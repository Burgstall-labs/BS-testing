from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from .assets import load_image_from_spec, resolve_font
from .compose import TONES, compose_cut, parse_hex

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]

def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)
_OPT_STR = ('STRING', {'forceInput': True})
_SKIP_SENTINELS = {'', '-', 'none', 'null', 'skip'}

def _clean_field(s) -> str:
    s = (s or '').strip()
    return '' if s.lower() in _SKIP_SENTINELS else s

class BSLoadImageFromURL:

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'url': ('STRING', {'default': ''})}}
    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('image',)
    FUNCTION = 'load'
    CATEGORY = 'BS-testing'

    def load(self, url):
        url = (url or '').strip()
        if not url:
            raise ValueError('BS Load Image From URL: `url` is empty — hero_image is required.')
        img = load_image_from_spec(url, target_width=2048)
        if img.mode == 'RGBA':
            bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
            bg.alpha_composite(img)
            img = bg
        return (_pil_to_tensor(img),)

class BSBrandKit:

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'logo_url': ('STRING', {'default': ''}), 'primary_color1': ('STRING', {'default': '#FFFFFF'}), 'primary_color2': ('STRING', {'default': '#111111'}), 'font': ('STRING', {'default': ''}), 'tone': ('STRING', {'default': 'minimal'})}}
    RETURN_TYPES = ('BS_BRAND_KIT',)
    RETURN_NAMES = ('brand_kit',)
    FUNCTION = 'build'
    CATEGORY = 'BS-testing'

    def build(self, logo_url, primary_color1, primary_color2, font, tone):
        logo = None
        logo_url = (logo_url or '').strip()
        if logo_url:
            logo = load_image_from_spec(logo_url, target_width=1200)
        font_regular, font_bold = resolve_font(font)
        return ({'logo': logo, 'color1': parse_hex(primary_color1, (255, 255, 255)), 'color2': parse_hex(primary_color2, (17, 17, 17)), 'font_regular': font_regular, 'font_bold': font_bold, 'tone': (tone or 'minimal').strip().lower()},)

class BSAudienceCutsRender:

    @classmethod
    def INPUT_TYPES(cls):
        optional = {'format': _OPT_STR}
        for i in range(1, 5):
            optional[f'cut{i}_label'] = _OPT_STR
            optional[f'cut{i}_headline'] = _OPT_STR
            optional[f'cut{i}_subhead'] = _OPT_STR
        return {'required': {'hero': ('IMAGE',), 'brand_kit': ('BS_BRAND_KIT',)}, 'optional': optional}
    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('images',)
    FUNCTION = 'render'
    CATEGORY = 'BS-testing'

    def render(self, hero, brand_kit, format=None, **cuts):
        fmt = _clean_field(format)
        if not fmt:
            return (torch.zeros((0, 8, 8, 3), dtype=torch.float32),)
        hero_pil = _tensor_to_pil(hero)
        frames = []
        for i in range(1, 5):
            label = _clean_field(cuts.get(f'cut{i}_label'))
            headline = _clean_field(cuts.get(f'cut{i}_headline'))
            subhead = _clean_field(cuts.get(f'cut{i}_subhead'))
            if not (label or headline or subhead):
                continue
            frames.append(_pil_to_tensor(compose_cut(hero_pil, brand_kit, headline, subhead, fmt)))
        if not frames:
            return (torch.zeros((0, 8, 8, 3), dtype=torch.float32),)
        return (torch.cat(frames, dim=0),)

def _round16(v: float) -> int:
    return max(64, int(round(v / 16) * 16))

class BSPadToAspect:
    EDIT_AREA = 2100000.0

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'skip_when_empty': ('BOOLEAN', {'default': True}), 'megapixels': ('FLOAT', {'default': 1.0, 'min': 0.25, 'max': 2.5, 'step': 0.05})}, 'optional': {'format': _OPT_STR}}
    RETURN_TYPES = ('IMAGE', 'MASK')
    RETURN_NAMES = ('padded', 'outpaint_mask')
    FUNCTION = 'pad'
    CATEGORY = 'BS-testing'

    def pad(self, image, skip_when_empty, megapixels, format=None):
        from .compose import canvas_size
        fmt = _clean_field(format)
        if not fmt and skip_when_empty:
            dummy = torch.full((1, 64, 64, 3), 0.5, dtype=torch.float32)
            return (dummy, torch.zeros((1, 64, 64), dtype=torch.float32))
        src = _tensor_to_pil(image)
        if not fmt:
            scale = (megapixels * 1000000.0 / (src.width * src.height)) ** 0.5
            w, h = (_round16(src.width * scale), _round16(src.height * scale))
            out = src.resize((w, h), Image.LANCZOS)
            return (_pil_to_tensor(out), torch.zeros((1, h, w), dtype=torch.float32))
        cw, ch = canvas_size(fmt)
        scale = min(1.0, (self.EDIT_AREA / (cw * ch)) ** 0.5)
        w, h = (_round16(cw * scale), _round16(ch * scale))
        fit = min(w / src.width, h / src.height)
        if max(w / src.width, h / src.height) / fit < 1.02:
            from .compose import cover_crop
            out = cover_crop(src, w, h)
            return (_pil_to_tensor(out), torch.zeros((1, h, w), dtype=torch.float32))
        sw, sh = (max(1, round(src.width * fit)), max(1, round(src.height * fit)))
        placed = src.resize((sw, sh), Image.LANCZOS)
        canvas = Image.new('RGB', (w, h), (127, 127, 127))
        ox, oy = ((w - sw) // 2, (h - sh) // 2)
        canvas.paste(placed, (ox, oy))
        mask = torch.ones((1, h, w), dtype=torch.float32)
        mask[:, oy:oy + sh, ox:ox + sw] = 0.0
        return (_pil_to_tensor(canvas), mask)

class BSCompositePreserved:

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'generated': ('IMAGE',), 'placed': ('IMAGE',), 'mask': ('MASK',), 'feather': ('INT', {'default': 24, 'min': 0, 'max': 256})}}
    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('image',)
    FUNCTION = 'composite'
    CATEGORY = 'BS-testing'

    def composite(self, generated, placed, mask, feather):
        if mask.max() <= 0:
            return (placed,)
        _, h, w, _ = placed.shape
        if generated.shape[1:3] != placed.shape[1:3]:
            gen = _tensor_to_pil(generated).resize((w, h), Image.LANCZOS)
            generated = _pil_to_tensor(gen)
        m = Image.fromarray((mask[0].cpu().numpy() * 255).astype(np.uint8), mode='L')
        if m.size != (w, h):
            m = m.resize((w, h), Image.BILINEAR)
        binary = np.asarray(m, dtype=np.float32) / 255.0
        if feather > 0:
            from PIL import ImageFilter
            m = m.filter(ImageFilter.GaussianBlur(feather))
        soft = np.clip(np.asarray(m, dtype=np.float32) / 255.0 * 2.0, 0.0, 1.0)
        m_t = torch.from_numpy(np.maximum(soft, binary))[None, ..., None]
        out = placed * (1.0 - m_t) + generated[:1] * m_t
        return (out,)

class BSCleanupComposite:

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'original': ('IMAGE',), 'cleaned': ('IMAGE',), 'threshold': ('FLOAT', {'default': 0.1, 'min': 0.02, 'max': 0.5, 'step': 0.01}), 'grow': ('INT', {'default': 10, 'min': 0, 'max': 128}), 'feather': ('INT', {'default': 8, 'min': 0, 'max': 128})}}
    RETURN_TYPES = ('IMAGE', 'MASK')
    RETURN_NAMES = ('image', 'patch_mask')
    FUNCTION = 'composite'
    CATEGORY = 'BS-testing'

    def composite(self, original, cleaned, threshold, grow, feather):
        from PIL import ImageFilter
        from scipy import ndimage
        orig = _tensor_to_pil(original)
        clean = _tensor_to_pil(cleaned)
        if clean.width * clean.height > orig.width * orig.height:
            size = clean.size
            orig = orig.resize(size, Image.LANCZOS)
        else:
            size = orig.size
            clean = clean.resize(size, Image.LANCZOS)
        w, h = size
        blur_r = max(1.0, 0.002 * max(w, h))
        ob = np.asarray(orig.filter(ImageFilter.GaussianBlur(blur_r)), dtype=np.float32)
        cb = np.asarray(clean.filter(ImageFilter.GaussianBlur(blur_r)), dtype=np.float32)
        diff = np.abs(ob - cb).max(axis=2) / 255.0
        mask = diff > threshold
        mask = ndimage.binary_closing(mask, iterations=3)
        labels, n = ndimage.label(mask)
        if n:
            min_area = max(64, int(0.0002 * w * h))
            sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=np.arange(1, n + 1))
            keep = np.flatnonzero(sizes >= min_area) + 1
            mask = np.isin(labels, keep)
        if grow > 0:
            mask = ndimage.maximum_filter(mask, size=2 * grow + 1)
        frac = float(mask.mean())
        if frac > 0.9:
            import logging
            logging.getLogger('BS-testing').warning('BSCleanupComposite: %.0f%% of the frame changed — keeping the cleaned render wholesale.', frac * 100)
            m_t = torch.ones((1, h, w), dtype=torch.float32)
            return (_pil_to_tensor(clean), m_t)
        if frac > 0.6:
            import logging
            logging.getLogger('BS-testing').warning('BSCleanupComposite: %.0f%% of the frame changed — compositing anyway; preserved region is small.', frac * 100)
        m = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        if feather > 0:
            m = m.filter(ImageFilter.GaussianBlur(feather))
        m_arr = np.asarray(m, dtype=np.float32) / 255.0
        m_t = torch.from_numpy(m_arr)[None, ..., None]
        out = _pil_to_tensor(orig) * (1.0 - m_t) + _pil_to_tensor(clean) * m_t
        return (out, m_t[..., 0])
NODE_CLASS_MAPPINGS = {'BSLoadImageFromURL': BSLoadImageFromURL, 'BSBrandKit': BSBrandKit, 'BSAudienceCutsRender': BSAudienceCutsRender, 'BSPadToAspect': BSPadToAspect, 'BSCompositePreserved': BSCompositePreserved, 'BSCleanupComposite': BSCleanupComposite}
NODE_DISPLAY_NAME_MAPPINGS = {'BSLoadImageFromURL': 'BS Load Image From URL', 'BSBrandKit': 'BS Brand Kit', 'BSAudienceCutsRender': 'BS Audience Cuts Render', 'BSPadToAspect': 'BS Pad To Aspect', 'BSCompositePreserved': 'BS Composite Preserved', 'BSCleanupComposite': 'BS Cleanup Composite'}
