from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from .assets import knockout_uniform_background, load_image_from_spec, resolve_font
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
            logo = knockout_uniform_background(load_image_from_spec(logo_url, target_width=1200))
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
        if src.width > 64 and src.height > 64:
            src = src.crop((2, 2, src.width - 2, src.height - 2))
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

def _membrane_blend(g_np, p_np, hard, strip=12, clamp=0.12):
    from scipy import ndimage
    h, w, _ = g_np.shape

    def banded(axis_all):
        n = 0
        while n < len(axis_all) and axis_all[n]:
            n += 1
        return n
    rows_all = hard.all(axis=1)
    cols_all = hard.all(axis=0)
    top = banded(rows_all)
    bottom = banded(rows_all[::-1])
    left = banded(cols_all)
    right = banded(cols_all[::-1])

    def edge_value(strip_arr, edge_at_start):
        n = strip_arr.shape[0]
        idx = np.arange(n, dtype=np.float32)
        mean_y = strip_arr.mean(axis=0)
        slope = ((idx - idx.mean())[:, None, None] * (strip_arr - mean_y)).sum(axis=0) / ((idx - idx.mean()) ** 2).sum()
        edge_idx = -0.5 if edge_at_start else n - 0.5
        return mean_y + slope * (edge_idx - idx.mean())

    def smooth_diff(orig_strip, gen_strip, orig_edge_at_start):
        d = edge_value(orig_strip, orig_edge_at_start) - edge_value(gen_strip, not orig_edge_at_start)
        valid = np.abs(d).max(axis=1) < clamp
        if valid.any() and (not valid.all()):
            idx = np.arange(d.shape[0], dtype=np.float32)
            for ch in range(d.shape[1]):
                d[~valid, ch] = np.interp(idx[~valid], idx[valid], d[valid, ch])
        d = np.clip(d, -clamp, clamp)
        return ndimage.gaussian_filter1d(d, sigma=16.0, axis=0)
    if top > 2 * strip:
        d = smooth_diff(p_np[top:top + strip], g_np[top - strip:top], True)
        fall = (np.arange(top, dtype=np.float32) + 1) / top
        g_np[:top] += d[None, :, :] * fall[:, None, None]
        g_np[top:top + 3] = p_np[top:top + 3]
    if bottom > 2 * strip:
        s = h - bottom
        d = smooth_diff(p_np[s - strip:s], g_np[s:s + strip], False)
        fall = (np.arange(bottom, dtype=np.float32)[::-1] + 1) / bottom
        g_np[s:] += d[None, :, :] * fall[:, None, None]
        g_np[s - 3:s] = p_np[s - 3:s]
    if left > 2 * strip:
        d = smooth_diff(p_np[:, left:left + strip].transpose(1, 0, 2), g_np[:, left - strip:left].transpose(1, 0, 2), True)
        fall = (np.arange(left, dtype=np.float32) + 1) / left
        g_np[:, :left] += d[:, None, :] * fall[None, :, None]
        g_np[:, left:left + 3] = p_np[:, left:left + 3]
    if right > 2 * strip:
        s = w - right
        d = smooth_diff(p_np[:, s - strip:s].transpose(1, 0, 2), g_np[:, s:s + strip].transpose(1, 0, 2), False)
        fall = (np.arange(right, dtype=np.float32)[::-1] + 1) / right
        g_np[:, s:] += d[:, None, :] * fall[None, :, None]
        g_np[:, s - 3:s] = p_np[:, s - 3:s]
    return g_np

def _correct_drift(g_np, p_np, hard, sigma=16.0, reach_frac=0.55, clamp=0.2):
    from scipy import ndimage
    h, w, _ = g_np.shape

    def band(flags):
        n = 0
        while n < len(flags) and flags[n]:
            n += 1
        return n
    rows_all = hard.all(axis=1)
    cols_all = hard.all(axis=0)
    top, bottom = (band(rows_all), band(rows_all[::-1]))
    left, right = (band(cols_all), band(cols_all[::-1]))
    D = 0.08

    def extend(edge, src_vals):
        return np.clip(edge + np.clip(edge - src_vals, -D, D), 0.0, 1.0)
    filled = p_np.copy()
    if top:
        src = np.clip(2 * top - 1 - np.arange(top), top + 2, h - bottom - 1)
        filled[:top] = extend(p_np[top + 2:top + 8].mean(axis=0), p_np[src])
    if bottom:
        s = h - bottom
        src = np.clip(2 * s - 1 - np.arange(s, h), top, s - 3)
        filled[s:] = extend(p_np[s - 8:s - 2].mean(axis=0), p_np[src])
    if left:
        src = np.clip(2 * left - 1 - np.arange(left), left + 2, w - right - 1)
        filled[:, :left] = extend(p_np[:, left + 2:left + 8].mean(axis=1)[:, None, :], p_np[:, src])
    if right:
        s = w - right
        src = np.clip(2 * s - 1 - np.arange(s, w), left, s - 3)
        filled[:, s:] = extend(p_np[:, s - 8:s - 2].mean(axis=1)[:, None, :], p_np[:, src])
    w_pad = hard.astype(np.float32)
    wb = np.maximum(ndimage.gaussian_filter(w_pad, sigma), 0.0001)
    T = np.empty_like(g_np)
    G_low = np.empty_like(g_np)
    for ch in range(3):
        T[..., ch] = ndimage.gaussian_filter(filled[..., ch], sigma)
        G_low[..., ch] = ndimage.gaussian_filter(g_np[..., ch] * w_pad, sigma) / wb
    dist = ndimage.distance_transform_edt(hard)
    reach = max(48.0, float(dist.max()) * reach_frac)
    weight = np.clip(1.0 - dist / reach, 0.0, 1.0) * (dist > 0)
    grad = ndimage.gaussian_gradient_magnitude(p_np.mean(axis=2), 1.5)
    gate = np.zeros_like(weight)
    if top:
        if float(np.percentile(grad[top + 6:top + 26], 90)) < 0.008:
            gate[:top] = 1.0
    if bottom:
        if float(np.percentile(grad[h - bottom - 26:h - bottom - 6], 90)) < 0.008:
            gate[h - bottom:] = 1.0
    if left:
        if float(np.percentile(grad[:, left + 6:left + 26], 90)) < 0.008:
            gate[:, :left] = 1.0
    if right:
        if float(np.percentile(grad[:, w - right - 26:w - right - 6], 90)) < 0.008:
            gate[:, w - right:] = 1.0
    weight = weight * gate
    corr = np.clip(T - G_low, -clamp, clamp) * weight[..., None]
    g_np = np.clip(g_np + corr, 0.0, 1.0)
    near_w = 24.0
    dev = np.abs(g_np - T).max(axis=2)
    pad_f = hard.astype(np.float32)
    row_dev = (dev * pad_f).sum(axis=1) / np.maximum(pad_f.sum(axis=1), 1.0)
    col_dev = (dev * pad_f).sum(axis=0) / np.maximum(pad_f.sum(axis=0), 1.0)
    line_strength = np.maximum(row_dev[:, None], col_dev[None, :])
    w_px = np.clip(dev * 6.0, 0.0, 1.0) * np.clip(1.0 - dist / near_w, 0.0, 1.0)
    w_line = np.clip(line_strength * 18.0, 0.0, 1.0) * (dist <= 48.0)
    w_fix = np.where(dist > 0, np.maximum(w_px, w_line), 0.0)
    w_fix = np.maximum(w_fix, ((dist > 0) & (dist <= 3)).astype(np.float32))
    w_fix = w_fix * gate
    g_np = g_np + (T - g_np) * w_fix[..., None]
    pin = ndimage.binary_dilation(hard, iterations=3) & ~hard
    g_np[pin] = p_np[pin]
    return g_np

class BSCompositePreserved:

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'generated': ('IMAGE',), 'placed': ('IMAGE',), 'mask': ('MASK',), 'feather': ('INT', {'default': 8, 'min': 0, 'max': 256})}}
    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('image',)
    FUNCTION = 'composite'
    CATEGORY = 'BS-testing'

    def composite(self, generated, placed, mask, feather):
        from PIL import ImageFilter
        from scipy import ndimage
        if mask.max() <= 0:
            return (placed,)
        _, h, w, _ = placed.shape
        if generated.shape[1:3] != placed.shape[1:3]:
            gen = _tensor_to_pil(generated).resize((w, h), Image.LANCZOS)
            generated = _pil_to_tensor(gen)
        generated = generated[:1]
        m = Image.fromarray((mask[0].cpu().numpy() * 255).astype(np.uint8), mode='L')
        if m.size != (w, h):
            m = m.resize((w, h), Image.BILINEAR)
        binary = np.asarray(m, dtype=np.float32) / 255.0
        import os
        dump_dir = os.environ.get('BSRF_DUMP')
        if dump_dir:
            from pathlib import Path
            d = Path(dump_dir)
            d.mkdir(parents=True, exist_ok=True)
            n = len(list(d.glob('*_mask.png')))
            _tensor_to_pil(placed).save(d / f'{n:02d}_placed.png')
            _tensor_to_pil(generated).save(d / f'{n:02d}_gen.png')
            m.save(d / f'{n:02d}_mask.png')
        hard = binary > 0.5
        p_np = placed[0].cpu().numpy()
        g_np = np.clip(_membrane_blend(generated[0].cpu().numpy().copy(), p_np, hard), 0.0, 1.0)
        g_np = _correct_drift(g_np, p_np, hard)
        generated = torch.from_numpy(g_np)[None, ...]
        if feather > 0:
            m = m.filter(ImageFilter.GaussianBlur(feather))
        soft = np.asarray(m, dtype=np.float32) / 255.0
        align = np.clip(1.0 - ndimage.gaussian_filter(np.abs(p_np - g_np).max(axis=2), 8.0) * 6.0, 0.0, 1.0)
        m_t = torch.from_numpy(np.maximum(soft * align, binary))[None, ..., None]
        out = placed * (1.0 - m_t) + generated * m_t
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
        o_arr = np.asarray(orig, dtype=np.float32)
        c_arr = np.asarray(clean, dtype=np.float32)
        ol = o_arr.mean(axis=2) - o_arr.mean()
        cl = c_arr.mean(axis=2) - c_arr.mean()
        cc = np.fft.irfft2(np.fft.rfft2(ol) * np.conj(np.fft.rfft2(cl)), s=ol.shape)
        dy, dx = np.unravel_index(np.argmax(cc), cc.shape)
        if dy > h // 2:
            dy -= h
        if dx > w // 2:
            dx -= w
        if 0 < max(abs(dy), abs(dx)) <= 16:
            clean = Image.fromarray(np.roll(np.asarray(clean), (dy, dx), axis=(0, 1)))
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
