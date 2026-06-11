from __future__ import annotations
import base64
import hashlib
import logging
import re
from io import BytesIO
from pathlib import Path
import requests
from PIL import Image
logger = logging.getLogger('BS-testing')
CACHE_DIR = Path(__file__).parent / '.cache'
_FALLBACK_FONT_REGULAR = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
_FALLBACK_FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
_URL_RE = re.compile('^https?://', re.IGNORECASE)

def _cache_path(key: str, suffix: str='') -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / (hashlib.sha1(key.encode('utf-8')).hexdigest() + suffix)

def fetch_bytes(url: str, *, timeout: float=30.0, user_agent: str='BS-testing/1.0') -> bytes:
    path = _cache_path(url)
    if path.is_file():
        return path.read_bytes()
    resp = requests.get(url, timeout=timeout, headers={'User-Agent': user_agent})
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return resp.content

def _looks_like_svg(data: bytes, spec: str) -> bool:
    head = data[:2048].lstrip().lower()
    return spec.lower().split('?')[0].endswith('.svg') or head.startswith(b'<svg') or b'<svg' in head[:512]

def _rasterize_svg(data: bytes, target_width: int) -> Image.Image:
    try:
        import cairosvg
    except ImportError as exc:
        raise RuntimeError('BS-testing: an SVG asset was supplied but cairosvg is not installed. Run `pip install cairosvg` in the ComfyUI environment.') from exc
    png = cairosvg.svg2png(bytestring=data, output_width=target_width)
    return Image.open(BytesIO(png))

def load_image_from_spec(spec: str, *, target_width: int=1024) -> Image.Image:
    spec = (spec or '').strip()
    if not spec:
        raise ValueError('BS-testing: empty image reference.')
    if spec.startswith('data:'):
        _, b64 = spec.split(',', 1)
        data = base64.b64decode(b64)
    elif _URL_RE.match(spec):
        data = fetch_bytes(spec)
    else:
        path = Path(spec)
        if not path.is_absolute():
            try:
                import folder_paths
                path = Path(folder_paths.get_annotated_filepath(spec))
            except Exception:
                pass
        if not path.is_file():
            raise FileNotFoundError(f'BS-testing: image not found: {spec!r}')
        data = path.read_bytes()
    if _looks_like_svg(data, spec):
        img = _rasterize_svg(data, target_width)
    else:
        img = Image.open(BytesIO(data))
        img.load()
    return img.convert('RGBA')
_FONT_FACE_RE = re.compile('@font-face\\s*\\{[^}]*\\}', re.DOTALL)
_WEIGHT_RE = re.compile('font-weight:\\s*(\\d+)')
_SRC_URL_RE = re.compile('url\\((https://[^)]+)\\)')

def _google_font_paths(family: str) -> tuple[str, str]:
    css_url = 'https://fonts.googleapis.com/css2?family=' + family.strip().replace(' ', '+') + ':wght@400;700'
    css_cache = _cache_path(css_url, '.css')
    if css_cache.is_file():
        css = css_cache.read_text()
    else:
        resp = requests.get(css_url, timeout=20, headers={'User-Agent': 'curl/8.5.0'})
        resp.raise_for_status()
        css = resp.text
        css_cache.write_text(css)
    by_weight: dict[int, str] = {}
    for block in _FONT_FACE_RE.findall(css):
        url_m = _SRC_URL_RE.search(block)
        if not url_m:
            continue
        weight_m = _WEIGHT_RE.search(block)
        by_weight[int(weight_m.group(1)) if weight_m else 400] = url_m.group(1)
    if not by_weight:
        raise RuntimeError(f'no font faces found for Google Font {family!r}')

    def _download(url: str) -> str:
        path = _cache_path(url, '.ttf')
        if not path.is_file():
            path.write_bytes(fetch_bytes(url))
        return str(path)
    regular_url = by_weight.get(400) or next(iter(by_weight.values()))
    bold_url = by_weight.get(700, regular_url)
    return (_download(regular_url), _download(bold_url))

def resolve_font(spec: str) -> tuple[str, str]:
    spec = (spec or '').strip()
    if not spec:
        return (_FALLBACK_FONT_REGULAR, _FALLBACK_FONT_BOLD)
    try:
        if _URL_RE.match(spec):
            suffix = Path(spec.split('?')[0]).suffix or '.ttf'
            path = _cache_path(spec, suffix)
            if not path.is_file():
                path.write_bytes(fetch_bytes(spec))
            return (str(path), str(path))
        return _google_font_paths(spec)
    except Exception:
        logger.warning('BS-testing: could not resolve font %r; falling back to DejaVu Sans.', spec, exc_info=True)
        return (_FALLBACK_FONT_REGULAR, _FALLBACK_FONT_BOLD)
