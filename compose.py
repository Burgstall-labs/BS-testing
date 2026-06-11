from __future__ import annotations
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
FORMATS: dict[str, tuple[int, int]] = {'1:1': (1080, 1080), '4:5': (1080, 1350), '9:16': (1080, 1920), '16:9': (1920, 1080)}
TONES = ('playful', 'premium', 'technical', 'minimal')
_FMT_RE = re.compile('^\\s*(\\d+)\\s*[:xX]\\s*(\\d+)\\s*$')

def canvas_size(fmt: str) -> tuple[int, int]:
    fmt = fmt.strip()
    if fmt in FORMATS:
        return FORMATS[fmt]
    m = _FMT_RE.match(fmt)
    if not m:
        raise ValueError(f"BS-testing: unknown format {fmt!r}; expected one of {sorted(FORMATS)} or a 'W:H' ratio.")
    a, b = (int(m.group(1)), int(m.group(2)))
    if a <= 0 or b <= 0:
        raise ValueError(f'BS-testing: invalid format ratio {fmt!r}.')
    if a >= b:
        h = 1080
        w = min(2048, 2 * round(1080 * a / b / 2))
    else:
        w = 1080
        h = min(2048, 2 * round(1080 * b / a / 2))
    return (w, h)

def parse_hex(s: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    s = (s or '').strip().lstrip('#')
    if re.fullmatch('[0-9a-fA-F]{3}', s):
        s = ''.join((ch * 2 for ch in s))
    if not re.fullmatch('[0-9a-fA-F]{6}', s):
        return default
    return tuple((int(s[i:i + 2], 16) for i in (0, 2, 4)))

def _lum(c: tuple[int, int, int]) -> float:
    r, g, b = (v / 255.0 for v in c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def _contrast(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    la, lb = (_lum(a), _lum(b))
    hi, lo = (max(la, lb), min(la, lb))
    return (hi + 0.05) / (lo + 0.05)

def _best_on(bg: tuple[int, int, int], *candidates: tuple[int, int, int]) -> tuple[int, int, int]:
    for c in candidates:
        if _contrast(c, bg) >= 3.0:
            return c
    return (255, 255, 255) if _lum(bg) < 0.5 else (18, 18, 18)
_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, max(1, int(size)))
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(key[0], key[1])
    return _font_cache[key]

def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
    lines: list[str] = []
    cur = ''
    for word in text.split():
        trial = f'{cur} {word}'.strip()
        if font.getlength(trial) <= max_w or not cur:
            if not cur and font.getlength(word) > max_w:
                piece = ''
                for ch in word:
                    if font.getlength(piece + ch) <= max_w or not piece:
                        piece += ch
                    else:
                        lines.append(piece)
                        piece = ch
                cur = piece
            else:
                cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines

def _fit_text(text: str, font_path: str, max_w: float, start_size: float, min_size: float, max_lines: int) -> tuple[ImageFont.FreeTypeFont, list[str], int]:
    size = start_size
    while True:
        f = _font(font_path, int(size))
        lines = _wrap(text, f, max_w)
        fits = len(lines) <= max_lines and all((f.getlength(l) <= max_w for l in lines))
        if fits or size <= min_size:
            return (f, lines, int(size))
        size *= 0.93

def _line_h(size: int) -> int:
    return int(size * 1.18)

def _draw_lines(draw: ImageDraw.ImageDraw, lines: list[str], font: ImageFont.FreeTypeFont, size: int, color: tuple[int, ...], x: float, y: float, *, align: str='left', box_w: float=0) -> float:
    lh = _line_h(size)
    for i, line in enumerate(lines):
        lx = x if align == 'left' else x + (box_w - font.getlength(line)) / 2
        draw.text((lx, y + i * lh), line, font=font, fill=color)
    return len(lines) * lh

def _draw_tracked_center(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, color: tuple[int, ...], cx: float, y: float, tracking: float) -> None:
    widths = [font.getlength(ch) for ch in text]
    total = sum(widths) + tracking * max(0, len(text) - 1)
    x = cx - total / 2
    for ch, wd in zip(text, widths):
        draw.text((x, y), ch, font=font, fill=color)
        x += wd + tracking

def _tracked_width(text: str, font: ImageFont.FreeTypeFont, tracking: float) -> float:
    return sum((font.getlength(ch) for ch in text)) + tracking * max(0, len(text) - 1)

def cover_crop(img: Image.Image, w: int, h: int) -> Image.Image:
    img = img.convert('RGB')
    scale = max(w / img.width, h / img.height)
    nw, nh = (max(w, round(img.width * scale)), max(h, round(img.height * scale)))
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = ((nw - w) // 2, (nh - h) // 2)
    return img.crop((left, top, left + w, top + h))

def _scrim(canvas: Image.Image, color: tuple[int, int, int], *, start_frac: float, max_alpha: int, from_top: bool=False) -> None:
    w, h = canvas.size
    grad = np.zeros((h, w), dtype=np.float32)
    span = h - int(start_frac * h)
    if span <= 0:
        return
    ramp = np.linspace(0.0, float(max_alpha), span) ** 1.0
    grad[h - span:, :] = ramp[:, None]
    if from_top:
        grad = grad[::-1].copy()
    overlay = Image.new('RGBA', canvas.size, color + (0,))
    overlay.putalpha(Image.fromarray(grad.astype(np.uint8), mode='L'))
    canvas.alpha_composite(overlay)

def _lum_ratio(a: float, b: float) -> float:
    hi, lo = (max(a, b), min(a, b))
    return (hi + 0.05) / (lo + 0.05)

def _logo_mean_lum(lg: Image.Image) -> float:
    arr = np.asarray(lg, dtype=np.float32)
    alpha = arr[..., 3:4] / 255.0
    if alpha.sum() < 1:
        return 1.0
    rgb = (arr[..., :3] * alpha).sum(axis=(0, 1)) / alpha.sum()
    return _lum(tuple(rgb))

def _logo_mono_color(lg: Image.Image) -> tuple[int, int, int] | None:
    arr = np.asarray(lg, dtype=np.float32)
    solid = arr[..., 3] > 80
    if solid.sum() < 10:
        return None
    px = arr[..., :3][solid]
    if px.std(axis=0).max() >= 18:
        return None
    return tuple((int(v) for v in px.mean(axis=0)))

def _tint(lg: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    out = Image.new('RGBA', lg.size, color + (0,))
    out.putalpha(lg.getchannel('A'))
    return out

def _region_lum_band(canvas: Image.Image, box: tuple[int, int, int, int]) -> tuple[float, float]:
    region = np.asarray(canvas.crop(box).convert('RGB'), dtype=np.float32)
    lums = region @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32) / 255.0
    return (float(np.percentile(lums, 15)), float(np.percentile(lums, 85)))

def _paste_logo(canvas: Image.Image, logo: Image.Image | None, x: float, y: float, target_h: float, *, center_x: bool=False, chip_colors: tuple[tuple[int, int, int], ...]=()) -> None:
    if logo is None:
        return
    w, _ = canvas.size
    target_h = max(8, int(target_h))
    ratio = target_h / logo.height
    lw = int(logo.width * ratio)
    if lw > 0.46 * w:
        ratio = 0.46 * w / logo.width
        lw = int(logo.width * ratio)
        target_h = max(8, int(logo.height * ratio))
    lg = logo.resize((lw, target_h), Image.LANCZOS)
    px = int((w - lw) / 2) if center_x else int(x)
    py = int(y)
    lo, hi = _region_lum_band(canvas, (px, py, px + lw, py + target_h))
    palette = tuple(chip_colors) + ((250, 250, 250), (15, 15, 15))
    if _logo_mono_color(lg) is not None:
        best = max(palette, key=lambda c: min(_lum_ratio(_lum(c), lo), _lum_ratio(_lum(c), hi)))
        lg = _tint(lg, best)
        logo_l = _lum(best)
    else:
        logo_l = _logo_mean_lum(lg)
    if min(_lum_ratio(logo_l, lo), _lum_ratio(logo_l, hi)) < 2.5:
        chip = max(palette, key=lambda c: _lum_ratio(_lum(c), logo_l))
        pad = int(0.4 * target_h)
        draw = ImageDraw.Draw(canvas, 'RGBA')
        draw.rounded_rectangle((px - pad, py - pad, px + lw + pad, py + target_h + pad), radius=int(0.5 * target_h), fill=chip + (235,))
    canvas.alpha_composite(lg, (px, py))

def compose_cut(hero: Image.Image, brand: dict, headline: str, subhead: str, fmt: str) -> Image.Image:
    w, h = canvas_size(fmt)
    base = min(w, h)
    m = int(0.07 * base)
    headline = (headline or '').strip()
    subhead = (subhead or '').strip()
    c1, c2 = (brand['color1'], brand['color2'])
    dark, light = sorted((c1, c2), key=_lum)
    scrim_col = dark if _lum(dark) <= 0.55 else (18, 18, 18)
    text_col = _best_on(scrim_col, light)
    tone = (brand.get('tone') or 'minimal').strip().lower()
    if tone not in TONES:
        tone = 'minimal'
    canvas = cover_crop(hero, w, h).convert('RGBA')
    draw = ImageDraw.Draw(canvas)
    font_reg, font_bold = (brand['font_regular'], brand['font_bold'])
    logo = brand.get('logo')
    neutral_top = (12, 12, 14)
    chip_colors = (dark, light)
    if tone == 'playful':
        _scrim(canvas, scrim_col, start_frac=0.4, max_alpha=225)
        _scrim(canvas, neutral_top, start_frac=0.84, max_alpha=90, from_top=True)
        _paste_logo(canvas, logo, m, m, 0.08 * base, chip_colors=chip_colors)
        max_w = w - 2 * m
        y_bottom = h - m
        pill_h = 0.0
        if subhead:
            sub_f, sub_lines, ss = _fit_text(subhead, font_bold, max_w - 1.6 * m, 0.038 * base, 0.024 * base, 2)
            pad_x, pad_y = (int(ss * 0.85), int(ss * 0.55))
            pill_w = max((sub_f.getlength(l) for l in sub_lines)) + 2 * pad_x
            pill_h = len(sub_lines) * _line_h(ss) + 2 * pad_y
            py = y_bottom - pill_h
            pill_col = light
            draw.rounded_rectangle((m, py, m + pill_w, py + pill_h), radius=pill_h / 2 if len(sub_lines) == 1 else ss, fill=pill_col + (255,))
            _draw_lines(draw, sub_lines, sub_f, ss, _best_on(pill_col, dark), m + pad_x, py + pad_y)
            y_bottom = py - int(0.55 * m)
        if headline:
            head_f, head_lines, hs = _fit_text(headline, font_bold, max_w, 0.095 * base, 0.045 * base, 3)
            head_h = len(head_lines) * _line_h(hs)
            _draw_lines(draw, head_lines, head_f, hs, text_col, m, y_bottom - head_h)
    elif tone == 'premium':
        _scrim(canvas, scrim_col, start_frac=0.46, max_alpha=200)
        _scrim(canvas, neutral_top, start_frac=0.86, max_alpha=80, from_top=True)
        _paste_logo(canvas, logo, 0, int(1.1 * m), 0.065 * base, center_x=True, chip_colors=chip_colors)
        max_w = int(0.86 * w) - 2 * int(0.0 * m)
        blocks_h = 0.0
        head_f = head_lines = hs = None
        sub_f = ss = None
        sub_tracked = None
        if headline:
            head_f, head_lines, hs = _fit_text(headline, font_bold, max_w, 0.082 * base, 0.04 * base, 3)
            blocks_h += len(head_lines) * _line_h(hs)
        if subhead:
            ss = int(0.03 * base)
            tracking = ss * 0.22
            sub_text = subhead.upper()
            while ss > 0.018 * base and _tracked_width(sub_text, _font(font_reg, ss), tracking) > max_w:
                ss = int(ss * 0.93)
                tracking = ss * 0.22
            sub_f = _font(font_reg, ss)
            sub_tracked = (sub_text, tracking)
            blocks_h += int(0.7 * m) + 2 + int(0.7 * m) + _line_h(ss)
        y = h - int(1.35 * m) - blocks_h
        if headline:
            y += _draw_lines(draw, head_lines, head_f, hs, text_col, (w - max_w) / 2, y, align='center', box_w=max_w)
        if subhead:
            y += int(0.7 * m)
            rule_w = int(0.07 * w)
            draw.rectangle(((w - rule_w) / 2, y, (w + rule_w) / 2, y + 2), fill=text_col + (150,))
            y += 2 + int(0.7 * m)
            sub_text, tracking = sub_tracked
            _draw_tracked_center(draw, sub_text, sub_f, text_col + (225,), w / 2, y, tracking)
    elif tone == 'technical':
        _scrim(canvas, neutral_top, start_frac=0.86, max_alpha=80, from_top=True)
        _paste_logo(canvas, logo, m, m, 0.075 * base, chip_colors=chip_colors)
        pad = int(0.85 * m)
        panel_w = w - 2 * m
        inner_w = panel_w - 2 * pad - int(0.012 * base) - int(0.5 * pad)
        head_f = head_lines = hs = None
        sub_f = sub_lines = ss = None
        content_h = 0
        if headline:
            head_f, head_lines, hs = _fit_text(headline, font_bold, inner_w, 0.07 * base, 0.036 * base, 3)
            content_h += len(head_lines) * _line_h(hs)
        if subhead:
            sub_f, sub_lines, ss = _fit_text(subhead, font_reg, inner_w, 0.036 * base, 0.022 * base, 2)
            if content_h:
                content_h += int(0.35 * m)
            content_h += len(sub_lines) * _line_h(ss)
        panel_h = content_h + 2 * pad
        x0, y0 = (m, h - m - panel_h)
        box = (x0, y0, x0 + panel_w, y0 + panel_h)
        radius = int(0.018 * base)
        blurred = canvas.crop(box).filter(ImageFilter.GaussianBlur(0.014 * base))
        mask = Image.new('L', (panel_w, panel_h), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, panel_w, panel_h), radius=radius, fill=255)
        canvas.paste(blurred, box[:2], mask)
        overlay = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).rounded_rectangle(box, radius=radius, fill=scrim_col + (175,))
        canvas.alpha_composite(overlay)
        draw = ImageDraw.Draw(canvas)
        accent = c1 if _contrast(c1, scrim_col) >= 1.8 else light
        bar_w = max(4, int(0.006 * base))
        draw.rounded_rectangle((x0 + pad, y0 + pad, x0 + pad + bar_w, y0 + panel_h - pad), radius=bar_w // 2, fill=accent + (255,))
        tx = x0 + pad + bar_w + int(0.5 * pad)
        ty = y0 + pad
        if headline:
            ty += _draw_lines(draw, head_lines, head_f, hs, text_col, tx, ty)
            ty += int(0.35 * m)
        if subhead:
            _draw_lines(draw, sub_lines, sub_f, ss, text_col + (210,), tx, ty)
    else:
        band_col = light
        on_band = _best_on(band_col, dark)
        max_w = w - 2 * m
        head_f = head_lines = hs = None
        sub_f = sub_lines = ss = None
        content_h = 0
        if headline:
            head_f, head_lines, hs = _fit_text(headline, font_bold, max_w, 0.058 * base, 0.03 * base, 2)
            content_h += len(head_lines) * _line_h(hs)
        if subhead:
            sub_f, sub_lines, ss = _fit_text(subhead, font_reg, max_w, 0.034 * base, 0.02 * base, 2)
            if content_h:
                content_h += int(0.3 * m)
            content_h += len(sub_lines) * _line_h(ss)
        rule_h = max(3, int(0.005 * base))
        content_h += rule_h + int(0.45 * m)
        band_h = content_h + int(1.5 * m)
        draw.rectangle((0, h - band_h, w, h), fill=band_col + (255,))
        accent = c1 if _contrast(c1, band_col) >= 1.8 else on_band
        y = h - band_h + int(0.75 * m)
        draw.rectangle((m, y, m + int(0.09 * w), y + rule_h), fill=accent + (255,))
        y += rule_h + int(0.45 * m)
        if headline:
            y += _draw_lines(draw, head_lines, head_f, hs, on_band, m, y)
            y += int(0.3 * m)
        if subhead:
            _draw_lines(draw, sub_lines, sub_f, ss, on_band + (215,), m, y)
        _paste_logo(canvas, logo, m, m, 0.07 * base, chip_colors=chip_colors)
    return canvas.convert('RGB')
