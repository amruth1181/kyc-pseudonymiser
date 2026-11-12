"""Visible marking for outputs served to the public.

A pseudonymised identity document with valid check digits looks genuine.
Outputs returned to anonymous callers are therefore marked across the whole
page, so they cannot pass as an unaltered document. Clean output is only
returned to callers holding the operator's access token.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .fonts import font_path

TEXT = "PSEUDONYMISED - NOT A VALID DOCUMENT"


def apply_watermark(rgb: np.ndarray, text: str = TEXT) -> np.ndarray:
    base = Image.fromarray(rgb).convert("RGBA")
    w, h = base.size
    size = max(18, int(min(w, h) * 0.045))
    font = ImageFont.truetype(font_path("sans_bold"), size)

    # Diagonal repeated text over the whole page.
    tile = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    step_x = int(font.getlength(text)) + size * 2
    step_y = size * 5
    for row, y in enumerate(range(0, h * 2, step_y)):
        offset = (row % 2) * step_x // 2
        for x in range(-step_x, w * 2, step_x):
            d.text((x + offset, y), text, font=font, fill=(200, 30, 30, 70))
    tile = tile.rotate(30, resample=Image.BICUBIC)
    left, top = (tile.width - w) // 2, (tile.height - h) // 2
    overlay = tile.crop((left, top, left + w, top + h))

    # Solid banner along the top edge.
    band = max(28, int(h * 0.035))
    d2 = ImageDraw.Draw(overlay)
    d2.rectangle((0, 0, w, band), fill=(200, 30, 30, 210))
    bfont = ImageFont.truetype(font_path("sans_bold"), int(band * 0.6))
    d2.text((band // 3, band * 0.18), text, font=bfont, fill=(255, 255, 255, 255))

    return np.array(Image.alpha_composite(base, overlay).convert("RGB"))
