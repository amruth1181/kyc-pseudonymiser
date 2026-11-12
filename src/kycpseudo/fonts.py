"""Font registry.

Rendering a surrogate in the same typeface as the original is what makes the
edit invisible and the text exact. Each document template maps its fields to
a font; real deployments add the issuing state's fonts here (and OCR-B for
the MRZ, which ICAO 9303 mandates). The synthetic corpus uses DejaVu and
Liberation fonts, which ship with most Linux distributions.
"""
from __future__ import annotations

import glob
import os
from functools import lru_cache

CANDIDATES = {
    "sans": ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"],
    "sans_bold": ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial Bold.ttf"],
    "mono": ["OCRB.ttf", "OCR-B.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "Courier New.ttf"],
}

SEARCH_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
               "/Library/Fonts", "/System/Library/Fonts", "C:/Windows/Fonts"]


@lru_cache(maxsize=None)
def font_path(name: str) -> str:
    env = os.environ.get(f"KYCPSEUDO_FONT_{name.upper()}")
    if env:
        return env
    for fname in CANDIDATES[name]:
        for root in SEARCH_DIRS:
            hits = glob.glob(os.path.join(root, "**", fname), recursive=True)
            if hits:
                return hits[0]
    raise FileNotFoundError(f"No font found for '{name}'. Install DejaVu fonts or set KYCPSEUDO_FONT_{name.upper()}.")


# Template field -> font. Bills and statements use the same family.
TEMPLATE_FONTS = {
    "passport": {"mrz_line1": "mono", "mrz_line2": "mono", "default": "sans_bold"},
    "utility_bill": {"default": "sans"},
    "bank_statement": {"default": "sans"},
    "unknown": {"default": "sans"},
}


def font_for(doc_type: str, ftype: str) -> str:
    t = TEMPLATE_FONTS.get(doc_type, TEMPLATE_FONTS["unknown"])
    return t.get(ftype, t["default"])
