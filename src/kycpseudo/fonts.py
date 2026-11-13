"""Font registry.

Rendering a surrogate in the same typeface as the original is what makes the
edit invisible and the text exact. Each document template maps its fields to
a font; real deployments add the issuing state's fonts here. OCR-B, which
ICAO 9303 mandates for the MRZ, is bundled in data/fonts. The synthetic corpus
otherwise uses DejaVu and Liberation fonts, which ship with most Linux
distributions.
"""
from __future__ import annotations

import glob
import os
from functools import lru_cache

CANDIDATES = {
    "sans": ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"],
    "sans_bold": ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial Bold.ttf"],
    "mono": ["OCRB.otf", "OCRB.ttf", "OCR-B.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "Courier New.ttf"],
}

BUNDLED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fonts")
SEARCH_DIRS = [BUNDLED, "/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
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


def font_candidates(doc_type: str, ftype: str) -> list[tuple[str, bool]]:
    """Fonts to try for a field, the template's own first, as (name, fixed pitch).

    Passports from different states print the visual zone in different faces; many use
    OCR-B there as well as in the MRZ. Stage 7 fits each candidate to the scan and keeps
    the one that reproduces the original text best. Only the MRZ is laid out on a fixed
    pitch, because its 44 characters must fill the line exactly.
    """
    first = font_for(doc_type, ftype)
    if ftype.startswith("mrz_"):
        return [(first, True)]
    names = [first]
    if doc_type == "passport":
        names += [n for n in ("mono", "sans_bold", "sans") if n != first]
    out = []
    for n in names:
        try:
            font_path(n)
        except FileNotFoundError:
            continue
        out.append((n, False))
    return out
