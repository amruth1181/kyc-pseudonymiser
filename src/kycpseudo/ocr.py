"""OCR adapter.

The pipeline only needs words with boxes and confidences, so the engine sits
behind two functions. Tesseract is the default because it installs anywhere.
PaddleOCR (PP-OCRv5, Apache 2.0) is the recommended production engine and
plugs in by returning the same ``Word`` list.
"""
from __future__ import annotations

import numpy as np
import pytesseract

from .models import Word

MRZ_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"


def read_words(img: np.ndarray, lang: str = "eng", psm: int = 3, thorough: bool = False) -> list[Word]:
    """Words with boxes in the coordinates of ``img``.

    ``thorough`` is for small scans (enlarged at ingest): no single setting reads them
    completely. Fine print needs more enlargement and values printed over a pattern are
    found by sparse-text mode, so three more passes fill the spots the page reading left
    empty, with confident words only.
    """
    primary = _read_words(img, lang, psm)
    if not thorough:
        return primary
    import cv2
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    extra = _read_words(img, lang, 11)
    for mode in (psm, 11):
        extra += [Word(w.text, (w.box[0] // 2, w.box[1] // 2, -(-w.box[2] // 2), -(-w.box[3] // 2)), w.conf, w.line_id)
                  for w in _read_words(big, lang, mode)]
    return _fill_gaps(primary, extra)


def _fill_gaps(primary: list[Word], extra: list[Word]) -> list[Word]:
    kept = list(primary)
    for w in sorted(extra, key=lambda w: -w.conf):
        solid = w.conf >= 40 and sum(c.isalnum() for c in w.text) >= 2     # not a speck or a guess
        if solid and not any(_overlaps(w.box, k.box) for k in kept):
            kept.append(w)
    return kept


def _overlaps(a, b) -> bool:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return False
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) or 1
    return ix * iy > 0.3 * small


def _read_words(img: np.ndarray, lang: str, psm: int) -> list[Word]:
    d = pytesseract.image_to_data(img, lang=lang, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    words = []
    for i, text in enumerate(d["text"]):
        text = text.strip()
        conf = float(d["conf"][i])
        if not text or conf < 0:
            continue
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        words.append(Word(text=text, box=(x, y, x + w, y + h), conf=conf,
                          line_id=(d["block_num"][i], d["par_num"][i], d["line_num"][i])))
    return words


def _border(img: np.ndarray, px: int = 10) -> np.ndarray:
    """A white margin; Tesseract reads tight crops poorly and can crash on near-blank ones."""
    import cv2
    val = 255 if img.ndim == 2 else (255, 255, 255)
    return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=val)


def read_line(img: np.ndarray, lang: str = "eng", whitelist: str | None = None) -> str:
    cfg = "--psm 7"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    try:
        return pytesseract.image_to_string(_border(img), lang=lang, config=cfg).strip()
    except pytesseract.TesseractError:
        return ""


def read_word(img: np.ndarray, whitelist: str, lang: str = "eng") -> str:
    """One word from a tight crop, limited to ``whitelist``. Reads print that page-level OCR misses."""
    import cv2
    # The margin continues the page's own background: a white frame around print on a tinted
    # or patterned page shifts the binarisation threshold and the word is lost.
    framed = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
    try:
        return pytesseract.image_to_string(framed, lang=lang,
                                           config=f"--psm 8 -c tessedit_char_whitelist={whitelist}").strip()
    except pytesseract.TesseractError:
        return ""


def read_text(img: np.ndarray, lang: str = "eng") -> str:
    return pytesseract.image_to_string(img, lang=lang, config="--psm 3")


def read_line_conf(img: np.ndarray, lang: str = "eng") -> tuple[str, float]:
    """Single-line reading with the mean word confidence (0-100)."""
    try:
        d = pytesseract.image_to_data(_border(img), lang=lang, config="--psm 7", output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractError:
        return "", 0.0
    words = [(t.strip(), float(c)) for t, c in zip(d["text"], d["conf"]) if t.strip() and float(c) >= 0]
    if not words:
        return "", 0.0
    return " ".join(t for t, _ in words), sum(c for _, c in words) / len(words)
