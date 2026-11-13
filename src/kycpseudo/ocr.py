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


OCR_MIN_WIDTH = 1400     # page-level OCR is unreliable below this; smaller scans are read enlarged


def ocr_scale(img: np.ndarray) -> int:
    """Whole-number enlargement used to read a small scan (1 for normal scans)."""
    return max(1, min(4, round(OCR_MIN_WIDTH / max(img.shape[1], 1))))


def read_words(img: np.ndarray, lang: str = "eng", psm: int = 3) -> list[Word]:
    """Words with boxes in the coordinates of ``img``. Small scans are enlarged for reading only."""
    s = ocr_scale(img)
    if s > 1:
        import cv2
        big = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        return [Word(w.text, (w.box[0] // s, w.box[1] // s, -(-w.box[2] // s), -(-w.box[3] // s)), w.conf, w.line_id)
                for w in _read_words(big, lang, psm)]
    return _read_words(img, lang, psm)


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
