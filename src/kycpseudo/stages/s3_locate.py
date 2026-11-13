"""Stage 3: locate fields and non-text regions.

* MRZ: candidate lines are re-read as whole lines with an MRZ-only character
  whitelist, parsed, and validated with ICAO check digits. A valid MRZ gives
  trusted values for name, number and dates, used in stage 4 as anchors.
* Photo: the largest portrait-shaped framed rectangle.
* Signature: ink below the 'Signature' label.
* Barcodes: interface only (decode with pyzbar/ZXing, rebuild payload, redraw).
"""
from __future__ import annotations

import cv2
import numpy as np

from ..layout import build_lines, find_label, text_of, union_box
from ..models import PageRecord, Region
from ..mrz import parse_line1, repair_td3
from ..ocr import MRZ_CHARS, read_line
from .s2_classify import looks_like_mrz


def locate_mrz(img: np.ndarray, rec: PageRecord) -> None:
    lines = build_lines(rec.words)
    cands = [l for l in lines if looks_like_mrz(text_of(l))]
    if len(cands) == 1:
        _locate_line1_only(img, rec, cands[0])
        return
    if len(cands) < 2:
        return
    l1w, l2w = cands[-2], cands[-1]
    boxes, crops = [], []
    h, w = img.shape[:2]
    for lw in (l1w, l2w):
        x0, y0, x1, y1 = union_box([x.box for x in lw])
        pad = (y1 - y0) // 3
        box = (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + 2 * pad), min(h, y1 + pad))
        crops.append(img[box[1]:box[3], box[0]:box[2]])
        boxes.append((x0, y0, x1, y1))
    # Several independent readings; the first whose check digits validate wins.
    readings = [[text_of(l1w), text_of(l2w)]]
    for prep in (_as_is, _upscale_binarize):
        readings.append([read_line(prep(c), whitelist=MRZ_CHARS) for c in crops])
    mrz = None
    for texts in readings[1:] + readings[:1]:
        cand = repair_td3(*texts)
        if cand and (mrz is None or (cand.valid and not mrz.valid)):
            mrz = cand
        if mrz and mrz.valid:
            break
    if mrz:
        mrz.line_boxes = boxes
        rec.mrz = mrz
        if not mrz.valid:
            rec.review_reasons.append("mrz_checksum_failed")


def _locate_line1_only(img: np.ndarray, rec: PageRecord, lw) -> None:
    """A scan cropped above MRZ line 2: the name line is still personal data and is replaced."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _ink_extent(img, union_box([x.box for x in lw]))
    pad = max(2, (y1 - y0) // 3)
    crop = img[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
    readings = [text_of(lw)] + [read_line(prep(crop), whitelist=MRZ_CHARS) for prep in (_as_is, _upscale_binarize)]
    parsed = [m for m in map(parse_line1, readings[1:] + readings[:1]) if m]
    if not parsed:
        return
    mrz = parsed[0]
    mrz.line_boxes = [(x0, y0, x1, y1)]
    rec.mrz = mrz


def _ink_extent(img: np.ndarray, box: tuple) -> tuple:
    """Widen a line box to the full run of characters; OCR often drops the last fillers."""
    x0, y0, x1, y1 = box
    band = cv2.cvtColor(img[y0:y1], cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cols = ink.any(axis=0)
    gap = int(1.5 * (y1 - y0))                     # wider than any space between MRZ characters
    while x1 < len(cols) and cols[x1:x1 + gap].any():
        x1 += 1
    while x0 > 0 and cols[max(0, x0 - gap):x0].any():
        x0 -= 1
    return x0, y0, x1, y1


def _as_is(crop: np.ndarray) -> np.ndarray:
    return crop


def _upscale_binarize(crop: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def locate_photo(img: np.ndarray) -> tuple | None:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    best = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 0.04 * W * H or not (1.1 <= h / max(w, 1) <= 1.7):
            continue
        if best is None or w * h > best[2] * best[3]:
            best = (x, y, w, h)
    if not best:
        return None
    x, y, w, h = best
    return (x, y, x + w, y + h)


def locate_signature(img: np.ndarray, rec: PageRecord) -> tuple | None:
    lines = build_lines(rec.words)
    hit = find_label(lines, "Signature")
    if not hit:
        return None
    _, lw = hit
    x0, _, _, y1 = union_box([w.box for w in lw])
    H, W = img.shape[:2]
    win = (x0, y1 + 4, min(W, x0 + 520), min(H, y1 + 150))
    gray = cv2.cvtColor(img[win[1]:win[3], win[0]:win[2]], cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ys, xs = np.nonzero(ink)
    if len(xs) < 50:
        return None
    return (win[0] + int(xs.min()) - 4, win[1] + int(ys.min()) - 4,
            win[0] + int(xs.max()) + 4, win[1] + int(ys.max()) + 4)


def locate(img: np.ndarray, rec: PageRecord) -> PageRecord:
    if rec.doc_type == "passport":
        locate_mrz(img, rec)
        photo = locate_photo(img)
        if photo:
            rec.regions.append(Region("photo", photo))
        else:
            rec.review_reasons.append("photo_not_found")
        sig = locate_signature(img, rec)
        if sig:
            rec.regions.append(Region("signature", sig))
    return rec
