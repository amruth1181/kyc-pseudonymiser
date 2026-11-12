"""Stage 4: label personal values with a type and a box.

Recall comes first: a missed field is a leak, an extra box only costs a
little fidelity. So several detectors run and their results are merged:

1. MRZ anchors   - values read from a valid MRZ are searched for in the
                   visual zone (fuzzy, with the date reformatted).
2. Label rules   - per document type: the value under or after a label.
3. Patterns      - regular expressions for structured values (IBAN, postcode).
4. NER (optional)- Presidio over free text when installed.
5. Corpus sweep  - runs after stage 5: names of known people found anywhere
                   (e.g. a transfer description), see s5_resolve.sweep.
"""
from __future__ import annotations

import re

import cv2
import numpy as np
from rapidfuzz import fuzz

from ..layout import (build_lines, find_label, find_value, segments, text_of, union_box, value_below,
                      value_right, vnorm)
from ..models import Field, PageRecord
from ..mrz import build_td3, yymmdd_to_viz
from ..ocr import read_line_conf

# (label text, field type, layout)
LABEL_RULES = {
    "passport": [
        ("Passport No.", "doc_number", "below"),
        ("Surname / Nom", "surname", "below"),
        ("Given names / Prenoms", "given_names", "below"),
        ("Date of birth", "dob", "below"),
        ("Place of birth", "place_of_birth", "below"),
    ],
    "utility_bill": [
        ("Customer:", "full_name", "right"),
        ("Address:", "address", "right"),
        ("Account number:", "account_number", "right"),
    ],
    "bank_statement": [
        ("Account holder:", "full_name", "right"),
        ("Address:", "address", "right"),
        ("IBAN:", "iban", "right"),
    ],
}

POSTCODE = re.compile(r"^[A-Z]{2}-\d{4}$")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}\b")


def _add(rec: PageRecord, ftype: str, words, source: str, conf: float = 1.0) -> None:
    if not words:
        return
    box = union_box([w.box for w in words])
    value = text_of(words)
    for f in rec.fields:
        if f.ftype == ftype and _overlap(f.box, box) > 0.5:
            f.source = f.source if source in f.source else f"{f.source}+{source}"
            if (box[2] - box[0]) > (f.box[2] - f.box[0]):   # keep the more complete reading
                f.box, f.value = box, value
            return
    rec.fields.append(Field(ftype=ftype, value=value, box=box, source=source, confidence=conf))


def _overlap(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) or 1
    return inter / small


def label(img: np.ndarray, rec: PageRecord) -> PageRecord:
    lines = build_lines(rec.words)

    # 1. MRZ anchors
    if rec.mrz and rec.mrz.valid:
        m = rec.mrz
        anchors = [("surname", m.surname), ("given_names", m.given_names), ("doc_number", m.doc_number),
                   ("dob", yymmdd_to_viz(m.dob))]
        mrz_words = {id(w) for l in lines for w in l if "<" in w.text}
        for ftype, value in anchors:
            if value:
                hit = find_value(lines, value, min_ratio=88, exclude=mrz_words)
                if hit:
                    _add(rec, ftype, hit[0], "mrz_anchor", hit[1] / 100)
        for i, (box, line) in enumerate(zip(m.line_boxes, m.raw)):
            rec.fields.append(Field(ftype=f"mrz_line{i + 1}", value=line, box=box, source="mrz"))

    # 2. Label rules
    for label_text, ftype, layout in LABEL_RULES.get(rec.doc_type, []):
        hit = find_label(lines, label_text)
        if not hit:
            continue
        li, lw = hit
        words = value_below(lines, li, lw) if layout == "below" else value_right(lines, li, lw)
        _add(rec, ftype, words, "label")
        if ftype == "address" and li + 1 < len(lines):
            # The line under the street holds 'POSTCODE CITY'; only the postcode is personal.
            for w in lines[li + 1]:
                if POSTCODE.match(w.text):
                    _add(rec, "postcode", [w], "pattern")

    # 3. Patterns anywhere on the page
    for line in lines:
        for seg in segments(line):
            t = text_of(seg)
            if IBAN.search(t.replace("O", "0")) and not any(f.ftype == "iban" for f in rec.fields):
                _add(rec, "iban", seg, "pattern", 0.9)

    # 4. Optional NER over free text
    _presidio(rec, lines)

    # 5. Second, closer reading of every value, reconciled with the MRZ where it is checksum-protected
    refine(img, rec)
    return rec


def refine(img: np.ndarray, rec: PageRecord) -> None:
    """Re-read each field from an upscaled crop.

    Page-level OCR is tuned for layout; a single-line re-read at 2x fixes
    most character slips (e.g. 'MANO}' -> 'MANOJ'). Number and date of birth
    are then taken from the MRZ when its check digits validated them.
    """
    h, w = img.shape[:2]
    for f in rec.fields:
        if f.ftype.startswith("mrz_"):
            continue
        x0, y0, x1, y1 = f.box
        pad = max(4, (y1 - y0) // 4)
        crop = img[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
        up = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        readings = [read_line_conf(crop), read_line_conf(up)]
        text, conf = max(readings, key=lambda r: r[1])
        if text and fuzz.ratio(vnorm(text), vnorm(f.value)) >= 70:
            f.value, f.confidence = text, round(conf / 100, 3)
    if rec.mrz and rec.mrz.valid:
        m = rec.mrz
        by_type = {f.ftype: f for f in rec.fields}
        for f in rec.fields:
            if f.ftype == "doc_number":
                f.value = m.doc_number
            elif f.ftype == "dob":
                f.value = yymmdd_to_viz(m.dob)
        # The validated MRZ can be rebuilt exactly; names come from the visual zone when they agree,
        # since check digits do not cover them. Exact text gives an exact style fit in stage 7.
        sur, giv = m.surname, m.given_names
        if "surname" in by_type and fuzz.ratio(vnorm(by_type["surname"].value), vnorm(sur)) >= 80:
            sur = vnorm(by_type["surname"].value)
        if "given_names" in by_type and fuzz.ratio(vnorm(by_type["given_names"].value), vnorm(giv)) >= 80:
            giv = vnorm(by_type["given_names"].value)
        l1, l2 = build_td3(m.doc_code or "P", m.issuer, sur, giv, m.doc_number, m.nationality, m.dob, m.sex,
                           m.expiry, m.personal)
        for f in rec.fields:
            if f.ftype == "mrz_line1":
                f.value = l1
            elif f.ftype == "mrz_line2":
                f.value = l2


def _presidio(rec: PageRecord, lines) -> None:
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError:
        return
    engine = AnalyzerEngine()
    for line in lines:
        for seg in segments(line):
            text = text_of(seg)
            for r in engine.analyze(text=text, language="en", entities=["PERSON"]):
                if r.score < 0.6:
                    continue
                span = text[r.start:r.end]
                hit = find_value([seg], span, min_ratio=90)
                if hit:
                    _add(rec, "third_party_name", hit[0], "ner", r.score)
