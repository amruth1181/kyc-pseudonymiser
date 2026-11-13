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
from datetime import date

import cv2
import numpy as np
from rapidfuzz import fuzz

from ..layout import (build_lines, find_label, find_value, segments, text_of, union_box, value_below,
                      value_right, vnorm)
from ..models import Field, PageRecord
from ..dates import format_viz, parse_viz
from ..mrz import build_line1, build_td3, viz_date_forms
from ..ocr import read_line_conf, read_word

# (label text, field type, layout)
LABEL_RULES = {
    # Several wordings per field: English-first (synthetic, UK, India), French-first (France)
    # and trilingual (Luxembourg). The first wording that leads to a value wins.
    "passport": [
        ("Passport No.", "doc_number", "below"),
        ("Passeport n", "doc_number", "below"),
        ("Surname / Nom", "surname", "below"),
        ("Surname", "surname", "below"),
        ("Nom", "surname", "below"),
        ("Given names / Prenoms", "given_names", "below"),
        ("Given names", "given_names", "below"),
        ("Given Name(s)", "given_names", "below"),
        ("Prenoms", "given_names", "below"),
        ("Date of birth", "dob", "below"),
        ("Date de naissance", "dob", "below"),
        ("Place of birth", "place_of_birth", "below"),
        ("Lieu de naissance", "place_of_birth", "below"),
        ("Domicile", "address", "below"),
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

PASSPORT_NUMBER = re.compile(r"^(?=(?:.*\d){6})[A-Z0-9]{8,9}$")
POSTCODE = re.compile(r"^[A-Z]{2}-\d{4}$")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}\b")


def _add(rec: PageRecord, ftype: str, words, source: str, conf: float = 1.0) -> None:
    if not words:
        return
    box = union_box([w.box for w in words])
    value = _trim(text_of(words))
    if not value:
        return
    for f in rec.fields:
        if f.ftype == ftype and _overlap(f.box, box) > 0.5:
            f.source = f.source if source in f.source else f"{f.source}+{source}"
            if (box[2] - box[0]) > (f.box[2] - f.box[0]):   # keep the more complete reading
                f.box, f.value = box, value
            return
    rec.fields.append(Field(ftype=ftype, value=value, box=box, source=source, confidence=conf))


def _trim(text: str) -> str:
    """Drop quote marks and specks OCR attaches to the ends of a value read off a patterned page."""
    t = re.sub(r"^[^0-9A-Za-zÀ-ÿ]+|[^0-9A-Za-zÀ-ÿ.)]+$", "", text.strip())
    return t if re.search(r"(^|\s)[A-Za-z]\.$", t) else t.rstrip(".")     # keep the dot of an initial


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
        anchors = [("surname", [m.surname]), ("given_names", [m.given_names]), ("doc_number", [m.doc_number]),
                   ("dob", viz_date_forms(m.dob))]
        mrz_words = {id(w) for l in lines for w in l if "<" in w.text}
        for ftype, forms in anchors:
            hits = [h for h in (find_value(lines, v, min_ratio=88, exclude=mrz_words) for v in forms if v) if h]
            if hits:
                best = max(hits, key=lambda h: h[1])
                _add(rec, ftype, best[0], "mrz_anchor", best[1] / 100)
        for i, (box, line) in enumerate(zip(m.line_boxes, m.raw)):
            rec.fields.append(Field(ftype=f"mrz_line{i + 1}", value=line, box=box, source="mrz"))

    # 2. Label rules
    for label_text, ftype, layout in LABEL_RULES.get(rec.doc_type, []):
        if rec.doc_type == "passport" and any(f.ftype == ftype for f in rec.fields):
            continue                    # already found (MRZ anchor or an earlier wording of the label)
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
    if rec.doc_type == "passport":
        _passport_extras(rec, lines)
    for line in lines:
        for seg in segments(line):
            t = text_of(seg)
            if IBAN.search(t.replace("O", "0")) and not any(f.ftype == "iban" for f in rec.fields):
                _add(rec, "iban", seg, "pattern", 0.9)

    # 4. Optional NER over free text
    _presidio(rec, lines)

    # A validated MRZ proves these values are printed in the visual zone too. One that no
    # detector located is hunted for directly; if still missing it is on the page somewhere,
    # so the page cannot be released.
    if rec.mrz and rec.mrz.valid:
        if not any(f.ftype == "doc_number" for f in rec.fields):
            _hunt(img, rec, "doc_number", rec.mrz.doc_number)
        have = {f.ftype for f in rec.fields}
        rec.review_reasons += [f"not_located:{t}" for t in ("surname", "given_names", "doc_number", "dob")
                               if t not in have and (t != "given_names" or rec.mrz.given_names)]

    # 5. Second, closer reading of every value, reconciled with the MRZ where it is checksum-protected
    refine(img, rec)
    return rec


def _hunt(img: np.ndarray, rec: PageRecord, ftype: str, value: str) -> None:
    """Find a known value that page-level OCR could not read (e.g. a number over a pattern).

    Every unread run of print of the right shape is read on its own as a single word and
    compared with the value. Knowing what to look for makes a weak reading usable.
    """
    target = vnorm(value).replace(" ", "")
    if len(target) < 6:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    taken = [f.box for f in rec.fields]
    heights = sorted(f.box[3] - f.box[1] for f in rec.fields if not f.ftype.startswith("mrz_"))
    h_ref = heights[len(heights) // 2] if heights else max(12, H // 40)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 2 * h_ref + 1, 12)
    # Labels are printed finer than values and often touch the value below them; eroding
    # removes the fine strokes so each value forms its own run.
    cands = []
    for strength in (2, 3):
        bold = cv2.erode(ink, np.ones((strength, strength), np.uint8))
        runs = cv2.morphologyEx(bold, cv2.MORPH_CLOSE, np.ones((1, max(3, int(0.7 * h_ref))), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(runs, connectivity=8)
        for x, y, w, h, _area in stats[1:n]:
            if not (0.5 * h_ref <= h <= 1.8 * h_ref and 0.35 * len(target) * h <= w <= 1.3 * len(target) * h):
                continue
            box = (max(0, int(x) - 3), int(y), int(x + w) + 3, int(y + h) + 1)      # snug: a label sits just above
            if not any(_overlap(box, t) > 0.2 for t in taken):
                cands.append(box)
    best = None
    for x0, y0, x1, y1 in cands[:30]:
        # A single-word reading is sensitive to its margins (a label line just above the value
        # can spoil it), so a few framings are tried and the best one counts.
        for pad in (0, 5, 10):
            crop = img[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad - 4):min(W, x1 + pad + 4)]
            for view in (crop, cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)):
                text = vnorm(read_word(view, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")).replace(" ", "")
                score = fuzz.ratio(text, target)
                if score >= 85 and (best is None or score > best[0]):
                    best = (score, (x0, y0, x1, y1))
        if best and best[0] >= 95:
            break
    if best:
        rec.fields.append(Field(ftype=ftype, value=value, box=best[1], source="mrz_hunt", confidence=best[0] / 100))


def _passport_extras(rec: PageRecord, lines) -> None:
    """Values that do not depend on one template's label wording."""
    have = {f.ftype for f in rec.fields}
    if "doc_number" not in have:
        # The number label is often lost in the header print; the value itself is distinctive.
        top = [w for line in lines for w in line if "<" not in w.text and PASSPORT_NUMBER.match(w.text)]
        if len(top) == 1:
            _add(rec, "doc_number", top, "pattern", 0.9)
    hit = find_label(lines, "Sex")
    if hit:
        words = value_below(lines, hit[0], hit[1])
        if words and words[0].text.upper() in ("F", "M"):
            rec.sex = words[0].text.upper()        # kept on the page; steers the choice of surrogate names


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
        text = _trim(text)
        if f.ftype == "dob" and parse_viz(f.value):
            continue                    # the page reading already is a date; a re-read may add dots
        if text and fuzz.ratio(vnorm(text), vnorm(f.value)) >= 70:
            f.value, f.confidence = text, round(conf / 100, 3)
    if rec.mrz and rec.mrz.line1_only:
        _reconcile_line1(rec)
    if rec.mrz and rec.mrz.valid:
        m = rec.mrz
        by_type = {f.ftype: f for f in rec.fields}
        for f in rec.fields:
            if f.ftype == "doc_number":
                f.value = m.doc_number
            elif f.ftype == "dob":
                # The check digit vouches for the date; the page decides how it is written.
                read = parse_viz(f.value)
                yy, mm, dd = int(m.dob[:2]), int(m.dob[2:4]), int(m.dob[4:6])
                if read:
                    century = read[0].year - read[0].year % 100
                    try:
                        f.value = format_viz(date(century + yy, mm, dd), read[1])
                    except ValueError:
                        pass
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


def _reconcile_line1(rec: PageRecord) -> None:
    """MRZ line 1 without line 2 has no check digits, so the visual zone confirms the name."""
    m = rec.mrz
    by_type = {f.ftype: f for f in rec.fields}
    sur, giv = m.surname, m.given_names
    agree = True
    for ftype, mrz_value in (("surname", sur), ("given_names", giv)):
        if ftype not in by_type or fuzz.ratio(vnorm(by_type[ftype].value), vnorm(mrz_value)) < 80:
            agree = False
    if agree:
        sur, giv = vnorm(by_type["surname"].value), vnorm(by_type["given_names"].value)
    else:
        rec.review_reasons.append("mrz_name_unconfirmed")
    m.surname, m.given_names = sur, giv
    rec.fields.append(Field(ftype="mrz_line1", value=build_line1(m.doc_code or "P", m.issuer, sur, giv),
                            box=m.line_boxes[0], source="mrz"))


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
