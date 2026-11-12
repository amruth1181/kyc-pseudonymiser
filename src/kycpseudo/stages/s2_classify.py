"""Stage 2: classify the document type.

The type decides which detectors, fonts and surrogate rules apply later.
This implementation uses transparent rules over the OCR text; a trained
image classifier (or a vision-language model used only as a classifier)
can replace ``classify`` with the same signature. Unknown types are sent
to human review rather than processed with the wrong rules.
"""
from __future__ import annotations

from ..layout import build_lines, text_of
from ..models import PageRecord

RULES = [
    ("passport", lambda t, mrz: mrz, 0.95),
    ("bank_statement", lambda t, mrz: "STATEMENT" in t and ("IBAN" in t or "ACCOUNT HOLDER" in t), 0.9),
    ("utility_bill", lambda t, mrz: ("BILL" in t or "KWH" in t) and "CUSTOMER" in t, 0.9),
]


def looks_like_mrz(line: str) -> bool:
    s = line.replace(" ", "")
    return len(s) >= 30 and s.count("<") >= 3


def classify(rec: PageRecord) -> PageRecord:
    lines = [text_of(l) for l in build_lines(rec.words)]
    text = " ".join(lines).upper()
    has_mrz = sum(looks_like_mrz(l) for l in lines) >= 1 and any(l.replace(" ", "").startswith("P<") for l in lines)
    for name, rule, conf in RULES:
        if rule(text, has_mrz):
            rec.doc_type, rec.type_confidence = name, conf
            return rec
    rec.doc_type, rec.type_confidence = "unknown", 0.0
    rec.review_reasons.append("unknown_document_type")
    return rec
