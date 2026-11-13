"""Data records passed between pipeline stages.

Every stage reads and writes plain dataclasses so that any stage can be
re-run, inspected or replaced on its own. Original values live only in
memory and in the encrypted vault; audit logs carry types and boxes only.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 in page pixels


@dataclass
class Word:
    text: str
    box: Box
    conf: float
    line_id: tuple[int, int, int]  # (block, paragraph, line) from the OCR engine


@dataclass
class Field:
    """One personal value found on a page."""
    ftype: str                 # surname, given_names, full_name, doc_number, dob, address, ...
    value: str                 # original text as read
    box: Box
    source: str                # which detector found it: mrz_anchor, label, regex, ...
    confidence: float = 1.0
    entity_id: Optional[str] = None
    mention_id: Optional[str] = None
    surrogate: Optional[str] = None  # text to render, set in stage 6


@dataclass
class Region:
    """A non-text identifier: face photo, signature, barcode."""
    rtype: str
    box: Box
    entity_id: Optional[str] = None


@dataclass
class MRZData:
    doc_code: str
    issuer: str
    surname: str
    given_names: str
    doc_number: str
    nationality: str
    dob: str          # YYMMDD
    sex: str
    expiry: str       # YYMMDD
    personal: str
    valid: bool
    composite_ok: bool = True
    line_boxes: list[Box] = field(default_factory=list)
    raw: tuple[str, str] = ("", "")
    line1_only: bool = False     # line 2 is not on the image (cropped scan): names only, no check digits


@dataclass
class PageRecord:
    doc_id: str
    path: str
    sha256: str = ""
    doc_type: str = "unknown"
    type_confidence: float = 0.0
    width: int = 0
    height: int = 0
    skew_deg: float = 0.0
    words: list[Word] = field(default_factory=list)
    mrz: Optional[MRZData] = None
    sex: Optional[str] = None        # read from the visual zone; kept, used to pick surrogate names
    fields: list[Field] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    edits: list[dict] = field(default_factory=list)
    verify: dict = field(default_factory=dict)
    status: str = "pending"          # pending, passed, review
    review_reasons: list[str] = field(default_factory=list)

    def audit_view(self) -> dict:
        """What may be written to logs: no original or surrogate values."""
        return {
            "doc_id": self.doc_id,
            "sha256": self.sha256,
            "doc_type": self.doc_type,
            "skew_deg": round(self.skew_deg, 2),
            "fields": [
                {"type": f.ftype, "box": f.box, "source": f.source, "entity": f.entity_id}
                for f in self.fields
            ],
            "regions": [{"type": r.rtype, "box": r.box, "entity": r.entity_id} for r in self.regions],
            "verify": self.verify,
            "status": self.status,
            "review_reasons": self.review_reasons,
        }


@dataclass
class Mention:
    """A name occurrence, the unit of entity resolution."""
    mention_id: str
    doc_id: str
    text: str
    normalized: str
    doc_type: str
    dob: Optional[str] = None
    doc_number: Optional[str] = None
    address: Optional[str] = None
    sex: Optional[str] = None
    given: Optional[str] = None      # known split, e.g. from the MRZ
    surname: Optional[str] = None


@dataclass
class Entity:
    entity_id: str
    mention_ids: list[str]
    surname: str
    given: list[str]
    sex: Optional[str] = None
    dob: Optional[str] = None
    locale: str = "generic"
    variants: list[str] = field(default_factory=list)   # every normalised spelling seen

    def surname_forms(self) -> set[str]:
        """Surname spellings across mentions (tokens that are not one of the given names)."""
        from rapidfuzz.distance import JaroWinkler
        forms = {self.surname}
        for v in self.variants:
            for t in v.split():
                if len(t) >= 3 and all(JaroWinkler.similarity(t, g) < 0.92 for g in self.given):
                    forms.add(t)
        return forms


@dataclass
class Identity:
    """The surrogate identity for one entity."""
    entity_id: str
    surname: str
    given: list[str]
    sex: Optional[str]
    dob_shift_days: int


def to_dict(obj) -> dict:
    return asdict(obj)
