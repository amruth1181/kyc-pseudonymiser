"""Layouts beyond the synthetic template: bilingual labels and dates, a cropped MRZ, OCR-B."""
from pathlib import Path

import numpy as np

from kycpseudo import ocr
from kycpseudo.fonts import font_candidates, font_path
from kycpseudo.layout import find_label, value_below
from kycpseudo.models import Word
from kycpseudo.mrz import build_line1, build_td3, parse_line1
from kycpseudo.stages.s6_generate import _shift_viz_date


def _line(y, *words):
    return [Word(text, (x, y, x + 9 * len(text), y + 12), 95.0, (0, 0, 0)) for x, text in words]


def test_bilingual_label_is_found_at_its_own_first_word():
    lines = [_line(0, (240, "Sex/Sexe"), (290, "(5)"), (316, "Place"), (345, "of"), (359, "birth/Lieu"), (460, "de")),
             _line(16, (245, "F"), (320, "LONDON"))]
    li, words = find_label(lines, "Place of birth")
    assert words[0].text == "Place"                       # not the '(5)' before it
    assert [w.text for w in value_below(lines, li, words)] == ["LONDON"]
    assert [w.text for w in value_below(lines, *find_label(lines, "Sex"))] == ["F"]


def test_date_shift_keeps_the_printed_layout():
    assert _shift_viz_date("12 AUG 1974", 30) == "11 SEP 1974"
    assert _shift_viz_date("18 SEP /SEPT 99", 36) == "24 OCT /OCT 99"
    assert _shift_viz_date("28 FEB /FEV 01", 1) == "01 MAR /MARS 01"


def test_unknown_date_layout_gives_no_surrogate():
    # Returning the text unchanged would leave the real date on the page.
    assert _shift_viz_date("1999-09-18", 36) is None
    assert _shift_viz_date("18 XXX 1999", 36) is None


def test_mrz_line1_alone_is_parsed_and_rebuilt():
    line1 = build_td3("P", "GBR", "TEST", "ANGELA ZOE", "999237144", "GBR", "990918", "F", "270313")[0]
    m = parse_line1(line1)
    assert (m.surname, m.given_names, m.issuer) == ("TEST", "ANGELA ZOE", "GBR")
    assert m.line1_only and not m.valid
    assert build_line1(m.doc_code, m.issuer, m.surname, m.given_names) == line1
    assert parse_line1("NOT<AN<MRZ<LINE") is None


def test_ocr_b_is_bundled_and_offered_for_passport_fields():
    assert Path(font_path("mono")).name == "OCRB.otf"
    assert font_candidates("passport", "mrz_line1") == [("mono", True)]
    names = [n for n, fixed in font_candidates("passport", "surname")]
    assert names[0] == "sans_bold" and "mono" in names     # the template's own font is tried first
    assert font_candidates("utility_bill", "full_name") == [("sans", False)]


def test_small_scans_are_enlarged_for_reading_only(monkeypatch):
    seen = {}

    def fake(img, lang, psm):
        seen["shape"] = img.shape
        return [Word("TEST", (100, 40, 201, 61), 90.0, (0, 0, 0))]

    monkeypatch.setattr(ocr, "_read_words", fake)
    words = ocr.read_words(np.zeros((432, 768, 3), np.uint8))
    assert seen["shape"][:2] == (864, 1536)
    assert words[0].box == (50, 20, 101, 31)               # back in the scan's own coordinates
    ocr.read_words(np.zeros((1060, 1500, 3), np.uint8))
    assert seen["shape"][:2] == (1060, 1500)
