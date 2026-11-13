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


def test_small_scans_are_enlarged_once_at_ingest():
    from kycpseudo.stages.s1_ingest import enlarge_small
    small, factor = enlarge_small(np.zeros((432, 768, 3), np.uint8))
    assert factor == 2 and small.shape[:2] == (864, 1536)
    page = np.zeros((1060, 1500, 3), np.uint8)
    same, factor = enlarge_small(page)
    assert factor == 1 and same is page


def test_extra_ocr_passes_only_fill_gaps(monkeypatch):
    def fake(img, lang, psm):
        if psm == 3 and img.shape[1] == 1500:                 # the page reading
            return [Word("Surname", (100, 100, 200, 120), 90.0, (0, 0, 0))]
        if psm == 11 and img.shape[1] == 1500:                # sparse-text pass
            return [Word("Sumame", (101, 100, 199, 120), 95.0, (0, 0, 0)),      # same spot: ignored
                    Word("JC3L7T2H", (700, 100, 820, 120), 70.0, (0, 0, 0)),    # new spot: kept
                    Word("~", (900, 100, 905, 120), 99.0, (0, 0, 0)),           # a speck: ignored
                    Word("gue55", (300, 300, 380, 320), 20.0, (0, 0, 0))]       # a guess: ignored
        return []

    monkeypatch.setattr(ocr, "_read_words", fake)
    page = np.zeros((1060, 1500, 3), np.uint8)
    assert [w.text for w in ocr.read_words(page)] == ["Surname"]
    assert [w.text for w in ocr.read_words(page, thorough=True)] == ["Surname", "JC3L7T2H"]


def test_mrz_keeps_a_personal_number_and_ignores_misread_fillers():
    from kycpseudo.mrz import parse_td3
    m = parse_td3("P<LUXSPECIMEN<<VANESSA<X<<<<<<<<<KK<K<<<<<<<<", "JC3L7T2H<4LUX9303092F200122901150000542<<<88")
    assert (m.surname, m.given_names) == ("SPECIMEN", "VANESSA")
    assert m.valid and m.personal == "01150000542"
    m = parse_td3("P<FRASPECIMEN<ABB<<NATACHA<<<<<<<<<<<<<<KK<K", "90RF023318FRA7307122F2903035<<<<<<<<<<<<<<02")
    assert (m.surname, m.given_names, m.valid) == ("SPECIMEN ABB", "NATACHA", True)


def test_numeric_dates_and_accents():
    from kycpseudo.dates import format_viz, parse_viz
    from kycpseudo.layout import vnorm
    from kycpseudo.mrz import viz_date_forms
    assert _shift_viz_date("09 03 1993", 30) == "08 04 1993"          # France, Luxembourg
    assert _shift_viz_date("09/03/1993", 30) == "08/04/1993"          # India
    assert _shift_viz_date("09.03.1993", -9) == "28.02.1993"
    d, layout = parse_viz("12 07 1973")
    assert format_viz(d, layout) == "12 07 1973"
    assert "12 07 1973" in viz_date_forms("730712")
    assert vnorm("Spécimen") == "SPECIMEN"


def test_trilingual_label_value_lines_up_with_the_label_start():
    lines = [_line(0, (360, "1."), (385, "Numm"), (440, "/"), (455, "Nom"), (495, "/"), (510, "Surname")),
             _line(22, (362, "Maus"))]
    li, words = find_label(lines, "Surname")
    assert words[0].text == "Surname"                         # matched on the English part, mid-label
    assert [w.text for w in value_below(lines, li, words)] == ["Maus"]
