from kycpseudo.mrz import build_td3, check_digit, parse_td3, repair_td3

# ICAO Doc 9303 specimen passport (fictional state Utopia).
SPEC_L1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
SPEC_L2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"


def test_check_digit_specimen():
    assert check_digit("L898902C3") == "6"
    assert check_digit("740812") == "2"
    assert check_digit("120415") == "9"


def test_build_reproduces_icao_specimen():
    assert build_td3("P", "UTO", "ERIKSSON", "ANNA MARIA", "L898902C3", "UTO",
                     "740812", "F", "120415", "ZE184226B") == (SPEC_L1, SPEC_L2)


def test_parse_specimen():
    m = parse_td3(SPEC_L1, SPEC_L2)
    assert m.valid and m.composite_ok
    assert (m.surname, m.given_names, m.doc_number, m.dob) == ("ERIKSSON", "ANNA MARIA", "L898902C3", "740812")


def test_repair_undoes_single_ocr_confusion():
    garbled = "L898902C36UT07408122F12O4159ZE184226B<<<<<10"   # 0 read as O in the expiry date
    m = repair_td3(SPEC_L1, garbled)
    assert m.valid and m.expiry == "120415"


def test_repair_handles_letter_for_digit_in_document_number():
    l1, l2 = build_td3("P", "UTO", "X", "Y", "B0042716G", "UTO", "580830", "M", "290414")
    garbled = l2[:1] + "Q" + l2[2:]                      # 0 read as Q inside an alphanumeric field
    m = repair_td3(l1, garbled)
    assert m.valid and m.doc_number == "B0042716G"


def test_garbled_filler_tail_is_rebuilt():
    l1, l2 = build_td3("P", "UTO", "X", "Y", "Y1049746Y", "UTO", "720823", "M", "290414")
    m = parse_td3(l1, l2[:28] + "<" * 13 + "8")         # one filler dropped, check digit lost
    assert m.valid
