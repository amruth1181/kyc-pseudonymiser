from kycpseudo.config import Config
from kycpseudo.models import Field, MRZData, Mention, PageRecord
from kycpseudo.mrz import parse_td3
from kycpseudo.names import normalize
from kycpseudo.stages import s5_resolve as s5
from kycpseudo.stages import s6_generate as s6


def _m(mid, text, **kw):
    return Mention(mid, mid.split(":")[0], text, normalize(text), "x", **kw)


def _cfg(key=b"k" * 32):
    c = Config()
    c.hmac_key = key
    return c


def test_twins_split_by_address():
    # Oskar and Olof Malmberg both appear as 'O. MALMBERG' on a statement.
    ms = [
        _m("P1:holder", "OSKAR MALMBERG", dob="580227", surname="MALMBERG", given="OSKAR"),
        _m("B1:holder", "Oskar Malmberg", address="41 HILL CREST NV 3961"),
        _m("S1:holder", "O. MALMBERG", address="41 HILL CREST NV 3961"),
        _m("P2:holder", "OLOF MALMBERG", dob="631105", surname="MALMBERG", given="OLOF"),
        _m("B2:holder", "Olof Malmberg", address="85 QUAY STREET NV 2210"),
        _m("S2:holder", "O. MALMBERG", address="85 QUAY STREET NV 2210"),
    ]
    roots, review = s5.resolve(ms, _cfg())
    assert roots["P1:holder"] == roots["B1:holder"] == roots["S1:holder"]
    assert roots["P2:holder"] == roots["B2:holder"] == roots["S2:holder"]
    assert roots["P1:holder"] != roots["P2:holder"]


def test_ambiguous_initials_go_to_review_not_merged():
    ms = [
        _m("A:holder", "Oskar Malmberg", address="41 HILL CREST"),
        _m("B:holder", "Olof Malmberg", address="85 QUAY STREET"),
        _m("C:holder", "O. MALMBERG"),                     # no evidence
    ]
    roots, review = s5.resolve(ms, _cfg())
    assert len({roots[m.mention_id] for m in ms}) == 3
    assert any(r["type"] == "ambiguous_initials" for r in review)


def test_dob_conflict_blocks_merge():
    ms = [_m("A:holder", "ANNA ERIKSSON", dob="740812"), _m("B:holder", "ANNA ERIKSSON", dob="810101")]
    roots, _ = s5.resolve(ms, _cfg())
    assert roots["A:holder"] != roots["B:holder"]


def _entities():
    ms = [_m("P:holder", "ANNA MARIA ERIKSSON", dob="740812", surname="ERIKSSON", given="ANNA MARIA", sex="F"),
          _m("B:holder", "Anna Maria Eriksson")]
    roots, _ = s5.resolve(ms, _cfg())
    return ms, s5.build_entities(ms, roots)


def test_identities_are_deterministic_per_key():
    ms, ents = _entities()
    toks = {"ANNA", "MARIA", "ERIKSSON"}
    a = s6.make_identities(ents, _cfg(), toks)
    b = s6.make_identities(ents, _cfg(), toks)
    c = s6.make_identities(ents, _cfg(b"z" * 32), toks)
    eid = ents[0].entity_id
    assert a[eid] == b[eid]
    assert (a[eid].surname, a[eid].given) != (c[eid].surname, c[eid].given)
    assert a[eid].surname != "ERIKSSON" and not set(a[eid].given) & toks
    assert len(a[eid].given) == 2 and a[eid].sex == "F"


def _surrogate(ftype, value, rec=None):
    ms, ents = _entities()
    ident = s6.make_identities(ents, _cfg(), set())[ents[0].entity_id]
    f = Field(ftype, value, (0, 0, 1, 1), "test", entity_id=ents[0].entity_id)
    return s6.surrogate_for(f, rec or PageRecord("d", "p"), ents[0], ident, _cfg())


def test_iban_check_digits_recomputed():
    new = _surrogate("iban", "UT15 3909 9603 0824 6281")
    compact = new.replace(" ", "")
    num = "".join(str(int(c, 36)) for c in compact[4:] + compact[:4])
    assert int(num) % 97 == 1 and len(new) == len("UT15 3909 9603 0824 6281")


def test_document_number_keeps_format_and_avoids_confusable_letters():
    new = _surrogate("doc_number", "U1590830C")
    assert len(new) == 9 and new[0].isalpha() and new[1:8].isdigit() and new[8].isalpha()
    assert not set(new) & set("OIQ")


def test_name_pattern_preserved():
    assert _surrogate("full_name", "A. M. ERIKSSON").count(".") == 2


def test_mrz_rebuilt_with_valid_check_digits():
    l1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
    l2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
    rec = PageRecord("d", "p")
    rec.mrz = parse_td3(l1, l2)
    new1, new2 = _surrogate("mrz_line1", l1, rec), _surrogate("mrz_line2", l2, rec)
    m = parse_td3(new1, new2)
    assert m.valid and m.composite_ok
    assert m.expiry == "120415" and m.sex == "F"          # kept attributes
    assert m.surname != "ERIKSSON" and m.dob != "740812"


def test_dob_shift_is_consistent():
    assert _surrogate("dob", "12 AUG 1974") == _surrogate("dob", "12 AUG 1974") != "12 AUG 1974"
