"""Operator-chosen name: only the name is chosen; everything else stays generated."""
import pytest

from kycpseudo import service
from kycpseudo.config import Config
from kycpseudo.models import Entity, Field, PageRecord
from kycpseudo.names import format_like
from kycpseudo.stages import s6_generate as s6


def _cfg(custom=None):
    cfg = Config()
    cfg.hmac_key = b"k" * 32
    cfg.custom_name = custom
    return cfg


def _entity(i=0, surname="TEST", given=("ANGELA", "ZOE")):
    return Entity(entity_id=f"P-{i:04d}", mention_ids=[f"d{i}:holder"], surname=surname, given=list(given), sex="F")


def test_input_is_validated():
    assert service.parse_custom_name("", "  ") is None
    assert service.parse_custom_name(" de silva ", "maria  ana") == ("DE SILVA", ("MARIA", "ANA"))
    for sur, giv in (("SMITH", ""), ("", "ANNA"), ("SM1TH", "ANNA"), ("O'NEIL", "ANNA"),
                     ("A B C", "ANNA"), ("SMITH", "A B C D"), ("X" * 21, "ANNA")):
        with pytest.raises(service.ServiceError):
            service.parse_custom_name(sur, giv)


def test_chosen_name_replaces_only_the_name():
    e = _entity()
    auto = s6.make_identities([e], _cfg(), set())[e.entity_id]
    ident = s6.make_identities([e], _cfg(("KUMAR", ("RAVI",))), set())[e.entity_id]
    assert (ident.surname, ident.given, ident.custom) == ("KUMAR", ["RAVI"], True)
    assert ident.dob_shift_days == auto.dob_shift_days       # dates are still key-derived
    rec = PageRecord(doc_id="d0", path="x", doc_type="passport")
    cfg = _cfg(("KUMAR", ("RAVI",)))
    box = (0, 0, 10, 10)
    assert s6.surrogate_for(Field("surname", "TEST", box, "label"), rec, e, ident, cfg) == "KUMAR"
    assert s6.surrogate_for(Field("given_names", "ANGELA ZOE", box, "label"), rec, e, ident, cfg) == "RAVI"
    assert s6.surrogate_for(Field("full_name", "Angela Zoe Test", box, "label"), rec, e, ident, cfg) == "Ravi Kumar"
    number = s6.surrogate_for(Field("doc_number", "999237144", box, "label"), rec, e, ident, cfg)
    assert number == s6.surrogate_for(Field("doc_number", "999237144", box, "label"), rec, e, auto, _cfg())


def test_chosen_name_is_not_shared_between_two_people():
    ents = [_entity(0), _entity(1, "MALMBERG", ("OSKAR",))]
    idents = s6.make_identities(ents, _cfg(("KUMAR", ("RAVI",))), set())
    assert not any(i.custom for i in idents.values())
    assert all(i.surname != "KUMAR" for i in idents.values())


def test_fewer_given_names_than_the_original():
    g, s = ["ANNA", "MARIA"], "ERIKSSON"
    assert format_like("ERIKSSON, Anna Maria", g, s, ["EVA"], "LINDQVIST") == "LINDQVIST, Eva"
    assert format_like("A. M. ERIKSSON", g, s, ["EVA"], "LINDQVIST") == "E. LINDQVIST"
    assert format_like("ERIKSSON<<ANNA<MARIA", g, s, ["EVA"], "LINDQVIST") == "LINDQVIST<<EVA"


@pytest.mark.slow
@pytest.mark.skipif(__import__("shutil").which("tesseract") is None, reason="needs tesseract")
def test_kept_photo_is_always_watermarked_and_public_callers_may_choose_a_name(tmp_path):
    import base64, io
    import numpy as np
    from PIL import Image

    images = service.make_samples(tmp_path / "samples")
    passport = next(p for p in sorted(images.glob("*.png")) if "passport" in p.name.lower() or p.name == "DOC-0000.png")
    files = [(passport.name, passport.read_bytes())]
    key = base64.urlsafe_b64encode(b"k" * 32).decode()
    name = service.parse_custom_name("Kumar", "Ravi")

    def top_rows(doc):
        return np.array(Image.open(io.BytesIO(base64.b64decode(doc["image_png_b64"]))).convert("RGB"))[:20]

    clean = service.process(files, True, hmac_key_b64=key, vault_key=key, custom_name=name)["documents"][0]
    kept = service.process(files, True, hmac_key_b64=key, vault_key=key, custom_name=name, keep_photo=True)["documents"][0]
    assert clean["doc_type"] == "passport" and not clean["watermarked"]
    assert kept["watermarked"]                                  # operator token does not lift it
    assert top_rows(kept)[..., 0].mean() > 150 and top_rows(kept)[..., 1].mean() < 100      # red banner
    assert "region_not_replaced" not in kept["review_reasons"]

    public = service.process(files, False, custom_name=name)
    assert public["custom_name"] == "applied" and public["documents"][0]["watermarked"]
    assert any(f["surrogate"] == "KUMAR" for f in public["documents"][0]["fields"])
