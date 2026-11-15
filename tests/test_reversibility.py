"""Reversibility: the key holder can undo a run exactly; nobody else can."""
import base64
import io
import shutil

import numpy as np
import pytest
from cryptography.fernet import Fernet
from PIL import Image

from kycpseudo import service
from kycpseudo.stages.s1_ingest import deskew, estimate_skew
from kycpseudo.stages.s6_generate import Keyed, _identifier, decrypt_digits

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="needs tesseract")


def _img(b64: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def test_document_number_is_recovered_from_the_key_alone():
    key = b"k" * 32
    surrogate = _identifier(Keyed(key), "999237144", "doc_number")
    assert surrogate != "999237144"
    assert decrypt_digits(key, surrogate, "doc_number") == "999237144"
    assert decrypt_digits(b"z" * 32, surrogate, "doc_number") != "999237144"     # another key gets nowhere
    assert decrypt_digits(key, "AB12", "doc_number") is None                      # too short for FF1


@pytest.mark.slow
def test_a_run_is_undone_exactly_and_only_with_the_right_key(tmp_path):
    images = service.make_samples(tmp_path / "samples")
    files = [(p.name, p.read_bytes()) for p in sorted(images.glob("*.png"))[:2]]
    hmac_key, vault_key = base64.urlsafe_b64encode(b"k" * 32).decode(), Fernet.generate_key().decode()

    run = service.process(files, True, hmac_key_b64=hmac_key, vault_key=vault_key)
    vault = base64.b64decode(run["vault_b64"])
    outputs = {d["output_name"]: base64.b64decode(d["image_png_b64"]) for d in run["documents"]}

    restored = service.restore(vault, outputs, vault_key=vault_key, hmac_key_b64=hmac_key)
    assert len(restored) == len(files)
    by_upload = {d["output_name"]: d["upload"] for d in run["documents"]}
    # The page the pipeline works on is the scan after straightening; that is what comes back.
    originals = {}
    for name, data in files:
        scan = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        originals[name] = deskew(scan, estimate_skew(scan))
    for item in restored:
        page = _img(item["restored_png_b64"])
        assert np.array_equal(page, originals[by_upload[item["output_name"]]])   # pixel for pixel
        assert not np.array_equal(page, _img(base64.b64encode(outputs[item["output_name"]]).decode()))
        assert all(f["original"] != f["surrogate"] for f in item["fields"])

    with pytest.raises(service.ServiceError) as wrong_key:
        service.restore(vault, outputs, vault_key=Fernet.generate_key().decode())
    assert wrong_key.value.status == 403

    assert "vault_b64" not in service.process(files[:1], False)                  # public runs: nothing to reverse


@pytest.mark.slow
def test_watermarked_pages_are_reported_not_half_restored(tmp_path):
    images = service.make_samples(tmp_path / "samples")
    first = sorted(images.glob("*.png"))[0]
    key = Fernet.generate_key().decode()
    run = service.process([(first.name, first.read_bytes())], True, vault_key=key,
                          hmac_key_b64=base64.urlsafe_b64encode(b"k" * 32).decode(), keep_photo=True)
    d = run["documents"][0]
    assert d["doc_type"] == "passport" and d["watermarked"]
    item = service.restore(base64.b64decode(run["vault_b64"]), {d["output_name"]: base64.b64decode(d["image_png_b64"])},
                           vault_key=key)[0]
    assert item["restored_png_b64"] is None and "watermark" in item["note"]
    assert item["fields"]                                                         # the mapping is still recovered
