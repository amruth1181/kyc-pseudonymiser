"""Engine API: access model. Needs fastapi and Tesseract; slow (runs the pipeline)."""
import base64
import importlib.util
import io
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(importlib.util.find_spec("fastapi") is None or shutil.which("tesseract") is None,
                       reason="needs fastapi and tesseract"),
]

ENGINE = Path(__file__).resolve().parents[1] / "deploy" / "engine"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from cryptography.fernet import Fernet
    mp = pytest.MonkeyPatch()
    mp.setenv("ENGINE_TOKEN", "test-token")
    mp.setenv("SAMPLES_DIR", str(tmp_path_factory.mktemp("samples")))
    mp.setenv("KYCPSEUDO_HMAC_KEY", base64.urlsafe_b64encode(b"k" * 32).decode())
    mp.setenv("KYCPSEUDO_VAULT_KEY", Fernet.generate_key().decode())
    sys.path.insert(0, str(ENGINE))
    import app as engine
    from fastapi.testclient import TestClient
    with TestClient(engine.app) as c:
        yield c
    mp.undo()


def _files(client, n=2):
    names = client.get("/samples").json()[:n]
    return [("files", (name, client.get(f"/samples/{name}").content, "image/png")) for name in names]


def _img(doc):
    from PIL import Image
    return np.array(Image.open(io.BytesIO(base64.b64decode(doc["image_png_b64"]))).convert("RGB"))


def test_public_requests_are_watermarked_and_unlinkable(client):
    files = _files(client)
    a = client.post("/pseudonymise", files=files).json()
    b = client.post("/pseudonymise", files=files).json()
    assert a["mode"] == "public" and a["watermarked"] and "vault_b64" not in a
    top = _img(a["documents"][0])[:20]
    assert top[..., 0].mean() > 150 and top[..., 1].mean() < 100      # red banner
    sa = [f["surrogate"] for d in a["documents"] for f in d["fields"]]
    sb = [f["surrogate"] for d in b["documents"] for f in d["fields"]]
    assert sa != sb                                                   # throwaway keys per request


def test_token_gives_clean_consistent_output(client):
    files = _files(client)
    h = {"Authorization": "Bearer test-token"}
    a = client.post("/pseudonymise", files=files, headers=h).json()
    b = client.post("/pseudonymise", files=files, headers=h).json()
    assert a["mode"] == "operator" and not a["watermarked"] and a["vault_b64"]
    assert [f["surrogate"] for d in a["documents"] for f in d["fields"]] == \
           [f["surrogate"] for d in b["documents"] for f in d["fields"]]


def test_wrong_token_is_public_and_bad_files_rejected(client):
    r = client.post("/pseudonymise", files=_files(client, 1), headers={"Authorization": "Bearer nope"}).json()
    assert r["mode"] == "public"
    assert client.post("/pseudonymise", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))]).status_code == 400
