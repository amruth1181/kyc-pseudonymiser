"""kycpseudo engine: HTTP API for a Hugging Face Docker Space.

Endpoints
  GET  /health                 liveness and limits
  GET  /samples                list of built-in synthetic documents
  GET  /samples/{name}         one synthetic document (PNG)
  POST /pseudonymise           files=<one or more scans>; optional header
                               Authorization: Bearer <ENGINE_TOKEN>

Access model
  * Anonymous callers: output is watermarked across the page; a fresh random
    key is used per request, so pseudonyms cannot be linked across requests
    or reversed; nothing is returned that allows re-identification.
  * Callers with ENGINE_TOKEN: clean output, pseudonyms from the operator's
    persistent keys (consistent across requests), and the encrypted vault.

Nothing is stored: uploads are processed in a temporary directory that is
deleted before the response is sent. Logs carry counts and timings only.

Space secrets
  ENGINE_TOKEN          access token for clean output (long random string)
  KYCPSEUDO_HMAC_KEY    base64 32-byte key for surrogates (token callers)
  KYCPSEUDO_VAULT_KEY   Fernet key for the vault (token callers)
"""
from __future__ import annotations

import base64
import hmac
import io
import json
import logging
import os
import secrets
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from cryptography.fernet import Fernet
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import Response
from PIL import Image

from kycpseudo.config import Config
from kycpseudo.pipeline import run
from kycpseudo.stages.s9_release import open_vault
from kycpseudo.synth.generator import generate
from kycpseudo.watermark import apply_watermark

log = logging.getLogger("engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

MAX_FILES = int(os.environ.get("MAX_FILES", 8))
MAX_BYTES = int(os.environ.get("MAX_FILE_MB", 8)) * 1024 * 1024
MAX_PIXELS = 25_000_000
PUBLIC_PER_HOUR = int(os.environ.get("PUBLIC_REQUESTS_PER_HOUR", 20))
ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pdf"}
SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR", "/tmp/kyc-samples"))

app = FastAPI(title="kycpseudo engine", version="0.1.0")
_hits: dict[str, deque] = defaultdict(deque)


@app.on_event("startup")
def _make_samples() -> None:
    if not (SAMPLES_DIR / "images").exists():
        generate(SAMPLES_DIR, n_people=4, seed=11)
        (SAMPLES_DIR / "truth.json").unlink(missing_ok=True)   # no ground truth on the server


def _is_operator(authorization: str | None) -> bool:
    token = os.environ.get("ENGINE_TOKEN", "")
    if not token or not authorization or not authorization.startswith("Bearer "):
        return False
    return hmac.compare_digest(authorization[7:].strip(), token)


def _rate_limit(ip: str) -> None:
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= PUBLIC_PER_HOUR:
        raise HTTPException(429, f"limit of {PUBLIC_PER_HOUR} requests per hour reached")
    q.append(now)


def _config(operator: bool) -> Config:
    cfg = Config()
    if operator and os.environ.get("KYCPSEUDO_HMAC_KEY") and os.environ.get("KYCPSEUDO_VAULT_KEY"):
        cfg.hmac_key = base64.urlsafe_b64decode(os.environ["KYCPSEUDO_HMAC_KEY"])
        cfg.vault_key = os.environ["KYCPSEUDO_VAULT_KEY"].encode()
    else:
        # Throwaway keys: pseudonyms differ every request and cannot be reversed.
        cfg.hmac_key = secrets.token_bytes(32)
        cfg.vault_key = Fernet.generate_key()
    return cfg


def _png_b64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "max_files": MAX_FILES, "max_file_mb": MAX_BYTES // (1024 * 1024),
            "token_configured": bool(os.environ.get("ENGINE_TOKEN"))}


@app.get("/samples")
def samples() -> list[str]:
    return sorted(p.name for p in (SAMPLES_DIR / "images").glob("*.png"))


@app.get("/samples/{name}")
def sample(name: str) -> Response:
    path = (SAMPLES_DIR / "images" / Path(name).name)
    if not path.exists():
        raise HTTPException(404, "no such sample")
    return Response(path.read_bytes(), media_type="image/png")


@app.post("/pseudonymise")
async def pseudonymise(request: Request, files: list[UploadFile] = File(...),
                       authorization: str | None = Header(default=None)) -> dict:
    operator = _is_operator(authorization)
    if not operator:
        _rate_limit(request.client.host if request.client else "unknown")
    if not files or len(files) > MAX_FILES:
        raise HTTPException(400, f"send between 1 and {MAX_FILES} files")

    t0 = time.time()
    cfg = _config(operator)
    with tempfile.TemporaryDirectory(prefix="kyc-") as tmp:
        inp, out = Path(tmp) / "in", Path(tmp) / "out"
        inp.mkdir()
        originals: dict[str, str] = {}
        for i, f in enumerate(files):
            ext = Path(f.filename or "").suffix.lower()
            if ext not in ALLOWED:
                raise HTTPException(400, f"unsupported file type: {ext or 'none'}")
            data = await f.read()
            if len(data) > MAX_BYTES:
                raise HTTPException(413, f"file {i + 1} is larger than {MAX_BYTES // (1024 * 1024)} MB")
            if ext != ".pdf":
                try:
                    with Image.open(io.BytesIO(data)) as im:
                        if im.width * im.height > MAX_PIXELS:
                            raise HTTPException(413, f"file {i + 1} has too many pixels")
                except HTTPException:
                    raise
                except Exception:
                    raise HTTPException(400, f"file {i + 1} is not a readable image")
            stem = f"upload-{i + 1:02d}"        # never reuse client filenames (they often contain names)
            (inp / f"{stem}{ext}").write_bytes(data)
            originals[stem] = f.filename or stem

        summary = run(inp, out, cfg, workers=1, use_cache=False)
        vault = open_vault(out / "vault.enc", cfg)

        docs = []
        for doc_id, d in sorted(vault["documents"].items()):
            folder = "pseudonymised" if d["status"] == "passed" else "review"
            rgb = np.array(Image.open(out / folder / d["output"]).convert("RGB"))
            if not operator:
                rgb = apply_watermark(rgb)
            audit = next(json.loads(l) for l in (out / "audit.jsonl").read_text().splitlines()
                         if json.loads(l)["doc_id"] == d["output"])
            docs.append({
                "upload": originals.get(Path(d["input"]).stem.split("-p")[0], doc_id),
                "output_name": d["output"],
                "doc_type": d["doc_type"],
                "status": d["status"],
                "review_reasons": audit["review_reasons"],
                "fields": [{"type": f["type"], "original": f["value"], "surrogate": f["surrogate"],
                            "entity": f["entity"]} for f in d["fields"] if not f["type"].startswith("mrz_")],
                "image_png_b64": _png_b64(rgb),
            })
        result = {
            "mode": "operator" if operator else "public",
            "watermarked": not operator,
            "summary": {k: summary[k] for k in ("documents", "passed", "sent_to_review", "entities")},
            "seconds": round(time.time() - t0, 1),
            "documents": docs,
        }
        if operator:
            result["vault_b64"] = base64.b64encode((out / "vault.enc").read_bytes()).decode()

    log.info("request mode=%s files=%d seconds=%.1f", result["mode"], len(files), result["seconds"])
    return result
