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

import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import Response

from kycpseudo import service

log = logging.getLogger("engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

MAX_FILES = int(os.environ.get("MAX_FILES", 8))
MAX_BYTES = int(os.environ.get("MAX_FILE_MB", 8)) * 1024 * 1024
PUBLIC_PER_HOUR = int(os.environ.get("PUBLIC_REQUESTS_PER_HOUR", 20))
SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR", "/tmp/kyc-samples"))

app = FastAPI(title="kycpseudo engine", version="0.1.0")
_hits: dict[str, deque] = defaultdict(deque)


@app.on_event("startup")
def _make_samples() -> None:
    service.make_samples(SAMPLES_DIR)


def _is_operator(authorization: str | None) -> bool:
    if not authorization or not authorization.startswith("Bearer "):
        return False
    return service.token_matches(authorization[7:], os.environ.get("ENGINE_TOKEN"))


def _rate_limit(ip: str) -> None:
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= PUBLIC_PER_HOUR:
        raise HTTPException(429, f"limit of {PUBLIC_PER_HOUR} requests per hour reached")
    q.append(now)


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
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"send between 1 and {MAX_FILES} files")
    payload = [(f.filename or "", await f.read(MAX_BYTES + 1)) for f in files]
    try:
        result = service.process(payload, operator,
                                 hmac_key_b64=os.environ.get("KYCPSEUDO_HMAC_KEY"),
                                 vault_key=os.environ.get("KYCPSEUDO_VAULT_KEY"),
                                 max_files=MAX_FILES, max_bytes=MAX_BYTES)
    except service.ServiceError as e:
        raise HTTPException(e.status, str(e)) from None
    log.info("request mode=%s files=%d seconds=%.1f", result["mode"], len(files), result["seconds"])
    return result
