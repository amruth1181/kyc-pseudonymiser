"""One-call service layer shared by the Streamlit app and the HTTP engine.

`process` takes uploaded files as bytes and returns each page pseudonymised,
as a plain dict. It owns the access model:

  * public callers: output is watermarked; a fresh random key is used per call,
    so pseudonyms cannot be linked across calls or reversed; no vault returned.
  * operator callers: clean output, pseudonyms from the persistent keys
    (consistent across calls), and the encrypted vault.

Nothing is stored: files are processed in a temporary directory that is deleted
before the result is returned.
"""
from __future__ import annotations

import base64
import hmac
import io
import json
import secrets
import tempfile
import time
from pathlib import Path

import re

import numpy as np
from cryptography.fernet import Fernet
from PIL import Image

from .config import Config
from .pipeline import run
from .stages.s9_release import open_vault
from .synth.generator import generate
from .watermark import apply_watermark

MAX_PIXELS = 25_000_000
ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pdf"}


class ServiceError(Exception):
    """A request the caller can fix. `status` is the matching HTTP status code."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def token_matches(given: str | None, expected: str | None) -> bool:
    """Constant-time check of an access token. An unset token never matches."""
    if not given or not expected:
        return False
    return hmac.compare_digest(given.strip().encode(), expected.encode())


def make_samples(samples_dir: Path, n_people: int = 4, seed: int = 11) -> Path:
    """Generate the synthetic sample documents once; returns the images folder."""
    samples_dir = Path(samples_dir)
    if not (samples_dir / "images").exists():
        generate(samples_dir, n_people=n_people, seed=seed)
        (samples_dir / "truth.json").unlink(missing_ok=True)   # no ground truth on the server
    return samples_dir / "images"


_NAME_PART = re.compile(r"^[A-Za-z]{1,20}$")


def parse_custom_name(surname: str, given_names: str) -> tuple[str, tuple[str, ...]] | None:
    """Validate an operator-chosen name. Both parts empty means 'generate one'."""
    sur, giv = surname.split(), given_names.split()
    if not sur and not giv:
        return None
    if not sur or not giv:
        raise ServiceError(400, "enter both a surname and at least one given name, or leave both empty")
    if len(sur) > 2 or len(giv) > 3 or not all(_NAME_PART.match(p) for p in sur + giv):
        raise ServiceError(400, "names may use the letters A-Z only: up to 2 words for the surname, "
                                "up to 3 given names, 20 letters each")
    return " ".join(sur).upper(), tuple(g.upper() for g in giv)


def _config(operator: bool, hmac_key_b64: str | None, vault_key: str | None) -> Config:
    cfg = Config()
    if operator and hmac_key_b64 and vault_key:
        cfg.hmac_key = base64.urlsafe_b64decode(hmac_key_b64)
        cfg.vault_key = vault_key.encode()
    else:
        # Throwaway keys: pseudonyms differ every call and cannot be reversed.
        cfg.hmac_key = secrets.token_bytes(32)
        cfg.vault_key = Fernet.generate_key()
    return cfg


def _png_b64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def process(files: list[tuple[str, bytes]], operator: bool = False, *,
            hmac_key_b64: str | None = None, vault_key: str | None = None,
            max_files: int = 8, max_bytes: int = 8 * 1024 * 1024,
            custom_name: tuple[str, tuple[str, ...]] | None = None, keep_photo: bool = False) -> dict:
    """Pseudonymise `files` (name, content) and return the result as a dict.

    `custom_name` (from `parse_custom_name`) replaces the generated name when the files hold
    one person. Document number, dates and places are always generated, never chosen.

    `keep_photo` leaves the portrait on passports untouched. A real face next to a new
    identity is what a forged document looks like, so such a page is always watermarked,
    for operators too. The same holds for a passport whose photo could not be replaced
    when a chosen name is on it.
    """
    if not files or len(files) > max_files:
        raise ServiceError(400, f"send between 1 and {max_files} files")

    t0 = time.time()
    cfg = _config(operator, hmac_key_b64, vault_key)
    cfg.custom_name = custom_name
    cfg.keep_photo = keep_photo
    with tempfile.TemporaryDirectory(prefix="kyc-") as tmp:
        inp, out = Path(tmp) / "in", Path(tmp) / "out"
        inp.mkdir()
        originals: dict[str, str] = {}
        for i, (name, data) in enumerate(files):
            ext = Path(name or "").suffix.lower()
            if ext not in ALLOWED:
                raise ServiceError(400, f"unsupported file type: {ext or 'none'}")
            if len(data) > max_bytes:
                raise ServiceError(413, f"file {i + 1} is larger than {max_bytes // (1024 * 1024)} MB")
            if ext != ".pdf":
                try:
                    with Image.open(io.BytesIO(data)) as im:
                        pixels = im.width * im.height
                except Exception:
                    raise ServiceError(400, f"file {i + 1} is not a readable image") from None
                if pixels > MAX_PIXELS:
                    raise ServiceError(413, f"file {i + 1} has too many pixels")
            stem = f"upload-{i + 1:02d}"        # never reuse client filenames (they often contain names)
            (inp / f"{stem}{ext}").write_bytes(data)
            originals[stem] = name or stem

        summary = run(inp, out, cfg, workers=1, use_cache=False)
        vault = open_vault(out / "vault.enc", cfg)
        audits = {a["doc_id"]: a for a in map(json.loads, (out / "audit.jsonl").read_text().splitlines())}

        name_applied = bool(cfg.custom_name) and summary["entities"] == 1
        docs = []
        for doc_id, d in sorted(vault["documents"].items()):
            folder = "pseudonymised" if d["status"] == "passed" else "review"
            rgb = np.array(Image.open(out / folder / d["output"]).convert("RGB"))
            reasons = audits[d["output"]]["review_reasons"]
            # The original face never goes out clean: not when it was kept on request, and not
            # with a chosen name when the photo could not be replaced.
            unreplaced = "photo_not_found" in reasons or "region_not_replaced" in reasons
            face_kept = d["doc_type"] == "passport" and (keep_photo or (name_applied and unreplaced))
            marked = not operator or face_kept
            if marked:
                rgb = apply_watermark(rgb)
            docs.append({
                "upload": originals.get(Path(d["input"]).stem.split("-p")[0], doc_id),
                "output_name": d["output"],
                "doc_type": d["doc_type"],
                "status": d["status"],
                "review_reasons": reasons,
                "watermarked": marked,
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
            "custom_name": ("applied" if name_applied else "ignored_several_people") if cfg.custom_name else None,
        }
        if operator:
            result["vault_b64"] = base64.b64encode((out / "vault.enc").read_bytes()).decode()
    return result
