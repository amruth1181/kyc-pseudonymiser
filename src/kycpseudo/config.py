"""Pipeline settings and secrets.

Two secrets, kept apart from the data:
  * the HMAC key that derives every surrogate (losing it breaks consistency
    across runs; leaking it lets someone recompute pseudonyms),
  * the vault key that encrypts the entity -> original mapping.
Both come from the environment. ``demo_keys`` creates throwaway keys in
.keys/ for local experiments only.
"""
from __future__ import annotations

import base64
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.fernet import Fernet


@dataclass
class Config:
    dpi: int = 300
    ocr_lang: str = "eng"
    # Entity resolution
    bm25_ngram: int = 3
    bm25_top_k: int = 10
    match_accept: float = 0.92      # name similarity accepted as same person
    match_review: float = 0.80      # below accept and above this: human review
    # Surrogates
    dob_shift_min_days: int = 30
    dob_shift_max_days: int = 400
    keep_fields: tuple[str, ...] = ("nationality", "issuer", "sex", "expiry", "issue_date", "city")
    # Rendering
    inpaint_radius: int = 3
    box_pad: int = 3
    use_lama: bool = False          # needs the optional 'lama' extra
    # Verification
    cer_max: float = 0.10           # max character error rate per rendered field
    leak_ratio: float = 90.0        # fuzzy ratio at which an original token counts as leaked
    # Keys
    hmac_key: bytes = field(default=b"", repr=False)
    vault_key: bytes = field(default=b"", repr=False)


def load_keys(cfg: Config, demo_keys: bool = False, keydir: Path = Path(".keys")) -> Config:
    hk = os.environ.get("KYCPSEUDO_HMAC_KEY")
    vk = os.environ.get("KYCPSEUDO_VAULT_KEY")
    if hk and vk:
        cfg.hmac_key = base64.urlsafe_b64decode(hk)
        cfg.vault_key = vk.encode()
        return cfg
    if not demo_keys:
        raise SystemExit(
            "Set KYCPSEUDO_HMAC_KEY and KYCPSEUDO_VAULT_KEY, or pass --demo-keys for a local test run."
        )
    keydir.mkdir(exist_ok=True)
    hpath, vpath = keydir / "hmac.key", keydir / "vault.key"
    if not hpath.exists():
        hpath.write_bytes(base64.urlsafe_b64encode(secrets.token_bytes(32)))
        vpath.write_bytes(Fernet.generate_key())
    cfg.hmac_key = base64.urlsafe_b64decode(hpath.read_bytes())
    cfg.vault_key = vpath.read_bytes()
    return cfg
