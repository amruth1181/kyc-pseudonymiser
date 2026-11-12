"""Stage 6: generate one consistent surrogate identity per entity.

* Names: HMAC(key, stable entity key) seeds a choice from same-locale,
  same-sex name lists, with lengths close to the original so the new text
  fits the old box. Names already in the corpus are never used as surrogates.
* Identifiers: FF1 format-preserving encryption on the digits (reversible
  with the key), deterministic HMAC substitution for short letter runs.
  IBAN check digits are recomputed (ISO 13616 mod-97).
* Dates of birth: shifted by a per-person offset; issue and expiry are kept
  so that whether a document is expired does not change.
* MRZ: rebuilt from the surrogate values with fresh ICAO check digits.

The entity key is built from the original canonical name and date of birth,
not from the run-specific entity id, so the same person gets the same
pseudonym in a later batch processed with the same key.
"""
from __future__ import annotations

import hashlib
import hmac
import re
from datetime import date, timedelta

from ..config import Config
from ..ff1 import FF1
from ..models import Entity, Field, Identity, PageRecord
from ..mrz import build_td3
from ..names import format_like, name_pools, normalize

SAFE_LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"
MONTHS = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()


class Keyed:
    def __init__(self, key: bytes):
        self.key = key
        self.ff1_10 = FF1(self._sub(b"ff1-radix10"), radix=10)

    def _sub(self, label: bytes) -> bytes:
        return hmac.new(self.key, label, hashlib.sha256).digest()

    def int(self, *parts: str) -> int:
        msg = "\x1f".join(parts).encode()
        return int.from_bytes(hmac.new(self.key, msg, hashlib.sha256).digest(), "big")

    def choice(self, options: list[str], *parts: str) -> str:
        opts = sorted(set(options))
        return opts[self.int(*parts) % len(opts)]


def entity_key(e: Entity) -> str:
    return f"{e.surname}|{' '.join(e.given)}|{e.dob or ''}"


def _pick_name(k: Keyed, pool: list[str], original: str, forbidden: set[str], *salt: str) -> str:
    for tol in (1, 2, 3, 99):
        cands = [n for n in pool if abs(len(n) - len(original)) <= tol and n not in forbidden and n != original]
        if cands:
            return k.choice(cands, *salt, original)
    raise ValueError("name pool exhausted")


def make_identities(entities: list[Entity], cfg: Config, corpus_tokens: set[str]) -> dict[str, Identity]:
    k = Keyed(cfg.hmac_key)
    pools = name_pools()
    used_full: set[tuple] = set()
    out: dict[str, Identity] = {}
    for e in sorted(entities, key=entity_key):
        loc = e.locale if e.locale in pools else "nordic"
        if e.sex in ("F", "M"):
            gpool = pools[loc][e.sex]
        else:
            gpool = pools[loc]["F"] + pools[loc]["M"]
        ek = entity_key(e)
        for attempt in range(20):
            sur = _pick_name(k, pools[loc]["S"], e.surname, corpus_tokens, ek, "surname", str(attempt))
            given, taken = [], set(corpus_tokens)
            for i, g in enumerate(e.given):
                ng = _pick_name(k, gpool, g, taken, ek, f"given{i}", str(attempt))
                given.append(ng)
                taken.add(ng)
            if (sur, tuple(given)) not in used_full:
                break
        used_full.add((sur, tuple(given)))
        span = cfg.dob_shift_max_days - cfg.dob_shift_min_days
        shift = cfg.dob_shift_min_days + k.int(ek, "dob") % span
        if k.int(ek, "dob-sign") % 2:
            shift = -shift
        out[e.entity_id] = Identity(e.entity_id, sur, given, e.sex, shift)
    return out


# ------------------------------------------------------------------- per-field surrogates

def _shift_viz_date(text: str, days: int) -> str:
    m = re.match(r"(\d{1,2})\s+([A-Z]{3})\s+(\d{4})", text.upper())
    if not m:
        return text
    d = date(int(m.group(3)), MONTHS.index(m.group(2)) + 1, int(m.group(1))) + timedelta(days=days)
    return f"{d.day:02d} {MONTHS[d.month - 1]} {d.year}"


def _shift_yymmdd(s: str, days: int, century_cut: int = 30) -> str:
    yy = int(s[:2])
    d = date(2000 + yy if yy < century_cut else 1900 + yy, int(s[2:4]), int(s[4:6])) + timedelta(days=days)
    return d.strftime("%y%m%d")


def _identifier(k: Keyed, value: str, label: str) -> str:
    """Digits via FF1 (when the run is long enough), letters via keyed substitution; format kept."""
    out = list(value)
    digits = [i for i, c in enumerate(value) if c.isdigit()]
    if len(digits) >= 6:
        enc = k.ff1_10.encrypt("".join(value[i] for i in digits), tweak=label.encode())
        for i, c in zip(digits, enc):
            out[i] = c
    else:
        for i in digits:
            out[i] = str(k.int(label, value, str(i)) % 10)
    for i, c in enumerate(value):
        if c.isalpha():
            # Issuers avoid O, I and Q in document numbers because readers confuse them with 0 and 1.
            out[i] = SAFE_LETTERS[k.int(label, value, "L", str(i)) % len(SAFE_LETTERS)]
            if c.islower():
                out[i] = out[i].lower()
    return "".join(out)


def _iban(k: Keyed, value: str) -> str:
    compact = value.replace(" ", "").upper()
    country, bban = compact[:2], compact[4:]
    new_bban = _identifier(k, bban, "iban")
    num = "".join(str(int(c, 36)) for c in new_bban + country + "00")
    new = f"{country}{98 - int(num) % 97:02d}{new_bban}"
    # Re-insert the original grouping.
    out, j = [], 0
    for c in value:
        if c == " ":
            out.append(" ")
        else:
            out.append(new[j]); j += 1
    return "".join(out)


def _address(k: Keyed, value: str, ek: str) -> str:
    pools = name_pools()
    m = re.match(r"\s*(\d+)\W*\s+(.*)", value)
    street = normalize(m.group(2)) if m else normalize(value)
    new_street = k.choice([s for s in pools["streets"] if s != street], ek, "street")
    digits = len(m.group(1)) if m else 2
    lo = 10 ** (digits - 1) if digits > 1 else 1
    number = lo + k.int(ek, "house") % (10 ** digits - lo)   # same number of digits as the original
    return f"{number} {new_street}"


def surrogate_for(f: Field, rec: PageRecord, ent: Entity, ident: Identity, cfg: Config) -> str | None:
    k = Keyed(cfg.hmac_key)
    ek = entity_key(ent)
    t = f.ftype
    if t in ("surname", "given_names", "full_name", "third_party_name"):
        return format_like(f.value, ent.given, ent.surname, ident.given, ident.surname)
    if t == "doc_number":
        return _identifier(k, f.value, "doc_number")
    if t == "dob":
        return _shift_viz_date(f.value, ident.dob_shift_days)
    if t == "place_of_birth":
        return k.choice([c for c in name_pools()["cities"] if c != normalize(f.value)], ek, "pob")
    if t == "address":
        return _address(k, f.value, ek)
    if t == "postcode":
        m = re.match(r"^([A-Z]+-?)(\d+)$", f.value.strip())
        if m:   # keep the region prefix, replace the digits
            return m.group(1) + _identifier(k, m.group(2), "postcode")
        return _identifier(k, f.value, "postcode")
    if t == "account_number":
        return _identifier(k, f.value, "account")
    if t == "iban":
        return _iban(k, f.value)
    if t in ("mrz_line1", "mrz_line2") and rec.mrz and rec.mrz.valid:
        m = rec.mrz
        num = _identifier(k, m.doc_number, "doc_number")
        l1, l2 = build_td3(m.doc_code or "P", m.issuer, ident.surname, " ".join(ident.given), num,
                           m.nationality, _shift_yymmdd(m.dob, ident.dob_shift_days), m.sex, m.expiry, m.personal)
        return l1 if t == "mrz_line1" else l2
    return None


def apply(records: list[PageRecord], entities: list[Entity], idents: dict[str, Identity], cfg: Config) -> None:
    by_id = {e.entity_id: e for e in entities}
    for rec in records:
        for f in rec.fields:
            if f.entity_id is None:
                rec.review_reasons.append(f"no_entity_for_{f.ftype}")
                continue
            f.surrogate = surrogate_for(f, rec, by_id[f.entity_id], idents[f.entity_id], cfg)
            if f.surrogate is None:
                rec.review_reasons.append(f"no_surrogate_for_{f.ftype}")
