"""Name normalisation, comparison and re-formatting.

The same person appears as 'ERIKSSON<<ANNA<MARIA', 'Anna Maria Eriksson',
'A. M. ERIKSSON' or, after OCR, 'Anna Maria Erikssen'. Comparison works on
token sets so that order and initials do not matter, and formatting a
surrogate copies the original mention's pattern (order, initials, case).
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from importlib import resources

from rapidfuzz.distance import JaroWinkler


@lru_cache(maxsize=1)
def name_pools() -> dict:
    return json.loads(resources.files("kycpseudo").joinpath("data/names.json").read_text())


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.upper().replace("<", " ").replace("|", "I")   # '|' is a misread I, J or l
    t = re.sub(r"[^A-Z\s]", " ", t)
    return " ".join(t.split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def _tok_sim(a: str, b: str) -> tuple[float, bool]:
    """Similarity of two tokens and whether the match relied on an initial."""
    if len(a) == 1 or len(b) == 1:
        return (1.0 if a[0] == b[0] else 0.0), True
    return JaroWinkler.similarity(a, b), False


def name_similarity(a: str, b: str) -> tuple[float, bool]:
    """Score in [0, 1] and a flag saying the match depends on initials.

    Each token of the shorter name is aligned to its best unused token in the
    longer one. At least one full (non-initial) token must match strongly,
    otherwise two people who merely share initials would score 1.0.
    """
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0, False
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    used: set[int] = set()
    scores, any_initial, strong_full = [], False, False
    for t in short:
        best, best_j, best_init = 0.0, None, False
        for j, u in enumerate(long_):
            if j in used:
                continue
            s, init = _tok_sim(t, u)
            if s > best:
                best, best_j, best_init = s, j, init
        if best_j is not None:
            used.add(best_j)
        scores.append(best)
        any_initial |= best_init and best > 0
        strong_full |= (not best_init) and best >= 0.9
    if not strong_full:
        return 0.0, any_initial
    score = sum(scores) / len(scores)
    # Unmatched extra tokens in the longer name cost a little (middle names are often dropped).
    score -= 0.02 * max(0, len(long_) - len(short))
    return max(0.0, score), any_initial


def guess_locale(surname: str, given: list[str]) -> str:
    pools = name_pools()
    best, best_hits = "generic", 0
    for loc in ("nordic", "indian", "french", "portuguese"):
        p = pools[loc]
        hits = (surname in p["S"]) * 2 + sum(g in p["F"] or g in p["M"] for g in given)
        if hits > best_hits:
            best, best_hits = loc, hits
    return best


def guess_sex(given: list[str], locale: str) -> str | None:
    pools = name_pools()
    locs = [locale] if locale in pools else ["nordic", "indian", "french", "portuguese"]
    for g in given:
        for loc in locs:
            if g in pools[loc]["F"]:
                return "F"
            if g in pools[loc]["M"]:
                return "M"
    return None


def _case_like(src: str, word: str) -> str:
    if src.isupper() or src == "|":
        return word.upper()
    if src[:1].isupper():
        return word.capitalize()
    return word.lower()


def format_like(original: str, canon_given: list[str], canon_surname: str,
                new_given: list[str], new_surname: str) -> str:
    """Rewrite ``original`` with the surrogate names, keeping its pattern.

    'A. M. ERIKSSON' with surrogate EVA KARIN LINDQVIST -> 'E. K. LINDQVIST'.
    Separators (spaces, '<', '.', ',') are kept in place.
    """
    canon = [(g, ("given", i)) for i, g in enumerate(canon_given)] + [(canon_surname, ("surname", 0))]
    used: set[int] = set()

    def replace(m: re.Match) -> str:
        word = m.group(0)
        norm = normalize(word)
        if not norm:
            return word
        best, best_k = 0.0, None
        for k, (c, _) in enumerate(canon):
            if k in used:
                continue
            s, _init = _tok_sim(norm, c)
            if s > best:
                best, best_k = s, k
        if best_k is None or best < 0.75:
            return word
        used.add(best_k)
        kind, idx = canon[best_k][1]
        target = new_surname if kind == "surname" else (new_given[idx] if idx < len(new_given) else new_given[-1])
        if len(norm) == 1:
            return _case_like(word, target[0])
        return _case_like(word, target)

    return re.sub(r"[A-Za-zÀ-ÿ|]+", replace, original)
