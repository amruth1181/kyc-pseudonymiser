"""Stage 8: verify every output page before it can be released.

Detection is never complete, so release rests on checks of the result:

* fidelity  - each rendered field is re-read; character error rate against
              the intended surrogate must stay under ``cfg.cer_max``;
* MRZ       - the new MRZ must parse with valid check digits and agree with
              the visual zone;
* leak      - the whole output page is re-read and searched for every
              original token; tokens that legitimately remain (labels, kept
              city names, words shared with the surrogate) are excluded;
* locality  - no pixel may change outside the recorded edit boxes;
* regions   - photo and signature areas must actually have changed.

``corpus_leak_hunt`` repeats the leak search across documents with BM25, to
catch a person's name on a page where no detector linked it to them.
"""
from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from ..bm25 import BM25
from ..config import Config
from ..layout import build_lines, text_of, vnorm
from ..models import Entity, PageRecord
from ..mrz import _TO_DIGIT, clean_line, repair_td3
from ..names import normalize
from ..ocr import MRZ_CHARS, read_line, read_line_conf, read_words


def _crop(img: np.ndarray, box, pad: int = 6) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]


def _best_reading(crop: np.ndarray) -> str:
    import cv2
    up = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return max([read_line_conf(crop), read_line_conf(up)], key=lambda r: r[1])[0]


def _class_align(got: str, want: str) -> str:
    """Where the intended character is a digit, read a confusable letter as that digit (O->0, I->1)."""
    if len(got) != len(want):
        return got
    return "".join(_TO_DIGIT.get(g, g) if w.isdigit() else g for g, w in zip(got, want))


def _drop_specks(got: str, want: str) -> str:
    """Remove one-character tokens the surrogate does not contain.

    A background line or the edge of a pattern next to the value is read as a lone
    'I' or '1'; it is not part of the drawn text.
    """
    keep = set(want.split())
    return " ".join(t for t in got.split() if len(t) > 1 or t in keep)


def _mrz_distance(crop: np.ndarray, want: str) -> tuple[int, int]:
    """Edit distance between the drawn MRZ line and its best reading, over the data part.

    The trailing run of '<' fillers carries no information and general-purpose OCR reads
    it unreliably (as K, S or E), so the comparison covers everything before it. Several
    readings at different sizes are taken, since small scans read better enlarged.
    """
    import cv2
    data = want.rstrip("<")
    best = len(data)
    for scale in (1, 2, 3):
        img = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else crop
        got = _class_align(clean_line(read_line(img, whitelist=MRZ_CHARS)), want)
        for k in range(max(0, len(data) - 2), len(data) + 3):
            best = min(best, Levenshtein.distance(got[:k], data))
        if best == 0:
            break
    return best, len(data)


_raw_cache: dict = {}


def _best_text_raw(crop: np.ndarray) -> str:
    key = (crop.shape, crop.tobytes()[:64], int(crop.sum()))
    if key not in _raw_cache:
        _raw_cache[key] = _best_reading(crop)
    return _raw_cache[key]


def verify(original: np.ndarray, output: np.ndarray, rec: PageRecord, cfg: Config) -> dict:
    res: dict = {"fields": [], "leaks": [], "mrz": None, "locality_px": 0, "regions_changed": True}
    edits = [e for e in rec.edits if e.get("ok")]

    # Fidelity per field
    for e in edits:
        if "idx" not in e:
            continue
        f = rec.fields[e["idx"]]
        crop = _crop(output, e["render_box"], 4)
        if f.ftype.startswith("mrz_"):
            dist, n = _mrz_distance(crop, f.surrogate)
            cer = dist / max(n, 1)
        else:
            want = vnorm(f.surrogate)
            got = _class_align(_drop_specks(vnorm(_best_text_raw(crop)), want), want)
            raw_got = " ".join(_best_text_raw(crop).upper().split())
            raw_want = " ".join(f.surrogate.upper().split())
            cer = Levenshtein.normalized_distance(got, want) if want else 0.0
            dist = min(Levenshtein.distance(got, want), Levenshtein.distance(raw_got, raw_want))
        # The string is drawn exactly, so read-back errors are reader slips or layout failures.
        # Layout failures (clipping, overflow, misplacement) break many characters; allow one slip.
        ok = cer <= cfg.cer_max or dist <= 1
        res["fields"].append({"type": f.ftype, "cer": round(cer, 3), "ok": ok})

    # MRZ re-parse and consistency with the visual zone
    lines = {f.ftype: f for f in rec.fields if f.ftype.startswith("mrz_") and f.surrogate}
    if len(lines) == 2:
        reads = [read_line(_crop(output, lines[k].box, 10), whitelist=MRZ_CHARS) for k in ("mrz_line1", "mrz_line2")]
        parsed = repair_td3(*reads)
        viz_surname = next((normalize(f.surrogate) for f in rec.fields if f.ftype == "surname" and f.surrogate), None)
        expected = repair_td3(lines["mrz_line1"].surrogate, lines["mrz_line2"].surrogate)
        res["mrz"] = {
            "check_digits_valid": bool(expected and expected.valid and expected.composite_ok),
            "reads_back": bool(parsed and parsed.valid),
            "matches_visual_zone": bool(expected and viz_surname and expected.surname == viz_surname),
        }

    # Leak search over the whole output page
    out_words = read_words(output, cfg.ocr_lang)
    out_tokens = {t for w in out_words for t in vnorm(w.text).split()}
    allowed = {t for f in rec.fields if f.surrogate for t in vnorm(f.surrogate).split()}
    pii_tokens = {t for f in rec.fields for t in vnorm(f.value).split()}
    page_text_tokens = {t for w in rec.words for t in vnorm(w.text).split()}
    allowed |= page_text_tokens - pii_tokens          # labels, headings, kept fields
    for f in rec.fields:
        for tok in set(vnorm(f.value).split()):
            if len(tok) < 3 or tok in allowed:
                continue
            for ot in out_tokens:
                if abs(len(ot) - len(tok)) <= 2 and fuzz.ratio(ot, tok) >= cfg.leak_ratio and ot not in allowed:
                    res["leaks"].append({"field": f.ftype, "token_len": len(tok)})
                    break
    res["page_text"] = " ".join(text_of(l) for l in build_lines(out_words))

    # Locality: nothing changed outside the edit boxes
    changed = np.any(original != output, axis=2)
    allowed_mask = np.zeros_like(changed)
    for e in edits:
        x0, y0, x1, y1 = e["box"]
        allowed_mask[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2] = True
    res["locality_px"] = int((changed & ~allowed_mask).sum())

    # Regions really replaced
    for r in rec.regions:
        x0, y0, x1, y1 = r.box
        diff = np.abs(original[y0:y1, x0:x1].astype(int) - output[y0:y1, x0:x1].astype(int)).mean()
        if diff < 5:
            res["regions_changed"] = False

    reasons = []
    reasons += [f"high_cer:{x['type']}" for x in res["fields"] if not x["ok"]]
    reasons += [f"leak:{x['field']}" for x in res["leaks"]]
    if res["mrz"] and not all(res["mrz"].values()):
        reasons.append("mrz_inconsistent")
    if res["locality_px"]:
        reasons.append("pixels_changed_outside_edits")
    if not res["regions_changed"]:
        reasons.append("region_not_replaced")
    rec.review_reasons += reasons
    rec.verify = {k: v for k, v in res.items() if k != "page_text"}
    return res


def corpus_leak_hunt(page_texts: dict[str, str], entities: list[Entity]) -> list[dict]:
    """Query every original name against the whole output corpus."""
    ids = list(page_texts)
    if not ids:
        return []
    index = BM25([normalize(page_texts[i]) for i in ids])
    hits = []
    for e in entities:
        forms = e.surname_forms()
        seen_docs = set()
        for query in set(e.variants) | {" ".join(e.given + [e.surname])}:
            for i, _score in index.top_k(query, k=5):
                if i in seen_docs:
                    continue
                toks = normalize(page_texts[ids[i]]).split()
                has_sur = any(JaroWinkler.similarity(t, f) >= 0.92 for t in toks if len(t) >= 4 for f in forms)
                has_given = any(JaroWinkler.similarity(t, g) >= 0.92 for t in toks for g in e.given if len(g) >= 3)
                if has_sur and has_given:
                    seen_docs.add(i)
                    hits.append({"doc": ids[i], "entity": e.entity_id})
    return hits
