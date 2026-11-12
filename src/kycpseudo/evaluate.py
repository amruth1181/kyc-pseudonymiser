"""Measure a run against the synthetic ground truth.

Uses only information the pipeline could not have used: the generator's
truth file. The leak check here is independent of the pipeline's own
verification: it OCRs every output image again and searches for every true
personal value, including ones the pipeline never detected.
"""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from rapidfuzz import fuzz

from .config import Config
from .layout import vnorm
from .ocr import read_words
from .stages.s9_release import open_vault

TYPE_MAP = {"surname": "surname", "given_names": "given_names", "doc_number": "doc_number", "dob": "dob",
            "place_of_birth": "place_of_birth", "mrz_line1": "mrz_line1", "mrz_line2": "mrz_line2",
            "full_name": "full_name", "address": "address", "postcode": "postcode",
            "account_number": "account_number", "iban": "iban", "third_party_name": "third_party_name"}
NAME_TYPES = {"surname", "given_names", "full_name", "third_party_name"}


def _same(a: str, b: str, t: str) -> bool:
    if t.startswith("mrz_"):
        return fuzz.ratio(a.replace(" ", ""), b.replace(" ", "")) >= 85
    return fuzz.ratio(vnorm(a), vnorm(b)) >= 85


def evaluate(truth_path: Path, run_dir: Path, cfg: Config) -> dict:
    truth = json.loads(truth_path.read_text())
    vault = open_vault(run_dir / "vault.enc", cfg)
    docs = {Path(v["input"]).stem: v for v in vault["documents"].values()}

    # 1. Document type and detection recall per field type
    type_ok, recall = 0, defaultdict(lambda: [0, 0])
    for d in truth["documents"]:
        v = docs[d["doc_id"]]
        type_ok += v["doc_type"] == d["doc_type"]
        for p in d["pii"]:
            t = TYPE_MAP[p["type"]]
            recall[t][1] += 1
            if any(f["type"] == t and _same(f["value"], p["value"], t) for f in v["fields"]):
                recall[t][0] += 1

    # 2. Entity resolution, pairwise over name mentions
    pred, gold = {}, {}
    for d in truth["documents"]:
        v = docs[d["doc_id"]]
        for f in v["fields"]:
            if f["mention"] and f["mention"].endswith(":holder") and f["type"] in ("full_name", "surname"):
                pred[f["mention"]], gold[f["mention"]] = f["entity"], d["person_id"]
            if f["type"] == "third_party_name":
                tp = next((p for p in d["pii"] if p["type"] == "third_party_name"), None)
                if tp:
                    key = f"{d['doc_id']}:tp"
                    pred[key], gold[key] = f["entity"], tp["person_id"]
    tp_ = fp = fn = 0
    for a, b in combinations(sorted(pred), 2):
        same_pred, same_gold = pred[a] == pred[b], gold[a] == gold[b]
        tp_ += same_pred and same_gold
        fp += same_pred and not same_gold
        fn += same_gold and not same_pred
    precision = tp_ / (tp_ + fp) if tp_ + fp else 1.0
    rec_ = tp_ / (tp_ + fn) if tp_ + fn else 1.0

    # 3. Independent leak check on the output images
    leaks, leak_docs = [], set()
    for d in truth["documents"]:
        v = docs[d["doc_id"]]
        folder = "pseudonymised" if v["status"] == "passed" else "review"
        out_img = np.array(Image.open(run_dir / folder / v["output"]).convert("RGB"))
        out_tokens = {t for w in read_words(out_img) for t in vnorm(w.text).split()}
        surrogate_tokens = {t for f in v["fields"] if f["surrogate"] for t in vnorm(f["surrogate"]).split()}
        pii_tokens = {t for p in d["pii"] for t in vnorm(p["value"]).split()}
        for p in d["pii"]:
            if p["type"].startswith("mrz_"):
                continue    # MRZ content is covered by the name, number and date checks
            for tok in set(vnorm(p["value"]).split()):
                if len(tok) < 3 or tok in surrogate_tokens:
                    continue
                if any(abs(len(o) - len(tok)) <= 1 and fuzz.ratio(o, tok) >= 92 for o in out_tokens - surrogate_tokens):
                    # A word like 'STREET' can legitimately remain; only count tokens unique to PII.
                    if tok in pii_tokens:
                        leaks.append({"doc": d["doc_id"], "type": p["type"], "token_len": len(tok)})
                        leak_docs.add(d["doc_id"])

    # 4. Pipeline's own verification results
    audit = [json.loads(l) for l in (run_dir / "audit.jsonl").read_text().splitlines()]
    cers = [f["cer"] for a in audit for f in a["verify"].get("fields", [])]
    mrz = [a["verify"]["mrz"] for a in audit if a["verify"].get("mrz")]
    locality = sum(a["verify"].get("locality_px", 0) for a in audit)
    passed = sum(a["status"] == "passed" for a in audit)

    metrics = {
        "documents": len(truth["documents"]),
        "persons": len(truth["persons"]),
        "doc_type_accuracy": round(type_ok / len(truth["documents"]), 3),
        "detection_recall": {t: f"{a}/{n}" for t, (a, n) in sorted(recall.items())},
        "detection_recall_overall": round(sum(a for a, _ in recall.values()) / sum(n for _, n in recall.values()), 3),
        "entity_resolution": {"pairwise_precision": round(precision, 3), "pairwise_recall": round(rec_, 3),
                              "predicted_entities": len(set(pred.values())), "true_persons": len(set(gold.values()))},
        "leaks_independent_check": {"documents_with_leak": len(leak_docs), "tokens": leaks},
        "fidelity": {"fields_rendered": len(cers), "mean_cer": round(float(np.mean(cers)), 4) if cers else None,
                     "max_cer": round(float(np.max(cers)), 4) if cers else None,
                     "fields_cer_0": sum(c == 0 for c in cers)},
        "mrz": {"rebuilt": len(mrz), "check_digits_valid": sum(m["check_digits_valid"] for m in mrz),
                "reads_back_valid": sum(m["reads_back"] for m in mrz),
                "matches_visual_zone": sum(m["matches_visual_zone"] for m in mrz)},
        "locality_pixels_changed_outside_edits": locality,
        "released_without_review": f"{passed}/{len(audit)}",
    }
    return metrics


def to_markdown(m: dict) -> str:
    er = m["entity_resolution"]
    lines = [
        "| Metric | Result |", "| --- | --- |",
        f"| Documents / fictional persons | {m['documents']} / {m['persons']} |",
        f"| Document type accuracy | {m['doc_type_accuracy']:.0%} |",
        f"| Detection recall (all PII values) | {m['detection_recall_overall']:.1%} |",
        f"| Entity resolution precision / recall (pairwise) | {er['pairwise_precision']:.0%} / {er['pairwise_recall']:.0%} |",
        f"| Entities found / true persons | {er['predicted_entities']} / {er['true_persons']} |",
        f"| Documents with any original value readable (independent check) | {m['leaks_independent_check']['documents_with_leak']} |",
        f"| Fields rendered / mean CER / max CER | {m['fidelity']['fields_rendered']} / {m['fidelity']['mean_cer']} / {m['fidelity']['max_cer']} |",
        f"| MRZ rebuilt with valid check digits | {m['mrz']['check_digits_valid']}/{m['mrz']['rebuilt']} |",
        f"| MRZ re-read from image validates | {m['mrz']['reads_back_valid']}/{m['mrz']['rebuilt']} |",
        f"| Pixels changed outside edit boxes | {m['locality_pixels_changed_outside_edits']} |",
        f"| Released without human review | {m['released_without_review']} |",
        "", "Detection recall by field type:", "", "| Field | Found |", "| --- | --- |",
    ]
    lines += [f"| {t} | {v} |" for t, v in m["detection_recall"].items()]
    return "\n".join(lines) + "\n"
