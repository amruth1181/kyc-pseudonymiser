"""Stage 9: review queue, audit log, vault and release.

* Passed pages are written under ``pseudonymised/`` with names built from
  the entity id and a keyed hash, never from the input filename (filenames
  of KYC scans often contain the customer's name).
* Pages with any review reason go to ``review/queue.jsonl`` and a Label
  Studio task file; they are not released.
* ``audit.jsonl`` records what was changed (types, boxes, checks), never the
  original or surrogate values.
* ``vault.enc`` holds everything needed to re-identify (originals, mapping,
  input paths) and to undo the edit (the original pixels of every changed
  area), encrypted with the vault key. Keep it, and the keys, apart from the
  released corpus.

Pseudonymisation, unlike anonymisation, is reversible by the key holder:
``restore_page`` puts the stored pixels back and returns the original page.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path

import cv2
import numpy as np
from cryptography.fernet import Fernet
from PIL import Image

from ..config import Config
from ..models import Entity, Identity, PageRecord


def original_patches(original: np.ndarray, rec: PageRecord, pad: int = 2) -> list[dict]:
    """The original pixels under every edit, as PNG, for the vault.

    Verification guarantees that nothing changed outside the edit boxes (plus the same
    2 px margin), so these patches are all that is needed to undo the page exactly.
    """
    h, w = original.shape[:2]
    patches = []
    for e in rec.edits:
        if not e.get("ok") or "box" not in e:
            continue
        x0, y0, x1, y1 = e["box"]
        x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
        if x1 <= x0 or y1 <= y0:
            continue
        ok, png = cv2.imencode(".png", cv2.cvtColor(original[y0:y1, x0:x1], cv2.COLOR_RGB2BGR))
        if ok:
            patches.append({"box": [x0, y0, x1, y1], "png_b64": base64.b64encode(png.tobytes()).decode()})
    return patches


def restore_page(pseudonymised: np.ndarray, patches: list[dict]) -> np.ndarray:
    """Undo a pseudonymised page with the patches from its vault entry."""
    out = pseudonymised.copy()
    for p in patches:
        x0, y0, x1, y1 = p["box"]
        patch = cv2.imdecode(np.frombuffer(base64.b64decode(p["png_b64"]), np.uint8), cv2.IMREAD_COLOR)
        if patch is None or out[y0:y1, x0:x1].shape != patch.shape:
            raise ValueError("this vault does not belong to this image")
        out[y0:y1, x0:x1] = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    return out


def output_name(rec: PageRecord, cfg: Config) -> str:
    holder = next((f.entity_id for f in rec.fields if f.entity_id and f.mention_id and f.mention_id.endswith(":holder")),
                  "UNRESOLVED")
    tag = hmac.new(cfg.hmac_key, rec.sha256.encode(), hashlib.sha256).hexdigest()[:8]
    return f"{holder}_{rec.doc_type}_{tag}.png"


def release(out_dir: Path, records: list[PageRecord], outputs: dict[str, np.ndarray], entities: list[Entity],
            idents: dict[str, Identity], resolution_review: list[dict], leak_hits: list[dict], cfg: Config) -> dict:
    (out_dir / "pseudonymised").mkdir(parents=True, exist_ok=True)
    (out_dir / "review").mkdir(parents=True, exist_ok=True)
    mention_doc = {}
    for item in resolution_review:
        for mid in item.get("mentions", []) + ([item["mention"]] if "mention" in item else []):
            mention_doc.setdefault(mid.split(":")[0], []).append(item["type"])
    for h in leak_hits:
        rec = next(r for r in records if r.doc_id == h["doc"])
        rec.review_reasons.append("corpus_leak_hunt_hit")

    names, queue, tasks = {}, [], []
    audit = (out_dir / "audit.jsonl").open("w")
    for rec in records:
        rec.review_reasons += [f"resolution:{t}" for t in mention_doc.get(rec.doc_id, [])]
        rec.review_reasons = sorted(set(rec.review_reasons))
        rec.status = "review" if rec.review_reasons else "passed"
        name = output_name(rec, cfg)
        names[rec.doc_id] = name
        target = out_dir / ("pseudonymised" if rec.status == "passed" else "review") / name
        Image.fromarray(outputs[rec.doc_id]).save(target)
        view = rec.audit_view()
        view["doc_id"] = name                          # input ids/paths stay in the vault only
        audit.write(json.dumps(view) + "\n")
        if rec.status == "review":
            queue.append({"output": str(target), "reasons": rec.review_reasons})
            tasks.append({"data": {"output": str(target)}, "meta": {"reasons": rec.review_reasons}})
    audit.close()
    (out_dir / "review" / "queue.jsonl").write_text("".join(json.dumps(q) + "\n" for q in queue))
    (out_dir / "review" / "labelstudio_tasks.json").write_text(json.dumps(tasks, indent=2))

    vault = {
        "entities": [e.__dict__ for e in entities],
        "identities": {k: v.__dict__ for k, v in idents.items()},
        "documents": {
            rec.doc_id: {"input": rec.path, "output": names[rec.doc_id], "doc_type": rec.doc_type,
                         "status": rec.status, "size": [rec.width, rec.height], "restore": rec.restore,
                         "fields": [{"type": f.ftype, "value": f.value, "surrogate": f.surrogate,
                                     "entity": f.entity_id, "mention": f.mention_id, "source": f.source}
                                    for f in rec.fields]}
            for rec in records
        },
        "resolution_review": resolution_review,
    }
    (out_dir / "vault.enc").write_bytes(Fernet(cfg.vault_key).encrypt(json.dumps(vault).encode()))

    passed = sum(r.status == "passed" for r in records)
    summary = {
        "documents": len(records),
        "passed": passed,
        "sent_to_review": len(records) - passed,
        "entities": len(entities),
        "review_reasons": _count(r for rec in records for r in rec.review_reasons),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _count(items) -> dict:
    out: dict = {}
    for x in items:
        k = x.split(":")[0]
        out[k] = out.get(k, 0) + 1
    return out


def open_vault(path: Path, cfg: Config) -> dict:
    return json.loads(Fernet(cfg.vault_key).decrypt(path.read_bytes()))
