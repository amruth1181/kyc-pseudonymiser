"""Orchestrates the nine stages.

    per document : 1 ingest -> 2 classify -> 3 locate -> 4 label
    corpus       : 5 resolve (BM25 candidates, linking, sweep, second pass)
    per document : 6 generate -> 7 render -> 8 verify
    corpus       : leak hunt -> 9 release

Per-document stages run in a process pool. Stage 1-4 results are cached by
file hash, encrypted with the vault key, so a re-run skips finished pages.
"""
from __future__ import annotations

import logging
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cryptography.fernet import Fernet

from .config import Config
from .models import PageRecord
from .ocr import read_words
from .stages import s5_resolve as s5
from .stages import s6_generate as s6
from .stages.s1_ingest import deskew, discover, enlarge_small, estimate_skew, load_pages, sha256
from .stages.s2_classify import classify
from .stages.s3_locate import locate
from .stages.s4_label import label
from .stages.s7_render import render
from .stages.s8_verify import corpus_leak_hunt, verify
from .stages.s9_release import release

log = logging.getLogger("kycpseudo")


@dataclass
class Page:
    rec: PageRecord
    image: np.ndarray


def _stages_1_to_4(args) -> Page:
    path, page_no, cfg, cache_dir = args
    digest = sha256(path)
    doc_id = f"{path.stem}" + (f"-p{page_no + 1}" if page_no else "")
    cache = cache_dir / f"{digest}-{page_no}.bin" if cache_dir else None
    if cache and cache.exists():
        return pickle.loads(Fernet(cfg.vault_key).decrypt(cache.read_bytes()))
    img, enlarged = enlarge_small(load_pages(path, cfg.dpi)[page_no])
    angle = estimate_skew(img)
    # Resampling softens the page, which costs a small scan more than a tilt that moves its
    # edges by a pixel or so; such a tilt is recorded but left alone.
    if abs(np.tan(np.radians(angle))) * img.shape[1] / 2 >= 1.5:
        img = deskew(img, angle)
    rec = PageRecord(doc_id=doc_id, path=str(path), sha256=digest, width=img.shape[1], height=img.shape[0],
                     skew_deg=angle)
    rec.words = read_words(img, cfg.ocr_lang, thorough=enlarged > 1)   # shared by stages 2-4
    classify(rec)
    locate(img, rec)
    label(img, rec)
    page = Page(rec, img)
    if cache:
        cache.write_bytes(Fernet(cfg.vault_key).encrypt(pickle.dumps(page)))
    return page


def _stages_7_to_8(args):
    page, cfg = args
    out = render(page.image, page.rec, cfg)
    res = verify(page.image, out, page.rec, cfg)
    return page.rec, out, res["page_text"]


def _map(fn, items, workers: int):
    if workers <= 1:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def run(input_dir: Path, out_dir: Path, cfg: Config, workers: int = 1, use_cache: bool = True) -> dict:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".cache" if use_cache else None
    if cache_dir:
        cache_dir.mkdir(exist_ok=True)
    jobs = []
    for path in discover(input_dir):
        n_pages = len(load_pages(path, 72)) if path.suffix.lower() == ".pdf" else 1
        jobs += [(path, i, cfg, cache_dir) for i in range(n_pages)]
    log.info("stage 1-4: %d pages", len(jobs))
    pages = _map(_stages_1_to_4, jobs, workers)
    records = [p.rec for p in pages]

    log.info("stage 5: entity resolution")
    mentions = s5.build_mentions(records)
    roots, _ = s5.resolve(mentions, cfg)
    entities = s5.build_entities(mentions, roots)
    swept = s5.sweep(records, entities)                         # names found in free text
    mentions = s5.build_mentions(records)                       # second pass includes them
    roots, review = s5.resolve(mentions, cfg)
    entities = s5.build_entities(mentions, roots)
    s5.assign(records, mentions, roots, entities)
    log.info("  %d mentions -> %d entities (%d names found by sweep, %d review items)",
             len(mentions), len(entities), swept, len(review))

    log.info("stage 6: surrogates")
    corpus_tokens = {t for e in entities for t in [e.surname] + e.given} | \
                    {t for m in mentions for t in m.normalized.split()}
    idents = s6.make_identities(entities, cfg, corpus_tokens)
    s6.apply(records, entities, idents, cfg)

    log.info("stage 7-8: render and verify")
    results = _map(_stages_7_to_8, [(p, cfg) for p in pages], workers)
    records = [r for r, _, _ in results]
    outputs = {r.doc_id: o for r, o, _ in results}
    page_texts = {r.doc_id: t for r, _, t in results}
    leak_hits = corpus_leak_hunt(page_texts, entities)

    log.info("stage 9: release")
    summary = release(out_dir, records, outputs, entities, idents, review, leak_hits, cfg)
    summary["seconds"] = round(time.time() - t0, 1)
    summary["corpus_leak_hunt_hits"] = len(leak_hits)
    return summary
