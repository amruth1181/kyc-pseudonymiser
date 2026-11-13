"""Stage 5: resolve name mentions into people across the whole corpus.

    mentions --BM25 (char n-grams)--> candidate pairs --similarity + evidence--> links --union-find--> entities

Decision rules, in order of trust:

1. Hard evidence: same date of birth, document number or address, and the
   names are compatible (similarity >= review threshold) -> link.
2. Full-name match: similarity >= accept threshold without relying on
   initials, and no conflicting evidence -> link.
3. Initial-only match ('O. MALMBERG'): linked only if every such candidate
   falls in a single existing entity. If the initials fit two different
   people (Oskar and Olof Malmberg), the mention is not linked by name and is
   sent to review. A wrong merge would give two real people one pseudonym.
4. Similarity in the review band -> review queue, not linked.

Unions that would join entities with different dates of birth or addresses
are refused.
"""
from __future__ import annotations

from collections import defaultdict

from ..bm25 import BM25
from ..config import Config
from ..layout import build_lines, segments, union_box
from ..models import Entity, Field, Mention, PageRecord
from ..names import guess_locale, guess_sex, name_similarity, normalize, tokens
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

NAME_FIELDS = {"full_name", "surname", "given_names", "mrz_line1", "third_party_name"}


def build_mentions(records: list[PageRecord]) -> list[Mention]:
    mentions: list[Mention] = []
    for rec in records:
        by_type = {f.ftype: f for f in rec.fields}
        mid = f"{rec.doc_id}:holder"
        if rec.doc_type == "passport":
            m = rec.mrz if (rec.mrz and rec.mrz.valid) else None
            # Check digits do not cover the name, so the visual zone (larger print) wins over the MRZ.
            sur = by_type["surname"].value if "surname" in by_type else (m.surname if m else "")
            giv = by_type["given_names"].value if "given_names" in by_type else (m.given_names if m else "")
            if not (sur or giv):
                continue
            mentions.append(Mention(mid, rec.doc_id, f"{giv} {sur}".strip(), normalize(f"{giv} {sur}"),
                                    rec.doc_type, dob=m.dob if m else None,
                                    doc_number=m.doc_number if m else None, sex=m.sex if m else rec.sex,
                                    given=normalize(giv), surname=normalize(sur)))
            for f in rec.fields:
                if f.ftype in {"surname", "given_names", "mrz_line1", "mrz_line2"}:
                    f.mention_id = mid
        elif "full_name" in by_type:
            f = by_type["full_name"]
            addr = " ".join(normalize(by_type[t].value) for t in ("address", "postcode") if t in by_type) or None
            mentions.append(Mention(mid, rec.doc_id, f.value, normalize(f.value), rec.doc_type, address=addr))
            f.mention_id = mid
        for i, f in enumerate(x for x in rec.fields if x.ftype == "third_party_name"):
            tid = f"{rec.doc_id}:tp{i}"
            f.mention_id = tid
            mentions.append(Mention(tid, rec.doc_id, f.value, normalize(f.value), rec.doc_type))
    return mentions


class _UF:
    def __init__(self, mentions: list[Mention]):
        self.parent = {m.mention_id: m.mention_id for m in mentions}
        self.dobs = {m.mention_id: {m.dob} - {None} for m in mentions}
        self.addrs = {m.mention_id: {m.address} - {None} for m in mentions}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def conflict(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        dob_clash = self.dobs[ra] and self.dobs[rb] and not (self.dobs[ra] & self.dobs[rb])
        addr_clash = (self.addrs[ra] and self.addrs[rb]
                      and not any(_same_address(x, y) for x in self.addrs[ra] for y in self.addrs[rb]))
        return bool(dob_clash or addr_clash)

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.conflict(ra, rb):
            return False
        self.parent[rb] = ra
        self.dobs[ra] |= self.dobs[rb]
        self.addrs[ra] |= self.addrs[rb]
        return True


def _same_address(a: str, b: str) -> bool:
    """Addresses are OCR'd text, so equality is fuzzy (one or two slipped characters)."""
    return fuzz.ratio(a, b) >= 85


def _evidence(a: Mention, b: Mention) -> bool:
    return bool((a.dob and a.dob == b.dob) or (a.doc_number and a.doc_number == b.doc_number)
                or (a.address and b.address and _same_address(a.address, b.address)))


def resolve(mentions: list[Mention], cfg: Config) -> tuple[dict[str, str], list[dict]]:
    """Return mention_id -> cluster root, and the review items."""
    index = BM25([m.normalized for m in mentions], n=cfg.bm25_ngram)
    uf = _UF(mentions)
    review: list[dict] = []
    weak: dict[str, set[str]] = defaultdict(set)
    uncertain: list = []
    seen: set[tuple[str, str]] = set()

    pairs = []
    for i, m in enumerate(mentions):
        for j, _score in index.top_k(m.normalized, k=cfg.bm25_top_k, exclude=i):
            key = tuple(sorted((m.mention_id, mentions[j].mention_id)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((m, mentions[j]))

    # Pass 1: evidence-backed and full-name links.
    for a, b in pairs:
        sim, by_initial = name_similarity(a.text, b.text)
        if _evidence(a, b) and sim >= cfg.match_review:
            uf.union(a.mention_id, b.mention_id)
        elif sim >= cfg.match_accept and not by_initial:
            if not uf.union(a.mention_id, b.mention_id):
                review.append({"type": "conflicting_evidence", "mentions": [a.mention_id, b.mention_id]})
        elif sim >= cfg.match_accept and by_initial:
            # Only the mention written with initials is ambiguous; record it on that side.
            for x, y in ((a, b), (b, a)):
                if any(len(t) == 1 for t in tokens(x.text)):
                    weak[x.mention_id].add(y.mention_id)
        elif sim >= cfg.match_review:
            uncertain.append((a, b, sim))

    # Pass 2: initial-only links, only when unambiguous.
    for mid, cands in weak.items():
        targets = {uf.find(c) for c in cands if uf.find(c) != uf.find(mid)}
        if len(targets) == 1:
            uf.union(mid, next(iter(cands)))
        elif len(targets) > 1:
            review.append({"type": "ambiguous_initials", "mention": mid, "candidates": len(targets)})

    # Pass 3: uncertain pairs need a human only if nothing already decided them
    # (same entity now, or different dates of birth / addresses at entity level).
    for a, b, sim in uncertain:
        if uf.find(a.mention_id) != uf.find(b.mention_id) and not uf.conflict(a.mention_id, b.mention_id):
            review.append({"type": "uncertain_pair", "similarity": round(sim, 3),
                           "mentions": [a.mention_id, b.mention_id]})

    return {m.mention_id: uf.find(m.mention_id) for m in mentions}, review


def build_entities(mentions: list[Mention], roots: dict[str, str]) -> list[Entity]:
    groups: dict[str, list[Mention]] = defaultdict(list)
    for m in mentions:
        groups[roots[m.mention_id]].append(m)
    entities = []
    for ms in groups.values():
        anchor = next((m for m in ms if m.surname is not None), None)
        if anchor:                                   # passport: MRZ gives the split
            surname, given = anchor.surname, anchor.given.split()
        else:
            surname, given = _split_name(max(ms, key=lambda m: len(m.normalized)))
        dob = next((m.dob for m in ms if m.dob), None)
        sex = next((m.sex for m in ms if m.sex in ("M", "F")), None)
        locale = guess_locale(surname, given)
        entities.append(Entity(entity_id="", mention_ids=sorted(m.mention_id for m in ms), surname=surname,
                               given=given, sex=sex or guess_sex(given, locale), dob=dob, locale=locale,
                               variants=sorted({m.normalized for m in ms})))
    # Stable ids: ordered by first appearance, so re-runs give the same labels.
    entities.sort(key=lambda e: e.mention_ids[0])
    for i, e in enumerate(entities):
        e.entity_id = f"P-{i:04d}"
    return entities


def _split_name(m: Mention) -> tuple[str, list[str]]:
    raw = m.text
    if "," in raw:                                   # 'DUBOIS, Etienne Thomas'
        sur, giv = raw.split(",", 1)
        return normalize(sur), tokens(giv)
    toks = tokens(raw)
    return toks[-1], toks[:-1]                        # 'Etienne Thomas Dubois'


def sweep(records: list[PageRecord], entities: list[Entity]) -> int:
    """Find names of known people where no detector looked (e.g. transfer descriptions).

    A pair of adjacent words counts when one matches a known surname and its
    neighbour matches one of that person's given names or initials.
    """
    added = 0
    for rec in records:
        covered = [f.box for f in rec.fields]
        for line in build_lines(rec.words):
            for seg in segments(line):
                for i in range(len(seg) - 1):
                    w1, w2 = seg[i], seg[i + 1]
                    box = union_box([w1.box, w2.box])
                    if any(_inside(box, c) for c in covered):
                        continue
                    hit = _match_pair(normalize(w1.text), normalize(w2.text), entities)
                    if hit:
                        # No entity id yet: the name becomes a mention and goes through resolution again.
                        rec.fields.append(Field("third_party_name", f"{w1.text} {w2.text}", box, "corpus_sweep"))
                        covered.append(box)
                        added += 1
    return added


def _inside(a, b) -> bool:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy > 0.3 * (a[2] - a[0]) * (a[3] - a[1])


def _match_pair(t1: str, t2: str, entities: list[Entity]) -> Entity | None:
    best, best_s = None, 0.0
    for e in entities:
        forms = e.surname_forms()                     # every spelling seen for this person
        for given_tok, sur_tok in ((t1, t2), (t2, t1)):
            if len(sur_tok) < 4 or max(JaroWinkler.similarity(sur_tok, f) for f in forms) < 0.92:
                continue
            g = max((JaroWinkler.similarity(given_tok, x) for x in e.given), default=0)
            if g >= 0.92 and g > best_s:
                best, best_s = e, g
    return best


def assign(records: list[PageRecord], mentions: list[Mention], roots: dict[str, str],
           entities: list[Entity]) -> None:
    """Write entity ids onto fields and regions. Non-name fields belong to the document holder."""
    root_to_eid = {}
    for e in entities:
        for mid in e.mention_ids:
            root_to_eid[roots[mid]] = e.entity_id
    mid_to_eid = {mid: root_to_eid[roots[mid]] for mid in roots}
    for rec in records:
        holder = mid_to_eid.get(f"{rec.doc_id}:holder")
        for f in rec.fields:
            if f.entity_id:
                continue
            f.entity_id = mid_to_eid.get(f.mention_id) if f.mention_id else holder
        for r in rec.regions:
            r.entity_id = holder
        if holder is None and rec.fields:
            rec.review_reasons.append("no_document_holder")
