"""BM25 over character n-grams.

Why character n-grams: OCR errors and transliteration change single letters.
'GANTA' and 'CANTA' share the trigrams ANT and NTA, so the true match still
scores, while whole-word BM25 would score zero.

Why BM25 and not embeddings: names carry no meaning, only spelling. BM25's
IDF term makes a rare surname dominate the score, which is what identity
matching needs, and every score can be explained by the n-grams that matched.

score(q, d) = sum_t IDF(t) * f(t,d) * (k1 + 1) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict


def char_ngrams(text: str, n: int = 3) -> list[str]:
    grams: list[str] = []
    for token in text.split():
        t = f"#{token}#"                       # boundary markers so short tokens still produce grams
        if len(t) <= n:
            grams.append(t)
        else:
            grams.extend(t[i:i + n] for i in range(len(t) - n + 1))
    return grams


class BM25:
    def __init__(self, docs: list[str], n: int = 3, k1: float = 1.5, b: float = 0.75):
        self.n, self.k1, self.b = n, k1, b
        self.tfs = [Counter(char_ngrams(d, n)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter = Counter()
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, tf in enumerate(self.tfs):
            for g in tf:
                df[g] += 1
                self.postings[g].append(i)
        n_docs = len(docs)
        # BM25+ style IDF floor keeps very common grams from going negative.
        self.idf = {g: math.log(1 + (n_docs - c + 0.5) / (c + 0.5)) for g, c in df.items()}

    def score(self, query: str) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for g in set(char_ngrams(query, self.n)):
            idf = self.idf.get(g)
            if idf is None:
                continue
            for i in self.postings[g]:
                f = self.tfs[i][g]
                norm = 1 - self.b + self.b * self.lens[i] / self.avgdl
                scores[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return scores

    def top_k(self, query: str, k: int = 10, exclude: int | None = None) -> list[tuple[int, float]]:
        ranked = sorted(((i, s) for i, s in self.score(query).items() if i != exclude),
                        key=lambda x: -x[1])
        return ranked[:k]

    def explain(self, query: str, i: int) -> list[tuple[str, float]]:
        """Which n-grams produced the match: shown to reviewers and auditors."""
        out = []
        for g in set(char_ngrams(query, self.n)) & set(self.tfs[i]):
            f = self.tfs[i][g]
            norm = 1 - self.b + self.b * self.lens[i] / self.avgdl
            out.append((g, self.idf[g] * f * (self.k1 + 1) / (f + self.k1 * norm)))
        return sorted(out, key=lambda x: -x[1])
