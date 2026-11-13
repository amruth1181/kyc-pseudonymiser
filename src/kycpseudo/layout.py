"""Geometry helpers on OCR words: lines, segments and fuzzy value search.

Lines are rebuilt from word positions (not from the OCR engine's own line
ids) because engines split a label and its value into separate blocks when
the gap between them is wide, as on forms.
"""
from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

from .models import Box, Word
from .names import normalize


def vnorm(text: str) -> str:
    """Value normalisation: keeps digits (unlike name normalisation), drops punctuation."""
    t = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^A-Z0-9]", " ", t.upper().replace("|", "I")).split())


def union_box(boxes: list[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def build_lines(words: list[Word]) -> list[list[Word]]:
    ws = sorted(words, key=lambda w: (w.box[1] + w.box[3]) / 2)
    lines: list[list[Word]] = []
    for w in ws:
        yc, h = (w.box[1] + w.box[3]) / 2, w.box[3] - w.box[1]
        if lines:
            last = lines[-1]
            lyc = sum((x.box[1] + x.box[3]) / 2 for x in last) / len(last)
            lh = sum(x.box[3] - x.box[1] for x in last) / len(last)
            if abs(yc - lyc) < 0.5 * max(h, lh):
                last.append(w)
                continue
        lines.append([w])
    return [sorted(l, key=lambda w: w.box[0]) for l in lines]


def segments(line: list[Word], gap_factor: float = 1.6) -> list[list[Word]]:
    """Split a line where the horizontal gap is much wider than a space."""
    if not line:
        return []
    h = sum(w.box[3] - w.box[1] for w in line) / len(line)
    segs, cur = [], [line[0]]
    for prev, w in zip(line, line[1:]):
        if w.box[0] - prev.box[2] > gap_factor * h:
            segs.append(cur)
            cur = [w]
        else:
            cur.append(w)
    segs.append(cur)
    return segs


def text_of(ws: list[Word]) -> str:
    return " ".join(w.text for w in ws)


def find_label(lines: list[list[Word]], label: str, min_ratio: float = 85) -> tuple[int, list[Word]] | None:
    """Locate a label phrase; returns (line index, the label's words)."""
    target = normalize(label)
    n = len(target.split())
    best = None
    for li, line in enumerate(lines):
        for i in range(len(line)):
            for k in (n, n + 1):
                win = line[i:i + k]
                if not win or not normalize(win[0].text):
                    continue                        # a label starts at a word, not at '(5)' or a bullet
                got = normalize(text_of(win))
                # Bilingual forms print 'Date of birth/Date de naissance': the label is the leading words.
                lead = " ".join(got.split()[:len(target.split())])
                r = max(fuzz.ratio(got, target), fuzz.ratio(lead, target))
                if r >= min_ratio and (best is None or r > best[0]):
                    best = (r, li, win)
    return (best[1], best[2]) if best else None


def find_value(lines: list[list[Word]], value: str, min_ratio: float = 85,
               exclude: set[int] | None = None) -> tuple[list[Word], float] | None:
    """Find the word window that best matches ``value`` anywhere on the page."""
    target = vnorm(value)
    n = max(1, len(value.split()))
    best = None
    for line in lines:
        for i in range(len(line)):
            for k in range(max(1, n - 1), n + 2):
                win = line[i:i + k]
                if len(win) < k or (exclude and any(id(w) in exclude for w in win)):
                    continue
                r = fuzz.ratio(vnorm(text_of(win)), target)
                if r >= min_ratio and (best is None or r > best[1]):
                    best = (win, r)
    return best


def value_below(lines: list[list[Word]], label_line: int, label_words: list[Word], max_dy: int = 80) -> list[Word] | None:
    """Form layout: the value sits on the next line, left-aligned with its label."""
    lbottom = max(w.box[3] for w in label_words)
    tol = max(30, 2 * (lbottom - min(w.box[1] for w in label_words)))
    # A trilingual label ('1. Numm / Nom / Surname') is matched on one of its languages, which
    # may sit mid-label; the value lines up with the start of the whole label.
    starts = [label_words[0].box[0]]
    for seg in segments(lines[label_line]):
        if any(w is label_words[0] for w in seg):
            starts.append(seg[0].box[0])
    for lx0 in starts:
        for line in lines[label_line + 1:label_line + 3]:
            top = min(w.box[1] for w in line)
            if top - lbottom > max_dy:
                break
            for seg in segments(line):
                if abs(seg[0].box[0] - lx0) < tol:
                    return seg
    return None


def value_right(lines: list[list[Word]], label_line: int, label_words: list[Word]) -> list[Word] | None:
    """Statement layout: the value follows its label on the same line."""
    lx1 = max(w.box[2] for w in label_words)
    rest = [w for w in lines[label_line] if w.box[0] > lx1]
    if not rest:
        return None
    return segments(rest)[0]
