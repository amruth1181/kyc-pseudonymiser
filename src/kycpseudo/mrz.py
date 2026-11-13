"""ICAO Doc 9303 machine-readable zone, TD3 layout (passports: 2 lines x 44).

Check digit: map 0-9 to themselves, A-Z to 10-35 and the filler '<' to 0,
multiply by the repeating weights 7, 3, 1, sum, and keep the last digit.
"""
from __future__ import annotations

import re
from itertools import combinations
from typing import Optional

from .models import MRZData

WEIGHTS = (7, 3, 1)
ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")


def char_value(ch: str) -> int:
    if ch == "<":
        return 0
    if ch.isdigit():
        return int(ch)
    if "A" <= ch <= "Z":
        return ord(ch) - 55
    raise ValueError(f"not an MRZ character: {ch!r}")


def check_digit(s: str) -> str:
    return str(sum(char_value(c) * WEIGHTS[i % 3] for i, c in enumerate(s)) % 10)


def _split_names(field: str) -> tuple[str, str]:
    """Surname and given names from the name field of line 1.

    Names are separated by one filler and the two parts by two, so three fillers in a
    row can only be padding. Cutting there discards the padding however OCR misread it
    (a run of '<' often comes back with stray K, S or E).
    """
    names, _, padding = field.partition("<<<")
    names = names.rstrip("<")
    sur, giv = names.split("<<", 1) if "<<" in names else (names, "")
    if padding and "<" in giv:
        # The first filler after the last name is sometimes read as a letter: 'VANESSA<X<<<<'.
        head, _, last = giv.rpartition("<")
        if re.fullmatch(r"[KXSEC]{1,2}", last):
            giv = head
    return (" ".join(_alpha(sur).replace("<", " ").split()), " ".join(_alpha(giv).replace("<", " ").split()))


def build_line1(doc_code: str, issuer: str, surname: str, given_names: str) -> str:
    """Line 1 of a TD3 MRZ: document code, issuing state and the name, padded to 44."""
    name = surname.replace(" ", "<") + "<<" + given_names.replace(" ", "<")
    return (doc_code.ljust(2, "<") + issuer + name)[:44].ljust(44, "<")


def parse_line1(line1: str) -> Optional[MRZData]:
    """Read line 1 alone, for scans where line 2 is cropped off. Carries no check digits."""
    l1 = clean_line(line1)
    if len(l1) < 30 or not l1.startswith("P") or "<<" not in l1[5:].rstrip("<"):
        return None
    l1 = l1[:44].ljust(44, "<")
    sur, giv = _split_names(l1[5:])
    return MRZData(doc_code=l1[0:2].replace("<", ""), issuer=_alpha(l1[2:5]), surname=sur, given_names=giv,
                   doc_number="", nationality="", dob="", sex="", expiry="", personal="", valid=False,
                   composite_ok=False, raw=(l1, ""), line1_only=True)


def build_td3(doc_code: str, issuer: str, surname: str, given_names: str, doc_number: str,
              nationality: str, dob: str, sex: str, expiry: str, personal: str = "") -> tuple[str, str]:
    """Return the two 44-character lines with every check digit computed."""
    line1 = build_line1(doc_code, issuer, surname, given_names)
    num = doc_number.ljust(9, "<")[:9]
    pers = personal.ljust(14, "<")[:14]
    parts = [num, check_digit(num), nationality, dob, check_digit(dob), sex, expiry,
             check_digit(expiry), pers, check_digit(pers)]
    composite = num + check_digit(num) + dob + check_digit(dob) + expiry + check_digit(expiry) + pers + check_digit(pers)
    line2 = "".join(parts) + check_digit(composite)
    assert len(line1) == 44 and len(line2) == 44
    return line1, line2


# OCR confusions that are only legal in one character class.
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
CONFUSIONS = {(a, b) for a, b in [("O", "0"), ("Q", "0"), ("D", "0"), ("I", "1"), ("L", "1"), ("Z", "2"),
                                   ("S", "5"), ("B", "8"), ("G", "6")]} | \
             {(b, a) for a, b in [("O", "0"), ("Q", "0"), ("D", "0"), ("I", "1"), ("L", "1"), ("Z", "2"),
                                   ("S", "5"), ("B", "8"), ("G", "6")]}
_TO_ALPHA = {v: k for k, v in {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}.items()}


def _digits(s: str) -> str:
    return "".join(_TO_DIGIT.get(c, c) for c in s)


def _alpha(s: str) -> str:
    return "".join(_TO_ALPHA.get(c, c) for c in s)


def clean_line(text: str) -> str:
    t = text.upper().replace(" ", "").replace("«", "<<").replace("‹", "<")
    t = "".join(c if c in ALLOWED else "<" for c in t)
    return t


def _normalize_line2(l2: str) -> str:
    """Rebuild line 2 to 44 characters.

    Positions 0-27 (number, nationality, dates, sex) are kept as read. The
    tail is the 14-character personal-number field plus two check digits; OCR
    often drops or invents characters inside its long filler run, so when the
    tail holds no real data it is rebuilt from the fillers and the last digits.
    A tail that carries a personal number (as Luxembourg prints) is kept as read.
    """
    head, tail = l2[:28], l2[28:]
    body = tail[:-2] if len(tail) >= 16 else tail.rstrip("<0123456789QDO()")
    if len(body.replace("<", "")) <= 1:
        digits = [_digits(c) for c in tail[-2:] if _digits(c).isdigit()]
        cd_p = digits[-2] if len(digits) == 2 else "0"
        comp = digits[-1] if digits else "<"
        return head.ljust(28, "<") + "<" * 14 + cd_p + comp
    return (head + tail)[:44].ljust(44, "<")


def parse_td3(line1: str, line2: str) -> Optional[MRZData]:
    l1, l2 = clean_line(line1), clean_line(line2)
    if len(l1) < 30 or len(l2) < 28 or not l1.startswith("P"):
        return None
    l1 = l1[:44].ljust(44, "<")
    l2 = _normalize_line2(l2)
    issuer = _alpha(l1[2:5])
    sur, giv = _split_names(l1[5:])
    num = l2[0:9]
    dob, exp = _digits(l2[13:19]), _digits(l2[21:27])
    pers = l2[28:42]
    cd_num, cd_dob, cd_exp, cd_pers, cd_comp = (_digits(c) for c in (l2[9], l2[19], l2[27], l2[42], l2[43]))
    try:
        fields_ok = check_digit(num) == cd_num and check_digit(dob) == cd_dob and check_digit(exp) == cd_exp
        comp = num + cd_num + dob + cd_dob + exp + check_digit(exp) + pers + check_digit(pers)
        composite_ok = check_digit(comp) == cd_comp
    except ValueError:
        fields_ok, composite_ok = False, False
    # Field check digits protect number and dates. An unreadable composite digit alone
    # does not reject the reading; it is recorded so verification can flag it.
    valid = fields_ok
    return MRZData(doc_code=l1[0:2].replace("<", ""), issuer=issuer, surname=sur, given_names=giv,
                   doc_number=num.replace("<", ""), nationality=_alpha(l2[10:13]), dob=dob,
                   sex=l2[20], expiry=exp, personal=pers.replace("<", ""), valid=valid,
                   composite_ok=composite_ok, raw=(l1, l2))


def viz_date_forms(d: str, century_cut: int = 30) -> list[str]:
    """How a YYMMDD date may be printed in the visual zone, across issuing states."""
    months = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()
    yy, mm, dd = int(d[:2]), int(d[2:4]), int(d[4:6])
    year = 2000 + yy if yy < century_cut else 1900 + yy
    if not (1 <= mm <= 12):
        return []
    return [f"{dd:02d} {months[mm - 1]} {year}",        # 12 AUG 1974
            f"{dd:02d} {mm:02d} {year}",                 # 12 08 1974, 12.08.1974, 12/08/1974
            f"{dd:02d} {months[mm - 1]} {yy:02d}"]       # 12 AUG /AOUT 74 (matched fuzzily)


def yymmdd_to_viz(d: str, century_cut: int = 30) -> str:
    """'740812' -> '12 AUG 1974' (the visual-zone format used by the synthetic template)."""
    months = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()
    yy, mm, dd = int(d[:2]), int(d[2:4]), int(d[4:6])
    year = 2000 + yy if yy < century_cut else 1900 + yy
    return f"{dd:02d} {months[mm - 1]} {year}"


def repair_td3(line1: str, line2: str) -> Optional[MRZData]:
    """Checksum-guided repair of OCR errors in the data part of line 2.

    A single mod-10 check digit is satisfied by about one random edit in ten,
    so validity alone is not enough to pick a repair. Candidates are tried in
    order of plausibility and accepted only if exactly one validates at that
    level:

    1. one known OCR confusion undone (Q->0, O->0, I->1, ...)
    2. two known confusions undone
    3. any single substitution (only if it is the only valid one)
    """
    first = parse_td3(line1, line2)
    if first is None or first.valid:
        return first
    head, tail = clean_line(line2)[:28], clean_line(line2)[28:]

    def valid_unique(cands) -> Optional[MRZData]:
        found = {}
        for c in cands:
            m = parse_td3(line1, c + tail)
            if m and m.valid:
                found[c] = m
        return next(iter(found.values())) if len(found) == 1 else None

    conf = [(i, b) for i, a in enumerate(head) for (x, b) in CONFUSIONS if x == a]
    one = [head[:i] + b + head[i + 1:] for i, b in conf]
    two = []
    for (i, a), (j, b) in combinations(conf, 2):
        if i != j:
            c = list(head); c[i], c[j] = a, b
            two.append("".join(c))
    alphabet = "<0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    anyone = [head[:i] + c + head[i + 1:] for i in range(len(head)) for c in alphabet if c != head[i]]
    for level in (one, two, anyone):
        m = valid_unique(level)
        if m:
            return m
    return first   # no unambiguous repair: leave invalid, send to review
