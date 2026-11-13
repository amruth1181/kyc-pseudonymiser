"""Dates as printed in the visual zone of identity documents.

Issuing states print the same date differently: '12 AUG 1974' (synthetic template),
'18 SEP /SEPT 99' (United Kingdom, English and French month), '12 07 1973' (France,
Luxembourg) or '12/07/1973' (India). A date is parsed together with its layout so a
shifted date can be written back exactly the way the original was printed.
"""
from __future__ import annotations

import re
from datetime import date

MONTHS = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()
MONTHS_FR = "JAN FEV MARS AVR MAI JUIN JUIL AOUT SEPT OCT NOV DEC".split()

_WORDED = re.compile(r"^\s*(\d{1,2})\s+([A-Z]{3})\s*(/\s*[A-Z]{3,5})?\s+(\d{4}|\d{2})\s*$")
_NUMERIC = re.compile(r"^\s*(\d{2})\s*([ ./-])\s*(\d{2})\s*[ ./-]\s*(\d{4}|\d{2})\s*$")


def _full_year(text: str) -> int:
    year = int(text)
    if len(text) == 2:                               # a birth year is never in the future
        year += 2000 if year <= date.today().year % 100 else 1900
    return year


def parse_viz(text: str) -> tuple[date, tuple] | None:
    """Return the date and its layout, or None when the text is not a date in a known layout."""
    t = text.upper()
    try:
        m = _WORDED.match(t)
        if m and m.group(2) in MONTHS:
            return (date(_full_year(m.group(4)), MONTHS.index(m.group(2)) + 1, int(m.group(1))),
                    ("worded", bool(m.group(3)), len(m.group(4))))
        m = _NUMERIC.match(t)
        if m:
            return (date(_full_year(m.group(4)), int(m.group(3)), int(m.group(1))),
                    ("numeric", m.group(2), len(m.group(4))))
    except ValueError:
        return None
    return None


def format_viz(d: date, layout: tuple) -> str:
    kind, variant, year_digits = layout
    year = f"{d.year}" if year_digits == 4 else f"{d.year % 100:02d}"
    if kind == "numeric":
        return f"{d.day:02d}{variant}{d.month:02d}{variant}{year}"
    month = MONTHS[d.month - 1] + (f" /{MONTHS_FR[d.month - 1]}" if variant else "")
    return f"{d.day:02d} {month} {year}"
