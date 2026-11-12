"""Synthetic KYC corpus: fictional people, fictional state (Utopia, code UTO).

Nothing here is a real person or a real document design. The corpus exists
so the pipeline can be built and measured without touching real identity
documents, and it deliberately contains the hard cases:

* one person on several documents, written differently each time
  (MRZ order, initials, mixed case, a data-entry typo);
* two people sharing a surname and a first initial, at different addresses;
* a person with no passport, linked only through their address;
* a third party's name inside a bank-transfer description.

Ground truth (persons and per-document PII values) is written next to the
images and is only used by ``kycpseudo evaluate``.
"""
from __future__ import annotations

import json
import math
import random
import zlib
from dataclasses import dataclass, asdict, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..mrz import build_td3, yymmdd_to_viz
from ..names import name_pools
from ..fonts import font_path

ISSUER = "UTO"


@dataclass
class Person:
    person_id: str
    locale: str
    sex: str
    given: list[str]
    surname: str
    dob: str              # YYMMDD
    passport_no: str
    place_of_birth: str
    street: str           # '14 HARBOUR ROAD'
    postcode: str
    city: str
    account_no: str
    iban: str
    issue: str = "190415"
    expiry: str = "290414"
    docs: list[str] = field(default_factory=list)


def _iban(country: str, bban: str) -> str:
    """Compute IBAN check digits (ISO 13616 mod-97) for a fictional country code."""
    rearranged = bban + country + "00"
    num = "".join(str(int(c, 36)) for c in rearranged)
    check = 98 - int(num) % 97
    return f"{country}{check:02d}{bban}"


def make_people(n: int, rng: random.Random) -> list[Person]:
    pools = name_pools()
    locales = ["nordic", "indian", "french", "portuguese"]
    people: list[Person] = []
    used_names: set[tuple] = set()
    for i in range(n):
        loc = locales[i % len(locales)]
        sex = rng.choice("FM")
        given = rng.sample(pools[loc][sex], k=rng.choice([1, 2, 2]))
        surname = rng.choice(pools[loc]["S"])
        if (tuple(given), surname) in used_names:
            surname = rng.choice(pools[loc]["S"])
        used_names.add((tuple(given), surname))
        yy, mm, dd = rng.randint(55, 99), rng.randint(1, 12), rng.randint(1, 28)
        num = rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") + "".join(rng.choice("0123456789") for _ in range(7)) + rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ")
        acct = "".join(rng.choice("0123456789") for _ in range(10))
        bban = "".join(rng.choice("0123456789") for _ in range(16))
        people.append(Person(
            person_id=f"PER-{i:03d}", locale=loc, sex=sex, given=given, surname=surname,
            dob=f"{yy:02d}{mm:02d}{dd:02d}", passport_no=num,
            place_of_birth=rng.choice(pools["cities"]),
            street=f"{rng.randint(2, 98)} {rng.choice(pools['streets'])}",
            postcode=f"NV-{rng.randint(1000, 9899)}", city=rng.choice(pools["cities"]),
            account_no=acct, iban=_iban("UT", bban),
        ))
    # Hard case: same surname, same first initial, different person and address.
    a = people[0]
    twin = [(g, sx) for sx in "FM" for g in pools[a.locale][sx] if g[0] == a.given[0][0] and g not in a.given]
    if twin:
        g, sx = twin[0]
        b = Person(**{**asdict(people[1]), "person_id": f"PER-{n:03d}", "locale": a.locale,
                      "sex": sx, "given": [g], "surname": a.surname, "docs": []})
        b.street = f"{rng.randint(2, 98)} {rng.choice(pools['streets'])}"
        people.append(b)
    return people


# --------------------------------------------------------------------------- drawing helpers

def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path(name), size)


def _guilloche(w: int, h: int, rng: random.Random) -> Image.Image:
    img = np.full((h, w, 3), (236, 241, 233), np.uint8)
    xs = np.arange(w)
    for k in range(26):
        amp, period, phase = rng.uniform(10, 40), rng.uniform(140, 420), rng.uniform(0, 6.28)
        base = int(h * k / 26)
        ys = (base + amp * np.sin(2 * math.pi * xs / period + phase)).astype(int)
        pts = np.stack([xs, ys], 1).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, (206, 222, 214), 1, cv2.LINE_AA)
    return Image.fromarray(img)


def draw_avatar(draw: ImageDraw.ImageDraw, box, seed: int, bg=(214, 214, 210)):
    """Procedural placeholder portrait (no real or generated face)."""
    r = random.Random(seed)
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=bg, outline=(120, 120, 120), width=2)
    cx, w, h = (x0 + x1) // 2, x1 - x0, y1 - y0
    tone = r.randint(110, 175)
    draw.ellipse((cx - w * 0.22, y0 + h * 0.16, cx + w * 0.22, y0 + h * 0.56), fill=(tone, tone - 8, tone - 16))
    draw.pieslice((x0 + w * 0.1, y0 + h * 0.6, x1 - w * 0.1, y1 + h * 0.35), 180, 360,
                  fill=(r.randint(40, 90), r.randint(40, 90), r.randint(60, 120)))
    hair = r.randint(20, 90)
    draw.chord((cx - w * 0.24, y0 + h * 0.12, cx + w * 0.24, y0 + h * 0.42), 180, 360, fill=(hair, hair - 10 if hair > 10 else 0, 10))


def draw_signature(draw: ImageDraw.ImageDraw, box, seed: int, ink=(30, 40, 90)):
    r = random.Random(seed)
    x0, y0, x1, y1 = box
    pts, x = [], x0 + 8
    mid = (y0 + y1) / 2
    while x < x1 - 10:
        pts.append((x, mid + r.uniform(-1, 1) * (y1 - y0) * 0.32))
        x += r.uniform(8, 22)
    for _ in range(2):
        draw.line(pts, fill=ink, width=3, joint="curve")
        pts = [(px + r.uniform(-3, 3), py + r.uniform(-4, 4)) for px, py in pts[::2]]


def scan(img: Image.Image, rng: random.Random, max_rot: float = 0.5) -> Image.Image:
    """Simulate a flatbed scan: small rotation, blur, sensor noise, JPEG."""
    a = np.array(img.convert("RGB"))
    h, w = a.shape[:2]
    ang = rng.uniform(-max_rot, max_rot)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    a = cv2.warpAffine(a, m, (w, h), borderValue=(250, 250, 250), flags=cv2.INTER_LINEAR)
    a = cv2.GaussianBlur(a, (0, 0), 0.6)
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, 3.5, a.shape)
    a = np.clip(a.astype(float) + noise, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(a, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    a = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return Image.fromarray(a)


# --------------------------------------------------------------------------- documents

def render_passport(p: Person, rng: random.Random) -> tuple[Image.Image, list[dict]]:
    W, H = 1500, 1060
    img = _guilloche(W, H, rng)
    d = ImageDraw.Draw(img)
    ink, lab = (28, 28, 38), (95, 100, 105)
    d.text((60, 40), "UTOPIA", font=_font("sans_bold", 46), fill=(30, 60, 90))
    d.text((60, 100), "PASSPORT / PASSEPORT", font=_font("sans", 26), fill=lab)
    draw_avatar(d, (60, 190, 400, 640), seed=zlib.crc32(p.person_id.encode()))
    L, V = _font("sans", 20), _font("sans_bold", 32)
    x1, x2, x3 = 450, 820, 1150
    rows = [
        (170, [(x1, "Type", "P"), (x2, "Code", ISSUER), (x3, "Passport No.", p.passport_no)]),
        (260, [(x1, "Surname / Nom", p.surname)]),
        (350, [(x1, "Given names / Prenoms", " ".join(p.given))]),
        (440, [(x1, "Nationality", "UTOPIAN"), (x3, "Sex", p.sex)]),
        (530, [(x1, "Date of birth", yymmdd_to_viz(p.dob)), (x2, "Place of birth", p.place_of_birth)]),
        (620, [(x1, "Date of issue", yymmdd_to_viz(p.issue)), (x2, "Date of expiry", yymmdd_to_viz(p.expiry))]),
    ]
    for y, cells in rows:
        for x, label, value in cells:
            d.text((x, y), label, font=L, fill=lab)
            d.text((x, y + 28), value, font=V, fill=ink)
    d.text((x1, 712), "Signature", font=L, fill=lab)
    draw_signature(d, (x1, 740, x1 + 360, 820), seed=zlib.crc32((p.person_id + "sig").encode()))
    l1, l2 = build_td3("P", ISSUER, p.surname, " ".join(p.given), p.passport_no, ISSUER,
                       p.dob, p.sex, p.expiry, "")
    mono = _font("mono", 50)
    pitch = 31
    for row, line in enumerate((l1, l2)):
        y = 880 + row * 78
        for i, ch in enumerate(line):
            d.text((60 + i * pitch, y), ch, font=mono, fill=ink)
    pii = [
        {"type": "surname", "value": p.surname},
        {"type": "given_names", "value": " ".join(p.given)},
        {"type": "doc_number", "value": p.passport_no},
        {"type": "dob", "value": yymmdd_to_viz(p.dob)},
        {"type": "place_of_birth", "value": p.place_of_birth},
        {"type": "mrz_line1", "value": l1},
        {"type": "mrz_line2", "value": l2},
    ]
    return img, pii


def _a4(rng):
    return Image.new("RGB", (1654, 2339), (251, 251, 248))


def render_bill(p: Person, name_text: str, rng: random.Random) -> tuple[Image.Image, list[dict]]:
    img = _a4(rng)
    d = ImageDraw.Draw(img)
    H1, B, R = _font("sans_bold", 56), _font("sans_bold", 34), _font("sans", 34)
    ink = (25, 25, 30)
    d.text((140, 150), "UTOPIA POWER & LIGHT", font=H1, fill=(20, 70, 120))
    d.text((140, 230), "Electricity bill - October 2026", font=R, fill=ink)
    d.line((140, 300, 1514, 300), fill=(150, 150, 150), width=2)
    y = 380
    for label, value in [("Customer:", name_text), ("Address:", p.street), ("", f"{p.postcode} {p.city}"),
                         ("Account number:", p.account_no)]:
        if label:
            d.text((140, y), label, font=B, fill=ink)
        d.text((520, y), value, font=R, fill=ink)
        y += 70
    y += 60
    d.text((140, y), "Billing period: 01 Oct 2026 - 31 Oct 2026", font=R, fill=ink); y += 70
    for item, amt in [("Energy used (412 kWh)", "98.88"), ("Standing charge", "12.40"), ("Taxes", "16.70")]:
        d.text((180, y), item, font=R, fill=ink); d.text((1300, y), amt, font=R, fill=ink); y += 60
    d.text((180, y + 20), "Total due", font=B, fill=ink); d.text((1300, y + 20), "127.98", font=B, fill=ink)
    d.text((140, 2150), "Thank you for choosing Utopia Power & Light.", font=_font("sans", 26), fill=(110, 110, 110))
    pii = [{"type": "full_name", "value": name_text}, {"type": "address", "value": p.street},
           {"type": "postcode", "value": p.postcode}, {"type": "account_number", "value": p.account_no}]
    return img, pii


def render_statement(p: Person, name_text: str, third_party: Person | None, rng: random.Random):
    img = _a4(rng)
    d = ImageDraw.Draw(img)
    H1, B, R = _font("sans_bold", 56), _font("sans_bold", 34), _font("sans", 32)
    ink = (25, 25, 30)
    d.text((140, 150), "BANK OF UTOPIA", font=H1, fill=(90, 30, 60))
    d.text((140, 230), "Account statement - October 2026", font=R, fill=ink)
    d.line((140, 300, 1514, 300), fill=(150, 150, 150), width=2)
    iban_fmt = " ".join(p.iban[i:i + 4] for i in range(0, len(p.iban), 4))
    y = 380
    for label, value in [("Account holder:", name_text), ("Address:", p.street), ("", f"{p.postcode} {p.city}"),
                         ("IBAN:", iban_fmt)]:
        if label:
            d.text((140, y), label, font=B, fill=ink)
        d.text((560, y), value, font=R, fill=ink)
        y += 70
    y += 60
    d.text((140, y), "Date", font=B, fill=ink); d.text((400, y), "Description", font=B, fill=ink)
    d.text((1300, y), "Amount", font=B, fill=ink); y += 70
    tx = [("03 Oct", "Card payment GROCERY MART", "-54.20"), ("07 Oct", "Salary NOVA PORTA LOGISTICS", "2410.00"),
          ("12 Oct", "Direct debit UTOPIA POWER", "-127.98")]
    pii = [{"type": "full_name", "value": name_text}, {"type": "address", "value": p.street},
           {"type": "postcode", "value": p.postcode}, {"type": "iban", "value": iban_fmt}]
    if third_party:
        tp_name = f"{third_party.given[0].capitalize()} {third_party.surname.capitalize()}"
        tx.append(("18 Oct", f"Transfer to {tp_name}", "-300.00"))
        pii.append({"type": "third_party_name", "value": tp_name, "person_id": third_party.person_id})
    tx.append(("25 Oct", "Card payment CITY TRANSIT", "-42.00"))
    for dt, desc, amt in tx:
        d.text((140, y), dt, font=R, fill=ink); d.text((400, y), desc, font=R, fill=ink)
        d.text((1300, y), amt, font=R, fill=ink); y += 60
    return img, pii


def _variant(p: Person, style: str, rng: random.Random) -> str:
    g = [x.capitalize() for x in p.given]
    s = p.surname.capitalize()
    if style == "full":
        return " ".join(g + [s])
    if style == "initials_upper":
        return " ".join(f"{x[0]}." for x in p.given) + " " + p.surname
    if style == "surname_first":
        return f"{p.surname}, {' '.join(g)}"
    if style == "typo":
        i = rng.randint(1, len(s) - 2)
        swapped = s[:i] + {"a": "e", "e": "a", "i": "y", "o": "u", "u": "o"}.get(s[i], "e") + s[i + 1:]
        return " ".join(g + [swapped])
    return " ".join(g + [s])


def generate(out: Path, n_people: int = 12, seed: int = 7) -> dict:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    people = make_people(n_people, rng)
    truth = {"persons": [], "documents": []}
    no_passport = {people[-2].person_id}          # linked only through address
    styles = ["full", "initials_upper", "surname_first", "typo"]
    k = 0

    def save(img, kind, p, pii):
        nonlocal k
        doc_id = f"DOC-{k:04d}"
        k += 1
        scan(img, rng).save(out / "images" / f"{doc_id}.png")
        p.docs.append(doc_id)
        truth["documents"].append({"doc_id": doc_id, "doc_type": kind, "person_id": p.person_id, "pii": pii})

    for idx, p in enumerate(people):
        if p.person_id not in no_passport:
            img, pii = render_passport(p, rng)
            save(img, "passport", p, pii)
        img, pii = render_bill(p, _variant(p, styles[idx % len(styles)], rng), rng)
        save(img, "utility_bill", p, pii)
        if idx % 2 == 0 or p.person_id in no_passport:
            tp = people[(idx + 3) % len(people)] if idx % 4 == 0 else None
            img, pii = render_statement(p, _variant(p, styles[(idx + 1) % len(styles)], rng), tp, rng)
            save(img, "bank_statement", p, pii)
    truth["persons"] = [asdict(p) for p in people]
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    return truth
