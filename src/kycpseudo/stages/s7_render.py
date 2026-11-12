"""Stage 7: erase the original text and draw the surrogate in the same style.

Deterministic by design: the surrogate string is drawn with a font, so the
text on the page is exactly the intended text. Generative text editors
(diffusion scene-text editing) do not guarantee the string and are left as
a fallback hook (``fallback_editor``) for fields that fail verification.

Style matching works by re-rendering the ORIGINAL text and fitting it to the
scan, then drawing the surrogate with the fitted parameters:

  1. size      - font size whose ink height equals the measured ink height
  2. position  - draw origin that puts the original's ink where it is on the
                 page; the surrogate reuses it, so baselines line up
  3. width     - horizontal scale that matches the original's ink width
  4. ink       - least-squares ink colour given the fitted blur
  5. blur      - Gaussian sigma that best reproduces the scanned edges
  6. noise     - standard deviation of the scan's sensor noise

The fit residual is kept as a style score; a poor fit (wrong font) is sent
to review.
"""
from __future__ import annotations

import hashlib
import hmac
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rapidfuzz import fuzz

from ..config import Config
from ..fonts import font_for, font_path
from ..layout import vnorm
from ..models import Field, PageRecord, Region
from ..ocr import read_line
from ..synth.generator import draw_avatar, draw_signature

Box = tuple[int, int, int, int]


def _seed(cfg: Config, *parts: str) -> int:
    return int.from_bytes(hmac.new(cfg.hmac_key, "|".join(parts).encode(), hashlib.sha256).digest()[:8], "big")


def _render_alpha(text: str, font: ImageFont.FreeTypeFont, mono_pitch: float | None = None) -> tuple[np.ndarray, int, int]:
    """Rasterise text; return alpha (0-1) and the draw origin inside the canvas."""
    pad = font.size
    if mono_pitch:
        w = int(mono_pitch * len(text)) + 2 * pad
    else:
        w = int(font.getlength(text)) + 2 * pad
    h = int(font.size * 1.6) + 2 * pad
    canvas = Image.new("L", (max(w, 1), max(h, 1)), 0)
    d = ImageDraw.Draw(canvas)
    if mono_pitch:
        for i, ch in enumerate(text):
            d.text((pad + i * mono_pitch, pad), ch, font=font, fill=255)
    else:
        d.text((pad, pad), text, font=font, fill=255)
    return np.asarray(canvas, np.float32) / 255.0, pad, pad


def _ink_bbox(a: np.ndarray, thr: float = 0.5) -> Box | None:
    ys, xs = np.nonzero(a > thr)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _measure(crop_rgb: np.ndarray) -> dict:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    thr, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = ink > 0
    # Box at the 50% level between paper and ink, the same level used on rendered text,
    # so the blur halo does not inflate the measured size.
    mid = (float(np.median(gray[~ink])) + float(np.median(gray[ink]))) / 2 if ink.any() and (~ink).any() else thr
    bbox = _ink_bbox((gray < mid).astype(np.float32))
    core = gray < thr * 0.6
    ink_rgb = np.median(crop_rgb[core if core.any() else ink], axis=0)
    resid = gray.astype(np.float32) - cv2.medianBlur(gray, 5).astype(np.float32)
    noise = float(np.std(resid[~cv2.dilate(ink.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)]) or 0.0)
    return {"gray": gray, "thr": thr, "ink": ink, "bbox": bbox, "ink_rgb": ink_rgb,
            "ink_level": float(np.median(gray[ink])), "bg_level": float(np.median(gray[~ink])), "noise": noise}


def _fit(text: str, fname: str, meas: dict, mono: bool, crop_rgb: np.ndarray) -> dict | None:
    """Fit size, origin, horizontal scale and blur by re-rendering the original text."""
    bbox = meas["bbox"]
    if bbox is None:
        return None
    target_h, target_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    pitch_ratio = None
    lo, hi, best = 6, 220, None
    while lo <= hi:                                  # font size by ink height
        size = (lo + hi) // 2
        font = ImageFont.truetype(font_path(fname), size)
        pitch = (target_w / max(len(text) - 0.4, 1)) if mono else None
        a, ox, oy = _render_alpha(text, font, pitch)
        bb = _ink_bbox(a)
        if bb is None:
            lo = size + 1
            continue
        h = bb[3] - bb[1]
        best = (size, a, ox, oy, bb, pitch)
        if h < target_h:
            lo = size + 1
        elif h > target_h:
            hi = size - 1
        else:
            break
    size, a, ox, oy, bb, pitch = best
    sx = 1.0 if mono else float(np.clip(target_w / max(bb[2] - bb[0], 1), 0.8, 1.25))
    origin = (bbox[0] - (bb[0] - ox) * sx, bbox[1] - (bb[1] - oy))
    # Blur and ink colour: which sigma, and which ink, make the re-rendered original look
    # most like the scan? For a given blur the best ink is a least-squares solution:
    #   pixel = bg + (ink - bg) * alpha   =>   ink = bg + sum((pixel - bg) * alpha) / sum(alpha^2)
    gray = meas["gray"].astype(np.float32)
    rgb = crop_rgb.astype(np.float32)
    bg_rgb = np.median(rgb[~meas["ink"]], axis=0)
    x0, y0, x1, y1 = bbox
    contrast = max(meas["bg_level"] - meas["ink_level"], 1.0)
    best = None
    for dx in (-1.0, -0.5, 0.0, 0.5, 1.0):               # sub-pixel alignment
        for dy in (-1.0, -0.5, 0.0, 0.5, 1.0):
            shifted = _place(a, ox, oy, (origin[0] + dx, origin[1] + dy), sx, gray.shape)
            for sigma in (0.0, 0.4, 0.6, 0.8, 1.0, 1.3):
                al = (cv2.GaussianBlur(shifted, (0, 0), sigma) if sigma else shifted)[y0:y1, x0:x1]
                denom = float((al * al).sum()) or 1.0
                ink = bg_rgb + ((rgb[y0:y1, x0:x1] - bg_rgb) * al[..., None]).sum(axis=(0, 1)) / denom
                synth = bg_rgb * (1 - al[..., None]) + ink * al[..., None]
                err = float(np.mean(np.abs(synth - rgb[y0:y1, x0:x1])))
                if best is None or err < best[1]:
                    best = (sigma, err, np.clip(ink, 0, 255), (origin[0] + dx, origin[1] + dy))
    sigma, err, ink, origin = best
    return {"size": size, "origin": origin, "sx": sx, "sigma": sigma, "pitch": pitch, "ink_rgb": ink,
            "style_score": round(1 - err / contrast, 3)}


def _place(a: np.ndarray, ox: int, oy: int, origin: tuple[float, float], sx: float, shape) -> np.ndarray:
    """Put a rendered alpha (drawn at canvas origin ox, oy) at ``origin`` in an image of ``shape``."""
    if sx != 1.0:
        a = cv2.resize(a, (max(1, int(round(a.shape[1] * sx))), a.shape[0]), interpolation=cv2.INTER_AREA)
        ox = ox * sx
    m = np.float32([[1, 0, origin[0] - ox], [0, 1, origin[1] - oy]])
    return cv2.warpAffine(a, m, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)


def _erase(region: np.ndarray, mask: np.ndarray, cfg: Config) -> np.ndarray:
    if cfg.use_lama:
        try:
            from simple_lama_inpainting import SimpleLama
            out = np.array(SimpleLama()(Image.fromarray(region), Image.fromarray(mask * 255)))
            return out[: region.shape[0], : region.shape[1]]
        except ImportError:
            pass
    return cv2.inpaint(region, mask.astype(np.uint8) * 255, cfg.inpaint_radius, cv2.INPAINT_TELEA)


def _stroke_mask(win_gray: np.ndarray, thr: float, box_rows: tuple[int, int], box_cols: tuple[int, int]) -> np.ndarray:
    """Ink strokes that belong to the field: connected components mostly inside the box.

    Catches descenders and accents that hang outside the OCR box (the tail of a 'g'),
    while a neighbouring line's descender reaching into the box stays untouched.
    """
    ink = (win_gray < thr).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    r0, r1 = box_rows
    c0, c1 = box_cols
    inside = np.zeros_like(ink, bool)
    inside[r0:r1, c0:c1] = True
    keep = np.zeros(n, bool)
    for lab in range(1, n):
        comp = labels == lab
        area = stats[lab, cv2.CC_STAT_AREA]
        keep[lab] = (comp & inside).sum() >= 0.5 * area
    return keep[labels]


def render_field(img: np.ndarray, rec: PageRecord, f: Field, cfg: Config) -> dict:
    H, W = img.shape[:2]
    p = cfg.box_pad
    x0, y0, x1, y1 = f.box
    x0, y0, x1, y1 = max(0, x0 - p), max(0, y0 - p), min(W, x1 + p), min(H, y1 + p)
    crop = img[y0:y1, x0:x1].copy()
    meas = _measure(crop)
    mono = f.ftype.startswith("mrz_")
    fname = font_for(rec.doc_type, f.ftype)
    fit = _fit(f.value if not mono else f.value[:44], fname, meas, mono, crop)
    if fit is None:
        return {"ok": False, "reason": "no_ink_found"}

    # Working window: the box, room on the right for a longer surrogate, and room
    # above/below for strokes that spill out of the OCR box.
    font = ImageFont.truetype(font_path(fname), fit["size"])
    a_new, ox, oy = _render_alpha(f.surrogate, font, fit["pitch"])
    bb_new = _ink_bbox(a_new)
    new_w = (bb_new[2] - bb_new[0]) * fit["sx"] if bb_new else 0
    extra = int(max(0, new_w - (x1 - x0) + 2 * p)) + 6
    wx1 = min(W, x1 + extra)
    mv = int(0.45 * (y1 - y0))
    ey0, ey1 = max(0, y0 - mv), min(H, y1 + mv)
    dy = y0 - ey0
    win = img[ey0:ey1, x0:wx1].copy()

    # 1. erase the field's ink strokes (plus anti-aliased edges)
    win_gray = cv2.cvtColor(win, cv2.COLOR_RGB2GRAY)
    mask = _stroke_mask(win_gray, meas["thr"], (dy, dy + (y1 - y0)), (0, x1 - x0)).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    ctx = 12
    cy0, cx0 = max(0, ey0 - ctx), max(0, x0 - ctx)
    cy1, cx1 = min(H, ey1 + ctx), min(W, wx1 + ctx)
    big = img[cy0:cy1, cx0:cx1].copy()
    big_mask = np.zeros(big.shape[:2], np.uint8)
    big_mask[ey0 - cy0:ey1 - cy0, x0 - cx0:wx1 - cx0] = mask
    erased_big = _erase(big, big_mask, cfg)
    erased = erased_big[ey0 - cy0:ey1 - cy0, x0 - cx0:wx1 - cx0]
    base = win.copy()
    base[mask > 0] = erased[mask > 0]

    residue = read_line(cv2.resize(base[:, : x1 - x0], None, fx=2, fy=2))
    residue_hit = len(vnorm(residue)) >= 3 and fuzz.partial_ratio(vnorm(residue), vnorm(f.value)) >= 80

    # 2. draw the surrogate with the fitted style
    origin = (fit["origin"][0], fit["origin"][1] + dy)
    alpha = _place(a_new, ox, oy, origin, fit["sx"], win.shape[:2])
    if fit["sigma"]:
        alpha = cv2.GaussianBlur(alpha, (0, 0), fit["sigma"])
    out = base.astype(np.float32) * (1 - alpha[..., None]) + fit["ink_rgb"][None, None, :] * alpha[..., None]

    # 3. match sensor noise where pixels were synthesised (inpainted area and new ink)
    touched = (mask > 0) | (alpha > 0.02)
    rng = np.random.default_rng(_seed(cfg, rec.doc_id, f.ftype, str(f.box)))
    noise = rng.normal(0, meas["noise"], out.shape[:2])[..., None]
    out = np.clip(np.rint(out + noise), 0, 255).astype(np.uint8)
    window = img[ey0:ey1, x0:wx1]
    window[touched] = out[touched]          # only synthesised pixels are written back

    ys, xs = np.nonzero(touched)
    changed = (x0 + int(xs.min()), ey0 + int(ys.min()), x0 + int(xs.max()) + 1, ey0 + int(ys.max()) + 1) \
        if len(xs) else f.box
    return {"ok": True, "field": f.ftype, "box": changed, "render_box": (x0, y0, wx1, y1),
            "font_size": fit["size"], "sx": round(fit["sx"], 3), "sigma": fit["sigma"],
            "style_score": fit["style_score"], "erase_residue": bool(residue_hit)}


def render_region(img: np.ndarray, rec: PageRecord, r: Region, cfg: Config) -> dict:
    x0, y0, x1, y1 = r.box
    seed = _seed(cfg, "region", r.rtype, r.entity_id or "")
    crop = img[y0:y1, x0:x1].copy()
    if r.rtype == "photo":
        # Placeholder portrait per entity. A production system plugs in a synthetic-face
        # generator here, one face per entity, reused on all that person's documents.
        canvas = Image.fromarray(crop)
        draw_avatar(ImageDraw.Draw(canvas), (0, 0, x1 - x0 - 1, y1 - y0 - 1), seed)
        new = np.asarray(canvas).astype(np.float32)
    elif r.rtype == "signature":
        meas = _measure(crop)
        mask = cv2.dilate(meas["ink"].astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
        base = cv2.inpaint(crop, mask * 255, cfg.inpaint_radius, cv2.INPAINT_TELEA)
        canvas = Image.fromarray(base)
        draw_signature(ImageDraw.Draw(canvas), (4, 4, x1 - x0 - 4, y1 - y0 - 4), seed,
                       ink=tuple(int(v) for v in meas["ink_rgb"]))
        new = np.asarray(canvas).astype(np.float32)
    else:
        return {"ok": False, "reason": f"no renderer for {r.rtype}"}
    new = cv2.GaussianBlur(new, (0, 0), 0.6)
    new += np.random.default_rng(seed).normal(0, 3.0, new.shape)
    img[y0:y1, x0:x1] = np.clip(new, 0, 255).astype(np.uint8)
    return {"ok": True, "field": r.rtype, "box": r.box}


def fallback_editor(img: np.ndarray, f: Field) -> bool:
    """Hook for a generative editor (e.g. TextCtrl) on fields the deterministic path cannot pass.

    Must be followed by verification; return False when no editor is configured.
    """
    return False


def render(img: np.ndarray, rec: PageRecord, cfg: Config) -> np.ndarray:
    out = img.copy()
    for i, f in enumerate(rec.fields):
        if not f.surrogate:
            continue
        res = render_field(out, rec, f, cfg)
        res["idx"] = i
        rec.edits.append(res)
        if not res.get("ok"):
            rec.review_reasons.append(f"render_failed:{f.ftype}")
        elif res["erase_residue"]:
            rec.review_reasons.append(f"erase_residue:{f.ftype}")
        elif res["style_score"] < 0.6:
            rec.review_reasons.append(f"style_mismatch:{f.ftype}")
    for r in rec.regions:
        rec.edits.append(render_region(out, rec, r, cfg))
    return out
