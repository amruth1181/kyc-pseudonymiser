"""Stage 1: ingest and normalise.

* hash the file (identity for idempotent re-runs),
* rasterise PDFs at a fixed DPI, which also drops the text layer,
  annotations and metadata that would otherwise carry the original text,
* correct small skews so text lines are horizontal.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pages(path: Path, dpi: int = 300) -> list[np.ndarray]:
    """Return RGB pages. Image files are re-encoded from pixels only (EXIF dropped)."""
    if path.suffix.lower() == ".pdf":
        try:
            import pypdfium2 as pdfium
        except ImportError as e:
            raise SystemExit("PDF input needs the 'pdf' extra: pip install kycpseudo[pdf]") from e
        pdf = pdfium.PdfDocument(str(path))
        return [np.array(pdf[i].render(scale=dpi / 72).to_pil().convert("RGB")) for i in range(len(pdf))]
    with Image.open(path) as im:
        return [np.array(im.convert("RGB"))]


def _shear_profiles(ink: np.ndarray, angles, n_cols: int = 160):
    """Row-profile sharpness for each candidate angle.

    Columns are averaged into ``n_cols`` strips but every row is kept (vertical
    resolution is what the profile needs). For a small angle, rotation is
    approximately a shear, so each strip is shifted by x * tan(angle).
    """
    h, w = ink.shape
    strips = cv2.resize(ink, (n_cols, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    xs = (np.arange(n_cols) + 0.5) * w / n_cols - w / 2
    max_shift = int(np.ceil(w / 2 * np.tan(np.radians(max(abs(a) for a in angles))))) + 1
    scores = []
    for a in angles:
        shifts = np.round(xs * np.tan(np.radians(a))).astype(int) + max_shift
        prof = np.zeros(h + 2 * max_shift, np.float32)
        for j, sft in enumerate(shifts):
            prof[sft:sft + h] += strips[:, j]
        scores.append(float(np.var(prof)))
    return scores


def estimate_skew(rgb: np.ndarray, max_deg: float = 2.0) -> float:
    """Projection-profile skew estimate: text rows give the sharpest row histogram when level.

    Coarse 0.1 degree search, then a 0.01 degree search around the best angle.
    A residual tilt of 0.3 degrees adds 2-3 px of height to a long line and
    would bias the font-size fit in stage 7, so the fine pass matters.
    Returns the rotation (degrees, OpenCV convention) that levels the page.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coarse_angles = list(np.arange(-max_deg, max_deg + 1e-9, 0.1))
    sc = _shear_profiles(ink, coarse_angles)
    coarse = coarse_angles[int(np.argmax(sc))]
    fine_angles = list(np.arange(coarse - 0.12, coarse + 0.12 + 1e-9, 0.01))
    sf = _shear_profiles(ink, fine_angles)
    tilt = fine_angles[int(np.argmax(sf))]
    return float(round(-tilt, 3))          # rotating by -tilt levels the page


def deskew(rgb: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.01:
        return rgb
    h, w = rgb.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    border = tuple(int(v) for v in np.median(rgb[:10].reshape(-1, 3), axis=0))
    return cv2.warpAffine(rgb, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=border)


def discover(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXT | {".pdf"})
