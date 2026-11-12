"""Image-level tests. Need the Tesseract binary; skipped when it is missing."""
import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from kycpseudo.config import Config
from kycpseudo.fonts import font_path
from kycpseudo.models import Field, PageRecord
from kycpseudo.ocr import read_line

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")


def _scan_with_name(name="Oskar Malmberg"):
    img = Image.new("RGB", (900, 160), (248, 248, 244))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path("sans"), 34)
    d.text((40, 30), "Customer:", font=font, fill=(25, 25, 30))
    d.text((300, 30), name, font=font, fill=(25, 25, 30))
    a = np.array(img)
    return np.clip(a + np.random.default_rng(0).normal(0, 3, a.shape), 0, 255).astype(np.uint8)


def _name_box(a):
    from kycpseudo.ocr import read_words
    ws = [w for w in read_words(a) if w.box[0] >= 290]
    return (min(w.box[0] for w in ws), min(w.box[1] for w in ws), max(w.box[2] for w in ws), max(w.box[3] for w in ws))


def _render_one(a, box):
    from kycpseudo.stages.s7_render import render

    rec = PageRecord("t", "t", doc_type="utility_bill")
    rec.fields = [Field("full_name", "Oskar Malmberg", box, "test", surrogate="Nils Rydberg")]
    cfg = Config()
    cfg.hmac_key = b"k" * 32
    return render(a, rec, cfg), rec.edits[0]


def test_render_replaces_text_exactly_and_locally():
    a = _scan_with_name()
    out, edit = _render_one(a, _name_box(a))
    assert read_line(out[20:85, 290:640]) == "Nils Rydberg"
    assert edit["ok"] and not edit["erase_residue"] and edit["style_score"] > 0.7
    changed = np.any(out != a, axis=2)
    x0, y0, x1, y1 = edit["box"]
    changed[y0:y1, x0:x1] = False
    assert changed.sum() == 0                        # nothing outside the edit box
    assert np.array_equal(out[:, :280], a[:, :280])  # the label is untouched


def test_descender_outside_the_box_is_erased():
    a = _scan_with_name()
    x0, y0, x1, y1 = _name_box(a)
    out, _ = _render_one(a, (x0, y0, x1, y1 - 7))     # box clips the tail of the 'g'
    assert read_line(out[20:85, 290:640]) == "Nils Rydberg"


@pytest.mark.slow
def test_end_to_end_small_corpus(tmp_path, monkeypatch):
    from kycpseudo.config import load_keys
    from kycpseudo.evaluate import evaluate
    from kycpseudo.pipeline import run
    from kycpseudo.synth.generator import generate

    monkeypatch.chdir(tmp_path)
    generate(tmp_path / "data", n_people=4, seed=3)
    cfg = load_keys(Config(), demo_keys=True, keydir=tmp_path / ".keys")
    summary = run(tmp_path / "data" / "images", tmp_path / "run", cfg, workers=1, use_cache=False)
    m = evaluate(tmp_path / "data" / "truth.json", tmp_path / "run", cfg)
    assert m["leaks_independent_check"]["documents_with_leak"] == 0
    assert m["entity_resolution"]["pairwise_precision"] == 1.0
    assert m["mrz"]["check_digits_valid"] == m["mrz"]["rebuilt"]
    assert m["locality_pixels_changed_outside_edits"] == 0
    assert summary["documents"] == m["documents"]
