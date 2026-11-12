import random

import cv2
import numpy as np
import pytest

from kycpseudo.stages.s1_ingest import deskew, estimate_skew
from kycpseudo.synth.generator import make_people, render_bill, render_passport


@pytest.mark.parametrize("angle", [0.37, -0.23, 0.5, -0.08])
@pytest.mark.parametrize("kind", ["bill", "passport"])
def test_skew_is_estimated_and_removed(angle, kind):
    p = make_people(2, random.Random(1))[0]
    img = render_bill(p, "Anna Maria Eriksson", random.Random(1))[0] if kind == "bill" else \
        render_passport(p, random.Random(2))[0]
    a = np.array(img)
    h, w = a.shape[:2]
    tilted = cv2.warpAffine(a, cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1), (w, h), borderValue=(250, 250, 250))
    correction = estimate_skew(tilted)
    assert abs(correction + angle) <= 0.05
    assert abs(estimate_skew(deskew(tilted, correction))) <= 0.05
