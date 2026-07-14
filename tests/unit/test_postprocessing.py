"""Tests des post-traitements ne dépendant pas de PyTorch."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from anomaly_detection.inference.postprocessing import (
    heatmap_to_base64_png,
    min_max_normalize,
)


def test_min_max_normalize_range():
    out = min_max_normalize(np.array([[0.0, 5.0], [10.0, 2.5]]))
    assert abs(float(out.min())) < 1e-6
    assert abs(float(out.max()) - 1.0) < 1e-6


def test_min_max_constant_map_no_nan():
    out = min_max_normalize(np.zeros((3, 3)))
    assert np.all(np.isfinite(out))


def test_heatmap_base64_is_valid_png():
    amap = np.linspace(0, 1, 16).reshape(4, 4)
    encoded = heatmap_to_base64_png(amap)
    assert isinstance(encoded, str) and len(encoded) > 0
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert image.size == (4, 4)
    assert image.mode == "RGB"
