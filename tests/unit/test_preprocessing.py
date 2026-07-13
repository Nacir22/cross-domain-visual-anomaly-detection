"""Tests du chargement et de la validation d'images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anomaly_detection.data.preprocessing import (
    InvalidImageError,
    is_valid_image,
    load_image,
    validate_image_file,
)


def _make_png(path: Path, size: int = 16) -> Path:
    array = np.zeros((size, size, 3), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)
    return path


def test_load_image_returns_rgb(tmp_path: Path):
    path = _make_png(tmp_path / "img.png")
    image = load_image(path)
    assert image.mode == "RGB"
    assert image.size == (16, 16)


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(InvalidImageError):
        load_image(tmp_path / "absent.png")


def test_corrupted_file_is_detected(tmp_path: Path):
    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"ceci n'est pas une image")
    assert is_valid_image(bad) is False
    with pytest.raises(InvalidImageError):
        load_image(bad)


def test_unsupported_extension_rejected(tmp_path: Path):
    txt = tmp_path / "note.txt"
    txt.write_text("hello")
    with pytest.raises(InvalidImageError):
        validate_image_file(txt)


def test_valid_image_passes_validation(tmp_path: Path):
    path = _make_png(tmp_path / "ok.png")
    assert validate_image_file(path) == path
    assert is_valid_image(path) is True
