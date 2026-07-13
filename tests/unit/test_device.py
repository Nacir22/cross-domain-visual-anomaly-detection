"""Tests de la sélection du device."""

from __future__ import annotations

import pytest

from anomaly_detection.utils.device import resolve_device


def test_cpu_is_always_available():
    assert resolve_device("cpu").type == "cpu"


def test_auto_returns_valid_device():
    assert resolve_device("auto").type in {"cpu", "cuda"}


def test_unknown_preference_raises():
    with pytest.raises(ValueError):
        resolve_device("quantum")
