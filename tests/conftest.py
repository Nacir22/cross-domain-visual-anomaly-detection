"""Fixtures partagées par la suite de tests."""

from __future__ import annotations

import pytest

from anomaly_detection.utils.paths import CONFIGS_DIR


@pytest.fixture
def configs_dir():
    """Retourne le dossier des configurations réelles du dépôt."""
    return CONFIGS_DIR
