"""Fixtures partagées par la suite de tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from anomaly_detection.utils.paths import CONFIGS_DIR


@pytest.fixture
def configs_dir() -> Path:
    """Retourne le dossier des configurations réelles du dépôt."""
    return CONFIGS_DIR


@pytest.fixture
def synthetic_mvtec(tmp_path: Path) -> Path:
    """Crée une catégorie MVTec synthétique dans un dossier temporaire.

    Permet de tester tout le pipeline de données sans télécharger les 5 Go du
    vrai MVTec AD. Le dossier est automatiquement supprimé après le test.

    Returns:
        La racine (``tmp_path``) contenant la catégorie ``synthetic``.
    """
    from anomaly_detection.data.synthetic import generate_synthetic_mvtec

    generate_synthetic_mvtec(
        tmp_path,
        category="synthetic",
        n_train_good=8,
        n_test_good=3,
        n_test_defect=3,
        size=32,
        seed=0,
    )
    return tmp_path
