"""Tests de la localisation des chemins du projet."""

from __future__ import annotations

from anomaly_detection.utils.paths import PROJECT_ROOT, get_project_root


def test_project_root_contains_pyproject():
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_get_project_root_is_idempotent():
    assert get_project_root() == PROJECT_ROOT
