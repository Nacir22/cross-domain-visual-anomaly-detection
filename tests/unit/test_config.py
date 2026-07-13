"""Tests de la configuration en cascade."""

from __future__ import annotations

import pytest

from anomaly_detection.config import RunConfig, deep_merge, load_config


def test_deep_merge_is_recursive_and_pure():
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    override = {"nested": {"y": 20, "z": 30}}
    result = deep_merge(base, override)
    assert result == {"a": 1, "nested": {"x": 1, "y": 20, "z": 30}}
    # L'original ne doit pas être modifié (fonction pure).
    assert base == {"a": 1, "nested": {"x": 1, "y": 2}}


def test_local_cpu_profile_forces_cpu():
    cfg = load_config("industrial", "autoencoder", "local_cpu")
    assert cfg.device == "cpu"
    assert cfg.batch_size == 2
    assert cfg.profile == "local_cpu"


def test_colab_profile_uses_gpu_settings():
    cfg = load_config("industrial", "patchcore", "colab_gpu")
    assert cfg.device == "cuda"
    assert cfg.mixed_precision is True
    assert cfg.batch_size == 16


def test_domain_specific_keys_go_to_extra():
    cfg = load_config("industrial", "autoencoder", "local_cpu")
    # 'category' et 'dataset_name' ne sont pas des champs du schéma.
    assert cfg.extra["dataset_name"] == "mvtec_ad"
    assert "category" in cfg.extra


def test_invalid_domain_is_rejected():
    with pytest.raises(ValueError):
        RunConfig(domain="banana", model="autoencoder")


def test_all_domains_load():
    for domain in ("industrial", "medical", "aerial"):
        cfg = load_config(domain, "autoencoder", "local_cpu")
        assert cfg.domain == domain
