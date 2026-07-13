"""Chargement et validation de la configuration en cascade.

Principe
--------
Le comportement du pipeline est piloté par des fichiers YAML, pas par du code
codé en dur. La configuration finale est construite en empilant, dans l'ordre,
quatre couches (la dernière l'emporte en cas de conflit) :

    base.yaml
      -> domains/<domaine>.yaml
        -> models/<modele>.yaml
          -> profiles/<profil>.yaml   (local_cpu ou colab_gpu)

Ainsi, passer du CPU local au GPU Colab ne change qu'une couche (le profil),
jamais le code. La configuration résolue est ensuite *validée* par Pydantic :
une valeur manquante ou d'un mauvais type échoue tôt, avec un message clair.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from anomaly_detection.constants import (
    SUPPORTED_DOMAINS,
    SUPPORTED_MODELS,
)
from anomaly_detection.utils.paths import CONFIGS_DIR


class RunConfig(BaseModel):
    """Configuration validée d'une exécution du pipeline.

    Attributes:
        domain: Domaine étudié (``industrial``, ``medical`` ou ``aerial``).
        model: Modèle utilisé (``autoencoder`` ou ``patchcore``).
        profile: Profil matériel (``local_cpu`` ou ``colab_gpu``).
        device: Préférence de device (``auto``, ``cpu`` ou ``cuda``).
        image_size: Côté (en pixels) des images carrées après redimension.
        batch_size: Nombre d'images traitées ensemble.
        num_workers: Processus de chargement des données (0 = synchrone).
        epochs: Nombre de passages complets sur les données d'entraînement.
        learning_rate: Pas d'apprentissage de l'optimiseur.
        seed: Graine aléatoire pour la reproductibilité.
        mixed_precision: Active le calcul en demi-précision (GPU uniquement).
        extra: Toute clé supplémentaire spécifique au domaine ou au modèle.
    """

    domain: str
    model: str
    profile: str = "local_cpu"
    device: str = "auto"
    image_size: int = Field(default=224, gt=0)
    batch_size: int = Field(default=2, gt=0)
    num_workers: int = Field(default=0, ge=0)
    epochs: int = Field(default=2, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    seed: int = 42
    mixed_precision: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("domain")
    @classmethod
    def _check_domain(cls, value: str) -> str:
        if value not in SUPPORTED_DOMAINS:
            raise ValueError(
                f"Domaine inconnu : {value!r}. Attendu : {SUPPORTED_DOMAINS}."
            )
        return value

    @field_validator("model")
    @classmethod
    def _check_model(cls, value: str) -> str:
        if value not in SUPPORTED_MODELS:
            raise ValueError(
                f"Modèle inconnu : {value!r}. Attendu : {SUPPORTED_MODELS}."
            )
        return value

    @field_validator("device")
    @classmethod
    def _check_device(cls, value: str) -> str:
        allowed = {"auto", "cpu", "cuda"}
        if value not in allowed:
            raise ValueError(f"device doit être dans {allowed}, reçu {value!r}.")
        return value


def _read_yaml(path: Path) -> dict[str, Any]:
    """Lit un fichier YAML et retourne un dictionnaire (vide si fichier vide).

    Args:
        path: Chemin du fichier YAML.

    Returns:
        Le contenu du fichier sous forme de dictionnaire.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        TypeError: Si le YAML ne représente pas un dictionnaire.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {path}")
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    if not isinstance(content, dict):
        raise TypeError(f"Le YAML {path} doit décrire un dictionnaire.")
    return content


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Fusionne récursivement ``override`` dans ``base`` sans les modifier.

    Les sous-dictionnaires sont fusionnés clé par clé ; toute autre valeur de
    ``override`` remplace celle de ``base``.

    Args:
        base: Dictionnaire de base (couche inférieure).
        override: Dictionnaire prioritaire (couche supérieure).

    Returns:
        Un nouveau dictionnaire résultant de la fusion.

    Example:
        >>> deep_merge({"a": 1, "b": {"x": 1}}, {"b": {"y": 2}})
        {'a': 1, 'b': {'x': 1, 'y': 2}}
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    domain: str,
    model: str,
    profile: str = "local_cpu",
    configs_dir: Path = CONFIGS_DIR,
) -> RunConfig:
    """Construit la configuration résolue et validée d'une exécution.

    Empile ``base`` -> domaine -> modèle -> profil, puis valide le résultat.

    Args:
        domain: Nom du domaine (fichier ``configs/domains/<domain>.yaml``).
        model: Nom du modèle (fichier ``configs/models/<model>.yaml``).
        profile: Nom du profil (fichier ``configs/profiles/<profile>.yaml``).
        configs_dir: Dossier racine des configurations.

    Returns:
        Une instance :class:`RunConfig` validée.

    Raises:
        FileNotFoundError: Si un fichier de couche est manquant.
        pydantic.ValidationError: Si la configuration finale est invalide.

    Example:
        >>> cfg = load_config("industrial", "autoencoder", "local_cpu")
        >>> cfg.domain
        'industrial'
    """
    merged = _read_yaml(configs_dir / "base.yaml")
    merged = deep_merge(merged, _read_yaml(configs_dir / "domains" / f"{domain}.yaml"))
    merged = deep_merge(merged, _read_yaml(configs_dir / "models" / f"{model}.yaml"))
    merged = deep_merge(
        merged, _read_yaml(configs_dir / "profiles" / f"{profile}.yaml")
    )

    # On garantit la cohérence des identifiants demandés.
    merged.setdefault("domain", domain)
    merged.setdefault("model", model)
    merged.setdefault("profile", profile)

    # Les clés non reconnues par le schéma sont regroupées dans ``extra``.
    known = set(RunConfig.model_fields)
    extra = merged.pop("extra", {})
    for key in list(merged):
        if key not in known:
            extra[key] = merged.pop(key)
    merged["extra"] = extra

    return RunConfig(**merged)
