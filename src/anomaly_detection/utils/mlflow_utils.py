"""Suivi d'expériences avec MLflow (dégradation gracieuse).

Qu'est-ce que MLflow ?
----------------------
MLflow est un carnet de laboratoire pour le ML : il enregistre, pour chaque
expérience, les *paramètres* (modèle, taille d'image, graine...), les
*métriques* (AUROC, F1...) et les *artefacts* (courbes, checkpoints). On peut
ensuite comparer les runs dans une interface web (``mlflow ui``).

Ce module encapsule MLflow derrière un gestionnaire de contexte qui NE plante
pas si MLflow n'est pas installé : le code d'entraînement reste exécutable
partout (tests, CI), et le suivi s'active quand MLflow est présent.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


@contextmanager
def mlflow_run(
    run_name: str,
    experiment: str = "anomaly-detection",
    tags: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Ouvre un run MLflow, ou un contexte neutre si MLflow est absent.

    Args:
        run_name: Nom lisible du run (ex. ``"industrial-patchcore-bottle"``).
        experiment: Nom de l'expérience regroupant les runs.
        tags: Tags à attacher au run (domaine, catégorie, modèle...).

    Yields:
        Le module ``mlflow`` prêt à l'emploi, ou ``None`` si indisponible.

    Example:
        >>> with mlflow_run("demo") as ml:  # doctest: +SKIP
        ...     if ml:
        ...         ml.log_param("seed", 42)
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow non installé : suivi d'expérience désactivé.")
        yield None
        return

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags(tags)
        yield mlflow
