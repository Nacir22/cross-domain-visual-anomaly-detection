"""Reproductibilité des expériences.

Pourquoi ?
----------
Un réseau de neurones s'initialise avec des nombres *aléatoires*. Sans
contrôle, deux exécutions donnent des résultats différents et rien n'est
comparable. On fixe donc une *graine* (seed) pour Python, NumPy et PyTorch.

Limite honnête : certaines opérations GPU restent non parfaitement
déterministes. On peut réduire ce phénomène mais rarement l'annuler à coût
raisonnable ; on le documente au lieu de le cacher.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

from anomaly_detection.constants import DEFAULT_SEED

logger = logging.getLogger(__name__)


def set_seed(seed: int = DEFAULT_SEED, *, deterministic: bool = False) -> None:
    """Fixe les graines aléatoires pour rendre une exécution reproductible.

    Args:
        seed: Valeur de la graine appliquée à Python, NumPy et PyTorch.
        deterministic: Si ``True``, tente de forcer des algorithmes
            déterministes sur GPU (plus lent, et pas toujours possible).

    Example:
        >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # Peut ralentir et lever une erreur si un op n'a pas de version
        # déterministe ; on reste tolérant pour ne pas casser le pipeline.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:  # pragma: no cover - dépend du backend
            logger.warning("Mode déterministe partiel uniquement : %s", exc)

    logger.info("Graine fixée à %d (deterministic=%s)", seed, deterministic)
