"""Sélection du matériel de calcul (CPU / GPU).

Idée pédagogique
-----------------
Un *device* est simplement l'endroit où PyTorch effectue les calculs :
le processeur (``cpu``) ou une carte graphique (``cuda``). Un GPU accélère
massivement l'entraînement, mais n'est pas toujours disponible.

Notre stratégie, décidée avec l'utilisateur :

- **Auto** : on prend le GPU s'il existe, sinon le CPU.
- **Forçage explicite** : la configuration peut imposer ``cpu`` ou ``cuda``.
  Forcer ``cuda`` sans GPU lève une erreur claire plutôt que de planter
  obscurément plus tard.

Le développement se fait en local sur CPU ; l'entraînement lourd se fait sur
Google Colab (GPU). Aucune installation CUDA locale n'est donc requise.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_device(preference: str = "auto") -> torch.device:
    """Résout le device à utiliser à partir d'une préférence.

    Args:
        preference: ``"auto"``, ``"cpu"`` ou ``"cuda"``.
            - ``"auto"`` : GPU si disponible, sinon CPU.
            - ``"cpu"`` : force le CPU.
            - ``"cuda"`` : force le GPU (erreur si indisponible).

    Returns:
        L'objet :class:`torch.device` correspondant.

    Raises:
        ValueError: Si ``preference`` n'est pas une valeur reconnue.
        RuntimeError: Si ``"cuda"`` est demandé alors qu'aucun GPU n'est
            disponible.

    Example:
        >>> dev = resolve_device("auto")
        >>> dev.type in {"cpu", "cuda"}
        True
    """
    normalized = preference.strip().lower()
    cuda_available = torch.cuda.is_available()

    if normalized == "auto":
        device = torch.device("cuda" if cuda_available else "cpu")
    elif normalized == "cpu":
        device = torch.device("cpu")
    elif normalized == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "device='cuda' demandé mais aucun GPU CUDA n'est disponible. "
                "Utilisez device='cpu' ou device='auto', ou lancez ce calcul "
                "sur Google Colab avec un runtime GPU."
            )
        device = torch.device("cuda")
    else:
        raise ValueError(
            f"Préférence de device inconnue : {preference!r}. "
            "Attendu : 'auto', 'cpu' ou 'cuda'."
        )

    logger.info("Device sélectionné : %s (préférence=%s)", device, preference)
    return device
