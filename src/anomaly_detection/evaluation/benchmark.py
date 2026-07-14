"""Mesure du coût des modèles : temps d'inférence et empreinte mémoire.

Comparer deux modèles ne se limite pas à la qualité de détection : il faut
aussi savoir combien ils coûtent. On mesure ici le temps d'inférence moyen par
image et une estimation de l'empreinte mémoire.
"""

from __future__ import annotations

import time

import torch


def count_parameters(model: torch.nn.Module) -> int:
    """Compte le nombre total de paramètres d'un modèle.

    Args:
        model: Le modèle à inspecter.

    Returns:
        Le nombre de paramètres.
    """
    return sum(p.numel() for p in model.parameters())


def memory_bank_footprint_mb(model: torch.nn.Module) -> float:
    """Estime en Mo la taille de la banque mémoire (spécifique PatchCore).

    Args:
        model: Modèle éventuellement doté d'un attribut ``memory_bank``.

    Returns:
        La taille en mégaoctets, ``0.0`` si pas de banque mémoire.
    """
    bank = getattr(model, "memory_bank", None)
    if bank is None:
        return 0.0
    return float(bank.numel() * bank.element_size()) / (1024 * 1024)


@torch.no_grad()
def measure_inference_time(
    model: torch.nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    n_iters: int = 20,
    warmup: int = 3,
) -> float:
    """Mesure le temps d'inférence moyen par image (millisecondes).

    Args:
        model: Modèle exposant ``anomaly_score``.
        sample: Batch d'images ``(B, 3, H, W)`` servant de charge de test.
        device: CPU ou GPU.
        n_iters: Nombre d'itérations chronométrées.
        warmup: Itérations de chauffe non comptées (caches, allocations).

    Returns:
        Temps moyen par image en millisecondes.
    """
    model.eval()
    sample = sample.to(device)
    batch_size = max(sample.shape[0], 1)

    for _ in range(warmup):
        model.anomaly_score(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        model.anomaly_score(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / (n_iters * batch_size)) * 1000.0
