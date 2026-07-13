"""Boucle d'entraînement de l'autoencodeur (cadre *normal-only*).

Vocabulaire
-----------
- **Époque (epoch)** : un passage complet sur toutes les images d'entraînement.
- **Batch** : un petit paquet d'images traité d'un coup (ex. 2 sur CPU, 16 sur GPU).
- **Optimiseur** : l'algorithme (ici Adam) qui ajuste les poids pour réduire la perte.
- **Validation** : on mesure la perte sur des images normales NON vues à
  l'entraînement, pour repérer le sur-apprentissage et choisir le meilleur modèle.

On n'entraîne QUE sur des images normales : le modèle apprend le « normal ».
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from anomaly_detection.training.callbacks import CheckpointSaver, LossHistory
from anomaly_detection.training.losses import reconstruction_loss
from anomaly_detection.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Entraîne le modèle sur une époque et retourne la perte moyenne.

    Args:
        model: Autoencodeur à entraîner.
        loader: DataLoader d'images normales d'entraînement.
        criterion: Fonction de perte (reconstruction).
        optimizer: Optimiseur mettant à jour les poids.
        device: CPU ou GPU.

    Returns:
        La perte d'entraînement moyenne sur l'époque.
    """
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device)
        optimizer.zero_grad()
        reconstruction = model(images)
        loss = criterion(reconstruction, images)
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
        n += images.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Calcule la perte de reconstruction moyenne sur un jeu (sans gradient).

    Args:
        model: Modèle évalué.
        loader: DataLoader (normalement le jeu de validation, normal-only).
        criterion: Fonction de perte.
        device: CPU ou GPU.

    Returns:
        La perte moyenne. ``0.0`` si le loader est vide.
    """
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device)
        reconstruction = model(images)
        loss = criterion(reconstruction, images)
        total += loss.item() * images.size(0)
        n += images.size(0)
    return total / max(n, 1)


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    checkpoint_path: str | Path,
    seed: int = 42,
    extra: dict | None = None,
) -> LossHistory:
    """Entraîne l'autoencodeur et sauvegarde le meilleur checkpoint.

    Args:
        model: Autoencodeur (déjà déplacé sur ``device`` ou déplacé ici).
        train_loader: Images normales d'entraînement.
        val_loader: Images normales de validation.
        epochs: Nombre d'époques.
        learning_rate: Pas d'apprentissage d'Adam.
        device: CPU ou GPU.
        checkpoint_path: Où sauvegarder le meilleur modèle.
        seed: Graine de reproductibilité.
        extra: Métadonnées à stocker dans le checkpoint (config, nom du modèle).

    Returns:
        L'historique des pertes (train et validation par époque).
    """
    set_seed(seed)
    model = model.to(device)
    criterion = reconstruction_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = LossHistory()
    saver = CheckpointSaver(checkpoint_path, extra=extra)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        history.update(train_loss, val_loss)
        improved = saver.step(model, val_loss, epoch)
        logger.info(
            "Époque %d/%d | train=%.4f | val=%.4f%s",
            epoch,
            epochs,
            train_loss,
            val_loss,
            "  <- meilleur" if improved else "",
        )

    # Si aucune validation n'a permis de sauvegarder (val vide), on sauve la fin.
    if saver.best_epoch == -1:
        saver.step(model, val_loss=history.train[-1] if history.train else 0.0, epoch=epochs)

    return history


def save_loss_curve(history: LossHistory, path: str | Path) -> Path:
    """Sauvegarde la courbe des pertes (train/val) en image PNG.

    Args:
        history: Historique produit par :func:`train_autoencoder`.
        path: Chemin du fichier PNG de sortie.

    Returns:
        Le chemin de l'image écrite.
    """
    import matplotlib

    matplotlib.use("Agg")  # backend sans écran (serveur / CI)
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.train) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history.train, label="train", marker="o")
    if history.val:
        plt.plot(epochs, history.val, label="validation", marker="o")
    plt.xlabel("Époque")
    plt.ylabel("Perte de reconstruction (MSE)")
    plt.title("Courbe d'apprentissage — autoencodeur")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out
