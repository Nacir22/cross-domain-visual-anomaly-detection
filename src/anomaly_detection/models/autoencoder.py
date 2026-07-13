"""Baseline : autoencodeur convolutif pour la détection d'anomalies.

Intuition
---------
Un **autoencodeur** apprend à *compresser* une image en une petite
représentation (l'**espace latent**), puis à la *reconstruire*. On l'entraîne
UNIQUEMENT sur des images normales.

- L'**encodeur** réduit progressivement l'image en une pile de caractéristiques.
- L'**espace latent** est ce goulot d'étranglement compressé.
- Le **décodeur** reconstruit l'image à partir du latent.

Comme le modèle n'a vu que du normal, il reconstruit bien le normal mais mal
les anomalies. L'**erreur de reconstruction** (différence entre l'image et sa
reconstruction) sert donc de signal d'anomalie : faible sur le normal, élevée
là où quelque chose d'inhabituel apparaît.

Limite honnête : un autoencodeur trop puissant peut « trop bien » généraliser
et reconstruire aussi certaines anomalies, ce qui baisse leur score. C'est
précisément pourquoi on le comparera à PatchCore (Phase 4).
"""

from __future__ import annotations

import torch
from torch import nn

from anomaly_detection.models.base import AnomalyModel


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Bloc encodeur : convolution qui divise la taille spatiale par 2."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _deconv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Bloc décodeur : convolution transposée qui double la taille spatiale."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ConvAutoencoder(AnomalyModel):
    """Autoencodeur convolutif entièrement convolutionnel.

    L'architecture est *fully convolutional* : elle fonctionne pour toute taille
    d'image divisible par ``2**4 = 16`` (ex. 224). Le goulot d'étranglement est
    une carte de caractéristiques à ``latent_dim`` canaux.

    Args:
        in_channels: Nombre de canaux d'entrée (3 pour RGB).
        base_channels: Nombre de canaux de la première couche (double ensuite).
        latent_dim: Nombre de canaux au goulot d'étranglement (espace latent).

    Example:
        >>> import torch
        >>> model = ConvAutoencoder()
        >>> x = torch.randn(2, 3, 64, 64)
        >>> model(x).shape
        torch.Size([2, 3, 64, 64])
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        # Encodeur : (3,H,W) -> (latent_dim, H/16, W/16)
        self.encoder = nn.Sequential(
            _conv_block(in_channels, c1),
            _conv_block(c1, c2),
            _conv_block(c2, c3),
            _conv_block(c3, latent_dim),
        )
        # Décodeur : (latent_dim, H/16, W/16) -> (3, H, W)
        self.decoder = nn.Sequential(
            _deconv_block(latent_dim, c3),
            _deconv_block(c3, c2),
            _deconv_block(c2, c1),
            nn.ConvTranspose2d(c1, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Reconstruit un batch d'images.

        Args:
            images: Batch ``(B, 3, H, W)``.

        Returns:
            Reconstruction ``(B, 3, H, W)`` (même forme que l'entrée).
        """
        latent = self.encoder(images)
        return self.decoder(latent)

    def anomaly_map(self, images: torch.Tensor) -> torch.Tensor:
        """Carte d'anomalie = erreur de reconstruction par pixel.

        On calcule l'erreur quadratique entre l'image et sa reconstruction, puis
        on moyenne sur les canaux couleur pour obtenir une carte à un canal.

        Args:
            images: Batch ``(B, 3, H, W)``.

        Returns:
            Carte ``(B, 1, H, W)`` de valeurs >= 0.
        """
        reconstruction = self.forward(images)
        squared_error = (images - reconstruction) ** 2
        return squared_error.mean(dim=1, keepdim=True)


def build_autoencoder(config: dict | None = None) -> ConvAutoencoder:
    """Construit un :class:`ConvAutoencoder` à partir d'un dictionnaire de config.

    Args:
        config: Dictionnaire pouvant contenir ``base_channels`` et ``latent_dim``
            (par exemple le champ ``extra`` d'une :class:`RunConfig`). Les clés
            absentes prennent leur valeur par défaut.

    Returns:
        Une instance de :class:`ConvAutoencoder`.
    """
    config = config or {}
    return ConvAutoencoder(
        in_channels=int(config.get("in_channels", 3)),
        base_channels=int(config.get("base_channels", 32)),
        latent_dim=int(config.get("latent_dim", 128)),
    )
