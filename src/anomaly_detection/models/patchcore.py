"""Modèle avancé : PatchCore (mémoire de features de patchs).

Intuition
---------
Contrairement à l'autoencodeur, PatchCore n'apprend RIEN par descente de
gradient. Il réutilise un réseau **pré-entraîné** (sur ImageNet) comme
extracteur de caractéristiques, puis :

1. **Construction de la mémoire** : sur les images NORMALES, il extrait des
   vecteurs de caractéristiques pour chaque petite région (« patch ») et les
   stocke dans une *banque mémoire*.
2. **Coreset** : cette banque est énorme ; on n'en garde qu'un sous-échantillon
   représentatif (``coreset_ratio``) pour rester rapide et léger.
3. **Détection** : pour une nouvelle image, chaque patch est comparé à son plus
   proche voisin dans la mémoire. Un patch loin de tout ce qui est « normal »
   est suspect. La distance devient la valeur d'anomalie du patch.

Avantages : cartes d'anomalie nettes, pas d'entraînement long, très bon sur
MVTec. Coûts : la banque mémoire occupe de la RAM et l'inférence fait une
recherche de plus proche voisin. On compare tout cela honnêtement à
l'autoencodeur (Phase 4).

Note sur le preprocessing : PatchCore ET l'autoencodeur reçoivent des images
normalisées avec les mêmes statistiques ImageNet (voir ``transforms.py``), ce
qui rend la comparaison équitable.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor

from anomaly_detection.models.base import AnomalyModel

logger = logging.getLogger(__name__)

_BACKBONES = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "wide_resnet50_2": wide_resnet50_2,
}


def _build_feature_extractor(
    backbone: str, layers: tuple[str, ...], pretrained: bool
) -> nn.Module:
    """Construit un extracteur figé renvoyant les sorties de couches choisies.

    Args:
        backbone: Nom du backbone (``resnet18``, ``resnet50``, ``wide_resnet50_2``).
        layers: Noms des couches à extraire (ex. ``("layer2", "layer3")``).
        pretrained: Si ``True``, charge les poids ImageNet (téléchargement).

    Returns:
        Un module qui, appelé sur un batch, renvoie un dict {couche: features}.

    Raises:
        ValueError: Si le backbone est inconnu.
    """
    if backbone not in _BACKBONES:
        raise ValueError(
            f"Backbone inconnu : {backbone!r}. Attendu : {tuple(_BACKBONES)}."
        )
    weights = "DEFAULT" if pretrained else None
    net = _BACKBONES[backbone](weights=weights)
    extractor = create_feature_extractor(net, return_nodes={ly: ly for ly in layers})
    for param in extractor.parameters():
        param.requires_grad = False
    extractor.eval()
    return extractor


class PatchCore(AnomalyModel):
    """Détecteur PatchCore.

    Args:
        backbone: Nom du backbone pré-entraîné.
        layers: Couches intermédiaires utilisées comme caractéristiques.
        pretrained: Charge les poids ImageNet (mettre ``False`` pour les tests
            hors ligne).
        coreset_ratio: Fraction de la banque mémoire conservée (0.01 = 1 %).
        n_neighbors: Nombre de voisins (réservé ; la carte utilise le 1er voisin).
        image_size: Taille des images (pour redimensionner la carte).
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: tuple[str, ...] = ("layer2", "layer3"),
        pretrained: bool = True,
        coreset_ratio: float = 0.01,
        n_neighbors: int = 9,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.layers = tuple(layers)
        self.coreset_ratio = float(coreset_ratio)
        self.n_neighbors = int(n_neighbors)
        self.image_size = int(image_size)
        self.feature_extractor = _build_feature_extractor(
            backbone, self.layers, pretrained
        )
        # La banque mémoire est remplie par fit(). Attribut simple (pas un
        # paramètre entraînable) : on la sauvegarde/recharge explicitement.
        self.memory_bank: torch.Tensor | None = None

    @torch.no_grad()
    def _extract_patches(self, images: torch.Tensor) -> torch.Tensor:
        """Extrait et agrège les caractéristiques de patchs.

        Args:
            images: Batch ``(B, 3, H, W)`` normalisé ImageNet.

        Returns:
            Caractéristiques ``(B, C, h, w)`` (h, w = taille de la carte de
            features de la première couche).
        """
        feats = self.feature_extractor(images)
        reference = feats[self.layers[0]]
        target_hw = reference.shape[-2:]

        aligned = []
        for layer in self.layers:
            f = feats[layer]
            if f.shape[-2:] != target_hw:
                f = F.interpolate(
                    f, size=target_hw, mode="bilinear", align_corners=False
                )
            aligned.append(f)
        x = torch.cat(aligned, dim=1)
        # Agrégation locale (voisinage 3x3) : chaque patch résume son entourage.
        return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    @torch.no_grad()
    def fit(self, loader: DataLoader, device: torch.device) -> PatchCore:
        """Construit la banque mémoire à partir d'images normales.

        Args:
            loader: DataLoader d'images NORMALES (jeu d'entraînement one-class).
            device: CPU ou GPU.

        Returns:
            ``self`` (pour chaîner les appels).
        """
        self.feature_extractor.to(device).eval()
        collected: list[torch.Tensor] = []
        for batch in loader:
            x = self._extract_patches(batch["image"].to(device))
            b, c, h, w = x.shape
            patches = x.permute(0, 2, 3, 1).reshape(-1, c)  # (B*h*w, C)
            collected.append(patches.cpu())

        bank = torch.cat(collected, dim=0)
        # Coreset : sous-échantillonnage aléatoire (simplification CPU-friendly du
        # coreset glouton d'origine). Documenté comme tel dans le rapport.
        keep = max(1, int(bank.shape[0] * self.coreset_ratio))
        idx = torch.randperm(bank.shape[0])[:keep]
        self.memory_bank = bank[idx].contiguous()
        logger.info(
            "Banque mémoire : %d patchs -> %d après coreset (%.1f%%)",
            bank.shape[0],
            self.memory_bank.shape[0],
            100 * self.coreset_ratio,
        )
        return self

    @torch.no_grad()
    def anomaly_map(self, images: torch.Tensor) -> torch.Tensor:
        """Carte d'anomalie = distance au plus proche patch normal.

        Args:
            images: Batch ``(B, 3, H, W)``.

        Returns:
            Carte ``(B, 1, image_size, image_size)``, valeurs >= 0.

        Raises:
            RuntimeError: Si la banque mémoire n'a pas été construite (fit).
        """
        if self.memory_bank is None:
            raise RuntimeError("Banque mémoire vide : appelez d'abord fit().")

        device = images.device
        bank = self.memory_bank.to(device)
        x = self._extract_patches(images)
        b, c, h, w = x.shape
        patches = x.permute(0, 2, 3, 1).reshape(-1, c)  # (B*h*w, C)

        # Distance au plus proche voisin, par morceaux (mémoire maîtrisée).
        min_dists = []
        for chunk in patches.split(4096):
            d = torch.cdist(chunk, bank)  # (chunk, M)
            min_dists.append(d.min(dim=1).values)
        distances = torch.cat(min_dists).reshape(b, 1, h, w)

        return F.interpolate(
            distances,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

    @torch.no_grad()
    def anomaly_score(self, images: torch.Tensor) -> torch.Tensor:
        """Score = distance maximale sur la carte (patch le plus anormal).

        Args:
            images: Batch ``(B, 3, H, W)``.

        Returns:
            Scores ``(B,)``.
        """
        maps = self.anomaly_map(images)
        return maps.flatten(start_dim=1).max(dim=1).values


def build_patchcore(config: dict | None = None) -> PatchCore:
    """Construit un :class:`PatchCore` depuis un dictionnaire de configuration.

    Args:
        config: Peut contenir ``backbone``, ``coreset_ratio``, ``n_neighbors``,
            ``pretrained``, ``image_size``. Les clés absentes prennent leur
            valeur par défaut.

    Returns:
        Une instance de :class:`PatchCore`.
    """
    config = config or {}
    return PatchCore(
        backbone=str(config.get("backbone", "resnet18")),
        pretrained=bool(config.get("pretrained", True)),
        coreset_ratio=float(config.get("coreset_ratio", 0.01)),
        n_neighbors=int(config.get("n_neighbors", 9)),
        image_size=int(config.get("image_size", 224)),
    )
