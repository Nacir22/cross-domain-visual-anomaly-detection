"""Transformations d'images (preprocessing différentiable pour les modèles).

Idée pédagogique
-----------------
Un réseau attend des images de taille fixe, sous forme de *tenseur* (un
tableau multidimensionnel de nombres) et souvent *normalisées*. Une
transformation est une suite d'opérations qui convertit une image PIL en un
tel tenseur.

Points importants pour la détection d'anomalies :

- **Déterminisme en évaluation.** On n'applique JAMAIS d'augmentation
  aléatoire (rotation, flip...) au moment de l'évaluation : le score doit être
  reproductible. Les augmentations éventuelles ne servent qu'à l'entraînement.
- **Normalisation ImageNet.** Le modèle avancé (PatchCore) réutilise un réseau
  pré-entraîné sur ImageNet ; on normalise donc avec les mêmes statistiques.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchvision import transforms

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import torch
    from PIL.Image import Image

# Statistiques de normalisation d'ImageNet (moyenne / écart-type par canal RGB).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(
    image_size: int = 224,
    *,
    train: bool = False,
    normalize: bool = True,
) -> Callable[[Image], torch.Tensor]:
    """Construit le pipeline de transformation d'une image.

    Args:
        image_size: Côté (en pixels) de l'image carrée en sortie.
        train: Si ``True``, autorise de légères augmentations. Par prudence,
            on garde ce pipeline volontairement minimal ; en détection
            d'anomalies *normal-only*, trop d'augmentation peut brouiller la
            notion de « normal ». Par défaut ``False`` (déterministe).
        normalize: Si ``True``, applique la normalisation ImageNet. À désactiver
            si un modèle attend des pixels dans ``[0, 1]``.

    Returns:
        Une fonction transformant une image PIL en tenseur PyTorch de forme
        ``(3, image_size, image_size)``.

    Example:
        >>> tf = build_transforms(224)
        >>> # tensor = tf(pil_image)  # forme (3, 224, 224)
    """
    steps: list[object] = [
        transforms.Resize((image_size, image_size)),
    ]
    if train:
        # Augmentation minimale et sûre (pas de rotation forte ni de couleur).
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    steps.append(transforms.ToTensor())  # -> tenseur float dans [0, 1]
    if normalize:
        steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(steps)


def build_mask_transform(image_size: int = 224) -> Callable[[Image], torch.Tensor]:
    """Construit la transformation d'un masque de vérité terrain (pixel-level).

    Un masque indique, pixel par pixel, où se trouve l'anomalie. On le
    redimensionne SANS interpolation lissante (``NEAREST``) pour conserver des
    valeurs binaires nettes, puis on le convertit en tenseur.

    Args:
        image_size: Côté (en pixels) du masque carré en sortie.

    Returns:
        Une fonction transformant un masque PIL en tenseur de forme
        ``(1, image_size, image_size)`` à valeurs dans ``{0, 1}``.
    """
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )
