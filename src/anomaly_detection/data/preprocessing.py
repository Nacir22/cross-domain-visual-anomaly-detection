"""Chargement et validation des images.

Rôle pédagogique
-----------------
Avant d'entraîner quoi que ce soit, il faut être sûr que chaque fichier est
bien une image lisible. Un fichier corrompu, tronqué ou d'un mauvais format
doit être détecté **tôt** et signalé clairement, plutôt que de faire planter
l'entraînement des heures plus tard avec un message obscur.

Ce module fournit donc :

- ``load_image`` : ouvre une image et la convertit en RGB ;
- ``is_valid_image`` : dit si un fichier est une image lisible (sans lever) ;
- ``validate_image_file`` : lève une exception claire si le fichier est invalide.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from anomaly_detection.constants import SUPPORTED_IMAGE_EXTENSIONS


class InvalidImageError(ValueError):
    """Levée lorsqu'un fichier n'est pas une image exploitable."""


def load_image(path: str | Path) -> Image.Image:
    """Charge une image depuis le disque et la convertit en RGB.

    La conversion en RGB garantit un nombre de canaux constant (3), même pour
    des images en niveaux de gris (radiographies) ou avec canal alpha.

    Args:
        path: Chemin du fichier image.

    Returns:
        L'image chargée au format :class:`PIL.Image.Image` en mode ``RGB``.

    Raises:
        InvalidImageError: Si le fichier n'existe pas, n'est pas une image, ou
            est corrompu/tronqué.

    Example:
        >>> img = load_image("data/raw/mvtec_ad/bottle/train/good/000.png")
        >>> img.mode
        'RGB'
    """
    path = Path(path)
    if not path.is_file():
        raise InvalidImageError(f"Fichier introuvable : {path}")
    try:
        with Image.open(path) as image:
            # load() force la lecture complète : détecte les fichiers tronqués.
            image.load()
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(f"Image illisible ou corrompue : {path}") from exc


def is_valid_image(path: str | Path) -> bool:
    """Indique si un fichier est une image lisible, sans lever d'exception.

    Utile pour filtrer un dossier sans interrompre le programme.

    Args:
        path: Chemin du fichier à tester.

    Returns:
        ``True`` si le fichier est une image exploitable, ``False`` sinon.

    Example:
        >>> is_valid_image("notes.txt")
        False
    """
    try:
        validate_image_file(path)
    except InvalidImageError:
        return False
    return True


def validate_image_file(path: str | Path) -> Path:
    """Valide qu'un fichier est une image d'extension supportée et lisible.

    Args:
        path: Chemin du fichier à valider.

    Returns:
        Le chemin validé sous forme de :class:`pathlib.Path`.

    Raises:
        InvalidImageError: Si l'extension n'est pas supportée, si le fichier
            n'existe pas, ou si le contenu n'est pas une image valide.

    Example:
        >>> validate_image_file("image.png")  # doctest: +SKIP
        PosixPath('image.png')
    """
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise InvalidImageError(
            f"Extension non supportée : {path.suffix!r}. "
            f"Attendu : {SUPPORTED_IMAGE_EXTENSIONS}."
        )
    if not path.is_file():
        raise InvalidImageError(f"Fichier introuvable : {path}")
    try:
        # verify() valide l'en-tête sans décoder toute l'image (rapide).
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(f"Image illisible ou corrompue : {path}") from exc
    return path
