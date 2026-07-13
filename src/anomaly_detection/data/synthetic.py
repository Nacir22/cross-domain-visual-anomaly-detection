"""Génération d'un mini-dataset synthétique au format MVTec AD.

À quoi ça sert ?
----------------
Le vrai MVTec AD pèse plusieurs gigaoctets et nécessite un téléchargement
manuel (licence non commerciale). Or, pour **tester** le code et faire tourner
la **CI**, on n'a pas besoin des vraies images : il suffit de quelques images
factices respectant EXACTEMENT la même arborescence de dossiers.

Ce module fabrique donc à la volée une catégorie synthétique :

    <root>/<category>/train/good/*.png
    <root>/<category>/test/good/*.png
    <root>/<category>/test/defect/*.png
    <root>/<category>/ground_truth/defect/*_mask.png

Ainsi, ``MVTecDataset`` et les tests fonctionnent sans télécharger 5 Go.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _save_random_image(path: Path, size: int, rng: np.random.Generator) -> None:
    """Écrit une image RGB de bruit aléatoire à l'emplacement donné."""
    array = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def _save_defect_pair(
    image_path: Path, mask_path: Path, size: int, rng: np.random.Generator
) -> None:
    """Écrit une image « défectueuse » et son masque binaire associé.

    Le défaut est un carré blanc inséré dans l'image ; le masque marque
    exactement ce carré (255 sur le défaut, 0 ailleurs).
    """
    array = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    # Carré de défaut dans le quart supérieur gauche.
    lo, hi = size // 4, size // 2
    array[lo:hi, lo:hi, :] = 255
    mask[lo:hi, lo:hi] = 255

    Image.fromarray(array, mode="RGB").save(image_path)
    Image.fromarray(mask, mode="L").save(mask_path)


def generate_synthetic_mvtec(
    root: str | Path,
    category: str = "synthetic",
    n_train_good: int = 8,
    n_test_good: int = 3,
    n_test_defect: int = 3,
    size: int = 32,
    seed: int = 42,
) -> Path:
    """Génère une catégorie synthétique au format MVTec AD.

    Args:
        root: Dossier racine où créer la catégorie.
        category: Nom de la catégorie synthétique.
        n_train_good: Nombre d'images normales d'entraînement.
        n_test_good: Nombre d'images normales de test.
        n_test_defect: Nombre d'images anormales de test (avec masques).
        size: Côté (en pixels) des images générées.
        seed: Graine du générateur aléatoire (reproductibilité).

    Returns:
        Le chemin du dossier de la catégorie créée.

    Example:
        >>> path = generate_synthetic_mvtec("/tmp/data", n_train_good=4)  # doctest: +SKIP
        >>> (path / "train" / "good").is_dir()  # doctest: +SKIP
        True
    """
    rng = np.random.default_rng(seed)
    category_dir = Path(root) / category

    train_good = category_dir / "train" / "good"
    test_good = category_dir / "test" / "good"
    test_defect = category_dir / "test" / "defect"
    gt_defect = category_dir / "ground_truth" / "defect"
    for directory in (train_good, test_good, test_defect, gt_defect):
        directory.mkdir(parents=True, exist_ok=True)

    for i in range(n_train_good):
        _save_random_image(train_good / f"{i:03d}.png", size, rng)
    for i in range(n_test_good):
        _save_random_image(test_good / f"{i:03d}.png", size, rng)
    for i in range(n_test_defect):
        _save_defect_pair(
            test_defect / f"{i:03d}.png",
            gt_defect / f"{i:03d}_mask.png",
            size,
            rng,
        )

    return category_dir
