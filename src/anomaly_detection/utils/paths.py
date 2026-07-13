"""Localisation reproductible des dossiers du projet.

Pourquoi ce module ?
--------------------
Coder des chemins absolus « en dur » (par ex. ``C:/Users/moi/...``) rend un
projet impossible à exécuter sur une autre machine. On calcule donc tous les
chemins **relativement à la racine du dépôt**, déterminée une seule fois ici.

La racine est trouvée en remontant l'arborescence depuis ce fichier jusqu'à
rencontrer un marqueur de projet (``pyproject.toml``).
"""

from __future__ import annotations

from pathlib import Path

# Marqueur qui identifie la racine du dépôt.
_PROJECT_MARKER = "pyproject.toml"


def get_project_root(start: Path | None = None) -> Path:
    """Retourne la racine du dépôt.

    Args:
        start: Dossier de départ pour la recherche. Par défaut, le dossier
            contenant ce fichier.

    Returns:
        Le chemin absolu de la racine du projet (dossier contenant
        ``pyproject.toml``).

    Raises:
        FileNotFoundError: Si aucun ``pyproject.toml`` n'est trouvé en
            remontant l'arborescence.

    Example:
        >>> root = get_project_root()
        >>> (root / "pyproject.toml").exists()
        True
    """
    current = (start or Path(__file__)).resolve()
    for parent in [current, *current.parents]:
        if (parent / _PROJECT_MARKER).is_file():
            return parent
    raise FileNotFoundError(
        f"Impossible de trouver '{_PROJECT_MARKER}' en remontant depuis {current}."
    )


# Chemins usuels, calculés une fois à l'import.
PROJECT_ROOT: Path = get_project_root()
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
MLRUNS_DIR: Path = PROJECT_ROOT / "mlruns"
