"""Configuration centralisée de la journalisation.

On utilise le module standard ``logging`` plutôt que des ``print`` : cela
permet de choisir le niveau de détail, d'horodater les messages et, plus
tard, de ne PAS journaliser de données sensibles (images médicales).
"""

from __future__ import annotations

import logging
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure la journalisation racine du projet.

    Idempotent : appeler plusieurs fois ne duplique pas les handlers.

    Args:
        level: Niveau minimal des messages affichés (``logging.INFO`` par
            défaut). Accepte un entier ou une chaîne comme ``"DEBUG"``.

    Example:
        >>> configure_logging("INFO")
        >>> logging.getLogger(__name__).info("prêt")
    """
    root = logging.getLogger()
    if root.handlers:
        # Déjà configuré : on ajuste seulement le niveau.
        root.setLevel(level)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
