"""Cross-Domain Visual Anomaly Detection.

Un pipeline unique de détection d'anomalies visuelles *normal-only*
(l'entraînement n'utilise que des images normales) appliqué à trois
domaines : industrie, santé et imagerie aérienne.

Ce package regroupe le code métier réutilisable. Les notebooks et les
scripts ne font que l'orchestrer ; ils n'en dupliquent jamais la logique.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
