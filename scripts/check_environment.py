"""Vérifie que l'environnement est correctement configuré.

Affiche la version de Python, la disponibilité des bibliothèques clés et du
GPU, puis charge une configuration d'exemple pour confirmer que la cascade
YAML fonctionne. À lancer en premier après l'installation.

Usage:
    python scripts/check_environment.py
"""

from __future__ import annotations

import importlib
import platform
import sys

# On rend le package importable même sans installation editable.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_REQUIRED = ["torch", "torchvision", "numpy", "yaml", "pydantic"]


def main() -> int:
    """Point d'entrée. Retourne 0 si tout est OK, 1 sinon."""
    print(f"Python      : {platform.python_version()} ({sys.executable})")
    ok = True

    for name in _REQUIRED:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "?")
            print(f"[ OK ] {name:<12} {version}")
        except ImportError:
            print(f"[FAIL] {name:<12} non installé")
            ok = False

    try:
        import torch

        print(f"CUDA dispo  : {torch.cuda.is_available()}")
    except ImportError:
        pass

    try:
        from anomaly_detection.config import load_config

        cfg = load_config("industrial", "autoencoder", "local_cpu")
        print(f"[ OK ] Config chargée : domaine={cfg.domain} device={cfg.device}")
    except Exception as exc:  # noqa: BLE001 - diagnostic volontairement large
        print(f"[FAIL] Chargement config : {exc}")
        ok = False

    print("\nRésultat :", "ENVIRONNEMENT PRÊT" if ok else "PROBLÈMES DÉTECTÉS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
