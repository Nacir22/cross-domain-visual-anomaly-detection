"""Script CLI 'export_model' — squelette. Implémenté dans une phase ultérieure.

Usage:
    python scripts/export_model.py --help
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="export_model (à implémenter)")
    parser.add_argument(
        "--profile",
        default="local_cpu",
        choices=["local_cpu", "colab_gpu"],
        help="Profil matériel à utiliser.",
    )
    parser.parse_args()
    print("[export_model] Non encore implémenté (voir le plan de développement).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
