"""Script CLI 'download_data' — squelette. Implémenté dans une phase ultérieure.

Usage:
    python scripts/download_data.py --help
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="download_data (à implémenter)")
    parser.add_argument("--profile", default="local_cpu",
                        choices=["local_cpu", "colab_gpu"],
                        help="Profil matériel à utiliser.")
    parser.parse_args()
    print("[download_data] Non encore implémenté (voir le plan de développement).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
