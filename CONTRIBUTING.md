# Contribuer

Merci de votre intérêt pour ce projet.

## Environnement de développement

```bash
python -m venv .venv
# Windows PowerShell : .venv\Scripts\Activate.ps1
# Linux/macOS       : source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Qualité du code

Avant chaque commit :

```bash
ruff format .      # formatage
ruff check .       # lint + tri des imports
pytest             # tests
```

`pre-commit` exécute automatiquement ces vérifications.

## Style

- Type hints obligatoires sur les fonctions publiques.
- Docstrings au format Google (rôle, arguments, retour, exceptions, exemple).
- Pas de chemins absolus codés en dur : utiliser `anomaly_detection.utils.paths`.
- Aucun secret ni donnée volumineuse dans le dépôt.

## Messages de commit

Convention *Conventional Commits* : `feat`, `fix`, `docs`, `test`, `refactor`,
`chore`, `ci`, `build`, `perf`.
