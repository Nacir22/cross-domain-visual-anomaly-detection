.PHONY: help install dev format lint test smoke report clean

help:
	@echo "Cibles disponibles :"
	@echo "  install  Installe le package (mode production)"
	@echo "  dev      Installe le package + outils de dev + hooks"
	@echo "  format   Formate le code avec Ruff"
	@echo "  lint     Vérifie le style avec Ruff"
	@echo "  test     Lance la suite de tests"
	@echo "  smoke    Lance uniquement les tests 'smoke'"
	@echo "  report   Compile le rapport LaTeX"
	@echo "  clean    Supprime les fichiers temporaires"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

format:
	ruff format .

lint:
	ruff check .

test:
	pytest

smoke:
	pytest -m smoke

report:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=docs docs/rapport.tex

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info

.PHONY: api demo cross-domain
api:
	uvicorn app.api.main:app --reload

demo:
	streamlit run app/demo/streamlit_app.py

cross-domain:
	python scripts/run_cross_domain.py --epochs 2 --image-size 64
