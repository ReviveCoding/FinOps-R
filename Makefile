PYTHON ?= python

.PHONY: install test quality smoke build verify clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest --cov=finops_r --cov-report=term-missing --cov-fail-under=80

quality:
	ruff check .
	ruff format --check .
	mypy

smoke:
	finops-r run --config configs/smoke.json

build:
	$(PYTHON) -m build

verify: test quality build
	$(PYTHON) -m pip check

clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('build', 'dist', '.pytest_cache', '.mypy_cache', '.ruff_cache')]"
