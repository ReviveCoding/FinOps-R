#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/finops-r-matplotlib}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PWD/.pip-cache}"

python -m pip install -r requirements-lock.txt
python -m pip install "pip-audit==2.10.1"
python -m pip install --no-build-isolation --no-deps .
python -m pip check
pip-audit --cache-dir "${TMPDIR:-/tmp}/finops-r-pip-audit-cache" --no-deps -r requirements-lock.txt --progress-spinner off
python -c "from pathlib import Path; import finops_r; assert not Path(finops_r.__file__).resolve().is_relative_to(Path('src').resolve())"
python -m pytest --cov=finops_r --cov-report=term-missing --cov-fail-under=80
ruff check .
ruff format --check .
mypy
python -m build --no-isolation
python -m pip install --force-reinstall --no-deps dist/*.whl
finops-r --version
finops-r data list
finops-r run --config configs/smoke.json
