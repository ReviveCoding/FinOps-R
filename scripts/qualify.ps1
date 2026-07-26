$ErrorActionPreference = "Stop"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
if (-not $env:MPLCONFIGDIR) {
    $env:MPLCONFIGDIR = Join-Path $PWD ".mplconfig"
}
if (-not $env:PIP_CACHE_DIR) {
    $env:PIP_CACHE_DIR = Join-Path $PWD ".pip-cache"
}

python -m pip install ".[dev]"
python -m pip install "pip-audit>=2.10,<3"
python -m pip check
$AuditCache = Join-Path ([System.IO.Path]::GetTempPath()) "finops-r-pip-audit-cache"
pip-audit --cache-dir $AuditCache --progress-spinner off
python -c "from pathlib import Path; import finops_r; assert not Path(finops_r.__file__).resolve().is_relative_to(Path('src').resolve())"
python -m pytest --cov=finops_r --cov-report=term-missing --cov-fail-under=80
ruff check .
ruff format --check .
mypy
python -m build
python -m pip install --force-reinstall --no-deps (Get-ChildItem dist\*.whl | Select-Object -First 1).FullName
finops-r --version
finops-r data list
finops-r run --config configs\smoke.json
