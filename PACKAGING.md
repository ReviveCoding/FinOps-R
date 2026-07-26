# Packaging and release

FinOps-R v0.4.0rc3 uses `pyproject.toml` as the build configuration and produces a wheel plus source distribution.

## Local qualification

Linux, Python 3.12:

```bash
./scripts/qualify.sh
```

Cross-platform development:

```bash
python -m pip install -e ".[dev]"
make verify
```

Windows uses `scripts/qualify.ps1`. The exact `requirements-lock.txt` is the Linux Python 3.12 qualification snapshot; it is intentionally not described as a universal lock.

## Build

```bash
python -m build
python -m pip install --force-reinstall --no-deps dist/finops_r-0.4.0rc3-py3-none-any.whl
finops-r --help
finops-r data list
```

The sdist and wheel must include `finops_r/dataset_registry.json`. CI verifies the installed version and CLI from the built wheel.

## GitHub release

1. Require the CI and security checks on the default branch.
2. Update `CHANGELOG.md` and run the locked qualification.
3. After E4 evidence passes, create an annotated `v0.4.0rc3` tag from the reviewed commit.
4. Build from a clean checkout and attach the wheel, sdist and SHA-256 checksum file to the GitHub release.
5. Do not attach source datasets, staged external data, run directories or untrusted joblib artifacts.

The archive delivered with this work is a Git-ready source snapshot. Generated experiment evidence remains reproducible from `configs/smoke.json` and `configs/full.json` rather than being committed as mutable binary output.
