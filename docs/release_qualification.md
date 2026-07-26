# FinOps-R 0.3.0 release qualification

## Verdict

**CONDITIONALLY QUALIFIED**

Linux x86_64 / CPython 3.12.13 source, network-resolved install, built-artifact, E2E, artifact and TCP API gates pass. GitHub-hosted, Windows, Python 3.11 and Docker execution are `NOT_RUN`, so this is not an unconditional multi-platform release claim. The repository has no Git commit or remote; `qualified_commit` remains null.

## Final local matrix

| Gate | Result | Evidence |
|---|---|---|
| Full source suite 1 | PASS | 70 tests, 36 subtests, branch coverage 86.81% |
| Full source suite 2 | PASS | 70 tests, 36 subtests |
| Lint / format / types | PASS | Ruff clean; 31 formatted files; mypy 17 source files |
| Build | PASS | wheel + sdist via `python -m build --no-isolation` |
| Wheel / sdist inventory | PASS | 24 / 80 entries; CRC and path checks |
| Wheel install and suite | PASS | fresh resolver, non-editable import, `pip check`, 70 tests |
| Sdist install | PASS | fresh resolver, isolated local build, CLI, registry, `pip check` |
| Installed-wheel E2E | PASS | two 1,200-case runs; distinct IDs; identical release decision |
| Artifact integrity | PASS | 59 files verified in each run; independent assessment reload |
| API lifecycle | PASS | 3 cycles; 96 concurrent health requests; 0 errors; occupied port non-zero |
| Attached ISO data | PASS | 5,000 aligned rows; immutable-source staging contract |
| Attached FUNSD archive | PASS | 824 members; CRC and safe-name audit |
| Dependency audit | PASS | no known vulnerabilities for `requirements-lock.txt` |

No final test was skipped or xfailed. The final smoke decision is `HOLD`; C1 remains the fallback because C5 does not pass every release gate.

## Runtime contract

- CPython 3.12.13 on Linux x86_64
- Latest-range wheel test: NumPy 2.4.6, pandas 2.3.3, scikit-learn 1.9.0, SciPy 1.18.0
- Qualification tools: pytest 9.0.3/9.1.1, Ruff 0.15.22, mypy 1.20.2, build 1.5.0
- API: FastAPI 0.139.2 and Uvicorn 0.51.0

## Commands

```bash
python -m pytest --cov=finops_r --cov-branch --cov-fail-under=80
ruff check .
ruff format --check .
mypy src/finops_r
python -m build --no-isolation
finops-r run --config configs/smoke.json
finops-r verify --run-dir artifacts/runs/<run_id>
python scripts/api_smoke.py --run-dir artifacts/runs/<run_id> --fixture data/fixtures/sample_case.json --cycles 3
```

## Conditions and claim boundary

- Workflows are hardened and matrixed, but no hosted run exists.
- The exact Linux lock pins versions but not distribution hashes or signatures.
- joblib loading is allowed only for trusted-local artifacts after inventory, checksum, stable-byte, runtime and cross-file checks; pickle is not a sandbox.
- ISO and other external datasets are governed proxy/stress inputs; only TradeOps-Sim drives the core model claim.

Machine-readable counts, fingerprints and distribution hashes are in `qualification_manifest.json`, `release_bundle_manifest.json` and the external SHA-256 file.
