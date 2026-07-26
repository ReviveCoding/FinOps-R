# FinOps-R v0.3.0 final repository verification

Verification date: 2026-07-20 (America/New_York)

## Outcome

The Git-ready source tree, wheel, source distribution and supplied ISO 20022-like adapter passed every check executable in the local workspace. GitHub CI remains the authoritative gate for the optional API dependency, pytest branch coverage, Ruff, mypy, dependency audit, Windows and Python 3.11 because this workspace prohibited installing the locked development toolchain.

## Local evidence

| Check | Result |
|---|---|
| Python syntax compilation | PASS |
| Core unit/contract/end-to-end suite | PASS, 48 tests via `unittest` |
| Dataset/CLI/runtime-contract regression subset | PASS, 15 tests |
| Dataset/security-adapter tests re-executed from final source ZIP | PASS, 5 tests |
| `pyproject.toml` and tracked JSON parsing | PASS |
| Git staged-tree whitespace check | PASS |
| Wheel build and isolated `--no-deps` install | PASS |
| Wheel version and packaged registry | PASS, 0.3.0 |
| Source distribution build and contents | PASS |
| Source and distribution ZIP CRC checks | PASS |
| Forbidden generated/source-data paths in source ZIP | PASS, none present |
| Supplied ISO synthetic snapshot | PASS, 3 CSVs and 5,000 paired rows |
| Supplied FUNSD ZIP CRC/safety inspection | PASS, 824 members |

## ISO snapshot contract

- Source SHA-256: `3d1b5885c874a2fc7d937c7bc11ab5d80b806d5aabf256474592ae2ea0888024`
- Rows: 5,000 each in analytics, pacs.002 and pacs.008.
- Statuses: 4,680 `ACSP`, 320 `RJCT`.
- Chronological proxy split: 3,000 train, 1,000 validation, 1,000 test.
- Validation covered archive safety, exact schemas, unique and paired IDs, status/rejection consistency, timestamp-derived processing time and positive instructed amount.
- Boundary: project-generated synthetic flattened CSV, not real bank data and not official ISO 20022 conformance evidence.

## Required GitHub checks

The committed workflows must pass before a release tag is created:

1. Ubuntu and Windows on Python 3.11 and 3.12.
2. Pytest with branch coverage at or above 80%.
3. Ruff lint/format and mypy.
4. Locked Linux dependency consistency and build.
5. Installed-wheel CLI and dataset registry.
6. Controlled smoke experiment.
7. CodeQL and pull-request dependency review.

Docker was not locally built because the workspace has no Docker executable. The Dockerfile is therefore CI/reviewer-verifiable, not claimed as locally executed.
