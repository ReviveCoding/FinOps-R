# Contributing

FinOps-R accepts changes that preserve its fail-closed release and claim boundaries.

1. Create a branch and a Python 3.11 or 3.12 environment.
2. Run `python -m pip install -e ".[dev]"`.
3. Add or update tests for every behavioral change.
4. Run `make verify` (or the equivalent commands in `scripts/qualify.ps1`).
5. Describe data provenance, leakage controls and metric changes in the pull request.

Do not commit source datasets, model artifacts, generated run directories, credentials, customer information or personally identifiable information. New external data adapters must declare an official source URL, snapshot/hash policy, license or terms, intended use and prohibited claims.

A model improvement is not accepted solely because its average score increases. Report uncertainty, worst slices, calibration, manual workload and the locked release decision. A valid `HOLD` is preferable to an unsupported promotion.
