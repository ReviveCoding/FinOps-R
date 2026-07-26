# Security policy

## Supported version

Security fixes are applied to the latest release on the default branch.

## Reporting

Do not open a public issue for a suspected vulnerability. Use the repository's private security-advisory reporting flow after the repository is published on GitHub. Do not include real financial records, credentials, access tokens or proprietary model artifacts in a report.

## Trust boundary

- Joblib model files are executable Python serialization. Load only artifacts created locally by a verified FinOps-R run.
- Every model load verifies the exact inventory, hashes, sizes, symlink boundary, Python minor version and core package versions before deserialization.
- External ZIPs are inspected for CRC failures, traversal, symlinks, encryption, duplicate names and expansion bounds before selected files are staged.
- JSON trust boundaries reject duplicate object keys, non-finite numeric extensions and
  finite-float overflow; assessment API JSON is limited to 1 MiB before model validation.
- The API is assessment-only. It does not connect to a bank, broker, custodian, payment rail or order-management system.
- Generated decisions operate only on controlled simulator state.
