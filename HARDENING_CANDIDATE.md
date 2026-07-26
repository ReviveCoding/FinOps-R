# FinOps-R JSON hardening candidate

This source tree keeps package version `0.3.0` while the JSON boundary hardening is
reviewed and requalified. It is not a `0.3.1` release.

## Added boundary behavior

- Reject duplicate object keys at every nesting depth.
- Reject `NaN`, `Infinity`, `-Infinity` and finite-float overflow on input.
- Reject non-finite Python values on JSON output.
- Require top-level objects for config and assessment request documents.
- Validate API request bytes before FastAPI/Pydantic parsing and enforce a 1 MiB limit.

## Evidence status

The checked-in `qualification_manifest.json`, `release_bundle_manifest.json`, release
notes and previously built distributions describe the original v0.3.0 release. They do
not qualify this candidate. Build fresh distributions and run the repository's complete
qualification workflow before changing the package version or publishing a release.

Local candidate verification completed before packaging:

- Python source compilation: PASS
- Strict JSON, dataset, artifact and CLI tests: 44/44 PASS
- Full source suite: 83/84 PASS; the only unexecuted completion point was the existing
  FastAPI app creation in the E2E test because the optional API dependency was absent in
  the verification environment. The E2E pipeline had already completed training,
  reporting and 59-file artifact integrity verification before that dependency check.

This partial evidence is development evidence only, not release qualification.
