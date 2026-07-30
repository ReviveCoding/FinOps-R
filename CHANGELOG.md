# Changelog

## 0.4.0

- Promoted the E4-qualified release candidate to the final 0.4.0 package.
- Required exact-commit hosted verification across Linux and Windows on Python
  3.11 and 3.12 before tagging or publication.
- Preserved the locked scientific decision: C5 remains HOLD and C1 remains the
  production reference.
- Added strict JSON boundaries that reject duplicate keys, non-finite numbers,
  finite-float overflow and non-object configuration or request payloads.
- Added a 1 MiB API request limit and bounded normal error responses while
  preserving explicit traceback diagnostics under debug mode.
- Closed the CodeQL stack-trace-exposure finding.
- Raised the verified suite to 202 passing tests with 81.13% branch coverage.

## 0.4.0rc3

- Added an exact-commit GitHub-hosted E4 evidence contract.
- Required Linux and Windows on Python 3.11 and 3.12 to reverify Stage 4 and Stage 5.
- Added a strict four-cell evidence-set verifier and failure tests.
- Corrected the CI wheel assertion from stale version 0.3.0 to 0.4.0rc3.
- Raised the full suite to 200 tests plus 61 subtests with 81.11% branch coverage.
- Preserved model promotion HOLD and C1 as the production reference.

## 0.4.0rc2

- Added a locked Stage 5 integrated qualification joining the model release and Stage 4
  orchestration evidence without retuning either test family.
- Added independent verification, artifact hashing and a human-readable integrated report.
- Kept model promotion and final release status separate from local software qualification.
- Preserved C1 as the safe production fallback when the locked C5 gates remain HOLD.

## 0.4.0rc1

- Added an optional LangGraph path that shares the typed Stage 4 reference
  contract and supports durable human-review interrupt/resume.
- Added hash-chained SQLite checkpoints, restart recovery and an idempotent
  TradeOps-Sim effect ledger.
- Added deterministic crash injection covering input persistence and the
  effect-commit-to-state-checkpoint gap.
- Added immutable release heads that reject any post-release checkpoint write.
- Added a reproducible frozen logistic risk verifier in veto-only mode, with
  separately hashed development training and threshold-selection snapshots.
- Added a nine-scenario locked orchestration qualification with direct/LangGraph
  parity, HITL, duplicate-effect, crash-replay, external-execution and trace
  immutability gates.
- Kept public auxiliary datasets outside the canonical Stage 4 execution path.

## 0.3.0

- Rejected multiply linked artifact files so a verified tree cannot retain an external hard-link mutation path.
- Made ZIP inspection hash the same open file that passed CRC/member validation, reject conflicting file-type markers and fail closed on invalid resource-limit arguments.
- Required rejected ISO payments to carry canonical non-empty reasons and valid non-backward settlement dates.
- Enforced exact runtime-contract keys, canonical Python versions and the complete expected package set.
- Made every supported minimum-size generated dataset contain the full training label universe and at least 20 healthy training references without changing already-valid deterministic outputs.
- Kept `--output-root` runs and release pointers under the override's publication root instead of contaminating the configuration repository.
- Prevented a verified model bundle and its metadata from being swapped between checksum verification and deserialization by loading hash-rechecked byte snapshots.
- Enforced semantic consistency across the production bundle, runtime contract, resolved config, release decision, run manifest and fitted decision policy.
- Made tied-score review thresholds conservatively capacity-safe during one-case online replay.
- Hardened ISO staging with a bounded immutable source snapshot, trimmed unique identifiers and special-member rejection.
- Rejected booleans, non-finite values, unknown fields, fractional counts, missing data splits and incomplete training label universes at the contract boundary.
- Added graceful Windows API smoke shutdown with a new process group and `CTRL_BREAK_EVENT`.
- Made same-configuration concurrent runs collision-resistant with a full UUID nonce.
- Made publication failure-atomic when private staging cleanup fails; no visible destination is left behind after a reported failure.
- Added shell-standard Ctrl+C handling (`130`) without default tracebacks while preserving `--debug` behavior.
- Hardened ZIP and artifact namespaces against Windows drive/device names, control characters, long components, Unicode-normalization collisions, case-fold collisions and file/directory prefix collisions.
- Added portable POSIX artifact manifest paths so manifests generated on Windows verify identically on Linux.
- Rejected negative generator seeds, entity counts larger than case counts and non-portable configured output paths before execution.
- Rejected non-finite ISO payment amounts, malformed registry roots/IDs and interrupt-created ISO staging residue.
- Expanded the suite to 76 tests plus 70 subtests with 87.91% branch coverage.
- Removed a conditional API bypass from the critical E2E suite; the API contract is now mandatory in the development qualification environment.
- Added public CLI `--version` and an opt-in `--debug` traceback path while keeping normal errors bounded, structured and non-zero.
- Added an actual TCP API lifecycle smoke covering three startup/shutdown cycles, validation failures, concurrent health checks and port conflicts.
- Changed qualification and CI installs from editable to non-editable package installs and added isolated built-wheel verification in CI.
- Added a writable pip cache contract so sdist builds work in read-only-home execution environments.
- Fixed four files that failed the canonical `ruff format --check` gate.
- Promoted the repository from release candidate to a GitHub-ready final package.
- Added a packaged dataset registry and an atomic, fail-closed adapter for the supplied ISO 20022-like synthetic ZIP.
- Added archive traversal, symlink, duplicate-name, encryption, CRC, expansion and compression-ratio checks.
- Added paired pacs.008/pacs.002/analytics schema, identifier, status, rejection-reason, timestamp and amount validation.
- Added chronological proxy split assignments and explicit dataset/claim governance.
- Separated the locked Python 3.12 Linux environment from cross-platform Python 3.11/3.12 CI installation.
- Raised the coverage gate to 80% and added dataset-contract tests.
- Added Docker, Make, contribution, citation and security assets.
- Added CodeQL, dependency review and Dependabot configuration.
- Pinned every GitHub Action to a verified full commit SHA while retaining release-version comments for Dependabot review.
- Removed unused GPU/LLM extras from the implemented package surface; language-model challengers remain future experiments, not implied functionality.
- Reduced the smoke configuration to a 1,200-case functional budget while preserving the 12,000-case locked scientific configuration.

## 0.2.0rc4

- Fixed paired bootstrap inference so repeated cases from the same entity are resampled as one cluster, preserving within-entity dependence.
- Froze the full 14-class label universe before every Macro-F1 bootstrap draw so missing rare classes cannot silently change the estimand.
- Added deterministic reference tests for fixed-label and complete entity-cluster resampling plus malformed, non-finite and unknown-label inputs.
- Added package/project/API version-consistency tests to prevent release metadata drift.
- Upgraded pytest to 9.0.3 and setuptools to 83.0.0 after a fresh `pip-audit` found known vulnerabilities in the previous qualification lock.
- Added `pip-audit` to Linux, PowerShell and CI qualification paths with an explicit writable cache; the rc4 exact lock reports no known vulnerabilities.
- Re-ran the 12,000-case experiment. The corrected cluster intervals preserve the scientific decision: C5 remains HOLD and C1 remains the production fallback.
- Expanded the source suite to 43 tests plus 14 subtests with 86.44% coverage.

## 0.2.0rc3

- Fixed a publication race that could expose a partially copied run to concurrent readers; publication now copies to a hidden same-filesystem sibling, verifies it, and atomically replaces the final name.
- Added concurrent JSON-pointer and run-publication reader/writer tests plus injected-copy-failure cleanup coverage.
- Replaced the Linux-only Matplotlib `/tmp` default with Python's platform temp-directory selection while preserving explicit overrides.
- Added mypy as a locked local/CI/qualification gate and resolved all findings across 15 source modules.
- Centralized runtime/API/model artifact version values on the package version to prevent drift.
- Added deterministic out-of-order-row, empty evaluation, single-class imbalance, and repeated-run edge contracts.
- Expanded the suite to 39 tests and 86.71% source coverage before the final clean-package qualification.

## 0.2.0rc2

- Added complete training-data validation, including duplicate columns, identifiers, timezone, categories, numeric finiteness, binary fields, targets and entity leakage.
- Fit candidate-specific review thresholds on the dedicated policy split and reused the production threshold for single-case inference.
- Moved run construction to an OS temporary tree, added exact-inventory verification before and after publication, and removed all run-owned staging on failure.
- Added atomic JSON/CSV/text/joblib cleanup, strict manifest schemas, hidden-file and component-wise symlink rejection, and portable lineage pointers.
- Made report regeneration publish a separately verified run copy so interruption cannot corrupt the source release.
- Made every release tolerance configurable and recorded the resolved thresholds in the release decision.
- Locked NumPy below 2.5 for Python 3.11 support and qualified the exact NumPy 2.3.5/scikit-learn 1.9.0 environment.
- Expanded failure-injection coverage to 31 tests; source coverage is 86.33%.
- Preserved the scientific result: C5 is HOLD and verified C1 remains the production fallback.

## 0.2.0rc1

- Added strict project, case, assessment and nested configuration contracts.
- Added fixed-total-manual-capacity routing and a positive-value EVR rule.
- Connected the stress split to numeric metrics, visualization and release gates.
- Added atomic staged run publication and previous-pointer preservation on failure.
- Added compressed model artifacts, reducing the validated full run from 353 MB to about 51 MB.
- Added manifest, checksum, symlink, path and runtime compatibility verification before model loading.
- Added a strict read-only FastAPI contract and cached verified bundle loading.
- Added locked dependencies, Linux and Windows qualification scripts, CI matrix, coverage, lint and build gates.
- Added failure tests for malformed data, path traversal, non-finite input, capacity, corruption, interrupted publication and incompatible model runtime.
- Preserved the negative result: C5 remains HOLD and C1 remains the fallback.
