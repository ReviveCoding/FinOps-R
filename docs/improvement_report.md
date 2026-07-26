# FinOps-R v0.2.0rc4 improvement report

## Executive verdict

`CONDITIONALLY COMPLETE`

The repository is an executable release candidate with E2 local execution and E3 clean-wheel reload evidence. Critical software defects found in the improvement rounds are resolved, the full suite passes, and the data-to-report-to-reload pipeline executes. The advanced C5 model is correctly held and C1 remains the fallback. GitHub-hosted, Windows and Python 3.11 qualification are prepared but not executed, so this is not `RELEASE QUALIFIED`.

## Project context and frozen gates

- Problem: controlled T+1 post-trade exception diagnosis and action routing.
- Users: ML engineer, operations scientist, model-risk reviewer and recruiter.
- Constraints: Python 3.11 or 3.12 design, Windows and Ubuntu scripts, CPU fallback, no paid API, public context plus synthetic case data, no external financial execution.
- Primary metric: C5 residual harm per 1,000 versus C1, lower is better.
- Promotion threshold: configured minimum relative improvement plus paired 95% interval above zero.
- Additional gates: critical false-greenlight, evidence precision, total manual capacity, hold rate, worst slice, shifted stress harm, shifted stress capacity, unauthorized action count and artifact integrity.
- Predictive tolerance: no more than 0.005 Macro-F1 regression for the intended champion comparison.
- Maximum major rounds: 3.
- Frozen rc2 baseline source fingerprint: `f4c3915354845db63351bfb0364adda7852ac9fe98a498b51c2446f3f7706430` over 45 declared files after excluding generated egg metadata and the corrupt upload residue.
- Git status: NOT_APPLICABLE. The supplied workspace was not a Git repository.
- Rollback point: `FinOps-R_v0.1.0_full.zip`, validated by `unzip -t`.

## Phase 0 baseline

| Check | Baseline result | Evidence level |
|---|---|---:|
| Python and OS | Python 3.12.13, Linux 6.12 x86_64 | E2 |
| Uninstalled unit command | 5 collection imports failed because `finops_r` was not installed | E2 |
| Editable install in active interpreter | Failed because `/root/.local` was read-only | E2 |
| `PYTHONPATH=src` suite | 8 passed, but about 47.5 seconds and no coverage gate | E2 |
| Clean dependency environment | Fresh venv installed project, dev and API dependencies | E3 |
| Ruff lint | 10 errors | E2 |
| Ruff format | 19 files unformatted | E2 |
| Build | No-isolation build lacked setuptools; isolated build was blocked by a transient proxy | E2 |
| CI | Missing | E0 |
| Artifact trust | Joblib loaded before checksum and runtime compatibility checks | E0 plus reproduced mismatch later |
| Full experiment | Existing C5 HOLD result, but hard HOLD demand was outside the stated review capacity | E2 |

## Priority analysis

Score is `severity × likelihood × scope × evidence confidence / cost` on the supplied scale.

| Problem | Severity | Score | Root cause | Decision |
|---|---|---:|---|---|
| Total manual load exceeded fixed capacity | Critical | 62.5 | HOLD and REVIEW budgets were added independently | Fixed in Round 2 |
| Incompatible sklearn artifact could deserialize | Critical | 50.0 | No external runtime contract before pickle load | Fixed in Round 3 |
| Artifact corruption was checked after load | Critical | 37.5 | Loader called joblib directly | Fixed in Round 1 |
| Config and path contracts were shallow | High | 21.6 | Required-key check only | Fixed in Round 1 |
| Stress split was generated but not gated | High | 19.2 | Evaluation stopped at normal test split | Fixed in Round 2 |
| Partial run directories could appear final | High | 18.0 | Direct writes to final run path | Fixed in Round 1 |
| Model artifacts were oversized | Medium | 20.0 | Uncompressed joblib plus transient files | Fixed in Round 1 |
| Build, coverage and CI contract missing | Medium | 20.0 | Dev tools were declared but not operationalized | Fixed in Round 3 |

## Improvement rounds

### Round 1: contracts, artifact trust and atomic publication

- Added Pydantic contracts for configuration, cases and assessments.
- Rejected absolute or parent-traversing output paths and non-finite values.
- Made malformed and empty dataset validation return explicit FAIL results.
- Added checksum, size, traversal, duplicate and symlink verification.
- Compressed joblib output and excluded hidden temporary files from manifests.
- Wrote to hidden staging directories and atomically published only verified runs.
- Cached one verified bundle for the API instead of loading on every request.

Validation: 7 targeted contract/data tests passed; E2E including corruption and interrupted publication passed.

### Round 2: scientifically valid capacity and stress evaluation

- Made mandatory HOLD consume the same total manual capacity as REVIEW.
- Made EVR use correction probability, normalized harm and review cost.
- Prevented EVR from filling capacity with negative-value cases.
- Evaluated all candidates on the shifted stress split.
- Added stress predictive, harm and manual-capacity gates plus the eighth diagnostic figure.
- Added atomic report and card writes.

Validation: decision and evidence tests passed. The 3,600-case smoke run was HOLD and exposed a 1.324 C5/C1 stress-harm ratio. Full-run metrics were deterministic across repeated runs.

### Round 3: packaging, runtime compatibility and handoff

- Added a pinned qualification environment, wheel and sdist build, CI matrix, Linux and PowerShell scripts, license and coverage gate.
- Added an external runtime JSON that is checksum-verified before joblib deserialization.
- Required exact joblib, NumPy, pandas, Pydantic, scikit-learn and SciPy versions plus matching Python minor.
- Verified a compatible wheel reload and an incompatible environment fail-closed path.
- Added pipeline, limitations, changelog and machine-readable handoff documents.

Validation: 31 source-tree tests passed in the exact locked runtime with 86.33% coverage. Ruff lint and format passed. Wheel and sdist builds passed.

### Round 4: policy parity, complete validation and publication isolation

- Activated the previously unused policy split and fitted one review-score threshold per candidate.
- Reused the selected candidate's learned threshold in single-case API and CLI inference, removing batch/API routing drift.
- Expanded data validation to the complete schema and all target invariants, while reporting multiple simultaneous failures.
- Built each run under an OS temporary owner, published only a fully verified tree, and reverified the copied destination.
- Required an exact file inventory and rejected hidden content, duplicate records, unsafe paths, and component symlinks.
- Moved every release tolerance into validated configuration and recorded it with the decision.

Validation: 31 failure-injection and E2E tests passed, including corruption, unsafe paths, duplicate columns, invalid probabilities, failed atomic writes, policy-threshold reuse, and interrupted publication.

### Round 5: concurrent publication, platform paths and executable static analysis

- Reproduced that publication copied directly to the final public directory, allowing a concurrent reader to observe partial content despite the atomic-publication claim.
- Copied to a hidden sibling on the destination filesystem, verified the complete copy, then exposed it with `os.replace`; injected copy failure leaves neither a public run nor hidden residue.
- Replaced the report module's Linux-only `/tmp` default with `tempfile.gettempdir()` and tested both the platform default and user override in isolated interpreters.
- Added a locked mypy gate to local scripts and CI, then resolved all findings in 15 source modules.
- Added concurrent pointer/read publication, out-of-order rows, empty evaluation, single-class imbalance, and repeated deterministic run checks.
- Quarantined a 47 MB corrupt upload residue that rc2 packaging had accidentally included.

Validation: 39 tests passed with 86.71% coverage; Ruff, mypy, 3,600-case smoke, 12,000-case full run, exact 59-file verification, and clean wheel reload passed.

### Round 6: clustered inference, fixed estimand and dependency audit

- Reproduced that the case-level bootstrap treated three observations from one entity as independent even though the split contract is entity-based.
- Changed paired uncertainty estimation to resample complete entity clusters while preserving candidate pairing within every draw.
- Froze the complete candidate class universe before Macro-F1 resampling so rare-class omission cannot change the metric denominator between draws.
- Added deterministic reference and rejection tests for cluster draws, fixed labels, non-finite costs and invalid group/label inputs.
- Audited the previous exact lock, upgraded vulnerable pytest/setuptools versions, and added `pip-audit` to local and CI qualification.
- Added executable version-consistency contracts across package metadata, runtime version and API.

Validation: 43 tests plus 14 subtests passed with 86.44% coverage; Ruff, mypy, Bandit medium/high scan, dependency audit, 3,600-case smoke, and corrected 12,000-case full run passed.

## Baseline versus final status

| Check | Baseline | Final candidate |
|---|---|---|
| Tests | 5 collection errors uninstalled, 8 pass with manual `PYTHONPATH` | 43 passed plus 14 subtests, 0 failed, 0 skipped in exact locked runtime |
| Coverage | Not configured | 86.44%, gate 70% |
| Lint | 10 errors | PASS |
| Format | 19 files failed | PASS |
| Build | Failed in baseline procedure | Wheel and sdist PASS |
| E2E | Executable but direct publication | Atomic staged full pipeline PASS |
| Artifact verification | Post-hoc only | 59-file exact inventory recomputed PASS before load and after publication |
| Clean reload | Not established | Exact-dependency wheel PASS from `/tmp` |
| Incompatible reload | Warned and continued | Rejected before deserialization |
| API | In-process factory only | Strict contract, cached verified bundle, Uvicorn startup PASS |
| CI readiness | Missing | Ubuntu/Windows and Python 3.11/3.12 matrix checked in, E1 only |
| GPU | No executable path | NOT_APPLICABLE, documented |
| Docker | Missing | NOT_APPLICABLE, documented |

## Final experiment

Canonical run: `finopsr_20260718T195414Z_0588f8c980`.

| Candidate | Macro-F1 | Brier | ECE | Harm/1,000 | Unsafe auto harm/1,000 | Manual rate |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 0.96310 | 0.06490 | 0.00986 | 74.75 | 30.15 | 19.10% |
| C3 | 0.96558 | 0.06248 | 0.01514 | 68.22 | 24.29 | 19.93% |
| C5 | 0.96449 | 0.06182 | 0.01348 | 136.93 | 56.95 | 19.93% |

C3 is the point-estimate predictive and safety champion, but its C3-minus-C1 Macro-F1 paired entity-cluster 95% interval is `[-0.00189, 0.00717]`; superiority is not established. C5 is 83.2% worse than C1 on the primary harm metric, and the paired entity-cluster C1-minus-C5 interval `[-0.09271, -0.03547]` excludes improvement. C5 is therefore HOLD.

On stress cases, C1 has Macro-F1 0.94360 and harm 141.45, while C5 has Macro-F1 0.94160, harm 362.71 and manual intervention 36.30%. Predictive degradation alone is small, but its operational failure is material.

## Evidence log

| Command | Exit | Result |
|---|---:|---|
| `python -m unittest discover -s tests -v` before install | 1 | 5 import errors |
| `python -m pip install -e . --no-deps` in active interpreter | 1 | read-only user site |
| clean venv dependency install | 0 | complete isolated dependency environment |
| `ruff check .` baseline | 1 | 10 errors |
| `ruff format --check .` baseline | 1 | 19 files |
| targeted Round 1 tests | 0 | 7 plus E2E passed |
| targeted Round 2 tests | 0 | 4 plus E2E passed |
| `finops-r run --config configs/smoke.json` | 0 | HOLD, integrity PASS |
| `finops-r run --config configs/full.json` final | 0 | 12,000 cases, HOLD, integrity PASS |
| exact-runtime source pytest with coverage | 0 | 43 passed plus 14 subtests, 86.44% |
| `pip-audit --no-deps -r requirements-lock.txt` | 0 | no known vulnerabilities in the exact lock |
| `bandit -q -r src -ll` | 0 | no medium/high findings |
| `mypy` | 0 | 15 source modules, no issues |
| `ruff check .` final | 0 | PASS |
| `ruff format --check .` final | 0 | PASS after canonical formatting |
| `python -m build --no-isolation` | 0 | wheel and sdist built |
| exact-runtime wheel `verify` and `assess` from `/tmp` | 0 | PASS |
| incompatible-runtime wheel `assess` | 2 | rejected before joblib load |
| Uvicorn service startup | 0 on controlled shutdown path | startup complete; cross-namespace curl unavailable |

## Evidence levels

- E0: repository and documentation inspection complete.
- E1: canonical local and CI commands defined.
- E2: Linux commands, tests, full experiment, API startup and artifact checks executed.
- E3: dependency environment created from scratch, built wheel installed, and artifact reloaded from a clean working directory.
- E4: NOT EXECUTED. An exact Git commit and GitHub-hosted run are required next.

## Final classification

- Critical unresolved: 0.
- Executable High software issues unresolved: 0.
- Release blocker: C5 model and policy performance, safely mitigated by C1 fallback.
- Recommended follow-up: multi-seed C3 validation and a redesigned capacity-aware C5 evidence policy.
- Optional: Qwen or LangGraph challenger only after a versioned policy corpus exists.
- Intentionally not implemented: external execution, GPU path, Docker and bank connectors.
