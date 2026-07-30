# FinOps-R v0.4.0

FinOps-R is an end-to-end, risk-controlled benchmark for T+1 post-trade exception resolution. It generates a leakage-safe controlled dataset, trains a fixed candidate ladder, calibrates decisions, allocates scarce human review, validates action evidence, measures residual harm and promotes a model only when every locked release gate passes.

The project is deliberately honest about its evidence: case labels and outcomes come from TradeOps-Sim, not a bank. Public datasets remain isolated proxy or stress benchmarks. The framework demonstrates experimental and operational discipline; it does not claim production settlement improvement.

## Why this repository is different

- **A decision system, not only a classifier.** The primary objective is residual harm under a fixed manual-review capacity.
- **Attributable candidates.** C1 is a strong tabular baseline; uncertainty, RSOP, evidence envelopes and calibration are added in separate candidates.
- **Fail-closed promotion.** Confidence intervals, critical false greenlights, evidence precision, worst slices, stress harm, workload and artifact integrity can block promotion.
- **Entity-safe evaluation.** Train, validation, calibration, policy, test and stress splits are assigned by entity.
- **Verified artifacts.** Runs are written atomically and loaded only after exact inventory, hash, size, symlink and runtime checks.
- **Governed external data.** The supplied ISO 20022-like snapshot has a safe ZIP and schema adapter; BPI, CFPB, SEC FTD and FUNSD have explicit roles and prohibited claims.
- **Integrated release evidence.** Stage 5 binds the locked model release posture to the durable-orchestration qualification while keeping software PASS, model HOLD and final-release readiness as separate decisions.
- **Local and GitHub ready.** Linux/Windows CI, Python 3.11/3.12 tests, CodeQL, dependency review, Dependabot, wheel/sdist builds and a non-root Docker image are included.

## Candidate ladder

| ID | Candidate | Increment over prior controls |
|---|---|---|
| C0 | Deterministic rules | Sanity check |
| C1 | Histogram gradient boosting | Industry baseline and production fallback |
| C2 | HGB + ExtraTrees | Ensemble disagreement |
| C3 | HGB + RSOP | Train-fitted domain specialty |
| C4 | Ensemble + RSOP + AEE | Evidence-aware actions |
| C5 | C4 + temperature calibration + EVR | Guarded production candidate |

RSOP is a reference-state operational-pressure transform. AEE is the Action Evidence Envelope. EVR is expected value of review.

## Quick start

Python 3.11 or 3.12 is required.

```bash
cd finops-r
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install ".[dev]"
finops-r --version
finops-r run --config configs/smoke.json
```

The run path is printed at completion. It contains trained models, decision policy, predictions, slice metrics, confidence intervals, failure cases, figures, HTML/Markdown reports, model/data/evaluation/release cards, a release decision and an integrity manifest.

Qualification:

```bash
make verify
finops-r run --config configs/smoke.json
python -m pip install ".[orchestration]"
python scripts/run_orchestration_qualification_stage4.py
```

Stage 4 adds a simulation-only LangGraph adapter, durable hash-chained SQLite
checkpoints, human-review interrupt/resume, crash replay, an idempotent
simulation-effect ledger, a frozen veto-only learned risk verifier and a locked
orchestration qualification. A verifier flag can reduce automation but can
never authorize an action. See
[Stage 4 orchestration](docs/ORCHESTRATION_QUALIFICATION_STAGE4.md).

Windows PowerShell:

```powershell
.\scripts\qualify.ps1
```

After a run exists, exercise the installed API over a real TCP socket:

```bash
python scripts/api_smoke.py --run-dir artifacts/runs/<run_id> \
  --fixture data/fixtures/sample_case.json --cycles 3
```

The Linux Python 3.12 release qualification uses `requirements-lock.txt`. Cross-platform CI installs the declared development extra because the Linux lock is intentionally not presented as a Windows lock.

## Dataset workflow

List the governed registry:

```bash
finops-r data list
```

Validate and stage the supplied synthetic ISO snapshot:

```bash
finops-r data verify-zip --archive /path/to/ISO20022_Payments_Analytics_Synthetic_v1.zip
finops-r data stage-iso \
  --archive /path/to/ISO20022_Payments_Analytics_Synthetic_v1.zip \
  --output data/external/iso20022_synthetic_v1
```

The adapter checks the archive before staging and writes a source hash, validation evidence and chronological split assignments. `data/external/` is ignored by Git. See [Dataset governance](docs/DATASETS.md) for official sources, snapshot rules and why unrelated attached datasets are excluded.

## Main commands

```bash
finops-r generate --config configs/smoke.json
finops-r run --config configs/full.json
finops-r inspect --run-dir artifacts/runs/<run_id>
finops-r verify --run-dir artifacts/runs/<run_id>
finops-r assess --run-dir artifacts/runs/<run_id> --request data/fixtures/sample_case.json
finops-r report --run-dir artifacts/runs/<run_id>
finops-r serve --run-dir artifacts/runs/<run_id> --host 127.0.0.1 --port 8000
finops-r rollback --run-id <promoted_run_id>
```

Add the global `--debug` option before a subcommand only when an original Python traceback is
needed for local diagnosis. Normal failures return exit code 2 and a bounded JSON error on stderr.

`report` never mutates a verified source run. It creates a new verified report-refresh run. Joblib files from outside a trusted local run must never be loaded.

## Docker

```bash
docker build -t finops-r:0.4.0 .
docker run --rm -v "$PWD/artifacts:/app/artifacts" finops-r:0.4.0 \
  run --config configs/smoke.json
```

The image runs as an unprivileged user and contains no source datasets or generated model artifacts.

## Locked experiment contract

The primary comparison is C5 versus C1 on entity-disjoint test cases at 20% simulated manual capacity. Promotion requires actual harm improvement and all safety, confidence-bound, evidence, workload, slice, stress, reproducibility and integrity gates to pass. A `HOLD` or fallback is a successful execution of the governance framework, not a fabricated model win.

The last locked 12,000-case reference study produced the following falsification result:

- C1 Macro-F1: 0.96310; residual harm: 74.75 per 1,000; manual intervention: 19.10%.
- C3 Macro-F1: 0.96558; residual harm: 68.22; C3-C1 Macro-F1 cluster-bootstrap interval: [-0.00189, 0.00717].
- C5 Macro-F1: 0.96449; residual harm: 136.93; paired C1-C5 harm interval: [-0.09271, -0.03547] per case.
- Stress harm: C1 141.45 versus C5 362.71 per 1,000; C5 manual intervention reached 36.30%.
- Decision: `HOLD`; production fallback: C1.

The valid conclusion is that RSOP was promising but not statistically confirmed, while the guarded C5 architecture was materially worse under the locked decision contract. FinOps-R correctly blocked promotion.

## Repository map

```text
src/finops_r/        package, CLI, data contracts, models, decisions, reports
configs/             smoke and locked full experiment configurations
data/fixtures/       small non-sensitive request fixture
docs/                architecture, datasets, runbook, limitations and evidence
tests/               unit, contract, security-boundary and end-to-end tests
scripts/             Linux and Windows qualification entry points
.github/              CI, security scanning and dependency automation
```

## Claim boundary and intended use

Use FinOps-R for offline controlled research, reproducibility demonstrations and local assessment workflows. Do not use it to execute trades, move funds, alter settlement instructions, make regulatory claims or estimate a financial institution's production performance. Synthetic, public proxy, stress and production data must remain clearly distinguished in every report.

## Primary references

- [SEC T+1 compliance guide](https://www.sec.gov/investment/settlement-cycle-small-entity-compliance-guide-15c6-1-15c6-2-204-2)
- [Python Packaging User Guide](https://packaging.python.org/en/latest/flow/)
- [GitHub Actions security hardening](https://docs.github.com/en/code-security/tutorials/secure-your-organization/protect-against-threats)
- [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [SEC Fails-to-Deliver data](https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data)

MIT licensed. See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md) and [CITATION.cff](CITATION.cff).
