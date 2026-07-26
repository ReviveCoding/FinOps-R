# FinOps-R pipeline map

| Stage | Implementation | Input contract | Output contract | Failure behavior | Test evidence |
|---|---|---|---|---|---|
| Configuration | `contracts.ProjectConfig`, `pipeline.load_config` | Strict JSON, finite values, bounded controls, relative output path | Normalized JSON-safe dict | Pydantic validation error, no run published | `test_contracts.py` |
| Data generation | `data.generate_cases` | `n_cases >= 200`, `n_entities >= 100`, fixed seed | Deterministic TradeOps-Sim frame | Explicit `ValueError` | `test_data.py` |
| Schema and leakage gate | `data.validate_cases` | DataFrame, including malformed and empty frames | Quality JSON with PASS or FAIL | Never silently coerces malformed input | `test_data.py` |
| Split | `data.entity_split` | Entity identifier | Train, validation, calibration, policy, test, stress | Hash-defined deterministic assignment | `test_data.py` |
| Specialty features | `specialty.RSOPTransformer` | Train-only healthy reference cohort | Bounded RSOP features and immutable reference manifest | Fails on insufficient reference data | `test_specialty.py` |
| Candidate training | `models.build_candidate` | Prefix-safe tabular or RSOP frame | C0 through C4 fitted models | Unknown candidate or unfitted use fails | Model paths covered by E2E |
| Calibration | `models.TemperatureScaler` | Calibration split only | Temperature-scaled probabilities | Bounded optimizer fallback to 1.0 | E2E and metrics |
| Evidence envelope | `evidence.build_action_envelopes` | Probabilities, observable evidence, policy state | Candidate, expanded, pruned and proposed actions | Conflict or missing evidence triggers HOLD | `test_evidence.py` |
| Fixed-capacity routing | `decision.route_decisions` | Probabilities, evidence gates, capacity, review economics | SIMULATE, REVIEW or HOLD per case | Hard HOLD consumes manual capacity; negative EVR is not filled | `test_decision.py` |
| Test evaluation | `pipeline.run_project`, `evaluation.py` | Entity-disjoint test cases and immutable class universe | Predictive, calibration, harm, latency, slice, envelope and paired entity-cluster intervals | Invalid bootstrap contract fails; undefined promotion gate yields HOLD | `test_evaluation.py`, E2E and full run |
| Stress evaluation | `pipeline.run_project` | Shifted entity-disjoint stress cases | `stress_comparison.csv` and stress gates | Harm or capacity regression blocks promotion | E2E and full run |
| Report | `reporting.generate_report` | Numeric artifacts only | Markdown, HTML, 8 PNG, 8 SVG, CSV underpinnings | Atomic text publication | E2E |
| Model publication | `artifacts.atomic_joblib`, `artifacts.publish_verified_tree`, `pipeline.run_project` | Selected fallback or promoted model | Compressed joblib plus external runtime contract | Hidden same-filesystem copy is verified then atomically exposed; failure leaves no public partial tree | `test_artifacts.py`, `test_e2e.py` |
| Artifact verification | `artifacts.verify_artifact_manifest` | Local run directory | Recomputed size and SHA-256 result | Corruption, symlink, traversal or duplicate path fails closed | `test_e2e.py` |
| Artifact loading | `pipeline.load_production_bundle` | Verified trusted-local run | Schema-checked bundle | Runtime mismatch rejected before deserialization | `test_contracts.py`, wheel reload |
| Inference | `pipeline.assess_payload_with_bundle` | Strict `CaseSnapshot` | Strict `Assessment` | Invalid schema rejected, external execution impossible | `test_contracts.py`, E2E, CLI reload |
| API | `api.create_app`, `cli serve` | Verified run and `CaseSnapshot` | Health and assessment endpoints | App creation fails if artifact is invalid | E2E plus Uvicorn startup smoke |

The run pointer is updated only after the staged directory is copied into a hidden same-filesystem sibling, fully verified, and atomically replaced into the public name. A failed run cannot overwrite `artifacts/releases/latest.json`.
