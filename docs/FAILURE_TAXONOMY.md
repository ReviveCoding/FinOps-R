# Failure Taxonomy

| Code | Layer | Meaning | Safe response |
|---|---|---|---|
| DQ_SCHEMA | Data | Invalid or missing input field | Reject input |
| DQ_LEAKAGE | Data | Entity, future-event, or preprocessing leakage | Fail experiment |
| RSOP_REFERENCE | Specialty | Missing or unstable reference cohort | Hierarchical backoff or remove RSOP |
| MODEL_LOW_CONFIDENCE | Model | Calibrated confidence below operating point | Review |
| MODEL_DISAGREEMENT | Model | Ensemble components disagree | Review |
| AEE_NO_COVERAGE | Specialty | Action envelope misses supported action | Hold |
| AEE_OVERREACH | Specialty | Too many actions survive pruning | Review |
| EVIDENCE_MISSING | Verification | Proposed action lacks observable support | Hold |
| POLICY_BLOCK | Policy | Action violates state preconditions | Hold |
| OOD_SHIFT | Reliability | Stress or unknown category detected | Review or hold |
| ARTIFACT_MISMATCH | Product | Checksum, model, config, or index mismatch | Refuse load and rollback |
| EXECUTION_PROHIBITED | Product | Attempt to call an external financial system | Block permanently |

Failures are emitted as reason codes in predictions, traces, release gates, and the single-case assessment response.
