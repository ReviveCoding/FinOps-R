# Operational Runbook

## Readiness

1. Confirm `artifact_integrity.json` is PASS.
2. Confirm the selected run has a release decision and production bundle.
3. Run `finops-r inspect --run-dir <run>`.
4. Run `finops-r assess` with the sample fixture.
5. Verify the response says `controlled simulation`.

## Failure response

- Invalid request: return the structured validation error.
- Missing artifact: do not reconstruct it silently; rerun the versioned workflow.
- Policy or evidence failure: HOLD.
- Model-load failure: use C0/C1 only in an offline rerun; do not auto-simulate.
- Drift or unexpected category: review or HOLD.
- Any external-execution attempt: block and record `EXECUTION_PROHIBITED`.

## Rollback

Only a previously promoted run may become the local champion:

```bash
finops-r rollback --run-id <promoted_run_id>
```

Rollback changes a local pointer atomically. It does not delete the failed run or its evidence.

## Retraining trigger

Start a new versioned experiment when target distribution, missingness, RSOP backoff rate, ECE, critical false greenlights, evidence precision, or latency crosses a preregistered boundary. Never refit on the locked test.
