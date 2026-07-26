# Stage 4 orchestration qualification

## Claim boundary

Stage 4 remains local, offline and simulation-only. The only effect target is
`TRADEOPS_SIM`, and every receipt must retain `external_execution=false`.
Neither the deterministic verifier nor the learned verifier can call a broker,
payment, settlement or bank-production interface.

The attached FUNSD, EDGAR, Kleister, HMDA, DocVQA, CFPB, SEC financial
statement, M5, Instacart and Criteo datasets are excluded from the canonical
Stage 4 run. They remain possible auxiliary benchmark or sensitivity inputs
under their existing dataset-governance restrictions.

## Execution paths

`run_stage4_reference` is the framework-independent semantic reference. The
optional LangGraph graph uses the same typed input and result contracts. Its
human-review node calls one JSON-serializable interrupt and resumes with the
same thread ID, review token, reviewer ID and decision.

The durable reference runner stores its state in SQLite with:

- full synchronous commits and write-ahead logging;
- an input-bound thread identity;
- sequence-checked, SHA-256 hash-chained checkpoints;
- a unique idempotency key for each simulated effect;
- restart recovery from the last committed input, waiting-review or review-input
  checkpoint;
- an immutable released trace head.

The simulation effect is committed before the final state checkpoint. The
qualification intentionally crashes in that gap and verifies that replay
returns the original receipt without creating a second effect.

## Frozen learned verifier

The learned verifier is a small logistic risk scorer trained only on the
engineered development feature snapshot
`learned_verifier_training_stage4.json`. Its threshold is selected on the
separate engineered development snapshot
`learned_verifier_threshold_stage4.json` under a zero-false-negative constraint
for that snapshot. The freeze script deterministically regenerates the JSON
artifact and binds it to both source hashes.

Its mode is `VETO_ONLY`: a flag changes the effective route to `HOLD`. It cannot
turn a deterministic rejection, abstention or hold into review or simulation.
The engineered snapshots are too small to support a general predictive-quality
claim.

## Locked qualification gates

The nine-scenario locked fixture covers three low-risk simulated requests,
review approval, review rejection, unavailable review capacity, an external
target request, an unsupported citation and planner abstention.

Qualification requires:

- reference and durable-path parity for every scenario;
- reference and LangGraph-path parity for every scenario;
- correct initial and final routes;
- exactly one effect for each final simulation and zero for every hold;
- zero external-execution receipts;
- successful recovery after an injected post-effect crash;
- one retained effect after crash replay;
- rejected writes after trace release;
- independent artifact-manifest verification.

Run:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -r requirements-orchestration-lock.txt
python -m pip install --no-deps -e .
python scripts/run_orchestration_qualification_stage4.py
python scripts/verify_orchestration_qualification_stage4.py
```

LangGraph documents that a resumed interrupt restarts its containing node and
therefore requires pre-interrupt side effects to be idempotent. Stage 4 places
the review interrupt before the simulated effect and separately enforces effect
idempotency at the SQLite ledger boundary.
