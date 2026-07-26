# Verification, policy, and orchestration Stage 3

## Scope

Stage 3 v1 implements the direct typed control path that follows retrieval and
planning. It is intentionally CPU-only and simulation-only. It does not add an
external execution adapter, claim bank-production performance, use a learned
verifier, or introduce a graph-framework dependency.

The implementation order is deliberate:

1. establish a framework-independent verifier, policy, trace, pause/resume, and
   replay contract;
2. validate fail-closed behavior with deterministic failure injection;
3. add an optional LangGraph adapter only after direct-pipeline equivalence can
   be tested;
4. evaluate a learned verifier only after a labeled adversarial verification
   benchmark is locked.

## Independent verifier contract

`verify_action_packet` independently binds the action packet to its request and
checks every citation against the retrieved hit and compiled policy card.

- case ID, decision time, and predicted root cause must match;
- a non-hold action must be permitted;
- every citation must have been retrieved as a policy card;
- the retrieved checksum must equal the compiled card checksum;
- the compiled card checksum must verify;
- the card must be effective at the decision time;
- every citation, including additional citations, must support the action;
- legal interpretation remains prohibited.

Any violation returns `REJECTED`. A valid explicit planner hold returns
`ABSTAINED`. Neither status can create a simulation receipt.

## Policy and routing contract

The only execution target is `TRADEOPS_SIM`, and every receipt records
`external_execution=false`.

- monitoring and low-risk operational requests may route to `SIMULATE`;
- corrections, retractions, funding or securities review, and compliance
  escalation require human review;
- missing review capacity converts a review-required action to `HOLD`;
- an external target, an external-execution request, a claim-boundary mismatch,
  verifier rejection, or planner abstention converts to `HOLD`.

Human review uses a deterministic review token. Replaying the same completed
resolution returns the byte-equivalent state. A different second resolution is
rejected.

## Audit trace

Every executed node records a stable sequence number plus SHA-256 fingerprints
of its logical input and output. The direct orchestrator validates route-specific
trace completeness before returning. Development artifacts are published only
after an independent manifest and metric recomputation passes.

## Development benchmark

`verification_policy_scenarios_stage3.json` is explicitly development-only. It
tests supported simulation, approval, rejection, capacity overflow, planner
abstention, external-target blocking, unsupported citation blocking, and
expired-policy blocking. It is not a qualification holdout and must not be used
to claim learned-verifier quality.
