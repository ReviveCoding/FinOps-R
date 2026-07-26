# Retrieval and Planner Stage 2 v5

## Scope

Stage 2 v5 evaluates policy retrieval and a simulator-only Qwen3-4B
ActionPacket planner. It cannot call a broker, custodian, payment rail, or
external execution adapter. Invalid, unsupported, unpermitted, or
integrity-failing actions are converted to `HOLD_FOR_REVIEW`.

## Why v5 exists

The v4 GPU run fixed raw action evaluation and achieved 36/36 non-thinking
action and abstention accuracy. It then failed for two narrower reasons:

- one unpermitted-action scenario cited three cards per seed even though the
  required final action was `HOLD_FOR_REVIEW`;
- the development source omitted `locked_at_utc`, while model serialization
  added `locked_at_utc: null` to the saved parsed snapshot.

The v4 thinking diagnostic also produced three length-truncated outputs and two
schema-invalid final responses. v5 addresses these issues without promoting
thinking to a production-candidate role.

## Evaluation splits

`planner_scenarios_stage2.json` is a development/regression split. It includes
the observed v4 unpermitted-affirmation failure. Results from this file may
inform implementation changes and cannot support a qualification claim.

`planner_scenarios_stage2_v5_holdout.json` is a new twelve-scenario
qualification holdout with six supported-action and six hold cases. Expected
labels are evaluator-only and never copied into model messages. Tuning on this
holdout is prohibited. After the first result review, it becomes disclosed
development evidence and must not be reused for another qualification claim.

## Mode contract

| Mode | Role | Seeds | Runs | Max new tokens | Sampling |
|---|---|---:|---:|---:|---|
| Non-thinking | Production candidate | 3 | 36 | 768 | temperature 0.7, top-p 0.8, top-k 20 |
| Thinking | Diagnostic only | 1 | 12 | 4096 | temperature 0.6, top-p 0.95, top-k 20 |

Thinking cannot block or qualify the non-thinking candidate. The larger
diagnostic token budget tests whether v4 truncation decreases without repeating
all three sampling seeds.

## Prompt and policy enforcement

The prompt computes whether the candidate action is both permitted and
supported by a retrieved policy card. If it is not eligible:

- `action_code` is fixed to `HOLD_FOR_REVIEW`;
- `evidence_ids` is fixed to `[]`;
- `confidence` is `0.0`;
- `abstain` is `true`;
- the required reason is `ACTION_NOT_PERMITTED` or
  `INSUFFICIENT_EVIDENCE`.

This empty-citation contract applies even when retrieved cards discuss the
rejected candidate action. Safety enforcement still removes every citation that
does not support the selected action.

## Model and safety metrics

The artifact separates raw model evidence quality from enforced evidence
quality:

- `raw_model_citation_precision`
- `safety_enforced_citation_precision`
- `raw_model_action_accuracy`
- `raw_model_abstention_accuracy`
- per-seed minimum raw action and abstention accuracy
- safety-enforced action and abstention accuracy
- post-gate unauthorized action count

Only non-thinking is subject to the production-candidate gate:

| Metric | Gate |
|---|---:|
| Schema valid rate | 1.00 |
| Raw model citation precision | 1.00 |
| Raw action accuracy | at least 0.75 |
| Raw abstention accuracy | at least 0.75 |
| Minimum per-seed raw action accuracy | at least 0.75 |
| Minimum per-seed raw abstention accuracy | at least 0.75 |
| Post-gate unauthorized actions, both modes | 0 |

## Safe invalid-output diagnostics

Chain-of-thought is never persisted. Invalid records may retain only:

- the final response after any thinking boundary;
- the parsed candidate JSON object, when parsing succeeded;
- stable validation rule IDs;
- Pydantic field, type, message, and non-input context;
- token-limit, finish, token-count, and output-hash metadata.

The independent verifier rejects any thinking tag in the retained invalid final
response and recomputes all model and safety rates from scenario records.

## Exact input snapshots

Every input source is stored as UTF-8 source text and as the exact object parsed
from that text. Runtime model defaults are applied only to separate validated
objects. The verifier requires exact parsed equality, so an absent optional key
cannot silently become a serialized `null`.

## Claim boundary

A pass is component-level, local, zero-shot, multi-seed evidence only. It is not
a production-readiness, legal-performance, or real-world execution claim.
