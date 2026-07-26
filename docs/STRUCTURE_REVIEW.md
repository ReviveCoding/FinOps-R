# Structural Effectiveness Review

## Verdict

The original FinOps-R safety boundary and release discipline are effective. Its initial implementation order was not. A Qwen planner, learned verifier, dense retrieval, temporal Transformer, and LangGraph runtime were all placed inside one C5 candidate. That makes a win difficult to attribute, raises compute and maintenance cost, and risks using language complexity where structured state is sufficient.

The implementation therefore freezes the product decision and safety invariants while revising the model ladder.

The rc3 re-review also found that the claimed atomic publication boundary ended one step too early: a verified OS-temporary tree was copied directly into its public name. The corrected boundary is now hidden same-filesystem copy, exact re-verification, and one atomic replace. This makes the artifact structure effective under concurrent readers, not only after a completed single-process run.

## What remains protected

- Observable case prefixes only.
- No external financial-system execution.
- Independent evidence and policy checks.
- Calibration, abstention, fixed review capacity, worst-slice gates, and reproducible lineage.
- Controlled-simulation claim boundary.

## What changed and why

| Original choice | Effectiveness finding | Implemented decision |
|---|---|---|
| Qwen in the main champion | Root cause and action are mostly structured; likely latency and correlated-error cost before demonstrated gain | Move to optional G1 challenger |
| LangGraph before direct pipeline | Persistence is valuable, but framework behavior can obscure core correctness | Implement direct typed orchestrator first; adapter is optional |
| Temporal Transformer as default advanced model | One case snapshot and modest synthetic data do not justify sequence complexity | Use HGB + ExtraTrees uncertainty ensemble |
| RSOP only as one extra feature block | Substantive but insufficient to express the user's broader specialty | Keep RSOP and add Action Evidence Envelope |
| One large C5 comparison | Component value cannot be identified | C0-C5 ladder and independent ablations |
| Fixed 50,000-case first run | Slows acceptance without changing contracts | 1,800-case smoke, 12,000-case full config, scalable generator |

## Technique selection gate

| Technique | Problem fit | Data fit | Expected gain | Reliability value | Compute | Product cost | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Histogram gradient boosting | High | High | High | Medium | Low | Low | Implement C1 |
| HGB + ExtraTrees ensemble | High | High | Medium | High through disagreement | Medium | Medium | Implement C2/C4 |
| RSOP | High for phase-dependent delay | High | Medium | High for state slices | Low | Low | Implement C3/C4/C5 |
| Action Evidence Envelope | High for candidate action safety | High | Medium | High | Low | Medium | Implement C4/C5 |
| Temperature calibration + EVR | High for review decisions | High | Medium | Very high | Low | Medium | Implement C5 |
| Temporal Transformer | Medium | Low to medium | Uncertain | Medium | High | High | Defer |
| Qwen3.5-4B QLoRA | Medium for ambiguous artifacts | No real text corpus yet | Uncertain | Uncertain | High | High | Optional G1 only |
| LangGraph persistence | High after multi-step HITL is proven | Not a model issue | Operational | High | Low | Medium | Optional adapter after direct E2E |

## Specialty mapping

### RSOP

RSOP is a domain extension of the user's PHT/reference-state principle. It replaces raw age as the sole progression coordinate with a train-only percentile of time spent in the same lifecycle state and product cohort. Hierarchical backoff and ablation make the contribution removable and testable.

### Action Evidence Envelope

AEE is inspired by Probability-Based Bbox and ABC Bbox. Instead of trusting one point action, it constructs a probability-ranked action region, expands it using independent evidence, and prunes it with policy constraints. It reports action-set recall, precision, F1, and overreach before selecting or abstaining. This transfers candidate aggregation, asymmetric expansion, and learned or rule-based correction into an operational action space.

## Official-source consistency

The SEC guide states that most covered transactions moved from T+2 to T+1 and notes operational and technology changes. Exact intraday deadlines in TradeOps-Sim are named scenario assumptions, not legal statements. Scikit-learn documents histogram gradient boosting and warns that classifier probabilities may require separate calibration. Qwen's official model card supports agentic use, but that does not demonstrate incremental value for this structured task. LangGraph checkpointing is appropriate for interruption and HITL persistence, but it is not required to establish predictive or decision efficacy.

## Promotion logic

The most complex candidate is not automatically selected. Predictive and production champions may differ. C5 must improve residual harm at fixed capacity and pass false-greenlight, evidence, workload, slice, artifact, and replay gates. Otherwise the implementation selects the best passing simpler candidate and records the failed hypotheses.
