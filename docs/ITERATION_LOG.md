# Implementation Iteration Log

## 1. Root-cause loss was not the product loss

The first implementation charged decision harm when the predicted root-cause label was wrong, even if the evidence layer proposed the correct operational action. The product objective is the next action, so decision cost and false-greenlight metrics were changed to action correctness. Predictive root-cause metrics remain separate diagnostics.

## 2. Evidence expansion must not silently override the learned action

Early AEE logic replaced `NO_ACTION` with the strongest rule-evidence action. Because simulator observations include realistic noise, this increased false positive actions. The corrected AEE expands a candidate set for coverage, but conflicting evidence triggers review or HOLD. It does not silently take control.

## 3. Raw harm dominated review allocation

The first EVR score over-prioritized confident, correctly predicted critical cases. This wasted the fixed review budget. The corrected score makes calibrated error and ensemble disagreement primary, uses data-quality risk explicitly, and limits harm to a tie-breaker.

## 4. Evidence coverage is not evidence precision

The initial metric mislabeled the fraction of supported cases as precision. The corrected contract reports:

- Evidence coverage: fraction passing independent evidence support.
- Evidence precision: action correctness among evidence-supported cases.
- Action-envelope recall, precision, F1, mean size, and overreach separately.

## 5. The complex candidate was not operationally best

On the final 12,000-case run, C3 had the highest Macro-F1 and lowest residual-harm point estimate, while C5 had the lowest multiclass Brier score. C3's advantage was not statistically confirmed, and C5's additional HOLD cost made total residual harm materially worse than C1. The framework therefore kept distinct labels:

- Predictive champion: C3.
- Safety champion: C3.
- Production reference: C1.

## 6. Supported conclusion

The original broad architecture is useful as a research portfolio, but ineffective as a default deployment recipe. Its components should be promoted independently. RSOP remains a plausible but unconfirmed predictive contribution. AEE shows high coverage and verified-action precision, but requires better evidence generation or lower-cost review handling before promotion. Qwen3.5 and LangGraph remain optional challengers because the structured CPU models already reach Macro-F1 near 0.966 on the controlled benchmark.

## 7. Release-hardening iterations

The release candidate was then attacked at its input, publication, packaging and serving boundaries. Successive rounds closed run-ID collisions, publication cleanup contradictions, portable-path collisions, interrupt residue, permissive numeric coercion, incomplete split/label validation, ISO source replacement, tied-score review overflow, semantically inconsistent model bundles and model-load TOCTOU. Each issue received a regression test before the entire qualification restarted.

The retained final loop contains 70 tests and 36 subtests, two clean source passes, a complete installed-wheel pass, independent wheel/sdist installations, two 1,200-case installed-wheel runs, 59-file verification per run and three TCP API cycles. No new defect was found in that exercised scope. C5 still remains `HOLD`; release hardening did not alter the scientific decision to retain C1.
