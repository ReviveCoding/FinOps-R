# Integrated qualification Stage 5

Stage 5 binds two independently locked evidence families:

1. the 12,000-case TradeOps-Sim model and decision release result; and
2. the nine-scenario Stage 4 durable-orchestration qualification.

It does not retune either family. A software qualification PASS does not override a model
promotion HOLD. For the current locked result, C5 remains HOLD and C1 remains the production
reference.

The integrated gate requires verified model artifacts, an internally consistent production
bundle, zero unauthorized actions, Stage 4 core and LangGraph parity, zero duplicate simulation
effects, zero external executions, immutable released traces and exactly-once crash recovery.

The result is local E3 evidence. A final 0.4.0 release remains blocked until the checked-in CI
matrix runs from an immutable Git commit and produces retained hosted Linux, Windows, Python
3.11 and Python 3.12 evidence.

Run from the repository root:

```bash
PYTHONPATH=src python scripts/run_integrated_qualification_stage5.py \
  --model-run-dir artifacts/stage5_model_runs/<run-id> \
  --output-dir artifacts/qualification/integrated_stage5_v1r1

PYTHONPATH=src python scripts/verify_integrated_qualification_stage5.py \
  --artifact-dir artifacts/qualification/integrated_stage5_v1r1
```
