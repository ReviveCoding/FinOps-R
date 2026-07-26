from __future__ import annotations

from pathlib import Path

from finops_r.stage3_benchmark import run_stage3_benchmark, verify_stage3_artifact


def test_stage3_development_benchmark_rebuilds_and_verifies(tmp_path: Path) -> None:
    repository_root = Path(__file__).parents[1]
    output = tmp_path / "stage3"
    result = run_stage3_benchmark(
        source_cards_path=repository_root / "data/policy_cards/source_cards.json",
        scenarios_path=(
            repository_root / "data/policy_cards/verification_policy_scenarios_stage3.json"
        ),
        output_dir=output,
    )
    independently_verified = verify_stage3_artifact(output)
    assert result == independently_verified
    assert result["status"] == "PASS"
    assert result["scenario_count"] == 8
    assert result["trace_complete_rate"] == 1.0
    assert result["external_execution_count"] == 0
