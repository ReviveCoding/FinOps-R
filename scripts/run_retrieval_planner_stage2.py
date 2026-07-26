from __future__ import annotations

import argparse
import gc
import tempfile
import time
from pathlib import Path

import pandas as pd

from finops_r.artifacts import (
    artifact_manifest,
    atomic_csv,
    atomic_json,
    publish_verified_tree,
    sha256_file,
    verify_artifact_manifest,
)
from finops_r.json_safety import strict_json_dumps
from finops_r.planner_stage2 import (
    DEFAULT_DIAGNOSTIC_SEEDS,
    DEFAULT_EVALUATION_SEEDS,
    DEVELOPMENT_SCENARIOS_SHA256,
    NON_THINKING_MAX_NEW_TOKENS,
    QwenGenerator,
    QUALIFICATION_HOLDOUT_SHA256,
    STAGE2_PROMPT_REVISION,
    THINKING_MAX_NEW_TOKENS,
    exact_snapshot_object,
    load_planner_scenarios,
    profile_planner,
)
from finops_r.policy_cards import compile_policy_cards, load_compiled_policy_cards
from finops_r.retrieval_stage2 import (
    BGE_MODEL_ID,
    BGE_MODEL_REVISION,
    QWEN_MODEL_ID,
    QWEN_MODEL_REVISION,
    STAGE2_BENCHMARK_SHA256,
    BGEEncoder,
    build_stage2_documents,
    compare_retrievers,
    load_stage2_benchmark,
    runtime_identity,
)
from finops_r.stage2_verification import (
    PLANNER_QUALITY_GATES,
    STAGE2_IMPLEMENTATION_REVISION,
    planner_quality_failures,
    verify_stage2_artifact,
)


def _require_gpu_runtime() -> dict[str, object]:
    identity = runtime_identity()
    if identity.get("cuda_available") is not True:
        raise RuntimeError("STAGE2_CUDA_REQUIRED")
    gpu = identity.get("gpu")
    if not isinstance(gpu, dict) or int(gpu.get("total_memory_bytes", 0)) < 12 * 1024**3:
        raise RuntimeError("STAGE2_REQUIRES_AT_LEAST_12_GIB_VRAM")
    packages = identity.get("packages")
    if not isinstance(packages, dict) or packages.get("transformers") is None:
        raise RuntimeError("STAGE2_TRANSFORMERS_REQUIRED")
    return identity


def main() -> int:
    parser = argparse.ArgumentParser(description="Run locked FinOps-R retrieval/planner Stage 2")
    parser.add_argument(
        "--source-cards", type=Path, default=Path("data/policy_cards/source_cards.json")
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/policy_cards/retrieval_benchmark_stage2.json"),
    )
    parser.add_argument(
        "--development-scenarios",
        type=Path,
        default=Path("data/policy_cards/planner_scenarios_stage2.json"),
    )
    parser.add_argument(
        "--qualification-holdout",
        type=Path,
        default=Path("data/policy_cards/planner_scenarios_stage2_v5_holdout.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/components/retrieval_planner_stage2"),
    )
    parser.add_argument("--model-cache", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"STAGE2_OUTPUT_ALREADY_EXISTS: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = _require_gpu_runtime()
    benchmark_path = args.benchmark.resolve()
    development_path = args.development_scenarios.resolve()
    holdout_path = args.qualification_holdout.resolve()
    source_cards_path = args.source_cards.resolve()
    benchmark_source_text = benchmark_path.read_text(encoding="utf-8")
    development_source_text = development_path.read_text(encoding="utf-8")
    holdout_source_text = holdout_path.read_text(encoding="utf-8")
    source_cards_source_text = source_cards_path.read_text(encoding="utf-8")
    benchmark = load_stage2_benchmark(benchmark_path)
    load_planner_scenarios(development_path)
    holdout = load_planner_scenarios(holdout_path)
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}.staging."
    ) as temporary:
        staging = Path(temporary)
        policy_manifest = compile_policy_cards(source_cards_path, staging)
        cards = load_compiled_policy_cards(staging / "policy_cards.json")
        documents = build_stage2_documents(cards, benchmark)

        import torch

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        bge_load_start = time.perf_counter()
        encoder = BGEEncoder(
            cache_dir=args.model_cache.resolve(),
            query_instruction=benchmark.query_instruction,
            local_files_only=args.local_files_only,
        )
        bge_load_seconds = time.perf_counter() - bge_load_start
        encoder.warmup()
        candidates, retrieval_records = compare_retrievers(documents, benchmark, encoder)
        bge_peak_vram = encoder.peak_vram_bytes()
        for candidate in candidates:
            candidate["bge_peak_vram_bytes"] = bge_peak_vram if candidate["method"] != "BM25" else 0
        atomic_csv(staging / "retrieval_candidate_comparison.csv", pd.DataFrame(candidates))
        atomic_json(
            staging / "retrieval_query_results.json",
            {
                "schema_version": "2.0",
                "benchmark_id": benchmark.benchmark_id,
                "benchmark_sha256": STAGE2_BENCHMARK_SHA256,
                "query_count": len(benchmark.queries),
                "future_evidence_count": sum(
                    item["future_evidence_count"] for item in retrieval_records
                ),
                "records": retrieval_records,
            },
        )
        del encoder
        gc.collect()
        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats(0)
        qwen_load_start = time.perf_counter()
        generator = QwenGenerator(
            cache_dir=args.model_cache.resolve(), local_files_only=args.local_files_only
        )
        qwen_load_seconds = time.perf_counter() - qwen_load_start
        generator.warmup()
        planner_metrics, planner_records = profile_planner(
            holdout,
            retrieval_records,
            documents,
            generator,
            production_seeds=DEFAULT_EVALUATION_SEEDS,
            diagnostic_seeds=DEFAULT_DIAGNOSTIC_SEEDS,
        )
        atomic_csv(staging / "planner_mode_metrics.csv", pd.DataFrame(planner_metrics))
        planner_quality_gate_failures = planner_quality_failures(pd.DataFrame(planner_metrics))
        if sum(item["post_gate_unauthorized_action_count"] for item in planner_metrics) != 0:
            planner_quality_gate_failures.append("POST_GATE_UNAUTHORIZED_ACTION")
        atomic_json(
            staging / "planner_scenario_results.json",
            {
                "schema_version": "4.0",
                "benchmark_id": holdout.benchmark_id,
                "qualification_holdout_sha256": QUALIFICATION_HOLDOUT_SHA256,
                "unique_scenario_count": len(holdout.scenarios),
                "seed_count_by_mode": {
                    "NON_THINKING": len(DEFAULT_EVALUATION_SEEDS),
                    "THINKING": len(DEFAULT_DIAGNOSTIC_SEEDS),
                },
                "run_count_by_mode": {
                    "NON_THINKING": (len(holdout.scenarios) * len(DEFAULT_EVALUATION_SEEDS)),
                    "THINKING": (len(holdout.scenarios) * len(DEFAULT_DIAGNOSTIC_SEEDS)),
                },
                "records": planner_records,
            },
        )
        atomic_json(
            staging / "evaluation_input_snapshot.json",
            {
                "schema_version": "1.0",
                "retrieval_benchmark_sha256": STAGE2_BENCHMARK_SHA256,
                "development_scenarios_sha256": DEVELOPMENT_SCENARIOS_SHA256,
                "qualification_holdout_sha256": QUALIFICATION_HOLDOUT_SHA256,
                "source_cards_sha256": sha256_file(source_cards_path),
                "retrieval_benchmark_source_text": benchmark_source_text,
                "development_scenarios_source_text": development_source_text,
                "qualification_holdout_source_text": holdout_source_text,
                "source_cards_source_text": source_cards_source_text,
                "retrieval_benchmark": exact_snapshot_object(benchmark_source_text),
                "development_scenarios": exact_snapshot_object(development_source_text),
                "qualification_holdout": exact_snapshot_object(holdout_source_text),
                "holdout_labels_excluded_from_prompt": True,
                "holdout_tuning_prohibited": True,
            },
        )

        runtime_manifest = {
            "schema_version": "4.0",
            "implementation_revision": STAGE2_IMPLEMENTATION_REVISION,
            "runtime": runtime,
            "benchmark_sha256": STAGE2_BENCHMARK_SHA256,
            "development_scenarios_sha256": DEVELOPMENT_SCENARIOS_SHA256,
            "qualification_holdout_sha256": QUALIFICATION_HOLDOUT_SHA256,
            "source_cards_sha256": sha256_file(source_cards_path),
            "compiled_policy_cards_sha256": policy_manifest["compiled_sha256"],
            "models": {
                "bge_model_id": BGE_MODEL_ID,
                "bge_revision": BGE_MODEL_REVISION,
                "bge_load_seconds": bge_load_seconds,
                "qwen_model_id": QWEN_MODEL_ID,
                "qwen_revision": QWEN_MODEL_REVISION,
                "qwen_tokenizer_revision": QWEN_MODEL_REVISION,
                "qwen_load_seconds": qwen_load_seconds,
                "qwen_dtype": "bfloat16",
            },
            "generation": {
                "prompt_revision": STAGE2_PROMPT_REVISION,
                "thinking": {
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "max_new_tokens": THINKING_MAX_NEW_TOKENS,
                    "role": "DIAGNOSTIC_ONLY",
                },
                "non_thinking": {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "max_new_tokens": NON_THINKING_MAX_NEW_TOKENS,
                    "role": "PRODUCTION_CANDIDATE",
                },
                "production_evaluation_seeds": list(DEFAULT_EVALUATION_SEEDS),
                "diagnostic_evaluation_seeds": list(DEFAULT_DIAGNOSTIC_SEEDS),
            },
            "external_execution": False,
            "chain_of_thought_persisted": False,
        }
        atomic_json(staging / "runtime_manifest.json", runtime_manifest)
        summary = {
            "schema_version": "4.0",
            "status": "PASS" if not planner_quality_gate_failures else "FAIL",
            "elapsed_seconds": time.perf_counter() - started,
            "retrieval_query_count": len(benchmark.queries),
            "planner_unique_scenario_count": len(holdout.scenarios),
            "planner_seed_count_by_mode": {
                "NON_THINKING": len(DEFAULT_EVALUATION_SEEDS),
                "THINKING": len(DEFAULT_DIAGNOSTIC_SEEDS),
            },
            "planner_run_count_by_mode": {
                "NON_THINKING": (len(holdout.scenarios) * len(DEFAULT_EVALUATION_SEEDS)),
                "THINKING": (len(holdout.scenarios) * len(DEFAULT_DIAGNOSTIC_SEEDS)),
            },
            "post_gate_unauthorized_action_count": sum(
                item["post_gate_unauthorized_action_count"] for item in planner_metrics
            ),
            "planner_quality_gates": PLANNER_QUALITY_GATES,
            "planner_quality_gate_failures": planner_quality_gate_failures,
            "production_claim_permitted": False,
            "production_candidate_mode": "NON_THINKING",
            "thinking_mode_status": "DIAGNOSTIC_ONLY",
            "tuning_on_qualification_holdout": False,
            "development_split_tuning_permitted": True,
            "claim_scope": (
                "Component-level private qualification holdout and local zero-shot "
                "multi-seed profiling only; "
                "not a production or legal-performance claim."
            ),
        }
        atomic_json(staging / "stage2_summary.json", summary)
        atomic_json(staging / "artifact_manifest.json", artifact_manifest(staging))
        verify_artifact_manifest(staging)
        publish_verified_tree(staging, output)

    verification = verify_stage2_artifact(output)
    print(
        strict_json_dumps(
            {
                "output": str(output),
                "retrieval_candidates": candidates,
                "planner_modes": planner_metrics,
                "verification": verification,
                "status": verification["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if verification["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
