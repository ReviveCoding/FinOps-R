from __future__ import annotations

import hashlib
import io
import re
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import __version__
from .artifacts import (
    ArtifactVerificationError,
    artifact_manifest,
    atomic_csv,
    atomic_joblib,
    atomic_json,
    atomic_text,
    cleanup_empty_joblib_staging,
    config_hash,
    publish_verified_tree,
    sha256_file,
    verify_artifact_manifest,
)
from .constants import ACTION_BY_CAUSE, CAUSES
from .contracts import Assessment, CaseSnapshot, ProjectConfig
from .data import dataframe_fingerprint, generate_cases, validate_cases
from .decision import ReviewPolicy, decision_metrics, fit_review_threshold, route_decisions
from .evaluation import (
    benchmark_latency,
    confusion_records,
    hardware_metadata,
    paired_bootstrap_delta,
    paired_bootstrap_macro_f1_delta,
    predictive_metrics,
    slice_metrics,
    wilson_upper,
)
from .evidence import allowed_actions, build_action_envelopes, envelope_metrics
from .json_safety import (
    StrictJSONError,
    require_json_object,
    strict_json_dumps,
    strict_json_loads,
)
from .models import TemperatureScaler, build_candidate
from .reporting import calibration_table, generate_report
from .specialty import RSOPTransformer


CANDIDATE_META = {
    "C0": {"type": "sanity", "algorithm": "deterministic rules", "specialty": []},
    "C1": {"type": "industry", "algorithm": "histogram gradient boosting", "specialty": []},
    "C2": {
        "type": "advanced",
        "algorithm": "HGB and ExtraTrees uncertainty ensemble",
        "specialty": [],
    },
    "C3": {"type": "specialty", "algorithm": "histogram gradient boosting", "specialty": ["RSOP"]},
    "C4": {
        "type": "hybrid",
        "algorithm": "HGB and ExtraTrees uncertainty ensemble",
        "specialty": ["RSOP", "Action Evidence Envelope"],
    },
    "C5": {
        "type": "production",
        "algorithm": "C4 with temperature calibration and EVR routing",
        "specialty": ["RSOP", "Action Evidence Envelope", "reliability guard"],
    },
}

RUNTIME_PACKAGES = ("joblib", "numpy", "pandas", "pydantic", "scikit-learn", "scipy")
_PYTHON_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


def _log(event: str, **fields: Any) -> None:
    print(
        strict_json_dumps(
            {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **fields},
            sort_keys=True,
        ),
        flush=True,
    )


def runtime_contract() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "packages": {name: version(name) for name in RUNTIME_PACKAGES},
    }


def verify_runtime_contract(contract: dict[str, Any]) -> None:
    packages = contract.get("packages")
    expected_python = contract.get("python")
    if (
        set(contract) != {"schema_version", "python", "packages"}
        or contract.get("schema_version") != "1.0"
        or not isinstance(expected_python, str)
        or _PYTHON_VERSION_PATTERN.fullmatch(expected_python) is None
        or not isinstance(packages, dict)
        or set(packages) != set(RUNTIME_PACKAGES)
        or any(not isinstance(value, str) or not value for value in packages.values())
    ):
        raise ArtifactVerificationError("runtime contract schema is incompatible")
    current_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if expected_python.split(".")[:2] != current_python.split(".")[:2]:
        raise ArtifactVerificationError(
            f"Python runtime mismatch: artifact={expected_python}, current={current_python}"
        )
    current_packages = {name: version(name) for name in RUNTIME_PACKAGES}
    mismatches = {
        name: {"artifact": packages[name], "current": current_packages[name]}
        for name in RUNTIME_PACKAGES
        if packages[name] != current_packages[name]
    }
    if mismatches:
        raise ArtifactVerificationError(f"package runtime mismatch: {mismatches}")


def load_config(path: Path) -> dict[str, Any]:
    payload = require_json_object(
        strict_json_loads(path.read_text(encoding="utf-8")), source=f"project config {path}"
    )
    return ProjectConfig.model_validate(payload).model_dump(mode="json")


def _run_identity(config: dict[str, Any]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    nonce = uuid.uuid4().hex
    return f"finopsr_{timestamp}_{config_hash(config)[:10]}_{nonce}"


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write_cards(
    run_dir: Path, quality: dict[str, Any], metrics: dict[str, Any], release: dict[str, Any]
) -> list[str]:
    cards = run_dir / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    data_card = f"""# Data Card

- Dataset: TradeOps-Sim v0.1.0
- Rows: {quality["row_count"]}
- Features: {quality["feature_count"]}
- Critical cases: {quality["critical_count"]}
- Entity split leakage: {quality["entity_split_leakage_count"]}
- Validation status: {quality["status"]}

All rows are controlled synthetic case snapshots. Aggregate public market sources may inform scenarios but are not case labels.
"""
    model_card = """# Model Card

FinOps-R compares C0-C5 under one entity-disjoint contract. C1 is the strong industry baseline. C2 adds an uncertainty ensemble. C3 isolates RSOP. C4 combines the ensemble with RSOP and the Action Evidence Envelope. C5 adds temperature calibration, hard evidence and policy gates, and Expected Value of Review routing.

Intended use is offline controlled benchmarking and local demonstration. External financial execution is prohibited.
"""
    evaluation_card = f"""# Evaluation Card

- Primary split: entity-disjoint test
- Primary decision metric: residual harm per 1,000 cases at fixed review capacity
- Review capacity: {release["review_capacity"]:.0%}
- Bootstrap: paired entity-cluster percentile resampling with a fixed label universe
- Predictive champion: {release["predictive_champion"]}
- Production champion: {release["production_champion"]}

Worst slices, calibration, false greenlights, latency, failure cases, and action-envelope coverage are reported with average metrics.
"""
    release_card = f"""# Release Card

- Status: {release["status"]}
- Rationale: {release["rationale"]}
- Production champion: {release["production_champion"]}
- Claim boundary: controlled TradeOps-Sim only

Gate details are stored in `release_decision.json`. A HOLD result is a valid completion, not a failed implementation.
"""
    values = {
        "data_card.md": data_card,
        "model_card.md": model_card,
        "evaluation_card.md": evaluation_card,
        "release_card.md": release_card,
    }
    for name, content in values.items():
        atomic_text(cards / name, content)
    return [str((cards / name).relative_to(run_dir)) for name in values]


def _calibrated_candidate_probabilities(
    model: Any, scaler: TemperatureScaler, frame: pd.DataFrame
) -> np.ndarray:
    return scaler.transform(model.predict_proba(frame))


def _candidate_frames(
    raw: dict[str, pd.DataFrame], rsop: RSOPTransformer
) -> dict[str, dict[str, pd.DataFrame]]:
    transformed = {name: rsop.transform(frame) for name, frame in raw.items()}
    return {
        candidate_id: (transformed if candidate_id in {"C3", "C4", "C5"} else raw)
        for candidate_id in CANDIDATE_META
    }


def _latency_function(
    candidate_id: str,
    model: Any,
    c1_scaler: TemperatureScaler,
    c5_scaler: TemperatureScaler,
    rsop: RSOPTransformer,
) -> Callable[[pd.DataFrame], np.ndarray]:
    if candidate_id == "C5":

        def predict(sample: pd.DataFrame) -> np.ndarray:
            return c5_scaler.transform(model.predict_proba(rsop.transform(sample)))

    elif candidate_id in {"C3", "C4"}:

        def predict(sample: pd.DataFrame) -> np.ndarray:
            return model.predict_proba(rsop.transform(sample))

    elif candidate_id == "C1":

        def predict(sample: pd.DataFrame) -> np.ndarray:
            return c1_scaler.transform(model.predict_proba(sample))

    else:
        predict = model.predict_proba
    return predict


def _evaluate_candidate(
    candidate_id: str,
    model: Any,
    frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    probabilities: np.ndarray,
    classes: np.ndarray,
    config: dict[str, Any],
    disagreement: np.ndarray | None,
    envelopes: pd.DataFrame | None,
    policy: ReviewPolicy,
    fit_seconds: float,
    latency_fn: Callable[[pd.DataFrame], np.ndarray],
    minimum_review_score: float,
) -> dict[str, Any]:
    predictions = classes[np.argmax(probabilities, axis=1)]
    routes = route_decisions(
        raw_frame,
        predictions,
        probabilities,
        policy=policy,
        review_capacity=float(config["review_capacity"]),
        human_correction_probability=float(config["human_correction_probability"]),
        review_cost=float(config["review_cost"]),
        disagreement=disagreement,
        envelopes=envelopes,
        minimum_review_score=minimum_review_score,
    )
    if envelopes is not None:
        proposed_actions = envelopes["proposed_action"].to_numpy()
        evidence_ids = envelopes["evidence_ids"].to_numpy()
        verifier = envelopes["verifier_status"].to_numpy()
        policy_status = envelopes["policy_status"].to_numpy()
    else:
        proposed_actions = np.asarray([ACTION_BY_CAUSE[str(value)] for value in predictions])
        evidence_ids = np.full(len(raw_frame), "", dtype=object)
        verifier = np.full(len(raw_frame), "NOT_APPLICABLE", dtype=object)
        policy_status = np.full(len(raw_frame), "NOT_APPLICABLE", dtype=object)
    decision, costs = decision_metrics(
        raw_frame,
        proposed_actions,
        routes,
        float(config["human_correction_probability"]),
        float(config["review_cost"]),
        float(config.get("hold_cost_fraction", 0.10)),
        float(config.get("hold_fixed_cost", 0.10)),
    )
    predictive = predictive_metrics(
        raw_frame["root_cause"].to_numpy(), predictions, probabilities, classes
    )
    slices = slice_metrics(raw_frame, predictions, candidate_id)
    worst_slice = float(slices["macro_f1"].min()) if len(slices) else 0.0
    evidence_values = (
        envelope_metrics(envelopes, raw_frame["oracle_action"]) if envelopes is not None else {}
    )
    operational = benchmark_latency(latency_fn, raw_frame)
    operational["training_seconds"] = float(fit_seconds)
    metrics = {
        "candidate_id": candidate_id,
        **predictive,
        **decision,
        **evidence_values,
        **operational,
        "worst_slice_macro_f1": worst_slice,
        "critical_false_greenlight_upper95": wilson_upper(
            int(decision["critical_false_greenlight_count"]), int(decision["critical_case_count"])
        ),
    }
    prediction_frame = pd.DataFrame(
        {
            "case_id": raw_frame["case_id"].to_numpy(),
            "candidate_id": candidate_id,
            "true_cause": raw_frame["root_cause"].to_numpy(),
            "predicted_cause": predictions,
            "oracle_action": raw_frame["oracle_action"].to_numpy(),
            "proposed_action": proposed_actions,
            "confidence": probabilities.max(axis=1),
            "correct": (predictions == raw_frame["root_cause"].to_numpy()).astype(int),
            "action_correct": (proposed_actions == raw_frame["oracle_action"].to_numpy()).astype(
                int
            ),
            "critical": raw_frame["critical"].to_numpy(),
            "product": raw_frame["product"].to_numpy(),
            "state": raw_frame["state"].to_numpy(),
            "evidence_ids": evidence_ids,
            "verifier_status": verifier,
            "policy_status": policy_status,
            "route": routes["route"].to_numpy(),
            "route_reason": routes["route_reason"].to_numpy(),
            "residual_cost": costs,
        }
    )
    return {
        "metrics": metrics,
        "predictions": prediction_frame,
        "slices": slices,
        "costs": costs,
        "routes": routes,
        "envelopes": envelopes,
        "probabilities": probabilities,
        "classes": classes,
    }


def run_project(config_path: Path, output_root_override: Path | None = None) -> Path:
    """Run in an isolated temporary tree and remove all staging on every exit path."""
    resolved_config = config_path.resolve()
    config = load_config(resolved_config)
    run_id = _run_identity(config)
    staging_owner = f"{run_id}.{uuid.uuid4().hex[:12]}"
    staging = Path(tempfile.mkdtemp(prefix=f".{staging_owner}.staging."))
    try:
        return _run_project(resolved_config, output_root_override, run_id, staging)
    finally:
        if staging.is_symlink():
            raise ArtifactVerificationError("run staging unexpectedly became a symlink")
        if staging.exists():
            shutil.rmtree(staging)


def _run_project(
    config_path: Path,
    output_root_override: Path | None,
    run_id: str,
    run_dir: Path,
) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    project_root = config_path.parent.parent
    output_root = output_root_override or (project_root / str(config["output_root"]))
    final_run_dir = output_root / run_id
    if final_run_dir.exists() or final_run_dir.is_symlink():
        raise FileExistsError(final_run_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("data", "models", "metrics", "traces", "cards"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _log("run_started", run_id=run_id, cases=int(config["n_cases"]))
    atomic_json(run_dir / "config.resolved.json", config)

    full_data = generate_cases(
        int(config["n_cases"]), int(config["n_entities"]), int(config["seed"])
    )
    quality = validate_cases(full_data)
    if quality["status"] != "PASS":
        raise RuntimeError(f"data quality gate failed: {quality}")
    atomic_csv(run_dir / "data" / "tradeops_sim.csv", full_data)
    atomic_json(run_dir / "metrics" / "data_quality.json", quality)
    _log("foundation_complete", rows=len(full_data), data_quality=quality["status"])
    dataset_fingerprint = dataframe_fingerprint(full_data)
    atomic_json(
        run_dir / "data" / "dataset_manifest.json",
        {
            "dataset": "TradeOps-Sim",
            "version": "0.1.0",
            "fingerprint": dataset_fingerprint,
            "rows": len(full_data),
            "split_counts": quality["split_distribution"],
            "claim_boundary": "controlled synthetic benchmark",
        },
    )

    required_splits = {"train", "validation", "calibration", "policy", "test", "stress"}
    if not required_splits.issubset(set(full_data["split"])):
        raise RuntimeError("one or more required splits are empty")
    raw = {
        name: full_data[full_data["split"] == name].copy().reset_index(drop=True)
        for name in required_splits
    }

    rsop = RSOPTransformer(
        min_reference_count=int(config["rsop"]["min_reference_count"]),
        epsilon=float(config["rsop"]["epsilon"]),
    )
    rsop.fit(raw["train"], raw["train"]["root_cause"])
    frames = _candidate_frames(raw, rsop)
    atomic_json(run_dir / "models" / "rsop_manifest.json", rsop.manifest())
    _log("rsop_fitted", references=rsop.manifest()["global_reference_rows"])

    models: dict[str, Any] = {}
    for candidate_id in ("C0", "C1", "C2", "C3", "C4"):
        model = build_candidate(candidate_id, config)
        model.fit(frames[candidate_id]["train"], raw["train"]["root_cause"])
        models[candidate_id] = model
        atomic_joblib(
            run_dir / "models" / f"{candidate_id}.joblib", model, staging_root=run_dir.parent
        )
        _log(
            "candidate_trained",
            candidate_id=candidate_id,
            seconds=float(getattr(model, "fit_seconds_", 0.0)),
        )

    c1_classes = np.asarray(models["C1"].classes_)
    c1_scaler = TemperatureScaler(
        lower=float(config["temperature_bounds"][0]),
        upper=float(config["temperature_bounds"][1]),
    )
    c1_scaler.fit(
        models["C1"].predict_proba(frames["C1"]["calibration"]),
        raw["calibration"]["root_cause"].to_numpy(),
        c1_classes,
    )
    c4_classes = np.asarray(models["C4"].classes_)
    c5_scaler = TemperatureScaler(
        lower=float(config["temperature_bounds"][0]),
        upper=float(config["temperature_bounds"][1]),
    )
    c5_scaler.fit(
        models["C4"].predict_proba(frames["C4"]["calibration"]),
        raw["calibration"]["root_cause"].to_numpy(),
        c4_classes,
    )
    atomic_joblib(
        run_dir / "models" / "C1_temperature.joblib", c1_scaler, staging_root=run_dir.parent
    )
    atomic_joblib(
        run_dir / "models" / "C5_temperature.joblib", c5_scaler, staging_root=run_dir.parent
    )
    _log(
        "calibration_complete",
        c1_temperature=c1_scaler.temperature_,
        c5_temperature=c5_scaler.temperature_,
    )

    results: dict[str, dict[str, Any]] = {}
    test_raw = raw["test"]
    policies: dict[str, ReviewPolicy] = {
        "C0": "fifo",
        "C1": "confidence",
        "C2": "disagreement",
        "C3": "confidence",
        "C4": "disagreement",
        "C5": "evr",
    }
    decision_thresholds: dict[str, float] = {}
    policy_raw = raw["policy"]
    for candidate_id in CANDIDATE_META:
        model_id = "C4" if candidate_id == "C5" else candidate_id
        model = models[model_id]
        policy_frame = frames[candidate_id]["policy"]
        probabilities = model.predict_proba(policy_frame)
        classes = np.asarray(model.classes_)
        if candidate_id == "C1":
            probabilities = c1_scaler.transform(probabilities)
        if candidate_id == "C5":
            probabilities = c5_scaler.transform(probabilities)
        disagreement = (
            model.disagreement(policy_frame) if candidate_id in {"C2", "C4", "C5"} else None
        )
        envelopes = (
            build_action_envelopes(policy_raw, probabilities, classes, config)
            if candidate_id in {"C4", "C5"}
            else None
        )
        predictions = classes[np.argmax(probabilities, axis=1)]
        decision_thresholds[candidate_id] = fit_review_threshold(
            policy_raw,
            predictions,
            probabilities,
            policies[candidate_id],
            float(config["review_capacity"]),
            float(config["human_correction_probability"]),
            float(config["review_cost"]),
            disagreement,
            envelopes,
        )
    atomic_json(
        run_dir / "models" / "decision_policy.json",
        {
            "schema_version": "1.0",
            "fit_split": "policy",
            "review_capacity": float(config["review_capacity"]),
            "policies": policies,
            "minimum_review_scores": decision_thresholds,
        },
    )
    for candidate_id in CANDIDATE_META:
        model_id = "C4" if candidate_id == "C5" else candidate_id
        model = models[model_id]
        candidate_frame = frames[candidate_id]["test"]
        probabilities = model.predict_proba(candidate_frame)
        classes = np.asarray(model.classes_)
        if candidate_id == "C1":
            probabilities = c1_scaler.transform(probabilities)
        if candidate_id == "C5":
            probabilities = c5_scaler.transform(probabilities)
        disagreement = (
            model.disagreement(candidate_frame) if candidate_id in {"C2", "C4", "C5"} else None
        )
        envelopes = (
            build_action_envelopes(test_raw, probabilities, classes, config)
            if candidate_id in {"C4", "C5"}
            else None
        )
        latency_fn = _latency_function(candidate_id, model, c1_scaler, c5_scaler, rsop)
        fit_seconds = float(getattr(model, "fit_seconds_", 0.0))
        results[candidate_id] = _evaluate_candidate(
            candidate_id,
            model,
            candidate_frame,
            test_raw,
            probabilities,
            classes,
            config,
            disagreement,
            envelopes,
            policies[candidate_id],
            fit_seconds,
            latency_fn,
            decision_thresholds[candidate_id],
        )
        _log(
            "candidate_evaluated",
            candidate_id=candidate_id,
            macro_f1=results[candidate_id]["metrics"]["macro_f1"],
        )

    stress_results: dict[str, dict[str, Any]] = {}
    stress_raw = raw["stress"]
    for candidate_id in CANDIDATE_META:
        model_id = "C4" if candidate_id == "C5" else candidate_id
        model = models[model_id]
        candidate_frame = frames[candidate_id]["stress"]
        probabilities = model.predict_proba(candidate_frame)
        classes = np.asarray(model.classes_)
        if candidate_id == "C1":
            probabilities = c1_scaler.transform(probabilities)
        if candidate_id == "C5":
            probabilities = c5_scaler.transform(probabilities)
        disagreement = (
            model.disagreement(candidate_frame) if candidate_id in {"C2", "C4", "C5"} else None
        )
        envelopes = (
            build_action_envelopes(stress_raw, probabilities, classes, config)
            if candidate_id in {"C4", "C5"}
            else None
        )
        stress_results[candidate_id] = _evaluate_candidate(
            candidate_id,
            model,
            candidate_frame,
            stress_raw,
            probabilities,
            classes,
            config,
            disagreement,
            envelopes,
            policies[candidate_id],
            float(getattr(model, "fit_seconds_", 0.0)),
            _latency_function(candidate_id, model, c1_scaler, c5_scaler, rsop),
            decision_thresholds[candidate_id],
        )

    comparison = pd.DataFrame([results[candidate_id]["metrics"] for candidate_id in CANDIDATE_META])
    stress_comparison = pd.DataFrame(
        [stress_results[candidate_id]["metrics"] for candidate_id in CANDIDATE_META]
    )
    predictions = pd.concat(
        [results[candidate_id]["predictions"] for candidate_id in CANDIDATE_META], ignore_index=True
    )
    slices = pd.concat(
        [results[candidate_id]["slices"] for candidate_id in CANDIDATE_META], ignore_index=True
    )
    atomic_csv(run_dir / "metrics" / "candidate_comparison.csv", comparison)
    atomic_csv(run_dir / "metrics" / "stress_comparison.csv", stress_comparison)
    atomic_csv(run_dir / "metrics" / "predictions.csv", predictions)
    atomic_csv(run_dir / "metrics" / "slice_metrics.csv", slices)
    atomic_json(
        run_dir / "metrics" / "metrics.json",
        {candidate_id: results[candidate_id]["metrics"] for candidate_id in CANDIDATE_META},
    )
    failure_cases = predictions[
        (predictions["candidate_id"] == "C5")
        & ((predictions["correct"] == 0) | (predictions["route"] != "SIMULATE"))
    ]
    atomic_csv(run_dir / "metrics" / "failure_cases.csv", failure_cases)
    if results["C5"]["envelopes"] is not None:
        atomic_csv(run_dir / "traces" / "action_envelopes.csv", results["C5"]["envelopes"])

    frontier_records: list[dict[str, Any]] = []
    frontier_policies: tuple[tuple[str, ReviewPolicy], ...] = (
        ("C1", "confidence"),
        ("C5", "evr"),
    )
    for capacity in np.linspace(0.0, 0.4, 9):
        for candidate_id, policy in frontier_policies:
            result = results[candidate_id]
            disagreement = (
                models["C4"].disagreement(frames["C5"]["test"]) if candidate_id == "C5" else None
            )
            routes = route_decisions(
                test_raw,
                result["predictions"]["predicted_cause"].to_numpy(),
                result["probabilities"],
                policy=policy,
                review_capacity=float(capacity),
                human_correction_probability=float(config["human_correction_probability"]),
                review_cost=float(config["review_cost"]),
                disagreement=disagreement,
                envelopes=result["envelopes"],
            )
            values, _ = decision_metrics(
                test_raw,
                result["predictions"]["proposed_action"].to_numpy(),
                routes,
                float(config["human_correction_probability"]),
                float(config["review_cost"]),
                float(config.get("hold_cost_fraction", 0.10)),
                float(config.get("hold_fixed_cost", 0.10)),
            )
            frontier_records.append(
                {"candidate_id": candidate_id, "review_capacity": float(capacity), **values}
            )
    frontier = pd.DataFrame(frontier_records)
    atomic_csv(run_dir / "metrics" / "decision_scenarios.csv", frontier)

    sensitivity_records: list[dict[str, Any]] = []
    for human_probability in (0.60, 0.85, 0.95):
        for hold_fraction in (0.00, 0.05, 0.10, 0.20):
            for candidate_id, policy in frontier_policies:
                result = results[candidate_id]
                disagreement = (
                    models["C4"].disagreement(frames["C5"]["test"])
                    if candidate_id == "C5"
                    else None
                )
                routes = route_decisions(
                    test_raw,
                    result["predictions"]["predicted_cause"].to_numpy(),
                    result["probabilities"],
                    policy=policy,
                    review_capacity=float(config["review_capacity"]),
                    human_correction_probability=human_probability,
                    review_cost=float(config["review_cost"]),
                    disagreement=disagreement,
                    envelopes=result["envelopes"],
                )
                values, _ = decision_metrics(
                    test_raw,
                    result["predictions"]["proposed_action"].to_numpy(),
                    routes,
                    human_probability,
                    float(config["review_cost"]),
                    hold_fraction,
                    float(config.get("hold_fixed_cost", 0.10)),
                )
                sensitivity_records.append(
                    {
                        "candidate_id": candidate_id,
                        "human_correction_probability": human_probability,
                        "hold_cost_fraction": hold_fraction,
                        **values,
                    }
                )
    sensitivity = pd.DataFrame(sensitivity_records)
    atomic_csv(run_dir / "metrics" / "decision_sensitivity.csv", sensitivity)

    bootstrap = paired_bootstrap_delta(
        results["C1"]["costs"],
        results["C5"]["costs"],
        int(config["seed"]) + 999,
        int(config["bootstrap_resamples"]),
        groups=test_raw["entity_id"].to_numpy(),
    )
    specialty_resamples = int(config["specialty_resamples"])
    y_test = test_raw["root_cause"].to_numpy()
    specialty_statistics = {
        "C3_vs_C1_macro_f1": paired_bootstrap_macro_f1_delta(
            y_test,
            results["C1"]["predictions"]["predicted_cause"].to_numpy(),
            results["C3"]["predictions"]["predicted_cause"].to_numpy(),
            int(config["seed"]) + 301,
            specialty_resamples,
            classes=np.asarray(results["C1"]["classes"]),
            groups=test_raw["entity_id"].to_numpy(),
        ),
        "C4_vs_C2_macro_f1": paired_bootstrap_macro_f1_delta(
            y_test,
            results["C2"]["predictions"]["predicted_cause"].to_numpy(),
            results["C4"]["predictions"]["predicted_cause"].to_numpy(),
            int(config["seed"]) + 302,
            specialty_resamples,
            classes=np.asarray(results["C2"]["classes"]),
            groups=test_raw["entity_id"].to_numpy(),
        ),
        "C3_vs_C1_residual_harm": paired_bootstrap_delta(
            results["C1"]["costs"],
            results["C3"]["costs"],
            int(config["seed"]) + 303,
            specialty_resamples,
            groups=test_raw["entity_id"].to_numpy(),
        ),
    }
    atomic_json(run_dir / "metrics" / "specialty_ablation.json", specialty_statistics)
    c1_harm = float(results["C1"]["metrics"]["residual_harm_per_1000"])
    c5_harm = float(results["C5"]["metrics"]["residual_harm_per_1000"])
    relative_improvement = (c1_harm - c5_harm) / max(c1_harm, 1e-12)
    slice_pivot = (
        slices[slices["candidate_id"].isin(["C1", "C5"])]
        .pivot_table(
            index=["slice_name", "slice_value"],
            columns="candidate_id",
            values="macro_f1",
            aggfunc="first",
        )
        .dropna()
    )
    worst_slice_delta = (
        float((slice_pivot["C5"] - slice_pivot["C1"]).min()) if len(slice_pivot) else -1.0
    )
    c5_envelopes = results["C5"]["envelopes"]
    c5_actions = results["C5"]["predictions"]["proposed_action"].to_numpy()
    c5_routes = results["C5"]["predictions"]["route"].to_numpy()
    unauthorized = 0
    for position, (_, row) in enumerate(test_raw.iterrows()):
        if c5_routes[position] == "SIMULATE" and c5_actions[position] not in allowed_actions(row):
            unauthorized += 1
    stress_c1 = stress_results["C1"]["metrics"]
    stress_c5 = stress_results["C5"]["metrics"]
    stress_f1_delta = float(stress_c5["macro_f1"] - stress_c1["macro_f1"])
    stress_harm_ratio = float(
        stress_c5["residual_harm_per_1000"] / max(stress_c1["residual_harm_per_1000"], 1e-12)
    )
    gates = {
        "data_quality": quality["status"] == "PASS",
        "relative_harm_improvement": relative_improvement
        >= float(config["min_relative_harm_improvement"]),
        "paired_ci_excludes_no_improvement": bootstrap["ci95_lower"] > 0.0,
        "critical_false_greenlight": results["C5"]["metrics"]["critical_false_greenlight_upper95"]
        <= float(config["critical_false_greenlight_limit"]),
        "evidence_precision": results["C5"]["metrics"].get("evidence_precision", 0.0)
        >= float(config["evidence_precision_limit"]),
        "manual_intervention_capacity": results["C5"]["metrics"]["manual_intervention_rate"]
        <= float(config["review_capacity"]) + float(config["manual_capacity_tolerance"]),
        "hold_rate": results["C5"]["metrics"]["hold_rate"] <= float(config["max_hold_rate"]),
        "worst_slice_non_regression": worst_slice_delta
        >= -float(config["max_worst_slice_macro_f1_regression"]),
        "stress_predictive_non_regression": stress_f1_delta
        >= -float(config["max_macro_f1_regression"]),
        "stress_harm_non_regression": stress_harm_ratio <= float(config["max_stress_harm_ratio"]),
        "stress_manual_intervention_capacity": stress_c5["manual_intervention_rate"]
        <= float(config["review_capacity"]) + float(config["manual_capacity_tolerance"]),
        "unauthorized_action_zero": unauthorized == 0,
    }
    predictive_champion = str(
        comparison.sort_values(["macro_f1", "multiclass_brier"], ascending=[False, True]).iloc[0][
            "candidate_id"
        ]
    )
    safety_champion = str(
        comparison.sort_values(
            [
                "unsafe_automated_harm_per_1000",
                "manual_intervention_rate",
                "residual_harm_per_1000",
            ],
            ascending=[True, True, True],
        ).iloc[0]["candidate_id"]
    )
    status = "PROMOTE" if all(gates.values()) else "HOLD"
    production_champion = "C5" if status == "PROMOTE" else "C1"
    failed = [name for name, value in gates.items() if not value]
    rationale = (
        f"C5 passed all gates with {relative_improvement:.2%} lower residual harm than C1."
        if status == "PROMOTE"
        else f"C5 was not promoted; failed gates: {', '.join(failed)}. C1 remains the production reference."
    )
    release = {
        "status": status,
        "predictive_champion": predictive_champion,
        "safety_champion": safety_champion,
        "production_champion": production_champion,
        "review_capacity": float(config["review_capacity"]),
        "gate_thresholds": {
            "max_macro_f1_regression": float(config["max_macro_f1_regression"]),
            "max_worst_slice_macro_f1_regression": float(
                config["max_worst_slice_macro_f1_regression"]
            ),
            "max_stress_harm_ratio": float(config["max_stress_harm_ratio"]),
            "max_hold_rate": float(config["max_hold_rate"]),
            "manual_capacity_tolerance": float(config["manual_capacity_tolerance"]),
        },
        "relative_harm_improvement": relative_improvement,
        "bootstrap": bootstrap,
        "worst_slice_delta": worst_slice_delta,
        "unauthorized_action_count": unauthorized,
        "stress": {
            "c1_macro_f1": stress_c1["macro_f1"],
            "c5_macro_f1": stress_c5["macro_f1"],
            "macro_f1_delta": stress_f1_delta,
            "c1_residual_harm_per_1000": stress_c1["residual_harm_per_1000"],
            "c5_residual_harm_per_1000": stress_c5["residual_harm_per_1000"],
            "harm_ratio": stress_harm_ratio,
        },
        "gates": gates,
        "rationale": rationale,
        "claim_boundary": "controlled TradeOps-Sim only",
    }
    atomic_json(run_dir / "release_decision.json", release)
    _log(
        "release_evaluated",
        status=status,
        production_champion=production_champion,
        failed_gates=failed,
    )
    atomic_json(
        run_dir / "policy_manifest.json",
        {
            "review_capacity": config["review_capacity"],
            "human_correction_probability": config["human_correction_probability"],
            "review_cost": config["review_cost"],
            "hold_cost_fraction": config.get("hold_cost_fraction", 0.10),
            "hold_fixed_cost": config.get("hold_fixed_cost", 0.10),
            "fallback": "HOLD",
            "external_execution": False,
            "fit_split": "policy",
            "minimum_review_scores": decision_thresholds,
        },
    )

    run_manifest = {
        "run_id": run_id,
        "project_version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": dataset_fingerprint,
        "config_hash": config_hash(config),
        "config_path": _portable_path(config_path, project_root),
        "seed": config["seed"],
        "review_capacity": config["review_capacity"],
        "hardware": hardware_metadata(),
        "elapsed_seconds_before_report": time.perf_counter() - started,
        "claim_boundary": "controlled simulation",
    }
    atomic_json(run_dir / "run_manifest.json", run_manifest)

    calibration = calibration_table(predictions)
    production_predictions = results[production_champion]["predictions"]
    production_classes = results[production_champion]["classes"]
    confusion = confusion_records(
        production_predictions["true_cause"].to_numpy(),
        production_predictions["predicted_cause"].to_numpy(),
        production_classes,
    )
    figures = generate_report(
        run_dir,
        full_data,
        frames["C5"]["test"],
        comparison,
        stress_comparison,
        calibration,
        frontier,
        slices,
        confusion,
        release,
        run_manifest,
    )
    _log("report_generated", figure_count=len(figures))
    cards = _write_cards(
        run_dir, quality, {key: value["metrics"] for key, value in results.items()}, release
    )

    selected_model = models["C4"] if production_champion == "C5" else models[production_champion]
    selected_calibrator = (
        c5_scaler
        if production_champion == "C5"
        else (c1_scaler if production_champion == "C1" else None)
    )
    selected_classes = np.asarray(selected_model.classes_)
    bundle_runtime = runtime_contract()
    bundle = {
        "run_id": run_id,
        "model_version": f"{production_champion}-{__version__}",
        "production_champion": production_champion,
        "model": selected_model,
        "rsop": rsop,
        "calibrator": selected_calibrator,
        "config": config,
        "classes": selected_classes,
        "use_rsop": production_champion in {"C3", "C4", "C5"},
        "use_action_envelope": production_champion in {"C4", "C5"},
        "claim_boundary": "controlled simulation",
        "runtime_contract": bundle_runtime,
        "decision_policy": {
            "policy": policies[production_champion],
            "minimum_review_score": decision_thresholds[production_champion],
        },
    }
    atomic_joblib(
        run_dir / "models" / "production_bundle.joblib", bundle, staging_root=run_dir.parent
    )
    atomic_json(run_dir / "models" / "production_bundle.runtime.json", bundle_runtime)
    core_checksums = {
        path.name: sha256_file(path)
        for path in (
            run_dir / "data" / "tradeops_sim.csv",
            run_dir / "models" / "C1.joblib",
            run_dir / "models" / "production_bundle.joblib",
            run_dir / "metrics" / "metrics.json",
        )
    }
    handoff = {
        "project": "FinOps-R",
        "version": __version__,
        "run_id": run_id,
        "project_depth": {
            "foundation_complete": True,
            "industry_baseline_complete": True,
            "advanced_technique_complete": True,
            "specialty_module_complete": True,
            "product_layer_complete": True,
        },
        "candidate_portfolio": [
            {
                "candidate_id": candidate_id,
                "type": CANDIDATE_META[candidate_id]["type"],
                "algorithm": CANDIDATE_META[candidate_id]["algorithm"],
                "features_or_representation": "RSOP"
                if "RSOP" in CANDIDATE_META[candidate_id]["specialty"]
                else "prefix-safe tabular",
                "specialty_components": CANDIDATE_META[candidate_id]["specialty"],
                "config_path": str(config_path.relative_to(project_root)),
                "artifact_path": f"models/{'C4' if candidate_id == 'C5' else candidate_id}.joblib",
                "supporting_artifacts": (
                    ["models/C5_temperature.joblib", "models/decision_policy.json"]
                    if candidate_id == "C5"
                    else (["models/C1_temperature.joblib"] if candidate_id == "C1" else [])
                ),
                "metrics": results[candidate_id]["metrics"],
                "operational_metrics": {
                    key: value
                    for key, value in results[candidate_id]["metrics"].items()
                    if key.startswith("latency_")
                    or key in {"throughput_cases_per_second", "training_seconds"}
                },
                "status": "selected"
                if candidate_id in {predictive_champion, production_champion}
                else "challenger",
            }
            for candidate_id in CANDIDATE_META
        ],
        "specialty_transfer": [
            {
                "source_method": "Physical Health Timestep and reference-state alignment",
                "adapted_name": "Reference-State Operational Progress",
                "adaptation_type": "domain_extension",
                "project_problem_addressed": "phase-dependent operational delay",
                "implementation_path": "src/finops_r/specialty.py",
                "test_path": "tests/test_specialty.py",
                "ablation_result": {
                    "C3_minus_C1_macro_f1": float(
                        results["C3"]["metrics"]["macro_f1"] - results["C1"]["metrics"]["macro_f1"]
                    ),
                    "C4_minus_C2_macro_f1": float(
                        results["C4"]["metrics"]["macro_f1"] - results["C2"]["metrics"]["macro_f1"]
                    ),
                    "paired_statistics": specialty_statistics,
                },
                "limitations": [
                    "reference cohorts are simulator-derived",
                    "benefit may be domain-specific",
                ],
            },
            {
                "source_method": "Probability-Based Bbox and ABC Bbox candidate coverage/correction",
                "adapted_name": "Action Evidence Envelope",
                "adaptation_type": "inspired",
                "project_problem_addressed": "single-action miss and unsupported-action overreach",
                "implementation_path": "src/finops_r/evidence.py",
                "test_path": "tests/test_evidence.py",
                "ablation_result": envelope_metrics(c5_envelopes, test_raw["oracle_action"]),
                "limitations": [
                    "action space is simulator-defined",
                    "rule evidence is not a bank policy corpus",
                ],
            },
        ],
        "champion_selection": {
            "predictive_champion": predictive_champion,
            "production_champion": production_champion,
            "safety_champion": safety_champion,
            "selection_criteria": list(gates),
            "tradeoffs": [rationale],
        },
        "outputs": {
            "numeric": [
                "metrics/metrics.json",
                "metrics/candidate_comparison.csv",
                "metrics/stress_comparison.csv",
                "metrics/slice_metrics.csv",
                "metrics/decision_scenarios.csv",
                "metrics/decision_sensitivity.csv",
                "metrics/specialty_ablation.json",
                "metrics/predictions.csv",
            ],
            "visualizations": figures,
            "reports": ["REPORT.md", "REPORT.html"],
            "cards": cards,
            "api_or_cli_examples": [
                "finops-r run",
                "finops-r assess",
                "finops-r report",
                "finops-r rollback",
            ],
        },
        "experiment_lineage": {
            "run_ids": [run_id],
            "dataset_fingerprints": [dataset_fingerprint],
            "config_hashes": [config_hash(config)],
            "artifact_checksums": core_checksums,
        },
        "release": release,
    }
    atomic_json(run_dir / "project_creation_handoff.json", handoff)

    cleanup_empty_joblib_staging(run_dir.parent, owner=run_dir.name.lstrip("."))
    manifest = artifact_manifest(run_dir)
    atomic_json(run_dir / "artifact_manifest.json", manifest)
    integrity = verify_artifact_manifest(run_dir)
    atomic_json(run_dir / "artifact_integrity.json", integrity)
    _log("artifact_integrity_complete", checked=len(manifest))

    publish_verified_tree(run_dir, final_run_dir)
    run_dir = final_run_dir

    # A caller-supplied output root must move all mutable run state, including
    # release pointers. Otherwise an isolated smoke run can contaminate the
    # source repository that merely supplied its configuration file.
    publication_root = output_root.parent
    releases = publication_root / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    atomic_json(
        releases / "latest.json",
        {
            "run_id": run_id,
            "run_dir": _portable_path(run_dir, publication_root),
            "status": status,
        },
    )
    if status == "PROMOTE":
        atomic_json(
            releases / "champion.json",
            {
                "run_id": run_id,
                "run_dir": _portable_path(run_dir, publication_root),
                "candidate": "C5",
            },
        )
    _log("run_complete", run_id=run_id, elapsed_seconds=time.perf_counter() - started)
    return run_dir


def generate_only(config_path: Path, output_path: Path) -> Path:
    config = load_config(config_path.resolve())
    frame = generate_cases(int(config["n_cases"]), int(config["n_entities"]), int(config["seed"]))
    quality = validate_cases(frame)
    if quality["status"] != "PASS":
        raise RuntimeError("generated data failed validation")
    atomic_csv(output_path, frame)
    atomic_json(
        output_path.with_suffix(".manifest.json"),
        {"fingerprint": dataframe_fingerprint(frame), "quality": quality},
    )
    return output_path


def load_production_bundle(run_dir: Path) -> dict[str, Any]:
    if run_dir.is_symlink():
        raise ArtifactVerificationError("artifact root must not be a symlink")
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "artifact_manifest.json"
    integrity_path = run_dir / "artifact_integrity.json"
    try:
        manifest_snapshot = manifest_path.read_bytes()
        integrity_snapshot = integrity_path.read_bytes()
        manifest_records = strict_json_loads(manifest_snapshot)
        integrity = strict_json_loads(integrity_snapshot)
    except (OSError, StrictJSONError) as exc:
        raise ArtifactVerificationError(
            "artifact manifest or integrity marker is missing or unreadable"
        ) from exc
    live_integrity = verify_artifact_manifest(run_dir)
    try:
        if manifest_path.read_bytes() != manifest_snapshot:
            raise ArtifactVerificationError("artifact manifest changed during verification")
    except OSError as exc:
        raise ArtifactVerificationError("artifact manifest changed during verification") from exc
    if integrity != live_integrity:
        raise ArtifactVerificationError("artifact integrity marker is stale or inconsistent")

    expected_files = {
        record["path"]: record
        for record in manifest_records
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }

    def verified_snapshot(relative: str) -> bytes:
        record = expected_files.get(relative)
        if record is None:
            raise ArtifactVerificationError(
                f"required production artifact is untracked: {relative}"
            )
        try:
            payload = (run_dir / Path(*PurePosixPath(relative).parts)).read_bytes()
        except OSError as exc:
            raise ArtifactVerificationError(
                f"required production artifact is unreadable: {relative}"
            ) from exc
        if len(payload) != record.get("bytes") or hashlib.sha256(payload).hexdigest() != record.get(
            "sha256"
        ):
            raise ArtifactVerificationError(
                f"production artifact changed after integrity verification: {relative}"
            )
        return payload

    bundle_snapshot = verified_snapshot("models/production_bundle.joblib")
    runtime_snapshot = verified_snapshot("models/production_bundle.runtime.json")
    try:
        expected_runtime = strict_json_loads(runtime_snapshot)
    except StrictJSONError as exc:
        raise ArtifactVerificationError(
            "production runtime contract is missing or unreadable"
        ) from exc
    verify_runtime_contract(expected_runtime)
    try:
        bundle = joblib.load(io.BytesIO(bundle_snapshot))
    except Exception as exc:
        raise ArtifactVerificationError("production bundle cannot be deserialized") from exc
    required = {
        "run_id",
        "model_version",
        "production_champion",
        "model",
        "rsop",
        "calibrator",
        "config",
        "classes",
        "use_rsop",
        "use_action_envelope",
        "claim_boundary",
        "runtime_contract",
        "decision_policy",
    }
    if not isinstance(bundle, dict) or set(bundle) != required:
        raise ArtifactVerificationError("production bundle schema is incompatible")
    champion = bundle["production_champion"]
    if champion not in {"C1", "C5"}:
        raise ArtifactVerificationError("production bundle champion is invalid")
    if (
        bundle["run_id"] != run_dir.name
        or bundle["claim_boundary"] != "controlled simulation"
        or bundle["model_version"] != f"{champion}-{__version__}"
    ):
        raise ArtifactVerificationError("production bundle identity or claim boundary is invalid")
    if bundle["runtime_contract"] != expected_runtime:
        raise ArtifactVerificationError("embedded and external runtime contracts differ")
    try:
        ProjectConfig.model_validate(bundle["config"])
    except Exception as exc:
        raise ArtifactVerificationError("embedded project config is invalid") from exc
    expected_use_rsop = champion == "C5"
    expected_use_action_envelope = champion == "C5"
    if (
        type(bundle["use_rsop"]) is not bool
        or type(bundle["use_action_envelope"]) is not bool
        or bundle["use_rsop"] != expected_use_rsop
        or bundle["use_action_envelope"] != expected_use_action_envelope
    ):
        raise ArtifactVerificationError("production bundle feature flags are invalid")
    decision_policy = bundle["decision_policy"]
    if (
        not isinstance(decision_policy, dict)
        or set(decision_policy) != {"policy", "minimum_review_score"}
        or decision_policy["policy"] not in {"fifo", "confidence", "disagreement", "evr"}
        or isinstance(decision_policy["minimum_review_score"], bool)
        or not isinstance(decision_policy["minimum_review_score"], (int, float))
        or not np.isfinite(float(decision_policy["minimum_review_score"]))
    ):
        raise ArtifactVerificationError("embedded decision policy is invalid")
    classes = np.asarray(bundle["classes"])
    if (
        classes.ndim != 1
        or len(classes) != len(CAUSES)
        or len(set(classes.tolist())) != len(classes)
        or set(classes.tolist()) != set(CAUSES)
    ):
        raise ArtifactVerificationError("embedded class labels are invalid")
    model_classes = np.asarray(getattr(bundle["model"], "classes_", []))
    calibrator_classes = np.asarray(getattr(bundle["calibrator"], "classes_", []))
    calibrator_temperature = getattr(bundle["calibrator"], "temperature_", None)
    if (
        not callable(getattr(bundle["model"], "predict_proba", None))
        or model_classes.ndim != 1
        or model_classes.tolist() != classes.tolist()
        or not callable(getattr(bundle["rsop"], "transform", None))
        or getattr(bundle["rsop"], "fitted_", None) is not True
        or not callable(getattr(bundle["calibrator"], "transform", None))
        or calibrator_classes.ndim != 1
        or calibrator_classes.tolist() != classes.tolist()
        or isinstance(calibrator_temperature, bool)
        or not isinstance(calibrator_temperature, (int, float))
        or not np.isfinite(float(calibrator_temperature))
        or float(calibrator_temperature) <= 0.0
    ):
        raise ArtifactVerificationError("production bundle components are incompatible")

    metadata_paths = {
        "config": "config.resolved.json",
        "release": "release_decision.json",
        "manifest": "run_manifest.json",
        "policy": "models/decision_policy.json",
    }
    try:
        metadata = {
            name: strict_json_loads(verified_snapshot(relative))
            for name, relative in metadata_paths.items()
        }
    except StrictJSONError as exc:
        raise ArtifactVerificationError("production metadata is missing or unreadable") from exc
    if any(not isinstance(value, dict) for value in metadata.values()):
        raise ArtifactVerificationError("production metadata schemas are incompatible")
    release = metadata["release"]
    run_manifest = metadata["manifest"]
    policy_record = metadata["policy"]
    recorded_policies = policy_record.get("policies")
    recorded_thresholds = policy_record.get("minimum_review_scores")
    if not isinstance(recorded_policies, dict) or not isinstance(recorded_thresholds, dict):
        raise ArtifactVerificationError("production policy metadata is incompatible")
    expected_status = "PROMOTE" if champion == "C5" else "HOLD"
    if (
        metadata["config"] != bundle["config"]
        or release.get("production_champion") != champion
        or release.get("status") != expected_status
        or run_manifest.get("run_id") != bundle["run_id"]
        or run_manifest.get("project_version") != __version__
        or run_manifest.get("config_hash") != config_hash(bundle["config"])
        or recorded_policies.get(champion) != decision_policy["policy"]
        or recorded_thresholds.get(champion) != decision_policy["minimum_review_score"]
    ):
        raise ArtifactVerificationError("production artifacts are internally inconsistent")
    return bundle


def assess_request(run_dir: Path, request_path: Path) -> dict[str, Any]:
    payload = require_json_object(
        strict_json_loads(request_path.read_text(encoding="utf-8")),
        source=f"assessment request {request_path}",
    )
    return assess_payload(run_dir, payload)


def assess_payload(run_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    bundle = load_production_bundle(run_dir)
    return assess_payload_with_bundle(bundle, payload)


def assess_payload_with_bundle(bundle: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    snapshot = CaseSnapshot.model_validate(payload)
    frame = pd.DataFrame([snapshot.model_dump(mode="json")])
    model_frame = bundle["rsop"].transform(frame) if bundle["use_rsop"] else frame
    probabilities = bundle["model"].predict_proba(model_frame)
    if bundle["calibrator"] is not None:
        probabilities = bundle["calibrator"].transform(probabilities)
    classes = np.asarray(bundle["classes"])
    predicted_cause = str(classes[int(np.argmax(probabilities[0]))])
    envelopes = build_action_envelopes(frame, probabilities, classes, bundle["config"])
    envelope = envelopes.iloc[0]
    confidence = float(probabilities[0].max())
    risk = 1.0 - confidence
    disagreement = None
    if bundle["decision_policy"]["policy"] in {"disagreement", "evr"}:
        if not hasattr(bundle["model"], "disagreement"):
            raise ArtifactVerificationError("decision policy requires model disagreement support")
        disagreement = bundle["model"].disagreement(model_frame)
    routed = route_decisions(
        frame,
        np.asarray([predicted_cause]),
        probabilities,
        policy=bundle["decision_policy"]["policy"],
        review_capacity=1.0,
        human_correction_probability=float(bundle["config"]["human_correction_probability"]),
        review_cost=float(bundle["config"]["review_cost"]),
        disagreement=disagreement,
        envelopes=envelopes if bundle["use_action_envelope"] else None,
        minimum_review_score=float(bundle["decision_policy"]["minimum_review_score"]),
    ).iloc[0]
    route = str(routed["route"])
    reason = str(routed["route_reason"])
    result = {
        "case_id": snapshot.case_id,
        "predicted_cause": predicted_cause,
        "proposed_action": (
            envelope["proposed_action"]
            if bundle["use_action_envelope"]
            else ACTION_BY_CAUSE[predicted_cause]
        ),
        "confidence": confidence,
        "calibrated_error_risk": risk,
        "route": route,
        "reason_codes": [reason]
        + (
            [item for item in str(envelope["evidence_reason_codes"]).split("|") if item]
            if bundle["use_action_envelope"]
            else []
        ),
        "evidence_ids": (
            [item for item in str(envelope["evidence_ids"]).split("|") if item]
            if bundle["use_action_envelope"]
            else []
        ),
        "policy_status": envelope["policy_status"]
        if bundle["use_action_envelope"]
        else "NOT_APPLICABLE",
        "verifier_status": envelope["verifier_status"]
        if bundle["use_action_envelope"]
        else "NOT_APPLICABLE",
        "model_version": bundle["model_version"],
        "run_id": bundle["run_id"],
        "claim_boundary": bundle["claim_boundary"],
    }
    return Assessment.model_validate(result).model_dump(mode="json")


def regenerate_report(run_dir: Path) -> tuple[Path, list[str]]:
    """Regenerate into a separately verified run copy, preserving the source release."""
    if run_dir.is_symlink():
        raise ArtifactVerificationError("artifact root must not be a symlink")
    source = run_dir.resolve()
    load_production_bundle(source)
    refresh_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination_parent = source.parent / "report_refreshes" / f"{refresh_id}_{uuid.uuid4().hex[:8]}"
    destination = destination_parent / source.name
    with tempfile.TemporaryDirectory(prefix=f".{source.name}.report-refresh.") as directory:
        staging = Path(directory) / source.name
        shutil.copytree(source, staging)
        figures = _regenerate_report_in_place(staging)
        destination_parent.mkdir(parents=True, exist_ok=False)
        try:
            publish_verified_tree(staging, destination)
        except BaseException:
            if destination_parent.exists() and not any(destination_parent.iterdir()):
                destination_parent.rmdir()
            raise
    return destination, figures


def _regenerate_report_in_place(run_dir: Path) -> list[str]:
    bundle = load_production_bundle(run_dir)
    run_dir = run_dir.resolve()
    full_data = pd.read_csv(run_dir / "data" / "tradeops_sim.csv")
    test_raw = full_data[full_data["split"] == "test"].copy().reset_index(drop=True)
    test_frame = bundle["rsop"].transform(test_raw)
    comparison = pd.read_csv(run_dir / "metrics" / "candidate_comparison.csv")
    predictions = pd.read_csv(run_dir / "metrics" / "predictions.csv")
    slices = pd.read_csv(run_dir / "metrics" / "slice_metrics.csv")
    frontier = pd.read_csv(run_dir / "metrics" / "decision_scenarios.csv")
    release = require_json_object(
        strict_json_loads((run_dir / "release_decision.json").read_text(encoding="utf-8")),
        source="release decision",
    )
    manifest = require_json_object(
        strict_json_loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")),
        source="run manifest",
    )
    calibration = calibration_table(predictions)
    production = predictions[predictions["candidate_id"] == release["production_champion"]]
    labels = np.asarray(sorted(set(production["true_cause"]) | set(production["predicted_cause"])))
    confusion = confusion_records(
        production["true_cause"].to_numpy(), production["predicted_cause"].to_numpy(), labels
    )
    figures = generate_report(
        run_dir,
        full_data,
        test_frame,
        comparison,
        (
            pd.read_csv(run_dir / "metrics" / "stress_comparison.csv")
            if (run_dir / "metrics" / "stress_comparison.csv").exists()
            else pd.DataFrame()
        ),
        calibration,
        frontier,
        slices,
        confusion,
        release,
        manifest,
    )
    artifacts = artifact_manifest(run_dir)
    atomic_json(run_dir / "artifact_manifest.json", artifacts)
    integrity = verify_artifact_manifest(run_dir)
    atomic_json(run_dir / "artifact_integrity.json", integrity)
    return figures
