from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "finops-r-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .artifacts import atomic_csv, atomic_text


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _target_distribution(data: pd.DataFrame, figures: Path, underlying: Path) -> None:
    table = (
        data.groupby(["split", "root_cause"], observed=True).size().rename("count").reset_index()
    )
    atomic_csv(underlying / "target_distribution.csv", table)
    pivot = table.pivot(index="root_cause", columns="split", values="count").fillna(0)
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("TradeOps-Sim target distribution by split")
    ax.set_ylabel("Cases")
    ax.set_xlabel("Root cause")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(title="Split", fontsize=8)
    _save_figure(fig, figures / "01_target_distribution.png")


def _candidate_comparison(comparison: pd.DataFrame, figures: Path, underlying: Path) -> None:
    atomic_csv(underlying / "candidate_comparison.csv", comparison)
    ordered = comparison.sort_values("candidate_id")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(ordered["candidate_id"], ordered["macro_f1"], color="#3A78B4")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Predictive quality")
    axes[0].set_ylabel("Macro-F1")
    axes[1].bar(ordered["candidate_id"], ordered["residual_harm_per_1000"], color="#C65D47")
    axes[1].set_title("Decision outcome at fixed review capacity")
    axes[1].set_ylabel("Residual harm per 1,000 cases")
    fig.suptitle("Candidate ladder under one evaluation contract")
    _save_figure(fig, figures / "02_candidate_comparison.png")


def _calibration(calibration: pd.DataFrame, figures: Path, underlying: Path) -> None:
    atomic_csv(underlying / "calibration_curve.csv", calibration)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for candidate_id, group in calibration.groupby("candidate_id", observed=True):
        ax.plot(
            group["mean_confidence"], group["observed_accuracy"], marker="o", label=candidate_id
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Ideal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Top-label calibration")
    ax.legend()
    _save_figure(fig, figures / "03_calibration.png")


def _capacity_frontier(
    frontier: pd.DataFrame, figures: Path, underlying: Path, primary_capacity: float
) -> None:
    atomic_csv(underlying / "capacity_frontier.csv", frontier)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for candidate_id, group in frontier.groupby("candidate_id", observed=True):
        ax.plot(
            group["review_capacity"],
            group["residual_harm_per_1000"],
            marker="o",
            label=candidate_id,
        )
    ax.axvline(
        primary_capacity, color="black", linestyle="--", linewidth=1, label="Primary capacity"
    )
    ax.set_xlabel("Review capacity")
    ax.set_ylabel("Residual harm per 1,000")
    ax.set_title("Automation and human-review frontier")
    ax.legend()
    _save_figure(fig, figures / "04_capacity_risk_frontier.png")


def _rsop_alignment(test_frame: pd.DataFrame, figures: Path, underlying: Path) -> None:
    sample = test_frame.sample(n=min(700, len(test_frame)), random_state=17).copy()
    atomic_csv(
        underlying / "rsop_alignment.csv",
        sample[["state", "product", "time_in_state_min", "rsop_percentile", "root_cause"]],
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = sample["state"].astype("category").cat.codes
    scatter = ax.scatter(
        sample["time_in_state_min"],
        sample["rsop_percentile"],
        c=colors,
        cmap="viridis",
        alpha=0.62,
        s=20,
    )
    ax.set_xlabel("Raw time in state, minutes")
    ax.set_ylabel("Train-only RSOP percentile")
    ax.set_title("Raw time versus reference-state aligned progress")
    legend = ax.legend(*scatter.legend_elements(), title="State code", loc="lower right")
    ax.add_artist(legend)
    _save_figure(fig, figures / "05_rsop_alignment.png")


def _worst_slices(slices: pd.DataFrame, figures: Path, underlying: Path) -> None:
    atomic_csv(underlying / "slice_metrics.csv", slices)
    subset = slices[slices["candidate_id"].isin(["C1", "C5"])].copy()
    subset["slice"] = subset["slice_name"] + ":" + subset["slice_value"]
    worst_names = subset.groupby("slice")["macro_f1"].min().sort_values().head(12).index.tolist()
    plot_data = subset[subset["slice"].isin(worst_names)]
    pivot = plot_data.pivot_table(
        index="slice", columns="candidate_id", values="macro_f1", aggfunc="first"
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    pivot.plot(kind="barh", ax=ax, color=["#777777", "#3A78B4"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1")
    ax.set_ylabel("Slice")
    ax.set_title("Lowest-performing slices for baseline and production candidate")
    _save_figure(fig, figures / "06_worst_slices.png")


def _confusion(confusion: pd.DataFrame, figures: Path, underlying: Path) -> None:
    atomic_csv(underlying / "production_confusion.csv", confusion)
    labels = list(dict.fromkeys(confusion["true_label"].tolist()))
    matrix = (
        confusion.pivot(index="true_label", columns="predicted_label", values="count")
        .reindex(index=labels, columns=labels)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=70, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Production-candidate confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.03)
    _save_figure(fig, figures / "07_confusion_matrix.png")


def _stress_comparison(comparison: pd.DataFrame, figures: Path, underlying: Path) -> None:
    atomic_csv(underlying / "stress_comparison.csv", comparison)
    subset = comparison[comparison["candidate_id"].isin(["C1", "C5"])].copy()
    subset = subset.sort_values("candidate_id")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    axes[0].bar(subset["candidate_id"], subset["macro_f1"], color=["#777777", "#3A78B4"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Stress predictive quality")
    axes[0].set_ylabel("Macro-F1")
    axes[1].bar(
        subset["candidate_id"],
        subset["residual_harm_per_1000"],
        color=["#777777", "#C65D47"],
    )
    axes[1].set_title("Stress decision outcome")
    axes[1].set_ylabel("Residual harm per 1,000")
    fig.suptitle("Entity-disjoint shifted stress split")
    _save_figure(fig, figures / "08_stress_comparison.png")


def generate_report(
    run_dir: Path,
    full_data: pd.DataFrame,
    test_frame: pd.DataFrame,
    comparison: pd.DataFrame,
    stress_comparison: pd.DataFrame,
    calibration: pd.DataFrame,
    frontier: pd.DataFrame,
    slices: pd.DataFrame,
    confusion: pd.DataFrame,
    release: dict[str, Any],
    run_manifest: dict[str, Any],
) -> list[str]:
    figures = run_dir / "figures"
    underlying = run_dir / "visualization_data"
    figures.mkdir(parents=True, exist_ok=True)
    underlying.mkdir(parents=True, exist_ok=True)
    _target_distribution(full_data, figures, underlying)
    _candidate_comparison(comparison, figures, underlying)
    _calibration(calibration, figures, underlying)
    _capacity_frontier(frontier, figures, underlying, float(release["review_capacity"]))
    _rsop_alignment(test_frame, figures, underlying)
    _worst_slices(slices, figures, underlying)
    _confusion(confusion, figures, underlying)
    if not stress_comparison.empty:
        _stress_comparison(stress_comparison, figures, underlying)

    champion = release["production_champion"]
    champion_row = comparison.loc[comparison["candidate_id"] == champion].iloc[0].to_dict()
    stress_summary = ""
    if "stress" in release:
        stress_summary = (
            f"- Stress C5 minus C1 Macro-F1: `{release['stress']['macro_f1_delta']:.4f}`\n"
            f"- Stress C5/C1 residual-harm ratio: `{release['stress']['harm_ratio']:.4f}`"
        )
    summary = f"""# FinOps-R Run Report

## Scope

This report evaluates controlled TradeOps-Sim cases. It is not evidence of production bank impact or regulatory compliance.

## Release decision

- Run ID: `{run_manifest["run_id"]}`
- Status: `{release["status"]}`
- Predictive champion: `{release["predictive_champion"]}`
- Production champion: `{champion}`
- Safety champion: `{release["safety_champion"]}`
- Primary comparator: `C1`
- Primary review capacity: `{run_manifest["review_capacity"]:.0%}`
{stress_summary}

## Production champion metrics

- Macro-F1: `{champion_row["macro_f1"]:.4f}`
- Multiclass Brier: `{champion_row["multiclass_brier"]:.4f}`
- Top-label ECE: `{champion_row["top_label_ece"]:.4f}`
- Residual harm per 1,000: `{champion_row["residual_harm_per_1000"]:.2f}`
- Unsafe automated-action harm per 1,000: `{champion_row["unsafe_automated_harm_per_1000"]:.2f}`
- Critical false-greenlight rate: `{champion_row["critical_false_greenlight_rate"]:.4f}`
- Review rate: `{champion_row["review_rate"]:.4f}`
- Hold rate: `{champion_row["hold_rate"]:.4f}`
- p95 inference latency, ms/case: `{champion_row["latency_ms_per_case_p95"]:.4f}`

## Primary gate

{release["rationale"]}

## Figures

1. `figures/01_target_distribution.png`: data and split composition.
2. `figures/02_candidate_comparison.png`: predictive and decision outcomes.
3. `figures/03_calibration.png`: confidence reliability.
4. `figures/04_capacity_risk_frontier.png`: decision sensitivity to review capacity.
5. `figures/05_rsop_alignment.png`: raw versus state-aligned progress.
6. `figures/06_worst_slices.png`: beyond-average reliability.
7. `figures/07_confusion_matrix.png`: failure structure.
8. `figures/08_stress_comparison.png`: shifted-split quality and decision harm.

Every figure has CSV underpinnings in `visualization_data/` and an SVG counterpart.
"""
    atomic_text(run_dir / "REPORT.md", summary)

    rows = comparison.to_html(index=False, float_format=lambda value: f"{value:.5f}")
    gate_rows = "".join(
        f"<li><b>{name}</b>: {value}</li>" for name, value in release["gates"].items()
    )
    image_tags = "".join(
        f'<section><img src="figures/{path.name}" alt="{path.stem}" style="max-width:100%"></section>'
        for path in sorted(figures.glob("*.png"))
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>FinOps-R {run_manifest["run_id"]}</title>
<style>body{{font-family:Arial,sans-serif;max-width:1180px;margin:2rem auto;line-height:1.45}} table{{border-collapse:collapse;font-size:12px}} th,td{{border:1px solid #ccc;padding:5px}} img{{border:1px solid #ddd;margin:1rem 0}}</style></head>
<body><h1>FinOps-R controlled benchmark</h1><p><b>Claim boundary:</b> controlled simulation only.</p>
<h2>Release</h2><p>Status: <b>{release["status"]}</b>; predictive champion: <b>{release["predictive_champion"]}</b>; safety champion: <b>{release["safety_champion"]}</b>; production champion: <b>{champion}</b>.</p>
<ul>{gate_rows}</ul><h2>Candidate comparison</h2>{rows}<h2>Diagnostics</h2>{image_tags}</body></html>"""
    atomic_text(run_dir / "REPORT.html", html)
    return [str(path.relative_to(run_dir)) for path in sorted(figures.glob("*"))]


def calibration_table(
    predictions: pd.DataFrame, candidate_ids: tuple[str, ...] = ("C1", "C5")
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    edges = np.linspace(0, 1, 11)
    for candidate_id in candidate_ids:
        subset = predictions[predictions["candidate_id"] == candidate_id]
        for bin_id, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            mask = (subset["confidence"] > lower) & (subset["confidence"] <= upper)
            if not mask.any():
                continue
            group = subset[mask]
            records.append(
                {
                    "candidate_id": candidate_id,
                    "bin": bin_id,
                    "lower": lower,
                    "upper": upper,
                    "support": len(group),
                    "mean_confidence": float(group["confidence"].mean()),
                    "observed_accuracy": float(group["correct"].mean()),
                }
            )
    return pd.DataFrame(records)
