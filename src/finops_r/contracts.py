from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, field_validator, model_validator

from .path_safety import is_portable_component


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @field_validator("*", mode="before")
    @classmethod
    def booleans_must_not_be_coerced_to_numbers(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("boolean values are not permitted in typed contract fields")
        return value


class EnsembleConfig(ContractModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    hist_max_iter: int = Field(ge=1, le=5000)
    hist_learning_rate: float = Field(gt=0.0, le=1.0)
    hist_max_leaf_nodes: int = Field(ge=2, le=4096)
    extra_trees_estimators: int = Field(ge=1, le=5000)
    extra_trees_min_samples_leaf: int = Field(ge=1)
    weight_hist: float = Field(ge=0.0)
    weight_extra: float = Field(ge=0.0)

    @model_validator(mode="after")
    def weights_must_have_positive_mass(self) -> "EnsembleConfig":
        if self.weight_hist + self.weight_extra <= 0.0:
            raise ValueError("ensemble weights must have positive total mass")
        return self


class RSOPConfig(ContractModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    min_reference_count: int = Field(ge=1)
    epsilon: float = Field(gt=0.0, lt=0.5)


class ActionEnvelopeConfig(ContractModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    cumulative_probability: float = Field(gt=0.0, le=1.0)
    max_candidates: int = Field(ge=1, le=20)
    rule_expand_threshold: float = Field(ge=0.0, le=1.0)


class ProjectConfig(ContractModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    project: str = Field(min_length=1)
    config_version: str = Field(min_length=1)
    seed: int = Field(ge=0)
    n_cases: int = Field(ge=200)
    n_entities: int = Field(ge=100)
    output_root: str = Field(min_length=1)
    review_capacity: float = Field(ge=0.0, le=1.0)
    human_correction_probability: float = Field(ge=0.0, le=1.0)
    review_cost: float = Field(ge=0.0)
    hold_cost_fraction: float = Field(ge=0.0, le=1.0)
    hold_fixed_cost: float = Field(ge=0.0)
    critical_false_greenlight_limit: float = Field(ge=0.0, le=1.0)
    evidence_precision_limit: float = Field(ge=0.0, le=1.0)
    min_relative_harm_improvement: float = Field(ge=-1.0, le=1.0)
    max_macro_f1_regression: float = Field(default=0.005, ge=0.0, le=1.0)
    max_worst_slice_macro_f1_regression: float = Field(default=0.05, ge=0.0, le=1.0)
    max_stress_harm_ratio: float = Field(default=1.20, gt=0.0)
    max_hold_rate: float = Field(default=0.18, ge=0.0, le=1.0)
    manual_capacity_tolerance: float = Field(default=0.005, ge=0.0, le=0.05)
    temperature_bounds: tuple[StrictFloat, StrictFloat]
    bootstrap_resamples: int = Field(default=1500, ge=100, le=100_000)
    specialty_resamples: int = Field(default=2000, ge=100, le=100_000)
    ensemble: EnsembleConfig
    rsop: RSOPConfig
    action_envelope: ActionEnvelopeConfig

    @field_validator("output_root")
    @classmethod
    def output_root_must_be_project_relative(cls, value: str) -> str:
        if value != value.strip() or "\\" in value:
            raise ValueError("output_root must use a trimmed POSIX-style relative path")
        path = Path(value)
        if (
            path.is_absolute()
            or PureWindowsPath(value).is_absolute()
            or bool(PureWindowsPath(value).drive)
            or ".." in path.parts
            or path == Path(".")
            or path.as_posix() != value
            or any(not is_portable_component(part) for part in path.parts)
        ):
            raise ValueError(
                "output_root must be a portable project-relative path without parent traversal"
            )
        return value

    @field_validator("project", "config_version")
    @classmethod
    def project_metadata_must_be_trimmed(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("project metadata must be non-empty and trimmed")
        return value

    @model_validator(mode="after")
    def temperature_bounds_must_be_ordered(self) -> "ProjectConfig":
        lower, upper = self.temperature_bounds
        if lower <= 0.0 or lower >= upper:
            raise ValueError("temperature_bounds must be positive and strictly increasing")
        if self.n_entities > self.n_cases:
            raise ValueError("n_entities cannot exceed n_cases")
        return self


class CaseSnapshot(ContractModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    case_id: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    decision_time: datetime
    state: Literal["EXECUTED", "ALLOCATED", "CONFIRMED", "INSTRUCTED", "MATCHED"]
    product: Literal["EQUITY", "ETF", "CORP_BOND", "OPTION"]
    route: Literal["DTC", "FEDWIRE", "EUROCLEAR", "UNKNOWN"]
    currency: Literal["USD", "EUR", "GBP"]
    side: Literal["BUY", "SELL"]
    time_in_state_min: float = Field(ge=0)
    case_age_min: float = Field(ge=0)
    deadline_slack_min: float
    event_count: int = Field(ge=0)
    artifact_count: int = Field(ge=0)
    missing_doc_count: int = Field(ge=0)
    ssi_version_gap: float | None = Field(default=None, ge=0.0)
    qty_diff_pct: float | None = Field(default=None, ge=0.0)
    price_diff_bps: float | None = Field(default=None, ge=0.0)
    duplicate_amendment_count: float | None = Field(default=None, ge=0.0)
    funding_ratio: float | None = Field(default=None, ge=0.0)
    securities_ratio: float | None = Field(default=None, ge=0.0)
    compliance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    allocation_present: float | None = Field(default=None, ge=0.0, le=1.0)
    affirmation_present: float | None = Field(default=None, ge=0.0, le=1.0)
    ssi_present: float | None = Field(default=None, ge=0.0, le=1.0)
    currency_match: float | None = Field(default=None, ge=0.0, le=1.0)
    counterparty_match: float | None = Field(default=None, ge=0.0, le=1.0)
    security_id_match: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_complete: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("case_id", "entity_id")
    @classmethod
    def identifiers_must_be_trimmed(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("identifiers must be non-empty and trimmed")
        return value

    @field_validator(
        "allocation_present",
        "affirmation_present",
        "ssi_present",
        "currency_match",
        "counterparty_match",
        "security_id_match",
        "evidence_complete",
    )
    @classmethod
    def binary_observables_must_be_binary(cls, value: float | None) -> float | None:
        if value is not None and value not in {0.0, 1.0}:
            raise ValueError("binary observables must be zero or one")
        return value

    @field_validator("decision_time")
    @classmethod
    def decision_time_requires_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("decision_time must be timezone-aware")
        return value.astimezone(timezone.utc)


class EvidenceItem(ContractModel):
    evidence_id: str
    reason_code: str
    strength: float = Field(ge=0.0, le=1.0)
    observable_at: datetime


class ActionEnvelope(ContractModel):
    candidate_actions: list[str]
    expanded_actions: list[str]
    pruned_actions: list[str]
    evidence_ids: list[str]
    coverage_probability: float = Field(ge=0.0, le=1.0)
    overreach_count: int = Field(ge=0)


class Assessment(ContractModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    case_id: str
    predicted_cause: str
    proposed_action: str
    confidence: float = Field(ge=0.0, le=1.0)
    calibrated_error_risk: float = Field(ge=0.0, le=1.0)
    route: Literal["SIMULATE", "REVIEW", "HOLD"]
    reason_codes: list[str]
    evidence_ids: list[str]
    policy_status: Literal["PASS", "BLOCK", "NOT_APPLICABLE"]
    verifier_status: Literal["SUPPORTED", "INSUFFICIENT", "CONTRADICTED", "NOT_APPLICABLE"]
    model_version: str
    run_id: str
    claim_boundary: str
    metadata: dict[str, Any] = Field(default_factory=dict)
