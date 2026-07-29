"""Apply fixed v0.1 hard gates without a weighted aggregate score."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from few_shot_anomaly_poc.calibration import CalibrationMethod
from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.cpu_latency import (
    CPULatencyResult,
    cpu_latency_result_is_valid,
)
from few_shot_anomaly_poc.errors import HardGateDecisionFailureCode
from few_shot_anomaly_poc.image_metrics import (
    ImageLevelMetricsResult,
    image_level_metrics_result_is_valid,
)

type GateIdentifier = Literal[
    "final_test_normal_fpr",
    "final_test_anomaly_recall",
    "cpu_p95_scoring_latency",
    "normal_reference_count",
    "anomaly_training_labels",
    "reproducibility",
]
type GateOperator = Literal["less_than_or_equal", "greater_than_or_equal", "equal"]
type GateValue = float | int | bool
type AdoptionDecision = Literal["ADOPT", "ADOPT WITH CONDITIONS", "REJECT"]


class FailureReviewDisposition(StrEnum):
    """Allowed qualitative dispositions after mechanical case selection."""

    NO_MATERIAL_BOUNDARY = "no_material_boundary"
    GUARDRAIL_REQUIRED = "guardrail_required"
    INTENDED_USE_CONTRADICTED = "intended_use_contradicted"


@dataclass(frozen=True)
class DecisionProcessEvidence:
    """Explicit process evidence used by non-metric decision gates."""

    normal_reference_count: int
    anomaly_training_labels_used: bool
    reproducibility_verified: bool
    test_leakage_detected: bool
    failure_review_disposition: FailureReviewDisposition
    failure_review_rationale: str
    condition: str | None


@dataclass(frozen=True)
class HardGateOutcome:
    """One ordered, independently evaluated preregistered gate."""

    gate_id: GateIdentifier
    order: int
    operator: GateOperator
    required_value: GateValue
    observed_value: GateValue
    passed: bool


@dataclass(frozen=True)
class HardGateDecisionResult:
    """One method decision produced after applying every fixed hard gate."""

    status: Literal["ok", "DECISION_FAILED"]
    failure_code: HardGateDecisionFailureCode | None
    method: CalibrationMethod | None
    decision: AdoptionDecision | None
    gate_outcomes: tuple[HardGateOutcome, ...] | None
    all_hard_gates_passed: bool | None
    first_failed_gate: GateIdentifier | None
    test_leakage_detected: bool | None
    failure_review_disposition: FailureReviewDisposition | None
    failure_review_rationale: str | None
    condition: str | None
    decision_reason: str | None

    @property
    def succeeded(self) -> bool:
        """Return whether a complete method decision was produced."""
        return self.status == "ok"


def _failed(
    code: HardGateDecisionFailureCode,
    *,
    metrics: object,
    latency: object,
) -> HardGateDecisionResult:
    method = None
    for evidence in (metrics, latency):
        candidate = getattr(evidence, "method", None)
        if isinstance(candidate, CalibrationMethod):
            method = candidate
            break
    return HardGateDecisionResult(
        status="DECISION_FAILED",
        failure_code=code,
        method=method,
        decision=None,
        gate_outcomes=None,
        all_hard_gates_passed=None,
        first_failed_gate=None,
        test_leakage_detected=None,
        failure_review_disposition=None,
        failure_review_rationale=None,
        condition=None,
        decision_reason=None,
    )


def _bounded_text_is_valid(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value == value.strip()
        and len(value) <= 500
    )


def _process_evidence_is_valid(evidence: object) -> bool:
    if (
        not isinstance(evidence, DecisionProcessEvidence)
        or not isinstance(evidence.normal_reference_count, int)
        or isinstance(evidence.normal_reference_count, bool)
        or evidence.normal_reference_count < 1
        or not isinstance(evidence.anomaly_training_labels_used, bool)
        or not isinstance(evidence.reproducibility_verified, bool)
        or not isinstance(evidence.test_leakage_detected, bool)
        or not isinstance(evidence.failure_review_disposition, FailureReviewDisposition)
        or not _bounded_text_is_valid(evidence.failure_review_rationale)
    ):
        return False
    if evidence.failure_review_disposition is FailureReviewDisposition.GUARDRAIL_REQUIRED:
        return _bounded_text_is_valid(evidence.condition)
    return evidence.condition is None


def _gate_outcomes(
    metrics: ImageLevelMetricsResult,
    latency: CPULatencyResult,
    evidence: DecisionProcessEvidence,
    *,
    config: ProjectConfig,
) -> tuple[HardGateOutcome, ...]:
    assert metrics.normal_false_positive_rate is not None
    assert metrics.anomaly_recall is not None
    assert latency.p95_latency_seconds is not None
    rules: dict[GateIdentifier, tuple[GateOperator, GateValue, GateValue, bool]] = {
        "final_test_normal_fpr": (
            "less_than_or_equal",
            config.hard_gate_decision.normal_fpr_max,
            metrics.normal_false_positive_rate,
            metrics.normal_false_positive_rate <= config.hard_gate_decision.normal_fpr_max,
        ),
        "final_test_anomaly_recall": (
            "greater_than_or_equal",
            config.hard_gate_decision.anomaly_recall_min,
            metrics.anomaly_recall,
            metrics.anomaly_recall >= config.hard_gate_decision.anomaly_recall_min,
        ),
        "cpu_p95_scoring_latency": (
            "less_than_or_equal",
            config.hard_gate_decision.cpu_p95_latency_seconds_max,
            latency.p95_latency_seconds,
            latency.p95_latency_seconds <= config.hard_gate_decision.cpu_p95_latency_seconds_max,
        ),
        "normal_reference_count": (
            "less_than_or_equal",
            config.hard_gate_decision.normal_reference_count_max,
            evidence.normal_reference_count,
            evidence.normal_reference_count <= config.hard_gate_decision.normal_reference_count_max,
        ),
        "anomaly_training_labels": (
            "equal",
            config.hard_gate_decision.anomaly_training_labels_used,
            evidence.anomaly_training_labels_used,
            evidence.anomaly_training_labels_used
            is config.hard_gate_decision.anomaly_training_labels_used,
        ),
        "reproducibility": (
            "equal",
            config.hard_gate_decision.reproducibility_required,
            evidence.reproducibility_verified,
            evidence.reproducibility_verified is config.hard_gate_decision.reproducibility_required,
        ),
    }
    return tuple(
        HardGateOutcome(
            gate_id=gate_id,
            order=order,
            operator=rules[gate_id][0],
            required_value=rules[gate_id][1],
            observed_value=rules[gate_id][2],
            passed=rules[gate_id][3],
        )
        for order, gate_id in enumerate(config.hard_gate_decision.gate_order, start=1)
    )


def _decision(
    outcomes: tuple[HardGateOutcome, ...],
    evidence: DecisionProcessEvidence,
) -> tuple[AdoptionDecision, str]:
    if evidence.test_leakage_detected:
        return ("REJECT", "test_leakage_detected")
    if not all(outcome.passed for outcome in outcomes):
        return ("REJECT", "hard_gate_failed")
    if evidence.failure_review_disposition is FailureReviewDisposition.INTENDED_USE_CONTRADICTED:
        return ("REJECT", "failure_review_contradicts_intended_use")
    if evidence.failure_review_disposition is FailureReviewDisposition.GUARDRAIL_REQUIRED:
        return ("ADOPT WITH CONDITIONS", "all_hard_gates_passed_guardrail_required")
    return ("ADOPT", "all_hard_gates_passed")


def _result_is_valid(
    result: HardGateDecisionResult,
    *,
    expected_outcomes: tuple[HardGateOutcome, ...],
    evidence: DecisionProcessEvidence,
) -> bool:
    expected_decision, expected_reason = _decision(expected_outcomes, evidence)
    expected_first_failure = next(
        (outcome.gate_id for outcome in expected_outcomes if not outcome.passed),
        None,
    )
    expected_condition = (
        evidence.condition
        if (
            expected_decision == "ADOPT WITH CONDITIONS"
            and evidence.failure_review_disposition is FailureReviewDisposition.GUARDRAIL_REQUIRED
        )
        else None
    )
    return (
        result.succeeded
        and result.failure_code is None
        and isinstance(result.method, CalibrationMethod)
        and result.decision == expected_decision
        and result.gate_outcomes == expected_outcomes
        and result.all_hard_gates_passed is all(outcome.passed for outcome in expected_outcomes)
        and result.first_failed_gate == expected_first_failure
        and result.test_leakage_detected is evidence.test_leakage_detected
        and result.failure_review_disposition is evidence.failure_review_disposition
        and result.failure_review_rationale == evidence.failure_review_rationale
        and result.condition == expected_condition
        and result.decision_reason == expected_reason
    )


def apply_hard_gate_decision(
    metrics: ImageLevelMetricsResult,
    latency: CPULatencyResult,
    process_evidence: DecisionProcessEvidence,
    *,
    config: ProjectConfig,
) -> HardGateDecisionResult:
    """Apply six fixed gates, then the explicit failure-review disposition."""
    if not image_level_metrics_result_is_valid(metrics):
        return _failed(
            HardGateDecisionFailureCode.DECISION_METRICS_INVALID,
            metrics=metrics,
            latency=latency,
        )
    if not cpu_latency_result_is_valid(latency, config=config):
        return _failed(
            HardGateDecisionFailureCode.DECISION_LATENCY_INVALID,
            metrics=metrics,
            latency=latency,
        )
    if metrics.method is not latency.method or metrics.item_count != latency.item_count:
        return _failed(
            HardGateDecisionFailureCode.DECISION_METHOD_MISMATCH,
            metrics=metrics,
            latency=latency,
        )
    if not _process_evidence_is_valid(process_evidence):
        return _failed(
            HardGateDecisionFailureCode.DECISION_PROCESS_EVIDENCE_INVALID,
            metrics=metrics,
            latency=latency,
        )

    outcomes = _gate_outcomes(
        metrics,
        latency,
        process_evidence,
        config=config,
    )
    decision, reason = _decision(outcomes, process_evidence)
    first_failed_gate = next(
        (outcome.gate_id for outcome in outcomes if not outcome.passed),
        None,
    )
    condition = process_evidence.condition if decision == "ADOPT WITH CONDITIONS" else None
    result = HardGateDecisionResult(
        status="ok",
        failure_code=None,
        method=metrics.method,
        decision=decision,
        gate_outcomes=outcomes,
        all_hard_gates_passed=all(outcome.passed for outcome in outcomes),
        first_failed_gate=first_failed_gate,
        test_leakage_detected=process_evidence.test_leakage_detected,
        failure_review_disposition=process_evidence.failure_review_disposition,
        failure_review_rationale=process_evidence.failure_review_rationale,
        condition=condition,
        decision_reason=reason,
    )
    if not _result_is_valid(
        result,
        expected_outcomes=outcomes,
        evidence=process_evidence,
    ):
        return _failed(
            HardGateDecisionFailureCode.DECISION_RESULT_INVALID,
            metrics=metrics,
            latency=latency,
        )
    return result
