"""Apply the frozen v0.2 ordered hard gates and project selection rule."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from few_shot_anomaly_poc.v0_2_boundary_preparation import RUN_ID
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    ARTIFACT_CONTRACT_VERSION,
    HARD_GATES,
    METHODS,
    validate_json_artifact,
)

RUN_KIND = "final_test"
NEXT_VALIDATION = (
    "Preregister a development-only pcb2 diagnostic study that reviews only the "
    "mechanically selected v0.2.7 cases and evaluates normal-score transfer without "
    "tuning or rescoring the completed final-test boundary."
)


class V0_2DecisionError(Exception):
    """Reject incomplete evidence or a decision outside the frozen rules."""


@dataclass(frozen=True)
class MethodDecisionEvidence:
    """Hold every fixed hard-gate input for one method."""

    fit_status: str
    test_leakage_detected: bool
    normal_false_positive_rate: float
    anomaly_recall: float
    cpu_median_latency_seconds: float
    cpu_p95_latency_seconds: float
    normal_reference_count: int
    anomaly_training_labels_used: bool
    reproducibility_verified: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V0_2DecisionError(message)


def _rules(
    evidence: MethodDecisionEvidence,
    *,
    config: Mapping[str, Any],
) -> dict[str, tuple[Any, str, Any, bool]]:
    gates = config["hard_gates"]
    return {
        "method_fit": (
            evidence.fit_status,
            "equal",
            "fit_ok",
            evidence.fit_status == "fit_ok",
        ),
        "test_leakage": (
            evidence.test_leakage_detected,
            "equal",
            False,
            evidence.test_leakage_detected is False,
        ),
        "final_test_normal_fpr": (
            evidence.normal_false_positive_rate,
            "less_than_or_equal",
            gates["normal_fpr_max"],
            evidence.normal_false_positive_rate <= gates["normal_fpr_max"],
        ),
        "final_test_anomaly_recall": (
            evidence.anomaly_recall,
            "greater_than_or_equal",
            gates["anomaly_recall_min"],
            evidence.anomaly_recall >= gates["anomaly_recall_min"],
        ),
        "cpu_p95_scoring_latency": (
            evidence.cpu_p95_latency_seconds,
            "less_than_or_equal",
            gates["cpu_p95_latency_seconds_max"],
            evidence.cpu_p95_latency_seconds <= gates["cpu_p95_latency_seconds_max"],
        ),
        "normal_reference_count": (
            evidence.normal_reference_count,
            "equal",
            gates["normal_reference_count_required"],
            evidence.normal_reference_count == gates["normal_reference_count_required"],
        ),
        "anomaly_training_labels": (
            evidence.anomaly_training_labels_used,
            "equal",
            gates["anomaly_training_labels_used"],
            evidence.anomaly_training_labels_used
            is gates["anomaly_training_labels_used"],
        ),
        "reproducibility": (
            evidence.reproducibility_verified,
            "equal",
            gates["reproducibility_required"],
            evidence.reproducibility_verified is gates["reproducibility_required"],
        ),
    }


def build_method_decision(
    method: str,
    evidence: MethodDecisionEvidence,
    *,
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Stop at the first failed gate and produce one validated method decision."""
    _require(method in METHODS, "method is outside the fixed inventory")
    _require(
        tuple(config["hard_gates"]["order"]) == HARD_GATES,
        "hard-gate order changed",
    )
    rules = _rules(evidence, config=config)
    first_failed_gate: str | None = None
    outcomes: list[dict[str, Any]] = []
    for order, gate in enumerate(HARD_GATES, start=1):
        observed, operator, requirement, passed = rules[gate]
        if first_failed_gate is not None:
            status = "not_evaluated"
            observed = None
        elif passed:
            status = "pass"
        else:
            status = "fail"
            first_failed_gate = gate
        outcomes.append(
            {
                "order": order,
                "name": gate,
                "status": status,
                "observed": observed,
                "operator": operator,
                "requirement": requirement,
            }
        )

    if first_failed_gate is None:
        decision = "ADOPT"
        reason = "Every ordered hard gate passed; no weighted score or waiver was used."
    else:
        failed = outcomes[HARD_GATES.index(first_failed_gate)]
        decision = "REJECT"
        reason = (
            f"The first failed ordered hard gate was {first_failed_gate}: observed "
            f"{failed['observed']!r} with requirement {failed['operator']} "
            f"{failed['requirement']!r}. Later gates were not evaluated. Selected images "
            "were not reviewed because this method was not otherwise gate-passing."
        )
    record = {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "run_id": RUN_ID,
        "run_kind": RUN_KIND,
        "method": method,
        "decision": decision,
        "gate_outcomes": outcomes,
        "first_failed_gate": first_failed_gate,
        # The frozen schema requires a non-null disposition. For an early hard-gate
        # rejection this neutral value is not an assertion that images were reviewed.
        "failure_review_disposition": "no_material_boundary",
        "failure_review_rationale": None,
        "condition": None,
        "decision_reason": reason,
        "weighted_score_used": False,
        "hard_gate_waiver_used": False,
    }
    return validate_json_artifact(
        "method_decision",
        record,
        config=config,
        schema=schema,
    )


def build_project_decision(
    decisions: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, MethodDecisionEvidence],
    *,
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Select only among non-rejected methods using the frozen lexicographic order."""
    _require(set(decisions) == set(METHODS), "method decision inventory changed")
    _require(set(evidence) == set(METHODS), "method evidence inventory changed")
    method_decisions = {method: decisions[method]["decision"] for method in METHODS}
    candidates = [method for method in METHODS if method_decisions[method] != "REJECT"]
    if not candidates:
        selected_method = None
        decision = "REJECT"
        trace = [
            "All three fixed methods are REJECT under their first failed ordered hard gate.",
            (
                "No method is selected; descriptive ranking metrics and latency cannot "
                "waive a failure."
            ),
        ]
        reason = (
            "No fixed method passed every preregistered hard gate, so the project decision is "
            "REJECT."
        )
    else:
        decision_rank = {"ADOPT": 0, "ADOPT WITH CONDITIONS": 1}
        complexity_rank = {method: index for index, method in enumerate(METHODS)}
        selected_method = min(
            candidates,
            key=lambda method: (
                decision_rank[method_decisions[method]],
                -evidence[method].anomaly_recall,
                evidence[method].normal_false_positive_rate,
                evidence[method].cpu_p95_latency_seconds,
                evidence[method].cpu_median_latency_seconds,
                complexity_rank[method],
            ),
        )
        decision = method_decisions[selected_method]
        trace = [
            "Non-rejected methods were compared by the frozen lexicographic selection order.",
            f"{selected_method} ranked first without a weighted aggregate score.",
        ]
        reason = (
            f"{selected_method} is the fixed next-validation candidate; this is not a deployment "
            "recommendation."
        )
    record = {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "run_id": RUN_ID,
        "run_kind": RUN_KIND,
        "decision": decision,
        "selected_method": selected_method,
        "method_decisions": method_decisions,
        "selection_trace": trace,
        "decision_reason": reason,
        "next_validation": NEXT_VALIDATION,
        "weighted_score_used": False,
    }
    return validate_json_artifact(
        "project_decision",
        record,
        config=config,
        schema=schema,
    )
