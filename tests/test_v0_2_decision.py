from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from few_shot_anomaly_poc.v0_2_decision import (
    MethodDecisionEvidence,
    build_method_decision,
    build_project_decision,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    HARD_GATES,
    METHODS,
    load_v0_2_artifact_schema,
    load_v0_2_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _contract() -> tuple[dict, dict]:
    return (
        load_v0_2_config(ROOT / "configs/v0.2.yaml"),
        load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json"),
    )


def _passing() -> MethodDecisionEvidence:
    return MethodDecisionEvidence(
        fit_status="fit_ok",
        test_leakage_detected=False,
        normal_false_positive_rate=0.05,
        anomaly_recall=0.90,
        cpu_median_latency_seconds=0.40,
        cpu_p95_latency_seconds=1.0,
        normal_reference_count=20,
        anomaly_training_labels_used=False,
        reproducibility_verified=True,
    )


def test_every_exact_hard_gate_boundary_passes_without_weighting() -> None:
    config, schema = _contract()

    decision = build_method_decision(
        "ecc_residual",
        _passing(),
        config=config,
        schema=schema,
    )

    assert decision["decision"] == "ADOPT"
    assert decision["first_failed_gate"] is None
    assert [outcome["name"] for outcome in decision["gate_outcomes"]] == list(HARD_GATES)
    assert all(outcome["status"] == "pass" for outcome in decision["gate_outcomes"])
    assert decision["weighted_score_used"] is False
    assert decision["hard_gate_waiver_used"] is False


def test_recall_failure_stops_every_later_gate() -> None:
    config, schema = _contract()
    evidence = replace(_passing(), anomaly_recall=0.34)

    decision = build_method_decision(
        "dinov2_vits14_224_nn",
        evidence,
        config=config,
        schema=schema,
    )

    assert decision["decision"] == "REJECT"
    assert decision["first_failed_gate"] == "final_test_anomaly_recall"
    statuses = [outcome["status"] for outcome in decision["gate_outcomes"]]
    assert statuses == ["pass", "pass", "pass", "fail", *["not_evaluated"] * 4]
    assert all(
        outcome["observed"] is None for outcome in decision["gate_outcomes"][4:]
    )


def test_reference_gate_requires_exactly_twenty_images() -> None:
    config, schema = _contract()

    decision = build_method_decision(
        "patch_hog_ocsvm",
        replace(_passing(), normal_reference_count=19),
        config=config,
        schema=schema,
    )

    assert decision["first_failed_gate"] == "normal_reference_count"
    assert decision["gate_outcomes"][5] == {
        "order": 6,
        "name": "normal_reference_count",
        "status": "fail",
        "observed": 19,
        "operator": "equal",
        "requirement": 20,
    }


def test_all_rejected_methods_force_an_unselected_project_reject() -> None:
    config, schema = _contract()
    evidence = {
        method: replace(_passing(), anomaly_recall=0.10 + index / 100)
        for index, method in enumerate(METHODS)
    }
    decisions = {
        method: build_method_decision(
            method,
            evidence[method],
            config=config,
            schema=schema,
        )
        for method in METHODS
    }

    project = build_project_decision(
        decisions,
        evidence,
        config=config,
        schema=schema,
    )

    assert project["decision"] == "REJECT"
    assert project["selected_method"] is None
    assert project["method_decisions"] == {method: "REJECT" for method in METHODS}
    assert project["weighted_score_used"] is False


def test_non_rejected_selection_uses_the_frozen_lexicographic_order() -> None:
    config, schema = _contract()
    evidence = {
        "ecc_residual": replace(_passing(), anomaly_recall=0.91),
        "patch_hog_ocsvm": replace(_passing(), anomaly_recall=0.95),
        "dinov2_vits14_224_nn": replace(_passing(), anomaly_recall=0.93),
    }
    decisions = {
        method: build_method_decision(
            method,
            evidence[method],
            config=config,
            schema=schema,
        )
        for method in METHODS
    }

    project = build_project_decision(
        decisions,
        evidence,
        config=config,
        schema=schema,
    )

    assert project["decision"] == "ADOPT"
    assert project["selected_method"] == "patch_hog_ocsvm"
