from __future__ import annotations

from pathlib import Path

from few_shot_anomaly_poc.opaque_boundary import run_synthetic_boundary_feasibility
from few_shot_anomaly_poc.v0_2_preflight_decision import (
    FIXED_EVIDENCE,
    evaluate_fixed_conditions,
    ordered_preflight_decision,
)

ROOT = Path(__file__).resolve().parents[1]


def test_every_fixed_preflight_evidence_file_has_exact_identity() -> None:
    feasibility = {
        "boundary": {
            "dataset_access": False,
            "dataset_labels_accessed": False,
            "official_split_access": False,
        },
        "checks": {"synthetic_check": True},
        "decision": {"status": "pass"},
        "schema_version": "v0.2-opaque-boundary-feasibility-v1",
    }
    conditions = evaluate_fixed_conditions(
        project_root=ROOT,
        execution_identity={"worktree_clean": True},
        boundary_feasibility=feasibility,
    )

    assert len(FIXED_EVIDENCE) == 15
    assert [condition["condition"] for condition in conditions] == list(range(1, 11))
    assert all(condition["passed"] for condition in conditions)
    assert all(
        evidence["verification"] == "pass"
        for condition in conditions[:9]
        for evidence in condition["evidence"]
    )


def test_ordered_hard_gate_proceeds_only_when_all_ten_conditions_pass() -> None:
    conditions = [
        {
            "condition": number,
            "evidence": [],
            "name": f"condition_{number}",
            "passed": True,
            "reason": "fixed fixture",
        }
        for number in range(1, 11)
    ]

    decision = ordered_preflight_decision(conditions)

    assert decision["decision"] == {
        "first_failed_condition": None,
        "next_step": "CREATE_SEPARATE_V0_2_METHOD_AND_EVALUATION_PREREGISTRATION",
        "outcome": "PROCEED",
        "weighted_score_used": False,
    }
    assert all(condition["status"] == "pass" for condition in decision["conditions"])


def test_ordered_hard_gate_cannot_waive_an_earlier_failure() -> None:
    conditions = [
        {
            "condition": number,
            "evidence": [],
            "name": f"condition_{number}",
            "passed": number != 8,
            "reason": "fixed fixture",
        }
        for number in range(1, 11)
    ]

    decision = ordered_preflight_decision(conditions)

    assert decision["decision"] == {
        "first_failed_condition": 8,
        "next_step": "STOP_V0_2_PREFLIGHT",
        "outcome": "DO_NOT_PROCEED",
        "weighted_score_used": False,
    }
    assert [condition["status"] for condition in decision["conditions"]] == [
        "pass",
        "pass",
        "pass",
        "pass",
        "pass",
        "pass",
        "pass",
        "fail",
        "not_evaluated",
        "not_evaluated",
    ]


def test_real_synthetic_boundary_feasibility_closes_condition_ten(
    tmp_path: Path,
) -> None:
    feasibility = run_synthetic_boundary_feasibility(tmp_path / "checkpoint")
    conditions = evaluate_fixed_conditions(
        project_root=ROOT,
        execution_identity={"worktree_clean": True},
        boundary_feasibility=feasibility,
    )

    assert conditions[-1]["name"] == "evaluation_boundary"
    assert conditions[-1]["passed"] is True
