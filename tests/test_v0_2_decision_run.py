from __future__ import annotations

from copy import deepcopy

import pytest

from few_shot_anomaly_poc.v0_2_decision_run import (
    V0_2DecisionRunError,
    _artifact_type,
    _validate_state,
)


def _state() -> dict:
    return {
        "schema_version": "v0.2-boundary-state-v1",
        "run_id": "visa-pcb2-v0-2-final",
        "boundary": {"final_test_label_revealed": True},
        "label_free_scoring": {
            "anomaly_labels_used": False,
            "semantic_paths_accessed": False,
            "sealed_mapping_accessed": False,
            "official_split_accessed": False,
        },
        "normal_only_fit_calibration": {
            "anomaly_labels_used": False,
            "final_test_accessed": False,
        },
        "offline_reproduction": {
            "labels_accessed": False,
            "semantic_paths_accessed": False,
            "sealed_mapping_accessed": False,
            "official_split_accessed": False,
        },
        "pre_reveal_checkpoint": {
            "git_push_verified": True,
            "labels_accessed": False,
        },
        "label_reveal_evaluation": {
            "milestone": "v0.2.7",
            "status": "completed",
            "labels_accessed": True,
            "image_content_accessed": False,
            "image_content_displayed": False,
            "official_split_accessed": False,
            "score_or_threshold_modified": False,
            "decision_generated": False,
        },
    }


def test_fixed_post_reveal_state_is_eligible_for_decision() -> None:
    _validate_state(_state())


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("label_free_scoring", "anomaly_labels_used"),
        ("label_free_scoring", "semantic_paths_accessed"),
        ("normal_only_fit_calibration", "final_test_accessed"),
        ("offline_reproduction", "labels_accessed"),
        ("label_reveal_evaluation", "score_or_threshold_modified"),
        ("label_reveal_evaluation", "image_content_accessed"),
    ],
)
def test_unsafe_or_changed_process_state_is_rejected(section: str, field: str) -> None:
    state = deepcopy(_state())
    state[section][field] = True

    with pytest.raises(V0_2DecisionRunError, match="pre-decision state"):
        _validate_state(state)


def test_decision_state_prevents_a_second_generation() -> None:
    state = _state()
    state["hard_gate_decision"] = {"status": "completed"}

    with pytest.raises(V0_2DecisionRunError, match="pre-decision state"):
        _validate_state(state)


@pytest.mark.parametrize(
    ("path", "artifact_type"),
    [
        ("ecc_residual/decision.json", "method_decision"),
        ("project-decision.json", "project_decision"),
        ("revealed-labels.csv", "label_reveal"),
        ("freeze/pre-evaluation-freeze.json", "pre_evaluation_freeze"),
    ],
)
def test_manifest_artifact_type_is_deterministic(path: str, artifact_type: str) -> None:
    assert _artifact_type(path) == artifact_type
