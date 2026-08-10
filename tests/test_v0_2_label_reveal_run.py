from __future__ import annotations

import json
from pathlib import Path

import pytest

from few_shot_anomaly_poc.v0_2_evaluation_contract import METHODS
from few_shot_anomaly_poc.v0_2_label_reveal_run import (
    EVIDENCE_COMMIT,
    LABEL_FREE_BUNDLE_SHA256,
    OPAQUE_SCORING_MANIFEST_SHA256,
    SEALED_MAPPING_SHA256,
    V0_2LabelRevealRunError,
    _mark_reveal_started,
    _validate_external_pre_reveal_state,
)


def _pre_reveal_state() -> dict:
    return {
        "schema_version": "v0.2-boundary-state-v1",
        "run_id": "visa-pcb2-v0-2-final",
        "boundary": {
            "anomaly_score_computed": True,
            "final_test_label_revealed": False,
        },
        "opaque_final_test": {
            "asset_count": 200,
            "scoring_manifest_sha256": OPAQUE_SCORING_MANIFEST_SHA256,
            "sealed_mapping_sha256": SEALED_MAPPING_SHA256,
        },
        "offline_reproduction": {
            "reproduction_status": {method: "pass" for method in METHODS},
            "labels_accessed": False,
        },
        "pre_reveal_checkpoint": {
            "label_free_evidence_commit": EVIDENCE_COMMIT,
            "label_free_bundle_sha256": LABEL_FREE_BUNDLE_SHA256,
            "git_push_verified": True,
            "labels_accessed": False,
        },
    }


def test_crossing_reveal_boundary_is_immediately_persistent_and_non_retryable(
    tmp_path: Path,
) -> None:
    state = _pre_reveal_state()
    path = tmp_path / "boundary-state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    _validate_external_pre_reveal_state(state)

    revealed = _mark_reveal_started(
        state_path=path,
        state=state,
        execution_commit="a" * 40,
    )

    assert revealed["boundary"]["final_test_label_revealed"] is True
    assert revealed["label_reveal_evaluation"]["status"] == "in_progress"
    assert revealed["label_reveal_evaluation"]["labels_accessed"] is True
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted == revealed
    with pytest.raises(V0_2LabelRevealRunError, match="unrevealed checkpoint state"):
        _validate_external_pre_reveal_state(revealed)


def test_pre_reveal_state_rejects_any_failed_reproduction() -> None:
    state = _pre_reveal_state()
    state["offline_reproduction"]["reproduction_status"]["ecc_residual"] = "fail"

    with pytest.raises(V0_2LabelRevealRunError, match="unrevealed checkpoint state"):
        _validate_external_pre_reveal_state(state)
