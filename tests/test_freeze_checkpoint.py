from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from few_shot_anomaly_poc.config import load_config
from few_shot_anomaly_poc.freeze_checkpoint import (
    FROZEN_PATHS,
    FreezeCheckpointError,
    build_pre_evaluation_freeze,
    read_and_verify_pre_evaluation_freeze,
    verify_pre_evaluation_freeze,
    write_pre_evaluation_freeze,
)

PROJECT_ROOT = Path(".").resolve()
CONFIG_PATH = PROJECT_ROOT / "configs/v0.1.yaml"
SOURCE_COMMIT = "fd9857acb29903fadb570680ecb5d4d8ebf5a5aa"
CI_RUN_ID = 30434900673
CI_RUN_URL = "https://github.com/cab0a/few-shot-anomaly-poc/actions/runs/30434900673"
COMMITTED_RECORD = PROJECT_ROOT / "artifacts/v0.1/freeze/pre-evaluation-freeze.json"


def _record() -> dict:
    return build_pre_evaluation_freeze(
        project_root=PROJECT_ROOT,
        source_commit=SOURCE_COMMIT,
        ci_run_id=CI_RUN_ID,
        ci_run_url=CI_RUN_URL,
        config=load_config(CONFIG_PATH),
    )


def test_freeze_records_fixed_source_ci_references_and_gates() -> None:
    record = _record()

    assert record["status"] == "FROZEN_BEFORE_FINAL_TEST"
    assert record["evaluation_source_commit"] == SOURCE_COMMIT
    assert record["ci"] == {
        "provider": "GitHub Actions",
        "workflow": "CI",
        "run_id": CI_RUN_ID,
        "run_url": CI_RUN_URL,
        "conclusion": "success",
    }
    assert record["selection"]["seed"] == 42
    assert record["selection"]["reference_count"] == 20
    assert len(record["selection"]["reference_ids"]) == 20
    assert record["partitions"]["calibration_count"] == 884
    assert record["partitions"]["overlap_allowed"] is False
    gates = record["evaluation_rules"]["hard_gate_decision"]
    assert gates["normal_fpr_max"] == 0.05
    assert gates["anomaly_recall_min"] == 0.9
    assert gates["cpu_p95_latency_seconds_max"] == 1.0
    assert gates["normal_reference_count_max"] == 20
    assert gates["weighted_score_allowed"] is False
    assert gates["hard_gate_waiver_allowed"] is False


def test_freeze_covers_every_declared_file_in_stable_order() -> None:
    record = _record()

    assert tuple(item["relative_path"] for item in record["frozen_files"]) == (FROZEN_PATHS)
    assert len({item["sha256"] for item in record["frozen_files"]}) == len(record["frozen_files"])
    assert len(record["frozen_tree_sha256"]) == 64


def test_freeze_boundary_is_unrevealed_and_unscored() -> None:
    record = _record()

    assert record["boundary_state"] == {
        "final_test_scoring_started": False,
        "final_test_label_join_performed": False,
        "final_test_metrics_computed": False,
        "final_test_decision_recorded": False,
    }
    assert not Path("artifacts/v0.1/evaluation/first-fixed-final-test").exists()


def test_freeze_write_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "freeze.json"
    write_pre_evaluation_freeze(
        _record(),
        path=output,
        project_root=PROJECT_ROOT,
    )

    with pytest.raises(FileExistsError):
        write_pre_evaluation_freeze(
            _record(),
            path=output,
            project_root=PROJECT_ROOT,
        )


def test_freeze_verifier_rejects_tampered_digest() -> None:
    record = deepcopy(_record())
    record["frozen_files"][0]["sha256"] = "0" * 64

    with pytest.raises(FreezeCheckpointError, match="frozen file"):
        verify_pre_evaluation_freeze(record, project_root=PROJECT_ROOT)


def test_committed_freeze_matches_generated_record() -> None:
    committed = read_and_verify_pre_evaluation_freeze(
        COMMITTED_RECORD,
        project_root=PROJECT_ROOT,
    )

    assert committed == _record()
    assert json.loads(COMMITTED_RECORD.read_text(encoding="utf-8")) == committed
