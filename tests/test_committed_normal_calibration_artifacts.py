from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

CHECKPOINT_DIR = Path("artifacts/v0.1/calibration/normal-only")
CHECKPOINT_PATH = CHECKPOINT_DIR / "normal-only-calibration.json"
SOURCE_COMMIT = "4fef91c1d1e339aa507cad80d51127e01046ae0b"
EXPECTED_METHODS = ("ecc_residual", "patch_hog_one_class_svm")
EXPECTED_COLUMNS = (
    "contract_version",
    "checkpoint_id",
    "method",
    "partition",
    "relative_path",
    "score_status",
    "score_failure_code",
    "anomaly_score",
    "diagnostics_json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _score_rows(method: str) -> tuple[dict[str, str], ...]:
    path = CHECKPOINT_DIR / method / "scores.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == EXPECTED_COLUMNS
        return tuple(reader)


def test_committed_checkpoint_fixes_normal_only_thresholds_before_final_test() -> None:
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))

    assert checkpoint["status"] == "THRESHOLDS_FIXED_BEFORE_FINAL_TEST"
    assert checkpoint["source_commit"] == SOURCE_COMMIT
    assert checkpoint["reference"]["count"] == 20
    assert len(checkpoint["reference"]["paths"]) == 20
    assert checkpoint["calibration"] == {
        "anomaly_labels_used": False,
        "count": 884,
        "final_test_paths_used": False,
        "normal_only": True,
    }
    assert checkpoint["evaluation_boundary"] == {
        "decision_recorded": False,
        "final_test_image_read": False,
        "final_test_label_join_performed": False,
        "final_test_scoring_started": False,
        "hard_gate_changed": False,
        "latency_measured": False,
        "metric_computed": False,
        "per_path_final_test_label_read": False,
        "threshold_rule_changed": False,
    }
    assert tuple(checkpoint["methods"]) == EXPECTED_METHODS


def test_committed_score_files_regenerate_each_fixed_threshold() -> None:
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))

    for method in EXPECTED_METHODS:
        method_record = checkpoint["methods"][method]
        artifact = method_record["score_artifact"]
        score_path = CHECKPOINT_DIR / artifact["relative_path"]
        rows = _score_rows(method)

        assert artifact["record_count"] == len(rows) == 884
        assert artifact["sha256"] == _sha256(score_path)
        assert len({row["relative_path"] for row in rows}) == 884
        assert all(
            row["contract_version"] == "normal-only-calibration/v0.1"
            and row["checkpoint_id"] == "v0.1-normal-reference-fit-and-calibration"
            and row["method"] == method
            and row["partition"] == "calibration"
            and row["relative_path"].startswith("pcb1/Data/Images/Normal/")
            and "\\" not in row["relative_path"]
            and row["score_status"] in {"ok", "failed"}
            for row in rows
        )
        diagnostics = tuple(json.loads(row["diagnostics_json"]) for row in rows)
        assert all(isinstance(item, dict) and "label" not in item for item in diagnostics)

        ordered = sorted(
            rows,
            key=lambda row: (float(row["anomaly_score"]), row["relative_path"]),
        )
        rank = math.ceil(0.95 * len(ordered))
        source = ordered[rank - 1]
        calibration = method_record["threshold_calibration"]
        threshold = float(source["anomaly_score"])
        predicted_anomalous_count = sum(
            row["score_status"] == "failed"
            or float(row["anomaly_score"]) > threshold
            for row in rows
        )
        failed_count = sum(row["score_status"] == "failed" for row in rows)

        assert rank == calibration["rank"] == 840
        assert source["relative_path"] == calibration["threshold_source_path"]
        assert threshold == calibration["threshold"]
        assert failed_count == calibration["failed_score_count"] == 0
        assert (
            predicted_anomalous_count
            == calibration["predicted_anomalous_count"]
            == 44
        )
        assert calibration["realized_normal_false_positive_rate"] == 44 / 884


def test_committed_calibration_artifacts_contain_no_raw_or_final_test_output() -> None:
    checkpoint_text = CHECKPOINT_PATH.read_text(encoding="utf-8")

    assert "final_test_image_read\": true" not in checkpoint_text
    assert "per_path_final_test_label_read\": true" not in checkpoint_text
    assert not (CHECKPOINT_DIR / "revealed-labels.csv").exists()
    assert not (CHECKPOINT_DIR / "metrics.json").exists()
    assert not (CHECKPOINT_DIR / "decision.json").exists()
    assert all(
        path.suffix in {".json", ".csv"}
        for path in CHECKPOINT_DIR.rglob("*")
        if path.is_file()
    )
