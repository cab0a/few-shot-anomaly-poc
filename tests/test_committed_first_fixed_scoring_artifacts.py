from __future__ import annotations

import csv
import json
from pathlib import Path

from few_shot_anomaly_poc.hashing import sha256_file

ROOT = Path("artifacts/v0.1/scoring/first-fixed-final-test")
CHECKPOINT_PATH = ROOT / "first-fixed-scoring.json"
EXPECTED_CHECKPOINT_SHA256 = (
    "70c3fb46e60e63b39616db54ea23f12c116dded693a8fab7e900b69591074dd7"
)
EXPECTED_SOURCE_COMMIT = "5b142f31c974334545ca2bb63bb7b2c6c514828a"
METHODS = ("ecc_residual", "patch_hog_one_class_svm")
EXPECTED = {
    "ecc_residual": {
        "threshold": 0.688464437424507,
        "predicted_normal_count": 170,
        "predicted_anomalous_count": 30,
        "score_failure_count": 0,
        "median_latency_seconds": 0.4369505,
        "p95_latency_seconds": 1.24699559,
    },
    "patch_hog_one_class_svm": {
        "threshold": 0.17611826509314352,
        "predicted_normal_count": 171,
        "predicted_anomalous_count": 29,
        "score_failure_count": 0,
        "median_latency_seconds": 0.4326800675,
        "p95_latency_seconds": 0.565766242,
    },
}


def _rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(encoding="utf-8", newline="") as stream:
        return tuple(csv.DictReader(stream))


def test_committed_checkpoint_fixes_source_inputs_and_evaluation_boundary() -> None:
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))

    assert sha256_file(CHECKPOINT_PATH) == EXPECTED_CHECKPOINT_SHA256
    assert checkpoint["contract_version"] == "first-fixed-final-test-scoring/v0.1"
    assert checkpoint["run_id"] == "visa-pcb1-v0-1-first-fixed"
    assert checkpoint["status"] == "SCORES_CLASSIFICATIONS_AND_LATENCY_FIXED"
    assert checkpoint["source_commit"] == EXPECTED_SOURCE_COMMIT
    assert checkpoint["final_test_item_count"] == 200
    assert checkpoint["normal_calibration_state_sha256"] == (
        "d0056a52225d5600e5db9d0c11076a1fbd919f273b66f1e1b65cd5895e883cb4"
    )
    assert checkpoint["final_test_manifest_sha256"] == (
        "04a5d9fbf1f0526f42a3705319cbcac2d37015525471b19b75f53482cc33c285"
    )
    assert checkpoint["evaluation_boundary"] == {
        "final_test_images_decoded": True,
        "final_test_scoring_completed": True,
        "fixed_threshold_classification_completed": True,
        "cpu_latency_measured": True,
        "per_path_final_test_class_read": False,
        "final_test_class_join_performed": False,
        "metric_computed": False,
        "failure_case_selected": False,
        "decision_recorded": False,
        "image_displayed": False,
        "threshold_rule_changed": False,
        "hard_gate_changed": False,
    }
    assert checkpoint["local_state"]["committed_to_git"] is False
    assert checkpoint["local_state"]["logical_path"].startswith("work/")


def test_committed_method_artifacts_match_hashes_counts_and_fixed_thresholds() -> None:
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    path_order = None

    for method in METHODS:
        method_record = checkpoint["methods"][method]
        expected = EXPECTED[method]
        assert {
            key: method_record[key]
            for key in (
                "threshold",
                "predicted_normal_count",
                "predicted_anomalous_count",
                "score_failure_count",
                "median_latency_seconds",
                "p95_latency_seconds",
            )
        } == expected
        assert method_record["item_count"] == 200

        artifacts = method_record["artifacts"]
        score_path = ROOT / artifacts["score"]["relative_path"]
        classification_path = ROOT / artifacts["classification"]["relative_path"]
        latency_path = ROOT / artifacts["latency"]["relative_path"]
        observation_path = ROOT / artifacts["latency"]["observation_relative_path"]
        assert sha256_file(score_path) == artifacts["score"]["sha256"]
        assert sha256_file(classification_path) == artifacts["classification"]["sha256"]
        assert sha256_file(latency_path) == artifacts["latency"]["sha256"]
        assert sha256_file(observation_path) == artifacts["latency"]["observation_sha256"]

        scores = _rows(score_path)
        classifications = _rows(classification_path)
        observations = _rows(observation_path)
        assert len(scores) == artifacts["score"]["record_count"] == 200
        assert len(classifications) == artifacts["classification"]["record_count"] == 200
        assert len(observations) == (
            artifacts["latency"]["observation_record_count"]
        ) == 600
        assert all(row["method"] == method for row in scores)
        assert all(row["method"] == method for row in classifications)
        assert all(row["method"] == method for row in observations)
        assert all(row["partition"] == "final_test" for row in scores)
        assert all(row["score_failure_code"] == "" for row in scores)
        assert all(row["score_failure_code"] == "" for row in observations)
        assert all(float(row["threshold"]) == expected["threshold"] for row in classifications)
        assert sum(row["predicted_class"] == "normal" for row in classifications) == (
            expected["predicted_normal_count"]
        )
        assert sum(row["predicted_class"] == "anomalous" for row in classifications) == (
            expected["predicted_anomalous_count"]
        )

        score_paths = tuple(row["relative_path"] for row in scores)
        classification_paths = tuple(row["relative_path"] for row in classifications)
        assert score_paths == tuple(sorted(score_paths))
        assert classification_paths == score_paths
        if path_order is None:
            path_order = score_paths
        else:
            assert score_paths == path_order

        latency = json.loads(latency_path.read_text(encoding="utf-8"))
        assert latency["status"] == "ok"
        assert latency["method"] == method
        assert latency["item_count"] == 200
        assert latency["warmup_passes"] == 1
        assert latency["timed_passes"] == 3
        assert latency["sample_count"] == 600
        assert latency["score_failure_timing_count"] == 0
        assert latency["median_latency_seconds"] == expected["median_latency_seconds"]
        assert latency["p95_latency_seconds"] == expected["p95_latency_seconds"]
        assert latency["measurement_boundary"] == (
            "decoded_grayscale_uint8_to_image_score"
        )
        assert tuple(latency["ordered_paths"]) == score_paths
        assert {int(row["pass_index"]) for row in observations} == {1, 2, 3}
        assert all(
            sum(row["relative_path"] == path for row in observations) == 3
            for path in score_paths
        )


def test_committed_scoring_directory_contains_no_evaluation_or_raw_image_artifact() -> None:
    files = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
    }
    assert files == {
        "first-fixed-scoring.json",
        "ecc_residual/scores.csv",
        "ecc_residual/classifications.csv",
        "ecc_residual/latency.json",
        "ecc_residual/latency-observations.csv",
        "patch_hog_one_class_svm/scores.csv",
        "patch_hog_one_class_svm/classifications.csv",
        "patch_hog_one_class_svm/latency.json",
        "patch_hog_one_class_svm/latency-observations.csv",
    }
    headers = []
    for path in ROOT.rglob("*.csv"):
        with path.open(encoding="utf-8", newline="") as stream:
            headers.extend(next(csv.reader(stream)))
    assert all("label" not in field and "true_class" not in field for field in headers)
    assert all(
        token not in path.name.lower()
        for path in ROOT.rglob("*")
        for token in ("metric", "failure", "decision", ".jpg", ".png")
    )
