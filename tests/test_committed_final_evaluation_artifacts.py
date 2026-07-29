from __future__ import annotations

import csv
import json
from pathlib import Path

from few_shot_anomaly_poc.hashing import sha256_file

ROOT = Path("artifacts/v0.1/evaluation/visa-pcb1-v0-1-final")
SCORING_ROOT = Path("artifacts/v0.1/scoring/first-fixed-final-test")
MANIFEST_PATH = ROOT / "artifact-manifest.json"
EXPECTED_MANIFEST_SHA256 = (
    "670c48bb57ebf26927de99388a0d966be2bc683bcaa8bcd9418ef023e3fe608d"
)
EXPECTED_SOURCE_COMMIT = "c6b4e5e164cc8788ff0428361406ada3e116543b"
METHODS = ("ecc_residual", "patch_hog_one_class_svm")
EXPECTED = {
    "ecc_residual": {
        "image_level_auroc": 0.8141,
        "image_level_auprc": 0.751334028468922,
        "normal_false_positive_rate": 0.09,
        "anomaly_recall": 0.21,
        "false_positive_count": 9,
        "false_negative_count": 79,
        "p95_latency_seconds": 1.24699559,
        "failed_gates": (
            "final_test_normal_fpr",
            "final_test_anomaly_recall",
            "cpu_p95_scoring_latency",
        ),
    },
    "patch_hog_one_class_svm": {
        "image_level_auroc": 0.7837999999999999,
        "image_level_auprc": 0.7241747321517565,
        "normal_false_positive_rate": 0.1,
        "anomaly_recall": 0.19,
        "false_positive_count": 10,
        "false_negative_count": 81,
        "p95_latency_seconds": 0.565766242,
        "failed_gates": (
            "final_test_normal_fpr",
            "final_test_anomaly_recall",
        ),
    },
}


def _rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(encoding="utf-8", newline="") as stream:
        return tuple(csv.DictReader(stream))


def test_final_manifest_fixes_source_and_hashes_every_contract_artifact() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert sha256_file(MANIFEST_PATH) == EXPECTED_MANIFEST_SHA256
    assert manifest["contract_version"] == "evaluation-artifacts/v0.1"
    assert manifest["run_id"] == "visa-pcb1-v0-1-final"
    assert manifest["run_kind"] == "final_test"
    assert manifest["dataset"] == "VisA"
    assert manifest["category"] == "pcb1"
    assert manifest["source_commit"] == EXPECTED_SOURCE_COMMIT
    assert manifest["partition_manifest_sha256"] == (
        "cde057e93700b90d473f52df680840ad9bc96668f6e6a8d476e712589dfa9f00"
    )
    assert len(manifest["files"]) == 16
    expected_files = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path != MANIFEST_PATH
    }
    assert {entry["relative_path"] for entry in manifest["files"]} == expected_files
    for entry in manifest["files"]:
        assert sha256_file(ROOT / entry["relative_path"]) == entry["sha256"]


def test_final_metrics_failures_and_ordered_hard_gate_decisions_are_exact() -> None:
    for method in METHODS:
        expected = EXPECTED[method]
        method_dir = ROOT / method
        metrics = json.loads((method_dir / "metrics.json").read_text(encoding="utf-8"))
        latency = json.loads((method_dir / "latency.json").read_text(encoding="utf-8"))
        decision = json.loads((method_dir / "decision.json").read_text(encoding="utf-8"))
        failures = _rows(method_dir / "failure-cases.csv")
        labels = _rows(method_dir / "revealed-labels.csv")

        assert metrics["normal_count"] == 100
        assert metrics["anomaly_count"] == 100
        assert metrics["item_count"] == 200
        assert metrics["score_failure_count"] == 0
        for key in (
            "image_level_auroc",
            "image_level_auprc",
            "normal_false_positive_rate",
            "anomaly_recall",
            "false_positive_count",
            "false_negative_count",
        ):
            assert metrics[key] == expected[key]
        assert latency["p95_latency_seconds"] == expected["p95_latency_seconds"]
        assert decision["decision"] == "REJECT"
        assert decision["decision_reason"] == "hard_gate_failed"
        assert decision["all_hard_gates_passed"] is False
        assert decision["first_failed_gate"] == "final_test_normal_fpr"
        assert decision["test_leakage_detected"] is False
        assert decision["failure_review_disposition"] == "guardrail_required"
        assert decision["condition"] is None
        outcomes = decision["gate_outcomes"]
        assert tuple(outcome["order"] for outcome in outcomes) == (1, 2, 3, 4, 5, 6)
        assert tuple(
            outcome["gate_id"] for outcome in outcomes if not outcome["passed"]
        ) == expected["failed_gates"]
        assert all(
            outcome["gate_id"]
            not in {"weighted_score", "aggregate_score", "hard_gate_waiver"}
            for outcome in outcomes
        )

        assert len(labels) == 200
        assert sum(row["true_class"] == "normal" for row in labels) == 100
        assert sum(row["true_class"] == "anomaly" for row in labels) == 100
        assert len(failures) == 10
        assert sum(row["case_type"] == "false_positive" for row in failures) == 5
        assert sum(row["case_type"] == "false_negative" for row in failures) == 5
        for case_type in ("false_positive", "false_negative"):
            assert tuple(
                int(row["rank"]) for row in failures if row["case_type"] == case_type
            ) == (1, 2, 3, 4, 5)


def test_final_bundle_reuses_first_fixed_scores_classifications_and_latency() -> None:
    score_core = (
        "relative_path",
        "score_status",
        "score_failure_code",
        "anomaly_score",
        "diagnostics_json",
    )
    classification_core = (
        "relative_path",
        "score_status",
        "score_failure_code",
        "anomaly_score",
        "threshold",
        "threshold_source_path",
        "calibration_sample_count",
        "calibration_rank",
        "predicted_class",
        "is_anomalous",
        "decision_reason",
        "score_margin",
    )
    for method in METHODS:
        final_scores = tuple(
            row
            for row in _rows(ROOT / method / "scores.csv")
            if row["partition"] == "final_test"
        )
        fixed_scores = _rows(SCORING_ROOT / method / "scores.csv")
        assert len(final_scores) == len(fixed_scores) == 200
        assert tuple(
            tuple(row[key] for key in score_core) for row in final_scores
        ) == tuple(tuple(row[key] for key in score_core) for row in fixed_scores)

        final_classifications = _rows(ROOT / method / "classifications.csv")
        fixed_classifications = _rows(SCORING_ROOT / method / "classifications.csv")
        assert tuple(
            tuple(row[key] for key in classification_core)
            for row in final_classifications
        ) == tuple(
            tuple(row[key] for key in classification_core)
            for row in fixed_classifications
        )

        final_latency = json.loads(
            (ROOT / method / "latency.json").read_text(encoding="utf-8")
        )
        fixed_latency = json.loads(
            (SCORING_ROOT / method / "latency.json").read_text(encoding="utf-8")
        )
        for key in (
            "measurement_boundary",
            "warmup_passes",
            "timed_passes",
            "sample_count",
            "score_failure_timing_count",
            "median_latency_seconds",
            "p95_latency_seconds",
        ):
            assert final_latency[key] == fixed_latency[key]


def test_final_bundle_contains_no_raw_image_or_pixel_level_artifact() -> None:
    suffixes = {path.suffix.lower() for path in ROOT.rglob("*") if path.is_file()}
    assert suffixes == {".csv", ".json"}
    assert not any(
        token in path.name.lower()
        for path in ROOT.rglob("*")
        for token in ("mask", "pixel", ".jpg", ".jpeg", ".png")
    )
