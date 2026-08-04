from __future__ import annotations

import csv
import json
from pathlib import Path

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.v0_2_calibration_artifacts import (
    CalibrationScore,
    calibrate_normal_scores,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    METHODS,
    load_v0_2_artifact_schema,
    load_v0_2_config,
    validate_json_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/v0.2/evaluation/visa-pcb2-v0-2-final"
REFERENCE_MANIFEST_SHA256 = (
    "e587f1808262480261ae8a7b940faff0d9ef5f83cf215028b31490ba48369b99"
)
EXPECTED_SHA256 = {
    "dinov2_vits14_224_nn/calibration-scores.csv": (
        "dc99f12c1a76c1421a9dd6258d14231afa82e75ebf1d2d22a20a20a4b09ba1bc"
    ),
    "dinov2_vits14_224_nn/calibration-summary.json": (
        "0a89bcb834f5e38a601bb956d9b099c613786187adbbd2ef8ba75226e2a75da7"
    ),
    "dinov2_vits14_224_nn/fit.json": (
        "ba2bc1b5d2b974cf6d700d775ac6c94a4b14d407e9e7fd6a8832a7b9877ef4e6"
    ),
    "ecc_residual/calibration-scores.csv": (
        "b0b513f16b6ce0d068658d43a34f474106095bbd52e57cbc388927cde6f21c26"
    ),
    "ecc_residual/calibration-summary.json": (
        "e68f519abaa6f1f08a1376eec444cae5777736e3414b882ed93bf73f93345735"
    ),
    "ecc_residual/fit.json": (
        "fee3b96824987a6d03f279f5fefc8cfe15a0c807050ec6a35700d4c2e567ae3d"
    ),
    "patch_hog_ocsvm/calibration-scores.csv": (
        "f6c40c3e4af1979f67ddf8916889887e022b58da0cd9fa04561c00f171a84d8e"
    ),
    "patch_hog_ocsvm/calibration-summary.json": (
        "9e687e80437de5f38f848677ad9372104c1bc78275ae536a7c58f4b02bdee356"
    ),
    "patch_hog_ocsvm/fit.json": (
        "d8da4c0704736e45495501618fcaa57d654e97acf4d3d10e12084e112e29b432"
    ),
}


def _load_score_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_committed_v0_2_4_inventory_and_hashes_are_fixed() -> None:
    actual = {
        path.relative_to(ARTIFACT_ROOT).as_posix()
        for method in METHODS
        for path in (ARTIFACT_ROOT / method).rglob("*")
        if path.is_file()
    }

    assert actual == set(EXPECTED_SHA256)
    for relative_path, expected_hash in EXPECTED_SHA256.items():
        assert sha256_file(ARTIFACT_ROOT / relative_path) == expected_hash


def test_committed_v0_2_4_fit_records_are_complete() -> None:
    config = load_v0_2_config(ROOT / "configs/v0.2.yaml")
    schema = load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json")

    for method in METHODS:
        fit_path = ARTIFACT_ROOT / method / "fit.json"
        fit = validate_json_artifact(
            "fit",
            json.loads(fit_path.read_text(encoding="utf-8")),
            config=config,
            schema=schema,
        )
        assert fit["method"] == method
        assert fit["status"] == "fit_ok"
        assert fit["reference_count"] == 20
        assert fit["successful_reference_count"] == 20
        assert fit["failed_reference_count"] == 0
        assert fit["reference_manifest_sha256"] == REFERENCE_MANIFEST_SHA256
        assert fit["failure_code"] is None


def test_committed_v0_2_4_thresholds_reproduce_from_normal_scores() -> None:
    config = load_v0_2_config(ROOT / "configs/v0.2.yaml")
    schema = load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json")

    for method in METHODS:
        method_root = ARTIFACT_ROOT / method
        raw_rows = _load_score_rows(method_root / "calibration-scores.csv")
        scores = tuple(
            CalibrationScore(
                source_path=row["source_path"],
                score_status=row["score_status"],
                score_failure_code=row["score_failure_code"] or None,
                anomaly_score=float(row["anomaly_score"]),
            )
            for row in raw_rows
        )
        regenerated = calibrate_normal_scores(
            scores,
            run_id="visa-pcb2-v0-2-final",
            method=method,
            config=config,
            schema=schema,
        )
        committed = json.loads(
            (method_root / "calibration-summary.json").read_text(encoding="utf-8")
        )

        assert len(raw_rows) == 881
        assert all(row["source_path"].startswith("pcb2/Data/Images/Normal/") for row in raw_rows)
        assert all(row["score_status"] == "ok" for row in raw_rows)
        assert regenerated.summary == committed
        assert committed["rank"] == 837
        assert committed["predicted_anomalous_count"] == 44
        assert committed["score_failure_count"] == 0
        assert committed["realized_normal_fpr"] == 44 / 881


def test_committed_v0_2_4_artifacts_do_not_cross_the_reveal_boundary() -> None:
    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for method in METHODS
        for path in (ARTIFACT_ROOT / method).rglob("*")
        if path.is_file()
    ).lower()

    assert "true_class" not in serialized
    assert "class_label" not in serialized
    assert "sealed_mapping" not in serialized
    assert "asset-000" not in serialized
    assert "/home/" not in serialized
    assert "\\\\wsl" not in serialized
    assert not any(
        path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        for method in METHODS
        for path in (ARTIFACT_ROOT / method).rglob("*")
        if path.is_file()
    )
