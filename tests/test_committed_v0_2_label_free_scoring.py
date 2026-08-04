from __future__ import annotations

import json
import shutil
from pathlib import Path

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    METHODS,
    load_v0_2_artifact_schema,
    load_v0_2_config,
    validate_json_artifact,
)
from few_shot_anomaly_poc.v0_2_scoring_artifacts import (
    ASSET_COUNT,
    SCORE_TOLERANCE,
    TIMED_PASS_COUNT,
    read_method_scoring_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "visa-pcb2-v0-2-final"
ARTIFACT_ROOT = ROOT / "artifacts/v0.2/evaluation" / RUN_ID
EXPECTED_HASHES = {
    "ecc_residual/scores.csv": (
        "43545c1ec1c75039a7fd73389861fa7de4a6125fd264901b1e367ee1170b22c4"
    ),
    "ecc_residual/classifications.csv": (
        "4e6b24d3553507a8d403900a209840c4d0e4d16ab3882001435e8875d93fb8c8"
    ),
    "ecc_residual/latency-observations.csv": (
        "3d2be6bab3bcbe35910b102c87dfed9e70786b0072b49d81d63dfc493fa2e50b"
    ),
    "patch_hog_ocsvm/scores.csv": (
        "fd403f19584c3170b15b4f9c3bef320ac8fbea23a0d715b26188bf850aef46e2"
    ),
    "patch_hog_ocsvm/classifications.csv": (
        "b17a977a570b1d2becfb9359941fe09d2936efc474253be1c11d133e369b95fa"
    ),
    "patch_hog_ocsvm/latency-observations.csv": (
        "5ff7424018220a7a8dd79e3dd646225de205a43373ff74c13b569100af45d1cb"
    ),
    "dinov2_vits14_224_nn/scores.csv": (
        "38885820447538763850bbc8820657ff75a8980e6e39ff09416970f32be6b282"
    ),
    "dinov2_vits14_224_nn/classifications.csv": (
        "71f4ae76d453583d030646ffbdf746683603d6d4eacad5784e3792d39cd468ba"
    ),
    "dinov2_vits14_224_nn/latency-observations.csv": (
        "1477493dac4a645283dc349d811d92bfcb8739c240a880ab9067e09d823ff63f"
    ),
}
FORBIDDEN_FIELDS = {
    "true_class",
    "class_label",
    "source_path",
    "official_split",
    "sealed_mapping",
    "hmac_key",
}


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_committed_v0_2_label_free_artifact_hashes_are_fixed() -> None:
    assert {
        relative_path: sha256_file(ARTIFACT_ROOT / relative_path)
        for relative_path in EXPECTED_HASHES
    } == EXPECTED_HASHES


def test_committed_v0_2_label_free_bundles_match_the_frozen_contract(tmp_path: Path) -> None:
    config = load_v0_2_config(ROOT / "configs/v0.2.yaml")
    schema = load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json")
    for method in METHODS:
        summary = validate_json_artifact(
            "calibration_summary",
            _read_json(ARTIFACT_ROOT / method / "calibration-summary.json"),
            config=config,
            schema=schema,
        )
        serialized_root = tmp_path / method
        serialized_root.mkdir()
        for name in ("scores.csv", "classifications.csv", "latency-observations.csv"):
            shutil.copyfile(ARTIFACT_ROOT / method / name, serialized_root / name)
        bundle = read_method_scoring_artifacts(serialized_root, schema=schema)
        assert len(bundle.score_records) == ASSET_COUNT
        assert len(bundle.classification_records) == ASSET_COUNT
        assert len(bundle.latency_records) == ASSET_COUNT * TIMED_PASS_COUNT

        canonical = {record["asset_id"]: record for record in bundle.score_records}
        for score, classification in zip(
            bundle.score_records,
            bundle.classification_records,
            strict=True,
        ):
            assert classification["asset_id"] == score["asset_id"]
            assert classification["method"] == score["method"] == method
            assert classification["run_id"] == score["run_id"] == RUN_ID
            assert classification["score_status"] == score["score_status"]
            assert classification["score_failure_code"] == score["score_failure_code"]
            assert classification["anomaly_score"] == score["anomaly_score"]
            assert classification["threshold"] == summary["threshold"]
        for observation in bundle.latency_records:
            score = canonical[observation["asset_id"]]
            assert observation["method"] == method
            assert observation["run_id"] == RUN_ID
            assert observation["score_status"] == score["score_status"]
            assert observation["score_failure_code"] == score["score_failure_code"]
            assert abs(observation["anomaly_score"] - score["anomaly_score"]) <= SCORE_TOLERANCE


def test_committed_v0_2_label_free_headers_exclude_protected_fields() -> None:
    for relative_path in EXPECTED_HASHES:
        header = (ARTIFACT_ROOT / relative_path).read_text(encoding="utf-8").splitlines()[0]
        assert not FORBIDDEN_FIELDS.intersection(header.split(","))
