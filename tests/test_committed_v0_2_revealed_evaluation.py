from __future__ import annotations

from pathlib import Path

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    METHODS,
    load_v0_2_artifact_schema,
    load_v0_2_config,
)
from few_shot_anomaly_poc.v0_2_revealed_evaluation import (
    ASSET_COUNT,
    read_failure_cases_csv,
    read_label_reveal_csv,
    read_metrics_json,
)
from few_shot_anomaly_poc.v0_2_scoring_artifacts import read_method_scoring_artifacts

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "visa-pcb2-v0-2-final"
ARTIFACT_ROOT = ROOT / "artifacts/v0.2/evaluation" / RUN_ID
EXPECTED_HASHES = {
    "revealed-labels.csv": (
        "f1602ebf21241e7579f9be9092b90db8e9ca1fc9822b984130eb464d804d595c"
    ),
    "ecc_residual/metrics.json": (
        "c7ff3c98e9c350d33406d647c1ea5100e25addb128c9baf39a78e02a25a286a7"
    ),
    "ecc_residual/failure-cases.csv": (
        "1bd73cf3b0050b3eff934467625ba118d662e442ceaa5829f8a47344b4f390e8"
    ),
    "patch_hog_ocsvm/metrics.json": (
        "fd8c61a99e00109e8f5ec8c68fbbb8138042d908cda286b38ec06b67bf79d233"
    ),
    "patch_hog_ocsvm/failure-cases.csv": (
        "7edde3c32fd23e9d53db0c665595bb3c6789a1ce52bf039810b9d7646d3387ff"
    ),
    "dinov2_vits14_224_nn/metrics.json": (
        "22f5cca3187d974d69efde8f495c70cbcef45c193123704010c10534f1a2fb29"
    ),
    "dinov2_vits14_224_nn/failure-cases.csv": (
        "ad8f05152f936b8b9da5a6d94f2f8dc9eab43035373836f949dbb3a7887c22e9"
    ),
}


def _contract() -> tuple[dict, dict]:
    return (
        load_v0_2_config(ROOT / "configs/v0.2.yaml"),
        load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json"),
    )


def test_committed_v0_2_revealed_artifact_hashes_are_fixed() -> None:
    observed = {
        relative_path: sha256_file(ARTIFACT_ROOT / relative_path)
        for relative_path in EXPECTED_HASHES
    }

    assert observed == EXPECTED_HASHES


def test_committed_reveal_exactly_covers_the_opaque_asset_ids() -> None:
    _, schema = _contract()
    labels = read_label_reveal_csv(ARTIFACT_ROOT / "revealed-labels.csv", schema=schema)

    assert len(labels) == ASSET_COUNT
    assert [record["asset_id"] for record in labels] == [
        f"asset-{index:06d}" for index in range(ASSET_COUNT)
    ]
    assert {record["true_class"] for record in labels} == {"normal", "anomaly"}
    assert len({record["source_path"] for record in labels}) == ASSET_COUNT
    assert all(not record["source_path"].startswith("/") for record in labels)


def test_committed_metrics_and_failures_preserve_the_label_free_join() -> None:
    config, schema = _contract()
    labels = read_label_reveal_csv(ARTIFACT_ROOT / "revealed-labels.csv", schema=schema)
    label_by_id = {record["asset_id"]: record for record in labels}
    for method in METHODS:
        scoring = read_method_scoring_artifacts(ARTIFACT_ROOT / method, schema=schema)
        metrics = read_metrics_json(
            ARTIFACT_ROOT / method / "metrics.json",
            config=config,
            schema=schema,
        )
        failures = read_failure_cases_csv(
            ARTIFACT_ROOT / method / "failure-cases.csv",
            schema=schema,
        )
        classification_by_id = {
            record["asset_id"]: record for record in scoring.classification_records
        }

        assert metrics["method"] == method
        assert metrics["item_count"] == ASSET_COUNT
        assert metrics["normal_count"] + metrics["anomaly_count"] == ASSET_COUNT
        assert len(failures) == min(metrics["false_positive_count"], 5) + min(
            metrics["false_negative_count"], 5
        )
        for failure in failures:
            label = label_by_id[failure["asset_id"]]
            classification = classification_by_id[failure["asset_id"]]
            assert failure["source_path"] == label["source_path"]
            assert failure["true_class"] == label["true_class"]
            assert failure["predicted_class"] == classification["predicted_class"]
            assert failure["anomaly_score"] == classification["anomaly_score"]
            assert failure["threshold"] == classification["threshold"]
            assert failure["score_margin"] == classification["score_margin"]


def test_revealed_evaluation_bundle_contains_no_image_artifact() -> None:
    suffixes = {path.suffix.lower() for path in ARTIFACT_ROOT.rglob("*") if path.is_file()}
    assert suffixes == {".csv", ".json"}
