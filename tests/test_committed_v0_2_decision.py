from __future__ import annotations

import csv
import json
from pathlib import Path

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    HARD_GATES,
    METHODS,
    load_v0_2_artifact_schema,
    load_v0_2_config,
    validate_json_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/v0.2/evaluation/visa-pcb2-v0-2-final"
SOURCE_COMMIT = "47e565e722c769fed060312725d14b4135fefbaf"
EXPECTED_HASHES = {
    "ecc_residual/decision.json": (
        "dcd80fd82ba206fdf0610ca9e3d735fcb2a3835496734d7dbb1f76706c8dd588"
    ),
    "patch_hog_ocsvm/decision.json": (
        "c5ba75cbe93d2f921cdf80d9a277fa208eb5f914c836eb7aaf18a8deaf49093e"
    ),
    "dinov2_vits14_224_nn/decision.json": (
        "31be925290499723016322aed12bb8b7ccb81f8140f26ebcd246a6173cec448b"
    ),
    "project-decision.json": (
        "aad34c5e3f0c085cd0f218e61d504df9539ddc68764694cc3d53ed0f4ec729e2"
    ),
    "artifact-manifest.json": (
        "dbf7bfc1d2429bc391720136ba76711fa0493eb7407487e185bf7d12c4b85d19"
    ),
}
EXPECTED_FIRST_FAILURES = {
    "ecc_residual": "final_test_anomaly_recall",
    "patch_hog_ocsvm": "final_test_anomaly_recall",
    "dinov2_vits14_224_nn": "final_test_normal_fpr",
}


def _json(relative_path: str) -> dict:
    return json.loads((ARTIFACT_ROOT / relative_path).read_text(encoding="utf-8"))


def _contract() -> tuple[dict, dict]:
    return (
        load_v0_2_config(ROOT / "configs/v0.2.yaml"),
        load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json"),
    )


def _record_count(path: Path) -> int:
    if path.suffix == ".json":
        return 1
    with path.open(encoding="utf-8", newline="") as stream:
        return len(list(csv.reader(stream))) - 1


def test_committed_decision_artifact_hashes_are_fixed() -> None:
    assert {
        path: sha256_file(ARTIFACT_ROOT / path) for path in EXPECTED_HASHES
    } == EXPECTED_HASHES


def test_each_method_stops_at_its_first_failed_ordered_gate() -> None:
    config, schema = _contract()
    for method in METHODS:
        decision = validate_json_artifact(
            "method_decision",
            _json(f"{method}/decision.json"),
            config=config,
            schema=schema,
        )
        assert decision["decision"] == "REJECT"
        assert decision["first_failed_gate"] == EXPECTED_FIRST_FAILURES[method]
        assert [outcome["name"] for outcome in decision["gate_outcomes"]] == list(
            HARD_GATES
        )
        failed_index = HARD_GATES.index(EXPECTED_FIRST_FAILURES[method])
        assert all(
            outcome["status"] == "pass"
            for outcome in decision["gate_outcomes"][:failed_index]
        )
        assert decision["gate_outcomes"][failed_index]["status"] == "fail"
        assert all(
            outcome["status"] == "not_evaluated" and outcome["observed"] is None
            for outcome in decision["gate_outcomes"][failed_index + 1 :]
        )
        assert decision["weighted_score_used"] is False
        assert decision["hard_gate_waiver_used"] is False


def test_project_decision_is_unselected_reject() -> None:
    config, schema = _contract()

    decision = validate_json_artifact(
        "project_decision",
        _json("project-decision.json"),
        config=config,
        schema=schema,
    )

    assert decision["decision"] == "REJECT"
    assert decision["selected_method"] is None
    assert decision["method_decisions"] == {method: "REJECT" for method in METHODS}
    assert decision["weighted_score_used"] is False
    assert "completed final-test boundary" in decision["next_validation"]


def test_manifest_covers_every_other_bundle_file_once() -> None:
    config, schema = _contract()
    manifest = validate_json_artifact(
        "bundle_manifest",
        _json("artifact-manifest.json"),
        config=config,
        schema=schema,
    )
    files = manifest["files"]
    actual_paths = sorted(
        path.relative_to(ARTIFACT_ROOT).as_posix()
        for path in ARTIFACT_ROOT.rglob("*")
        if path.is_file() and path.name != "artifact-manifest.json"
    )

    assert manifest["source_commit"] == SOURCE_COMMIT
    assert len(files) == 35
    assert [entry["relative_path"] for entry in files] == actual_paths
    for entry in files:
        path = ARTIFACT_ROOT / entry["relative_path"]
        assert entry["sha256"] == sha256_file(path)
        assert entry["record_count"] == _record_count(path)


def test_complete_bundle_contains_no_raw_or_image_artifact() -> None:
    suffixes = {
        path.suffix.lower() for path in ARTIFACT_ROOT.rglob("*") if path.is_file()
    }

    assert suffixes == {".csv", ".json"}
