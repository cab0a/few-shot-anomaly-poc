from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.jsonio import write_json_atomic
from few_shot_anomaly_poc.v0_2_boundary_preparation import (
    BOUNDARY_STATE_SCHEMA,
    RUN_ID,
    build_public_boundary_record,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_SCHEMA_SHA256,
    HARD_GATES,
    METHODS,
    PREREGISTRATION_COMMIT,
    PREREGISTRATION_DOCUMENT_SHA256,
    load_v0_2_artifact_schema,
    load_v0_2_config,
)
from few_shot_anomaly_poc.v0_2_freeze_checkpoint import (
    BoundaryFreezeEvidence,
    V0_2FreezeCheckpointError,
    build_v0_2_freeze_record,
    validate_boundary_freeze_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "a" * 40


def _contract() -> tuple[dict, dict]:
    return (
        load_v0_2_config(ROOT / "configs/v0.2.yaml"),
        load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json"),
    )


def _write_lines(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"record": index}) + "\n" for index in range(count)),
        encoding="utf-8",
    )


def _boundary_fixture(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    config, schema = _contract()
    external = tmp_path / "external"
    public = tmp_path / "public"
    for name in ("source", "normal-manifests", "scorer", "sealed"):
        (external / name).mkdir(parents=True, exist_ok=True)
    write_json_atomic(external / "archive-identity.json", {"fixture": True})
    write_json_atomic(external / "extraction.json", {"fixture": True})

    reference_path = external / "normal-manifests/reference.jsonl"
    calibration_path = external / "normal-manifests/calibration.jsonl"
    _write_lines(reference_path, 20)
    _write_lines(calibration_path, 2)
    write_json_atomic(external / "normal-manifests/manifest-set.json", {"fixture": True})

    scoring_path = external / "scorer/scoring-manifest.json"
    scoring = {
        "schema_version": "v0.2-opaque-scoring-manifest-v1",
        "records": [
            {
                "asset_id": f"asset-{index:06d}",
                "byte_count": index + 1,
                "relative_path": f"assets/asset-{index:06d}.jpg",
                "sha256": f"{index + 1:x}" * 64,
            }
            for index in range(2)
        ],
    }
    write_json_atomic(scoring_path, scoring)
    sealed_path = external / "sealed/mapping.json"
    write_json_atomic(sealed_path, {"protected_fixture": True})
    key_path = external / "sealed/ordering-key.bin"
    key_path.write_bytes(b"k" * 32)
    key_path.chmod(0o600)

    state = {
        "schema_version": BOUNDARY_STATE_SCHEMA,
        "run_id": RUN_ID,
        "execution": {"execution_commit": "b" * 40},
        "contract": {
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "schema_sha256": EXPECTED_SCHEMA_SHA256,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "preregistration_document_sha256": PREREGISTRATION_DOCUMENT_SHA256,
        },
        "normal_partitions": {
            "reference_count": 20,
            "reference_manifest_sha256": sha256_file(reference_path),
            "calibration_count": 2,
            "calibration_manifest_sha256": sha256_file(calibration_path),
        },
        "opaque_final_test": {
            "asset_count": 2,
            "scoring_manifest_sha256": sha256_file(scoring_path),
            "sealed_mapping_sha256": sha256_file(sealed_path),
            "ordering_key_sha256": sha256_file(key_path),
            "class_counts_published": False,
        },
        "boundary": {
            "anomaly_score_computed": False,
            "final_test_label_revealed": False,
            "image_content_decoded": False,
            "image_content_displayed": False,
            "method_fit_performed": False,
            "raw_data_in_git": False,
            "threshold_calibrated": False,
        },
    }
    write_json_atomic(external / "boundary-state.json", state)

    boundary = build_public_boundary_record(
        run_kind="synthetic",
        archive_sha256=config["dataset"]["archive_sha256"],
        split_sha256=config["dataset"]["split_sha256"],
        reference_count=20,
        calibration_count=2,
        final_test_count=2,
        scoring_manifest_sha256=sha256_file(scoring_path),
        sealed_mapping_sha256=sha256_file(sealed_path),
        config=config,
        schema=schema,
    )
    write_json_atomic(public / "boundary/boundary-record.json", boundary)
    return external, public, config, schema


def test_boundary_evidence_validates_without_parsing_sealed_mapping(
    tmp_path: Path,
) -> None:
    external, public, config, schema = _boundary_fixture(tmp_path)

    evidence = validate_boundary_freeze_evidence(
        external_root=external,
        public_artifact_root=public,
        config=config,
        schema=schema,
    )

    assert evidence == BoundaryFreezeEvidence(
        calibration_manifest_sha256=sha256_file(external / "normal-manifests/calibration.jsonl"),
        reference_manifest_sha256=sha256_file(external / "normal-manifests/reference.jsonl"),
        scoring_manifest_sha256=sha256_file(external / "scorer/scoring-manifest.json"),
        sealed_mapping_sha256=sha256_file(external / "sealed/mapping.json"),
    )


def test_boundary_evidence_rejects_changed_sealed_mapping(tmp_path: Path) -> None:
    external, public, config, schema = _boundary_fixture(tmp_path)
    (external / "sealed/mapping.json").write_text("changed\n", encoding="utf-8")

    with pytest.raises(V0_2FreezeCheckpointError, match="sealed mapping SHA-256"):
        validate_boundary_freeze_evidence(
            external_root=external,
            public_artifact_root=public,
            config=config,
            schema=schema,
        )


def test_boundary_evidence_rejects_evaluation_started_before_freeze(
    tmp_path: Path,
) -> None:
    external, public, config, schema = _boundary_fixture(tmp_path)
    state_path = external / "boundary-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["boundary"]["method_fit_performed"] = True
    state_path.unlink()
    write_json_atomic(state_path, state)

    with pytest.raises(V0_2FreezeCheckpointError, match="advanced before"):
        validate_boundary_freeze_evidence(
            external_root=external,
            public_artifact_root=public,
            config=config,
            schema=schema,
        )


def test_boundary_evidence_rejects_any_pre_freeze_public_result(
    tmp_path: Path,
) -> None:
    external, public, config, schema = _boundary_fixture(tmp_path)
    write_json_atomic(public / "fit/result.json", {"too_early": True})

    with pytest.raises(V0_2FreezeCheckpointError, match="started before"):
        validate_boundary_freeze_evidence(
            external_root=external,
            public_artifact_root=public,
            config=config,
            schema=schema,
        )


def test_freeze_record_fixes_methods_gates_and_unrevealed_boundary() -> None:
    config, schema = _contract()
    evidence = BoundaryFreezeEvidence(
        calibration_manifest_sha256="1" * 64,
        reference_manifest_sha256="2" * 64,
        scoring_manifest_sha256="3" * 64,
        sealed_mapping_sha256="4" * 64,
    )

    record = build_v0_2_freeze_record(
        source_commit=SOURCE_COMMIT,
        evidence=evidence,
        config=config,
        schema=schema,
    )

    assert record["source_commit"] == SOURCE_COMMIT
    assert record["config_sha256"] == EXPECTED_CONFIG_SHA256
    assert record["schema_sha256"] == EXPECTED_SCHEMA_SHA256
    assert record["method_order"] == list(METHODS)
    assert record["hard_gate_order"] == list(HARD_GATES)
    assert record["label_reveal_completed"] is False


def test_boundary_evidence_rejects_tampered_public_record(tmp_path: Path) -> None:
    external, public, config, schema = _boundary_fixture(tmp_path)
    boundary_path = public / "boundary/boundary-record.json"
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    boundary["calibration_count"] += 1
    boundary_path.unlink()
    write_json_atomic(boundary_path, boundary)

    with pytest.raises(V0_2FreezeCheckpointError, match="partition counts"):
        validate_boundary_freeze_evidence(
            external_root=external,
            public_artifact_root=public,
            config=config,
            schema=schema,
        )


def test_boundary_evidence_does_not_modify_the_fixture(tmp_path: Path) -> None:
    external, public, config, schema = _boundary_fixture(tmp_path)
    state_before = deepcopy(
        json.loads((external / "boundary-state.json").read_text(encoding="utf-8"))
    )

    validate_boundary_freeze_evidence(
        external_root=external,
        public_artifact_root=public,
        config=config,
        schema=schema,
    )

    assert (
        json.loads((external / "boundary-state.json").read_text(encoding="utf-8")) == state_before
    )
    assert sorted(
        path.relative_to(public).as_posix() for path in public.rglob("*") if path.is_file()
    ) == ["boundary/boundary-record.json"]
