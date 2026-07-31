from __future__ import annotations

import json
from pathlib import Path

from few_shot_anomaly_poc.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "artifacts/v0.2/model-assets/acquisition.json"
PREREGISTRATION_PATH = ROOT / "docs/v0.2-preflight-preregistration.md"
EXPECTED_REPORT_SHA256 = "ba976ed08369fd80423d241129b8a86b05fcef650a39befa4ee67c8314233dac"
EXPECTED_SOURCE_SHA256 = "c27dcdaf50e9fb5bbdf2bb529da357716372e19c6afab17d5350f3f0094aed4b"
EXPECTED_CHECKPOINT_SHA256 = (
    "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
)


def _report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_committed_model_asset_record_has_fixed_identity() -> None:
    report = _report()

    assert sha256_file(REPORT_PATH) == EXPECTED_REPORT_SHA256
    assert report["schema_version"] == "v0.2-model-asset-acquisition-v1"
    assert report["acquisition"] == {
        "acquisition_base_commit": "64a1634056ce452a4080848f9a4cc0177f4d0a69",
        "acquisition_date": "2026-07-31",
        "external_cache": "data/external/v0.2/model-assets",
        "preregistration_commit": "e9330be10742947e4227ced4c99acafe4d098566",
        "preregistration_document": "docs/v0.2-preflight-preregistration.md",
        "preregistration_document_sha256": sha256_file(PREREGISTRATION_PATH),
        "preregistration_id": "v0.2-dinov2-cpu-preflight-1",
    }
    assert report["decision"]["next_step"] == (
        "PROCEED_TO_WEIGHTS_ONLY_STRICT_LOAD_VERIFICATION"
    )


def test_committed_source_record_preserves_observed_identity_and_license_scope() -> None:
    source = _report()["source"]

    assert source["identity"] == {
        "project": "facebookresearch/dinov2",
        "revision": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
    }
    assert source["artifact"]["observed_sha256"] == EXPECTED_SOURCE_SHA256
    assert source["artifact"]["published_sha256"] is None
    assert source["artifact"]["checksum_status"] == "observed_only"
    assert source["artifact"]["byte_count"] == 2_869_642
    assert source["artifact"]["storage"] == "outside_git"
    assert source["license"]["identifier"] == "Apache-2.0"
    assert source["inspection"]["safe_structure"] == "pass"
    assert source["inspection"]["member_count"] == 271
    assert {
        item["path"] for item in source["inspection"]["license_material"]
    } == {
        "LICENSE",
        "LICENSE_CELL_DINO_CODE",
        "LICENSE_CELL_DINO_MODELS",
        "LICENSE_XRAY_DINO_MODEL",
        "dinov2/thirdparty/CLIP/LICENSE",
    }


def test_committed_checkpoint_record_is_structural_not_executable_evidence() -> None:
    checkpoint = _report()["checkpoint"]

    assert checkpoint["artifact"]["observed_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert checkpoint["artifact"]["published_sha256"] is None
    assert checkpoint["artifact"]["checksum_status"] == "observed_only"
    assert checkpoint["artifact"]["byte_count"] == 88_283_115
    assert checkpoint["artifact"]["storage"] == "outside_git"
    assert checkpoint["identity"] == {
        "architecture": "ViT-S/14",
        "model_identifier": "dinov2_vits14",
        "pretraining_identity": "LVD142M",
        "register_tokens": False,
    }
    inspection = checkpoint["inspection"]
    assert inspection["safe_structure"] == "pass"
    assert inspection["crc_verification"] == "pass"
    assert inspection["member_count"] == 177
    assert inspection["data_member_count"] == 175
    assert inspection["byteorder_member_present"] is False
    assert inspection["version_member_present"] is True
    assert inspection["pickle"]["candidate_state_key_count"] == 175
    assert inspection["pickle"]["pickle_deserialized"] is False


def test_committed_record_preserves_non_execution_boundary() -> None:
    report = _report()

    assert report["boundary"] == {
        "checkpoint_acquired": True,
        "checkpoint_deserialized": False,
        "checkpoint_pickle_executed": False,
        "checkpoint_tensor_values_inspected": False,
        "dataset_access": False,
        "model_constructed": False,
        "model_inference_performed": False,
        "source_acquired": True,
        "source_executed": False,
        "source_extracted": False,
        "tensor_operation_performed": False,
    }
    serialized = REPORT_PATH.read_text(encoding="utf-8")
    assert "/home/" not in serialized
    assert "\\\\wsl" not in serialized
    assert "downloaded_at" not in serialized
    assert "verified_at" not in serialized
