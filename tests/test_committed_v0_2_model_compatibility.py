from __future__ import annotations

import json
from pathlib import Path

from few_shot_anomaly_poc.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "artifacts/v0.2/model-compatibility/strict-load.json"
ACQUISITION_PATH = ROOT / "artifacts/v0.2/model-assets/acquisition.json"
IMPORT_SMOKE_PATH = ROOT / "artifacts/v0.2/environment/import-smoke.json"
EXPECTED_REPORT_SHA256 = "4491f2fb472df813642d296d92d396e62476a2fd257d6b9da431c3a90b6aa604"


def _report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_committed_strict_load_record_has_fixed_identity() -> None:
    report = _report()

    assert sha256_file(REPORT_PATH) == EXPECTED_REPORT_SHA256
    assert report["schema_version"] == "v0.2-weights-only-strict-load-v1"
    assert report["execution"] == {
        "execution_commit": "37d45145de18b8f9d98d6bfe18b98978b10c0c1c",
        "verification_date": "2026-07-31",
    }
    assert report["inputs"] == {
        "acquisition_record": "artifacts/v0.2/model-assets/acquisition.json",
        "acquisition_record_sha256": sha256_file(ACQUISITION_PATH),
        "import_smoke_record": "artifacts/v0.2/environment/import-smoke.json",
        "import_smoke_record_sha256": sha256_file(IMPORT_SMOKE_PATH),
        "preregistration_id": "v0.2-dinov2-cpu-preflight-1",
    }
    assert report["decision"]["next_step"] == (
        "PROCEED_TO_FIXED_DINOV2_SCORING_PATH_IMPLEMENTATION"
    )


def test_committed_strict_load_record_covers_fixed_source_and_environment() -> None:
    report = _report()
    source = report["source"]
    environment = report["environment"]

    assert source["identity"] == {
        "byte_count": 2_869_642,
        "revision": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        "sha256": "c27dcdaf50e9fb5bbdf2bb529da357716372e19c6afab17d5350f3f0094aed4b",
        "verification": "pass",
    }
    assert source["extraction"] == {
        "archive_sha256": (
            "c27dcdaf50e9fb5bbdf2bb529da357716372e19c6afab17d5350f3f0094aed4b"
        ),
        "directory_count": 68,
        "file_count": 203,
        "import_root": "dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        "safe_extraction": "pass",
        "tree_manifest_sha256": (
            "c49e1f0ff377e5f591cd4a9e87e47ea6e75b921545f24c8312216948566494b6"
        ),
    }
    assert source["module_import_count"] == 15
    assert all(
        item["origin"].startswith("dinov2/") for item in source["module_origins"]
    )
    assert environment["torch_version"] == "2.13.0+cpu"
    assert environment["intraop_threads"] == 4
    assert environment["interop_threads"] == 1
    assert environment["xformers_imported"] is False
    assert environment["accelerator_distribution_count"] == 0
    assert environment["cuda_build_version"] is None
    assert environment["hip_build_version"] is None


def test_committed_strict_load_record_covers_every_checkpoint_tensor() -> None:
    state = _report()["checkpoint"]["state_dictionary"]

    assert state["root_type"] == "builtins.dict"
    assert state["key_count"] == 175
    assert state["tensor_count"] == 175
    assert state["finite_tensor_count"] == 175
    assert state["total_tensor_elements"] == 22_056_576
    assert state["total_tensor_bytes"] == 88_226_304
    assert state["dtype_counts"] == {"float32": 175}
    assert state["device_counts"] == {"cpu": 175}
    assert state["state_key_manifest_sha256"] == (
        "21dec8566e545b724414a5881a72aa9590525cb81b8b38d416d21e7952eff0f1"
    )
    assert len(state["tensors"]) == 175
    assert [item["key"] for item in state["tensors"]] == sorted(
        item["key"] for item in state["tensors"]
    )
    assert all(
        item["dtype"] == "float32"
        and item["device"] == "cpu"
        and item["finite"] is True
        for item in state["tensors"]
    )
    tensors = {item["key"]: item for item in state["tensors"]}
    assert tensors["patch_embed.proj.weight"]["shape"] == [384, 3, 14, 14]
    assert tensors["pos_embed"]["shape"] == [1, 1370, 384]


def test_committed_strict_load_record_proves_exact_non_register_model_load() -> None:
    model = _report()["model"]

    assert model == {
        "buffer_element_count": 0,
        "class": "dinov2.models.vision_transformer.DinoVisionTransformer",
        "embedding_dimension": 384,
        "entry_point": "dinov2.hub.backbones.dinov2_vits14",
        "eval_mode": True,
        "exact_value_match_count": 175,
        "missing_keys": [],
        "num_heads": 6,
        "num_register_tokens": 0,
        "parameter_count": 22_056_576,
        "patch_size": 14,
        "register_tokens_is_none": True,
        "state_key_count": 175,
        "strict_load": "pass",
        "trainable_parameter_count": 22_056_576,
        "transformer_block_count": 12,
        "unexpected_keys": [],
    }


def test_committed_strict_load_record_preserves_non_inference_boundary() -> None:
    report = _report()

    assert report["boundary"] == {
        "accelerator_runtime_probe_performed": False,
        "checkpoint_deserialized": True,
        "checkpoint_pickle_executed_by_weights_only_loader": True,
        "checkpoint_tensor_values_inspected": True,
        "dataset_access": False,
        "feature_extraction_performed": False,
        "latency_measurement_performed": False,
        "model_constructed": True,
        "model_inference_performed": False,
        "network_access": False,
        "source_executed": True,
        "source_extracted": True,
        "synthetic_workload_generated": False,
        "tensor_operations_performed": True,
    }
    serialized = REPORT_PATH.read_text(encoding="utf-8")
    assert "/home/" not in serialized
    assert "\\\\wsl" not in serialized
    assert "generated_at" not in serialized
    assert "verified_at" not in serialized
