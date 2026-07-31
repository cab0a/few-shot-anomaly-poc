from __future__ import annotations

import json
from pathlib import Path

from few_shot_anomaly_poc.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "artifacts/v0.2/scoring-path/synthetic-smoke.json"
ACQUISITION_PATH = ROOT / "artifacts/v0.2/model-assets/acquisition.json"
IMPORT_SMOKE_PATH = ROOT / "artifacts/v0.2/environment/import-smoke.json"
STRICT_LOAD_PATH = ROOT / "artifacts/v0.2/model-compatibility/strict-load.json"
EXPECTED_REPORT_SHA256 = "56b5f342c3b8875df6c9baec61fdd8339c0f40d1f69c83245bdbf580ac23f7b8"


def _report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_committed_scoring_smoke_has_fixed_execution_identity() -> None:
    report = _report()

    assert sha256_file(REPORT_PATH) == EXPECTED_REPORT_SHA256
    assert report["schema_version"] == "v0.2-fixed-dinov2-scoring-smoke-v1"
    assert report["execution"] == {
        "execution_commit": "d88a0339b7da7370d95fcf05865f1646b205cb95",
        "verification_date": "2026-07-31",
    }
    assert report["inputs"]["acquisition_record_sha256"] == sha256_file(ACQUISITION_PATH)
    assert report["inputs"]["import_smoke_record_sha256"] == sha256_file(IMPORT_SMOKE_PATH)
    assert report["inputs"]["strict_load_record_sha256"] == sha256_file(STRICT_LOAD_PATH)


def test_committed_scoring_smoke_preserves_non_performance_boundary() -> None:
    report = _report()

    assert report["boundary"] == {
        "accelerator_runtime_probe_performed": False,
        "dataset_access": False,
        "formal_latency_measurement_performed": False,
        "labels_accessed": False,
        "model_inference_performed": True,
        "network_access": False,
        "performance_claim": False,
        "synthetic_inputs_only": True,
        "threshold_calibration_performed": False,
    }
    assert report["decision"] == {
        "next_step": "PROCEED_TO_PREREGISTERED_CPU_TIMING_WORKLOAD",
        "reason": (
            "The fixed scoring path completed for both resolutions and matched "
            "an independent exact NumPy calculation within tolerance."
        ),
        "status": "PASS",
    }


def test_committed_scoring_smoke_uses_fixed_model_and_arithmetic_contract() -> None:
    report = _report()

    assert report["fixed_contract"] == {
        "allowed_resolutions": [224, 448],
        "embedding_dimension": 384,
        "l2_epsilon": 1e-12,
        "memory_block_size": 2_048,
        "model_entry_point": "dinov2.hub.backbones.dinov2_vits14",
        "reference_count": 20,
        "top_fraction": 0.01,
    }
    assert report["model"] == {
        "embedding_dimension": 384,
        "entry_point": "dinov2.hub.backbones.dinov2_vits14",
        "eval_mode": True,
        "num_register_tokens": 0,
        "parameter_count": 22_056_576,
        "patch_size": 14,
        "strict_load": "pass",
    }
    assert report["environment"]["torch_version"] == "2.13.0+cpu"
    assert report["environment"]["deterministic_algorithms"] is True
    assert report["environment"]["intraop_threads"] == 4
    assert report["environment"]["interop_threads"] == 1
    assert report["environment"]["xformers_imported"] is False


def test_committed_scoring_smoke_records_both_fixed_resolution_results() -> None:
    report = _report()
    results = {item["resolution"]: item for item in report["resolutions"]}

    assert sorted(results) == [224, 448]
    assert results[224]["patch_count"] == 256
    assert results[224]["top_patch_count"] == 2
    assert results[224]["memory_bank"]["patch_count"] == 5_120
    assert results[224]["score"] == 0.4834079146385193
    assert results[224]["independent_numpy_check"] == {
        "maximum_absolute_patch_distance_difference": 7.748603820800781e-07,
        "patch_distance_difference_is_gating": False,
        "score": 0.4834078550338745,
        "score_absolute_difference": 5.960464477539063e-08,
        "score_tolerance": 1e-06,
        "score_verification": "pass",
    }

    assert results[448]["patch_count"] == 1_024
    assert results[448]["top_patch_count"] == 10
    assert results[448]["memory_bank"]["patch_count"] == 20_480
    assert results[448]["score"] == 0.10851246118545532
    assert results[448]["independent_numpy_check"] == {
        "maximum_absolute_patch_distance_difference": 1.1920928955078125e-06,
        "patch_distance_difference_is_gating": False,
        "score": 0.10851237922906876,
        "score_absolute_difference": 8.195638656616211e-08,
        "score_tolerance": 1e-06,
        "score_verification": "pass",
    }
    assert all(all(item["finite"].values()) for item in results.values())
    assert all(
        item["memory_bank"]["unique_reference_image_count"] == 1
        and item["memory_bank"]["reference_count_contract"] == 20
        for item in results.values()
    )


def test_committed_scoring_smoke_contains_no_dataset_or_machine_path() -> None:
    serialized = REPORT_PATH.read_text(encoding="utf-8")
    report = _report()

    assert report["inputs"]["synthetic_reference"]["sha256"] == (
        "2177c217fa47c84eac86530410e60ebbf9e7c8ea35c9ffed8ce36d2ce172a550"
    )
    assert report["inputs"]["synthetic_query"]["sha256"] == (
        "ea629e99f050dea9639f24dfc11e9c74f2f329f4c4fbee332cc268a201e5e8d2"
    )
    assert "/home/" not in serialized
    assert "\\\\wsl" not in serialized
    assert "VisA" not in serialized
    assert "pcb1" not in serialized
    assert "pcb2" not in serialized
