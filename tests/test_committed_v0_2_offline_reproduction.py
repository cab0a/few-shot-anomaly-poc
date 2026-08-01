from __future__ import annotations

import json
import math
from pathlib import Path

from few_shot_anomaly_poc.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts/v0.2/offline-reproduction/first-fixed-run"
TIMING_PATH = ROOT / "artifacts/v0.2/cpu-timing/first-fixed-memory-bounded-run/resolution-224.json"
EXPECTED_SHA256 = {
    "input-manifest.json": ("0760e26a49e11130b66bae57e287d8593dea666feda0a6c08041baf77e2c7dec"),
    "reproduction.json": ("3b53ad4370377fa86394bb757b204f9f6dc05c38448e203d16d77ace8bf89aeb"),
    "summary.json": ("c6036f0214bf55cb0d7b208f51d54304aef9a3fe9688e4afffaf2dcbc0eafdd3"),
}


def _json(name: str) -> dict[str, object]:
    return json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))


def test_committed_reproduction_files_have_fixed_identity() -> None:
    actual_names = {path.name for path in ARTIFACT_DIR.iterdir() if path.is_file()}

    assert actual_names == set(EXPECTED_SHA256)
    assert {
        name: sha256_file(ARTIFACT_DIR / name) for name in sorted(actual_names)
    } == EXPECTED_SHA256


def test_committed_reproduction_preserves_execution_and_boundary() -> None:
    report = _json("reproduction.json")

    assert report["schema_version"] == "v0.2-dinov2-offline-reproduction-v1"
    assert report["execution"] == {
        "execution_commit": "50c9a3f1e02bd370b428989315f9ee8bca52c5fa",
        "fresh_process_required": True,
        "process_id_recorded": False,
        "resolution": 224,
        "verification_date": "2026-08-01",
    }
    assert report["boundary"] == {
        "dataset_access": False,
        "labels_accessed": False,
        "latency_measurement_performed": False,
        "model_constructed": True,
        "model_inference_performed": True,
        "network_access": False,
        "synthetic_inputs_only": True,
        "threshold_calibration_performed": False,
    }
    assert report["failure"] is None
    assert report["decision"] == {
        "next_step": "PROCEED_TO_EVALUATION_BOUNDARY_CHECK",
        "status": "pass",
    }


def test_committed_reproduction_exactly_matches_first_ten_timing_scores() -> None:
    report = _json("reproduction.json")
    timing = json.loads(TIMING_PATH.read_text(encoding="utf-8"))
    comparisons = report["reproduction"]["comparisons"]
    timing_rows = timing["loop"]["observations"][:10]

    assert len(comparisons) == 10
    assert [row["query_index"] for row in comparisons] == list(range(10))
    assert [row["asset_id"] for row in comparisons] == [
        f"synthetic/query/{index:03d}" for index in range(10)
    ]
    assert [row["expected_score"] for row in comparisons] == [row["score"] for row in timing_rows]
    assert [row["reproduced_score"] for row in comparisons] == [row["score"] for row in timing_rows]
    assert all(row["absolute_difference"] == 0.0 for row in comparisons)
    assert all(row["within_tolerance"] is True for row in comparisons)
    assert all(
        math.isfinite(row["expected_score"]) and math.isfinite(row["reproduced_score"])
        for row in comparisons
    )
    assert report["reproduction"]["summary"] == {
        "all_scores_finite": True,
        "asset_id_order_match": True,
        "attempted_count": 10,
        "complete_observation_set": True,
        "failure_count": 0,
        "maximum_absolute_difference": 0.0,
        "missing_count": 0,
        "required_count": 10,
        "status": "pass",
        "tolerance": 1e-6,
    }


def test_committed_reproduction_matches_every_fixed_identity() -> None:
    report = _json("reproduction.json")
    identities = report["identities"]

    assert identities["all_match"] is True
    assert set(identities["records"]) == {
        "checkpoint_sha256",
        "configuration_sha256",
        "environment_lock_sha256",
        "generated_input_manifest_sha256",
        "raw_input_store_sha256",
        "source_archive_sha256",
        "source_revision",
    }
    assert all(
        record["match"] is True and record["expected"] == record["observed"]
        for record in identities["records"].values()
    )
    assert report["memory"] == {
        "after_memory_bank_peak_rss_bytes": 498_180_096,
        "after_model_load_peak_rss_bytes": 498_180_096,
        "after_reproduction_peak_rss_bytes": 498_180_096,
        "peak_rss_is_gating": False,
        "process_start_peak_rss_bytes": 40_640_512,
        "units": "bytes",
    }


def test_committed_reproduction_parent_validates_worker_result() -> None:
    summary = _json("summary.json")

    assert summary["schema_version"] == ("v0.2-dinov2-offline-reproduction-parent-v1")
    assert summary["decision"] == {
        "next_step": "PROCEED_TO_EVALUATION_BOUNDARY_CHECK",
        "status": "pass",
    }
    assert summary["worker"] == {
        "artifact": "reproduction.json",
        "artifact_sha256": EXPECTED_SHA256["reproduction.json"],
        "fresh_process": True,
        "return_code": 0,
        "validation_failure": None,
    }


def test_committed_reproduction_excludes_raw_data_logs_and_machine_paths() -> None:
    serialized = "\n".join(
        (ARTIFACT_DIR / name).read_text(encoding="utf-8") for name in sorted(EXPECTED_SHA256)
    )

    assert not (ARTIFACT_DIR / "synthetic-inputs.npy").exists()
    assert not list(ARTIFACT_DIR.glob("*.log"))
    assert "/home/" not in serialized
    assert "\\\\wsl" not in serialized
    assert "VisA" not in serialized
    assert "pcb1" not in serialized
    assert "pcb2" not in serialized
