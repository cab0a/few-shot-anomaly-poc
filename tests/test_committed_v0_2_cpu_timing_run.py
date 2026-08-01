from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from few_shot_anomaly_poc.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts/v0.2/cpu-timing/first-fixed-memory-bounded-run"
EXPECTED_SHA256 = {
    "input-manifest.json": ("0760e26a49e11130b66bae57e287d8593dea666feda0a6c08041baf77e2c7dec"),
    "preconditions.json": ("331ba1549f33fe0da920472a5c3667051b09e09e1fa08f370331df7484c5229e"),
    "resolution-224-observations.csv": (
        "59492ba66335313c3d124900c940008bfc5b62c735a73fc9e2c925b3b50bf90d"
    ),
    "resolution-224.json": ("2c0820771cc435f23b79c053d1c705f2bf1fcd875a7bb46a9d12706fddc4c3d4"),
    "resolution-448-observations.csv": (
        "0b3e91ff89671185e9226a1be09a65d0fb9d4e15aa4ce32dc74497b050059802"
    ),
    "resolution-448.json": ("7988270f82eb2a6e34b7f429fe755dd55d8b1e287fefbd2353191e50966c875a"),
    "summary.json": ("64f8ca7b56ad861f8b3dd821f1b0d07dac3cff6809fa331e1042dbae7a9dfd71"),
}


def _json(name: str) -> dict[str, object]:
    return json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))


def _csv_rows(resolution: int) -> list[dict[str, str]]:
    path = ARTIFACT_DIR / f"resolution-{resolution}-observations.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_committed_timing_files_have_fixed_identity() -> None:
    actual_names = {path.name for path in ARTIFACT_DIR.iterdir() if path.is_file()}

    assert actual_names == set(EXPECTED_SHA256)
    assert {
        name: sha256_file(ARTIFACT_DIR / name) for name in sorted(actual_names)
    } == EXPECTED_SHA256


def test_committed_timing_summary_preserves_execution_and_evaluation_boundary() -> None:
    summary = _json("summary.json")

    assert summary["schema_version"] == "v0.2-dinov2-timing-parent-v1"
    assert summary["execution"] == {
        "execution_commit": "d02da60f622090746c8348704e550dccf57358d5",
        "process_order": [224, 448],
        "verification_date": "2026-08-01",
    }
    assert summary["boundary"] == {
        "dataset_access": False,
        "labels_accessed": False,
        "synthetic_inputs_only": True,
        "threshold_calibration_performed": False,
        "timing_invocation_count": 600,
    }
    assert summary["decision"] == {
        "next_step": "PROCEED_TO_OFFLINE_REPRODUCTION",
        "resolution_pass": {"224": True, "448": False},
        "selected_resolution_candidate": 224,
        "status": "pass",
    }


def test_committed_timing_input_manifest_records_bounded_raw_store_exclusion() -> None:
    manifest = _json("input-manifest.json")

    assert manifest["preregistration_id"] == "v0.2-dinov2-cpu-preflight-2"
    assert len(manifest["references"]) == 20
    assert len(manifest["queries"]) == 100
    assert manifest["logical_store"] == {
        "byte_count": 94_371_968,
        "dtype": "uint8",
        "file_sha256": ("b57319a8aa9fc8c27d1daa22acf8640a31cf366074a2c42e14e65ff55f4501b7"),
        "format": "NumPy .npy memory map",
        "logical_id": "synthetic-pcg64-42-memory-bounded-v1",
        "path_recorded": False,
        "shape": [120, 512, 512, 3],
    }
    assert manifest["resident_policy"] == {
        "all_source_images_retained_in_process_memory": False,
        "current_source_image_count": 1,
        "memory_map_to_contiguous_copy_outside_timer": True,
    }


def test_committed_timing_observations_match_json_and_recomputed_latency() -> None:
    expected = {
        224: {
            "median_ns": 417_283_103.0,
            "p95_ns": 679_490_250,
            "peak_rss_bytes": 543_424_512,
            "status": "pass",
            "latency_gate_passed": True,
        },
        448: {
            "median_ns": 2_502_270_558.5,
            "p95_ns": 3_625_274_025,
            "peak_rss_bytes": 605_278_208,
            "status": "fail",
            "latency_gate_passed": False,
        },
    }

    for resolution, fixed in expected.items():
        report = _json(f"resolution-{resolution}.json")
        rows = _csv_rows(resolution)
        observations = report["loop"]["observations"]
        durations = [int(row["duration_ns"]) for row in rows]
        scores = [float(row["score"]) for row in rows]
        nearest_rank = math.ceil(0.95 * len(durations))

        assert len(rows) == len(observations) == 300
        assert [int(row["invocation_index"]) for row in rows] == list(range(300))
        assert [int(row["pass_index"]) for row in rows] == [
            pass_index for pass_index in range(3) for _ in range(100)
        ]
        assert [int(row["query_index"]) for row in rows] == list(range(100)) * 3
        assert all(row["status"] == "success" for row in rows)
        assert all(math.isfinite(score) for score in scores)
        assert all(
            int(row["invocation_index"]) == observation["invocation_index"]
            and int(row["pass_index"]) == observation["pass_index"]
            and int(row["query_index"]) == observation["query_index"]
            and row["asset_id"] == observation["asset_id"]
            and row["status"] == observation["status"]
            and int(row["duration_ns"]) == observation["duration_ns"]
            and float(row["score"]) == observation["score"]
            for row, observation in zip(rows, observations, strict=True)
        )

        loop_summary = report["loop"]["summary"]
        assert statistics.median(durations) == loop_summary["median_ns"]
        assert sorted(durations)[nearest_rank - 1] == loop_summary["p95_ns"]
        assert loop_summary["failure_count"] == 0
        assert loop_summary["missing_invocation_count"] == 0
        assert loop_summary["median_ns"] == fixed["median_ns"]
        assert loop_summary["p95_ns"] == fixed["p95_ns"]
        assert loop_summary["latency_gate_passed"] is fixed["latency_gate_passed"]
        assert report["decision"]["status"] == fixed["status"]
        assert report["memory"]["after_timing_peak_rss_bytes"] == fixed["peak_rss_bytes"]


def test_committed_timing_artifacts_exclude_raw_data_logs_and_machine_paths() -> None:
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
