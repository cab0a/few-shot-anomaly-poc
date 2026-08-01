from __future__ import annotations

import json
from pathlib import Path

from few_shot_anomaly_poc.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "artifacts/v0.2/cpu-preflight/attempt-002-memory-bounded-pass.json"
ATTEMPT_1_PATH = ROOT / "artifacts/v0.2/cpu-preflight/attempt-001-target-machine-stop.json"
PREREGISTRATION_PATH = ROOT / "docs/v0.2-memory-bounded-cpu-preflight.md"
EXPECTED_REPORT_SHA256 = "f9befbd3df1c980f1dc0a8dc48563fd4dffbecd24530b2a4e2b413b4e688715d"


def _report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_committed_memory_bounded_preflight_has_fixed_identity() -> None:
    report = _report()

    assert sha256_file(REPORT_PATH) == EXPECTED_REPORT_SHA256
    assert report["schema_version"] == "v0.2-cpu-timing-preconditions-v2"
    assert report["execution"] == {
        "execution_commit": "ee93021f0120645b072b357907dab2b2b98d1861",
        "preregistration_commit": "a177b5648c450b1e33ca3bbf5c16a051410ef756",
        "superseded_preregistration_commit": "e9330be10742947e4227ced4c99acafe4d098566",
        "verification_date": "2026-08-01",
        "worktree_clean": True,
    }
    assert report["inputs"]["preregistration_id"] == "v0.2-dinov2-cpu-preflight-2"
    assert report["inputs"]["preregistration_sha256"] == sha256_file(
        PREREGISTRATION_PATH
    )
    assert report["inputs"]["required_records"]["first_cpu_preflight_attempt"] == {
        "path": "artifacts/v0.2/cpu-preflight/attempt-001-target-machine-stop.json",
        "sha256": sha256_file(ATTEMPT_1_PATH),
        "verification": "pass",
    }


def test_committed_memory_bounded_preflight_passes_without_waiving_cpu_identity() -> None:
    report = _report()

    assert report["decision"] == {
        "first_failed_condition": None,
        "next_step": "PROCEED_TO_FRESH_PROCESS_TIMING_RUN",
        "outcome": "PENDING",
        "status": "pass",
    }
    assert [item["status"] for item in report["ordered_stop_conditions"]] == [
        "pass",
        "pass",
        "pass",
        "pass",
        "pass",
        "pass",
        "pending",
        "pending",
        "pending",
        "pending",
    ]
    target = report["target_machine"]
    assert target["evaluation"]["status"] == "pass"
    assert target["evaluation"]["failures"] == []
    assert all(target["evaluation"]["checks"].values())


def test_committed_memory_bounded_preflight_records_ram_as_diagnostic() -> None:
    target = _report()["target_machine"]

    assert target["observed"]["ram_bytes"] == 4_045_004_800
    assert target["observed"]["mem_total_bytes"] == 4_045_004_800
    assert target["observed"]["mem_available_bytes"] == 3_086_802_944
    assert target["observed"]["swap_total_bytes"] == 1_073_741_824
    assert target["observed"]["swap_free_bytes"] == 1_073_741_824
    assert target["evaluation"]["diagnostics"] == {
        "meminfo_status": "available",
        "ram_bytes_status": "available",
        "superseded_ram_snapshot_bytes": 4_045_017_088,
        "total_ram_is_gating": False,
        "total_ram_meets_superseded_snapshot": False,
    }
    assert "ram_bytes" not in target["evaluation"]["checks"]
    assert target["resource_policy"]["total_ram_is_gating"] is False
    assert target["resource_policy"]["peak_rss_is_gating"] is False


def test_committed_memory_bounded_preflight_preserves_no_timing_boundary() -> None:
    report = _report()

    assert report["boundary"] == {
        "dataset_access": False,
        "labels_accessed": False,
        "model_constructed": False,
        "model_inference_performed": False,
        "network_access": False,
        "scoring_performed": False,
        "timing_invocation_count": 0,
    }
    serialized = REPORT_PATH.read_text(encoding="utf-8")
    assert "/home/" not in serialized
    assert "\\\\wsl" not in serialized
    assert "VisA" not in serialized
    assert "pcb1" not in serialized
    assert "pcb2" not in serialized
