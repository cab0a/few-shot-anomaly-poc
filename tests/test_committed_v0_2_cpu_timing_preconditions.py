from __future__ import annotations

import json
from pathlib import Path

from few_shot_anomaly_poc.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "artifacts/v0.2/cpu-preflight/attempt-001-target-machine-stop.json"
EXPECTED_REPORT_SHA256 = "b334ae369437636cc7c4e368e48e73687f3344a12a8e329134ba3871ed35a283"


def _report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_committed_cpu_preflight_record_has_fixed_identity() -> None:
    report = _report()

    assert sha256_file(REPORT_PATH) == EXPECTED_REPORT_SHA256
    assert report["schema_version"] == "v0.2-cpu-timing-preconditions-v1"
    assert report["execution"] == {
        "execution_commit": "1dd2f9b45a9e3b4bfd63c6083e738895e0e97d2f",
        "preregistration_commit": "e9330be10742947e4227ced4c99acafe4d098566",
        "verification_date": "2026-08-01",
        "worktree_clean": True,
    }
    assert report["inputs"]["preregistration_sha256"] == (
        "19d4cf4079c6df7c9042be464859ccf98d41108656ba0259c8940ace740ebf42"
    )
    assert all(
        item["verification"] == "pass"
        for item in report["inputs"]["required_records"].values()
    )


def test_committed_cpu_preflight_applies_ordered_stop_at_target_machine() -> None:
    report = _report()

    assert report["decision"] == {
        "first_failed_condition": 6,
        "next_step": "DO_NOT_START_TIMING_WORKLOAD",
        "outcome": "DO NOT PROCEED",
        "status": "stop",
    }
    assert [item["status"] for item in report["ordered_stop_conditions"]] == [
        "pass",
        "pass",
        "pass",
        "pass",
        "pass",
        "fail",
        "not_evaluated",
        "not_evaluated",
        "not_evaluated",
        "not_evaluated",
    ]


def test_committed_cpu_preflight_records_exact_ram_shortfall() -> None:
    target = _report()["target_machine"]

    assert target["evaluation"] == {
        "checks": {
            "ac_power": True,
            "cpu_affinity": True,
            "cpu_model": True,
            "logical_core_count": True,
            "machine": True,
            "nice": True,
            "physical_core_count": True,
            "ram_bytes": False,
            "scheduler": True,
            "sys_platform": True,
            "wsl2": True,
        },
        "failures": ["ram_bytes"],
        "status": "fail",
    }
    assert target["observed"]["ram_bytes"] == 4_045_004_800
    assert target["required"]["minimum_ram_bytes"] == 4_045_017_088
    assert target["required"]["minimum_ram_bytes"] - target["observed"]["ram_bytes"] == 12_288


def test_committed_cpu_preflight_preserves_no_timing_boundary() -> None:
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
