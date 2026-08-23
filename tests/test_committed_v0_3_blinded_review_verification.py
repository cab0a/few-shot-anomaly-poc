from __future__ import annotations

import json
from pathlib import Path

from few_shot_anomaly_poc.v0_3_diagnostic_contract import sha256_file

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "artifacts/v0.3/synthetic/blinded-review-verification.json"
REAL_DIAGNOSTIC_ROOT = ROOT / "artifacts/v0.3/diagnostics/pcb2-development"


def test_committed_v0_3_3_report_identity_and_checks_are_exact() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert sha256_file(REPORT_PATH) == (
        "4ef11f5097af2d7ce3dcc850ebd75bbefd9f96a80854286ef344eb30369da685"
    )
    assert report["report_version"] == "v0.3.3-synthetic-blinded-review-verification-v1"
    assert report["source_commit"] == "c2b5992863a861fd2baabd8a51121017cc0d0666"
    assert report["synthetic_fixture_only"] is True
    assert report["synthetic_asset_count"] == 28
    assert report["synthetic_observation_count"] == 28
    assert report["all_checks_passed"] is True
    assert len(report["checks"]) == 18
    assert all(report["checks"].values())


def test_committed_v0_3_3_boundary_records_only_synthetic_decoding() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    boundary = report["boundary"]

    assert boundary == {
        "anomaly_scorer_executed": False,
        "diagnostic_score_computed": False,
        "image_displayed": False,
        "raw_or_derived_image_committed": False,
        "real_diagnostic_artifacts_modified": False,
        "real_observation_written": False,
        "synthetic_image_decoded": True,
        "v0_3_selected_image_accessed": False,
        "visa_image_accessed": False,
    }
    assert not (REAL_DIAGNOSTIC_ROOT / "blind-observations.csv").exists()
    assert not (REAL_DIAGNOSTIC_ROOT / "review-completion-checkpoint.json").exists()


def test_committed_v0_3_3_report_contains_no_paths_or_protected_sentinel() -> None:
    text = REPORT_PATH.read_text(encoding="utf-8")

    assert "PROTECTED-SENTINEL-MUST-NOT-LEAK" not in text
    assert "/home/" not in text
    assert "\\\\" not in text
    assert "Anomaly/" not in text
    assert "Normal/" not in text
