from __future__ import annotations

import json
from pathlib import Path

from few_shot_anomaly_poc.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "artifacts/v0.2/environment/import-smoke.json"
INSPECTION_PATH = ROOT / "artifacts/v0.2/dependencies/wheel-inspection.json"
EXPECTED_REPORT_SHA256 = "b0f38afb103f7084a0e5e09e8fd00e4cf2e0e5825d7a3fe8d5e3b48afd7b1f74"


def _report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_committed_import_smoke_has_fixed_identity_and_input() -> None:
    report = _report()

    assert sha256_file(REPORT_PATH) == EXPECTED_REPORT_SHA256
    assert report["schema_version"] == "v0.2-isolated-import-smoke-v1"
    assert report["inputs"] == {
        "wheel_inspection_record": (
            "artifacts/v0.2/dependencies/wheel-inspection.json"
        ),
        "wheel_inspection_record_sha256": sha256_file(INSPECTION_PATH),
    }
    assert report["decision"]["next_step"] == (
        "PROCEED_TO_CONTROLLED_MODEL_ASSET_ACQUISITION"
    )


def test_committed_import_smoke_covers_exact_environment() -> None:
    report = _report()

    assert report["summary"] == {
        "accelerator_distribution_count": 0,
        "development_distribution_count": 6,
        "distribution_count": 17,
        "import_pass_count": 16,
        "metadata_only_count": 1,
        "runtime_distribution_count": 11,
    }
    assert report["environment"] == {
        "environment_path": "environments/v0.2-preflight/.venv",
        "forbidden_root_environment_visible": False,
        "implementation": "CPython",
        "isolated_prefix": True,
        "platform": "linux-x86_64",
        "python_executable": "environments/v0.2-preflight/.venv/bin/python",
        "python_version": "3.13.14",
        "user_site_enabled": False,
    }
    assert report["installation"] == {
        "artifact_installation_source": "verified_external_wheel_set",
        "completed_method": "uv_pip_install_exact_local_wheels_no_deps",
        "dependency_compatibility_check": "pass",
        "initial_locked_sync_attempt": {
            "package_installation_occurred": False,
            "reason_code": "EXPLICIT_INDEX_NOT_REPLACED_BY_NO_INDEX_FIND_LINKS",
            "result": "stopped_before_install",
        },
        "installer": "uv 0.11.32",
        "network_access": False,
        "source_build": False,
        "wheel_count": 17,
    }


def test_committed_import_smoke_preserves_cpu_only_and_non_execution_boundary() -> None:
    report = _report()

    assert report["torch"] == {
        "build": "cpu",
        "cuda_build_version": None,
        "distribution_version": "2.13.0+cpu",
        "hip_build_version": None,
        "module_version": "2.13.0+cpu",
        "runtime_accelerator_probe_performed": False,
    }
    assert report["boundary"] == {
        "accelerator_runtime_probe_performed": False,
        "dataset_access": False,
        "dinov2_checkpoint_acquired": False,
        "dinov2_source_acquired": False,
        "model_constructed": False,
        "model_inference_performed": False,
        "network_access_during_verification": False,
        "tensor_operation_performed": False,
    }
    assert all(
        item["origin"] is None
        or item["origin"].startswith("lib/python3.13/site-packages/")
        for item in report["imports"]
    )
    assert {
        item["distribution"]
        for item in report["imports"]
        if item["status"] == "metadata_only_no_python_module"
    } == {"ruff"}

    serialized = REPORT_PATH.read_text(encoding="utf-8")
    assert "/home/" not in serialized
    assert "\\\\wsl" not in serialized
    assert "verified_at" not in serialized
