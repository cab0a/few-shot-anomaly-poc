from __future__ import annotations

import json
from pathlib import Path

from few_shot_anomaly_poc.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "artifacts/v0.2/dependencies/wheel-inspection.json"
LOCK_PATH = ROOT / "environments/v0.2-preflight/uv.lock"
ENVIRONMENT_PATH = ROOT / "environments/v0.2-preflight/pyproject.toml"
EXPECTED_REPORT_SHA256 = "402e35c32a7c31e2fd2470877f8047685a372e933d48167046366553eea1d0ad"
EXPECTED_NAMES = (
    "filelock",
    "fsspec",
    "iniconfig",
    "jinja2",
    "markupsafe",
    "mpmath",
    "networkx",
    "numpy",
    "packaging",
    "pluggy",
    "pygments",
    "pytest",
    "ruff",
    "setuptools",
    "sympy",
    "torch",
    "typing-extensions",
)


def _report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_committed_dependency_inspection_has_fixed_identity() -> None:
    report = _report()

    assert sha256_file(REPORT_PATH) == EXPECTED_REPORT_SHA256
    assert report["schema_version"] == "v0.2-dependency-artifact-inspection-v1"
    assert report["environment"]["lock_sha256"] == sha256_file(LOCK_PATH)
    assert report["environment"]["environment_definition_sha256"] == sha256_file(
        ENVIRONMENT_PATH
    )
    assert report["decision"]["installation"] == "INSTALL"


def test_committed_dependency_inspection_covers_every_locked_distribution() -> None:
    report = _report()
    packages = report["packages"]

    assert tuple(package["dependency"]["name"] for package in packages) == EXPECTED_NAMES
    assert report["summary"] == {
        "all_artifact_checksums_verified": True,
        "all_license_material_inventoried": True,
        "all_metadata_identities_verified": True,
        "all_wheel_records_verified": True,
        "all_wheels_safe_to_inspect": True,
        "development_distribution_count": 6,
        "distribution_count": 17,
        "license_material_count": 154,
        "native_file_count": 35,
        "runtime_distribution_count": 11,
    }
    assert sum(package["dependency"]["direct"] for package in packages) == 4
    assert {
        package["dependency"]["name"]: package["native_file_count"]
        for package in packages
        if package["native_file_count"]
    } == {
        "markupsafe": 1,
        "numpy": 22,
        "torch": 12,
    }


def test_committed_dependency_inspection_preserves_all_verification_boundaries() -> None:
    report = _report()

    assert report["acquisition_boundary"] == {
        "artifact_execution": False,
        "artifact_installation": False,
        "artifact_storage": "outside_git",
        "dataset_access": False,
        "dinov2_checkpoint_acquired": False,
        "dinov2_source_acquired": False,
        "package_import": False,
        "wheel_extraction": False,
    }
    for package in report["packages"]:
        assert package["artifact"]["observed_sha256"] == package["artifact"]["published_sha256"]
        assert package["artifact"]["checksum_status"] == "upstream_verified"
        assert package["archive"]["record_verification"] == "pass"
        assert package["archive"]["safe_member_validation"] == "pass"
        assert package["license"]["material_count"] >= 1

    serialized = REPORT_PATH.read_text(encoding="utf-8")
    assert "/home/" not in serialized
    assert "\\\\wsl" not in serialized
    assert "downloaded_at" not in serialized
