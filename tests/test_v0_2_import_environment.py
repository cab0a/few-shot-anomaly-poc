from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/verify_v0_2_import_environment.py"
INSPECTION_PATH = ROOT / "artifacts/v0.2/dependencies/wheel-inspection.json"
SCRIPT = runpy.run_path(str(SCRIPT_PATH))


def test_inspection_record_defines_every_expected_import() -> None:
    versions, roles = SCRIPT["_load_expected_distributions"](INSPECTION_PATH)

    assert set(versions) == set(SCRIPT["EXPECTED_IMPORTS"])
    assert len(versions) == 17
    assert versions["torch"] == "2.13.0+cpu"
    assert sum(role == "runtime" for role in roles.values()) == 11
    assert sum(role == "development" for role in roles.values()) == 6
    assert SCRIPT["EXPECTED_IMPORTS"]["ruff"] is None


def test_accelerator_distribution_detection_is_explicit() -> None:
    detect = SCRIPT["_accelerator_distributions"]

    assert detect({"numpy", "torch"}) == []
    assert detect({"torch", "nvidia-cublas-cu13", "triton"}) == [
        "nvidia-cublas-cu13",
        "triton",
    ]


def test_atomic_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    writer = SCRIPT["_write_json_atomic"]

    writer(output, {"schema_version": "test"})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema_version": "test"
    }
    with pytest.raises(FileExistsError, match="overwrite"):
        writer(output, {"schema_version": "changed"})
