from __future__ import annotations

import json
from pathlib import Path

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    HARD_GATES,
    METHODS,
    load_v0_2_artifact_schema,
    load_v0_2_config,
    validate_json_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
FREEZE_DIR = ROOT / "artifacts/v0.2/evaluation/visa-pcb2-v0-2-final/freeze"
FREEZE_PATH = FREEZE_DIR / "pre-evaluation-freeze.json"
EXPECTED_FREEZE_SHA256 = "ae552d805dd9648163a48683bad828c7e1b7ecc4f1d69f1fa28511363b08ce3b"


def _record() -> dict:
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


def test_committed_freeze_directory_contains_only_the_fixed_record() -> None:
    files = sorted(
        path.relative_to(FREEZE_DIR).as_posix() for path in FREEZE_DIR.rglob("*") if path.is_file()
    )

    assert files == ["pre-evaluation-freeze.json"]
    assert sha256_file(FREEZE_PATH) == EXPECTED_FREEZE_SHA256


def test_committed_freeze_record_validates_against_the_fixed_contract() -> None:
    config = load_v0_2_config(ROOT / "configs/v0.2.yaml")
    schema = load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json")

    validated = validate_json_artifact(
        "pre_evaluation_freeze",
        _record(),
        config=config,
        schema=schema,
    )

    assert validated["source_commit"] == "462738321da403ce1f77ef23b0cb823008c32694"
    assert validated["reference_manifest_sha256"] == (
        "e587f1808262480261ae8a7b940faff0d9ef5f83cf215028b31490ba48369b99"
    )
    assert validated["calibration_manifest_sha256"] == (
        "77d5adb588e7d463e7fcab1c10b841b9ad23d827b51138c31f42dac35bd99ca3"
    )
    assert validated["scoring_manifest_sha256"] == (
        "32ea52ed1b9872f39ae27f5d58a353ea84b8b143642e3a7f0fabe940184705e8"
    )
    assert validated["sealed_mapping_sha256"] == (
        "0e40e3777e797fd5099fd7ea8fd307547e396785dc33219f0bc0b3120e3dfe39"
    )
    assert validated["method_order"] == list(METHODS)
    assert validated["hard_gate_order"] == list(HARD_GATES)
    assert validated["label_reveal_completed"] is False


def test_committed_freeze_record_excludes_protected_or_machine_local_values() -> None:
    record = _record()
    serialized = FREEZE_PATH.read_text(encoding="utf-8")

    assert "true_class" not in record
    assert "class_label" not in record
    assert "source_path" not in record
    assert "relative_path" not in record
    assert "ordering_key" not in record
    assert "hmac_key" not in record
    assert "class_count" not in record
    assert "/home/" not in serialized
    assert "\\\\wsl" not in serialized
    assert not any(
        path.suffix.lower() in {".jpg", ".jpeg", ".png"} for path in FREEZE_DIR.rglob("*")
    )
