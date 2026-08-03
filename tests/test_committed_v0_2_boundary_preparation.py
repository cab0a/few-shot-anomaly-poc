from __future__ import annotations

import json
from pathlib import Path

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    load_v0_2_artifact_schema,
    load_v0_2_config,
    validate_json_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_DIR = ROOT / "artifacts/v0.2/evaluation/visa-pcb2-v0-2-final/boundary"
BOUNDARY_PATH = BOUNDARY_DIR / "boundary-record.json"
EXPECTED_BOUNDARY_SHA256 = "e122bfa51ce618e0588a580f2cf66447c44a2cf801f08f851cce9d5271a4c698"


def _record() -> dict:
    return json.loads(BOUNDARY_PATH.read_text(encoding="utf-8"))


def test_committed_boundary_directory_contains_only_the_allowlisted_record() -> None:
    files = sorted(
        path.relative_to(BOUNDARY_DIR).as_posix()
        for path in BOUNDARY_DIR.rglob("*")
        if path.is_file()
    )

    assert files == ["boundary-record.json"]
    assert sha256_file(BOUNDARY_PATH) == EXPECTED_BOUNDARY_SHA256


def test_committed_boundary_record_validates_against_the_fixed_contract() -> None:
    config = load_v0_2_config(ROOT / "configs/v0.2.yaml")
    schema = load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json")

    validated = validate_json_artifact(
        "boundary_record",
        _record(),
        config=config,
        schema=schema,
    )

    assert validated == {
        "archive_sha256": ("2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362"),
        "calibration_count": 881,
        "contract_version": "evaluation-artifacts/v0.2",
        "final_test_class_counts_published": False,
        "final_test_count": 200,
        "preregistration_commit": "b873bacc4f677a4c82f3944c09a7374037cb7c50",
        "preregistration_document_sha256": (
            "6306c2122f69aa96dcfe1f377518e7c6795a096eceb085be5b829262b55482b9"
        ),
        "raw_data_in_git": False,
        "reference_count": 20,
        "run_id": "visa-pcb2-v0-2-final",
        "run_kind": "final_test",
        "scoring_manifest_sha256": (
            "32ea52ed1b9872f39ae27f5d58a353ea84b8b143642e3a7f0fabe940184705e8"
        ),
        "sealed_mapping_sha256": (
            "0e40e3777e797fd5099fd7ea8fd307547e396785dc33219f0bc0b3120e3dfe39"
        ),
        "split_sha256": ("a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995"),
    }


def test_committed_boundary_record_excludes_protected_or_machine_local_values() -> None:
    record = _record()
    serialized = BOUNDARY_PATH.read_text(encoding="utf-8")

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
        path.suffix.lower() in {".jpg", ".jpeg", ".png"} for path in BOUNDARY_DIR.rglob("*")
    )
