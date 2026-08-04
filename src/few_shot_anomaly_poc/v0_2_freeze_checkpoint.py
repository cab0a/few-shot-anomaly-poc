"""Create the v0.2.3 pre-evaluation freeze without reading labels or images."""

from __future__ import annotations

import json
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.jsonio import write_json_atomic
from few_shot_anomaly_poc.opaque_boundary import load_scoring_manifest
from few_shot_anomaly_poc.v0_2_boundary_preparation import (
    BOUNDARY_STATE_SCHEMA,
    RUN_ID,
    validate_boundary_execution_identity,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    ARTIFACT_CONTRACT_VERSION,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_SCHEMA_SHA256,
    HARD_GATES,
    METHODS,
    PREREGISTRATION_COMMIT,
    PREREGISTRATION_DOCUMENT_SHA256,
    load_v0_2_artifact_schema,
    load_v0_2_config,
    validate_json_artifact,
)

MILESTONE_LABEL = "v0.2.3"
BOUNDARY_EXECUTION_COMMIT = "0fd5ec2d9d477fe9298f6b5468cec35a1111314f"
BOUNDARY_RECORD_COMMIT = "6ba9f59648ae909b5104869b4737091b0ceacd71"
MILESTONE_MAP_COMMIT = "73128e71ade5e175e5bbb65d0037ffb21cb4296b"
BOUNDARY_RECORD_SHA256 = "e122bfa51ce618e0588a580f2cf66447c44a2cf801f08f851cce9d5271a4c698"
EXPECTED_EXTERNAL_TOP_LEVEL = {
    "archive-identity.json",
    "boundary-state.json",
    "extraction.json",
    "normal-manifests",
    "scorer",
    "sealed",
    "source",
}


class V0_2FreezeCheckpointError(Exception):
    """Reject an incomplete, changed, revealed, or already-started boundary."""


@dataclass(frozen=True)
class BoundaryFreezeEvidence:
    """Return only identities permitted in the public freeze record."""

    calibration_manifest_sha256: str
    reference_manifest_sha256: str
    scoring_manifest_sha256: str
    sealed_mapping_sha256: str


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V0_2FreezeCheckpointError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise V0_2FreezeCheckpointError(f"{label} must contain one JSON object")
    return value


def _require(value: bool, message: str) -> None:
    if not value:
        raise V0_2FreezeCheckpointError(message)


def _line_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return sum(1 for line in stream if line.strip())
    except (OSError, UnicodeError) as error:
        raise V0_2FreezeCheckpointError("cannot count normal manifest records") from error


def _validate_external_inventory(external_root: Path) -> None:
    try:
        names = {path.name for path in external_root.iterdir()}
    except OSError as error:
        raise V0_2FreezeCheckpointError("cannot inspect external boundary root") from error
    _require(names == EXPECTED_EXTERNAL_TOP_LEVEL, "external boundary inventory changed")


def _validate_pre_freeze_public_inventory(public_artifact_root: Path) -> None:
    files = sorted(
        path.relative_to(public_artifact_root).as_posix()
        for path in public_artifact_root.rglob("*")
        if path.is_file()
    )
    _require(
        files == ["boundary/boundary-record.json"],
        "public evaluation work started before the freeze",
    )


def validate_boundary_freeze_evidence(
    *,
    external_root: Path,
    public_artifact_root: Path,
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
    expected_boundary_record_sha256: str | None = None,
    expected_boundary_execution_commit: str | None = None,
) -> BoundaryFreezeEvidence:
    """Validate boundary identities without parsing the sealed label mapping."""
    if not external_root.is_dir() or external_root.is_symlink():
        raise V0_2FreezeCheckpointError("external boundary root is invalid")
    if not public_artifact_root.is_dir() or public_artifact_root.is_symlink():
        raise V0_2FreezeCheckpointError("public artifact root is invalid")
    _validate_external_inventory(external_root)
    _validate_pre_freeze_public_inventory(public_artifact_root)

    state = _read_json(external_root / "boundary-state.json", label="boundary state")
    _require(state.get("schema_version") == BOUNDARY_STATE_SCHEMA, "boundary state changed")
    _require(state.get("run_id") == RUN_ID, "boundary run ID changed")
    if expected_boundary_execution_commit is not None:
        _require(
            state.get("execution", {}).get("execution_commit")
            == expected_boundary_execution_commit,
            "boundary execution commit changed",
        )
    _require(
        state.get("contract")
        == {
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "schema_sha256": EXPECTED_SCHEMA_SHA256,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "preregistration_document_sha256": PREREGISTRATION_DOCUMENT_SHA256,
        },
        "boundary contract identities changed",
    )
    _require(
        state.get("boundary")
        == {
            "anomaly_score_computed": False,
            "final_test_label_revealed": False,
            "image_content_decoded": False,
            "image_content_displayed": False,
            "method_fit_performed": False,
            "raw_data_in_git": False,
            "threshold_calibrated": False,
        },
        "evaluation advanced before the freeze",
    )

    normal = state.get("normal_partitions")
    opaque = state.get("opaque_final_test")
    _require(isinstance(normal, dict), "normal partition evidence is missing")
    _require(isinstance(opaque, dict), "opaque boundary evidence is missing")
    reference_path = external_root / "normal-manifests/reference.jsonl"
    calibration_path = external_root / "normal-manifests/calibration.jsonl"
    scoring_path = external_root / "scorer/scoring-manifest.json"
    sealed_path = external_root / "sealed/mapping.json"
    key_path = external_root / "sealed/ordering-key.bin"
    _require(reference_path.is_file(), "reference manifest is missing")
    _require(calibration_path.is_file(), "calibration manifest is missing")
    _require(scoring_path.is_file(), "scoring manifest is missing")
    _require(sealed_path.is_file(), "sealed mapping is missing")
    _require(key_path.is_file() and not key_path.is_symlink(), "ordering key is invalid")

    reference_sha256 = sha256_file(reference_path)
    calibration_sha256 = sha256_file(calibration_path)
    scoring_sha256 = sha256_file(scoring_path)
    sealed_sha256 = sha256_file(sealed_path)
    _require(
        normal.get("reference_count") == _line_count(reference_path)
        and normal.get("reference_count") == 20,
        "reference manifest count changed",
    )
    _require(
        normal.get("calibration_count") == _line_count(calibration_path)
        and normal.get("calibration_count") > 0,
        "calibration manifest count changed",
    )
    _require(
        normal.get("reference_manifest_sha256") == reference_sha256,
        "reference manifest SHA-256 changed",
    )
    _require(
        normal.get("calibration_manifest_sha256") == calibration_sha256,
        "calibration manifest SHA-256 changed",
    )

    scoring = load_scoring_manifest(scoring_path)
    _require(
        opaque.get("asset_count") == len(scoring["records"]) and opaque.get("asset_count") > 0,
        "opaque asset count changed",
    )
    _require(
        opaque.get("scoring_manifest_sha256") == scoring_sha256,
        "scoring manifest SHA-256 changed",
    )
    _require(
        opaque.get("sealed_mapping_sha256") == sealed_sha256,
        "sealed mapping SHA-256 changed",
    )
    _require(opaque.get("class_counts_published") is False, "class counts were published")
    key_mode = stat.S_IMODE(key_path.stat().st_mode)
    _require(key_path.stat().st_size == 32 and key_mode == 0o600, "ordering key changed")
    _require(
        opaque.get("ordering_key_sha256") == sha256_file(key_path),
        "ordering key SHA-256 changed",
    )

    boundary_path = public_artifact_root / "boundary/boundary-record.json"
    boundary_sha256 = sha256_file(boundary_path)
    if expected_boundary_record_sha256 is not None:
        _require(
            boundary_sha256 == expected_boundary_record_sha256,
            "public boundary record SHA-256 changed",
        )
    boundary = validate_json_artifact(
        "boundary_record",
        _read_json(boundary_path, label="public boundary record"),
        config=config,
        schema=schema,
    )
    _require(
        boundary["reference_count"] == normal["reference_count"]
        and boundary["calibration_count"] == normal["calibration_count"]
        and boundary["final_test_count"] == opaque["asset_count"],
        "public and external partition counts differ",
    )
    _require(
        boundary["scoring_manifest_sha256"] == scoring_sha256
        and boundary["sealed_mapping_sha256"] == sealed_sha256,
        "public and external opaque identities differ",
    )
    _require(
        boundary["final_test_class_counts_published"] is False
        and boundary["raw_data_in_git"] is False,
        "public boundary assertions changed",
    )
    return BoundaryFreezeEvidence(
        calibration_manifest_sha256=calibration_sha256,
        reference_manifest_sha256=reference_sha256,
        scoring_manifest_sha256=scoring_sha256,
        sealed_mapping_sha256=sealed_sha256,
    )


def build_v0_2_freeze_record(
    *,
    source_commit: str,
    evidence: BoundaryFreezeEvidence,
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact public pre-evaluation freeze artifact."""
    record = {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "run_id": RUN_ID,
        "run_kind": "final_test",
        "source_commit": source_commit,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "schema_sha256": EXPECTED_SCHEMA_SHA256,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_document_sha256": PREREGISTRATION_DOCUMENT_SHA256,
        "reference_manifest_sha256": evidence.reference_manifest_sha256,
        "calibration_manifest_sha256": evidence.calibration_manifest_sha256,
        "scoring_manifest_sha256": evidence.scoring_manifest_sha256,
        "sealed_mapping_sha256": evidence.sealed_mapping_sha256,
        "method_order": list(METHODS),
        "hard_gate_order": list(HARD_GATES),
        "label_reveal_completed": False,
    }
    return validate_json_artifact(
        "pre_evaluation_freeze",
        record,
        config=config,
        schema=schema,
    )


def _require_freeze_ancestors(project_root: Path, source_commit: str) -> None:
    for commit in (BOUNDARY_RECORD_COMMIT, MILESTONE_MAP_COMMIT):
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, source_commit],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise V0_2FreezeCheckpointError(
                "freeze source does not contain the boundary record and milestone map"
            )


def prepare_v0_2_freeze(
    *,
    project_root: Path,
    source_commit: str,
    external_root: Path,
    public_artifact_root: Path,
) -> tuple[dict[str, Any], str]:
    """Validate the untouched boundary and write one non-overwritable freeze."""
    project_root = project_root.resolve()
    external_root = external_root.resolve()
    public_artifact_root = public_artifact_root.resolve()
    expected_external_root = (project_root / "data/external/v0.2/evaluation" / RUN_ID).resolve()
    expected_public_root = (project_root / "artifacts/v0.2/evaluation" / RUN_ID).resolve()
    if external_root != expected_external_root:
        raise V0_2FreezeCheckpointError("external_root differs from the fixed path")
    if public_artifact_root != expected_public_root:
        raise V0_2FreezeCheckpointError("public artifact root differs from the contract")
    validate_boundary_execution_identity(
        project_root=project_root,
        execution_commit=source_commit,
    )
    _require_freeze_ancestors(project_root, source_commit)

    config = load_v0_2_config(project_root / "configs/v0.2.yaml")
    schema = load_v0_2_artifact_schema(project_root / "schemas/v0.2/evaluation-artifacts.json")
    evidence = validate_boundary_freeze_evidence(
        external_root=external_root,
        public_artifact_root=public_artifact_root,
        config=config,
        schema=schema,
        expected_boundary_record_sha256=BOUNDARY_RECORD_SHA256,
        expected_boundary_execution_commit=BOUNDARY_EXECUTION_COMMIT,
    )
    record = build_v0_2_freeze_record(
        source_commit=source_commit,
        evidence=evidence,
        config=config,
        schema=schema,
    )
    freeze_path = public_artifact_root / "freeze/pre-evaluation-freeze.json"
    write_json_atomic(freeze_path, record)
    return record, sha256_file(freeze_path)
