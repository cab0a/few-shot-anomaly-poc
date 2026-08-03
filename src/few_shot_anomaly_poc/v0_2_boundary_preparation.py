"""Prepare the fixed v0.2 VisA boundary without decoding or scoring images."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.jsonio import write_json_atomic
from few_shot_anomaly_poc.manifests import SplitRow, load_official_rows
from few_shot_anomaly_poc.opaque_boundary import (
    BoundarySourceRecord,
    prepare_opaque_boundary,
)
from few_shot_anomaly_poc.safe_tar import extract_archive_safely
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    ARTIFACT_CONTRACT_VERSION,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_SCHEMA_SHA256,
    PREREGISTRATION_COMMIT,
    PREREGISTRATION_DOCUMENT_SHA256,
    load_v0_2_artifact_schema,
    load_v0_2_config,
    validate_json_artifact,
)

BOUNDARY_STATE_SCHEMA = "v0.2-boundary-state-v1"
NORMAL_MANIFEST_SCHEMA = "v0.2-normal-partition-manifest-v1"
NORMAL_MANIFEST_SET_SCHEMA = "v0.2-normal-partition-manifest-set-v1"
RUN_ID = "visa-pcb2-v0-2-final"
CATEGORY = "pcb2"
REFERENCE_COUNT = 20
SELECTION_SEED = 42
SELECTION_NAMESPACE = "few-shot-anomaly-poc:v0.2:pcb2"
ARCHIVE_BYTE_COUNT = 1_929_840_640
CONTRACT_COMMIT = "c2adf6d9a0a849612414cca05404acc4dfc36274"
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class V0_2BoundaryPreparationError(Exception):
    """Reject an input or output that violates the fixed boundary."""


@dataclass(frozen=True)
class NormalPartitionRecord:
    """Identify one normal-only input without embedding image content."""

    byte_count: int
    partition: str
    relative_path: str
    selection_rank: int
    selection_sha256: str
    sha256: str


@dataclass(frozen=True)
class PreparedBoundary:
    """Return aggregate identities without exposing labels or semantic test paths."""

    calibration_count: int
    calibration_manifest_sha256: str
    final_test_count: int
    normal_manifest_set_sha256: str
    public_boundary_record_sha256: str
    reference_count: int
    reference_manifest_sha256: str
    scoring_manifest_sha256: str
    sealed_mapping_sha256: str


def _git(project_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise V0_2BoundaryPreparationError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def validate_boundary_execution_identity(
    *,
    project_root: Path,
    execution_commit: str,
) -> dict[str, Any]:
    """Require a clean pushed commit containing the fixed evaluation contract."""
    if not _COMMIT_PATTERN.fullmatch(execution_commit):
        raise V0_2BoundaryPreparationError("execution_commit must be a full Git commit")
    observed_commit = _git(project_root, "rev-parse", "HEAD")
    if observed_commit != execution_commit:
        raise V0_2BoundaryPreparationError("execution_commit does not match HEAD")
    if _git(project_root, "status", "--porcelain", "--untracked-files=all"):
        raise V0_2BoundaryPreparationError("worktree must be clean before boundary preparation")
    remote_commit = _git(project_root, "rev-parse", "origin/main")
    if remote_commit != execution_commit:
        raise V0_2BoundaryPreparationError("execution_commit must already be pushed to origin/main")
    for controlling_commit in (PREREGISTRATION_COMMIT, CONTRACT_COMMIT):
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", controlling_commit, execution_commit],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise V0_2BoundaryPreparationError(
                "execution commit does not contain every controlling contract"
            )
    return {
        "execution_commit": execution_commit,
        "origin_main_commit": remote_commit,
        "push_verified": True,
        "worktree_clean": True,
    }


def _selection_digest(relative_path: str) -> str:
    value = f"{SELECTION_NAMESPACE}:{SELECTION_SEED}:{relative_path}".encode()
    return hashlib.sha256(value).hexdigest()


def select_v0_2_partitions(
    rows: Sequence[SplitRow],
) -> tuple[tuple[SplitRow, ...], tuple[SplitRow, ...], tuple[SplitRow, ...]]:
    """Apply the fixed normal ranking and preserve all official test records."""
    if not rows:
        raise V0_2BoundaryPreparationError("official split contains no pcb2 records")
    train_rows = [row for row in rows if row.split == "train"]
    test_rows = [row for row in rows if row.split == "test"]
    if any(row.label != "normal" for row in train_rows):
        raise V0_2BoundaryPreparationError("training partition contains a non-normal label")
    ranked_train = sorted(
        train_rows,
        key=lambda row: (_selection_digest(row.relative_path), row.relative_path),
    )
    if len(ranked_train) <= REFERENCE_COUNT:
        raise V0_2BoundaryPreparationError(
            "normal training partition cannot provide reference and calibration sets"
        )
    if not test_rows:
        raise V0_2BoundaryPreparationError("official split contains no final-test records")
    all_paths = [row.relative_path for row in rows]
    if len(all_paths) != len(set(all_paths)):
        raise V0_2BoundaryPreparationError("official pcb2 paths are not unique")
    return (
        tuple(ranked_train[:REFERENCE_COUNT]),
        tuple(ranked_train[REFERENCE_COUNT:]),
        tuple(sorted(test_rows, key=lambda row: row.relative_path)),
    )


def _safe_source_path(source_root: Path, relative_path: str) -> Path:
    pure_path = PurePosixPath(relative_path)
    if (
        pure_path.is_absolute()
        or pure_path.as_posix() != relative_path
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise V0_2BoundaryPreparationError("split contains an unsafe source path")
    path = source_root.joinpath(*pure_path.parts)
    if not path.is_file() or path.is_symlink():
        raise V0_2BoundaryPreparationError("split source must be a regular non-symlink file")
    return path


def verify_category_image_inventory(
    *,
    source_root: Path,
    rows: Sequence[SplitRow],
) -> None:
    """Require exact split-to-extraction image-path equality without decoding bytes."""
    expected = {row.relative_path for row in rows}
    image_root = source_root / CATEGORY / "Data" / "Images"
    if not image_root.is_dir() or image_root.is_symlink():
        raise V0_2BoundaryPreparationError("extracted pcb2 image root is invalid")
    observed: set[str] = set()
    for path in image_root.rglob("*"):
        if path.is_symlink():
            raise V0_2BoundaryPreparationError("extracted image tree contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise V0_2BoundaryPreparationError("extracted image tree contains a special file")
        observed.add(path.relative_to(source_root).as_posix())
    if observed != expected:
        raise V0_2BoundaryPreparationError(
            "official split and extracted pcb2 image paths are not identical"
        )


def _normal_partition_record(
    *,
    source_root: Path,
    row: SplitRow,
    partition: str,
    rank: int,
) -> NormalPartitionRecord:
    source_path = _safe_source_path(source_root, row.relative_path)
    return NormalPartitionRecord(
        byte_count=source_path.stat().st_size,
        partition=partition,
        relative_path=row.relative_path,
        selection_rank=rank,
        selection_sha256=_selection_digest(row.relative_path),
        sha256=sha256_file(source_path),
    )


def _write_jsonl_exclusive(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            stream.write("\n")


def write_normal_partition_manifests(
    *,
    source_root: Path,
    manifest_root: Path,
    reference_rows: Sequence[SplitRow],
    calibration_rows: Sequence[SplitRow],
) -> dict[str, Any]:
    """Write ignored normal-only manifests with byte identities."""
    if manifest_root.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_root}")
    if len(reference_rows) != REFERENCE_COUNT or not calibration_rows:
        raise V0_2BoundaryPreparationError("normal partition counts differ from the contract")
    all_rows = [*reference_rows, *calibration_rows]
    if len({row.relative_path for row in all_rows}) != len(all_rows):
        raise V0_2BoundaryPreparationError("normal partitions overlap")

    reference_records = tuple(
        _normal_partition_record(
            source_root=source_root,
            row=row,
            partition="reference",
            rank=rank,
        )
        for rank, row in enumerate(reference_rows, start=1)
    )
    calibration_records = tuple(
        _normal_partition_record(
            source_root=source_root,
            row=row,
            partition="calibration",
            rank=rank,
        )
        for rank, row in enumerate(calibration_rows, start=REFERENCE_COUNT + 1)
    )
    manifest_root.mkdir(parents=True)
    reference_path = manifest_root / "reference.jsonl"
    calibration_path = manifest_root / "calibration.jsonl"
    _write_jsonl_exclusive(
        reference_path,
        tuple(
            {
                "schema_version": NORMAL_MANIFEST_SCHEMA,
                **asdict(record),
            }
            for record in reference_records
        ),
    )
    _write_jsonl_exclusive(
        calibration_path,
        tuple(
            {
                "schema_version": NORMAL_MANIFEST_SCHEMA,
                **asdict(record),
            }
            for record in calibration_records
        ),
    )
    manifest_set = {
        "schema_version": NORMAL_MANIFEST_SET_SCHEMA,
        "category": CATEGORY,
        "selection": {
            "namespace": SELECTION_NAMESPACE,
            "procedure": "sha256_path_ranking_v1",
            "reference_count": REFERENCE_COUNT,
            "seed": SELECTION_SEED,
            "tie_breaker": "posix_relative_path_ascending",
        },
        "partitions": {
            "calibration": {
                "file": "calibration.jsonl",
                "record_count": len(calibration_records),
                "sha256": sha256_file(calibration_path),
            },
            "reference": {
                "file": "reference.jsonl",
                "record_count": len(reference_records),
                "sha256": sha256_file(reference_path),
            },
        },
        "labels": "known_normal_only",
        "image_content_decoded": False,
    }
    manifest_set_path = manifest_root / "manifest-set.json"
    write_json_atomic(manifest_set_path, manifest_set)
    return {
        "calibration_count": len(calibration_records),
        "calibration_manifest_path": calibration_path,
        "calibration_manifest_sha256": sha256_file(calibration_path),
        "manifest_set_path": manifest_set_path,
        "manifest_set_sha256": sha256_file(manifest_set_path),
        "reference_count": len(reference_records),
        "reference_manifest_path": reference_path,
        "reference_manifest_sha256": sha256_file(reference_path),
    }


def _write_ordering_key(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    key = os.urandom(32)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(key)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    if path.stat().st_size != 32 or path.stat().st_mode & 0o077:
        raise V0_2BoundaryPreparationError("ordering key permissions or size are invalid")
    return key


def prepare_partition_assets(
    *,
    source_root: Path,
    external_root: Path,
    rows: Sequence[SplitRow],
    ordering_key: bytes,
) -> dict[str, Any]:
    """Create normal manifests and opaque test assets from a verified source tree."""
    reference_rows, calibration_rows, final_test_rows = select_v0_2_partitions(rows)
    verify_category_image_inventory(source_root=source_root, rows=rows)
    normal = write_normal_partition_manifests(
        source_root=source_root,
        manifest_root=external_root / "normal-manifests",
        reference_rows=reference_rows,
        calibration_rows=calibration_rows,
    )
    opaque = prepare_opaque_boundary(
        source_root=source_root,
        scorer_root=external_root / "scorer",
        sealed_mapping_path=external_root / "sealed" / "mapping.json",
        records=tuple(
            BoundarySourceRecord(
                source_path=row.relative_path,
                class_label=row.label,
            )
            for row in final_test_rows
        ),
        ordering_key=ordering_key,
    )
    return {
        "calibration_count": normal["calibration_count"],
        "calibration_manifest_sha256": normal["calibration_manifest_sha256"],
        "final_test_count": opaque["asset_count"],
        "normal_manifest_set_sha256": normal["manifest_set_sha256"],
        "reference_count": normal["reference_count"],
        "reference_manifest_sha256": normal["reference_manifest_sha256"],
        "scoring_manifest_sha256": opaque["scoring_manifest_sha256"],
        "sealed_mapping_sha256": opaque["sealed_mapping_sha256"],
    }


def build_public_boundary_record(
    *,
    run_kind: str,
    archive_sha256: str,
    split_sha256: str,
    reference_count: int,
    calibration_count: int,
    final_test_count: int,
    scoring_manifest_sha256: str,
    sealed_mapping_sha256: str,
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the allowlisted public boundary record without protected values."""
    record = {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "run_id": RUN_ID,
        "run_kind": run_kind,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_document_sha256": PREREGISTRATION_DOCUMENT_SHA256,
        "archive_sha256": archive_sha256,
        "split_sha256": split_sha256,
        "reference_count": reference_count,
        "calibration_count": calibration_count,
        "final_test_count": final_test_count,
        "scoring_manifest_sha256": scoring_manifest_sha256,
        "sealed_mapping_sha256": sealed_mapping_sha256,
        "final_test_class_counts_published": False,
        "raw_data_in_git": False,
    }
    return validate_json_artifact(
        "boundary_record",
        record,
        config=config,
        schema=schema,
    )


def _require_fixed_input(path: Path, *, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise V0_2BoundaryPreparationError(f"{label} must be a regular non-symlink file")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise V0_2BoundaryPreparationError(f"{label} SHA-256 differs from the contract")
    return {"byte_count": path.stat().st_size, "sha256": observed_sha256}


def prepare_v0_2_boundary(
    *,
    project_root: Path,
    execution_commit: str,
    archive_path: Path,
    split_path: Path,
    external_root: Path,
    public_artifact_root: Path,
) -> PreparedBoundary:
    """Prepare the one fixed real-data boundary and publish aggregate identities."""
    project_root = project_root.resolve()
    archive_path = archive_path.resolve()
    split_path = split_path.resolve()
    external_root = external_root.resolve()
    public_artifact_root = public_artifact_root.resolve()
    if external_root.exists():
        raise FileExistsError(f"refusing to overwrite {external_root}")
    if public_artifact_root.exists():
        raise FileExistsError(f"refusing to overwrite {public_artifact_root}")
    expected_external_root = (project_root / "data/external/v0.2/evaluation" / RUN_ID).resolve()
    expected_public_root = (project_root / "artifacts/v0.2/evaluation" / RUN_ID).resolve()
    if external_root != expected_external_root:
        raise V0_2BoundaryPreparationError("external_root differs from the fixed ignored path")
    if public_artifact_root != expected_public_root:
        raise V0_2BoundaryPreparationError("public artifact root differs from the contract")

    execution = validate_boundary_execution_identity(
        project_root=project_root,
        execution_commit=execution_commit,
    )
    config_path = project_root / "configs/v0.2.yaml"
    schema_path = project_root / "schemas/v0.2/evaluation-artifacts.json"
    config = load_v0_2_config(config_path)
    schema = load_v0_2_artifact_schema(schema_path)
    archive_identity = _require_fixed_input(
        archive_path,
        expected_sha256=config["dataset"]["archive_sha256"],
        label="archive",
    )
    if archive_identity["byte_count"] != ARCHIVE_BYTE_COUNT:
        raise V0_2BoundaryPreparationError("archive byte count differs from the fixed record")
    split_identity = _require_fixed_input(
        split_path,
        expected_sha256=config["dataset"]["split_sha256"],
        label="official split",
    )
    rows = tuple(
        load_official_rows(
            split_path,
            expected_sha256=config["dataset"]["split_sha256"],
            category=CATEGORY,
        )
    )
    external_root.mkdir(parents=True)
    archive_record_path = external_root / "archive-identity.json"
    write_json_atomic(
        archive_record_path,
        {
            "sha256": archive_identity["sha256"],
            "byte_count": archive_identity["byte_count"],
            "verified_against_contract": True,
        },
    )
    source_root = external_root / "source"
    extraction_summary = extract_archive_safely(
        archive_path=archive_path,
        archive_provenance_path=archive_record_path,
        destination=source_root,
        extraction_provenance_path=external_root / "extraction.json",
        project_root=project_root,
        member_prefix=CATEGORY,
    )
    ordering_key_path = external_root / "sealed" / "ordering-key.bin"
    ordering_key = _write_ordering_key(ordering_key_path)
    prepared = prepare_partition_assets(
        source_root=source_root,
        external_root=external_root,
        rows=rows,
        ordering_key=ordering_key,
    )

    boundary_record = build_public_boundary_record(
        run_kind="final_test",
        archive_sha256=archive_identity["sha256"],
        split_sha256=split_identity["sha256"],
        reference_count=prepared["reference_count"],
        calibration_count=prepared["calibration_count"],
        final_test_count=prepared["final_test_count"],
        scoring_manifest_sha256=prepared["scoring_manifest_sha256"],
        sealed_mapping_sha256=prepared["sealed_mapping_sha256"],
        config=config,
        schema=schema,
    )
    state = {
        "schema_version": BOUNDARY_STATE_SCHEMA,
        "run_id": RUN_ID,
        "execution": execution,
        "contract": {
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "schema_sha256": EXPECTED_SCHEMA_SHA256,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "preregistration_document_sha256": PREREGISTRATION_DOCUMENT_SHA256,
        },
        "dataset": {
            "archive_byte_count": archive_identity["byte_count"],
            "archive_sha256": archive_identity["sha256"],
            "category": CATEGORY,
            "license": config["dataset"]["license"],
            "split_byte_count": split_identity["byte_count"],
            "split_sha256": split_identity["sha256"],
        },
        "extraction": {
            **asdict(extraction_summary),
            "all_archive_members_validated": True,
            "image_content_decoded": False,
        },
        "normal_partitions": {
            "reference_count": prepared["reference_count"],
            "reference_manifest_sha256": prepared["reference_manifest_sha256"],
            "calibration_count": prepared["calibration_count"],
            "calibration_manifest_sha256": prepared["calibration_manifest_sha256"],
            "manifest_set_sha256": prepared["normal_manifest_set_sha256"],
        },
        "opaque_final_test": {
            "asset_count": prepared["final_test_count"],
            "scoring_manifest_sha256": prepared["scoring_manifest_sha256"],
            "sealed_mapping_sha256": prepared["sealed_mapping_sha256"],
            "ordering_key_sha256": sha256_file(ordering_key_path),
            "class_counts_published": False,
        },
        "boundary": {
            "anomaly_score_computed": False,
            "final_test_label_revealed": False,
            "image_content_decoded": False,
            "image_content_displayed": False,
            "method_fit_performed": False,
            "raw_data_in_git": False,
            "threshold_calibrated": False,
        },
    }
    write_json_atomic(external_root / "boundary-state.json", state)

    public_boundary_dir = public_artifact_root / "boundary"
    public_boundary_dir.mkdir(parents=True)
    public_boundary_path = public_boundary_dir / "boundary-record.json"
    write_json_atomic(public_boundary_path, boundary_record)
    return PreparedBoundary(
        calibration_count=prepared["calibration_count"],
        calibration_manifest_sha256=prepared["calibration_manifest_sha256"],
        final_test_count=prepared["final_test_count"],
        normal_manifest_set_sha256=prepared["normal_manifest_set_sha256"],
        public_boundary_record_sha256=sha256_file(public_boundary_path),
        reference_count=prepared["reference_count"],
        reference_manifest_sha256=prepared["reference_manifest_sha256"],
        scoring_manifest_sha256=prepared["scoring_manifest_sha256"],
        sealed_mapping_sha256=prepared["sealed_mapping_sha256"],
    )
