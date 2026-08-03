"""Prepare a label-free opaque scoring boundary without decoding images."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.jsonio import write_json_atomic

SCORING_MANIFEST_SCHEMA = "v0.2-opaque-scoring-manifest-v1"
SEALED_MAPPING_SCHEMA = "v0.2-sealed-label-mapping-v1"
SCORING_RECORD_FIELDS = frozenset(
    {"asset_id", "byte_count", "relative_path", "sha256"}
)
SCORING_MANIFEST_FIELDS = frozenset({"records", "schema_version"})
SEALED_RECORD_FIELDS = frozenset({"asset_id", "class_label", "source_path"})
SEALED_MANIFEST_FIELDS = frozenset({"records", "schema_version"})
_ASSET_ID_PATTERN = re.compile(r"asset-[0-9]{6}")
_EXTENSION_PATTERN = re.compile(r"\.[A-Za-z0-9]{1,8}")


class OpaqueBoundaryError(Exception):
    """Reject input or output that would weaken the opaque boundary."""


@dataclass(frozen=True)
class BoundarySourceRecord:
    """Describe one source file and its protected final-test label."""

    source_path: str
    class_label: str


def _require_safe_relative_path(value: str, *, field: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise OpaqueBoundaryError(f"{field} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise OpaqueBoundaryError(f"{field} must not escape its declared root")
    if path.as_posix() != value:
        raise OpaqueBoundaryError(f"{field} is not canonical POSIX syntax")
    return path


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OpaqueBoundaryError(f"{field} must be a lowercase SHA-256")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OpaqueBoundaryError(f"cannot read JSON from {path.name}") from error
    if not isinstance(value, dict):
        raise OpaqueBoundaryError(f"{path.name} must contain a JSON object")
    return value


def _opaque_order(
    records: tuple[BoundarySourceRecord, ...], *, ordering_key: bytes
) -> list[BoundarySourceRecord]:
    if len(ordering_key) < 32:
        raise OpaqueBoundaryError("ordering_key must contain at least 32 bytes")
    seen_paths: set[str] = set()
    ranked: list[tuple[bytes, str, BoundarySourceRecord]] = []
    for record in records:
        _require_safe_relative_path(record.source_path, field="source_path")
        if not record.class_label:
            raise OpaqueBoundaryError("class_label must be a non-empty string")
        if record.source_path in seen_paths:
            raise OpaqueBoundaryError("source_path values must be unique")
        seen_paths.add(record.source_path)
        digest = hmac.digest(ordering_key, record.source_path.encode("utf-8"), "sha256")
        ranked.append((digest, record.source_path, record))
    return [record for _, _, record in sorted(ranked)]


def validate_scoring_manifest(value: object) -> dict[str, Any]:
    """Require the exact label-free scorer allowlist and opaque ordering."""
    if not isinstance(value, dict) or set(value) != SCORING_MANIFEST_FIELDS:
        raise OpaqueBoundaryError("scoring manifest fields differ from the allowlist")
    if value.get("schema_version") != SCORING_MANIFEST_SCHEMA:
        raise OpaqueBoundaryError("scoring manifest schema is invalid")
    records = value.get("records")
    if not isinstance(records, list) or not records:
        raise OpaqueBoundaryError("scoring manifest records must be a non-empty list")

    seen_paths: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != SCORING_RECORD_FIELDS:
            raise OpaqueBoundaryError("scoring record fields differ from the allowlist")
        expected_id = f"asset-{index:06d}"
        if record.get("asset_id") != expected_id:
            raise OpaqueBoundaryError("asset IDs must be opaque and sequential")
        relative_path = record.get("relative_path")
        if not isinstance(relative_path, str):
            raise OpaqueBoundaryError("relative_path must be a string")
        path = _require_safe_relative_path(relative_path, field="relative_path")
        if path.parent.as_posix() != "assets" or path.stem != expected_id:
            raise OpaqueBoundaryError("staged filenames must contain only the opaque ID")
        if not _EXTENSION_PATTERN.fullmatch(path.suffix):
            raise OpaqueBoundaryError("staged file extension is invalid")
        if relative_path in seen_paths:
            raise OpaqueBoundaryError("relative_path values must be unique")
        seen_paths.add(relative_path)
        byte_count = record.get("byte_count")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
            raise OpaqueBoundaryError("byte_count must be a non-negative integer")
        _require_sha256(record.get("sha256"), field="sha256")
    return value


def validate_sealed_mapping(value: object) -> dict[str, Any]:
    """Validate the reveal-side mapping without exposing it to the scorer."""
    if not isinstance(value, dict) or set(value) != SEALED_MANIFEST_FIELDS:
        raise OpaqueBoundaryError("sealed manifest fields are invalid")
    if value.get("schema_version") != SEALED_MAPPING_SCHEMA:
        raise OpaqueBoundaryError("sealed manifest schema is invalid")
    records = value.get("records")
    if not isinstance(records, list) or not records:
        raise OpaqueBoundaryError("sealed records must be a non-empty list")
    seen_sources: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != SEALED_RECORD_FIELDS:
            raise OpaqueBoundaryError("sealed record fields are invalid")
        if record.get("asset_id") != f"asset-{index:06d}":
            raise OpaqueBoundaryError("sealed asset IDs must match scoring order")
        source_path = record.get("source_path")
        class_label = record.get("class_label")
        if not isinstance(source_path, str):
            raise OpaqueBoundaryError("source_path must be a string")
        _require_safe_relative_path(source_path, field="source_path")
        if source_path in seen_sources:
            raise OpaqueBoundaryError("sealed source paths must be unique")
        seen_sources.add(source_path)
        if not isinstance(class_label, str) or not class_label:
            raise OpaqueBoundaryError("class_label must be a non-empty string")
    return value


def load_scoring_manifest(path: Path) -> dict[str, Any]:
    """Load only the public scorer input contract."""
    return validate_scoring_manifest(_read_json(path))


def load_sealed_mapping(path: Path) -> dict[str, Any]:
    """Load the protected reveal-side mapping."""
    return validate_sealed_mapping(_read_json(path))


def prepare_opaque_boundary(
    *,
    source_root: Path,
    scorer_root: Path,
    sealed_mapping_path: Path,
    records: tuple[BoundarySourceRecord, ...],
    ordering_key: bytes,
) -> dict[str, Any]:
    """Copy files under opaque IDs and separate scorer and reveal metadata."""
    source_root = source_root.resolve()
    scorer_root = scorer_root.resolve()
    sealed_mapping_path = sealed_mapping_path.resolve()
    if not source_root.is_dir():
        raise OpaqueBoundaryError("source_root must be an existing directory")
    if not records:
        raise OpaqueBoundaryError("records must not be empty")
    if scorer_root.exists():
        raise FileExistsError(f"refusing to overwrite {scorer_root}")
    if sealed_mapping_path.exists():
        raise FileExistsError(f"refusing to overwrite {sealed_mapping_path}")
    if sealed_mapping_path.is_relative_to(scorer_root):
        raise OpaqueBoundaryError("sealed mapping must be outside the scorer root")

    ordered = _opaque_order(records, ordering_key=ordering_key)
    prepared: list[tuple[Path, str, str, int, str, str]] = []
    for index, record in enumerate(ordered):
        relative_source = _require_safe_relative_path(
            record.source_path, field="source_path"
        )
        source_path = source_root.joinpath(*relative_source.parts)
        if not source_path.is_file() or source_path.is_symlink():
            raise OpaqueBoundaryError("every source record must name a regular file")
        extension = source_path.suffix.lower()
        if not _EXTENSION_PATTERN.fullmatch(extension):
            raise OpaqueBoundaryError("source extension cannot be retained safely")
        asset_id = f"asset-{index:06d}"
        relative_staged = f"assets/{asset_id}{extension}"
        prepared.append(
            (
                source_path,
                record.source_path,
                record.class_label,
                source_path.stat().st_size,
                sha256_file(source_path),
                relative_staged,
            )
        )

    scorer_root.mkdir(parents=True)
    scoring_records: list[dict[str, Any]] = []
    sealed_records: list[dict[str, str]] = []
    for index, (source, source_name, label, byte_count, digest, staged_name) in enumerate(
        prepared
    ):
        destination = scorer_root.joinpath(*PurePosixPath(staged_name).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream)
        if destination.stat().st_size != byte_count or sha256_file(destination) != digest:
            raise OpaqueBoundaryError("staged file identity differs from its source")
        asset_id = f"asset-{index:06d}"
        scoring_records.append(
            {
                "asset_id": asset_id,
                "byte_count": byte_count,
                "relative_path": staged_name,
                "sha256": digest,
            }
        )
        sealed_records.append(
            {
                "asset_id": asset_id,
                "class_label": label,
                "source_path": source_name,
            }
        )

    scoring_manifest = validate_scoring_manifest(
        {"records": scoring_records, "schema_version": SCORING_MANIFEST_SCHEMA}
    )
    sealed_mapping = validate_sealed_mapping(
        {"records": sealed_records, "schema_version": SEALED_MAPPING_SCHEMA}
    )
    scoring_manifest_path = scorer_root / "scoring-manifest.json"
    write_json_atomic(scoring_manifest_path, scoring_manifest)
    write_json_atomic(sealed_mapping_path, sealed_mapping)
    return {
        "asset_count": len(scoring_records),
        "scoring_manifest_path": scoring_manifest_path,
        "scoring_manifest_sha256": sha256_file(scoring_manifest_path),
        "sealed_mapping_path": sealed_mapping_path,
        "sealed_mapping_sha256": sha256_file(sealed_mapping_path),
    }


def run_synthetic_boundary_feasibility(work_root: Path) -> dict[str, Any]:
    """Exercise the boundary with synthetic bytes and return no protected values."""
    work_root = work_root.resolve()
    if work_root.exists():
        raise FileExistsError(f"refusing to overwrite {work_root}")
    source_root = work_root / "source"
    source_files = {
        "group-a/visual-one.png": b"synthetic-png-a\x00\x01",
        "group-b/visual-two.jpg": b"synthetic-jpg-b\x02\x03",
        "group-c/visual-three.png": b"synthetic-png-c\x04\x05",
    }
    labels = ("fixture-class-a", "fixture-class-b", "fixture-class-c")
    for relative_path, content in source_files.items():
        target = source_root.joinpath(*PurePosixPath(relative_path).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    records = tuple(
        BoundarySourceRecord(source_path=source_path, class_label=label)
        for source_path, label in zip(source_files, labels, strict=True)
    )
    ordering_key = hashlib.sha256(b"synthetic opaque boundary feasibility only").digest()

    first = prepare_opaque_boundary(
        source_root=source_root,
        scorer_root=work_root / "scorer-a",
        sealed_mapping_path=work_root / "sealed-a/mapping.json",
        records=records,
        ordering_key=ordering_key,
    )
    second = prepare_opaque_boundary(
        source_root=source_root,
        scorer_root=work_root / "scorer-b",
        sealed_mapping_path=work_root / "sealed-b/mapping.json",
        records=records,
        ordering_key=ordering_key,
    )
    scoring_manifest = load_scoring_manifest(first["scoring_manifest_path"])
    sealed_mapping = load_sealed_mapping(first["sealed_mapping_path"])

    overwrite_refused = False
    try:
        prepare_opaque_boundary(
            source_root=source_root,
            scorer_root=work_root / "scorer-a",
            sealed_mapping_path=work_root / "sealed-a/mapping.json",
            records=records,
            ordering_key=ordering_key,
        )
    except FileExistsError:
        overwrite_refused = True

    scorer_files = sorted(
        path.relative_to(work_root / "scorer-a").as_posix()
        for path in (work_root / "scorer-a").rglob("*")
        if path.is_file()
    )
    expected_scorer_files = sorted(
        ["scoring-manifest.json"]
        + [record["relative_path"] for record in scoring_manifest["records"]]
    )
    scoring_text = first["scoring_manifest_path"].read_text(encoding="utf-8")
    scorer_bytes = b"".join(
        path.read_bytes()
        for path in (work_root / "scorer-a").rglob("*")
        if path.is_file()
    )
    source_names_absent = all(name not in scoring_text for name in source_files)
    protected_labels_absent = all(label not in scoring_text for label in labels)
    key_absent = ordering_key not in scorer_bytes and ordering_key.hex() not in scoring_text
    copied_identities_match = all(
        (work_root / "scorer-a" / record["relative_path"]).stat().st_size
        == record["byte_count"]
        and sha256_file(work_root / "scorer-a" / record["relative_path"])
        == record["sha256"]
        for record in scoring_manifest["records"]
    )
    sealed_id_set_matches = [record["asset_id"] for record in sealed_mapping["records"]] == [
        record["asset_id"] for record in scoring_manifest["records"]
    ]
    checks = {
        "copied_byte_and_hash_identities_match": copied_identities_match,
        "deterministic_hmac_ordering": (
            first["scoring_manifest_sha256"] == second["scoring_manifest_sha256"]
            and first["sealed_mapping_sha256"] == second["sealed_mapping_sha256"]
        ),
        "non_overwrite_enforced": overwrite_refused,
        "opaque_sequential_filenames": all(
            _ASSET_ID_PATTERN.fullmatch(Path(record["relative_path"]).stem)
            for record in scoring_manifest["records"]
        ),
        "protected_labels_absent_from_scorer_manifest": protected_labels_absent,
        "scorer_root_contains_only_allowlisted_files": scorer_files
        == expected_scorer_files,
        "scoring_manifest_exact_field_allowlist": all(
            set(record) == SCORING_RECORD_FIELDS
            for record in scoring_manifest["records"]
        ),
        "sealed_and_scoring_asset_ids_match": sealed_id_set_matches,
        "sealed_mapping_outside_scorer_root": not first[
            "sealed_mapping_path"
        ].is_relative_to(work_root / "scorer-a"),
        "secret_key_absent_from_scorer_view": key_absent,
        "semantic_source_paths_absent_from_scorer_manifest": source_names_absent,
    }
    return {
        "boundary": {
            "boundary_prepared": False,
            "dataset_access": False,
            "dataset_labels_accessed": False,
            "image_decode_performed": False,
            "network_access": False,
            "official_split_access": False,
            "scoring_performed": False,
            "synthetic_fixture_only": True,
        },
        "checks": checks,
        "decision": {
            "status": "pass" if all(checks.values()) else "fail",
        },
        "fixture": {
            "asset_count": len(records),
            "protected_values_published": False,
            "raw_fixture_bytes_published": False,
        },
        "identities": {
            "scoring_manifest_sha256": first["scoring_manifest_sha256"],
            "sealed_mapping_sha256": first["sealed_mapping_sha256"],
        },
        "schema_version": "v0.2-opaque-boundary-feasibility-v1",
    }
