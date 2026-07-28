"""Tar extraction with explicit member validation and no image decoding."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from few_shot_anomaly_poc.errors import ChecksumMismatchError, UnsafeArchiveError
from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.jsonio import write_json_atomic


@dataclass(frozen=True)
class ExtractionLimits:
    max_members: int = 100_000
    max_file_bytes: int = 2_000_000_000
    max_total_bytes: int = 50_000_000_000


@dataclass(frozen=True)
class ExtractionSummary:
    member_count: int
    file_count: int
    directory_count: int
    total_file_bytes: int


DEFAULT_EXTRACTION_LIMITS = ExtractionLimits()


def _validated_prefix(value: str) -> tuple[str, ...]:
    if not value or "\\" in value or "\x00" in value:
        raise UnsafeArchiveError(f"unsafe archive member prefix: {value!r}")
    parts = tuple(value.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise UnsafeArchiveError(f"unsafe archive member prefix: {value!r}")
    return parts


def _validated_parts(member: tarfile.TarInfo) -> tuple[str, ...]:
    name = member.name
    if not name or "\\" in name or "\x00" in name:
        raise UnsafeArchiveError(f"unsafe archive member name: {name!r}")

    pure_path = PurePosixPath(name)
    raw_parts = tuple(name.split("/"))
    if (
        not pure_path.parts
        or pure_path.is_absolute()
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise UnsafeArchiveError(f"unsafe archive member path: {name!r}")
    if not (member.isdir() or member.isreg()):
        raise UnsafeArchiveError(f"unsupported archive member type: {name!r}")
    return pure_path.parts


def inspect_archive(
    archive_path: Path,
    *,
    member_prefix: str | None = None,
    limits: ExtractionLimits = DEFAULT_EXTRACTION_LIMITS,
) -> tuple[list[tuple[tarfile.TarInfo, tuple[str, ...]]], ExtractionSummary]:
    """Validate every member and select only the requested subtree."""
    validated: list[tuple[tarfile.TarInfo, tuple[str, ...]]] = []
    targets: set[tuple[str, ...]] = set()
    prefix_parts = _validated_prefix(member_prefix) if member_prefix is not None else None
    file_count = 0
    directory_count = 0
    total_file_bytes = 0
    selected_file_bytes = 0

    with tarfile.open(archive_path, mode="r:*") as archive:
        for index, member in enumerate(archive, start=1):
            if index > limits.max_members:
                raise UnsafeArchiveError(f"archive exceeds {limits.max_members} members")
            parts = _validated_parts(member)
            if parts in targets:
                raise UnsafeArchiveError(f"duplicate archive target: {member.name!r}")
            targets.add(parts)

            if member.isreg():
                if member.size < 0 or member.size > limits.max_file_bytes:
                    raise UnsafeArchiveError(f"archive member is too large: {member.name!r}")
                total_file_bytes += member.size
                if total_file_bytes > limits.max_total_bytes:
                    raise UnsafeArchiveError(
                        f"archive expands beyond {limits.max_total_bytes} bytes"
                    )
            if prefix_parts is not None and parts[: len(prefix_parts)] != prefix_parts:
                continue
            validated.append((member, parts))
            if member.isreg():
                file_count += 1
                selected_file_bytes += member.size
            else:
                directory_count += 1

    if not validated:
        raise UnsafeArchiveError(f"archive contains no members under {member_prefix!r}")

    return validated, ExtractionSummary(
        member_count=len(validated),
        file_count=file_count,
        directory_count=directory_count,
        total_file_bytes=selected_file_bytes,
    )


def _load_recorded_archive_sha256(provenance_path: Path) -> str:
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        recorded_sha256 = provenance["sha256"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ChecksumMismatchError(f"invalid archive provenance: {error}") from error
    if not isinstance(recorded_sha256, str) or len(recorded_sha256) != 64:
        raise ChecksumMismatchError("archive provenance has no valid SHA-256")
    return recorded_sha256


def extract_archive_safely(
    *,
    archive_path: Path,
    archive_provenance_path: Path,
    destination: Path,
    extraction_provenance_path: Path,
    project_root: Path,
    member_prefix: str | None = None,
    limits: ExtractionLimits = DEFAULT_EXTRACTION_LIMITS,
) -> ExtractionSummary:
    """Verify the archive, validate all members, then extract to a fresh directory."""
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    if extraction_provenance_path.exists():
        raise FileExistsError(f"refusing to overwrite {extraction_provenance_path}")

    recorded_sha256 = _load_recorded_archive_sha256(archive_provenance_path)
    observed_sha256 = sha256_file(archive_path)
    if observed_sha256 != recorded_sha256:
        raise ChecksumMismatchError(
            "archive does not match its provenance: "
            f"recorded {recorded_sha256}, observed {observed_sha256}"
        )

    validated, summary = inspect_archive(
        archive_path,
        member_prefix=member_prefix,
        limits=limits,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.extracting.",
        )
    )

    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            for member, parts in validated:
                target = staging.joinpath(*parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise UnsafeArchiveError(f"cannot read archive member: {member.name!r}")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)

        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    try:
        relative_destination = destination.relative_to(project_root).as_posix()
    except ValueError:
        relative_destination = destination.as_posix()
    write_json_atomic(
        extraction_provenance_path,
        {
            "schema_version": 1,
            "asset_type": "safe_extraction",
            "archive_sha256": observed_sha256,
            "extracted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "destination": relative_destination,
            "member_prefix": member_prefix,
            "all_archive_members_validated": True,
            "limits": asdict(limits),
            "summary": asdict(summary),
            "content_decoding": False,
        },
    )
    return summary
