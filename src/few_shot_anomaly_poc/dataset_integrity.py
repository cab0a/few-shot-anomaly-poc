"""Verify one local VisA category without decoding or scoring images."""

from __future__ import annotations

import csv
import hashlib
import json
import tarfile
from pathlib import Path
from typing import BinaryIO

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.manifests import (
    load_official_rows,
    normalize_relative_path,
)
from few_shot_anomaly_poc.safe_tar import inspect_archive

HASH_CHUNK_SIZE = 1024 * 1024


class DatasetIntegrityError(Exception):
    """Reject a local dataset that differs from the fixed public record."""


def _read_dataset_record(path: Path, *, category: str) -> dict:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetIntegrityError("cannot read the fixed dataset record") from error
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != 1
        or record.get("dataset", {}).get("category") != category
        or record.get("dataset", {}).get("license") != "CC BY 4.0"
        or record.get("archive", {}).get("checksum_status") != "observed_only"
    ):
        raise DatasetIntegrityError("the fixed dataset record is invalid")
    return record


def _require_nonnegative_integer(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DatasetIntegrityError(f"invalid fixed integer: {field}")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DatasetIntegrityError(f"invalid fixed SHA-256: {field}")
    return value


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while block := stream.read(HASH_CHUNK_SIZE):
        digest.update(block)
    return digest.hexdigest()


def _split_inventory(
    split_csv: Path,
    *,
    split_sha256: str,
    category: str,
) -> tuple[set[str], set[str], dict[str, int]]:
    rows = load_official_rows(
        split_csv,
        expected_sha256=split_sha256,
        category=category,
    )
    expected_images = {row.relative_path for row in rows}
    expected_masks: set[str] = set()

    try:
        with split_csv.open("r", encoding="utf-8-sig", newline="") as stream:
            raw_rows = tuple(csv.DictReader(stream))
    except (OSError, csv.Error) as error:
        raise DatasetIntegrityError("cannot read the pinned official split") from error

    relevant_raw_rows = tuple(row for row in raw_rows if row.get("object") == category)
    if len(relevant_raw_rows) != len(rows):
        raise DatasetIntegrityError("official split row reconstruction failed")
    for raw in relevant_raw_rows:
        image_path = normalize_relative_path(raw.get("image", ""))
        mask_value = raw.get("mask", "")
        if raw.get("label") == "anomaly":
            mask_path = normalize_relative_path(mask_value)
            if mask_path in expected_masks:
                raise DatasetIntegrityError(f"duplicate official mask path: {mask_path}")
            expected_masks.add(mask_path)
        elif mask_value:
            raise DatasetIntegrityError(f"normal row unexpectedly has a mask: {image_path}")

    counts = {
        "official_train_normal": sum(
            row.split == "train" and row.label == "normal" for row in rows
        ),
        "official_test_normal": sum(
            row.split == "test" and row.label == "normal" for row in rows
        ),
        "official_test_anomaly": sum(
            row.split == "test" and row.label == "anomaly" for row in rows
        ),
    }
    return expected_images, expected_masks, counts


def _category_files(dataset_root: Path, *, category: str) -> dict[str, Path]:
    category_root = dataset_root / category
    if not category_root.is_dir() or category_root.is_symlink():
        raise DatasetIntegrityError("local category root must be a real directory")

    files: dict[str, Path] = {}
    for path in category_root.rglob("*"):
        if path.is_symlink():
            raise DatasetIntegrityError(f"symlink found in local category: {path.name}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise DatasetIntegrityError(f"unsupported local filesystem entry: {path.name}")
        relative_path = path.relative_to(dataset_root).as_posix()
        if relative_path in files:
            raise DatasetIntegrityError(f"duplicate local path: {relative_path}")
        files[relative_path] = path
    return files


def _compare_archive_and_extraction(
    *,
    archive_path: Path,
    dataset_root: Path,
    category: str,
) -> tuple[dict[str, object], set[str]]:
    try:
        selected_members, selected_summary = inspect_archive(
            archive_path,
            member_prefix=category,
        )
    except (OSError, tarfile.TarError) as error:
        raise DatasetIntegrityError("cannot inspect the local archive") from error

    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            archive_member_count = sum(1 for _ in archive)
    except (OSError, tarfile.TarError) as error:
        raise DatasetIntegrityError("cannot count archive members") from error

    regular_members = sorted(
        (member for member, _ in selected_members if member.isreg()),
        key=lambda member: member.name,
    )
    expected_paths = {member.name for member in regular_members}
    actual_files = _category_files(dataset_root, category=category)
    actual_paths = set(actual_files)
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    if missing or extra:
        raise DatasetIntegrityError(
            f"archive/extraction path mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    tree_digest = hashlib.sha256()
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            for member in regular_members:
                extracted_path = actual_files[member.name]
                if extracted_path.stat().st_size != member.size:
                    raise DatasetIntegrityError(
                        f"archive/extraction size mismatch: {member.name}"
                    )
                archive_stream = archive.extractfile(member)
                if archive_stream is None:
                    raise DatasetIntegrityError(
                        f"cannot read regular archive member: {member.name}"
                    )
                with archive_stream:
                    archive_sha256 = _sha256_stream(archive_stream)
                extracted_sha256 = sha256_file(extracted_path)
                if archive_sha256 != extracted_sha256:
                    raise DatasetIntegrityError(
                        f"archive/extraction content mismatch: {member.name}"
                    )
                tree_digest.update(
                    f"{member.name}\t{member.size}\t{extracted_sha256}\n".encode()
                )
    except (OSError, tarfile.TarError) as error:
        raise DatasetIntegrityError("cannot compare archive and extracted files") from error

    return (
        {
            "whole_archive_member_count": archive_member_count,
            "whole_archive_member_validation_passed": True,
            "selected_member_count": selected_summary.member_count,
            "selected_file_count": selected_summary.file_count,
            "selected_directory_count": selected_summary.directory_count,
            "selected_file_bytes": selected_summary.total_file_bytes,
            "extracted_file_count": len(actual_files),
            "byte_for_byte_files_compared": len(regular_members),
            "missing_extracted_files": 0,
            "extra_extracted_files": 0,
            "content_tree_sha256": tree_digest.hexdigest(),
        },
        actual_paths,
    )


def verify_visa_pcb1_integrity(
    *,
    archive_path: Path,
    dataset_root: Path,
    split_csv: Path,
    dataset_record_path: Path,
    category: str = "pcb1",
) -> dict:
    """Return a deterministic aggregate record after strict local verification."""
    record = _read_dataset_record(dataset_record_path, category=category)
    archive_fixed = record["archive"]
    split_fixed = record["official_split"]
    structure_fixed = record["pcb1_structure"]

    expected_archive_sha256 = _require_sha256(
        archive_fixed.get("observed_sha256"),
        field="archive.observed_sha256",
    )
    expected_archive_bytes = _require_nonnegative_integer(
        archive_fixed.get("content_length_bytes"),
        field="archive.content_length_bytes",
    )
    expected_tar_entries = _require_nonnegative_integer(
        archive_fixed.get("tar_entry_count"),
        field="archive.tar_entry_count",
    )
    expected_split_sha256 = _require_sha256(
        split_fixed.get("sha256"),
        field="official_split.sha256",
    )
    expected_annotation_sha256 = _require_sha256(
        structure_fixed.get("image_annotation_sha256"),
        field="pcb1_structure.image_annotation_sha256",
    )

    if not archive_path.is_file() or archive_path.is_symlink():
        raise DatasetIntegrityError("local archive must be a real file")
    observed_archive_bytes = archive_path.stat().st_size
    if observed_archive_bytes != expected_archive_bytes:
        raise DatasetIntegrityError("local archive byte count differs from the fixed record")
    observed_archive_sha256 = sha256_file(archive_path)
    if observed_archive_sha256 != expected_archive_sha256:
        raise DatasetIntegrityError("local archive SHA-256 differs from the fixed observation")
    observed_split_sha256 = sha256_file(split_csv)
    if observed_split_sha256 != expected_split_sha256:
        raise DatasetIntegrityError("local split SHA-256 differs from the pinned value")

    expected_images, expected_masks, split_counts = _split_inventory(
        split_csv,
        split_sha256=expected_split_sha256,
        category=category,
    )
    archive_comparison, actual_paths = _compare_archive_and_extraction(
        archive_path=archive_path,
        dataset_root=dataset_root,
        category=category,
    )
    if archive_comparison["whole_archive_member_count"] != expected_tar_entries:
        raise DatasetIntegrityError("whole archive member count differs from the fixed record")

    actual_images = {
        path for path in actual_paths if path.startswith(f"{category}/Data/Images/")
    }
    actual_masks = {
        path for path in actual_paths if path.startswith(f"{category}/Data/Masks/")
    }
    if actual_images != expected_images:
        raise DatasetIntegrityError(
            "official split and extracted image path sets are not identical"
        )
    if actual_masks != expected_masks:
        raise DatasetIntegrityError(
            "official split and extracted mask path sets are not identical"
        )

    normal_image_count = sum("/Images/Normal/" in path for path in actual_images)
    anomaly_image_count = sum("/Images/Anomaly/" in path for path in actual_images)
    expected_counts = {
        "normal_images": normal_image_count,
        "anomaly_images": anomaly_image_count,
        "anomaly_masks": len(actual_masks),
        **split_counts,
    }
    for field, observed in expected_counts.items():
        expected = _require_nonnegative_integer(
            structure_fixed.get(field),
            field=f"pcb1_structure.{field}",
        )
        if observed != expected:
            raise DatasetIntegrityError(
                f"local count differs from the fixed record: {field}"
            )

    annotation_path = dataset_root / category / "image_anno.csv"
    observed_annotation_sha256 = sha256_file(annotation_path)
    if observed_annotation_sha256 != expected_annotation_sha256:
        raise DatasetIntegrityError("image_anno.csv differs from the fixed record")

    return {
        "schema_version": 1,
        "record_type": "visa_pcb1_local_integrity",
        "status": "PASS",
        "dataset": {
            "name": record["dataset"]["name"],
            "category": category,
            "license": "CC BY 4.0",
            "raw_data_committed_to_git": False,
        },
        "archive": {
            "identifier": archive_fixed["identifier"],
            "byte_count": observed_archive_bytes,
            "observed_sha256": observed_archive_sha256,
            "matches_fixed_prior_observation": True,
            "independently_published_upstream_sha256_available": False,
        },
        "official_split": {
            "revision": split_fixed["revision"],
            "path": split_fixed["path"],
            "sha256": observed_split_sha256,
            "matches_pinned_sha256": True,
        },
        "archive_extraction_comparison": archive_comparison,
        "pcb1": {
            **expected_counts,
            "image_annotation_sha256": observed_annotation_sha256,
            "missing_split_images": 0,
            "extra_images_outside_split": 0,
            "missing_anomaly_masks": 0,
            "extra_masks_outside_split": 0,
        },
        "evaluation_boundary": {
            "image_content_decoded": False,
            "image_content_displayed": False,
            "anomaly_score_computed": False,
            "per_path_final_test_label_exported": False,
            "final_test_label_join_performed": False,
            "metric_computed": False,
            "threshold_changed": False,
        },
    }
