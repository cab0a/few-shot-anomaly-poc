"""Create and validate metadata-only dataset manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.errors import ChecksumMismatchError, ManifestIntegrityError
from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.jsonio import write_json_atomic

SPLIT_COLUMNS = {"object", "split", "label", "image", "mask"}
MANIFEST_NAMES = {
    "reference": "reference.jsonl",
    "calibration": "calibration.jsonl",
    "final-test": "final-test.jsonl",
}
BASE_RECORD_KEYS = {
    "id",
    "schema_version",
    "partition",
    "category",
    "relative_path",
    "source_split",
    "source_row",
}


@dataclass(frozen=True)
class SplitRow:
    source_row: int
    split: str
    label: str
    relative_path: str


@dataclass(frozen=True)
class ManifestSummary:
    reference_count: int
    calibration_count: int
    final_test_count: int
    archive_sha256_recorded: bool


def normalize_relative_path(value: str) -> str:
    """Normalize a split path without touching the referenced file."""
    if not value or "\\" in value or "\x00" in value or ":" in value:
        raise ManifestIntegrityError(f"invalid relative path: {value!r}")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ManifestIntegrityError(f"unsafe relative path: {value!r}")
    return candidate.as_posix()


def load_official_rows(
    split_csv: Path,
    *,
    expected_sha256: str,
    category: str,
) -> list[SplitRow]:
    """Load only official split metadata; no dataset file is opened or inspected."""
    observed_sha256 = sha256_file(split_csv)
    if observed_sha256 != expected_sha256:
        raise ChecksumMismatchError(
            f"official split checksum mismatch: expected {expected_sha256}, "
            f"observed {observed_sha256}"
        )

    rows: list[SplitRow] = []
    seen_paths: set[str] = set()
    with split_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not SPLIT_COLUMNS.issubset(reader.fieldnames):
            raise ManifestIntegrityError(
                f"official split must contain columns {sorted(SPLIT_COLUMNS)}"
            )
        for source_row, raw in enumerate(reader, start=2):
            if raw["object"] != category:
                continue
            relative_path = normalize_relative_path(raw["image"])
            if not relative_path.startswith(f"{category}/"):
                raise ManifestIntegrityError(
                    f"row {source_row} path is outside category {category!r}"
                )
            if relative_path in seen_paths:
                raise ManifestIntegrityError(f"duplicate split path: {relative_path}")
            seen_paths.add(relative_path)

            split = raw["split"]
            label = raw["label"]
            if split == "train" and label != "normal":
                raise ManifestIntegrityError(
                    f"row {source_row} assigns a non-normal label to training"
                )
            if split == "test" and label not in {"normal", "anomaly"}:
                raise ManifestIntegrityError(f"row {source_row} has an invalid test label")
            if split not in {"train", "test"}:
                raise ManifestIntegrityError(f"row {source_row} has an invalid split")
            rows.append(
                SplitRow(
                    source_row=source_row,
                    split=split,
                    label=label,
                    relative_path=relative_path,
                )
            )

    if not rows:
        raise ManifestIntegrityError(f"official split has no rows for {category!r}")
    return rows


def _selection_digest(namespace: str, seed: int, relative_path: str) -> str:
    value = f"{namespace}:{seed}:{relative_path}".encode()
    return hashlib.sha256(value).hexdigest()


def _record_id(category: str, relative_path: str) -> str:
    suffix = hashlib.sha256(relative_path.encode()).hexdigest()[:16]
    return f"visa-{category}-{suffix}"


def _base_record(
    *,
    config: ProjectConfig,
    row: SplitRow,
    partition: str,
    include_label: bool = True,
) -> dict[str, Any]:
    record = {
        "schema_version": 1,
        "id": _record_id(config.category, row.relative_path),
        "partition": partition,
        "category": config.category,
        "relative_path": row.relative_path,
        "source_split": row.split,
        "source_row": row.source_row,
    }
    if include_label:
        record["label"] = row.label
    return record


def partition_rows(
    rows: Iterable[SplitRow],
    *,
    config: ProjectConfig,
) -> dict[str, list[dict[str, Any]]]:
    """Apply the preregistered deterministic reference ranking."""
    train_rows = [row for row in rows if row.split == "train"]
    test_rows = [row for row in rows if row.split == "test"]
    ranked_train = sorted(
        train_rows,
        key=lambda row: (
            _selection_digest(
                config.selection.namespace,
                config.selection.seed,
                row.relative_path,
            ),
            row.relative_path,
        ),
    )
    if len(ranked_train) < config.selection.reference_count:
        raise ManifestIntegrityError(
            "official training partition is smaller than the requested reference count"
        )

    reference_rows = ranked_train[: config.selection.reference_count]
    calibration_rows = ranked_train[config.selection.reference_count :]
    if not calibration_rows:
        raise ManifestIntegrityError("calibration partition must not be empty")
    if not test_rows:
        raise ManifestIntegrityError("final-test partition must not be empty")

    reference = []
    for rank, row in enumerate(reference_rows, start=1):
        record = _base_record(config=config, row=row, partition="reference")
        record["selection_rank"] = rank
        record["selection_sha256"] = _selection_digest(
            config.selection.namespace,
            config.selection.seed,
            row.relative_path,
        )
        reference.append(record)

    calibration = []
    for rank, row in enumerate(
        calibration_rows,
        start=config.selection.reference_count + 1,
    ):
        record = _base_record(config=config, row=row, partition="calibration")
        record["selection_rank"] = rank
        record["selection_sha256"] = _selection_digest(
            config.selection.namespace,
            config.selection.seed,
            row.relative_path,
        )
        calibration.append(record)

    final_test = [
        _base_record(
            config=config,
            row=row,
            partition="final-test",
            include_label=False,
        )
        for row in sorted(test_rows, key=lambda item: item.relative_path)
    ]
    return {
        "reference": reference,
        "calibration": calibration,
        "final-test": final_test,
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            stream.write("\n")


def _recorded_archive_sha256(path: Path) -> str:
    if not path.exists():
        raise ManifestIntegrityError("archive provenance is required before manifests can be fixed")
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("sha256")
    except (OSError, json.JSONDecodeError, AttributeError) as error:
        raise ManifestIntegrityError(f"invalid archive provenance: {error}") from error
    if not isinstance(value, str) or len(value) != 64:
        raise ManifestIntegrityError("archive provenance does not contain a valid SHA-256")
    return value


def build_manifests(config: ProjectConfig) -> ManifestSummary:
    """Build manifests solely from the pinned split CSV and provenance metadata."""
    rows = load_official_rows(
        config.paths.split_csv,
        expected_sha256=config.split.sha256,
        category=config.category,
    )
    partitions = partition_rows(rows, config=config)
    archive_sha256 = _recorded_archive_sha256(config.paths.archive_provenance)

    destination = config.paths.manifest_dir
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.building.",
        )
    )

    try:
        manifest_metadata: dict[str, dict[str, Any]] = {}
        for partition, filename in MANIFEST_NAMES.items():
            path = staging / filename
            _write_jsonl(path, partitions[partition])
            manifest_metadata[partition] = {
                "file": filename,
                "record_count": len(partitions[partition]),
                "sha256": sha256_file(path),
            }

        metadata = {
            "schema_version": 1,
            "dataset": {
                "name": config.dataset_name,
                "category": config.category,
                "license": config.dataset_license,
                "archive_identifier": config.archive.identifier,
                "archive_sha256": archive_sha256,
            },
            "official_split": {
                "repository": config.split.repository,
                "revision": config.split.revision,
                "path": config.split.path,
                "sha256": config.split.sha256,
            },
            "selection": asdict(config.selection),
            "manifests": manifest_metadata,
            "final_test_access_policy": {
                "stage": "manifest_only",
                "image_content_reading": False,
                "image_display": False,
                "class_label_exposure": False,
                "score_calculation": False,
                "statistics": False,
                "parameter_selection": False,
            },
        }
        write_json_atomic(staging / "manifest-set.json", metadata)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return ManifestSummary(
        reference_count=len(partitions["reference"]),
        calibration_count=len(partitions["calibration"]),
        final_test_count=len(partitions["final-test"]),
        archive_sha256_recorded=True,
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ManifestIntegrityError(
                    f"{path.name}:{line_number} is not valid JSON"
                ) from error
            if not isinstance(record, dict):
                raise ManifestIntegrityError(f"{path.name}:{line_number} must contain an object")
            records.append(record)
    return records


def validate_manifests(config: ProjectConfig) -> ManifestSummary:
    """Validate metadata and recorded hashes; never open dataset images."""
    directory = config.paths.manifest_dir
    metadata_path = directory / "manifest-set.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestIntegrityError(f"cannot read manifest-set.json: {error}") from error

    split_metadata = metadata.get("official_split", {})
    if split_metadata.get("revision") != config.split.revision:
        raise ManifestIntegrityError("manifest split revision does not match configuration")
    if split_metadata.get("sha256") != config.split.sha256:
        raise ManifestIntegrityError("manifest split checksum does not match configuration")
    if metadata.get("selection") != asdict(config.selection):
        raise ManifestIntegrityError("manifest selection does not match configuration")
    expected_policy = {
        "stage": "manifest_only",
        "image_content_reading": False,
        "image_display": False,
        "class_label_exposure": False,
        "score_calculation": False,
        "statistics": False,
        "parameter_selection": False,
    }
    if metadata.get("final_test_access_policy") != expected_policy:
        raise ManifestIntegrityError("final-test access policy is missing or changed")

    all_ids: set[str] = set()
    all_paths: set[str] = set()
    counts: dict[str, int] = {}
    actual_partitions: dict[str, list[dict[str, Any]]] = {}
    manifest_metadata = metadata.get("manifests")
    if not isinstance(manifest_metadata, dict):
        raise ManifestIntegrityError("manifest-set.json has no manifest inventory")

    for partition, filename in MANIFEST_NAMES.items():
        path = directory / filename
        expected = manifest_metadata.get(partition)
        if not isinstance(expected, dict):
            raise ManifestIntegrityError(f"missing metadata for {partition}")
        if expected.get("file") != filename:
            raise ManifestIntegrityError(f"unexpected filename for {partition}")
        if sha256_file(path) != expected.get("sha256"):
            raise ManifestIntegrityError(f"checksum mismatch for {filename}")

        records = _load_jsonl(path)
        actual_partitions[partition] = records
        if len(records) != expected.get("record_count"):
            raise ManifestIntegrityError(f"record count mismatch for {filename}")
        counts[partition] = len(records)

        for record in records:
            required_keys = BASE_RECORD_KEYS | (
                {"label"} if partition in {"reference", "calibration"} else set()
            )
            if not required_keys.issubset(record):
                raise ManifestIntegrityError(f"{filename} contains an incomplete record")
            if record["schema_version"] != 1:
                raise ManifestIntegrityError(f"{filename} contains another schema version")
            if (
                not isinstance(record["source_row"], int)
                or isinstance(record["source_row"], bool)
                or record["source_row"] < 2
            ):
                raise ManifestIntegrityError(f"{filename} contains an invalid source row")
            if record["partition"] != partition:
                raise ManifestIntegrityError(f"{filename} contains another partition")
            if record["category"] != config.category:
                raise ManifestIntegrityError(f"{filename} contains another category")
            if "mask" in record or "mask_path" in record:
                raise ManifestIntegrityError(f"{filename} contains pixel-level metadata")
            relative_path = normalize_relative_path(record["relative_path"])
            if not relative_path.startswith(f"{config.category}/"):
                raise ManifestIntegrityError(f"{filename} contains a path outside the category")
            if record["id"] != _record_id(config.category, relative_path):
                raise ManifestIntegrityError(f"{filename} contains an invalid record ID")
            if record["id"] in all_ids:
                raise ManifestIntegrityError(f"duplicate manifest ID: {record['id']}")
            if relative_path in all_paths:
                raise ManifestIntegrityError(f"partition overlap: {relative_path}")
            all_ids.add(record["id"])
            all_paths.add(relative_path)

            if partition in {"reference", "calibration"}:
                if record["source_split"] != "train" or record["label"] != "normal":
                    raise ManifestIntegrityError(
                        f"{filename} contains a non-normal training record"
                    )
                expected_digest = _selection_digest(
                    config.selection.namespace,
                    config.selection.seed,
                    relative_path,
                )
                if record.get("selection_sha256") != expected_digest:
                    raise ManifestIntegrityError(f"{filename} contains an invalid selection digest")
            elif record["source_split"] != "test":
                raise ManifestIntegrityError("final-test contains an invalid split record")
            elif (
                "label" in record
                or "selection_sha256" in record
                or "selection_rank" in record
            ):
                raise ManifestIntegrityError(
                    "final-test exposes labels or reference-selection metadata"
                )

    if counts["reference"] != config.selection.reference_count:
        raise ManifestIntegrityError("reference count does not match configuration")
    if counts["calibration"] == 0 or counts["final-test"] == 0:
        raise ManifestIntegrityError("calibration and final-test must not be empty")

    expected_rows = load_official_rows(
        config.paths.split_csv,
        expected_sha256=config.split.sha256,
        category=config.category,
    )
    expected_partitions = partition_rows(expected_rows, config=config)
    if actual_partitions != expected_partitions:
        raise ManifestIntegrityError("manifests do not match the pinned official split")

    archive_sha256 = metadata.get("dataset", {}).get("archive_sha256")
    if archive_sha256 != _recorded_archive_sha256(config.paths.archive_provenance):
        raise ManifestIntegrityError("manifest archive checksum does not match provenance")
    return ManifestSummary(
        reference_count=counts["reference"],
        calibration_count=counts["calibration"],
        final_test_count=counts["final-test"],
        archive_sha256_recorded=isinstance(archive_sha256, str),
    )
