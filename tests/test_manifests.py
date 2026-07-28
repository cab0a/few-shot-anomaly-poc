from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from few_shot_anomaly_poc.errors import ChecksumMismatchError, ManifestIntegrityError
from few_shot_anomaly_poc.manifests import (
    build_manifests,
    load_official_rows,
    validate_manifests,
)
from tests.helpers import create_config, final_test_row, normal_train, write_split


def _rows() -> list[dict[str, str]]:
    return [
        *(normal_train(index) for index in range(1, 7)),
        final_test_row(1001, "normal"),
        final_test_row(1002, "anomaly"),
    ]


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_build_and_validate_manifests_without_extracted_dataset(tmp_path: Path) -> None:
    config = create_config(tmp_path, rows=_rows(), reference_count=2)

    summary = build_manifests(config)
    validated = validate_manifests(config)

    assert summary == validated
    assert summary.reference_count == 2
    assert summary.calibration_count == 4
    assert summary.final_test_count == 2
    assert not config.paths.extracted.exists()

    reference = _jsonl(config.paths.manifest_dir / "reference.jsonl")
    calibration = _jsonl(config.paths.manifest_dir / "calibration.jsonl")
    final_test = _jsonl(config.paths.manifest_dir / "final-test.jsonl")
    all_paths = {
        partition: {record["relative_path"] for record in records}
        for partition, records in {
            "reference": reference,
            "calibration": calibration,
            "final-test": final_test,
        }.items()
    }
    assert all_paths["reference"].isdisjoint(all_paths["calibration"])
    assert all_paths["reference"].isdisjoint(all_paths["final-test"])
    assert all_paths["calibration"].isdisjoint(all_paths["final-test"])
    assert all("mask" not in record and "label" not in record for record in final_test)


def test_reference_selection_is_deterministic(tmp_path: Path) -> None:
    config = create_config(tmp_path, rows=_rows(), reference_count=2)
    build_manifests(config)

    selected = _jsonl(config.paths.manifest_dir / "reference.jsonl")
    ranked_paths = [record["relative_path"] for record in selected]
    candidates = [row["image"] for row in _rows() if row["split"] == "train"]
    expected = sorted(
        candidates,
        key=lambda path: (
            hashlib.sha256(f"few-shot-anomaly-poc:v0.1:42:{path}".encode()).hexdigest(),
            path,
        ),
    )
    assert ranked_paths == expected[:2]

    calibration = _jsonl(config.paths.manifest_dir / "calibration.jsonl")
    combined = selected + calibration
    assert [record["relative_path"] for record in combined] == expected
    assert [record["selection_rank"] for record in combined] == list(
        range(1, len(expected) + 1)
    )
    assert all(
        record["selection_sha256"]
        == hashlib.sha256(
            f"few-shot-anomaly-poc:v0.1:42:{record['relative_path']}".encode()
        ).hexdigest()
        for record in combined
    )


def test_manifest_metadata_records_final_test_access_policy(tmp_path: Path) -> None:
    config = create_config(tmp_path, rows=_rows(), reference_count=2)
    build_manifests(config)

    metadata = json.loads(
        (config.paths.manifest_dir / "manifest-set.json").read_text(encoding="utf-8")
    )
    policy = metadata["final_test_access_policy"]
    assert policy["stage"] == "manifest_only"
    assert policy["image_content_reading"] is False
    assert policy["image_display"] is False
    assert policy["class_label_exposure"] is False
    assert policy["statistics"] is False
    assert metadata["dataset"]["archive_sha256"] == "b" * 64


def test_build_requires_archive_provenance(tmp_path: Path) -> None:
    config = create_config(
        tmp_path,
        rows=_rows(),
        reference_count=2,
        write_archive_provenance=False,
    )

    with pytest.raises(ManifestIntegrityError, match="provenance"):
        build_manifests(config)


def test_split_checksum_is_required(tmp_path: Path) -> None:
    config = create_config(tmp_path, rows=_rows(), reference_count=2)

    with pytest.raises(ChecksumMismatchError):
        load_official_rows(
            config.paths.split_csv,
            expected_sha256="0" * 64,
            category="pcb1",
        )


def test_split_rejects_duplicate_paths(tmp_path: Path) -> None:
    split = tmp_path / "split.csv"
    duplicate = normal_train(1)
    digest = write_split(
        split,
        [duplicate, duplicate, final_test_row(1001, "normal")],
    )

    with pytest.raises(ManifestIntegrityError, match="duplicate"):
        load_official_rows(split, expected_sha256=digest, category="pcb1")


def test_split_rejects_anomaly_training_row(tmp_path: Path) -> None:
    split = tmp_path / "split.csv"
    invalid = normal_train(1)
    invalid["label"] = "anomaly"
    digest = write_split(split, [invalid, final_test_row(1001, "normal")])

    with pytest.raises(ManifestIntegrityError, match="non-normal"):
        load_official_rows(split, expected_sha256=digest, category="pcb1")


def test_validator_detects_partition_overlap(tmp_path: Path) -> None:
    config = create_config(tmp_path, rows=_rows(), reference_count=2)
    build_manifests(config)
    reference = _jsonl(config.paths.manifest_dir / "reference.jsonl")
    final_path = config.paths.manifest_dir / "final-test.jsonl"
    final_records = _jsonl(final_path)
    final_records[0]["relative_path"] = reference[0]["relative_path"]
    final_records[0]["id"] = reference[0]["id"]
    final_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in final_records),
        encoding="utf-8",
    )

    metadata_path = config.paths.manifest_dir / "manifest-set.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["manifests"]["final-test"]["sha256"] = hashlib.sha256(
        final_path.read_bytes()
    ).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ManifestIntegrityError, match=r"duplicate|overlap"):
        validate_manifests(config)


def test_validator_rejects_exposed_final_test_label(tmp_path: Path) -> None:
    config = create_config(tmp_path, rows=_rows(), reference_count=2)
    build_manifests(config)
    final_path = config.paths.manifest_dir / "final-test.jsonl"
    final_records = _jsonl(final_path)
    final_records[0]["label"] = "normal"
    final_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in final_records),
        encoding="utf-8",
    )

    metadata_path = config.paths.manifest_dir / "manifest-set.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["manifests"]["final-test"]["sha256"] = hashlib.sha256(
        final_path.read_bytes()
    ).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ManifestIntegrityError, match="exposes labels"):
        validate_manifests(config)
