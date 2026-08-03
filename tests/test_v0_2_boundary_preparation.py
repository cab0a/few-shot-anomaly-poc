from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.manifests import SplitRow
from few_shot_anomaly_poc.opaque_boundary import (
    load_scoring_manifest,
    load_sealed_mapping,
)
from few_shot_anomaly_poc.v0_2_boundary_preparation import (
    CATEGORY,
    NORMAL_MANIFEST_SCHEMA,
    NORMAL_MANIFEST_SET_SCHEMA,
    REFERENCE_COUNT,
    RUN_ID,
    SELECTION_NAMESPACE,
    SELECTION_SEED,
    V0_2BoundaryPreparationError,
    build_public_boundary_record,
    prepare_partition_assets,
    select_v0_2_partitions,
    verify_category_image_inventory,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    ARTIFACT_CONTRACT_VERSION,
    load_v0_2_artifact_schema,
    load_v0_2_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _rows() -> tuple[SplitRow, ...]:
    train = tuple(
        SplitRow(
            source_row=index + 2,
            split="train",
            label="normal",
            relative_path=f"pcb2/Data/Images/Normal/train-{index:03d}.JPG",
        )
        for index in range(22)
    )
    final_test = (
        SplitRow(
            source_row=100,
            split="test",
            label="normal",
            relative_path="pcb2/Data/Images/Normal/test-normal.JPG",
        ),
        SplitRow(
            source_row=101,
            split="test",
            label="anomaly",
            relative_path="pcb2/Data/Images/Anomaly/test-anomaly.JPG",
        ),
    )
    return (*train, *final_test)


def _write_sources(source_root: Path, rows: tuple[SplitRow, ...]) -> None:
    for index, row in enumerate(rows):
        path = source_root.joinpath(*row.relative_path.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic-boundary-bytes-{index}".encode())


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_fixed_partition_selection_is_deterministic_and_normal_only() -> None:
    rows = _rows()

    reference, calibration, final_test = select_v0_2_partitions(rows)

    expected_train = sorted(
        rows[:22],
        key=lambda row: (
            hashlib.sha256(
                (f"{SELECTION_NAMESPACE}:{SELECTION_SEED}:{row.relative_path}").encode()
            ).hexdigest(),
            row.relative_path,
        ),
    )
    assert reference == tuple(expected_train[:REFERENCE_COUNT])
    assert calibration == tuple(expected_train[REFERENCE_COUNT:])
    assert len(reference) == 20
    assert len(calibration) == 2
    assert all(row.label == "normal" for row in (*reference, *calibration))
    assert [row.relative_path for row in final_test] == sorted(
        row.relative_path for row in rows[22:]
    )


def test_partition_selection_rejects_overlap_and_missing_calibration() -> None:
    duplicate = (*_rows(), _rows()[0])
    too_small = _rows()[:20] + _rows()[22:]

    with pytest.raises(V0_2BoundaryPreparationError, match="not unique"):
        select_v0_2_partitions(duplicate)
    with pytest.raises(V0_2BoundaryPreparationError, match="reference and calibration"):
        select_v0_2_partitions(too_small)


def test_category_inventory_requires_exact_split_path_set(tmp_path: Path) -> None:
    rows = _rows()
    source_root = tmp_path / "source"
    _write_sources(source_root, rows)

    verify_category_image_inventory(source_root=source_root, rows=rows)
    extra = source_root / CATEGORY / "Data/Images/Normal/extra.JPG"
    extra.write_bytes(b"extra")
    with pytest.raises(V0_2BoundaryPreparationError, match="not identical"):
        verify_category_image_inventory(source_root=source_root, rows=rows)


def test_partition_assets_separate_normal_manifests_and_final_labels(
    tmp_path: Path,
) -> None:
    rows = _rows()
    external_root = tmp_path / "external"
    source_root = external_root / "source"
    _write_sources(source_root, rows)

    result = prepare_partition_assets(
        source_root=source_root,
        external_root=external_root,
        rows=rows,
        ordering_key=b"k" * 32,
    )

    assert result["reference_count"] == 20
    assert result["calibration_count"] == 2
    assert result["final_test_count"] == 2
    reference_path = external_root / "normal-manifests/reference.jsonl"
    calibration_path = external_root / "normal-manifests/calibration.jsonl"
    manifest_set_path = external_root / "normal-manifests/manifest-set.json"
    assert result["reference_manifest_sha256"] == sha256_file(reference_path)
    assert result["calibration_manifest_sha256"] == sha256_file(calibration_path)
    assert result["normal_manifest_set_sha256"] == sha256_file(manifest_set_path)

    normal_records = [*_read_jsonl(reference_path), *_read_jsonl(calibration_path)]
    assert [record["selection_rank"] for record in normal_records] == list(range(1, 23))
    assert all(record["schema_version"] == NORMAL_MANIFEST_SCHEMA for record in normal_records)
    assert all(record["partition"] in {"reference", "calibration"} for record in normal_records)
    assert all("label" not in record and "class_label" not in record for record in normal_records)
    manifest_set = json.loads(manifest_set_path.read_text(encoding="utf-8"))
    assert manifest_set["schema_version"] == NORMAL_MANIFEST_SET_SCHEMA
    assert manifest_set["image_content_decoded"] is False

    scoring = load_scoring_manifest(external_root / "scorer/scoring-manifest.json")
    sealed = load_sealed_mapping(external_root / "sealed/mapping.json")
    scoring_text = json.dumps(scoring, sort_keys=True)
    assert [record["asset_id"] for record in scoring["records"]] == [
        "asset-000000",
        "asset-000001",
    ]
    assert [record["asset_id"] for record in sealed["records"]] == [
        "asset-000000",
        "asset-000001",
    ]
    assert "Anomaly" not in scoring_text
    assert "Normal" not in scoring_text
    assert "test-anomaly" not in scoring_text
    assert "test-normal" not in scoring_text
    assert "class_label" not in scoring_text
    assert {record["class_label"] for record in sealed["records"]} == {
        "normal",
        "anomaly",
    }


def test_partition_assets_refuse_to_overwrite_a_completed_boundary(
    tmp_path: Path,
) -> None:
    rows = _rows()
    external_root = tmp_path / "external"
    source_root = external_root / "source"
    _write_sources(source_root, rows)
    prepare_partition_assets(
        source_root=source_root,
        external_root=external_root,
        rows=rows,
        ordering_key=b"m" * 32,
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_partition_assets(
            source_root=source_root,
            external_root=external_root,
            rows=rows,
            ordering_key=b"m" * 32,
        )


def test_public_boundary_record_contains_only_aggregate_fixed_evidence() -> None:
    config = load_v0_2_config(ROOT / "configs/v0.2.yaml")
    schema = load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json")

    record = build_public_boundary_record(
        run_kind="synthetic",
        archive_sha256=config["dataset"]["archive_sha256"],
        split_sha256=config["dataset"]["split_sha256"],
        reference_count=20,
        calibration_count=2,
        final_test_count=2,
        scoring_manifest_sha256="a" * 64,
        sealed_mapping_sha256="b" * 64,
        config=config,
        schema=schema,
    )
    serialized = json.dumps(record, sort_keys=True)

    assert record["contract_version"] == ARTIFACT_CONTRACT_VERSION
    assert record["run_id"] == RUN_ID
    assert record["final_test_class_counts_published"] is False
    assert record["raw_data_in_git"] is False
    for protected in (
        "class_label",
        "source_path",
        "relative_path",
        "ordering_key",
        "normal",
        "anomaly",
    ):
        assert protected not in serialized


def test_source_fixture_contains_no_decodable_image_requirement(tmp_path: Path) -> None:
    rows = _rows()
    external_root = tmp_path / "external"
    source_root = external_root / "source"
    _write_sources(source_root, rows)

    prepare_partition_assets(
        source_root=source_root,
        external_root=external_root,
        rows=rows,
        ordering_key=b"n" * 32,
    )

    assert all(
        path.read_bytes().startswith(b"synthetic-boundary-bytes-")
        for path in source_root.rglob("*.JPG")
    )
