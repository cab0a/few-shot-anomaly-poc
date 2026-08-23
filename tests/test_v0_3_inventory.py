from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from few_shot_anomaly_poc.v0_3_diagnostic_contract import (
    EXPECTED_ASSET_IDS,
    EXPECTED_METHOD_CASES,
    load_v0_3_schema,
)
from few_shot_anomaly_poc.v0_3_inventory import (
    NORMAL_PARTITION_FIELDS,
    REVIEW_ASSET_FIELDS,
    REVIEW_CASE_FIELDS,
    V0_3InventoryError,
    build_review_assets,
    build_review_case_links,
    select_normal_partition,
    serialize_csv,
    write_inventory_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = load_v0_3_schema(ROOT / "schemas/v0.3/diagnostic-artifacts.json")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _scoring_records() -> dict[str, dict]:
    return {
        asset_id: {
            "asset_id": asset_id,
            "relative_path": f"assets/{asset_id}.jpg",
            "byte_count": index + 1,
            "sha256": _sha(asset_id),
        }
        for index, asset_id in enumerate(EXPECTED_ASSET_IDS)
    }


def _normal_records(count: int = 80) -> list[dict]:
    return [
        {
            "relative_path": f"pcb2/Data/Images/Normal/{index:04d}.JPG",
            "byte_count": 1000 + index,
            "sha256": _sha(f"normal-{index}"),
        }
        for index in range(count)
    ]


def test_review_asset_projection_is_exactly_the_first_pass_safe_allowlist() -> None:
    records = build_review_assets(_scoring_records(), schema=SCHEMA)

    assert [record["asset_id"] for record in records] == list(EXPECTED_ASSET_IDS)
    assert all(tuple(record) == REVIEW_ASSET_FIELDS for record in records)
    forbidden = {
        "relative_path",
        "method",
        "case_type",
        "true_class",
        "anomaly_score",
        "threshold",
        "rank",
    }
    assert not any(set(record) & forbidden for record in records)


def test_review_asset_projection_rejects_missing_selected_asset() -> None:
    records = _scoring_records()
    del records[EXPECTED_ASSET_IDS[0]]

    with pytest.raises(V0_3InventoryError, match="absent"):
        build_review_assets(records, schema=SCHEMA)


def test_case_links_are_exact_and_remain_separate_from_review_assets() -> None:
    hashes = {
        "ecc_residual": "a" * 64,
        "patch_hog_ocsvm": "b" * 64,
        "dinov2_vits14_224_nn": "c" * 64,
    }
    records = build_review_case_links(failure_hashes=hashes, schema=SCHEMA)

    identities = tuple(
        (record["method"], record["case_type"], record["rank"], record["asset_id"])
        for record in records
    )
    assert identities == EXPECTED_METHOD_CASES
    assert all(tuple(record) == REVIEW_CASE_FIELDS for record in records)
    assert sum(record["asset_id"] == "asset-000112" for record in records) == 2


def test_normal_selection_matches_independent_digest_sort() -> None:
    records = _normal_records()
    prefix = "synthetic-selection:"
    selected = select_normal_partition(records, selection_prefix=prefix, selected_count=60)
    expected = sorted(
        records,
        key=lambda record: (
            _sha(f"{prefix}{record['relative_path']}"),
            record["relative_path"],
        ),
    )[:60]

    assert [record["relative_path"] for record in selected] == [
        record["relative_path"] for record in expected
    ]
    assert [record["selection_rank"] for record in selected] == list(range(1, 61))
    assert all(tuple(record) == NORMAL_PARTITION_FIELDS for record in selected)


def test_normal_selection_rejects_duplicate_or_unsafe_input_path() -> None:
    records = _normal_records()
    records[1]["relative_path"] = "../outside.JPG"

    with pytest.raises(V0_3InventoryError, match="unsafe"):
        select_normal_partition(records, selection_prefix="synthetic:", selected_count=60)


def test_csv_serialization_is_deterministic_lf_utf8() -> None:
    records = build_review_assets(_scoring_records(), schema=SCHEMA)

    first = serialize_csv(REVIEW_ASSET_FIELDS, records)
    second = serialize_csv(REVIEW_ASSET_FIELDS, records)

    assert first == second
    assert b"\r\n" not in first
    assert first.endswith(b"\n")
    assert first.count(b"\n") == 29


def test_bundle_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "diagnostics"
    files = {
        "review-assets.csv": b"a\n",
        "review-case-linkage.csv": b"b\n",
        "normal-diagnostic-partition.csv": b"c\n",
        "pre-access-checkpoint.json": b"{}\n",
    }
    write_inventory_bundle(output, files)

    with pytest.raises(V0_3InventoryError, match="already exists"):
        write_inventory_bundle(output, files)
    assert {path.name for path in output.iterdir()} == set(files)


def test_bundle_writer_rejects_extra_file_mapping_without_publishing(tmp_path: Path) -> None:
    output = tmp_path / "diagnostics"
    files = {
        "review-assets.csv": b"a\n",
        "review-case-linkage.csv": b"b\n",
        "normal-diagnostic-partition.csv": b"c\n",
        "pre-access-checkpoint.json": b"{}\n",
        "unexpected.jpg": b"synthetic",
    }

    with pytest.raises(V0_3InventoryError, match="unexpected"):
        write_inventory_bundle(output, files)
    assert not output.exists()
