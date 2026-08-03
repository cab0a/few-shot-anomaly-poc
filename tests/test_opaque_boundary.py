from __future__ import annotations

import json
from pathlib import Path

import pytest

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.opaque_boundary import (
    SCORING_RECORD_FIELDS,
    BoundarySourceRecord,
    OpaqueBoundaryError,
    load_scoring_manifest,
    load_sealed_mapping,
    prepare_opaque_boundary,
    run_synthetic_boundary_feasibility,
    validate_scoring_manifest,
)


def _source_fixture(root: Path) -> tuple[BoundarySourceRecord, ...]:
    files = {
        "class-a/semantic-first.png": b"first-image-bytes",
        "class-b/semantic-second.jpg": b"second-image-bytes",
    }
    labels = ("protected-a", "protected-b")
    for relative_path, content in files.items():
        path = root.joinpath(*relative_path.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return tuple(
        BoundarySourceRecord(source_path=path, class_label=label)
        for path, label in zip(files, labels, strict=True)
    )


def test_prepare_opaque_boundary_separates_scorer_and_reveal_metadata(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    records = _source_fixture(source_root)
    result = prepare_opaque_boundary(
        source_root=source_root,
        scorer_root=tmp_path / "scorer",
        sealed_mapping_path=tmp_path / "sealed/mapping.json",
        records=records,
        ordering_key=b"a" * 32,
    )

    scoring = load_scoring_manifest(result["scoring_manifest_path"])
    sealed = load_sealed_mapping(result["sealed_mapping_path"])
    scoring_text = result["scoring_manifest_path"].read_text(encoding="utf-8")

    assert result["asset_count"] == 2
    assert all(set(record) == SCORING_RECORD_FIELDS for record in scoring["records"])
    assert [record["asset_id"] for record in scoring["records"]] == [
        "asset-000000",
        "asset-000001",
    ]
    assert [record["asset_id"] for record in sealed["records"]] == [
        "asset-000000",
        "asset-000001",
    ]
    assert "semantic-first" not in scoring_text
    assert "semantic-second" not in scoring_text
    assert "protected-a" not in scoring_text
    assert "protected-b" not in scoring_text
    assert result["sealed_mapping_path"].is_relative_to(tmp_path / "sealed")
    for record in scoring["records"]:
        copied = tmp_path / "scorer" / record["relative_path"]
        assert copied.stat().st_size == record["byte_count"]
        assert sha256_file(copied) == record["sha256"]


def test_prepare_opaque_boundary_is_deterministic_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    records = _source_fixture(source_root)
    first = prepare_opaque_boundary(
        source_root=source_root,
        scorer_root=tmp_path / "scorer-a",
        sealed_mapping_path=tmp_path / "sealed-a/mapping.json",
        records=records,
        ordering_key=b"b" * 32,
    )
    second = prepare_opaque_boundary(
        source_root=source_root,
        scorer_root=tmp_path / "scorer-b",
        sealed_mapping_path=tmp_path / "sealed-b/mapping.json",
        records=records,
        ordering_key=b"b" * 32,
    )

    assert first["scoring_manifest_sha256"] == second["scoring_manifest_sha256"]
    assert first["sealed_mapping_sha256"] == second["sealed_mapping_sha256"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_opaque_boundary(
            source_root=source_root,
            scorer_root=tmp_path / "scorer-a",
            sealed_mapping_path=tmp_path / "sealed-a/mapping.json",
            records=records,
            ordering_key=b"b" * 32,
        )


def test_scoring_manifest_rejects_protected_or_unregistered_fields() -> None:
    base = {
        "records": [
            {
                "asset_id": "asset-000000",
                "byte_count": 1,
                "relative_path": "assets/asset-000000.png",
                "sha256": "a" * 64,
            }
        ],
        "schema_version": "v0.2-opaque-scoring-manifest-v1",
    }

    for forbidden_field in ("class_label", "source_path", "split", "hmac_key"):
        changed = json.loads(json.dumps(base))
        changed["records"][0][forbidden_field] = "forbidden"
        with pytest.raises(OpaqueBoundaryError, match="allowlist"):
            validate_scoring_manifest(changed)


def test_boundary_rejects_path_escape_short_key_and_unsealed_location(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "image.png").write_bytes(b"image")

    with pytest.raises(OpaqueBoundaryError, match="ordering_key"):
        prepare_opaque_boundary(
            source_root=source_root,
            scorer_root=tmp_path / "scorer-short-key",
            sealed_mapping_path=tmp_path / "sealed-short-key.json",
            records=(BoundarySourceRecord("image.png", "label"),),
            ordering_key=b"short",
        )
    with pytest.raises(OpaqueBoundaryError, match="escape"):
        prepare_opaque_boundary(
            source_root=source_root,
            scorer_root=tmp_path / "scorer-escape",
            sealed_mapping_path=tmp_path / "sealed-escape.json",
            records=(BoundarySourceRecord("../image.png", "label"),),
            ordering_key=b"c" * 32,
        )
    with pytest.raises(OpaqueBoundaryError, match="outside"):
        prepare_opaque_boundary(
            source_root=source_root,
            scorer_root=tmp_path / "scorer-unsealed",
            sealed_mapping_path=tmp_path / "scorer-unsealed/mapping.json",
            records=(BoundarySourceRecord("image.png", "label"),),
            ordering_key=b"c" * 32,
        )


def test_synthetic_feasibility_report_excludes_protected_fixture_values(
    tmp_path: Path,
) -> None:
    report = run_synthetic_boundary_feasibility(tmp_path / "checkpoint")
    serialized = json.dumps(report, sort_keys=True)

    assert report["decision"]["status"] == "pass"
    assert all(report["checks"].values())
    assert report["boundary"] == {
        "boundary_prepared": False,
        "dataset_access": False,
        "dataset_labels_accessed": False,
        "image_decode_performed": False,
        "network_access": False,
        "official_split_access": False,
        "scoring_performed": False,
        "synthetic_fixture_only": True,
    }
    assert "visual-one" not in serialized
    assert "fixture-class" not in serialized
    assert "synthetic opaque boundary feasibility only" not in serialized
    assert "/home/" not in serialized
