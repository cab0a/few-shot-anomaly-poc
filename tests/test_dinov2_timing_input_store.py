from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from few_shot_anomaly_poc.dinov2_timing import (
    INPUT_STORE_SCHEMA,
    LOGICAL_STORE_ID,
    QUERY_COUNT,
    QUERY_IDS,
    REFERENCE_COUNT,
    REFERENCE_IDS,
    STORE_SHAPE,
    DINOv2TimingError,
    copy_store_image,
    create_synthetic_input_store,
    open_verified_synthetic_input_store,
    validate_synthetic_input_manifest,
)


@pytest.fixture(scope="module")
def generated_store(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    root = tmp_path_factory.mktemp("dinov2-timing-inputs")
    store_path = root / "inputs.npy"
    manifest_path = root / "manifest.json"

    manifest = create_synthetic_input_store(
        store_path=store_path,
        manifest_path=manifest_path,
    )

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    return store_path, manifest


def test_fixed_input_store_has_exact_order_and_bounded_policy(
    generated_store: tuple[Path, dict[str, object]],
) -> None:
    _, manifest = generated_store

    assert manifest["schema_version"] == INPUT_STORE_SCHEMA
    assert manifest["logical_store"]["logical_id"] == LOGICAL_STORE_ID
    assert manifest["logical_store"]["shape"] == list(STORE_SHAPE)
    assert manifest["logical_store"]["byte_count"] == 94_371_968
    assert manifest["logical_store"]["file_sha256"] == (
        "b57319a8aa9fc8c27d1daa22acf8640a31cf366074a2c42e14e65ff55f4501b7"
    )
    assert [item["asset_id"] for item in manifest["references"]] == list(REFERENCE_IDS)
    assert [item["asset_id"] for item in manifest["queries"]] == list(QUERY_IDS)
    assert len(manifest["references"]) == REFERENCE_COUNT
    assert len(manifest["queries"]) == QUERY_COUNT
    assert manifest["references"][0]["sha256"] == (
        "023bb0389ea38626636c3b42cd8ea40ddd297d7a4103ba8343eb0af694bdbc15"
    )
    assert manifest["queries"][0]["sha256"] == (
        "751b0ba5f549fafd2e6f77d9adb0f2d991845a4b273ef832790fd66a7ff16734"
    )
    assert manifest["queries"][-1]["sha256"] == (
        "7f68d98dc55025ddb09d5ce707428205e1bade9f0f50ddf49b08cfcbcc391310"
    )
    assert manifest["resident_policy"] == {
        "all_source_images_retained_in_process_memory": False,
        "current_source_image_count": 1,
        "memory_map_to_contiguous_copy_outside_timer": True,
    }


def test_fixed_input_store_is_repeatable_and_hash_addressed(
    generated_store: tuple[Path, dict[str, object]],
    tmp_path: Path,
) -> None:
    _, first = generated_store
    second = create_synthetic_input_store(
        store_path=tmp_path / "inputs.npy",
        manifest_path=tmp_path / "manifest.json",
    )

    assert second["logical_store"]["file_sha256"] == first["logical_store"]["file_sha256"]
    assert second["logical_store"]["byte_count"] == first["logical_store"]["byte_count"]
    assert [item["sha256"] for item in second["references"]] == [
        item["sha256"] for item in first["references"]
    ]
    assert [item["sha256"] for item in second["queries"]] == [
        item["sha256"] for item in first["queries"]
    ]


def test_open_and_copy_verify_store_and_per_image_identity(
    generated_store: tuple[Path, dict[str, object]],
) -> None:
    store_path, manifest = generated_store
    store = open_verified_synthetic_input_store(store_path=store_path, manifest=manifest)

    reference = copy_store_image(
        store,
        index=0,
        expected_sha256=manifest["references"][0]["sha256"],
    )
    query = copy_store_image(
        store,
        index=REFERENCE_COUNT,
        expected_sha256=manifest["queries"][0]["sha256"],
    )

    assert reference.shape == (512, 512, 3)
    assert query.shape == (512, 512, 3)
    assert reference.dtype == np.uint8
    assert query.dtype == np.uint8
    assert reference.flags.c_contiguous
    assert query.flags.c_contiguous
    assert reference.base is None
    assert query.base is None


def test_input_manifest_rejects_changed_order(
    generated_store: tuple[Path, dict[str, object]],
) -> None:
    _, manifest = generated_store
    changed = {**manifest, "references": list(reversed(manifest["references"]))}

    with pytest.raises(DINOv2TimingError, match="IDs or order"):
        validate_synthetic_input_manifest(changed)


def test_input_store_refuses_overwrite(tmp_path: Path) -> None:
    store_path = tmp_path / "inputs.npy"
    store_path.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        create_synthetic_input_store(
            store_path=store_path,
            manifest_path=tmp_path / "manifest.json",
        )
