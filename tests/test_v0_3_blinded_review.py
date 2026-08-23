from __future__ import annotations

import hashlib
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
import pytest

import few_shot_anomaly_poc.v0_3_blinded_review as blinded_module
from few_shot_anomaly_poc.v0_3_blinded_review import (
    FORM_FIELDS,
    BlankObservationForm,
    BlindReviewItem,
    ReviewAssetIdentity,
    V0_3BlindedReviewError,
    build_review_completion_checkpoint,
    collect_blind_observations,
    decode_verified_opaque_asset,
    read_blind_observations_csv,
    read_review_assets_csv,
    reviewer_visible_field_names,
    serialize_blind_observations,
    serialize_review_completion_checkpoint,
    validate_review_assets,
    validate_review_completion_checkpoint,
    write_blind_observations_csv,
)
from few_shot_anomaly_poc.v0_3_diagnostic_contract import (
    EXPECTED_ASSET_IDS,
    load_v0_3_schema,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = load_v0_3_schema(ROOT / "schemas/v0.3/diagnostic-artifacts.json")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identities() -> list[ReviewAssetIdentity]:
    return [
        ReviewAssetIdentity(asset_id=asset_id, byte_count=100 + index, sha256=f"{index:064x}")
        for index, asset_id in enumerate(EXPECTED_ASSET_IDS, start=1)
    ]


def _response(
    *,
    visible_observation: str = "A synthetic square is visible.",
    inferred_cause: str | None = None,
) -> dict:
    return {
        "review_status": "reviewed",
        "framing_or_crop_difference_visible": "no",
        "pose_or_registration_offset_visible": "no",
        "global_exposure_difference_visible": "uncertain",
        "blur_visible": "no",
        "localized_change_visible": "yes",
        "diffuse_change_visible": "no",
        "no_obvious_visible_change": "no",
        "visible_observation": visible_observation,
        "inferred_cause": inferred_cause,
    }


def _immutable_pixels(value: int = 0) -> np.ndarray:
    encoded = bytes([value] * (4 * 5 * 3))
    return np.frombuffer(encoded, dtype=np.uint8).reshape(4, 5, 3)


def _collected_records() -> list[dict]:
    return collect_blind_observations(
        _identities(),
        image_loader=lambda _identity: _immutable_pixels(),
        reviewer=lambda _item: _response(),
        schema=SCHEMA,
    )


def test_reviewer_view_model_is_an_exact_positive_allowlist() -> None:
    assert reviewer_visible_field_names() == ("asset_id", "pixels", "form")
    assert tuple(field.name for field in fields(BlindReviewItem)) == (
        "asset_id",
        "pixels",
        "form",
    )
    assert tuple(field.name for field in fields(BlankObservationForm)) == FORM_FIELDS
    forbidden = {
        "path",
        "sha256",
        "byte_count",
        "method",
        "case_type",
        "true_class",
        "anomaly_score",
        "threshold",
        "rank",
    }
    assert not (set(reviewer_visible_field_names()) & forbidden)
    assert not (set(FORM_FIELDS) & forbidden)


def test_review_inventory_requires_exact_membership_order_and_count() -> None:
    identities = _identities()
    assert validate_review_assets(identities, schema=SCHEMA) == tuple(identities)

    with pytest.raises(V0_3BlindedReviewError, match="exactly 28"):
        validate_review_assets(identities[:-1], schema=SCHEMA)
    with pytest.raises(V0_3BlindedReviewError, match="order or membership"):
        validate_review_assets(list(reversed(identities)), schema=SCHEMA)
    duplicated = [*identities]
    duplicated[-1] = duplicated[0]
    with pytest.raises(V0_3BlindedReviewError, match="order or membership"):
        validate_review_assets(duplicated, schema=SCHEMA)


def test_committed_review_inventory_reader_accepts_only_safe_fields() -> None:
    path = ROOT / "artifacts/v0.3/diagnostics/pcb2-development/review-assets.csv"
    assets = read_review_assets_csv(
        path,
        expected_sha256="399ebef79498917ff0cf5b2bf83467541a5417dce74bdd8e0e462fec70a7fd58",
        schema=SCHEMA,
    )

    assert [asset.asset_id for asset in assets] == list(EXPECTED_ASSET_IDS)
    assert all(not hasattr(asset, "relative_path") for asset in assets)
    assert all(not hasattr(asset, "method") for asset in assets)


def test_verified_decoder_preserves_synthetic_pixels_and_makes_them_immutable(
    tmp_path: Path,
) -> None:
    assets_root = tmp_path / "assets"
    assets_root.mkdir()
    original = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
    success, encoded = cv2.imencode(".png", original)
    assert success
    payload = encoded.tobytes()
    asset_id = EXPECTED_ASSET_IDS[0]
    (assets_root / f"{asset_id}.jpg").write_bytes(payload)
    identity = ReviewAssetIdentity(asset_id, len(payload), _sha(payload))

    pixels = decode_verified_opaque_asset(identity, assets_root=assets_root)

    assert pixels.shape == original.shape
    assert np.array_equal(pixels, original)
    assert pixels.dtype == np.uint8
    assert pixels.flags.c_contiguous
    assert not pixels.flags.writeable
    with pytest.raises(ValueError):
        pixels.setflags(write=True)


def test_decoder_rejects_hash_mismatch_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets_root = tmp_path / "assets"
    assets_root.mkdir()
    asset_id = EXPECTED_ASSET_IDS[0]
    payload = b"not-decoded-because-hash-check-fails"
    (assets_root / f"{asset_id}.jpg").write_bytes(payload)
    identity = ReviewAssetIdentity(asset_id, len(payload), "0" * 64)
    decode_called = False

    def forbidden_decode(*_args: object, **_kwargs: object) -> None:
        nonlocal decode_called
        decode_called = True
        return None

    monkeypatch.setattr(blinded_module.cv2, "imdecode", forbidden_decode)
    with pytest.raises(V0_3BlindedReviewError, match="SHA-256 changed"):
        decode_verified_opaque_asset(identity, assets_root=assets_root)
    assert decode_called is False


def test_decoder_rejects_unselected_asset_and_symlink(tmp_path: Path) -> None:
    assets_root = tmp_path / "assets"
    assets_root.mkdir()
    with pytest.raises(V0_3BlindedReviewError, match="not authorized"):
        decode_verified_opaque_asset(
            ReviewAssetIdentity("asset-999999", 1, "0" * 64),
            assets_root=assets_root,
        )

    target = tmp_path / "target.jpg"
    target.write_bytes(b"synthetic")
    asset_id = EXPECTED_ASSET_IDS[0]
    (assets_root / f"{asset_id}.jpg").symlink_to(target)
    with pytest.raises(V0_3BlindedReviewError, match="regular non-symlink"):
        decode_verified_opaque_asset(
            ReviewAssetIdentity(asset_id, len(b"synthetic"), _sha(b"synthetic")),
            assets_root=assets_root,
        )


def test_collector_calls_each_fixed_asset_once_in_order_with_read_only_pixels() -> None:
    loader_calls: list[str] = []
    reviewer_calls: list[str] = []

    def loader(identity: ReviewAssetIdentity) -> np.ndarray:
        loader_calls.append(identity.asset_id)
        return _immutable_pixels(len(loader_calls) % 255)

    def reviewer(item: BlindReviewItem) -> dict:
        reviewer_calls.append(item.asset_id)
        assert not item.pixels.flags.writeable
        assert item.form == BlankObservationForm()
        assert not hasattr(item, "sha256")
        assert not hasattr(item, "method")
        return _response()

    records = collect_blind_observations(
        _identities(),
        image_loader=loader,
        reviewer=reviewer,
        schema=SCHEMA,
    )

    assert loader_calls == reviewer_calls == list(EXPECTED_ASSET_IDS)
    assert [record["asset_id"] for record in records] == list(EXPECTED_ASSET_IDS)
    assert len(set(loader_calls)) == 28


def test_collector_rejects_protected_field_injection() -> None:
    def reviewer(_item: BlindReviewItem) -> dict:
        return {**_response(), "method": "ecc_residual"}

    with pytest.raises(V0_3BlindedReviewError, match="fields differ"):
        collect_blind_observations(
            _identities(),
            image_loader=lambda _identity: _immutable_pixels(),
            reviewer=reviewer,
            schema=SCHEMA,
        )


@pytest.mark.parametrize(
    ("note", "accepted"),
    [
        ("x", True),
        ("x" * 300, True),
        ("", False),
        ("x" * 301, False),
        ("line one\nline two", False),
    ],
)
def test_visible_note_boundaries_are_fixed(note: str, accepted: bool) -> None:
    reviewer = lambda _item: _response(visible_observation=note)  # noqa: E731
    if accepted:
        records = collect_blind_observations(
            _identities(),
            image_loader=lambda _identity: _immutable_pixels(),
            reviewer=reviewer,
            schema=SCHEMA,
        )
        assert records[0]["visible_observation"] == note
    else:
        with pytest.raises(V0_3BlindedReviewError, match="violates"):
            collect_blind_observations(
                _identities(),
                image_loader=lambda _identity: _immutable_pixels(),
                reviewer=reviewer,
                schema=SCHEMA,
            )


def test_inferred_cause_is_null_or_one_to_300_code_points() -> None:
    assert _collected_records()[0]["inferred_cause"] is None

    for value in ("", "x" * 301, "line\nbreak"):
        with pytest.raises(V0_3BlindedReviewError, match="violates"):
            collect_blind_observations(
                _identities(),
                image_loader=lambda _identity: _immutable_pixels(),
                reviewer=lambda _item, cause=value: _response(inferred_cause=cause),
                schema=SCHEMA,
            )


def test_blind_csv_round_trip_is_deterministic_and_exclusive(tmp_path: Path) -> None:
    records = _collected_records()
    output = tmp_path / "blind-observations.csv"

    first_bytes = serialize_blind_observations(records)
    second_bytes = serialize_blind_observations(records)
    digest = write_blind_observations_csv(output, records, schema=SCHEMA)

    assert first_bytes == second_bytes == output.read_bytes()
    assert b"\r\n" not in first_bytes
    assert digest == _sha(first_bytes)
    assert read_blind_observations_csv(
        output,
        expected_sha256=digest,
        schema=SCHEMA,
    ) == records
    with pytest.raises(V0_3BlindedReviewError, match="already exists"):
        write_blind_observations_csv(output, records, schema=SCHEMA)


def test_review_completion_checkpoint_stays_closed_before_metadata_join() -> None:
    checkpoint = build_review_completion_checkpoint(
        review_assets_sha256="a" * 64,
        blind_observations_sha256="b" * 64,
    )

    assert validate_review_completion_checkpoint(checkpoint) == checkpoint
    assert checkpoint["first_pass_complete"] is True
    assert checkpoint["protected_metadata_exposed"] is False
    assert checkpoint["method_metadata_joined"] is False
    assert serialize_review_completion_checkpoint(checkpoint).endswith(b"\n")

    checkpoint["method_metadata_joined"] = True
    with pytest.raises(V0_3BlindedReviewError, match="too early"):
        validate_review_completion_checkpoint(checkpoint)
