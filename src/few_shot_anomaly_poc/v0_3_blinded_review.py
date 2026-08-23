"""Positive-allowlist first-pass review primitive for the v0.3 diagnostic."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from few_shot_anomaly_poc.v0_3_diagnostic_contract import (
    ARTIFACT_CONTRACT_VERSION,
    BLIND_OBSERVATION_FIELDS,
    DIAGNOSTIC_ID,
    EXPECTED_ASSET_IDS,
    V0_3DiagnosticContractError,
    sha256_file,
    validate_blind_observation_records,
    validate_tabular_record,
)

FORM_FIELDS = (
    "review_status",
    "framing_or_crop_difference_visible",
    "pose_or_registration_offset_visible",
    "global_exposure_difference_visible",
    "blur_visible",
    "localized_change_visible",
    "diffuse_change_visible",
    "no_obvious_visible_change",
    "visible_observation",
    "inferred_cause",
)


class V0_3BlindedReviewError(Exception):
    """Reject a first-pass review operation outside the fixed safe interface."""


@dataclass(frozen=True, slots=True)
class ReviewAssetIdentity:
    """Internal byte identity; this object must never reach the reviewer callback."""

    asset_id: str
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class BlankObservationForm:
    """Reviewer-visible blank form with no method or result metadata."""

    review_status: None = None
    framing_or_crop_difference_visible: None = None
    pose_or_registration_offset_visible: None = None
    global_exposure_difference_visible: None = None
    blur_visible: None = None
    localized_change_visible: None = None
    diffuse_change_visible: None = None
    no_obvious_visible_change: None = None
    visible_observation: str = ""
    inferred_cause: None = None


@dataclass(frozen=True, slots=True)
class BlindReviewItem:
    """The complete positive allowlist passed to one reviewer invocation."""

    asset_id: str
    pixels: NDArray[np.uint8]
    form: BlankObservationForm


Reviewer = Callable[[BlindReviewItem], Mapping[str, Any]]
ImageLoader = Callable[[ReviewAssetIdentity], NDArray[np.uint8]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V0_3BlindedReviewError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_sha256(value: object, *, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"invalid {label}",
    )
    return value


def validate_review_assets(
    assets: Sequence[ReviewAssetIdentity],
    *,
    schema: Mapping[str, Any],
) -> tuple[ReviewAssetIdentity, ...]:
    """Validate the exact fixed 28-row first-pass identity inventory."""
    _require(len(assets) == 28, "review inventory must contain exactly 28 assets")
    observed_ids = [asset.asset_id for asset in assets]
    _require(observed_ids == list(EXPECTED_ASSET_IDS), "review asset order or membership changed")
    _require(len(observed_ids) == len(set(observed_ids)), "duplicate review asset")
    for asset in assets:
        _require(asset.byte_count > 0, "review asset byte count must be positive")
        _validate_sha256(asset.sha256, label="review asset SHA-256")
        record = {
            "contract_version": ARTIFACT_CONTRACT_VERSION,
            "diagnostic_id": DIAGNOSTIC_ID,
            "asset_id": asset.asset_id,
            "byte_count": asset.byte_count,
            "sha256": asset.sha256,
        }
        try:
            validate_tabular_record("review_asset", record, schema=schema)
        except V0_3DiagnosticContractError as error:
            raise V0_3BlindedReviewError("review asset violates the fixed contract") from error
    return tuple(assets)


def read_review_assets_csv(
    path: Path,
    *,
    expected_sha256: str,
    schema: Mapping[str, Any],
) -> tuple[ReviewAssetIdentity, ...]:
    """Read only the safe review inventory; no path or method metadata is accepted."""
    _require(path.is_file() and not path.is_symlink(), "review inventory must be a regular file")
    _require(sha256_file(path) == expected_sha256, "review inventory SHA-256 changed")
    expected_header = (
        "contract_version",
        "diagnostic_id",
        "asset_id",
        "byte_count",
        "sha256",
    )
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            _require(tuple(reader.fieldnames or ()) == expected_header, "review header changed")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise V0_3BlindedReviewError("cannot read review inventory") from error
    assets: list[ReviewAssetIdentity] = []
    for row in rows:
        _require(set(row) == set(expected_header), "review inventory contains an extra field")
        _require(row["contract_version"] == ARTIFACT_CONTRACT_VERSION, "review version changed")
        _require(row["diagnostic_id"] == DIAGNOSTIC_ID, "review diagnostic ID changed")
        try:
            byte_count = int(row["byte_count"])
        except ValueError as error:
            raise V0_3BlindedReviewError("review byte count is not an integer") from error
        assets.append(
            ReviewAssetIdentity(
                asset_id=row["asset_id"],
                byte_count=byte_count,
                sha256=row["sha256"],
            )
        )
    return validate_review_assets(assets, schema=schema)


def decode_verified_opaque_asset(
    identity: ReviewAssetIdentity,
    *,
    assets_root: Path,
) -> NDArray[np.uint8]:
    """Verify and decode one authorized opaque asset without exposing its path."""
    _require(
        identity.asset_id in EXPECTED_ASSET_IDS, "asset is not authorized for first-pass review"
    )
    _require(assets_root.is_dir() and not assets_root.is_symlink(), "asset root is invalid")
    path = assets_root / f"{identity.asset_id}.jpg"
    try:
        metadata = path.lstat()
    except OSError as error:
        raise V0_3BlindedReviewError("authorized opaque asset is unavailable") from error
    _require(stat.S_ISREG(metadata.st_mode), "opaque asset must be a regular non-symlink file")
    _require(not path.is_symlink(), "opaque asset symlink is forbidden")
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise V0_3BlindedReviewError("cannot read authorized opaque asset") from error
    _require(len(encoded) == identity.byte_count, "opaque asset byte count changed")
    _require(_sha256_bytes(encoded) == identity.sha256, "opaque asset SHA-256 changed")
    buffer = np.frombuffer(encoded, dtype=np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    _require(decoded is not None and decoded.size > 0, "opaque asset decode failed")
    _require(decoded.dtype == np.uint8, "decoded asset dtype changed")
    _require(decoded.ndim == 3 and decoded.shape[2] == 3, "decoded asset shape is invalid")
    _require(decoded.flags.c_contiguous, "decoded asset must be C-contiguous")
    immutable_bytes = decoded.tobytes(order="C")
    pixels = np.frombuffer(immutable_bytes, dtype=np.uint8).reshape(decoded.shape)
    _require(not pixels.flags.writeable, "review pixels must be read-only")
    return pixels


def _form_record(
    asset_id: str,
    value: Mapping[str, Any],
    *,
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    _require(set(value) == set(FORM_FIELDS), "reviewer response fields differ from the blank form")
    record = {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "asset_id": asset_id,
        **value,
    }
    try:
        return validate_tabular_record("blind_observation", record, schema=schema)
    except V0_3DiagnosticContractError as error:
        raise V0_3BlindedReviewError("reviewer response violates the blind schema") from error


def collect_blind_observations(
    assets: Sequence[ReviewAssetIdentity],
    *,
    image_loader: ImageLoader,
    reviewer: Reviewer,
    schema: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Collect exactly one ordered observation through the positive safe view model."""
    fixed_assets = validate_review_assets(assets, schema=schema)
    observed: list[dict[str, Any]] = []
    for identity in fixed_assets:
        pixels = image_loader(identity)
        _require(isinstance(pixels, np.ndarray), "image loader must return a NumPy array")
        _require(
            pixels.dtype == np.uint8
            and pixels.ndim == 3
            and pixels.shape[2] == 3
            and pixels.size > 0
            and pixels.flags.c_contiguous,
            "review pixels are invalid",
        )
        _require(not pixels.flags.writeable, "review pixels must be read-only")
        item = BlindReviewItem(
            asset_id=identity.asset_id,
            pixels=pixels,
            form=BlankObservationForm(),
        )
        response = reviewer(item)
        _require(isinstance(response, Mapping), "reviewer response must be a mapping")
        observed.append(_form_record(identity.asset_id, response, schema=schema))
    try:
        return validate_blind_observation_records(
            observed,
            expected_asset_ids=EXPECTED_ASSET_IDS,
            schema=schema,
            require_complete=True,
        )
    except V0_3DiagnosticContractError as error:
        raise V0_3BlindedReviewError("blind observation sequence is invalid") from error


def serialize_blind_observations(records: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize fixed observations as deterministic UTF-8 CSV with LF endings."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(BLIND_OBSERVATION_FIELDS),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for record in records:
        writer.writerow(record)
    return stream.getvalue().encode("utf-8")


def write_blind_observations_csv(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    schema: Mapping[str, Any],
) -> str:
    """Validate and exclusively write one complete first-pass observation file."""
    try:
        validated = validate_blind_observation_records(
            records,
            expected_asset_ids=EXPECTED_ASSET_IDS,
            schema=schema,
            require_complete=True,
        )
    except V0_3DiagnosticContractError as error:
        raise V0_3BlindedReviewError("cannot write invalid blind observations") from error
    content = serialize_blind_observations(validated)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(content)
    except FileExistsError as error:
        raise V0_3BlindedReviewError("blind observation output already exists") from error
    return _sha256_bytes(content)


def read_blind_observations_csv(
    path: Path,
    *,
    expected_sha256: str,
    schema: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Read and revalidate a complete immutable first-pass observation file."""
    _require(path.is_file() and not path.is_symlink(), "blind observation file is invalid")
    _require(sha256_file(path) == expected_sha256, "blind observation SHA-256 changed")
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            _require(
                tuple(reader.fieldnames or ()) == BLIND_OBSERVATION_FIELDS, "blind header changed"
            )
            records: list[dict[str, Any]] = []
            for row in reader:
                _require(set(row) == set(BLIND_OBSERVATION_FIELDS), "blind row fields changed")
                row["inferred_cause"] = row["inferred_cause"] or None
                records.append(row)
    except (OSError, UnicodeError, csv.Error) as error:
        raise V0_3BlindedReviewError("cannot read blind observations") from error
    try:
        return validate_blind_observation_records(
            records,
            expected_asset_ids=EXPECTED_ASSET_IDS,
            schema=schema,
            require_complete=True,
        )
    except V0_3DiagnosticContractError as error:
        raise V0_3BlindedReviewError("stored blind observations are invalid") from error


def build_review_completion_checkpoint(
    *,
    review_assets_sha256: str,
    blind_observations_sha256: str,
) -> dict[str, Any]:
    """Hash-lock the complete first pass before any method metadata join."""
    _validate_sha256(review_assets_sha256, label="review inventory SHA-256")
    _validate_sha256(blind_observations_sha256, label="blind observation SHA-256")
    return {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "review_assets_sha256": review_assets_sha256,
        "blind_observations_sha256": blind_observations_sha256,
        "expected_asset_count": 28,
        "attempted_asset_count": 28,
        "unique_asset_count": 28,
        "asset_order": "asset_id_ascending",
        "first_pass_complete": True,
        "protected_metadata_exposed": False,
        "method_metadata_joined": False,
    }


def validate_review_completion_checkpoint(value: object) -> dict[str, Any]:
    """Validate the exact closed-before-join review completion boundary."""
    _require(isinstance(value, dict), "review completion checkpoint must be an object")
    expected_keys = {
        "contract_version",
        "diagnostic_id",
        "review_assets_sha256",
        "blind_observations_sha256",
        "expected_asset_count",
        "attempted_asset_count",
        "unique_asset_count",
        "asset_order",
        "first_pass_complete",
        "protected_metadata_exposed",
        "method_metadata_joined",
    }
    _require(set(value) == expected_keys, "review completion checkpoint fields changed")
    _require(value["contract_version"] == ARTIFACT_CONTRACT_VERSION, "checkpoint version changed")
    _require(value["diagnostic_id"] == DIAGNOSTIC_ID, "checkpoint diagnostic ID changed")
    _validate_sha256(value["review_assets_sha256"], label="review inventory SHA-256")
    _validate_sha256(value["blind_observations_sha256"], label="blind observation SHA-256")
    _require(
        value["expected_asset_count"]
        == value["attempted_asset_count"]
        == value["unique_asset_count"]
        == 28,
        "review completion count changed",
    )
    _require(value["asset_order"] == "asset_id_ascending", "review order changed")
    _require(value["first_pass_complete"] is True, "first pass is incomplete")
    _require(value["protected_metadata_exposed"] is False, "protected metadata was exposed")
    _require(value["method_metadata_joined"] is False, "method metadata was joined too early")
    return value


def serialize_review_completion_checkpoint(value: Mapping[str, Any]) -> bytes:
    """Serialize a validated completion checkpoint deterministically."""
    validated = validate_review_completion_checkpoint(dict(value))
    return (
        json.dumps(validated, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def reviewer_visible_field_names() -> tuple[str, ...]:
    """Expose the safe view-model field names for audit tests."""
    return tuple(field.name for field in fields(BlindReviewItem))
