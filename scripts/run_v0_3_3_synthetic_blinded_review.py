from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np

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
    reviewer_visible_field_names,
    serialize_blind_observations,
    serialize_review_completion_checkpoint,
    validate_review_completion_checkpoint,
    write_blind_observations_csv,
)
from few_shot_anomaly_poc.v0_3_diagnostic_contract import (
    EXPECTED_ASSET_IDS,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_SCHEMA_SHA256,
    load_v0_3_schema,
    sha256_file,
)

REPORT_VERSION = "v0.3.3-synthetic-blinded-review-verification-v1"
PROTECTED_SENTINEL = "PROTECTED-SENTINEL-MUST-NOT-LEAK"


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _expect_rejection(operation: Callable[[], object]) -> bool:
    try:
        operation()
    except V0_3BlindedReviewError:
        return True
    return False


def _response(asset_id: str) -> dict:
    return {
        "review_status": "reviewed",
        "framing_or_crop_difference_visible": "no",
        "pose_or_registration_offset_visible": "no",
        "global_exposure_difference_visible": "uncertain",
        "blur_visible": "no",
        "localized_change_visible": "yes",
        "diffuse_change_visible": "no",
        "no_obvious_visible_change": "no",
        "visible_observation": f"Synthetic pattern for {asset_id} is visible.",
        "inferred_cause": None,
    }


def _artifact_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        return {}
    return {path.name: sha256_file(path) for path in sorted(root.iterdir()) if path.is_file()}


def _run_synthetic_verification(repository_root: Path, source_commit: str) -> dict:
    schema = load_v0_3_schema(repository_root / "schemas/v0.3/diagnostic-artifacts.json")
    real_root = repository_root / "artifacts/v0.3/diagnostics/pcb2-development"
    real_before = _artifact_hashes(real_root)
    protected_metadata = {
        "method": PROTECTED_SENTINEL,
        "case_type": PROTECTED_SENTINEL,
        "class_label": PROTECTED_SENTINEL,
        "score": PROTECTED_SENTINEL,
        "semantic_path": PROTECTED_SENTINEL,
    }
    del protected_metadata

    with tempfile.TemporaryDirectory(prefix="v0-3-3-synthetic-") as temporary_name:
        temporary = Path(temporary_name)
        assets_root = temporary / "assets"
        assets_root.mkdir()
        expected_pixels: dict[str, np.ndarray] = {}
        identities: list[ReviewAssetIdentity] = []
        for index, asset_id in enumerate(EXPECTED_ASSET_IDS):
            height = 7 + index % 3
            width = 9 + index % 5
            pixels = np.zeros((height, width, 3), dtype=np.uint8)
            pixels[:, :, 0] = (index * 7) % 256
            pixels[:, :, 1] = np.arange(width, dtype=np.uint8)
            pixels[:, :, 2] = np.arange(height, dtype=np.uint8)[:, None]
            success, encoded = cv2.imencode(".png", pixels)
            if not success:
                raise V0_3BlindedReviewError("synthetic PNG encode failed")
            payload = encoded.tobytes()
            (assets_root / f"{asset_id}.jpg").write_bytes(payload)
            expected_pixels[asset_id] = pixels
            identities.append(ReviewAssetIdentity(asset_id, len(payload), _sha256_bytes(payload)))

        loader_calls: list[str] = []
        reviewer_calls: list[str] = []
        visible_text: list[str] = []

        def loader(identity: ReviewAssetIdentity) -> np.ndarray:
            loader_calls.append(identity.asset_id)
            return decode_verified_opaque_asset(identity, assets_root=assets_root)

        def reviewer(item: BlindReviewItem) -> dict:
            reviewer_calls.append(item.asset_id)
            visible_text.append(repr((item.asset_id, item.form)))
            if item.pixels.flags.writeable:
                raise V0_3BlindedReviewError("synthetic reviewer received mutable pixels")
            if not np.array_equal(item.pixels, expected_pixels[item.asset_id]):
                raise V0_3BlindedReviewError("synthetic decoded extent or value changed")
            if hasattr(item, "method") or hasattr(item, "sha256") or hasattr(item, "path"):
                raise V0_3BlindedReviewError("protected metadata entered the safe view")
            return _response(item.asset_id)

        records = collect_blind_observations(
            identities,
            image_loader=loader,
            reviewer=reviewer,
            schema=schema,
        )
        first_serialization = serialize_blind_observations(records)
        second_serialization = serialize_blind_observations(records)
        first_csv = temporary / "run-one.csv"
        second_csv = temporary / "run-two.csv"
        first_sha256 = write_blind_observations_csv(first_csv, records, schema=schema)
        second_sha256 = write_blind_observations_csv(second_csv, records, schema=schema)
        round_trip = read_blind_observations_csv(
            first_csv,
            expected_sha256=first_sha256,
            schema=schema,
        )
        completion = build_review_completion_checkpoint(
            review_assets_sha256=(
                "399ebef79498917ff0cf5b2bf83467541a5417dce74bdd8e0e462fec70a7fd58"
            ),
            blind_observations_sha256=first_sha256,
        )
        validate_review_completion_checkpoint(completion)
        completion_bytes = serialize_review_completion_checkpoint(completion)

        injected_response = {**_response(EXPECTED_ASSET_IDS[0]), "method": PROTECTED_SENTINEL}
        protected_injection_rejected = _expect_rejection(
            lambda: collect_blind_observations(
                identities,
                image_loader=loader,
                reviewer=lambda _item: injected_response,
                schema=schema,
            )
        )
        missing_inventory_rejected = _expect_rejection(
            lambda: collect_blind_observations(
                identities[:-1],
                image_loader=loader,
                reviewer=lambda item: _response(item.asset_id),
                schema=schema,
            )
        )
        reordered_inventory_rejected = _expect_rejection(
            lambda: collect_blind_observations(
                list(reversed(identities)),
                image_loader=loader,
                reviewer=lambda item: _response(item.asset_id),
                schema=schema,
            )
        )
        duplicated_identities = [*identities]
        duplicated_identities[-1] = duplicated_identities[0]
        duplicate_inventory_rejected = _expect_rejection(
            lambda: collect_blind_observations(
                duplicated_identities,
                image_loader=loader,
                reviewer=lambda item: _response(item.asset_id),
                schema=schema,
            )
        )
        wrong_identity = ReviewAssetIdentity(
            EXPECTED_ASSET_IDS[0], identities[0].byte_count, "0" * 64
        )
        hash_mismatch_rejected = _expect_rejection(
            lambda: decode_verified_opaque_asset(wrong_identity, assets_root=assets_root)
        )
        unselected_asset_rejected = _expect_rejection(
            lambda: decode_verified_opaque_asset(
                ReviewAssetIdentity("asset-999999", 1, "0" * 64),
                assets_root=assets_root,
            )
        )
        overwrite_rejected = _expect_rejection(
            lambda: write_blind_observations_csv(first_csv, records, schema=schema)
        )
        symlink_root = temporary / "symlink-assets"
        symlink_root.mkdir()
        symlink_target = temporary / "symlink-target.jpg"
        symlink_target.write_bytes(b"synthetic-symlink-target")
        (symlink_root / f"{EXPECTED_ASSET_IDS[0]}.jpg").symlink_to(symlink_target)
        symlink_rejected = _expect_rejection(
            lambda: decode_verified_opaque_asset(
                ReviewAssetIdentity(
                    EXPECTED_ASSET_IDS[0],
                    len(b"synthetic-symlink-target"),
                    _sha256_bytes(b"synthetic-symlink-target"),
                ),
                assets_root=symlink_root,
            )
        )

        checks = {
            "all_28_assets_processed_once": (
                loader_calls[:28] == list(EXPECTED_ASSET_IDS)
                and reviewer_calls == list(EXPECTED_ASSET_IDS)
                and Counter(reviewer_calls) == Counter(EXPECTED_ASSET_IDS)
            ),
            "asset_order_fixed": [record["asset_id"] for record in records]
            == list(EXPECTED_ASSET_IDS),
            "completion_checkpoint_closed_before_join": (
                completion["first_pass_complete"] is True
                and completion["protected_metadata_exposed"] is False
                and completion["method_metadata_joined"] is False
            ),
            "csv_round_trip_exact": round_trip == records,
            "deterministic_csv_bytes": (
                first_serialization
                == second_serialization
                == first_csv.read_bytes()
                == second_csv.read_bytes()
                and first_sha256 == second_sha256
            ),
            "duplicate_inventory_rejected": duplicate_inventory_rejected,
            "fixed_form_fields_only": tuple(
                field.name for field in fields(BlankObservationForm)
            )
            == FORM_FIELDS,
            "hash_mismatch_rejected": hash_mismatch_rejected,
            "missing_inventory_rejected": missing_inventory_rejected,
            "original_extent_and_values_preserved": True,
            "output_overwrite_rejected": overwrite_rejected,
            "pixels_read_only": True,
            "protected_field_injection_rejected": protected_injection_rejected,
            "protected_sentinel_absent": (
                PROTECTED_SENTINEL not in "".join(visible_text)
                and PROTECTED_SENTINEL not in first_serialization.decode("utf-8")
            ),
            "safe_view_fields_only": reviewer_visible_field_names()
            == ("asset_id", "pixels", "form"),
            "symlink_rejected": symlink_rejected,
            "reordered_inventory_rejected": reordered_inventory_rejected,
            "unselected_asset_rejected": unselected_asset_rejected,
        }
        real_after = _artifact_hashes(real_root)

    report = {
        "report_version": REPORT_VERSION,
        "source_commit": source_commit,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "schema_sha256": EXPECTED_SCHEMA_SHA256,
        "synthetic_fixture_only": True,
        "synthetic_asset_count": 28,
        "synthetic_observation_count": 28,
        "synthetic_observation_csv_sha256": first_sha256,
        "synthetic_review_completion_checkpoint_sha256": _sha256_bytes(completion_bytes),
        "checks": checks,
        "all_checks_passed": all(checks.values()) and real_before == real_after,
        "boundary": {
            "v0_3_selected_image_accessed": False,
            "visa_image_accessed": False,
            "synthetic_image_decoded": True,
            "image_displayed": False,
            "anomaly_scorer_executed": False,
            "diagnostic_score_computed": False,
            "real_observation_written": False,
            "real_diagnostic_artifacts_modified": real_before != real_after,
            "raw_or_derived_image_committed": False,
        },
        "dependencies": {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
    }
    if PROTECTED_SENTINEL in json.dumps(report, sort_keys=True):
        raise V0_3BlindedReviewError("protected sentinel entered the verification report")
    if not report["all_checks_passed"]:
        raise V0_3BlindedReviewError("synthetic blinded-review verification failed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the v0.3.3 synthetic-only blinded-review verification."
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]

    try:
        if _git(repository_root, "status", "--porcelain"):
            raise V0_3BlindedReviewError("worktree must be clean before synthetic verification")
        source_commit = _git(repository_root, "rev-parse", "HEAD")
        if source_commit != _git(repository_root, "rev-parse", "origin/main"):
            raise V0_3BlindedReviewError("HEAD must equal pushed origin/main")
        report = _run_synthetic_verification(repository_root, source_commit)
        output = arguments.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        content = (
            json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        with output.open("xb") as stream:
            stream.write(content)
    except (
        FileExistsError,
        OSError,
        subprocess.CalledProcessError,
        V0_3BlindedReviewError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print("v0.3.3 synthetic blinded-review verification passed")
    print(f"source_commit={source_commit}")
    print("visa_image_accessed=false")
    print("anomaly_scorer_executed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
