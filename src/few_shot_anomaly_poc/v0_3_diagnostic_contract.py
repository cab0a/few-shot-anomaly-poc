"""Machine-readable contract for the preregistered v0.3 diagnostic study."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

CONFIG_SCHEMA_VERSION = "v0.3-diagnostic-contract-v1"
ARTIFACT_CONTRACT_VERSION = "diagnostic-artifacts/v0.3"
DIAGNOSTIC_ID = "v0.3.0-pcb2-development-diagnostic-1"
PREREGISTRATION_COMMIT = "88fe9a04f990be06c478599867e1c2f704f3c19c"
PREREGISTRATION_DOCUMENT_SHA256 = (
    "2165851d9e0f1f8bbdfb81774a61ea3079a55730e04b938c203df3016c9bba93"
)
EXPECTED_CONFIG_SHA256 = "e3108c837b3c9818b25b86594a7b103091e145f8ee6392e8424c7163f220f548"
EXPECTED_SCHEMA_SHA256 = "173018d554380910147a826a42364dfc8499f0cb6ec6ff76dc1e24f233d87834"
METHODS = ("ecc_residual", "patch_hog_ocsvm", "dinov2_vits14_224_nn")
CONDITIONS = (
    "control",
    "brightness_085",
    "brightness_115",
    "gaussian_blur",
    "translate_4_4",
)
REVIEW_STATUS_VALUES = frozenset({"reviewed", "unreadable"})
VISIBILITY_VALUES = frozenset({"yes", "no", "uncertain"})
BLIND_OBSERVATION_FIELDS = (
    "contract_version",
    "diagnostic_id",
    "asset_id",
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
EXPECTED_ASSET_IDS = (
    "asset-000003",
    "asset-000027",
    "asset-000044",
    "asset-000046",
    "asset-000047",
    "asset-000055",
    "asset-000062",
    "asset-000078",
    "asset-000087",
    "asset-000090",
    "asset-000093",
    "asset-000095",
    "asset-000102",
    "asset-000104",
    "asset-000106",
    "asset-000111",
    "asset-000112",
    "asset-000117",
    "asset-000141",
    "asset-000143",
    "asset-000151",
    "asset-000155",
    "asset-000156",
    "asset-000168",
    "asset-000169",
    "asset-000171",
    "asset-000172",
    "asset-000174",
)
EXPECTED_METHOD_CASES = (
    ("ecc_residual", "false_positive", 1, "asset-000141"),
    ("ecc_residual", "false_positive", 2, "asset-000090"),
    ("ecc_residual", "false_positive", 3, "asset-000169"),
    ("ecc_residual", "false_positive", 4, "asset-000046"),
    ("ecc_residual", "false_negative", 1, "asset-000095"),
    ("ecc_residual", "false_negative", 2, "asset-000117"),
    ("ecc_residual", "false_negative", 3, "asset-000027"),
    ("ecc_residual", "false_negative", 4, "asset-000112"),
    ("ecc_residual", "false_negative", 5, "asset-000087"),
    ("patch_hog_ocsvm", "false_positive", 1, "asset-000062"),
    ("patch_hog_ocsvm", "false_positive", 2, "asset-000171"),
    ("patch_hog_ocsvm", "false_positive", 3, "asset-000172"),
    ("patch_hog_ocsvm", "false_positive", 4, "asset-000106"),
    ("patch_hog_ocsvm", "false_positive", 5, "asset-000143"),
    ("patch_hog_ocsvm", "false_negative", 1, "asset-000044"),
    ("patch_hog_ocsvm", "false_negative", 2, "asset-000155"),
    ("patch_hog_ocsvm", "false_negative", 3, "asset-000102"),
    ("patch_hog_ocsvm", "false_negative", 4, "asset-000111"),
    ("patch_hog_ocsvm", "false_negative", 5, "asset-000003"),
    ("dinov2_vits14_224_nn", "false_positive", 1, "asset-000078"),
    ("dinov2_vits14_224_nn", "false_positive", 2, "asset-000093"),
    ("dinov2_vits14_224_nn", "false_positive", 3, "asset-000055"),
    ("dinov2_vits14_224_nn", "false_positive", 4, "asset-000047"),
    ("dinov2_vits14_224_nn", "false_positive", 5, "asset-000104"),
    ("dinov2_vits14_224_nn", "false_negative", 1, "asset-000112"),
    ("dinov2_vits14_224_nn", "false_negative", 2, "asset-000151"),
    ("dinov2_vits14_224_nn", "false_negative", 3, "asset-000156"),
    ("dinov2_vits14_224_nn", "false_negative", 4, "asset-000174"),
    ("dinov2_vits14_224_nn", "false_negative", 5, "asset-000168"),
)
PROTECTED_FIRST_PASS_FIELD_ORDER = (
    "method",
    "case_type",
    "true_class",
    "predicted_class",
    "anomaly_score",
    "threshold",
    "score_margin",
    "rank",
    "source_path",
    "other_method_selection",
)
PROTECTED_FIRST_PASS_FIELDS = frozenset(PROTECTED_FIRST_PASS_FIELD_ORDER)
ASSET_ID_PATTERN = re.compile(r"^asset-[0-9]{6}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class V0_3DiagnosticContractError(Exception):
    """Reject state that differs from the preregistered diagnostic contract."""


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 identity of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V0_3DiagnosticContractError(message)


def _object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise V0_3DiagnosticContractError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise V0_3DiagnosticContractError(f"{label} fields differ from the contract")


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V0_3DiagnosticContractError(f"cannot load {label}") from error
    return _object(value, label=label)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _is_commit(value: object) -> bool:
    return isinstance(value, str) and COMMIT_PATTERN.fullmatch(value) is not None


def _relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise V0_3DiagnosticContractError(f"{label} must be a POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise V0_3DiagnosticContractError(f"{label} must use canonical relative POSIX syntax")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise V0_3DiagnosticContractError(f"{label} escapes its declared root")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise V0_3DiagnosticContractError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise V0_3DiagnosticContractError(f"{label} must be a finite number")
    return float(value)


def validate_v0_3_config(value: object) -> dict[str, Any]:
    """Validate the identities, access boundary, and fixed diagnostic rules."""
    config = _object(value, label="v0.3 config")
    _exact_keys(
        config,
        {
            "access_boundary",
            "artifact_root",
            "conditions",
            "dataset",
            "decision_precedence",
            "dependencies",
            "decision_values",
            "diagnostic_id",
            "method_order",
            "methods",
            "normal_probe",
            "parent_evidence",
            "preregistration",
            "probe_record_count",
            "review",
            "schema_version",
            "serialization",
            "signal_rule",
            "summaries",
        },
        label="v0.3 config",
    )
    _require(config["schema_version"] == CONFIG_SCHEMA_VERSION, "config version changed")
    _require(config["diagnostic_id"] == DIAGNOSTIC_ID, "diagnostic identity changed")

    prereg = _object(config["preregistration"], label="preregistration")
    _require(
        prereg
        == {
            "commit": PREREGISTRATION_COMMIT,
            "document": "docs/v0.3-development-diagnostic-preregistration.md",
            "document_sha256": PREREGISTRATION_DOCUMENT_SHA256,
        },
        "preregistration identity changed",
    )
    parent = _object(config["parent_evidence"], label="parent_evidence")
    expected_parent = {
        "artifact_manifest_sha256": (
            "dbf7bfc1d2429bc391720136ba76711fa0493eb7407487e185bf7d12c4b85d19"
        ),
        "project_decision_sha256": (
            "aad34c5e3f0c085cd0f218e61d504df9539ddc68764694cc3d53ed0f4ec729e2"
        ),
        "v0_2_freeze_source_commit": "462738321da403ce1f77ef23b0cb823008c32694",
        "v0_2_config_sha256": (
            "9ea3a7156aeb3c6efc87c8ae3811444421bd3e86ea25b7ac014893c7e5892265"
        ),
        "v0_2_schema_sha256": (
            "4178d24f7210f8f859f6c99386204022ac3b2e4ffe4b7b63cdfea00d0c79f31d"
        ),
        "boundary_record_sha256": (
            "e122bfa51ce618e0588a580f2cf66447c44a2cf801f08f851cce9d5271a4c698"
        ),
        "opaque_scoring_manifest_sha256": (
            "32ea52ed1b9872f39ae27f5d58a353ea84b8b143642e3a7f0fabe940184705e8"
        ),
        "normal_manifest_set_sha256": (
            "bb6b633d7c7645f767276f47ec74f709dd20afc83e0f444c4df3579418185e4b"
        ),
        "failure_case_sha256": {
            "ecc_residual": (
                "1bd73cf3b0050b3eff934467625ba118d662e442ceaa5829f8a47344b4f390e8"
            ),
            "patch_hog_ocsvm": (
                "7edde3c32fd23e9d53db0c665595bb3c6789a1ce52bf039810b9d7646d3387ff"
            ),
            "dinov2_vits14_224_nn": (
                "ad8f05152f936b8b9da5a6d94f2f8dc9eab43035373836f949dbb3a7887c22e9"
            ),
        },
        "normal_calibration_manifest_sha256": (
            "77d5adb588e7d463e7fcab1c10b841b9ad23d827b51138c31f42dac35bd99ca3"
        ),
        "normal_reference_manifest_sha256": (
            "e587f1808262480261ae8a7b940faff0d9ef5f83cf215028b31490ba48369b99"
        ),
    }
    _require(parent == expected_parent, "parent evidence identity changed")

    dataset = _object(config["dataset"], label="dataset")
    _require(
        dataset
        == {
            "name": "VisA",
            "category": "pcb2",
            "license": "CC BY 4.0",
            "raw_data_in_git": False,
            "derived_images_in_git": False,
        },
        "dataset boundary changed",
    )
    dependencies = _object(config["dependencies"], label="dependencies")
    _require(
        dependencies
        == {
            "python": "3.13.14",
            "numpy": "2.5.1",
            "opencv_python_headless": "4.13.0.92",
            "torch_cpu": "2.13.0+cpu",
            "root_lock_sha256": (
                "770d3655ea925007f33d5a22988943d09184b222c7cb7e886fd73623bf9039d4"
            ),
            "dinov2_lock_sha256": (
                "28a0878f3425bf6ea6fcc1631b168d67e6ddcc747d71f9c92c85b3aa9c706ec8"
            ),
        },
        "dependency identities changed",
    )

    review = _object(config["review"], label="review")
    _exact_keys(
        review,
        {
            "asset_id_pattern",
            "method_case_count",
            "method_cases",
            "note_character_limit",
            "order",
            "protected_fields",
            "review_status_values",
            "unique_asset_count",
            "unique_asset_ids",
            "visibility_values",
            "visible_fields",
        },
        label="review",
    )
    _require(review.get("asset_id_pattern") == ASSET_ID_PATTERN.pattern, "asset pattern changed")
    asset_ids = review.get("unique_asset_ids")
    _require(isinstance(asset_ids, list), "review asset IDs must be a list")
    _require(asset_ids == sorted(asset_ids), "review asset IDs must be in ascending order")
    _require(len(asset_ids) == 28 == review.get("unique_asset_count"), "review count changed")
    _require(tuple(asset_ids) == EXPECTED_ASSET_IDS, "review asset membership changed")
    _require(len(set(asset_ids)) == 28, "review asset IDs must be unique")
    _require(
        all(
            isinstance(asset_id, str) and ASSET_ID_PATTERN.fullmatch(asset_id)
            for asset_id in asset_ids
        ),
        "review asset ID is invalid",
    )
    cases = review.get("method_cases")
    _require(isinstance(cases, list) and len(cases) == 29, "method-case count changed")
    _require(review.get("method_case_count") == 29, "declared method-case count changed")
    seen_case_keys: set[tuple[str, str, int]] = set()
    case_asset_ids: set[str] = set()
    for index, case in enumerate(cases):
        item = _object(case, label=f"review.method_cases[{index}]")
        _exact_keys(item, {"method", "case_type", "rank", "asset_id"}, label="method case")
        _require(item["method"] in METHODS, "method case has an unknown method")
        _require(item["case_type"] in {"false_positive", "false_negative"}, "case type changed")
        rank = _integer(item["rank"], label="case rank", minimum=1)
        _require(item["asset_id"] in asset_ids, "method case references an unknown asset")
        key = (item["method"], item["case_type"], rank)
        _require(key not in seen_case_keys, "duplicate method-case key")
        seen_case_keys.add(key)
        case_asset_ids.add(item["asset_id"])
    observed_cases = tuple(
        (item["method"], item["case_type"], item["rank"], item["asset_id"]) for item in cases
    )
    _require(observed_cases == EXPECTED_METHOD_CASES, "fixed method-case associations changed")
    _require(case_asset_ids == set(asset_ids), "method cases do not cover the fixed asset set")
    _require(review.get("order") == "asset_id_ascending", "review order changed")
    _require(
        review.get("visible_fields") == ["asset_id", "image_pixels", "observation_form"],
        "visible review fields changed",
    )
    _require(
        review.get("protected_fields") == list(PROTECTED_FIRST_PASS_FIELD_ORDER),
        "protected review fields changed",
    )
    _require(
        review.get("review_status_values") == ["reviewed", "unreadable"],
        "review status changed",
    )
    _require(
        review.get("visibility_values") == ["yes", "no", "uncertain"],
        "visibility values changed",
    )
    _require(review.get("note_character_limit") == 300, "review note limit changed")

    probe = _object(config["normal_probe"], label="normal_probe")
    _exact_keys(
        probe,
        {
            "anomaly_labels_allowed",
            "exclude_final_test",
            "exclude_reference",
            "seed",
            "selected_count",
            "selection_digest",
            "selection_prefix",
            "source_count",
            "source_partition",
            "tie_breaker",
        },
        label="normal_probe",
    )
    _require(
        probe.get("source_partition") == "v0.2_normal_calibration"
        and probe.get("source_count") == 881
        and probe.get("selected_count") == 60
        and probe.get("seed") == 42
        and probe.get("selection_prefix")
        == "few-shot-anomaly-poc:v0.3.0:pcb2:normal-diagnostic:42:"
        and probe.get("selection_digest") == "sha256_utf8_lowercase_hex"
        and probe.get("tie_breaker") == "posix_relative_path_ascending"
        and probe.get("exclude_reference") is True
        and probe.get("exclude_final_test") is True
        and probe.get("anomaly_labels_allowed") is False,
        "normal probe selection changed",
    )

    _require(config["method_order"] == list(METHODS), "method order changed")
    methods = _object(config["methods"], label="methods")
    _require(tuple(methods) == METHODS, "method inventory or order changed")
    expected_methods: dict[str, dict[str, Any]] = {
        "ecc_residual": {
            "fit_artifact_sha256": (
                "fee3b96824987a6d03f279f5fefc8cfe15a0c807050ec6a35700d4c2e567ae3d"
            ),
            "calibration_summary_sha256": (
                "e68f519abaa6f1f08a1376eec444cae5777736e3414b882ed93bf73f93345735"
            ),
            "fitted_state_sha256": (
                "f796dcef8fb7b6197f656c2a57800766c0398ac4842fd65f85372781063b800b"
            ),
            "threshold": 0.6219927985456926,
            "success_score_minimum": 0.0,
            "success_score_minimum_inclusive": True,
            "success_score_maximum": 1.0,
            "success_score_maximum_inclusive": True,
            "failure_score": 1.0,
        },
        "patch_hog_ocsvm": {
            "fit_artifact_sha256": (
                "d8da4c0704736e45495501618fcaa57d654e97acf4d3d10e12084e112e29b432"
            ),
            "calibration_summary_sha256": (
                "9e687e80437de5f38f848677ad9372104c1bc78275ae536a7c58f4b02bdee356"
            ),
            "fitted_state_sha256": (
                "ba7e1d47e8ff6fd7873edfce84c027c6ada37f6aab85d1b26cf92426463056f9"
            ),
            "threshold": 0.14960269900890077,
            "success_score_minimum": -1e12,
            "success_score_minimum_inclusive": False,
            "success_score_maximum": 1e12,
            "success_score_maximum_inclusive": False,
            "failure_score": 1e12,
        },
        "dinov2_vits14_224_nn": {
            "fit_artifact_sha256": (
                "ba2bc1b5d2b974cf6d700d775ac6c94a4b14d407e9e7fd6a8832a7b9877ef4e6"
            ),
            "calibration_summary_sha256": (
                "0a89bcb834f5e38a601bb956d9b099c613786187adbbd2ef8ba75226e2a75da7"
            ),
            "fitted_state_sha256": (
                "11ac0a0a4b0c082e2450fcf708e1c96a804ec738967aff392b727959ec425f8d"
            ),
            "threshold": 0.22208529710769653,
            "success_score_minimum": 0.0,
            "success_score_minimum_inclusive": True,
            "success_score_maximum": 2.0,
            "success_score_maximum_inclusive": True,
            "failure_score": 2.0,
        },
    }
    for method, expected_method in expected_methods.items():
        method_config = _object(methods[method], label=f"methods.{method}")
        _require(
            method_config == expected_method,
            f"{method} state or threshold changed",
        )

    conditions = config["conditions"]
    expected_conditions = [
        {"id": "control", "operation": "identity"},
        {
            "id": "brightness_085",
            "operation": "brightness",
            "factor": 0.85,
            "round": "numpy.rint",
            "clip": [0, 255],
            "dtype": "uint8",
        },
        {
            "id": "brightness_115",
            "operation": "brightness",
            "factor": 1.15,
            "round": "numpy.rint",
            "clip": [0, 255],
            "dtype": "uint8",
        },
        {
            "id": "gaussian_blur",
            "operation": "gaussian_blur",
            "kernel": [5, 5],
            "sigma": 1.0,
            "border": "BORDER_REFLECT_101",
        },
        {
            "id": "translate_4_4",
            "operation": "translate",
            "x_pixels": 4,
            "y_pixels": 4,
            "interpolation": "INTER_LINEAR",
            "border": "BORDER_REFLECT_101",
        },
    ]
    _require(conditions == expected_conditions, "transform definition changed")
    _require(config["probe_record_count"] == 900, "probe record count changed")

    summaries = _object(config["summaries"], label="summaries")
    _require(
        summaries
        == {
            "paired_statistics": [
                "attempted_count",
                "successful_count",
                "failed_count",
                "median_signed_delta",
                "median_absolute_delta",
                "nearest_rank_p95_absolute_delta",
                "above_threshold_count",
                "above_threshold_rate",
                "normal_to_anomalous_crossing_count",
                "anomalous_to_normal_crossing_count",
            ],
            "forbidden_metrics": [
                "auroc",
                "auprc",
                "recall",
                "precision",
                "accuracy",
                "confusion_matrix",
            ],
            "latency_measured": False,
        },
        "summary rules changed",
    )
    signal = _object(config["signal_rule"], label="signal_rule")
    _require(
        signal
        == {
            "minimum_visual_yes_unique_assets": 3,
            "minimum_normal_to_anomalous_crossings": 6,
            "minimum_median_absolute_delta_threshold_fraction": 0.10,
            "priority": ["framing_or_pose", "global_exposure", "blur"],
            "families": {
                "framing_or_pose": {
                    "visual_fields": [
                        "framing_or_crop_difference_visible",
                        "pose_or_registration_offset_visible",
                    ],
                    "conditions": ["translate_4_4"],
                },
                "global_exposure": {
                    "visual_fields": ["global_exposure_difference_visible"],
                    "conditions": ["brightness_085", "brightness_115"],
                },
                "blur": {
                    "visual_fields": ["blur_visible"],
                    "conditions": ["gaussian_blur"],
                },
            },
        },
        "diagnostic signal rule changed",
    )
    _require(
        config["decision_values"]
        == [
            "PROCEED TO ONE BOUNDED FOLLOW-UP PREREGISTRATION",
            "STOP: NO BOUNDED DIAGNOSTIC SIGNAL",
            "INCONCLUSIVE: REQUIRED EVIDENCE INCOMPLETE",
            "INVALID: BOUNDARY VIOLATION",
        ],
        "diagnostic decision values changed",
    )
    _require(
        config["decision_precedence"]
        == [
            "INVALID: BOUNDARY VIOLATION",
            "INCONCLUSIVE: REQUIRED EVIDENCE INCOMPLETE",
            "PROCEED TO ONE BOUNDED FOLLOW-UP PREREGISTRATION",
            "STOP: NO BOUNDED DIAGNOSTIC SIGNAL",
        ],
        "diagnostic decision precedence changed",
    )
    boundary = _object(config["access_boundary"], label="access_boundary")
    _require(
        boundary
        == {
            "v0_2_final_test_rescoring_allowed": False,
            "unselected_final_test_image_access_allowed": False,
            "selected_image_access_before_checkpoint_allowed": False,
            "diagnostic_scoring_before_checkpoint_allowed": False,
            "protected_metadata_exposure_allowed": False,
            "v0_2_artifact_modification_allowed": False,
            "overwrite_allowed": False,
            "timestamps_allowed": False,
        },
        "access boundary was relaxed",
    )
    _require(
        config["artifact_root"] == "artifacts/v0.3/diagnostics/pcb2-development",
        "artifact root changed",
    )
    _require(
        config["serialization"]
        == {
            "encoding": "UTF-8",
            "line_ending": "LF",
            "json_indent": 2,
            "json_sort_keys": True,
            "final_newline": True,
        },
        "serialization rules changed",
    )
    return config


def validate_v0_3_schema(value: object) -> dict[str, Any]:
    """Validate the declared diagnostic artifact inventory and blind fields."""
    schema = _object(value, label="v0.3 schema")
    _exact_keys(
        schema,
        {"common_rules", "contract_version", "contracts", "protected_first_pass_fields", "status"},
        label="v0.3 schema",
    )
    _require(schema["contract_version"] == ARTIFACT_CONTRACT_VERSION, "artifact version changed")
    _require(schema["status"] == "frozen_before_diagnostic_access", "schema status changed")
    common_rules = _object(schema["common_rules"], label="common_rules")
    _require(
        common_rules
        == {
            "additional_fields_allowed": False,
            "absolute_paths_forbidden": True,
            "duplicate_primary_keys_forbidden": True,
            "non_finite_numbers_forbidden": True,
            "overwrite_forbidden": True,
            "raw_or_derived_image_content_forbidden": True,
            "timestamps_forbidden": True,
        },
        "common artifact rules changed",
    )
    _require(
        schema["protected_first_pass_fields"] == list(PROTECTED_FIRST_PASS_FIELD_ORDER),
        "schema protected fields changed",
    )
    contracts = _object(schema["contracts"], label="contracts")
    expected = {
        "review_asset",
        "review_case_link",
        "normal_partition",
        "pre_access_checkpoint",
        "blind_observation",
        "review_completion_checkpoint",
        "observation_join",
        "diagnostic_score",
        "condition_summary",
        "diagnostic_decision",
        "bundle_manifest",
    }
    _require(set(contracts) == expected, "artifact contract inventory changed")
    csv_with_columns = {
        "review_asset",
        "review_case_link",
        "normal_partition",
        "blind_observation",
        "observation_join",
        "diagnostic_score",
        "condition_summary",
    }
    json_contracts = {
        "pre_access_checkpoint",
        "review_completion_checkpoint",
        "diagnostic_decision",
        "bundle_manifest",
    }
    for name in csv_with_columns:
        _exact_keys(
            contracts[name],
            {
                "columns",
                "format",
                "label_boundary",
                "path",
                "primary_key",
                "record_count",
                "sort_key",
            },
            label=f"contracts.{name}",
        )
        _require(contracts[name]["format"] == "csv", f"{name} format changed")
    for name in json_contracts:
        _exact_keys(
            contracts[name],
            {"format", "label_boundary", "path", "required_keys"},
            label=f"contracts.{name}",
        )
        _require(contracts[name]["format"] == "json", f"{name} format changed")
    expected_paths = {
        "review_asset": "review-assets.csv",
        "review_case_link": "review-case-linkage.csv",
        "normal_partition": "normal-diagnostic-partition.csv",
        "pre_access_checkpoint": "pre-access-checkpoint.json",
        "blind_observation": "blind-observations.csv",
        "review_completion_checkpoint": "review-completion-checkpoint.json",
        "observation_join": "observation-case-join.csv",
        "diagnostic_score": "controlled-scores.csv",
        "condition_summary": "condition-summaries.csv",
        "diagnostic_decision": "diagnostic-decision.json",
        "bundle_manifest": "artifact-manifest.json",
    }
    _require(
        all(contracts[name]["path"] == path for name, path in expected_paths.items()),
        "artifact path changed",
    )
    expected_boundaries = {
        "review_asset": "opaque_identity_only",
        "review_case_link": "post_review_join_only",
        "normal_partition": "known_normal_development_only",
        "pre_access_checkpoint": "no_diagnostic_access",
        "blind_observation": "first_pass_blinded",
        "review_completion_checkpoint": "completed_first_pass_before_join",
        "observation_join": "after_immutable_blind_observations",
        "diagnostic_score": "known_normal_development_only",
        "condition_summary": "known_normal_development_only",
        "diagnostic_decision": "development_only",
        "bundle_manifest": "development_only",
    }
    _require(
        all(
            contracts[name]["label_boundary"] == boundary
            for name, boundary in expected_boundaries.items()
        ),
        "artifact label boundary changed",
    )
    expected_keys = {
        "review_asset": (["asset_id"], ["asset_id"]),
        "review_case_link": (
            ["method", "case_type", "rank"],
            ["method_order", "case_type_order", "rank"],
        ),
        "normal_partition": (["relative_path"], ["selection_rank"]),
        "blind_observation": (["asset_id"], ["asset_id"]),
        "observation_join": (
            ["method", "case_type", "rank"],
            ["method_order", "case_type_order", "rank"],
        ),
        "diagnostic_score": (
            ["method", "selection_rank", "condition"],
            ["method_order", "selection_rank", "condition_order"],
        ),
        "condition_summary": (
            ["method", "condition"],
            ["method_order", "condition_order"],
        ),
    }
    for name, (primary_key, sort_key) in expected_keys.items():
        _require(contracts[name]["primary_key"] == primary_key, f"{name} primary key changed")
        _require(contracts[name]["sort_key"] == sort_key, f"{name} sort key changed")
    counts = {
        "review_asset": 28,
        "review_case_link": 29,
        "normal_partition": 60,
        "blind_observation": 28,
        "observation_join": 29,
        "diagnostic_score": 900,
        "condition_summary": 15,
    }
    for name, count in counts.items():
        contract = _object(contracts[name], label=f"contracts.{name}")
        _require(contract.get("record_count") == count, f"{name} count changed")
    expected_columns = {
        "review_asset": [
            "contract_version",
            "diagnostic_id",
            "asset_id",
            "byte_count",
            "sha256",
        ],
        "review_case_link": [
            "contract_version",
            "diagnostic_id",
            "method",
            "case_type",
            "rank",
            "asset_id",
            "parent_failure_sha256",
        ],
        "normal_partition": [
            "contract_version",
            "diagnostic_id",
            "selection_rank",
            "selection_sha256",
            "relative_path",
            "byte_count",
            "sha256",
        ],
        "blind_observation": list(BLIND_OBSERVATION_FIELDS),
        "observation_join": [
            "contract_version",
            "diagnostic_id",
            "method",
            "case_type",
            "rank",
            "asset_id",
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
        ],
        "diagnostic_score": [
            "contract_version",
            "diagnostic_id",
            "method",
            "selection_rank",
            "condition",
            "base_image_sha256",
            "score_status",
            "score_failure_code",
            "anomaly_score",
            "threshold",
            "above_threshold",
        ],
        "condition_summary": [
            "contract_version",
            "diagnostic_id",
            "method",
            "condition",
            "attempted_count",
            "successful_count",
            "failed_count",
            "median_signed_delta",
            "median_absolute_delta",
            "nearest_rank_p95_absolute_delta",
            "above_threshold_count",
            "above_threshold_rate",
            "normal_to_anomalous_crossing_count",
            "anomalous_to_normal_crossing_count",
        ],
    }
    for name, expected_names in expected_columns.items():
        columns = contracts[name].get("columns")
        _require(isinstance(columns, list), f"{name} columns must be a list")
        _require(
            [column.get("name") for column in columns] == expected_names,
            f"{name} columns changed",
        )
    expected_types = {
        "review_asset": ["string", "string", "asset_id", "integer", "sha256"],
        "review_case_link": [
            "string",
            "string",
            "method",
            "case_type",
            "integer",
            "asset_id",
            "sha256",
        ],
        "normal_partition": [
            "string",
            "string",
            "integer",
            "sha256",
            "relative_path",
            "integer",
            "sha256",
        ],
        "blind_observation": [
            "string",
            "string",
            "asset_id",
            "review_status",
            "visibility",
            "visibility",
            "visibility",
            "visibility",
            "visibility",
            "visibility",
            "visibility",
            "note_1_300",
            "note_1_300",
        ],
        "observation_join": [
            "string",
            "string",
            "method",
            "case_type",
            "integer",
            "asset_id",
            "review_status",
            "visibility",
            "visibility",
            "visibility",
            "visibility",
            "visibility",
            "visibility",
            "visibility",
            "note_1_300",
            "note_1_300",
        ],
        "diagnostic_score": [
            "string",
            "string",
            "method",
            "integer",
            "condition",
            "sha256",
            "score_status",
            "string",
            "number",
            "number",
            "boolean",
        ],
        "condition_summary": [
            "string",
            "string",
            "method",
            "condition",
            "integer",
            "integer",
            "integer",
            "number",
            "number",
            "number",
            "integer",
            "number",
            "integer",
            "integer",
        ],
    }
    for name, expected in expected_types.items():
        columns = contracts[name]["columns"]
        _require([column.get("type") for column in columns] == expected, f"{name} types changed")
    expected_nullable = {
        "review_asset": [False] * 5,
        "review_case_link": [False] * 7,
        "normal_partition": [False] * 7,
        "blind_observation": [False] * 12 + [True],
        "observation_join": [False] * 15 + [True],
        "diagnostic_score": [False] * 7 + [True] + [False] * 3,
        "condition_summary": [False] * 7 + [True] * 3 + [False] * 4,
    }
    for name, expected in expected_nullable.items():
        _require(
            [column.get("nullable") for column in contracts[name]["columns"]] == expected,
            f"{name} nullability changed",
        )
    checkpoint_keys = {
        "contract_version",
        "diagnostic_id",
        "source_commit",
        "config_sha256",
        "schema_sha256",
        "preregistration_commit",
        "preregistration_document_sha256",
        "parent_artifact_manifest_sha256",
        "opaque_scoring_manifest_sha256",
        "normal_calibration_manifest_sha256",
        "normal_reference_manifest_sha256",
        "ecc_failure_cases_sha256",
        "patch_hog_failure_cases_sha256",
        "dinov2_failure_cases_sha256",
        "review_assets_sha256",
        "review_case_linkage_sha256",
        "normal_partition_sha256",
        "review_asset_count",
        "method_case_count",
        "normal_partition_count",
        "diagnostic_image_accessed",
        "diagnostic_image_decoded",
        "diagnostic_image_displayed",
        "diagnostic_score_computed",
        "v0_2_final_test_rescored",
        "unselected_final_test_image_accessed",
        "protected_metadata_exposed",
        "v0_2_artifacts_modified",
        "preexisting_output_detected",
        "output_overwrite_performed",
        "raw_data_in_git",
    }
    _require(
        set(contracts["pre_access_checkpoint"].get("required_keys", [])) == checkpoint_keys,
        "pre-access checkpoint fields changed",
    )
    review_completion_keys = {
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
    _require(
        set(contracts["review_completion_checkpoint"]["required_keys"])
        == review_completion_keys,
        "review-completion checkpoint fields changed",
    )
    _require(
        set(contracts["diagnostic_decision"]["required_keys"])
        == {
            "contract_version",
            "diagnostic_id",
            "status",
            "decision",
            "selected_family",
            "family_evaluations",
            "reason",
            "weighted_score_used",
            "v0_2_decision_changed",
            "performance_claim_made",
        },
        "diagnostic decision fields changed",
    )
    _require(
        set(contracts["bundle_manifest"]["required_keys"])
        == {
            "contract_version",
            "diagnostic_id",
            "source_commit",
            "config_sha256",
            "schema_sha256",
            "files",
        },
        "bundle manifest fields changed",
    )
    blind = _object(contracts["blind_observation"], label="blind_observation")
    blind_columns = [column["name"] for column in blind["columns"]]
    _require(blind_columns == list(BLIND_OBSERVATION_FIELDS), "blind observation fields changed")
    _require(
        not (set(blind_columns) & PROTECTED_FIRST_PASS_FIELDS),
        "blind observation exposes a protected field",
    )
    _require(blind.get("label_boundary") == "first_pass_blinded", "blind boundary changed")
    return schema


def load_v0_3_config(path: Path, *, verify_identity: bool = True) -> dict[str, Any]:
    """Load and validate the JSON-compatible v0.3 configuration."""
    if verify_identity and sha256_file(path) != EXPECTED_CONFIG_SHA256:
        raise V0_3DiagnosticContractError("v0.3 config SHA-256 differs from the fixed identity")
    return validate_v0_3_config(_load_json_object(path, label="v0.3 config"))


def load_v0_3_schema(path: Path, *, verify_identity: bool = True) -> dict[str, Any]:
    """Load and validate the v0.3 diagnostic artifact schema."""
    if verify_identity and sha256_file(path) != EXPECTED_SCHEMA_SHA256:
        raise V0_3DiagnosticContractError("v0.3 schema SHA-256 differs from the fixed identity")
    return validate_v0_3_schema(_load_json_object(path, label="v0.3 schema"))


def validate_repository_contract(*, config_path: Path, schema_path: Path) -> tuple[dict, dict]:
    """Cross-check the fixed repository configuration and schema."""
    config = load_v0_3_config(config_path)
    schema = load_v0_3_schema(schema_path)
    _require(
        schema["protected_first_pass_fields"] == config["review"]["protected_fields"],
        "config and schema protected fields differ",
    )
    return config, schema


def _validate_typed_value(value: object, value_type: str, *, nullable: bool, label: str) -> None:
    if value is None:
        if nullable:
            return
        raise V0_3DiagnosticContractError(f"{label} may not be null")
    if value_type == "string":
        _require(isinstance(value, str), f"{label} must be a string")
    elif value_type == "asset_id":
        _require(
            isinstance(value, str) and ASSET_ID_PATTERN.fullmatch(value) is not None,
            f"{label} must be an opaque asset ID",
        )
    elif value_type == "integer":
        _integer(value, label=label)
    elif value_type == "sha256":
        _require(_is_sha256(value), f"{label} must be a lowercase SHA-256")
    elif value_type == "relative_path":
        _relative_path(value, label=label)
    elif value_type == "method":
        _require(value in METHODS, f"{label} contains an unknown method")
    elif value_type == "case_type":
        _require(value in {"false_positive", "false_negative"}, f"{label} is invalid")
    elif value_type == "review_status":
        _require(value in REVIEW_STATUS_VALUES, f"{label} is invalid")
    elif value_type == "visibility":
        _require(value in VISIBILITY_VALUES, f"{label} is invalid")
    elif value_type == "condition":
        _require(value in CONDITIONS, f"{label} is invalid")
    elif value_type == "score_status":
        _require(value in {"ok", "failed"}, f"{label} is invalid")
    elif value_type == "boolean":
        _require(isinstance(value, bool), f"{label} must be boolean")
    elif value_type == "note_1_300":
        _require(
            isinstance(value, str) and 1 <= len(value) <= 300,
            f"{label} must contain 1 to 300 Unicode code points",
        )
        _require(
            not any(ord(character) < 32 or ord(character) == 127 for character in value),
            f"{label} contains a control character",
        )
    elif value_type == "number":
        _finite_number(value, label=label)
    else:
        raise V0_3DiagnosticContractError(f"unknown schema type: {value_type}")


def validate_tabular_record(
    contract_name: str,
    record: Mapping[str, Any],
    *,
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one fully declared CSV record without coercing values."""
    validate_v0_3_schema(schema)
    contracts = _object(schema.get("contracts"), label="contracts")
    contract = _object(contracts.get(contract_name), label=f"contracts.{contract_name}")
    columns = contract.get("columns")
    if not isinstance(columns, list):
        raise V0_3DiagnosticContractError(f"{contract_name} has no declared columns")
    expected = {column["name"] for column in columns}
    _exact_keys(record, expected, label=f"{contract_name} record")
    for column in columns:
        _validate_typed_value(
            record[column["name"]],
            column["type"],
            nullable=column["nullable"],
            label=f"{contract_name}.{column['name']}",
        )
    _require(
        record.get("contract_version") == ARTIFACT_CONTRACT_VERSION,
        f"{contract_name} contract version changed",
    )
    _require(record.get("diagnostic_id") == DIAGNOSTIC_ID, "diagnostic ID changed")
    return dict(record)


def validate_blind_observation_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_asset_ids: Sequence[str],
    schema: Mapping[str, Any],
    require_complete: bool,
) -> list[dict[str, Any]]:
    """Validate first-pass observations, order, uniqueness, and completeness."""
    if len(records) > len(expected_asset_ids):
        raise V0_3DiagnosticContractError("too many blind observation records")
    validated = [
        validate_tabular_record("blind_observation", record, schema=schema) for record in records
    ]
    ids = [record["asset_id"] for record in validated]
    _require(len(ids) == len(set(ids)), "duplicate blind observation asset ID")
    _require(ids == list(expected_asset_ids[: len(ids)]), "blind observations are out of order")
    if require_complete:
        _require(ids == list(expected_asset_ids), "blind observations are incomplete")
    return validated


def validate_checkpoint_record(value: object) -> dict[str, Any]:
    """Validate the no-access checkpoint's exact keys and closed boundary flags."""
    record = _object(value, label="pre-access checkpoint")
    expected = {
        "contract_version",
        "diagnostic_id",
        "source_commit",
        "config_sha256",
        "schema_sha256",
        "preregistration_commit",
        "preregistration_document_sha256",
        "parent_artifact_manifest_sha256",
        "opaque_scoring_manifest_sha256",
        "normal_calibration_manifest_sha256",
        "normal_reference_manifest_sha256",
        "ecc_failure_cases_sha256",
        "patch_hog_failure_cases_sha256",
        "dinov2_failure_cases_sha256",
        "review_assets_sha256",
        "review_case_linkage_sha256",
        "normal_partition_sha256",
        "review_asset_count",
        "method_case_count",
        "normal_partition_count",
        "diagnostic_image_accessed",
        "diagnostic_image_decoded",
        "diagnostic_image_displayed",
        "diagnostic_score_computed",
        "v0_2_final_test_rescored",
        "unselected_final_test_image_accessed",
        "protected_metadata_exposed",
        "v0_2_artifacts_modified",
        "preexisting_output_detected",
        "output_overwrite_performed",
        "raw_data_in_git",
    }
    _exact_keys(record, expected, label="pre-access checkpoint")
    _require(record["contract_version"] == ARTIFACT_CONTRACT_VERSION, "checkpoint version changed")
    _require(record["diagnostic_id"] == DIAGNOSTIC_ID, "checkpoint diagnostic ID changed")
    _require(_is_commit(record["source_commit"]), "checkpoint source commit is invalid")
    _require(record["config_sha256"] == EXPECTED_CONFIG_SHA256, "checkpoint config changed")
    _require(record["schema_sha256"] == EXPECTED_SCHEMA_SHA256, "checkpoint schema changed")
    _require(
        record["preregistration_commit"] == PREREGISTRATION_COMMIT
        and record["preregistration_document_sha256"] == PREREGISTRATION_DOCUMENT_SHA256,
        "checkpoint preregistration changed",
    )
    expected_parent = {
        "parent_artifact_manifest_sha256": (
            "dbf7bfc1d2429bc391720136ba76711fa0493eb7407487e185bf7d12c4b85d19"
        ),
        "opaque_scoring_manifest_sha256": (
            "32ea52ed1b9872f39ae27f5d58a353ea84b8b143642e3a7f0fabe940184705e8"
        ),
        "normal_calibration_manifest_sha256": (
            "77d5adb588e7d463e7fcab1c10b841b9ad23d827b51138c31f42dac35bd99ca3"
        ),
        "normal_reference_manifest_sha256": (
            "e587f1808262480261ae8a7b940faff0d9ef5f83cf215028b31490ba48369b99"
        ),
        "ecc_failure_cases_sha256": (
            "1bd73cf3b0050b3eff934467625ba118d662e442ceaa5829f8a47344b4f390e8"
        ),
        "patch_hog_failure_cases_sha256": (
            "7edde3c32fd23e9d53db0c665595bb3c6789a1ce52bf039810b9d7646d3387ff"
        ),
        "dinov2_failure_cases_sha256": (
            "ad8f05152f936b8b9da5a6d94f2f8dc9eab43035373836f949dbb3a7887c22e9"
        ),
    }
    _require(
        all(record[field] == sha256 for field, sha256 in expected_parent.items()),
        "checkpoint parent evidence changed",
    )
    for field in (
        "config_sha256",
        "schema_sha256",
        "preregistration_document_sha256",
        "parent_artifact_manifest_sha256",
        "opaque_scoring_manifest_sha256",
        "normal_calibration_manifest_sha256",
        "normal_reference_manifest_sha256",
        "ecc_failure_cases_sha256",
        "patch_hog_failure_cases_sha256",
        "dinov2_failure_cases_sha256",
        "review_assets_sha256",
        "review_case_linkage_sha256",
        "normal_partition_sha256",
    ):
        _require(_is_sha256(record[field]), f"checkpoint {field} is invalid")
    _require(_is_commit(record["preregistration_commit"]), "checkpoint prereg commit is invalid")
    _require(record["review_asset_count"] == 28, "checkpoint review count changed")
    _require(record["method_case_count"] == 29, "checkpoint method-case count changed")
    _require(record["normal_partition_count"] == 60, "checkpoint normal count changed")
    closed_flags = (
        "diagnostic_image_accessed",
        "diagnostic_image_decoded",
        "diagnostic_image_displayed",
        "diagnostic_score_computed",
        "v0_2_final_test_rescored",
        "unselected_final_test_image_accessed",
        "protected_metadata_exposed",
        "v0_2_artifacts_modified",
        "preexisting_output_detected",
        "output_overwrite_performed",
        "raw_data_in_git",
    )
    _require(
        all(record[field] is False for field in closed_flags),
        "checkpoint access boundary is open",
    )
    return record
