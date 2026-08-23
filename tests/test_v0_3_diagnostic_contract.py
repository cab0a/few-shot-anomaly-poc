from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from few_shot_anomaly_poc.v0_3_diagnostic_contract import (
    ARTIFACT_CONTRACT_VERSION,
    DIAGNOSTIC_ID,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_SCHEMA_SHA256,
    PREREGISTRATION_COMMIT,
    PREREGISTRATION_DOCUMENT_SHA256,
    V0_3DiagnosticContractError,
    load_v0_3_config,
    load_v0_3_schema,
    validate_blind_observation_records,
    validate_checkpoint_record,
    validate_repository_contract,
    validate_tabular_record,
    validate_v0_3_config,
    validate_v0_3_schema,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/v0.3.yaml"
SCHEMA_PATH = ROOT / "schemas/v0.3/diagnostic-artifacts.json"
SHA256 = "a" * 64


def _contract() -> tuple[dict, dict]:
    return validate_repository_contract(config_path=CONFIG_PATH, schema_path=SCHEMA_PATH)


def _blind_record(asset_id: str) -> dict:
    return {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "asset_id": asset_id,
        "review_status": "reviewed",
        "framing_or_crop_difference_visible": "no",
        "pose_or_registration_offset_visible": "uncertain",
        "global_exposure_difference_visible": "no",
        "blur_visible": "no",
        "localized_change_visible": "yes",
        "diffuse_change_visible": "no",
        "no_obvious_visible_change": "no",
        "visible_observation": "A synthetic localized mark is visible.",
        "inferred_cause": None,
    }


def test_fixed_config_and_schema_identities_cross_validate() -> None:
    config, schema = _contract()

    assert load_v0_3_config(CONFIG_PATH) == config
    assert load_v0_3_schema(SCHEMA_PATH) == schema
    assert EXPECTED_CONFIG_SHA256 == (
        "e3108c837b3c9818b25b86594a7b103091e145f8ee6392e8424c7163f220f548"
    )
    assert EXPECTED_SCHEMA_SHA256 == (
        "173018d554380910147a826a42364dfc8499f0cb6ec6ff76dc1e24f233d87834"
    )
    assert config["preregistration"]["commit"] == PREREGISTRATION_COMMIT
    assert schema["contract_version"] == ARTIFACT_CONTRACT_VERSION


def test_config_rejects_review_expansion_and_threshold_change() -> None:
    config = load_v0_3_config(CONFIG_PATH)
    expanded = deepcopy(config)
    expanded["review"]["unique_asset_ids"].append("asset-999999")
    changed_threshold = deepcopy(config)
    changed_threshold["methods"]["ecc_residual"]["threshold"] = 0.5

    with pytest.raises(V0_3DiagnosticContractError):
        validate_v0_3_config(expanded)
    with pytest.raises(V0_3DiagnosticContractError):
        validate_v0_3_config(changed_threshold)


def test_config_rejects_relaxed_access_boundary_and_metric_addition() -> None:
    config = load_v0_3_config(CONFIG_PATH)
    relaxed = deepcopy(config)
    relaxed["access_boundary"]["v0_2_final_test_rescoring_allowed"] = True
    changed_metrics = deepcopy(config)
    changed_metrics["summaries"]["forbidden_metrics"].remove("auroc")

    with pytest.raises(V0_3DiagnosticContractError):
        validate_v0_3_config(relaxed)
    with pytest.raises(V0_3DiagnosticContractError):
        validate_v0_3_config(changed_metrics)


def test_schema_rejects_protected_field_in_blind_observation() -> None:
    schema = load_v0_3_schema(SCHEMA_PATH)
    changed = deepcopy(schema)
    changed["contracts"]["blind_observation"]["columns"].append(
        {"name": "anomaly_score", "type": "number", "nullable": False}
    )

    with pytest.raises(V0_3DiagnosticContractError, match="blind_observation columns"):
        validate_v0_3_schema(changed)


def test_review_asset_record_accepts_only_opaque_identity_fields() -> None:
    _, schema = _contract()
    record = {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "asset_id": "asset-000003",
        "byte_count": 123,
        "sha256": SHA256,
    }

    assert validate_tabular_record("review_asset", record, schema=schema) == record
    injected = {**record, "source_path": "pcb2/Data/Images/Anomaly/001.JPG"}
    with pytest.raises(V0_3DiagnosticContractError, match="fields differ"):
        validate_tabular_record("review_asset", injected, schema=schema)


def test_review_case_link_is_a_separate_post_review_contract() -> None:
    _, schema = _contract()
    record = {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "method": "ecc_residual",
        "case_type": "false_positive",
        "rank": 1,
        "asset_id": "asset-000141",
        "parent_failure_sha256": SHA256,
    }

    assert validate_tabular_record("review_case_link", record, schema=schema) == record


def test_normal_partition_requires_canonical_relative_path_and_identity() -> None:
    _, schema = _contract()
    record = {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "selection_rank": 1,
        "selection_sha256": SHA256,
        "relative_path": "pcb2/Data/Images/Normal/0001.JPG",
        "byte_count": 456,
        "sha256": SHA256,
    }

    assert validate_tabular_record("normal_partition", record, schema=schema) == record
    record["relative_path"] = "../outside.JPG"
    with pytest.raises(V0_3DiagnosticContractError, match="escapes"):
        validate_tabular_record("normal_partition", record, schema=schema)


def test_blind_observation_accepts_fixed_fields_and_nullable_inference() -> None:
    config, schema = _contract()
    asset_id = config["review"]["unique_asset_ids"][0]
    record = _blind_record(asset_id)

    assert validate_tabular_record("blind_observation", record, schema=schema) == record
    record["inferred_cause"] = "A synthetic lighting change may explain the mark."
    assert validate_tabular_record("blind_observation", record, schema=schema) == record


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("review_status", "skipped"),
        ("blur_visible", "likely"),
        ("visible_observation", "x" * 301),
        ("inferred_cause", "x" * 301),
    ],
)
def test_blind_observation_rejects_invalid_values(field: str, value: str) -> None:
    config, schema = _contract()
    record = _blind_record(config["review"]["unique_asset_ids"][0])
    record[field] = value

    with pytest.raises(V0_3DiagnosticContractError):
        validate_tabular_record("blind_observation", record, schema=schema)


def test_blind_observation_rejects_protected_field_injection() -> None:
    config, schema = _contract()
    record = _blind_record(config["review"]["unique_asset_ids"][0])
    record["method"] = "ecc_residual"

    with pytest.raises(V0_3DiagnosticContractError, match="fields differ"):
        validate_tabular_record("blind_observation", record, schema=schema)


def test_blind_observation_sequence_rejects_duplicate_and_out_of_order_rows() -> None:
    config, schema = _contract()
    asset_ids = config["review"]["unique_asset_ids"]
    valid = [_blind_record(asset_ids[0]), _blind_record(asset_ids[1])]
    assert len(
        validate_blind_observation_records(
            valid,
            expected_asset_ids=asset_ids,
            schema=schema,
            require_complete=False,
        )
    ) == 2

    with pytest.raises(V0_3DiagnosticContractError, match="duplicate"):
        validate_blind_observation_records(
            [valid[0], valid[0]],
            expected_asset_ids=asset_ids,
            schema=schema,
            require_complete=False,
        )
    with pytest.raises(V0_3DiagnosticContractError, match="out of order"):
        validate_blind_observation_records(
            list(reversed(valid)),
            expected_asset_ids=asset_ids,
            schema=schema,
            require_complete=False,
        )


def test_complete_blind_observation_sequence_rejects_missing_rows() -> None:
    config, schema = _contract()
    asset_ids = config["review"]["unique_asset_ids"]

    with pytest.raises(V0_3DiagnosticContractError, match="incomplete"):
        validate_blind_observation_records(
            [_blind_record(asset_ids[0])],
            expected_asset_ids=asset_ids,
            schema=schema,
            require_complete=True,
        )


def test_pre_access_checkpoint_requires_all_boundary_flags_false() -> None:
    checkpoint = {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "source_commit": "b" * 40,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "schema_sha256": EXPECTED_SCHEMA_SHA256,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_document_sha256": PREREGISTRATION_DOCUMENT_SHA256,
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
        "review_assets_sha256": SHA256,
        "review_case_linkage_sha256": SHA256,
        "normal_partition_sha256": SHA256,
        "review_asset_count": 28,
        "method_case_count": 29,
        "normal_partition_count": 60,
        "diagnostic_image_accessed": False,
        "diagnostic_image_decoded": False,
        "diagnostic_image_displayed": False,
        "diagnostic_score_computed": False,
        "v0_2_final_test_rescored": False,
        "unselected_final_test_image_accessed": False,
        "protected_metadata_exposed": False,
        "v0_2_artifacts_modified": False,
        "preexisting_output_detected": False,
        "output_overwrite_performed": False,
        "raw_data_in_git": False,
    }

    assert validate_checkpoint_record(checkpoint) == checkpoint
    checkpoint["diagnostic_score_computed"] = True
    with pytest.raises(V0_3DiagnosticContractError, match="boundary is open"):
        validate_checkpoint_record(checkpoint)
