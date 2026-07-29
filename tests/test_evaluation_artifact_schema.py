from __future__ import annotations

import json
import re
from pathlib import Path

from few_shot_anomaly_poc.evaluation_artifacts import (
    CLASSIFICATION_COLUMNS,
    FAILURE_COLUMNS,
    LABEL_COLUMNS,
    LATENCY_OBSERVATION_COLUMNS,
    SCORE_COLUMNS,
)

SCHEMA_PATH = Path("schemas/v0.1/evaluation-artifacts.json")
CONTRACT_VERSION = "evaluation-artifacts/v0.1"
REQUIRED_CONTRACTS = {
    "score",
    "classification",
    "label_reveal",
    "metrics",
    "latency",
    "failure",
    "decision",
    "bundle_manifest",
}
REQUIRED_PRIMARY_CONTRACTS = {
    "score",
    "classification",
    "metrics",
    "latency",
    "failure",
    "decision",
}


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _csv_specs(contract: dict):
    if contract["format"] == "csv":
        yield contract
    elif contract["format"] == "json_and_csv":
        yield from (
            file_spec for file_spec in contract["files"].values() if file_spec["format"] == "csv"
        )


def test_schema_is_valid_json_with_fixed_contract_inventory() -> None:
    schema = _schema()

    assert schema["contract_version"] == CONTRACT_VERSION
    assert schema["status"] == "frozen_before_final_test"
    assert set(schema["contracts"]) == REQUIRED_CONTRACTS
    assert set(schema["contracts"]) >= REQUIRED_PRIMARY_CONTRACTS
    assert schema["allowed_run_kinds"] == ["synthetic", "final_test"]
    assert schema["allowed_methods"] == [
        "ecc_residual",
        "patch_hog_one_class_svm",
    ]
    assert re.fullmatch(schema["run_id_pattern"], "synthetic-e2e")
    assert re.fullmatch(schema["run_id_pattern"], "first-fixed-final-test")
    assert not re.fullmatch(schema["run_id_pattern"], "2026-07-29T12:34:56")


def test_serialization_policy_is_deterministic_and_rejects_non_finite_values() -> None:
    schema = _schema()
    serialization = schema["serialization"]
    common = schema["common_rules"]

    assert serialization == {
        "text_encoding": "UTF-8",
        "line_ending": "LF",
        "final_newline": True,
        "json": {
            "indent": 2,
            "sort_keys": True,
            "ensure_ascii": False,
            "allow_nan": False,
        },
        "csv": {
            "dialect": "RFC 4180",
            "delimiter": ",",
            "header": True,
            "quoting": "minimal",
            "null": "",
            "boolean_true": "true",
            "boolean_false": "false",
            "finite_float": "CPython 3.13 shortest round-trip decimal",
        },
    }
    assert common["row_order_must_match_sort_key"] is True
    assert common["duplicate_primary_keys_forbidden"] is True
    assert common["overwrite_forbidden"] is True
    assert common["absolute_paths_forbidden"] is True
    assert common["timestamps_forbidden"] is True
    assert common["non_finite_numbers_forbidden"] is True
    assert common["raw_dataset_content_forbidden"] is True


def test_every_csv_contract_has_unique_columns_and_key_columns() -> None:
    schema = _schema()

    for contract in schema["contracts"].values():
        for csv_spec in _csv_specs(contract):
            columns = tuple(column["name"] for column in csv_spec["columns"])
            assert len(columns) == len(set(columns))
            assert columns[0] == "contract_version"
            assert {"run_id", "run_kind"} <= set(columns)
            assert set(csv_spec["primary_key"]) <= set(columns)
            assert all(
                key in columns
                or key
                in {
                    "case_type_fixed_false_positive_first",
                    "partition_fixed_calibration_first",
                }
                for key in csv_spec["sort_key"]
            )
            assert all(isinstance(column["nullable"], bool) for column in csv_spec["columns"])


def test_writer_headers_match_the_machine_readable_contract() -> None:
    contracts = _schema()["contracts"]

    assert tuple(column["name"] for column in contracts["score"]["columns"]) == SCORE_COLUMNS
    assert (
        tuple(column["name"] for column in contracts["classification"]["columns"])
        == CLASSIFICATION_COLUMNS
    )
    assert tuple(column["name"] for column in contracts["label_reveal"]["columns"]) == LABEL_COLUMNS
    assert (
        tuple(column["name"] for column in contracts["latency"]["files"]["observations"]["columns"])
        == LATENCY_OBSERVATION_COLUMNS
    )
    assert tuple(column["name"] for column in contracts["failure"]["columns"]) == FAILURE_COLUMNS


def test_label_free_contracts_do_not_contain_true_class() -> None:
    contracts = _schema()["contracts"]
    score_columns = {column["name"] for column in contracts["score"]["columns"]}
    classification_columns = {column["name"] for column in contracts["classification"]["columns"]}
    reveal_columns = {column["name"] for column in contracts["label_reveal"]["columns"]}

    assert "true_class" not in score_columns
    assert "partition" in score_columns
    assert "true_class" not in classification_columns
    assert "true_class" in reveal_columns
    assert "anomaly_score" not in reveal_columns
    assert "predicted_class" not in reveal_columns


def test_metric_and_decision_json_contracts_cover_required_evidence() -> None:
    contracts = _schema()["contracts"]
    metrics = set(contracts["metrics"]["required_keys"])
    decision = set(contracts["decision"]["required_keys"])

    assert {
        "image_level_auroc",
        "image_level_auprc",
        "normal_false_positive_rate",
        "anomaly_recall",
        "false_positive_count",
        "false_negative_count",
        "score_failure_count",
    } <= metrics
    assert {
        "decision",
        "gate_outcomes",
        "first_failed_gate",
        "test_leakage_detected",
        "failure_review_disposition",
        "failure_review_rationale",
        "condition",
        "decision_reason",
    } <= decision
    assert contracts["metrics"]["additional_keys_allowed"] is False
    assert contracts["decision"]["additional_keys_allowed"] is False
    assert set(contracts["metrics"]["field_types"]) == set(contracts["metrics"]["required_keys"])
    assert set(contracts["decision"]["field_types"]) == set(contracts["decision"]["required_keys"])
    assert contracts["decision"]["field_enums"]["decision"] == [
        "ADOPT",
        "ADOPT WITH CONDITIONS",
        "REJECT",
    ]
    assert set(contracts["decision"]["gate_outcome_field_types"]) == set(
        contracts["decision"]["gate_outcome_required_keys"]
    )


def test_latency_contract_preserves_every_timing_and_environment() -> None:
    latency = _schema()["contracts"]["latency"]
    summary = latency["files"]["summary"]
    observations = latency["files"]["observations"]
    observation_columns = {column["name"] for column in observations["columns"]}

    assert {
        "measurement_boundary",
        "warmup_passes",
        "timed_passes",
        "sample_count",
        "median_latency_ns",
        "p95_latency_ns",
        "environment",
    } <= set(summary["required_keys"])
    assert {
        "pass_index",
        "relative_path",
        "duration_ns",
        "score_status",
        "score_failure_code",
    } <= observation_columns
    assert observations["sort_key"] == [
        "method",
        "pass_index",
        "relative_path",
    ]
    assert set(summary["field_types"]) == set(summary["required_keys"])
    assert set(summary["environment_field_types"]) == set(summary["environment_required_keys"])


def test_manifest_covers_provenance_without_self_digest() -> None:
    manifest = _schema()["contracts"]["bundle_manifest"]

    assert manifest["path_template"] == "artifact-manifest.json"
    assert {
        "source_commit",
        "config_sha256",
        "partition_manifest_sha256",
        "files",
    } <= set(manifest["required_keys"])
    assert manifest["file_entry_required_keys"] == [
        "artifact_type",
        "method",
        "relative_path",
        "sha256",
        "record_count",
    ]
    assert manifest["file_sort_key"] == ["relative_path"]
    assert set(manifest["field_types"]) == set(manifest["required_keys"])
    assert set(manifest["file_entry_field_types"]) == set(manifest["file_entry_required_keys"])


def test_schema_creates_no_result_bundle_or_machine_specific_path() -> None:
    serialized = SCHEMA_PATH.read_text(encoding="utf-8")

    assert not Path("artifacts/v0.1/evaluation").exists()
    assert "/home/" not in serialized
    assert "\\\\wsl.localhost" not in serialized
