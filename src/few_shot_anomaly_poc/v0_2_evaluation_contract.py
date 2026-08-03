"""Validate the fixed v0.2 configuration and evaluation artifact records."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

CONFIG_SCHEMA_VERSION = "v0.2-evaluation-contract-v1"
ARTIFACT_CONTRACT_VERSION = "evaluation-artifacts/v0.2"
EXPECTED_CONFIG_SHA256 = "9ea3a7156aeb3c6efc87c8ae3811444421bd3e86ea25b7ac014893c7e5892265"
EXPECTED_SCHEMA_SHA256 = "4178d24f7210f8f859f6c99386204022ac3b2e4ffe4b7b63cdfea00d0c79f31d"
PREREGISTRATION_COMMIT = "b873bacc4f677a4c82f3944c09a7374037cb7c50"
PREREGISTRATION_DOCUMENT_SHA256 = "6306c2122f69aa96dcfe1f377518e7c6795a096eceb085be5b829262b55482b9"
METHODS = (
    "ecc_residual",
    "patch_hog_ocsvm",
    "dinov2_vits14_224_nn",
)
HARD_GATES = (
    "method_fit",
    "test_leakage",
    "final_test_normal_fpr",
    "final_test_anomaly_recall",
    "cpu_p95_scoring_latency",
    "normal_reference_count",
    "anomaly_training_labels",
    "reproducibility",
)
TABULAR_CONTRACTS = (
    "calibration_score",
    "score",
    "classification",
    "latency_observation",
    "reproduction",
    "label_reveal",
    "failure_case",
)
JSON_CONTRACTS = (
    "boundary_record",
    "pre_evaluation_freeze",
    "fit",
    "calibration_summary",
    "pre_reveal_checkpoint",
    "metrics",
    "method_decision",
    "project_decision",
    "bundle_manifest",
)
REQUIRED_CONTRACTS = frozenset((*TABULAR_CONTRACTS, *JSON_CONTRACTS))
RUN_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
ASSET_ID_PATTERN = re.compile(r"^asset-[0-9]{6}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class V0_2EvaluationContractError(Exception):
    """Reject configuration, schema, or evidence outside the fixed contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V0_2EvaluationContractError(f"cannot load {label}") from error
    if not isinstance(value, dict):
        raise V0_2EvaluationContractError(f"{label} must contain a JSON object")
    return value


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise V0_2EvaluationContractError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise V0_2EvaluationContractError(f"{label} fields differ from the contract")


def _require(value: bool, message: str) -> None:
    if not value:
        raise V0_2EvaluationContractError(message)


def _finite_number(value: object, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise V0_2EvaluationContractError(f"{label} must be a finite number")
    return float(value)


def _integer(value: object, *, label: str, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise V0_2EvaluationContractError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise V0_2EvaluationContractError(f"{label} is below its minimum")
    return value


def _relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise V0_2EvaluationContractError(f"{label} must be a POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise V0_2EvaluationContractError(f"{label} escapes its declared root")
    if path.as_posix() != value:
        raise V0_2EvaluationContractError(f"{label} must be canonical POSIX syntax")
    return value


def _canonical_json_object(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise V0_2EvaluationContractError(f"{label} must be canonical JSON text")
    try:
        parsed = json.loads(value)
        canonical = json.dumps(
            parsed,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise V0_2EvaluationContractError(f"{label} is invalid JSON") from error
    if not isinstance(parsed, dict) or value != canonical:
        raise V0_2EvaluationContractError(f"{label} must be one canonical JSON object")
    return value


def validate_v0_2_config(value: object) -> dict[str, Any]:
    """Validate the semantic identities and fixed rules in one v0.2 config."""
    config = _mapping(value, label="v0.2 config")
    expected_top = {
        "artifact_contract",
        "calibration",
        "dataset",
        "decision",
        "dinov2_input_adapter",
        "entry_preflight",
        "failure_selection",
        "final_test_scoring",
        "hard_gates",
        "latency",
        "method_order",
        "methods",
        "metrics",
        "opaque_boundary",
        "partitions",
        "preregistration",
        "reproduction",
        "schema_version",
    }
    _exact_keys(config, expected_top, label="v0.2 config")
    _require(
        config.get("schema_version") == CONFIG_SCHEMA_VERSION,
        "config schema_version changed",
    )

    preregistration = _mapping(config.get("preregistration"), label="preregistration")
    _require(
        preregistration.get("id") == "v0.2-pcb2-method-evaluation-1"
        and preregistration.get("commit") == PREREGISTRATION_COMMIT
        and preregistration.get("document") == "docs/v0.2-method-and-evaluation-preregistration.md"
        and preregistration.get("document_sha256") == PREREGISTRATION_DOCUMENT_SHA256,
        "preregistration identity changed",
    )
    entry = _mapping(config.get("entry_preflight"), label="entry_preflight")
    _require(
        entry.get("decision") == "PROCEED"
        and entry.get("artifact_sha256")
        == "04da3c64f181c05411b8fdaec2b56db15a38812978815f4c2a1fcac37a9e0298",
        "entry preflight identity changed",
    )

    dataset = _mapping(config.get("dataset"), label="dataset")
    _require(
        dataset.get("name") == "VisA"
        and dataset.get("category") == "pcb2"
        and dataset.get("license") == "CC BY 4.0"
        and dataset.get("archive_sha256")
        == "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362"
        and dataset.get("split_revision") == "2a692ab575001cbde74d402d897a7286086c6199"
        and dataset.get("split_sha256")
        == "a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995"
        and dataset.get("raw_data_in_git") is False,
        "dataset identity changed",
    )

    partitions = _mapping(config.get("partitions"), label="partitions")
    reference = _mapping(partitions.get("reference"), label="partitions.reference")
    _require(
        reference.get("count") == 20
        and reference.get("seed") == 42
        and reference.get("selection") == "sha256_path_ranking_v1"
        and reference.get("selection_input")
        == "few-shot-anomaly-poc:v0.2:pcb2:42:<posix_relative_path>"
        and partitions.get("duplicate_paths_forbidden") is True
        and partitions.get("overlap_forbidden") is True,
        "partition rules changed",
    )

    boundary = _mapping(config.get("opaque_boundary"), label="opaque_boundary")
    _require(
        boundary.get("ordering_algorithm") == "HMAC-SHA-256"
        and boundary.get("ordering_key_bytes") == 32
        and boundary.get("asset_id_pattern") == ASSET_ID_PATTERN.pattern
        and boundary.get("scorer_record_fields")
        == ["asset_id", "relative_path", "byte_count", "sha256"]
        and boundary.get("overwrite_allowed") is False
        and boundary.get("manual_final_test_image_access_before_reveal") is False,
        "opaque boundary rules changed",
    )

    method_order = config.get("method_order")
    methods = _mapping(config.get("methods"), label="methods")
    _require(method_order == list(METHODS), "method order changed")
    _require(set(methods) == set(METHODS), "method inventory changed")
    expected_scores = {
        "ecc_residual": (0.0, True, 1.0, True, 1.0),
        "patch_hog_ocsvm": (-1e12, False, 1e12, False, 1e12),
        "dinov2_vits14_224_nn": (0.0, True, 2.0, True, 2.0),
    }
    for method, expected in expected_scores.items():
        method_config = _mapping(methods.get(method), label=f"methods.{method}")
        actual = (
            method_config.get("success_score_minimum"),
            method_config.get("success_score_minimum_inclusive"),
            method_config.get("success_score_maximum"),
            method_config.get("success_score_maximum_inclusive"),
            method_config.get("failure_score"),
        )
        _require(actual == expected, f"{method} score rules changed")
        _require(method_config.get("reference_count") == 20, f"{method} reference count changed")
    dino = methods["dinov2_vits14_224_nn"]
    _require(
        dino.get("source_revision") == "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
        and dino.get("checkpoint_sha256")
        == "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
        and dino.get("resolution") == 224
        and dino.get("device") == "cpu"
        and dino.get("dtype") == "float32"
        and dino.get("patch_tokens_per_image") == 256
        and dino.get("memory_bank_patch_count") == 5120,
        "DINOv2 method identity changed",
    )

    calibration = _mapping(config.get("calibration"), label="calibration")
    _require(
        calibration.get("normal_only") is True
        and calibration.get("quantile") == 0.95
        and calibration.get("prediction_rule") == "failed_or_strictly_greater"
        and calibration.get("parameter_changes_allowed") is False,
        "calibration rules changed",
    )
    scoring = _mapping(config.get("final_test_scoring"), label="final_test_scoring")
    _require(
        scoring.get("warmup_passes") == 1
        and scoring.get("timed_passes") == 3
        and scoring.get("canonical_score_pass_index") == 0
        and scoring.get("retry_allowed") is False
        and scoring.get("label_access") is False
        and scoring.get("score_repetition_absolute_tolerance") == 1e-6,
        "final-test scoring rules changed",
    )
    latency = _mapping(config.get("latency"), label="latency")
    _require(
        latency.get("device") == "cpu"
        and latency.get("timer") == "perf_counter_ns"
        and latency.get("timed_passes") == 3
        and latency.get("p95_quantile") == 0.95
        and latency.get("dinov2_duration_rule") == "adapter_ns_plus_scorer_ns"
        and latency.get("preflight_latency_is_final_gate") is False,
        "latency rules changed",
    )
    reproduction = _mapping(config.get("reproduction"), label="reproduction")
    _require(
        reproduction.get("asset_count") == 10
        and reproduction.get("absolute_score_tolerance") == 1e-6
        and reproduction.get("offline") is True
        and reproduction.get("network_allowed") is False,
        "reproduction rules changed",
    )
    gates = _mapping(config.get("hard_gates"), label="hard_gates")
    _require(
        gates.get("order") == list(HARD_GATES)
        and gates.get("normal_fpr_max") == 0.05
        and gates.get("anomaly_recall_min") == 0.90
        and gates.get("cpu_p95_latency_seconds_max") == 1.0
        and gates.get("normal_reference_count_required") == 20
        and gates.get("weighted_score_allowed") is False
        and gates.get("hard_gate_waiver_allowed") is False,
        "hard-gate rules changed",
    )
    artifacts = _mapping(config.get("artifact_contract"), label="artifact_contract")
    _require(
        artifacts.get("contract_version") == ARTIFACT_CONTRACT_VERSION
        and artifacts.get("schema_path") == "schemas/v0.2/evaluation-artifacts.json"
        and artifacts.get("overwrite_allowed") is False
        and artifacts.get("absolute_paths_allowed") is False
        and artifacts.get("non_finite_numbers_allowed") is False
        and artifacts.get("raw_dataset_content_allowed") is False,
        "artifact rules changed",
    )
    return config


def load_v0_2_config(path: Path, *, require_fixed_identity: bool = True) -> dict[str, Any]:
    """Load a JSON-compatible YAML config and optionally require its fixed hash."""
    if require_fixed_identity and (
        not path.is_file() or _sha256_file(path) != EXPECTED_CONFIG_SHA256
    ):
        raise V0_2EvaluationContractError("v0.2 config SHA-256 changed")
    return validate_v0_2_config(_load_json_object(path, label="v0.2 config"))


def _column_names(contract: Mapping[str, Any]) -> tuple[str, ...]:
    columns = contract.get("columns")
    if not isinstance(columns, list) or not columns:
        raise V0_2EvaluationContractError("tabular contract must contain columns")
    names = tuple(column.get("name") if isinstance(column, dict) else None for column in columns)
    if any(not isinstance(name, str) or not name for name in names) or len(names) != len(
        set(names)
    ):
        raise V0_2EvaluationContractError("contract columns must have unique names")
    return names  # type: ignore[return-value]


def validate_v0_2_artifact_schema(value: object) -> dict[str, Any]:
    """Validate the contract inventory and its label-separation declarations."""
    schema = _mapping(value, label="v0.2 artifact schema")
    _require(
        schema.get("contract_version") == ARTIFACT_CONTRACT_VERSION,
        "artifact contract version changed",
    )
    _require(
        schema.get("status") == "frozen_before_boundary_preparation",
        "artifact schema status changed",
    )
    _require(schema.get("allowed_methods") == list(METHODS), "schema methods changed")
    _require(
        schema.get("allowed_run_kinds") == ["synthetic", "final_test"],
        "run kinds changed",
    )
    contracts = _mapping(schema.get("contracts"), label="contracts")
    _require(set(contracts) == REQUIRED_CONTRACTS, "artifact contract inventory changed")
    for name in TABULAR_CONTRACTS:
        contract = _mapping(contracts.get(name), label=f"contracts.{name}")
        _require(contract.get("format") == "csv", f"{name} format changed")
        columns = _column_names(contract)
        _require(
            "contract_version" in columns and "run_id" in columns, f"{name} common fields missing"
        )
        _require(
            set(contract.get("primary_key", [])) <= set(columns)
            and all(
                key in columns or key == "case_type_false_positive_first"
                for key in contract.get("sort_key", [])
            ),
            f"{name} key declaration is invalid",
        )
    for name in JSON_CONTRACTS:
        contract = _mapping(contracts.get(name), label=f"contracts.{name}")
        _require(contract.get("format") == "json", f"{name} format changed")
        required = contract.get("required_keys")
        types = contract.get("field_types")
        _require(
            isinstance(required, list)
            and len(required) == len(set(required))
            and isinstance(types, dict)
            and set(types) == set(required),
            f"{name} JSON fields are invalid",
        )
    label_free = schema.get("final_test_label_free_contracts")
    forbidden = schema.get("final_test_label_free_forbidden_fields")
    _require(
        label_free == ["score", "classification", "latency_observation", "reproduction"]
        and isinstance(forbidden, list),
        "label-free boundary declaration changed",
    )
    for name in label_free:
        columns = set(_column_names(contracts[name]))
        _require(not columns.intersection(forbidden), f"{name} leaks a protected field")
        _require(
            contracts[name].get("label_boundary") == "final_test_label_free",
            f"{name} label boundary changed",
        )
    score_rules = _mapping(schema.get("method_score_rules"), label="method_score_rules")
    _require(set(score_rules) == set(METHODS), "schema score-rule methods changed")
    return schema


def load_v0_2_artifact_schema(path: Path, *, require_fixed_identity: bool = True) -> dict[str, Any]:
    """Load the v0.2 artifact schema and optionally require its fixed hash."""
    if require_fixed_identity and (
        not path.is_file() or _sha256_file(path) != EXPECTED_SCHEMA_SHA256
    ):
        raise V0_2EvaluationContractError("v0.2 artifact schema SHA-256 changed")
    return validate_v0_2_artifact_schema(_load_json_object(path, label="v0.2 artifact schema"))


def validate_repository_contract(
    *, config_path: Path, schema_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the exact repository contract and verify cross-file score rules."""
    config = load_v0_2_config(config_path)
    schema = load_v0_2_artifact_schema(schema_path)
    _require(
        config["artifact_contract"]["contract_version"] == schema["contract_version"],
        "config and artifact contract versions differ",
    )
    project_root = config_path.parent.parent
    declared_schema_path = project_root / config["artifact_contract"]["schema_path"]
    _require(
        declared_schema_path.resolve() == schema_path.resolve(),
        "config artifact schema path differs",
    )
    for method in METHODS:
        method_config = config["methods"][method]
        rule = schema["method_score_rules"][method]
        expected = {
            "failure_score": method_config["failure_score"],
            "success_maximum": method_config["success_score_maximum"],
            "success_maximum_inclusive": method_config["success_score_maximum_inclusive"],
            "success_minimum": method_config["success_score_minimum"],
            "success_minimum_inclusive": method_config["success_score_minimum_inclusive"],
        }
        _require(rule == expected, f"{method} config and schema score rules differ")
    return config, schema


def _validate_field_type(value: object, field_type: str, *, label: str) -> None:
    if field_type == "string":
        if not isinstance(value, str) or not value:
            raise V0_2EvaluationContractError(f"{label} must be a non-empty string")
    elif field_type == "integer":
        _integer(value, label=label)
    elif field_type == "number":
        _finite_number(value, label=label)
    elif field_type == "boolean":
        if not isinstance(value, bool):
            raise V0_2EvaluationContractError(f"{label} must be boolean")
    elif field_type == "sha256":
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            raise V0_2EvaluationContractError(f"{label} must be a SHA-256")
    elif field_type == "commit":
        if not isinstance(value, str) or not COMMIT_PATTERN.fullmatch(value):
            raise V0_2EvaluationContractError(f"{label} must be a full Git commit")
    elif field_type == "asset_id":
        if not isinstance(value, str) or not ASSET_ID_PATTERN.fullmatch(value):
            raise V0_2EvaluationContractError(f"{label} must be an opaque asset ID")
    elif field_type == "relative_path":
        _relative_path(value, label=label)
    elif field_type == "canonical_json_object":
        _canonical_json_object(value, label=label)
    elif field_type == "array":
        if not isinstance(value, list):
            raise V0_2EvaluationContractError(f"{label} must be an array")
    elif field_type == "string_array":
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise V0_2EvaluationContractError(f"{label} must be a string array")
    elif field_type == "object":
        if not isinstance(value, dict):
            raise V0_2EvaluationContractError(f"{label} must be an object")
    else:
        raise V0_2EvaluationContractError(f"unsupported field type {field_type}")


def _score_rule(schema: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    rules = _mapping(schema.get("method_score_rules"), label="method_score_rules")
    return _mapping(rules.get(method), label=f"method_score_rules.{method}")


def _score_in_success_range(score: float, rule: Mapping[str, Any]) -> bool:
    minimum = float(rule["success_minimum"])
    maximum = float(rule["success_maximum"])
    lower = score >= minimum if rule["success_minimum_inclusive"] else score > minimum
    upper = score <= maximum if rule["success_maximum_inclusive"] else score < maximum
    return lower and upper


def _validate_score_state(
    record: Mapping[str, Any],
    *,
    method: str,
    status_field: str,
    failure_field: str,
    score_field: str,
    schema: Mapping[str, Any],
) -> None:
    status = record[status_field]
    failure = record[failure_field]
    score = _finite_number(record[score_field], label=score_field)
    rule = _score_rule(schema, method)
    if status == "ok":
        _require(failure is None, f"{failure_field} must be null for an ok score")
        _require(_score_in_success_range(score, rule), f"{score_field} is outside its method range")
    elif status == "failed":
        _require(isinstance(failure, str) and bool(failure), f"{failure_field} is required")
        _require(
            score == float(rule["failure_score"]), f"{score_field} is not the fixed failure score"
        )
    else:
        raise V0_2EvaluationContractError(f"{status_field} is invalid")


def _validate_tabular_semantics(
    contract_name: str, record: Mapping[str, Any], schema: Mapping[str, Any]
) -> None:
    method = record.get("method")
    if contract_name in {
        "calibration_score",
        "score",
        "classification",
        "latency_observation",
        "reproduction",
        "failure_case",
    }:
        _require(method in METHODS, "method is outside the fixed inventory")
    if contract_name in {"calibration_score", "score", "classification", "latency_observation"}:
        _validate_score_state(
            record,
            method=str(method),
            status_field="score_status",
            failure_field="score_failure_code",
            score_field="anomaly_score",
            schema=schema,
        )
    if contract_name == "classification":
        score = float(record["anomaly_score"])
        threshold = _finite_number(record["threshold"], label="threshold")
        margin = _finite_number(record["score_margin"], label="score_margin")
        failed = record["score_status"] == "failed"
        anomalous = failed or score > threshold
        _require(margin == score - threshold, "classification score_margin is inconsistent")
        _require(record["is_anomalous"] is anomalous, "classification boolean is inconsistent")
        _require(
            record["predicted_class"] == ("anomalous" if anomalous else "normal"),
            "classification class is inconsistent",
        )
        expected_reason = (
            "score_failed"
            if failed
            else "score_strictly_greater_than_threshold"
            if score > threshold
            else "score_not_greater_than_threshold"
        )
        _require(
            record["decision_reason"] == expected_reason, "classification reason is inconsistent"
        )
    elif contract_name == "latency_observation":
        adapter = record["adapter_duration_ns"]
        scorer = _integer(record["scorer_duration_ns"], label="scorer_duration_ns", minimum=1)
        duration = _integer(record["duration_ns"], label="duration_ns", minimum=1)
        if method == "dinov2_vits14_224_nn":
            adapter_ns = _integer(adapter, label="adapter_duration_ns", minimum=1)
            _require(
                duration == adapter_ns + scorer, "DINOv2 duration must equal adapter plus scorer"
            )
        else:
            _require(adapter is None, "classical latency must not contain adapter duration")
            _require(duration == scorer, "classical duration must equal scorer duration")
    elif contract_name == "reproduction":
        _validate_score_state(
            record,
            method=str(method),
            status_field="expected_score_status",
            failure_field="expected_failure_code",
            score_field="expected_score",
            schema=schema,
        )
        _validate_score_state(
            record,
            method=str(method),
            status_field="reproduced_score_status",
            failure_field="reproduced_failure_code",
            score_field="reproduced_score",
            schema=schema,
        )
        expected_score = float(record["expected_score"])
        reproduced_score = float(record["reproduced_score"])
        difference = _finite_number(record["absolute_difference"], label="absolute_difference")
        _require(
            difference == abs(expected_score - reproduced_score),
            "reproduction difference is inconsistent",
        )
        identity_match = (
            record["expected_score_status"] == record["reproduced_score_status"]
            and record["expected_failure_code"] == record["reproduced_failure_code"]
        )
        _require(
            record["within_tolerance"] is (identity_match and difference <= 1e-6),
            "reproduction tolerance result is inconsistent",
        )
    elif contract_name == "failure_case":
        _validate_score_state(
            record,
            method=str(method),
            status_field="score_status",
            failure_field="score_failure_code",
            score_field="anomaly_score",
            schema=schema,
        )
        score = float(record["anomaly_score"])
        threshold = float(record["threshold"])
        _require(
            record["score_margin"] == score - threshold,
            "failure-case margin is inconsistent",
        )
        if record["case_type"] == "false_positive":
            _require(
                record["true_class"] == "normal" and record["predicted_class"] == "anomalous",
                "false-positive classes are inconsistent",
            )
        else:
            _require(
                record["true_class"] == "anomaly" and record["predicted_class"] == "normal",
                "false-negative classes are inconsistent",
            )


def validate_tabular_record(
    contract_name: str,
    record: object,
    *,
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one CSV-ready record against fields and cross-field rules."""
    validate_v0_2_artifact_schema(schema)
    if contract_name not in TABULAR_CONTRACTS:
        raise V0_2EvaluationContractError("unknown tabular contract")
    contract = schema["contracts"][contract_name]
    row = _mapping(record, label=f"{contract_name} record")
    columns = contract["columns"]
    expected_fields = {column["name"] for column in columns}
    _exact_keys(row, expected_fields, label=f"{contract_name} record")
    for column in columns:
        name = column["name"]
        value = row[name]
        if value is None:
            if column["nullable"] is not True:
                raise V0_2EvaluationContractError(f"{contract_name}.{name} must not be null")
            continue
        _validate_field_type(value, column["type"], label=f"{contract_name}.{name}")
        if "fixed" in column:
            _require(value == column["fixed"], f"{contract_name}.{name} fixed value changed")
        if "enum" in column:
            _require(value in column["enum"], f"{contract_name}.{name} is outside its enum")
        if "minimum" in column:
            _require(
                float(value) >= float(column["minimum"]), f"{contract_name}.{name} below minimum"
            )
        if "maximum" in column:
            _require(
                float(value) <= float(column["maximum"]), f"{contract_name}.{name} above maximum"
            )
    if contract_name in schema["final_test_label_free_contracts"]:
        _require(
            not set(row).intersection(schema["final_test_label_free_forbidden_fields"]),
            f"{contract_name} contains a protected field",
        )
    _require(RUN_ID_PATTERN.fullmatch(row["run_id"]) is not None, "run_id is invalid")
    _validate_tabular_semantics(contract_name, row, schema)
    return row


def _record_sort_key(contract_name: str, record: Mapping[str, Any]) -> tuple[Any, ...]:
    method_rank = METHODS.index(record["method"]) if "method" in record else -1
    if contract_name == "calibration_score":
        return (method_rank, record["source_path"])
    if contract_name in {"score", "classification", "reproduction"}:
        return (method_rank, record["asset_id"])
    if contract_name == "latency_observation":
        return (method_rank, record["pass_index"], record["asset_id"])
    if contract_name == "label_reveal":
        return (record["asset_id"],)
    if contract_name == "failure_case":
        case_rank = 0 if record["case_type"] == "false_positive" else 1
        return (method_rank, case_rank, record["rank"])
    raise V0_2EvaluationContractError("unknown sequence contract")


def validate_tabular_records(
    contract_name: str,
    records: Sequence[object],
    *,
    schema: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Validate ordering and primary-key uniqueness for one record collection."""
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise V0_2EvaluationContractError("record collection must be non-empty")
    validated = tuple(
        validate_tabular_record(contract_name, record, schema=schema) for record in records
    )
    expected_order = tuple(sorted(validated, key=lambda row: _record_sort_key(contract_name, row)))
    _require(validated == expected_order, f"{contract_name} record order is invalid")
    primary_key = schema["contracts"][contract_name]["primary_key"]
    keys = [tuple(record[field] for field in primary_key) for record in validated]
    _require(len(keys) == len(set(keys)), f"{contract_name} primary keys are not unique")
    _validate_collection_completeness(contract_name, validated)
    return validated


def _asset_index(asset_id: str) -> int:
    return int(asset_id.removeprefix("asset-"))


def _require_contiguous_assets(records: Sequence[Mapping[str, Any]], *, label: str) -> None:
    indexes = [_asset_index(str(record["asset_id"])) for record in records]
    _require(indexes == list(range(len(indexes))), f"{label} asset IDs are not contiguous")


def _validate_collection_completeness(
    contract_name: str,
    records: Sequence[Mapping[str, Any]],
) -> None:
    run_identities = {(record["run_id"], record["run_kind"]) for record in records}
    _require(len(run_identities) == 1, f"{contract_name} mixes run identities")

    if contract_name == "label_reveal":
        _require_contiguous_assets(records, label=contract_name)
        return
    if contract_name in {"score", "classification", "reproduction"}:
        for method in METHODS:
            method_records = [record for record in records if record["method"] == method]
            if method_records:
                _require_contiguous_assets(
                    method_records,
                    label=f"{contract_name}.{method}",
                )
                if contract_name == "reproduction" and records[0]["run_kind"] == "final_test":
                    _require(
                        len(method_records) == 10,
                        "final-test reproduction must contain the fixed first ten assets",
                    )
        return
    if contract_name == "latency_observation":
        for method in METHODS:
            method_records = [record for record in records if record["method"] == method]
            if not method_records:
                continue
            passes = {record["pass_index"] for record in method_records}
            _require(passes == {0, 1, 2}, f"{method} latency passes are incomplete")
            expected_assets: tuple[str, ...] | None = None
            for pass_index in range(3):
                pass_records = [
                    record for record in method_records if record["pass_index"] == pass_index
                ]
                _require_contiguous_assets(
                    pass_records,
                    label=f"latency_observation.{method}.pass_{pass_index}",
                )
                assets = tuple(str(record["asset_id"]) for record in pass_records)
                if expected_assets is None:
                    expected_assets = assets
                else:
                    _require(
                        assets == expected_assets,
                        f"{method} latency passes cover different assets",
                    )


def _validate_json_semantics(
    contract_name: str, record: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    if contract_name == "boundary_record":
        _require(record["reference_count"] == 20, "boundary reference count changed")
        _require(
            record["calibration_count"] > 0 and record["final_test_count"] > 0,
            "boundary counts must be positive",
        )
    elif contract_name == "pre_evaluation_freeze":
        _require(record["method_order"] == list(METHODS), "freeze method order changed")
        _require(record["hard_gate_order"] == list(HARD_GATES), "freeze gate order changed")
    elif contract_name == "fit":
        reference_count = record["reference_count"]
        successful = record["successful_reference_count"]
        failed = record["failed_reference_count"]
        _require(
            reference_count == 20 and successful + failed == 20,
            "fit reference counts are inconsistent",
        )
        if record["status"] == "fit_ok":
            _require(
                record["fitted_state_sha256"] is not None, "successful fit needs state identity"
            )
            _require(record["failure_code"] is None, "successful fit cannot have failure code")
        else:
            _require(record["fitted_state_sha256"] is None, "failed fit cannot have state identity")
            _require(isinstance(record["failure_code"], str), "failed fit needs failure code")
    elif contract_name == "calibration_summary":
        count = record["sample_count"]
        rank = record["rank"]
        anomalous = record["predicted_anomalous_count"]
        _require(count > 0 and rank == math.ceil(0.95 * count), "calibration rank is inconsistent")
        _require(0 <= anomalous <= count, "calibration anomalous count is invalid")
        _require(
            record["realized_normal_fpr"] == anomalous / count, "calibration FPR is inconsistent"
        )
    elif contract_name == "metrics":
        item_count = record["item_count"]
        normal_count = record["normal_count"]
        anomaly_count = record["anomaly_count"]
        tp = record["true_positive_count"]
        fn = record["false_negative_count"]
        tn = record["true_negative_count"]
        fp = record["false_positive_count"]
        _require(item_count == normal_count + anomaly_count, "metric item count is inconsistent")
        _require(
            anomaly_count == tp + fn and normal_count == tn + fp,
            "confusion counts are inconsistent",
        )
        _require(normal_count > 0 and anomaly_count > 0, "both metric classes are required")
        _require(
            record["normal_false_positive_rate"] == fp / normal_count, "normal FPR is inconsistent"
        )
        _require(record["anomaly_recall"] == tp / anomaly_count, "anomaly recall is inconsistent")
        for field in (
            "image_level_auroc",
            "image_level_auprc",
            "normal_false_positive_rate",
            "anomaly_recall",
        ):
            value = float(record[field])
            _require(0.0 <= value <= 1.0, f"{field} is outside [0, 1]")
    elif contract_name == "method_decision":
        outcomes = record["gate_outcomes"]
        _require(len(outcomes) == len(HARD_GATES), "gate outcome count changed")
        first_failed: str | None = None
        for index, (gate, outcome) in enumerate(zip(HARD_GATES, outcomes, strict=True), start=1):
            item = _mapping(outcome, label="gate outcome")
            _exact_keys(
                item,
                {"name", "observed", "operator", "order", "requirement", "status"},
                label="gate outcome",
            )
            _require(item["order"] == index and item["name"] == gate, "gate order changed")
            _require(item["status"] in {"pass", "fail", "not_evaluated"}, "gate status is invalid")
            if first_failed is None:
                _require(
                    item["status"] in {"pass", "fail"},
                    "gate was not evaluated before any failure",
                )
                if item["status"] == "fail":
                    first_failed = gate
            else:
                _require(item["status"] == "not_evaluated", "later gate evaluated after failure")
        _require(record["first_failed_gate"] == first_failed, "first failed gate is inconsistent")
        disposition = record["failure_review_disposition"]
        if first_failed is not None or disposition == "intended_use_contradicted":
            expected_decision = "REJECT"
        elif disposition == "guardrail_required":
            expected_decision = "ADOPT WITH CONDITIONS"
        else:
            expected_decision = "ADOPT"
        _require(record["decision"] == expected_decision, "method decision is inconsistent")
        _require(
            (record["condition"] is not None) is (expected_decision == "ADOPT WITH CONDITIONS"),
            "method condition is inconsistent",
        )
        _require(
            (record["failure_review_rationale"] is not None)
            is (disposition != "no_material_boundary"),
            "failure review rationale is inconsistent",
        )
    elif contract_name == "project_decision":
        method_decisions = record["method_decisions"]
        _require(set(method_decisions) == set(METHODS), "project method decisions changed")
        _require(
            all(
                value in config["decision"]["allowed_method_decisions"]
                for value in method_decisions.values()
            ),
            "project method decision is invalid",
        )
        selected = record["selected_method"]
        if all(value == "REJECT" for value in method_decisions.values()):
            _require(
                selected is None and record["decision"] == "REJECT",
                "all-reject project decision is inconsistent",
            )
        else:
            _require(selected in METHODS, "selected method is invalid")
            _require(
                method_decisions[selected] != "REJECT",
                "selected method must not be rejected",
            )
            _require(
                record["decision"] == method_decisions[selected],
                "project decision differs from selected method",
            )
        _require(bool(record["selection_trace"]), "selection trace must not be empty")
    elif contract_name == "bundle_manifest":
        files = record["files"]
        _require(bool(files), "bundle manifest files must not be empty")
        required = {"artifact_type", "method", "record_count", "relative_path", "sha256"}
        paths: list[str] = []
        for item in files:
            entry = _mapping(item, label="bundle manifest file")
            _exact_keys(entry, required, label="bundle manifest file")
            _relative_path(entry["relative_path"], label="bundle manifest relative_path")
            _require(
                SHA256_PATTERN.fullmatch(entry["sha256"]) is not None,
                "bundle file SHA-256 is invalid",
            )
            _integer(entry["record_count"], label="bundle record_count", minimum=0)
            _require(
                entry["relative_path"] != "artifact-manifest.json", "manifest cannot hash itself"
            )
            paths.append(entry["relative_path"])
        _require(
            paths == sorted(paths) and len(paths) == len(set(paths)),
            "bundle files are not unique and sorted",
        )


def validate_json_artifact(
    contract_name: str,
    record: object,
    *,
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one JSON artifact against exact fields and decision semantics."""
    validate_v0_2_config(config)
    validate_v0_2_artifact_schema(schema)
    if contract_name not in JSON_CONTRACTS:
        raise V0_2EvaluationContractError("unknown JSON contract")
    contract = schema["contracts"][contract_name]
    value = _mapping(record, label=f"{contract_name} artifact")
    required = set(contract["required_keys"])
    _exact_keys(value, required, label=f"{contract_name} artifact")
    nullable = set(contract.get("nullable_keys", []))
    enums = contract.get("field_enums", {})
    for name, field_type in contract["field_types"].items():
        item = value[name]
        if item is None:
            _require(name in nullable, f"{contract_name}.{name} must not be null")
            continue
        _validate_field_type(item, field_type, label=f"{contract_name}.{name}")
        if name in enums:
            _require(item in enums[name], f"{contract_name}.{name} is outside its enum")
    _require(RUN_ID_PATTERN.fullmatch(value["run_id"]) is not None, "run_id is invalid")
    _validate_json_semantics(contract_name, value, config)
    return value
