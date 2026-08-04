from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    ARTIFACT_CONTRACT_VERSION,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_SCHEMA_SHA256,
    HARD_GATES,
    METHODS,
    V0_2EvaluationContractError,
    load_v0_2_artifact_schema,
    load_v0_2_config,
    validate_json_artifact,
    validate_repository_contract,
    validate_tabular_record,
    validate_tabular_records,
    validate_v0_2_artifact_schema,
    validate_v0_2_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/v0.2.yaml"
SCHEMA_PATH = ROOT / "schemas/v0.2/evaluation-artifacts.json"
RUN_ID = "synthetic-contract"
SHA256 = "a" * 64
COMMIT = "b" * 40


def _repository_contract() -> tuple[dict, dict]:
    return validate_repository_contract(
        config_path=CONFIG_PATH,
        schema_path=SCHEMA_PATH,
    )


def _score_record(
    method: str = "ecc_residual",
    *,
    asset_index: int = 0,
    score: float = 0.2,
    status: str = "ok",
    failure_code: str | None = None,
) -> dict:
    return {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "run_id": RUN_ID,
        "run_kind": "synthetic",
        "method": method,
        "asset_id": f"asset-{asset_index:06d}",
        "score_status": status,
        "score_failure_code": failure_code,
        "anomaly_score": score,
        "diagnostics_json": '{"patch_count":256}',
    }


def _classification_record(
    method: str = "ecc_residual",
    *,
    asset_index: int = 0,
    score: float = 0.4,
    threshold: float = 0.3,
) -> dict:
    anomalous = score > threshold
    return {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "run_id": RUN_ID,
        "run_kind": "synthetic",
        "method": method,
        "asset_id": f"asset-{asset_index:06d}",
        "score_status": "ok",
        "score_failure_code": None,
        "anomaly_score": score,
        "threshold": threshold,
        "predicted_class": "anomalous" if anomalous else "normal",
        "is_anomalous": anomalous,
        "decision_reason": (
            "score_strictly_greater_than_threshold"
            if anomalous
            else "score_not_greater_than_threshold"
        ),
        "score_margin": score - threshold,
    }


def _latency_record(
    method: str,
    *,
    pass_index: int,
    asset_index: int,
) -> dict:
    is_dino = method == "dinov2_vits14_224_nn"
    adapter_ns = 20 if is_dino else None
    scorer_ns = 80
    return {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "run_id": RUN_ID,
        "run_kind": "synthetic",
        "method": method,
        "pass_index": pass_index,
        "asset_id": f"asset-{asset_index:06d}",
        "adapter_duration_ns": adapter_ns,
        "scorer_duration_ns": scorer_ns,
        "duration_ns": scorer_ns + (adapter_ns or 0),
        "score_status": "ok",
        "score_failure_code": None,
        "anomaly_score": 0.2,
    }


def _reproduction_record(method: str, *, asset_index: int = 0) -> dict:
    expected_score = 0.2
    reproduced_score = 0.2000005
    return {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "run_id": RUN_ID,
        "run_kind": "synthetic",
        "method": method,
        "asset_id": f"asset-{asset_index:06d}",
        "expected_score_status": "ok",
        "reproduced_score_status": "ok",
        "expected_failure_code": None,
        "reproduced_failure_code": None,
        "expected_score": expected_score,
        "reproduced_score": reproduced_score,
        "absolute_difference": abs(expected_score - reproduced_score),
        "within_tolerance": True,
    }


def _json_common(method: str | None = None) -> dict:
    common = {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "run_id": RUN_ID,
        "run_kind": "synthetic",
    }
    if method is not None:
        common["method"] = method
    return common


def _gate_outcomes(failed_gate: str | None = None) -> list[dict]:
    outcomes = []
    failure_seen = False
    for order, gate in enumerate(HARD_GATES, start=1):
        if failure_seen:
            status = "not_evaluated"
        elif gate == failed_gate:
            status = "fail"
            failure_seen = True
        else:
            status = "pass"
        outcomes.append(
            {
                "order": order,
                "name": gate,
                "status": status,
                "observed": "synthetic observation",
                "operator": "fixed",
                "requirement": "preregistered requirement",
            }
        )
    return outcomes


def _method_decision(
    *,
    failed_gate: str | None = None,
    disposition: str = "no_material_boundary",
) -> dict:
    if failed_gate is not None or disposition == "intended_use_contradicted":
        decision = "REJECT"
    elif disposition == "guardrail_required":
        decision = "ADOPT WITH CONDITIONS"
    else:
        decision = "ADOPT"
    return {
        **_json_common("ecc_residual"),
        "decision": decision,
        "gate_outcomes": _gate_outcomes(failed_gate),
        "first_failed_gate": failed_gate,
        "failure_review_disposition": disposition,
        "failure_review_rationale": (
            None if disposition == "no_material_boundary" else "Synthetic failure-review rationale."
        ),
        "condition": "Synthetic guardrail" if decision == "ADOPT WITH CONDITIONS" else None,
        "decision_reason": "Synthetic decision fixture.",
        "weighted_score_used": False,
        "hard_gate_waiver_used": False,
    }


def test_fixed_config_and_schema_identities_cross_validate() -> None:
    config, schema = _repository_contract()

    assert load_v0_2_config(CONFIG_PATH) == config
    assert load_v0_2_artifact_schema(SCHEMA_PATH) == schema
    assert config["schema_version"] == "v0.2-evaluation-contract-v1"
    assert config["method_order"] == list(METHODS)
    assert config["hard_gates"]["order"] == list(HARD_GATES)
    assert schema["contract_version"] == ARTIFACT_CONTRACT_VERSION
    assert (
        EXPECTED_CONFIG_SHA256 == "9ea3a7156aeb3c6efc87c8ae3811444421bd3e86ea25b7ac014893c7e5892265"
    )
    assert (
        EXPECTED_SCHEMA_SHA256 == "4178d24f7210f8f859f6c99386204022ac3b2e4ffe4b7b63cdfea00d0c79f31d"
    )


def test_semantic_config_rejects_method_resolution_and_gate_changes() -> None:
    config = load_v0_2_config(CONFIG_PATH)
    changed_resolution = deepcopy(config)
    changed_resolution["methods"]["dinov2_vits14_224_nn"]["resolution"] = 448
    changed_gate = deepcopy(config)
    changed_gate["hard_gates"]["anomaly_recall_min"] = 0.80

    with pytest.raises(V0_2EvaluationContractError):
        validate_v0_2_config(changed_resolution)
    with pytest.raises(V0_2EvaluationContractError):
        validate_v0_2_config(changed_gate)


def test_label_free_schema_rejects_protected_fields() -> None:
    schema = load_v0_2_artifact_schema(SCHEMA_PATH)
    changed = deepcopy(schema)
    changed["contracts"]["score"]["columns"].append(
        {"name": "true_class", "type": "string", "nullable": False}
    )

    with pytest.raises(V0_2EvaluationContractError, match="protected field"):
        validate_v0_2_artifact_schema(changed)


@pytest.mark.parametrize("method", METHODS)
def test_successful_scores_validate_for_every_fixed_method(method: str) -> None:
    _, schema = _repository_contract()

    assert (
        validate_tabular_record("score", _score_record(method), schema=schema)["method"] == method
    )


@pytest.mark.parametrize(
    ("method", "failure_score"),
    [
        ("ecc_residual", 1.0),
        ("patch_hog_ocsvm", 1e12),
        ("dinov2_vits14_224_nn", 2.0),
    ],
)
def test_failed_scores_require_the_method_specific_fixed_score(
    method: str,
    failure_score: float,
) -> None:
    _, schema = _repository_contract()
    valid = _score_record(
        method,
        score=failure_score,
        status="failed",
        failure_code="synthetic_failure",
    )
    wrong_score = {**valid, "anomaly_score": failure_score - 0.1}

    validate_tabular_record("score", valid, schema=schema)
    with pytest.raises(V0_2EvaluationContractError, match="fixed failure score"):
        validate_tabular_record("score", wrong_score, schema=schema)


def test_score_rejects_noncanonical_diagnostics_and_extra_label() -> None:
    _, schema = _repository_contract()
    noncanonical = {**_score_record(), "diagnostics_json": '{"z": 1, "a": 2}'}
    label_leak = {**_score_record(), "true_class": "anomaly"}

    with pytest.raises(V0_2EvaluationContractError, match="canonical JSON"):
        validate_tabular_record("score", noncanonical, schema=schema)
    with pytest.raises(V0_2EvaluationContractError, match="fields differ"):
        validate_tabular_record("score", label_leak, schema=schema)


def test_classification_uses_strict_threshold_and_rejects_inconsistent_margin() -> None:
    _, schema = _repository_contract()
    equal = _classification_record(score=0.3, threshold=0.3)
    wrong_margin = {**equal, "score_margin": 0.1}

    validated = validate_tabular_record("classification", equal, schema=schema)
    assert validated["predicted_class"] == "normal"
    assert validated["is_anomalous"] is False
    with pytest.raises(V0_2EvaluationContractError, match="score_margin"):
        validate_tabular_record("classification", wrong_margin, schema=schema)


def test_latency_enforces_classical_and_dinov2_measurement_boundaries() -> None:
    _, schema = _repository_contract()
    classical = _latency_record("ecc_residual", pass_index=0, asset_index=0)
    dino = _latency_record("dinov2_vits14_224_nn", pass_index=0, asset_index=0)
    wrong_dino = {**dino, "duration_ns": 99}

    validate_tabular_record("latency_observation", classical, schema=schema)
    validate_tabular_record("latency_observation", dino, schema=schema)
    with pytest.raises(V0_2EvaluationContractError, match="adapter plus scorer"):
        validate_tabular_record("latency_observation", wrong_dino, schema=schema)


def test_latency_collection_requires_three_equal_complete_passes() -> None:
    _, schema = _repository_contract()
    records = [
        _latency_record("ecc_residual", pass_index=pass_index, asset_index=asset_index)
        for pass_index in range(3)
        for asset_index in range(2)
    ]

    validate_tabular_records("latency_observation", records, schema=schema)
    with pytest.raises(V0_2EvaluationContractError, match="different assets"):
        validate_tabular_records("latency_observation", records[:-1], schema=schema)


def test_reproduction_computes_difference_and_tolerance_from_scores() -> None:
    _, schema = _repository_contract()
    valid = _reproduction_record("dinov2_vits14_224_nn")
    wrong = {**valid, "within_tolerance": False}

    validate_tabular_record("reproduction", valid, schema=schema)
    with pytest.raises(V0_2EvaluationContractError, match="tolerance result"):
        validate_tabular_record("reproduction", wrong, schema=schema)


def test_ordered_collections_reject_gaps_duplicates_and_mixed_runs() -> None:
    _, schema = _repository_contract()
    records = [_score_record(asset_index=0), _score_record(asset_index=1)]
    gap = [_score_record(asset_index=0), _score_record(asset_index=2)]
    duplicate = [_score_record(asset_index=0), _score_record(asset_index=0)]
    mixed_run = [records[0], {**records[1], "run_id": "another-run"}]

    validate_tabular_records("score", records, schema=schema)
    with pytest.raises(V0_2EvaluationContractError, match="contiguous"):
        validate_tabular_records("score", gap, schema=schema)
    with pytest.raises(V0_2EvaluationContractError, match="primary keys"):
        validate_tabular_records("score", duplicate, schema=schema)
    with pytest.raises(V0_2EvaluationContractError, match="mixes run identities"):
        validate_tabular_records("score", mixed_run, schema=schema)


def test_label_reveal_is_separate_and_contains_no_score_or_prediction() -> None:
    _, schema = _repository_contract()
    labels = [
        {
            "contract_version": ARTIFACT_CONTRACT_VERSION,
            "run_id": RUN_ID,
            "run_kind": "synthetic",
            "asset_id": f"asset-{index:06d}",
            "source_path": f"Data/Images/Anomaly/example-{index}.JPG",
            "true_class": "anomaly",
        }
        for index in range(2)
    ]

    validated = validate_tabular_records("label_reveal", labels, schema=schema)
    assert all("anomaly_score" not in record for record in validated)
    assert all("predicted_class" not in record for record in validated)


def test_fit_and_calibration_summaries_enforce_preregistered_counts() -> None:
    config, schema = _repository_contract()
    fit = {
        **_json_common("ecc_residual"),
        "status": "fit_ok",
        "reference_count": 20,
        "successful_reference_count": 20,
        "failed_reference_count": 0,
        "reference_manifest_sha256": SHA256,
        "fitted_state_sha256": SHA256,
        "failure_code": None,
    }
    calibration = {
        **_json_common("ecc_residual"),
        "sample_count": 20,
        "rank": 19,
        "quantile": 0.95,
        "threshold": 0.3,
        "threshold_source_path": "Data/Images/Normal/example.JPG",
        "predicted_anomalous_count": 1,
        "score_failure_count": 0,
        "realized_normal_fpr": 0.05,
    }

    validate_json_artifact("fit", fit, config=config, schema=schema)
    validate_json_artifact("calibration_summary", calibration, config=config, schema=schema)
    with pytest.raises(V0_2EvaluationContractError, match="fit reference counts"):
        validate_json_artifact(
            "fit",
            {**fit, "successful_reference_count": 19},
            config=config,
            schema=schema,
        )


def test_metrics_enforce_counts_rates_and_bounded_ranking_metrics() -> None:
    config, schema = _repository_contract()
    metrics = {
        **_json_common("ecc_residual"),
        "positive_class": "anomaly",
        "item_count": 20,
        "normal_count": 10,
        "anomaly_count": 10,
        "true_positive_count": 9,
        "false_negative_count": 1,
        "true_negative_count": 9,
        "false_positive_count": 1,
        "score_failure_count": 0,
        "image_level_auroc": 0.91,
        "image_level_auprc": 0.92,
        "normal_false_positive_rate": 0.1,
        "anomaly_recall": 0.9,
        "threshold": 0.3,
    }

    validate_json_artifact("metrics", metrics, config=config, schema=schema)
    with pytest.raises(V0_2EvaluationContractError, match="normal FPR"):
        validate_json_artifact(
            "metrics",
            {**metrics, "normal_false_positive_rate": 0.05},
            config=config,
            schema=schema,
        )


@pytest.mark.parametrize(
    ("record", "expected_decision"),
    [
        (_method_decision(), "ADOPT"),
        (
            _method_decision(disposition="guardrail_required"),
            "ADOPT WITH CONDITIONS",
        ),
        (_method_decision(failed_gate="final_test_normal_fpr"), "REJECT"),
    ],
)
def test_method_decision_follows_ordered_hard_gates(
    record: dict,
    expected_decision: str,
) -> None:
    config, schema = _repository_contract()

    validated = validate_json_artifact(
        "method_decision",
        record,
        config=config,
        schema=schema,
    )
    assert validated["decision"] == expected_decision


def test_method_decision_rejects_evaluating_a_gate_after_failure() -> None:
    config, schema = _repository_contract()
    record = _method_decision(failed_gate="final_test_normal_fpr")
    record["gate_outcomes"][3]["status"] = "pass"

    with pytest.raises(V0_2EvaluationContractError, match="after failure"):
        validate_json_artifact(
            "method_decision",
            record,
            config=config,
            schema=schema,
        )


def test_method_decision_rejects_an_unexplained_unevaluated_gate() -> None:
    config, schema = _repository_contract()
    record = _method_decision()
    record["gate_outcomes"][0]["status"] = "not_evaluated"

    with pytest.raises(V0_2EvaluationContractError, match="before any failure"):
        validate_json_artifact(
            "method_decision",
            record,
            config=config,
            schema=schema,
        )


def test_project_decision_and_manifest_enforce_selection_and_provenance() -> None:
    config, schema = _repository_contract()
    project_decision = {
        **_json_common(),
        "decision": "ADOPT WITH CONDITIONS",
        "selected_method": "patch_hog_ocsvm",
        "method_decisions": {
            "ecc_residual": "REJECT",
            "patch_hog_ocsvm": "ADOPT WITH CONDITIONS",
            "dinov2_vits14_224_nn": "REJECT",
        },
        "selection_trace": ["patch_hog_ocsvm is the only non-rejected method"],
        "decision_reason": "Synthetic selection fixture.",
        "next_validation": "Validate the stated condition.",
        "weighted_score_used": False,
    }
    manifest = {
        **_json_common(),
        "source_commit": COMMIT,
        "config_sha256": SHA256,
        "schema_sha256": SHA256,
        "preregistration_commit": COMMIT,
        "preregistration_document_sha256": SHA256,
        "scoring_manifest_sha256": SHA256,
        "sealed_mapping_sha256": SHA256,
        "files": [
            {
                "artifact_type": "score",
                "method": "ecc_residual",
                "record_count": 2,
                "relative_path": "ecc_residual/scores.csv",
                "sha256": SHA256,
            }
        ],
    }

    validate_json_artifact(
        "project_decision",
        project_decision,
        config=config,
        schema=schema,
    )
    rejected_selection = {
        **project_decision,
        "decision": "REJECT",
        "selected_method": "ecc_residual",
    }
    with pytest.raises(V0_2EvaluationContractError, match="must not be rejected"):
        validate_json_artifact(
            "project_decision",
            rejected_selection,
            config=config,
            schema=schema,
        )
    validate_json_artifact(
        "bundle_manifest",
        manifest,
        config=config,
        schema=schema,
    )
    manifest["files"][0]["relative_path"] = "artifact-manifest.json"
    with pytest.raises(V0_2EvaluationContractError, match="cannot hash itself"):
        validate_json_artifact(
            "bundle_manifest",
            manifest,
            config=config,
            schema=schema,
        )


def test_pre_reveal_checkpoint_requires_all_fixed_methods_and_statuses() -> None:
    config, schema = _repository_contract()
    checkpoint = {
        **_json_common(),
        "source_commit": COMMIT,
        "label_free_bundle_sha256": SHA256,
        "method_order": list(METHODS),
        "method_score_counts": {method: 200 for method in METHODS},
        "reproduction_status": {method: "pass" for method in METHODS},
        "git_commit": COMMIT,
        "git_push_verified": True,
        "labels_accessed": False,
    }

    validate_json_artifact(
        "pre_reveal_checkpoint",
        checkpoint,
        config=config,
        schema=schema,
    )
    changed = deepcopy(checkpoint)
    changed["reproduction_status"]["ecc_residual"] = "retry"
    with pytest.raises(V0_2EvaluationContractError, match="is invalid"):
        validate_json_artifact(
            "pre_reveal_checkpoint",
            changed,
            config=config,
            schema=schema,
        )


def test_contract_validation_does_not_create_evaluation_artifacts(tmp_path: Path) -> None:
    config, schema = _repository_contract()
    validate_tabular_record("score", _score_record(), schema=schema)
    validate_json_artifact(
        "method_decision",
        _method_decision(),
        config=config,
        schema=schema,
    )

    assert list(tmp_path.iterdir()) == []
