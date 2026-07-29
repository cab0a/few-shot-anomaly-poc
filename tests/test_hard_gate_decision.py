from __future__ import annotations

import inspect
import math
from dataclasses import fields, replace
from pathlib import Path

import pytest

import few_shot_anomaly_poc.hard_gate_decision as decision_module
from few_shot_anomaly_poc.calibration import CalibrationMethod
from few_shot_anomaly_poc.config import ProjectConfig, load_config
from few_shot_anomaly_poc.cpu_latency import (
    DEPENDENCY_DISTRIBUTIONS,
    THREAD_ENVIRONMENT_VARIABLES,
    CPUEnvironmentRecord,
    CPULatencyObservation,
    CPULatencyResult,
)
from few_shot_anomaly_poc.errors import HardGateDecisionFailureCode
from few_shot_anomaly_poc.hard_gate_decision import (
    DecisionProcessEvidence,
    FailureReviewDisposition,
    HardGateDecisionResult,
    apply_hard_gate_decision,
)
from few_shot_anomaly_poc.image_metrics import ImageLevelMetricsResult


@pytest.fixture(scope="module")
def project_config() -> ProjectConfig:
    return load_config(Path("configs/v0.1.yaml"))


def _metrics(
    *,
    method: CalibrationMethod = CalibrationMethod.ECC_RESIDUAL,
    normal_false_positive_count: int = 1,
    anomaly_true_positive_count: int = 18,
) -> ImageLevelMetricsResult:
    normal_count = 20
    anomaly_count = 20
    return ImageLevelMetricsResult(
        status="ok",
        failure_code=None,
        method=method,
        positive_class="anomaly",
        item_count=normal_count + anomaly_count,
        normal_count=normal_count,
        anomaly_count=anomaly_count,
        true_positive_count=anomaly_true_positive_count,
        false_negative_count=anomaly_count - anomaly_true_positive_count,
        true_negative_count=normal_count - normal_false_positive_count,
        false_positive_count=normal_false_positive_count,
        score_failure_count=0,
        image_level_auroc=0.9,
        image_level_auprc=0.9,
        normal_false_positive_rate=normal_false_positive_count / normal_count,
        anomaly_recall=anomaly_true_positive_count / anomaly_count,
        threshold=0.5,
        threshold_source_path="pcb1/Data/Images/Normal/0018.JPG",
    )


def _latency(
    p95_seconds: float = 1.0,
    *,
    method: CalibrationMethod = CalibrationMethod.ECC_RESIDUAL,
) -> CPULatencyResult:
    duration_ns = int(p95_seconds * 1e9)
    paths = tuple(f"pcb1/Data/Images/Test/{index:04d}.JPG" for index in range(40))
    observations = tuple(
        CPULatencyObservation(
            pass_index=pass_index,
            relative_path=path,
            duration_ns=duration_ns,
            score_status="ok",
            score_failure_code=None,
        )
        for pass_index in range(1, 4)
        for path in paths
    )
    sample_count = len(observations)
    return CPULatencyResult(
        status="ok",
        failure_code=None,
        method=method,
        measurement_boundary="decoded_grayscale_uint8_to_image_score",
        timer="perf_counter_ns",
        warmup_passes=1,
        timed_passes=3,
        item_count=len(paths),
        completed_warmup_invocations=len(paths),
        completed_timed_invocations=sample_count,
        sample_count=sample_count,
        ordered_paths=paths,
        observations=observations,
        score_failure_timing_count=0,
        score_failure_paths=(),
        median_latency_ns=float(duration_ns),
        p95_latency_ns=duration_ns,
        median_latency_seconds=duration_ns / 1e9,
        p95_latency_seconds=duration_ns / 1e9,
        p95_rank=math.ceil(0.95 * sample_count),
        environment=CPUEnvironmentRecord(
            cpu_model="Synthetic CPU",
            logical_core_count=8,
            physical_core_count=4,
            ram_bytes=16 * 1024**3,
            operating_system="Synthetic OS",
            machine="x86_64",
            python_version="3.13.14",
            opencv_thread_count=1,
            dependency_versions=tuple(
                (name, "synthetic-version") for name in DEPENDENCY_DISTRIBUTIONS
            ),
            thread_environment=tuple((name, None) for name in THREAD_ENVIRONMENT_VARIABLES),
        ),
        failed_phase=None,
        failed_path=None,
        failed_pass_index=None,
    )


def _process(
    *,
    normal_reference_count: int = 20,
    anomaly_training_labels_used: bool = False,
    reproducibility_verified: bool = True,
    test_leakage_detected: bool = False,
    disposition: FailureReviewDisposition = (FailureReviewDisposition.NO_MATERIAL_BOUNDARY),
    condition: str | None = None,
) -> DecisionProcessEvidence:
    return DecisionProcessEvidence(
        normal_reference_count=normal_reference_count,
        anomaly_training_labels_used=anomaly_training_labels_used,
        reproducibility_verified=reproducibility_verified,
        test_leakage_detected=test_leakage_detected,
        failure_review_disposition=disposition,
        failure_review_rationale="Synthetic failure review for a primitive test.",
        condition=condition,
    )


def test_decision_interface_exposes_no_gate_or_score_override() -> None:
    parameters = inspect.signature(apply_hard_gate_decision).parameters
    result_fields = {field.name for field in fields(HardGateDecisionResult)}

    assert tuple(parameters) == ("metrics", "latency", "process_evidence", "config")
    assert "weighted_score" not in parameters
    assert "gate_overrides" not in parameters
    assert "threshold" not in parameters
    assert "weighted_score" not in result_fields


def test_decision_passes_exact_gate_boundaries_in_fixed_order(
    project_config: ProjectConfig,
) -> None:
    result = apply_hard_gate_decision(
        _metrics(),
        _latency(),
        _process(),
        config=project_config,
    )

    assert result.succeeded
    assert result.decision == "ADOPT"
    assert result.all_hard_gates_passed is True
    assert result.first_failed_gate is None
    assert result.decision_reason == "all_hard_gates_passed"
    assert result.failure_review_rationale == ("Synthetic failure review for a primitive test.")
    assert result.gate_outcomes is not None
    assert tuple(outcome.gate_id for outcome in result.gate_outcomes) == (
        "final_test_normal_fpr",
        "final_test_anomaly_recall",
        "cpu_p95_scoring_latency",
        "normal_reference_count",
        "anomaly_training_labels",
        "reproducibility",
    )
    assert tuple(outcome.order for outcome in result.gate_outcomes) == (
        1,
        2,
        3,
        4,
        5,
        6,
    )
    assert all(outcome.passed for outcome in result.gate_outcomes)


def test_each_hard_gate_failure_forces_reject(
    project_config: ProjectConfig,
) -> None:
    cases = (
        (
            _metrics(normal_false_positive_count=2),
            _latency(),
            _process(),
            "final_test_normal_fpr",
        ),
        (
            _metrics(anomaly_true_positive_count=17),
            _latency(),
            _process(),
            "final_test_anomaly_recall",
        ),
        (
            _metrics(),
            _latency(1.000000001),
            _process(),
            "cpu_p95_scoring_latency",
        ),
        (
            _metrics(),
            _latency(),
            _process(normal_reference_count=21),
            "normal_reference_count",
        ),
        (
            _metrics(),
            _latency(),
            _process(anomaly_training_labels_used=True),
            "anomaly_training_labels",
        ),
        (
            _metrics(),
            _latency(),
            _process(reproducibility_verified=False),
            "reproducibility",
        ),
    )

    for metrics, latency, process, expected_gate in cases:
        result = apply_hard_gate_decision(
            metrics,
            latency,
            process,
            config=project_config,
        )
        assert result.succeeded
        assert result.decision == "REJECT"
        assert result.all_hard_gates_passed is False
        assert result.first_failed_gate == expected_gate
        assert result.decision_reason == "hard_gate_failed"


def test_first_failed_gate_follows_preregistered_order(
    project_config: ProjectConfig,
) -> None:
    result = apply_hard_gate_decision(
        _metrics(
            normal_false_positive_count=2,
            anomaly_true_positive_count=17,
        ),
        _latency(1.1),
        _process(normal_reference_count=21),
        config=project_config,
    )

    assert result.decision == "REJECT"
    assert result.first_failed_gate == "final_test_normal_fpr"
    assert result.gate_outcomes is not None
    assert sum(not outcome.passed for outcome in result.gate_outcomes) == 4


def test_guardrail_requires_adopt_with_conditions(
    project_config: ProjectConfig,
) -> None:
    result = apply_hard_gate_decision(
        _metrics(),
        _latency(0.5),
        _process(
            disposition=FailureReviewDisposition.GUARDRAIL_REQUIRED,
            condition="Route low-contrast inputs to manual review.",
        ),
        config=project_config,
    )

    assert result.decision == "ADOPT WITH CONDITIONS"
    assert result.all_hard_gates_passed is True
    assert result.condition == "Route low-contrast inputs to manual review."
    assert result.decision_reason == "all_hard_gates_passed_guardrail_required"


def test_failure_review_can_reject_but_cannot_waive_a_hard_gate(
    project_config: ProjectConfig,
) -> None:
    contradicted = apply_hard_gate_decision(
        _metrics(),
        _latency(0.5),
        _process(
            disposition=FailureReviewDisposition.INTENDED_USE_CONTRADICTED,
        ),
        config=project_config,
    )
    conditional_failure = apply_hard_gate_decision(
        _metrics(anomaly_true_positive_count=17),
        _latency(0.5),
        _process(
            disposition=FailureReviewDisposition.GUARDRAIL_REQUIRED,
            condition="Manual review.",
        ),
        config=project_config,
    )

    assert contradicted.decision == "REJECT"
    assert contradicted.all_hard_gates_passed is True
    assert contradicted.decision_reason == "failure_review_contradicts_intended_use"
    assert conditional_failure.decision == "REJECT"
    assert conditional_failure.all_hard_gates_passed is False
    assert conditional_failure.condition is None
    assert conditional_failure.decision_reason == "hard_gate_failed"


def test_test_leakage_forces_reject_after_gate_recording(
    project_config: ProjectConfig,
) -> None:
    result = apply_hard_gate_decision(
        _metrics(),
        _latency(0.5),
        _process(test_leakage_detected=True),
        config=project_config,
    )

    assert result.decision == "REJECT"
    assert result.all_hard_gates_passed is True
    assert result.first_failed_gate is None
    assert result.test_leakage_detected is True
    assert result.decision_reason == "test_leakage_detected"


@pytest.mark.parametrize(
    "process",
    [
        replace(
            _process(),
            failure_review_rationale=" ",
        ),
        _process(
            disposition=FailureReviewDisposition.GUARDRAIL_REQUIRED,
            condition=None,
        ),
        _process(condition="Unexpected condition."),
        replace(_process(), normal_reference_count=True),
    ],
)
def test_decision_rejects_invalid_process_evidence(
    process: DecisionProcessEvidence,
    project_config: ProjectConfig,
) -> None:
    result = apply_hard_gate_decision(
        _metrics(),
        _latency(0.5),
        process,
        config=project_config,
    )

    assert not result.succeeded
    assert result.failure_code is HardGateDecisionFailureCode.DECISION_PROCESS_EVIDENCE_INVALID
    assert result.decision is None
    assert result.gate_outcomes is None
    assert result.failure_review_rationale is None


def test_decision_rejects_invalid_or_mismatched_metric_and_latency_evidence(
    project_config: ProjectConfig,
) -> None:
    invalid_metrics = apply_hard_gate_decision(
        replace(_metrics(), image_level_auroc=float("nan")),
        _latency(0.5),
        _process(),
        config=project_config,
    )
    mismatched_method = apply_hard_gate_decision(
        _metrics(),
        _latency(0.5, method=CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM),
        _process(),
        config=project_config,
    )

    assert invalid_metrics.failure_code is HardGateDecisionFailureCode.DECISION_METRICS_INVALID
    assert mismatched_method.failure_code is HardGateDecisionFailureCode.DECISION_METHOD_MISMATCH


def test_decision_is_repeatable_and_does_not_mutate_inputs(
    project_config: ProjectConfig,
) -> None:
    metrics = _metrics()
    latency = _latency(0.5)
    process = _process()

    first = apply_hard_gate_decision(
        metrics,
        latency,
        process,
        config=project_config,
    )
    second = apply_hard_gate_decision(
        metrics,
        latency,
        process,
        config=project_config,
    )

    assert first == second
    assert metrics == _metrics()
    assert latency == _latency(0.5)
    assert process == _process()


def test_decision_rejects_invalid_internal_result(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
) -> None:
    monkeypatch.setattr(decision_module, "_result_is_valid", lambda *args, **kwargs: False)

    result = apply_hard_gate_decision(
        _metrics(),
        _latency(0.5),
        _process(),
        config=project_config,
    )

    assert result.failure_code is HardGateDecisionFailureCode.DECISION_RESULT_INVALID
    assert result.decision is None
