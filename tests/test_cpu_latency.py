from __future__ import annotations

import inspect
import math
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

import few_shot_anomaly_poc.cpu_latency as latency_module
from few_shot_anomaly_poc.calibration import CalibrationMethod
from few_shot_anomaly_poc.config import ProjectConfig, load_config
from few_shot_anomaly_poc.cpu_latency import (
    DEPENDENCY_DISTRIBUTIONS,
    THREAD_ENVIRONMENT_VARIABLES,
    CPUEnvironmentRecord,
    CPULatencyResult,
    measure_cpu_latency,
)
from few_shot_anomaly_poc.ecc_residual import ECCResidualScoreResult
from few_shot_anomaly_poc.errors import (
    CPULatencyFailureCode,
    HOGScoringFailureCode,
    PreprocessingFailureCode,
)
from few_shot_anomaly_poc.hog_scoring import PatchHOGScoreResult


@pytest.fixture(scope="module")
def project_config() -> ProjectConfig:
    return load_config(Path("configs/v0.1.yaml"))


@pytest.fixture
def fixed_environment(monkeypatch: pytest.MonkeyPatch) -> CPUEnvironmentRecord:
    environment = CPUEnvironmentRecord(
        cpu_model="Synthetic CPU",
        logical_core_count=8,
        physical_core_count=4,
        ram_bytes=16 * 1024**3,
        operating_system="Synthetic OS",
        machine="x86_64",
        python_version="3.13.14",
        opencv_thread_count=1,
        dependency_versions=tuple((name, "synthetic-version") for name in DEPENDENCY_DISTRIBUTIONS),
        thread_environment=tuple((name, None) for name in THREAD_ENVIRONMENT_VARIABLES),
    )
    monkeypatch.setattr(
        latency_module,
        "_capture_cpu_environment",
        lambda: environment,
    )
    return environment


def _ecc_success(score: float = 0.1) -> ECCResidualScoreResult:
    return ECCResidualScoreResult(
        score_status="ok",
        failure_code=None,
        anomaly_score=score,
        registration_status="ok",
        correlation=1.0,
        warp_matrix=np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        rotation_degrees=0.0,
        translation_x_pixels=0.0,
        translation_y_pixels=0.0,
        registration_valid_fraction=1.0,
        effective_support_fraction=1.0,
        effective_pixel_count=1,
        top_pixel_count=1,
    )


def _ecc_failure() -> ECCResidualScoreResult:
    return ECCResidualScoreResult(
        score_status="failed",
        failure_code=PreprocessingFailureCode.INVALID_PREPROCESSED_IMAGE,
        anomaly_score=1.0,
        registration_status="not_run",
        correlation=None,
        warp_matrix=None,
        rotation_degrees=None,
        translation_x_pixels=None,
        translation_y_pixels=None,
        registration_valid_fraction=None,
        effective_support_fraction=None,
        effective_pixel_count=None,
        top_pixel_count=None,
    )


def _hog_success(score: float = -0.1) -> PatchHOGScoreResult:
    return PatchHOGScoreResult(
        score_status="ok",
        failure_code=None,
        anomaly_score=score,
        patch_anomaly_scores=tuple(score for _ in range(225)),
        top_patch_count=12,
        top_patch_indices=tuple(range(12)),
        successful_patch_count=225,
        failed_patch_index=None,
    )


def _hog_failure() -> PatchHOGScoreResult:
    return PatchHOGScoreResult(
        score_status="failed",
        failure_code=HOGScoringFailureCode.HOG_SCORE_DECISION_FAILED,
        anomaly_score=1e12,
        patch_anomaly_scores=None,
        top_patch_count=None,
        top_patch_indices=(),
        successful_patch_count=3,
        failed_patch_index=3,
    )


def _timer_for_durations(durations: tuple[int, ...]):
    values = []
    cursor = 1_000
    for duration in durations:
        values.extend((cursor, cursor + duration))
        cursor += duration + 1_000
    iterator = iter(values)
    return lambda: next(iterator)


def test_latency_interface_exposes_no_label_threshold_or_gate() -> None:
    parameters = inspect.signature(measure_cpu_latency).parameters
    result_fields = {field.name for field in fields(CPULatencyResult)}

    assert tuple(parameters) == ("decoded_images", "score_one", "method", "config")
    assert all("label" not in name for name in parameters)
    assert "threshold" not in parameters
    assert {"gate", "passes_gate", "decision"}.isdisjoint(result_fields)


def test_cpu_environment_capture_records_required_context() -> None:
    environment = latency_module._capture_cpu_environment()

    assert latency_module._environment_is_valid(environment)
    assert environment.logical_core_count is None or environment.logical_core_count > 0
    assert tuple(name for name, _ in environment.dependency_versions) == (DEPENDENCY_DISTRIBUTIONS)
    assert all(value is not None for _, value in environment.dependency_versions)
    assert tuple(name for name, _ in environment.thread_environment) == (
        THREAD_ENVIRONMENT_VARIABLES
    )


def test_latency_runs_fixed_warmup_and_timed_passes_in_path_order(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    fixed_environment: CPUEnvironmentRecord,
) -> None:
    decoded_images = {
        "pcb1/Data/Images/Test/b.JPG": "decoded-b",
        "pcb1/Data/Images/Test/a.JPG": "decoded-a",
    }
    original_items = tuple(decoded_images.items())
    calls = []

    def score_one(decoded):
        calls.append(decoded)
        return _ecc_success()

    monkeypatch.setattr(
        latency_module.time,
        "perf_counter_ns",
        _timer_for_durations((10, 20, 30, 40, 50, 60)),
    )

    result = measure_cpu_latency(
        decoded_images,
        score_one,
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )

    assert result.succeeded
    assert result.status == "ok"
    assert result.failure_code is None
    assert result.method is CalibrationMethod.ECC_RESIDUAL
    assert result.measurement_boundary == "decoded_grayscale_uint8_to_image_score"
    assert result.timer == "perf_counter_ns"
    assert result.warmup_passes == 1
    assert result.timed_passes == 3
    assert result.item_count == 2
    assert result.completed_warmup_invocations == 2
    assert result.completed_timed_invocations == 6
    assert result.sample_count == 6
    assert result.ordered_paths == (
        "pcb1/Data/Images/Test/a.JPG",
        "pcb1/Data/Images/Test/b.JPG",
    )
    assert result.observations is not None
    assert tuple(
        (item.pass_index, item.relative_path, item.duration_ns) for item in result.observations
    ) == (
        (1, "pcb1/Data/Images/Test/a.JPG", 10),
        (1, "pcb1/Data/Images/Test/b.JPG", 20),
        (2, "pcb1/Data/Images/Test/a.JPG", 30),
        (2, "pcb1/Data/Images/Test/b.JPG", 40),
        (3, "pcb1/Data/Images/Test/a.JPG", 50),
        (3, "pcb1/Data/Images/Test/b.JPG", 60),
    )
    assert result.score_failure_timing_count == 0
    assert not result.score_failure_paths
    assert result.median_latency_ns == 35.0
    assert result.p95_rank == 6
    assert result.p95_latency_ns == 60
    assert result.median_latency_seconds == 35 / 1e9
    assert result.p95_latency_seconds == 60 / 1e9
    assert result.environment == fixed_environment
    assert result.failed_phase is None
    assert result.failed_path is None
    assert result.failed_pass_index is None
    assert calls == ["decoded-a", "decoded-b"] * 4
    assert tuple(decoded_images.items()) == original_items


def test_latency_supports_hog_score_records(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    fixed_environment: CPUEnvironmentRecord,
) -> None:
    monkeypatch.setattr(
        latency_module.time,
        "perf_counter_ns",
        _timer_for_durations((11, 12, 13)),
    )

    result = measure_cpu_latency(
        {"pcb1/Data/Images/Test/a.JPG": object()},
        lambda _: _hog_success(),
        method=CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM,
        config=project_config,
    )

    assert result.succeeded
    assert result.method is CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM
    assert result.sample_count == 3
    assert result.median_latency_ns == 12.0
    assert result.p95_rank == 3
    assert result.p95_latency_ns == 13
    assert result.environment == fixed_environment


@pytest.mark.parametrize(
    ("method", "failure_factory", "expected_code"),
    [
        (
            CalibrationMethod.ECC_RESIDUAL,
            _ecc_failure,
            PreprocessingFailureCode.INVALID_PREPROCESSED_IMAGE,
        ),
        (
            CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM,
            _hog_failure,
            HOGScoringFailureCode.HOG_SCORE_DECISION_FAILED,
        ),
    ],
)
def test_latency_retains_failed_score_timings(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    fixed_environment: CPUEnvironmentRecord,
    method: CalibrationMethod,
    failure_factory,
    expected_code,
) -> None:
    failed_input = object()
    decoded_images = {
        "pcb1/Data/Images/Test/a.JPG": failed_input,
        "pcb1/Data/Images/Test/b.JPG": object(),
    }

    def score_one(decoded):
        return (
            failure_factory()
            if decoded is failed_input
            else (_ecc_success() if method is CalibrationMethod.ECC_RESIDUAL else _hog_success())
        )

    monkeypatch.setattr(
        latency_module.time,
        "perf_counter_ns",
        _timer_for_durations((60, 10, 50, 20, 40, 30)),
    )

    result = measure_cpu_latency(
        decoded_images,
        score_one,
        method=method,
        config=project_config,
    )

    assert result.succeeded
    assert result.sample_count == 6
    assert result.score_failure_timing_count == 3
    assert result.score_failure_paths == ("pcb1/Data/Images/Test/a.JPG",)
    assert result.observations is not None
    failed_observations = tuple(
        item for item in result.observations if item.score_status == "failed"
    )
    assert len(failed_observations) == 3
    assert all(item.score_failure_code == expected_code for item in failed_observations)
    assert result.median_latency_ns == 35.0
    assert result.p95_latency_ns == 60


def test_latency_uses_unicode_code_point_order(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    fixed_environment: CPUEnvironmentRecord,
) -> None:
    paths = (
        "pcb1/Data/Images/Test/A.JPG",
        "pcb1/Data/Images/Test/a.JPG",
        "pcb1/Data/Images/Test/é.JPG",
        "pcb1/Data/Images/Test/あ.JPG",
    )
    decoded_images = {path: path for path in reversed(paths)}
    calls = []

    def score_one(decoded):
        calls.append(decoded)
        return _ecc_success()

    monkeypatch.setattr(
        latency_module.time,
        "perf_counter_ns",
        _timer_for_durations(tuple(range(1, 13))),
    )

    result = measure_cpu_latency(
        decoded_images,
        score_one,
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )

    assert result.succeeded
    assert result.ordered_paths == paths
    assert calls == list(paths) * 4


def test_latency_is_repeatable_with_same_timer_evidence(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    fixed_environment: CPUEnvironmentRecord,
) -> None:
    inputs = {"pcb1/Data/Images/Test/a.JPG": object()}
    durations = (10, 20, 30)
    monkeypatch.setattr(
        latency_module.time,
        "perf_counter_ns",
        _timer_for_durations(durations),
    )
    first = measure_cpu_latency(
        inputs,
        lambda _: _ecc_success(),
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )
    monkeypatch.setattr(
        latency_module.time,
        "perf_counter_ns",
        _timer_for_durations(durations),
    )
    second = measure_cpu_latency(
        inputs,
        lambda _: _ecc_success(),
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )

    assert first == second
    assert first.environment == fixed_environment


def test_latency_rejects_empty_input(
    project_config: ProjectConfig,
    fixed_environment: CPUEnvironmentRecord,
) -> None:
    result = measure_cpu_latency(
        {},
        lambda _: _ecc_success(),
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )

    assert result.failure_code is CPULatencyFailureCode.LATENCY_INPUT_EMPTY
    assert result.failed_phase == "input"
    assert result.item_count == 0
    assert result.observations is None


@pytest.mark.parametrize(
    "invalid_path",
    ["", "/absolute.JPG", "../escape.JPG", "windows\\path.JPG"],
)
def test_latency_rejects_invalid_path_before_environment_capture(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    invalid_path: str,
) -> None:
    capture_calls = 0

    def capture():
        nonlocal capture_calls
        capture_calls += 1
        raise AssertionError("environment capture must not run")

    monkeypatch.setattr(latency_module, "_capture_cpu_environment", capture)

    result = measure_cpu_latency(
        {invalid_path: object()},
        lambda _: _ecc_success(),
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )

    assert result.failure_code is CPULatencyFailureCode.LATENCY_PATH_INVALID
    assert result.failed_path == invalid_path
    assert capture_calls == 0


def test_latency_rejects_invalid_method_and_scorer(
    project_config: ProjectConfig,
    fixed_environment: CPUEnvironmentRecord,
) -> None:
    inputs = {"pcb1/Data/Images/Test/a.JPG": object()}

    invalid_method = measure_cpu_latency(
        inputs,
        lambda _: _ecc_success(),
        method="ecc_residual",
        config=project_config,
    )
    invalid_scorer = measure_cpu_latency(
        inputs,
        None,
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )

    assert invalid_method.failure_code is CPULatencyFailureCode.LATENCY_METHOD_INVALID
    assert invalid_scorer.failure_code is CPULatencyFailureCode.LATENCY_SCORER_INVALID
    assert invalid_method.observations is None
    assert invalid_scorer.observations is None


def test_latency_rejects_environment_capture_failure(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
) -> None:
    def fail_capture():
        raise OSError("synthetic environment failure")

    monkeypatch.setattr(latency_module, "_capture_cpu_environment", fail_capture)

    result = measure_cpu_latency(
        {"pcb1/Data/Images/Test/a.JPG": object()},
        lambda _: _ecc_success(),
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )

    assert result.failure_code is CPULatencyFailureCode.LATENCY_ENVIRONMENT_CAPTURE_FAILED
    assert result.failed_phase == "environment"
    assert result.observations is None


def test_latency_rejects_warmup_exception_without_partial_output(
    project_config: ProjectConfig,
    fixed_environment: CPUEnvironmentRecord,
) -> None:
    calls = 0

    def score_one(_):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic warmup failure")
        return _ecc_success()

    result = measure_cpu_latency(
        {
            "pcb1/Data/Images/Test/a.JPG": object(),
            "pcb1/Data/Images/Test/b.JPG": object(),
        },
        score_one,
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )

    assert result.failure_code is CPULatencyFailureCode.LATENCY_SCORE_CALL_FAILED
    assert result.failed_phase == "warmup"
    assert result.failed_path == "pcb1/Data/Images/Test/b.JPG"
    assert result.completed_warmup_invocations == 1
    assert result.completed_timed_invocations == 0
    assert result.observations is None


def test_latency_rejects_invalid_timed_score_without_partial_output(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    fixed_environment: CPUEnvironmentRecord,
) -> None:
    calls = 0

    def score_one(_):
        nonlocal calls
        calls += 1
        return _ecc_success() if calls == 1 else object()

    monkeypatch.setattr(
        latency_module.time,
        "perf_counter_ns",
        _timer_for_durations((10,)),
    )

    result = measure_cpu_latency(
        {"pcb1/Data/Images/Test/a.JPG": object()},
        score_one,
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )

    assert result.failure_code is CPULatencyFailureCode.LATENCY_SCORE_RECORD_INVALID
    assert result.failed_phase == "timed"
    assert result.failed_path == "pcb1/Data/Images/Test/a.JPG"
    assert result.failed_pass_index == 1
    assert result.completed_warmup_invocations == 1
    assert result.completed_timed_invocations == 0
    assert result.observations is None


def test_latency_rejects_nonmonotonic_timer_without_partial_output(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    fixed_environment: CPUEnvironmentRecord,
) -> None:
    timer_values = iter((20, 10))
    monkeypatch.setattr(
        latency_module.time,
        "perf_counter_ns",
        lambda: next(timer_values),
    )

    result = measure_cpu_latency(
        {"pcb1/Data/Images/Test/a.JPG": object()},
        lambda _: _ecc_success(),
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )

    assert result.failure_code is CPULatencyFailureCode.LATENCY_TIMER_INVALID
    assert result.failed_phase == "timed"
    assert result.failed_pass_index == 1
    assert result.observations is None


def test_latency_rejects_method_score_type_mismatch(
    project_config: ProjectConfig,
    fixed_environment: CPUEnvironmentRecord,
) -> None:
    result = measure_cpu_latency(
        {"pcb1/Data/Images/Test/a.JPG": object()},
        lambda _: _hog_success(),
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )

    assert result.failure_code is CPULatencyFailureCode.LATENCY_SCORE_RECORD_INVALID
    assert result.failed_phase == "warmup"
    assert result.observations is None


def test_latency_rejects_invalid_summary(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    fixed_environment: CPUEnvironmentRecord,
) -> None:
    monkeypatch.setattr(
        latency_module.time,
        "perf_counter_ns",
        _timer_for_durations((10, 20, 30)),
    )
    monkeypatch.setattr(latency_module.statistics, "median", lambda _: math.nan)

    result = measure_cpu_latency(
        {"pcb1/Data/Images/Test/a.JPG": object()},
        lambda _: _ecc_success(),
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )

    assert result.failure_code is CPULatencyFailureCode.LATENCY_RESULT_INVALID
    assert result.failed_phase == "summary"
    assert result.completed_timed_invocations == 3
    assert result.observations is None
