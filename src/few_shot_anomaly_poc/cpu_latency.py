"""Measure the preregistered CPU preprocessing-and-scoring boundary."""

from __future__ import annotations

import math
import os
import platform
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

import cv2

from few_shot_anomaly_poc.calibration import CalibrationMethod
from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.ecc_residual import ECCResidualScoreResult
from few_shot_anomaly_poc.errors import (
    CPULatencyFailureCode,
    ManifestIntegrityError,
)
from few_shot_anomaly_poc.hog_scoring import PatchHOGScoreResult
from few_shot_anomaly_poc.manifests import normalize_relative_path

DEPENDENCY_DISTRIBUTIONS = (
    "numpy",
    "opencv-python-headless",
    "scikit-image",
    "scikit-learn",
)
THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
)


@dataclass(frozen=True)
class CPUEnvironmentRecord:
    """Machine and runtime context required to interpret CPU latency."""

    cpu_model: str | None
    logical_core_count: int | None
    physical_core_count: int | None
    ram_bytes: int | None
    operating_system: str
    machine: str
    python_version: str
    opencv_thread_count: int | None
    dependency_versions: tuple[tuple[str, str | None], ...]
    thread_environment: tuple[tuple[str, str | None], ...]


@dataclass(frozen=True)
class CPULatencyObservation:
    """One measured image invocation from one timed pass."""

    pass_index: int
    relative_path: str
    duration_ns: int
    score_status: Literal["ok", "failed"]
    score_failure_code: str | None


@dataclass(frozen=True)
class CPULatencyResult:
    """Complete CPU latency evidence or one stable measurement failure."""

    status: Literal["ok", "LATENCY_FAILED"]
    failure_code: CPULatencyFailureCode | None
    method: CalibrationMethod | None
    measurement_boundary: str
    timer: str
    warmup_passes: int
    timed_passes: int
    item_count: int
    completed_warmup_invocations: int
    completed_timed_invocations: int
    sample_count: int | None
    ordered_paths: tuple[str, ...]
    observations: tuple[CPULatencyObservation, ...] | None
    score_failure_timing_count: int | None
    score_failure_paths: tuple[str, ...]
    median_latency_ns: float | None
    p95_latency_ns: int | None
    median_latency_seconds: float | None
    p95_latency_seconds: float | None
    p95_rank: int | None
    environment: CPUEnvironmentRecord | None
    failed_phase: Literal["input", "environment", "warmup", "timed", "summary"] | None
    failed_path: str | None
    failed_pass_index: int | None

    @property
    def succeeded(self) -> bool:
        """Return whether the complete preregistered timing run succeeded."""
        return self.status == "ok"


def _relative_path_is_valid(path: object) -> bool:
    if not isinstance(path, str):
        return False
    try:
        return normalize_relative_path(path) == path
    except ManifestIntegrityError:
        return False


def _read_proc_cpuinfo() -> str | None:
    try:
        return Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return None


def _cpu_model(cpuinfo: str | None) -> str | None:
    if cpuinfo is not None:
        for line in cpuinfo.splitlines():
            if line.lower().startswith("model name") and ":" in line:
                value = line.split(":", maxsplit=1)[1].strip()
                if value:
                    return value
    fallback = platform.processor().strip()
    return fallback or None


def _physical_core_count(cpuinfo: str | None) -> int | None:
    if cpuinfo is None:
        return None
    pairs: set[tuple[str, str]] = set()
    physical_id: str | None = None
    core_id: str | None = None
    for line in (*cpuinfo.splitlines(), ""):
        stripped = line.strip()
        if not stripped:
            if physical_id is not None and core_id is not None:
                pairs.add((physical_id, core_id))
            physical_id = None
            core_id = None
        elif ":" in line:
            key, value = (part.strip() for part in line.split(":", maxsplit=1))
            if key == "physical id":
                physical_id = value
            elif key == "core id":
                core_id = value
    return len(pairs) or None


def _ram_bytes() -> int | None:
    try:
        page_count = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if (
        not isinstance(page_count, int)
        or isinstance(page_count, bool)
        or page_count <= 0
        or not isinstance(page_size, int)
        or isinstance(page_size, bool)
        or page_size <= 0
    ):
        return None
    return page_count * page_size


def _distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _capture_cpu_environment() -> CPUEnvironmentRecord:
    cpuinfo = _read_proc_cpuinfo()
    logical_core_count = os.cpu_count()
    try:
        opencv_thread_count = int(cv2.getNumThreads())
    except (AttributeError, TypeError, ValueError):
        opencv_thread_count = None
    return CPUEnvironmentRecord(
        cpu_model=_cpu_model(cpuinfo),
        logical_core_count=logical_core_count,
        physical_core_count=_physical_core_count(cpuinfo),
        ram_bytes=_ram_bytes(),
        operating_system=platform.platform(),
        machine=platform.machine(),
        python_version=platform.python_version(),
        opencv_thread_count=opencv_thread_count,
        dependency_versions=tuple(
            (name, _distribution_version(name)) for name in DEPENDENCY_DISTRIBUTIONS
        ),
        thread_environment=tuple(
            (name, os.environ.get(name)) for name in THREAD_ENVIRONMENT_VARIABLES
        ),
    )


def _environment_is_valid(result: object) -> bool:
    if (
        not isinstance(result, CPUEnvironmentRecord)
        or not isinstance(result.operating_system, str)
        or not result.operating_system
        or not isinstance(result.machine, str)
        or not result.machine
        or not isinstance(result.python_version, str)
        or not result.python_version
        or not isinstance(result.dependency_versions, tuple)
        or any(not isinstance(item, tuple) or len(item) != 2 for item in result.dependency_versions)
        or tuple(name for name, _ in result.dependency_versions) != DEPENDENCY_DISTRIBUTIONS
        or any(
            value is not None and (not isinstance(value, str) or not value)
            for _, value in result.dependency_versions
        )
        or not isinstance(result.thread_environment, tuple)
        or any(not isinstance(item, tuple) or len(item) != 2 for item in result.thread_environment)
        or tuple(name for name, _ in result.thread_environment) != THREAD_ENVIRONMENT_VARIABLES
        or any(
            value is not None and not isinstance(value, str)
            for _, value in result.thread_environment
        )
    ):
        return False
    optional_positive_integers = (
        result.logical_core_count,
        result.physical_core_count,
        result.ram_bytes,
        result.opencv_thread_count,
    )
    return (
        result.cpu_model is None or (isinstance(result.cpu_model, str) and bool(result.cpu_model))
    ) and all(
        value is None or (isinstance(value, int) and not isinstance(value, bool) and value > 0)
        for value in optional_positive_integers
    )


def _score_status(
    result: object,
    *,
    method: CalibrationMethod,
    config: ProjectConfig,
) -> tuple[Literal["ok", "failed"], str | None] | None:
    if method is CalibrationMethod.ECC_RESIDUAL:
        if not isinstance(result, ECCResidualScoreResult):
            return None
    else:
        if not isinstance(result, PatchHOGScoreResult):
            return None

    if not isinstance(result.anomaly_score, float) or not math.isfinite(result.anomaly_score):
        return None
    if method is CalibrationMethod.ECC_RESIDUAL:
        successful_score_is_valid = 0.0 <= result.anomaly_score <= 1.0
        failure_score = config.ecc_residual_scoring.failure_score
    else:
        limit = config.patch_hog_scoring.maximum_absolute_patch_score_exclusive
        successful_score_is_valid = abs(result.anomaly_score) < limit
        failure_score = config.patch_hog_scoring.failure_score
    if result.score_status == "ok":
        if result.failure_code is None and successful_score_is_valid:
            return ("ok", None)
        return None
    if result.score_status == "failed":
        if result.failure_code is not None and result.anomaly_score == failure_score:
            return ("failed", str(result.failure_code))
        return None
    return None


def _failed(
    code: CPULatencyFailureCode,
    *,
    config: ProjectConfig,
    method: object,
    item_count: int,
    completed_warmup_invocations: int = 0,
    completed_timed_invocations: int = 0,
    failed_phase: Literal["input", "environment", "warmup", "timed", "summary"],
    failed_path: str | None = None,
    failed_pass_index: int | None = None,
) -> CPULatencyResult:
    return CPULatencyResult(
        status="LATENCY_FAILED",
        failure_code=code,
        method=method if isinstance(method, CalibrationMethod) else None,
        measurement_boundary=config.latency_measurement.boundary,
        timer=config.latency_measurement.timer,
        warmup_passes=config.latency_measurement.warmup_passes,
        timed_passes=config.latency_measurement.timed_passes,
        item_count=item_count,
        completed_warmup_invocations=completed_warmup_invocations,
        completed_timed_invocations=completed_timed_invocations,
        sample_count=None,
        ordered_paths=(),
        observations=None,
        score_failure_timing_count=None,
        score_failure_paths=(),
        median_latency_ns=None,
        p95_latency_ns=None,
        median_latency_seconds=None,
        p95_latency_seconds=None,
        p95_rank=None,
        environment=None,
        failed_phase=failed_phase,
        failed_path=failed_path,
        failed_pass_index=failed_pass_index,
    )


def _result_is_valid(
    result: CPULatencyResult,
    *,
    config: ProjectConfig,
) -> bool:
    if (
        not result.succeeded
        or result.failure_code is not None
        or not isinstance(result.method, CalibrationMethod)
        or result.measurement_boundary != config.latency_measurement.boundary
        or result.timer != config.latency_measurement.timer
        or result.warmup_passes != config.latency_measurement.warmup_passes
        or result.timed_passes != config.latency_measurement.timed_passes
        or not isinstance(result.item_count, int)
        or isinstance(result.item_count, bool)
        or result.item_count < 1
        or result.completed_warmup_invocations != result.item_count * result.warmup_passes
        or result.completed_timed_invocations != result.item_count * result.timed_passes
        or result.sample_count != result.completed_timed_invocations
        or not isinstance(result.ordered_paths, tuple)
        or len(result.ordered_paths) != result.item_count
        or any(not _relative_path_is_valid(path) for path in result.ordered_paths)
        or tuple(sorted(result.ordered_paths)) != result.ordered_paths
        or not isinstance(result.observations, tuple)
        or len(result.observations) != result.sample_count
        or any(not isinstance(item, CPULatencyObservation) for item in result.observations)
        or not isinstance(result.score_failure_timing_count, int)
        or isinstance(result.score_failure_timing_count, bool)
        or not isinstance(result.score_failure_paths, tuple)
        or not isinstance(result.median_latency_ns, float)
        or not math.isfinite(result.median_latency_ns)
        or result.median_latency_ns < 0.0
        or not isinstance(result.p95_latency_ns, int)
        or isinstance(result.p95_latency_ns, bool)
        or result.p95_latency_ns < 0
        or not isinstance(result.median_latency_seconds, float)
        or not math.isfinite(result.median_latency_seconds)
        or not isinstance(result.p95_latency_seconds, float)
        or not math.isfinite(result.p95_latency_seconds)
        or not isinstance(result.p95_rank, int)
        or isinstance(result.p95_rank, bool)
        or not _environment_is_valid(result.environment)
        or result.failed_phase is not None
        or result.failed_path is not None
        or result.failed_pass_index is not None
    ):
        return False
    assert result.observations is not None
    assert result.sample_count is not None
    assert result.score_failure_timing_count is not None
    assert result.median_latency_ns is not None
    assert result.p95_latency_ns is not None
    assert result.median_latency_seconds is not None
    assert result.p95_latency_seconds is not None
    assert result.p95_rank is not None

    expected_order = tuple(
        (pass_index, path)
        for pass_index in range(1, result.timed_passes + 1)
        for path in result.ordered_paths
    )
    if (
        tuple(
            (observation.pass_index, observation.relative_path)
            for observation in result.observations
        )
        != expected_order
    ):
        return False
    if any(
        not isinstance(observation.duration_ns, int)
        or isinstance(observation.duration_ns, bool)
        or observation.duration_ns < 0
        or observation.score_status not in ("ok", "failed")
        or (observation.score_status == "ok" and observation.score_failure_code is not None)
        or (
            observation.score_status == "failed"
            and (
                not isinstance(observation.score_failure_code, str)
                or not observation.score_failure_code
            )
        )
        for observation in result.observations
    ):
        return False

    durations = tuple(observation.duration_ns for observation in result.observations)
    expected_p95_rank = math.ceil(config.latency_measurement.p95_quantile * result.sample_count)
    failed_observations = tuple(
        observation for observation in result.observations if observation.score_status == "failed"
    )
    expected_failure_paths = tuple(
        path
        for path in result.ordered_paths
        if any(observation.relative_path == path for observation in failed_observations)
    )
    return (
        result.score_failure_timing_count == len(failed_observations)
        and result.score_failure_paths == expected_failure_paths
        and result.median_latency_ns == float(statistics.median(durations))
        and result.p95_rank == expected_p95_rank
        and result.p95_latency_ns == sorted(durations)[expected_p95_rank - 1]
        and result.median_latency_seconds == result.median_latency_ns / 1e9
        and result.p95_latency_seconds == result.p95_latency_ns / 1e9
    )


def measure_cpu_latency(
    decoded_images: Mapping[str, object],
    score_one: Callable[[object], object],
    *,
    method: CalibrationMethod,
    config: ProjectConfig,
) -> CPULatencyResult:
    """Run one warm-up and three timed passes in deterministic path order."""
    item_count = len(decoded_images)
    if not isinstance(method, CalibrationMethod):
        return _failed(
            CPULatencyFailureCode.LATENCY_METHOD_INVALID,
            config=config,
            method=method,
            item_count=item_count,
            failed_phase="input",
        )
    if item_count == 0:
        return _failed(
            CPULatencyFailureCode.LATENCY_INPUT_EMPTY,
            config=config,
            method=method,
            item_count=0,
            failed_phase="input",
        )
    invalid_paths = tuple(path for path in decoded_images if not _relative_path_is_valid(path))
    if invalid_paths:
        invalid_string_paths = tuple(path for path in invalid_paths if isinstance(path, str))
        failed_path = (
            min(invalid_string_paths) if len(invalid_string_paths) == len(invalid_paths) else None
        )
        return _failed(
            CPULatencyFailureCode.LATENCY_PATH_INVALID,
            config=config,
            method=method,
            item_count=item_count,
            failed_phase="input",
            failed_path=failed_path,
        )
    if not callable(score_one):
        return _failed(
            CPULatencyFailureCode.LATENCY_SCORER_INVALID,
            config=config,
            method=method,
            item_count=item_count,
            failed_phase="input",
        )
    ordered_paths = tuple(sorted(decoded_images))

    try:
        environment = _capture_cpu_environment()
    except Exception:
        return _failed(
            CPULatencyFailureCode.LATENCY_ENVIRONMENT_CAPTURE_FAILED,
            config=config,
            method=method,
            item_count=item_count,
            failed_phase="environment",
        )
    if not _environment_is_valid(environment):
        return _failed(
            CPULatencyFailureCode.LATENCY_ENVIRONMENT_CAPTURE_FAILED,
            config=config,
            method=method,
            item_count=item_count,
            failed_phase="environment",
        )

    completed_warmup_invocations = 0
    for _ in range(config.latency_measurement.warmup_passes):
        for path in ordered_paths:
            try:
                score_result = score_one(decoded_images[path])
            except Exception:
                return _failed(
                    CPULatencyFailureCode.LATENCY_SCORE_CALL_FAILED,
                    config=config,
                    method=method,
                    item_count=item_count,
                    completed_warmup_invocations=completed_warmup_invocations,
                    failed_phase="warmup",
                    failed_path=path,
                )
            if _score_status(score_result, method=method, config=config) is None:
                return _failed(
                    CPULatencyFailureCode.LATENCY_SCORE_RECORD_INVALID,
                    config=config,
                    method=method,
                    item_count=item_count,
                    completed_warmup_invocations=completed_warmup_invocations,
                    failed_phase="warmup",
                    failed_path=path,
                )
            completed_warmup_invocations += 1

    observations = []
    completed_timed_invocations = 0
    for pass_index in range(1, config.latency_measurement.timed_passes + 1):
        for path in ordered_paths:
            try:
                start_ns = time.perf_counter_ns()
            except Exception:
                return _failed(
                    CPULatencyFailureCode.LATENCY_TIMER_INVALID,
                    config=config,
                    method=method,
                    item_count=item_count,
                    completed_warmup_invocations=completed_warmup_invocations,
                    completed_timed_invocations=completed_timed_invocations,
                    failed_phase="timed",
                    failed_path=path,
                    failed_pass_index=pass_index,
                )
            try:
                score_result = score_one(decoded_images[path])
            except Exception:
                return _failed(
                    CPULatencyFailureCode.LATENCY_SCORE_CALL_FAILED,
                    config=config,
                    method=method,
                    item_count=item_count,
                    completed_warmup_invocations=completed_warmup_invocations,
                    completed_timed_invocations=completed_timed_invocations,
                    failed_phase="timed",
                    failed_path=path,
                    failed_pass_index=pass_index,
                )
            try:
                end_ns = time.perf_counter_ns()
            except Exception:
                return _failed(
                    CPULatencyFailureCode.LATENCY_TIMER_INVALID,
                    config=config,
                    method=method,
                    item_count=item_count,
                    completed_warmup_invocations=completed_warmup_invocations,
                    completed_timed_invocations=completed_timed_invocations,
                    failed_phase="timed",
                    failed_path=path,
                    failed_pass_index=pass_index,
                )
            if (
                not isinstance(start_ns, int)
                or isinstance(start_ns, bool)
                or not isinstance(end_ns, int)
                or isinstance(end_ns, bool)
                or end_ns < start_ns
            ):
                return _failed(
                    CPULatencyFailureCode.LATENCY_TIMER_INVALID,
                    config=config,
                    method=method,
                    item_count=item_count,
                    completed_warmup_invocations=completed_warmup_invocations,
                    completed_timed_invocations=completed_timed_invocations,
                    failed_phase="timed",
                    failed_path=path,
                    failed_pass_index=pass_index,
                )
            status = _score_status(score_result, method=method, config=config)
            if status is None:
                return _failed(
                    CPULatencyFailureCode.LATENCY_SCORE_RECORD_INVALID,
                    config=config,
                    method=method,
                    item_count=item_count,
                    completed_warmup_invocations=completed_warmup_invocations,
                    completed_timed_invocations=completed_timed_invocations,
                    failed_phase="timed",
                    failed_path=path,
                    failed_pass_index=pass_index,
                )
            score_status, score_failure_code = status
            observations.append(
                CPULatencyObservation(
                    pass_index=pass_index,
                    relative_path=path,
                    duration_ns=end_ns - start_ns,
                    score_status=score_status,
                    score_failure_code=score_failure_code,
                )
            )
            completed_timed_invocations += 1

    durations = tuple(observation.duration_ns for observation in observations)
    sample_count = len(durations)
    p95_rank = math.ceil(config.latency_measurement.p95_quantile * sample_count)
    median_latency_ns = float(statistics.median(durations))
    p95_latency_ns = sorted(durations)[p95_rank - 1]
    failed_observations = tuple(
        observation for observation in observations if observation.score_status == "failed"
    )
    score_failure_paths = tuple(
        path
        for path in ordered_paths
        if any(observation.relative_path == path for observation in failed_observations)
    )
    result = CPULatencyResult(
        status="ok",
        failure_code=None,
        method=method,
        measurement_boundary=config.latency_measurement.boundary,
        timer=config.latency_measurement.timer,
        warmup_passes=config.latency_measurement.warmup_passes,
        timed_passes=config.latency_measurement.timed_passes,
        item_count=item_count,
        completed_warmup_invocations=completed_warmup_invocations,
        completed_timed_invocations=completed_timed_invocations,
        sample_count=sample_count,
        ordered_paths=ordered_paths,
        observations=tuple(observations),
        score_failure_timing_count=len(failed_observations),
        score_failure_paths=score_failure_paths,
        median_latency_ns=median_latency_ns,
        p95_latency_ns=p95_latency_ns,
        median_latency_seconds=median_latency_ns / 1e9,
        p95_latency_seconds=p95_latency_ns / 1e9,
        p95_rank=p95_rank,
        environment=environment,
        failed_phase=None,
        failed_path=None,
        failed_pass_index=None,
    )
    if not _result_is_valid(result, config=config):
        return _failed(
            CPULatencyFailureCode.LATENCY_RESULT_INVALID,
            config=config,
            method=method,
            item_count=item_count,
            completed_warmup_invocations=completed_warmup_invocations,
            completed_timed_invocations=completed_timed_invocations,
            failed_phase="summary",
        )
    return result
