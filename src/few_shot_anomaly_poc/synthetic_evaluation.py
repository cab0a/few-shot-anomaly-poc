"""Connect the v0.1 evaluation primitives using synthetic records only."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from few_shot_anomaly_poc.calibration import (
    CalibrationMethod,
    calibrate_normal_threshold,
    classify_fixed_threshold_batch,
)
from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.cpu_latency import (
    DEPENDENCY_DISTRIBUTIONS,
    THREAD_ENVIRONMENT_VARIABLES,
    CPUEnvironmentRecord,
    CPULatencyObservation,
    CPULatencyResult,
)
from few_shot_anomaly_poc.ecc_residual import ECCResidualScoreResult
from few_shot_anomaly_poc.evaluation_artifacts import (
    EvaluationArtifactBundle,
    MethodEvaluationArtifacts,
    write_evaluation_artifact_bundle,
)
from few_shot_anomaly_poc.failure_cases import select_failure_cases
from few_shot_anomaly_poc.hard_gate_decision import (
    DecisionProcessEvidence,
    FailureReviewDisposition,
    apply_hard_gate_decision,
)
from few_shot_anomaly_poc.hog_scoring import PatchHOGScoreResult
from few_shot_anomaly_poc.image_metrics import calculate_image_level_metrics
from few_shot_anomaly_poc.label_reveal import (
    FinalTestLabelRecord,
    reveal_final_test_labels,
)

SYNTHETIC_RUN_ID = "synthetic-e2e"
SYNTHETIC_REFERENCE_PATHS = tuple(f"synthetic/reference/{index:04d}.png" for index in range(20))
SYNTHETIC_CALIBRATION_PATHS = tuple(f"synthetic/calibration/{index:04d}.png" for index in range(20))
SYNTHETIC_FINAL_PATHS = tuple(f"synthetic/final/{index:04d}.png" for index in range(40))


def _ecc_score(score: float) -> ECCResidualScoreResult:
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
        effective_pixel_count=262_144,
        top_pixel_count=2_622,
    )


def _hog_score(score: float) -> PatchHOGScoreResult:
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


def _scores(
    method: CalibrationMethod,
) -> tuple[
    dict[str, ECCResidualScoreResult | PatchHOGScoreResult],
    dict[str, ECCResidualScoreResult | PatchHOGScoreResult],
]:
    if method is CalibrationMethod.ECC_RESIDUAL:
        calibration_values = tuple((index + 1) / 100 for index in range(20))
        final_values = (
            *(0.10 for _ in range(19)),
            0.20,
            0.10,
            0.15,
            *(0.30 + index / 100 for index in range(18)),
        )
        builder = _ecc_score
    else:
        calibration_values = tuple(float(index - 10) for index in range(20))
        final_values = (
            *(0.0 for _ in range(19)),
            9.0,
            0.0,
            1.0,
            *(float(10 + index) for index in range(18)),
        )
        builder = _hog_score
    return (
        {
            path: builder(value)
            for path, value in zip(
                SYNTHETIC_CALIBRATION_PATHS,
                calibration_values,
                strict=True,
            )
        },
        {
            path: builder(value)
            for path, value in zip(
                SYNTHETIC_FINAL_PATHS,
                final_values,
                strict=True,
            )
        },
    )


def _synthetic_latency(
    method: CalibrationMethod,
    *,
    duration_ns: int,
    config: ProjectConfig,
) -> CPULatencyResult:
    observations = tuple(
        CPULatencyObservation(
            pass_index=pass_index,
            relative_path=path,
            duration_ns=duration_ns,
            score_status="ok",
            score_failure_code=None,
        )
        for pass_index in range(1, config.latency_measurement.timed_passes + 1)
        for path in SYNTHETIC_FINAL_PATHS
    )
    sample_count = len(observations)
    return CPULatencyResult(
        status="ok",
        failure_code=None,
        method=method,
        measurement_boundary=config.latency_measurement.boundary,
        timer=config.latency_measurement.timer,
        warmup_passes=config.latency_measurement.warmup_passes,
        timed_passes=config.latency_measurement.timed_passes,
        item_count=len(SYNTHETIC_FINAL_PATHS),
        completed_warmup_invocations=(
            len(SYNTHETIC_FINAL_PATHS) * config.latency_measurement.warmup_passes
        ),
        completed_timed_invocations=sample_count,
        sample_count=sample_count,
        ordered_paths=SYNTHETIC_FINAL_PATHS,
        observations=observations,
        score_failure_timing_count=0,
        score_failure_paths=(),
        median_latency_ns=float(duration_ns),
        p95_latency_ns=duration_ns,
        median_latency_seconds=duration_ns / 1e9,
        p95_latency_seconds=duration_ns / 1e9,
        p95_rank=math.ceil(config.latency_measurement.p95_quantile * sample_count),
        environment=CPUEnvironmentRecord(
            cpu_model="Synthetic CPU record; not measured hardware",
            logical_core_count=8,
            physical_core_count=4,
            ram_bytes=16 * 1024**3,
            operating_system="Synthetic OS record",
            machine="synthetic-x86_64",
            python_version="3.13.14",
            opencv_thread_count=1,
            dependency_versions=tuple(
                (name, "synthetic-record") for name in DEPENDENCY_DISTRIBUTIONS
            ),
            thread_environment=tuple((name, None) for name in THREAD_ENVIRONMENT_VARIABLES),
        ),
        failed_phase=None,
        failed_path=None,
        failed_pass_index=None,
    )


def _partition_manifest_sha256() -> str:
    records = (
        *((path, "reference") for path in SYNTHETIC_REFERENCE_PATHS),
        *((path, "calibration") for path in SYNTHETIC_CALIBRATION_PATHS),
        *((path, "final_test") for path in SYNTHETIC_FINAL_PATHS),
    )
    serialized = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _method_artifacts(
    method: CalibrationMethod,
    *,
    config: ProjectConfig,
) -> MethodEvaluationArtifacts:
    calibration_scores, final_scores = _scores(method)
    calibration = calibrate_normal_threshold(
        calibration_scores,
        method=method,
        config=config,
    )
    if not calibration.succeeded:
        raise RuntimeError("synthetic calibration fixture is invalid")
    classifications = classify_fixed_threshold_batch(
        final_scores,
        calibration=calibration,
        config=config,
    )
    if not classifications.succeeded:
        raise RuntimeError("synthetic classification fixture is invalid")

    labels = tuple(
        FinalTestLabelRecord(
            relative_path=path,
            label="normal" if index < 20 else "anomaly",
        )
        for index, path in enumerate(SYNTHETIC_FINAL_PATHS)
    )
    revealed = reveal_final_test_labels(
        classifications,
        labels,
        config=config,
    )
    metrics = calculate_image_level_metrics(revealed, config=config)
    failures = select_failure_cases(revealed, config=config)
    latency = _synthetic_latency(
        method,
        duration_ns=50_000_000 if method is CalibrationMethod.ECC_RESIDUAL else 70_000_000,
        config=config,
    )
    process_evidence = DecisionProcessEvidence(
        normal_reference_count=len(SYNTHETIC_REFERENCE_PATHS),
        anomaly_training_labels_used=False,
        reproducibility_verified=True,
        test_leakage_detected=False,
        failure_review_disposition=FailureReviewDisposition.NO_MATERIAL_BOUNDARY,
        failure_review_rationale=(
            "Synthetic records exercise fixed false-positive and false-negative "
            "selection without an image-content claim."
        ),
        condition=None,
    )
    decision = apply_hard_gate_decision(
        metrics,
        latency,
        process_evidence,
        config=config,
    )
    if not all(
        result.succeeded
        for result in (
            revealed,
            metrics,
            failures,
            latency,
            decision,
        )
    ):
        raise RuntimeError("synthetic evaluation fixture is invalid")
    return MethodEvaluationArtifacts(
        method=method,
        calibration_scores=calibration_scores,
        final_test_scores=final_scores,
        classifications=classifications,
        revealed=revealed,
        metrics=metrics,
        latency=latency,
        failures=failures,
        process_evidence=process_evidence,
        decision=decision,
    )


def build_synthetic_evaluation_bundle(
    *,
    source_commit: str,
    config: ProjectConfig,
) -> EvaluationArtifactBundle:
    """Build deterministic primitive outputs without reading an image or VisA."""
    return EvaluationArtifactBundle(
        run_id=SYNTHETIC_RUN_ID,
        run_kind="synthetic",
        dataset="synthetic-records",
        category="not-applicable",
        source_commit=source_commit,
        partition_manifest_sha256=_partition_manifest_sha256(),
        methods=tuple(
            _method_artifacts(method, config=config)
            for method in (
                CalibrationMethod.ECC_RESIDUAL,
                CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM,
            )
        ),
    )


def run_synthetic_evaluation(
    *,
    output_root: Path,
    source_commit: str,
    config_path: Path,
    config: ProjectConfig,
) -> Path:
    """Create the deterministic synthetic artifact bundle once."""
    return write_evaluation_artifact_bundle(
        build_synthetic_evaluation_bundle(
            source_commit=source_commit,
            config=config,
        ),
        output_root=output_root,
        config_path=config_path,
        config=config,
    )
