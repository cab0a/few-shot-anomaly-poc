"""Calibrate and apply one fixed normal-only threshold per v0.1 method."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.ecc_residual import ECCResidualScoreResult
from few_shot_anomaly_poc.errors import (
    ThresholdCalibrationFailureCode,
    ThresholdClassificationFailureCode,
)
from few_shot_anomaly_poc.hog_scoring import PatchHOGScoreResult


class CalibrationMethod(StrEnum):
    """Method identifiers whose score scales must remain separate."""

    ECC_RESIDUAL = "ecc_residual"
    PATCH_HOG_ONE_CLASS_SVM = "patch_hog_one_class_svm"


type ClassificationDecisionReason = Literal[
    "score_at_or_below_threshold",
    "score_above_threshold",
    "score_failure",
]


@dataclass(frozen=True)
class NormalThresholdCalibrationResult:
    """Threshold and normal-only calibration diagnostics."""

    status: Literal["ok", "CALIBRATION_FAILED"]
    failure_code: ThresholdCalibrationFailureCode | None
    method: CalibrationMethod | None
    quantile: float
    sample_count: int
    rank: int | None
    threshold: float | None
    threshold_source_path: str | None
    score_order_paths: tuple[str, ...]
    failed_score_count: int | None
    failed_score_paths: tuple[str, ...]
    predicted_anomalous_count: int | None
    predicted_anomalous_paths: tuple[str, ...]
    realized_false_positive_rate: float | None
    invalid_score_path: str | None

    @property
    def succeeded(self) -> bool:
        """Return whether a fixed threshold was selected."""
        return self.status == "ok"


@dataclass(frozen=True)
class FixedThresholdClassificationResult:
    """One label-free image decision tied to a validated calibration result."""

    status: Literal["ok", "CLASSIFICATION_FAILED"]
    failure_code: ThresholdClassificationFailureCode | None
    method: CalibrationMethod | None
    relative_path: str | None
    score_status: Literal["ok", "failed"] | None
    score_failure_code: str | None
    anomaly_score: float | None
    threshold: float | None
    threshold_source_path: str | None
    calibration_sample_count: int | None
    calibration_rank: int | None
    predicted_class: Literal["normal", "anomalous"] | None
    is_anomalous: bool | None
    decision_reason: ClassificationDecisionReason | None
    score_margin: float | None

    @property
    def succeeded(self) -> bool:
        """Return whether one fixed-threshold decision was produced."""
        return self.status == "ok"


def _failed(
    code: ThresholdCalibrationFailureCode,
    *,
    method: CalibrationMethod | None,
    config: ProjectConfig,
    sample_count: int,
    invalid_score_path: str | None = None,
) -> NormalThresholdCalibrationResult:
    return NormalThresholdCalibrationResult(
        status="CALIBRATION_FAILED",
        failure_code=code,
        method=method,
        quantile=config.threshold_calibration.quantile,
        sample_count=sample_count,
        rank=None,
        threshold=None,
        threshold_source_path=None,
        score_order_paths=(),
        failed_score_count=None,
        failed_score_paths=(),
        predicted_anomalous_count=None,
        predicted_anomalous_paths=(),
        realized_false_positive_rate=None,
        invalid_score_path=invalid_score_path,
    )


def _relative_path_is_valid(path: object) -> bool:
    if not isinstance(path, str) or not path or "\\" in path or path.startswith("/"):
        return False
    return all(part not in {"", ".", ".."} for part in path.split("/"))


def _score_type_matches(result: object, method: CalibrationMethod) -> bool:
    if method is CalibrationMethod.ECC_RESIDUAL:
        return isinstance(result, ECCResidualScoreResult)
    return isinstance(result, PatchHOGScoreResult)


def _score_record_is_valid(
    result: ECCResidualScoreResult | PatchHOGScoreResult,
    *,
    method: CalibrationMethod,
    config: ProjectConfig,
) -> bool:
    score = result.anomaly_score
    if not isinstance(score, float) or not math.isfinite(score):
        return False

    if method is CalibrationMethod.ECC_RESIDUAL:
        valid_success_score = 0.0 <= score <= 1.0
        expected_failure_score = config.ecc_residual_scoring.failure_score
    else:
        valid_success_score = (
            abs(score) < config.patch_hog_scoring.maximum_absolute_patch_score_exclusive
        )
        expected_failure_score = config.patch_hog_scoring.failure_score

    if result.score_status == "ok":
        return result.failure_code is None and valid_success_score
    if result.score_status == "failed":
        return result.failure_code is not None and score == expected_failure_score
    return False


def normal_threshold_calibration_result_is_valid(
    result: object,
    *,
    config: ProjectConfig,
) -> bool:
    """Return whether a calibration result can control image classification."""
    if (
        not isinstance(result, NormalThresholdCalibrationResult)
        or not result.succeeded
        or result.failure_code is not None
        or not isinstance(result.method, CalibrationMethod)
        or result.quantile != config.threshold_calibration.quantile
        or not isinstance(result.sample_count, int)
        or isinstance(result.sample_count, bool)
        or result.sample_count < 1
        or not isinstance(result.rank, int)
        or isinstance(result.rank, bool)
        or result.rank != math.ceil(result.quantile * result.sample_count)
        or result.rank < 1
        or result.rank > result.sample_count
        or not isinstance(result.threshold, float)
        or not math.isfinite(result.threshold)
        or not _relative_path_is_valid(result.threshold_source_path)
        or not isinstance(result.score_order_paths, tuple)
        or len(result.score_order_paths) != result.sample_count
        or any(not _relative_path_is_valid(path) for path in result.score_order_paths)
        or len(set(result.score_order_paths)) != result.sample_count
        or result.score_order_paths[result.rank - 1] != result.threshold_source_path
        or not isinstance(result.failed_score_count, int)
        or isinstance(result.failed_score_count, bool)
        or not isinstance(result.failed_score_paths, tuple)
        or result.failed_score_count != len(result.failed_score_paths)
        or tuple(sorted(result.failed_score_paths)) != result.failed_score_paths
        or len(set(result.failed_score_paths)) != len(result.failed_score_paths)
        or any(not _relative_path_is_valid(path) for path in result.failed_score_paths)
        or not isinstance(result.predicted_anomalous_count, int)
        or isinstance(result.predicted_anomalous_count, bool)
        or not isinstance(result.predicted_anomalous_paths, tuple)
        or result.predicted_anomalous_count != len(result.predicted_anomalous_paths)
        or tuple(sorted(result.predicted_anomalous_paths)) != result.predicted_anomalous_paths
        or len(set(result.predicted_anomalous_paths)) != len(result.predicted_anomalous_paths)
        or any(not _relative_path_is_valid(path) for path in result.predicted_anomalous_paths)
        or result.failed_score_count > result.predicted_anomalous_count
        or not set(result.failed_score_paths).issubset(result.predicted_anomalous_paths)
        or not set(result.predicted_anomalous_paths).issubset(result.score_order_paths)
        or not isinstance(result.realized_false_positive_rate, float)
        or not math.isfinite(result.realized_false_positive_rate)
        or not math.isclose(
            result.realized_false_positive_rate,
            result.predicted_anomalous_count / result.sample_count,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or result.invalid_score_path is not None
    ):
        return False

    if result.method is CalibrationMethod.ECC_RESIDUAL:
        return 0.0 <= result.threshold <= config.ecc_residual_scoring.failure_score
    limit = config.patch_hog_scoring.maximum_absolute_patch_score_exclusive
    return -limit < result.threshold <= config.patch_hog_scoring.failure_score


def _classification_failed(
    code: ThresholdClassificationFailureCode,
    *,
    calibration: object,
    relative_path: object,
) -> FixedThresholdClassificationResult:
    method = (
        calibration.method
        if (
            isinstance(calibration, NormalThresholdCalibrationResult)
            and isinstance(calibration.method, CalibrationMethod)
        )
        else None
    )
    return FixedThresholdClassificationResult(
        status="CLASSIFICATION_FAILED",
        failure_code=code,
        method=method,
        relative_path=relative_path if isinstance(relative_path, str) else None,
        score_status=None,
        score_failure_code=None,
        anomaly_score=None,
        threshold=None,
        threshold_source_path=None,
        calibration_sample_count=None,
        calibration_rank=None,
        predicted_class=None,
        is_anomalous=None,
        decision_reason=None,
        score_margin=None,
    )


def _record_is_anomalous(
    result: ECCResidualScoreResult | PatchHOGScoreResult,
    *,
    threshold: float,
) -> bool:
    return result.score_status == "failed" or result.anomaly_score > threshold


def classify_fixed_threshold(
    relative_path: str,
    score_result: object,
    *,
    calibration: NormalThresholdCalibrationResult,
    config: ProjectConfig,
) -> FixedThresholdClassificationResult:
    """Classify one score without accepting an observed class label."""
    if not _relative_path_is_valid(relative_path):
        return _classification_failed(
            ThresholdClassificationFailureCode.CLASSIFICATION_PATH_INVALID,
            calibration=calibration,
            relative_path=relative_path,
        )
    if not normal_threshold_calibration_result_is_valid(
        calibration,
        config=config,
    ):
        return _classification_failed(
            ThresholdClassificationFailureCode.CLASSIFICATION_CALIBRATION_INVALID,
            calibration=calibration,
            relative_path=relative_path,
        )
    assert calibration.method is not None
    assert calibration.threshold is not None
    assert calibration.threshold_source_path is not None
    assert calibration.rank is not None

    if not _score_type_matches(score_result, calibration.method):
        return _classification_failed(
            ThresholdClassificationFailureCode.CLASSIFICATION_SCORE_TYPE_MISMATCH,
            calibration=calibration,
            relative_path=relative_path,
        )
    assert isinstance(score_result, (ECCResidualScoreResult, PatchHOGScoreResult))
    if not _score_record_is_valid(
        score_result,
        method=calibration.method,
        config=config,
    ):
        return _classification_failed(
            ThresholdClassificationFailureCode.CLASSIFICATION_SCORE_RECORD_INVALID,
            calibration=calibration,
            relative_path=relative_path,
        )

    score_margin = score_result.anomaly_score - calibration.threshold
    if not math.isfinite(score_margin):
        return _classification_failed(
            ThresholdClassificationFailureCode.CLASSIFICATION_RESULT_INVALID,
            calibration=calibration,
            relative_path=relative_path,
        )

    is_anomalous = _record_is_anomalous(
        score_result,
        threshold=calibration.threshold,
    )
    if score_result.score_status == "failed":
        decision_reason: ClassificationDecisionReason = "score_failure"
    elif is_anomalous:
        decision_reason = "score_above_threshold"
    else:
        decision_reason = "score_at_or_below_threshold"

    return FixedThresholdClassificationResult(
        status="ok",
        failure_code=None,
        method=calibration.method,
        relative_path=relative_path,
        score_status=score_result.score_status,
        score_failure_code=(
            str(score_result.failure_code) if score_result.failure_code is not None else None
        ),
        anomaly_score=score_result.anomaly_score,
        threshold=calibration.threshold,
        threshold_source_path=calibration.threshold_source_path,
        calibration_sample_count=calibration.sample_count,
        calibration_rank=calibration.rank,
        predicted_class="anomalous" if is_anomalous else "normal",
        is_anomalous=is_anomalous,
        decision_reason=decision_reason,
        score_margin=score_margin,
    )


def calibrate_normal_threshold(
    normal_calibration_scores: Mapping[str, object],
    *,
    method: CalibrationMethod,
    config: ProjectConfig,
) -> NormalThresholdCalibrationResult:
    """Select the preregistered threshold without accepting any label input."""
    sample_count = len(normal_calibration_scores)
    if not isinstance(method, CalibrationMethod):
        return _failed(
            ThresholdCalibrationFailureCode.CALIBRATION_METHOD_INVALID,
            method=None,
            config=config,
            sample_count=sample_count,
        )
    if sample_count == 0:
        return _failed(
            ThresholdCalibrationFailureCode.CALIBRATION_EMPTY,
            method=method,
            config=config,
            sample_count=0,
        )

    for path in normal_calibration_scores:
        if not _relative_path_is_valid(path):
            return _failed(
                ThresholdCalibrationFailureCode.CALIBRATION_PATH_INVALID,
                method=method,
                config=config,
                sample_count=sample_count,
                invalid_score_path=path if isinstance(path, str) else None,
            )

    relative_paths = tuple(sorted(normal_calibration_scores))
    score_records: dict[str, ECCResidualScoreResult | PatchHOGScoreResult] = {}
    for path in relative_paths:
        result = normal_calibration_scores[path]
        if not _score_type_matches(result, method):
            return _failed(
                ThresholdCalibrationFailureCode.CALIBRATION_SCORE_TYPE_MISMATCH,
                method=method,
                config=config,
                sample_count=sample_count,
                invalid_score_path=path,
            )
        assert isinstance(result, (ECCResidualScoreResult, PatchHOGScoreResult))
        if not _score_record_is_valid(result, method=method, config=config):
            return _failed(
                ThresholdCalibrationFailureCode.CALIBRATION_SCORE_RECORD_INVALID,
                method=method,
                config=config,
                sample_count=sample_count,
                invalid_score_path=path,
            )
        score_records[path] = result

    ordered = tuple(
        sorted(
            relative_paths,
            key=lambda path: (score_records[path].anomaly_score, path),
        )
    )
    rank = math.ceil(config.threshold_calibration.quantile * sample_count)
    if rank < 1 or rank > sample_count:
        return _failed(
            ThresholdCalibrationFailureCode.CALIBRATION_RESULT_INVALID,
            method=method,
            config=config,
            sample_count=sample_count,
        )
    threshold_source_path = ordered[rank - 1]
    threshold = score_records[threshold_source_path].anomaly_score
    if not math.isfinite(threshold):
        return _failed(
            ThresholdCalibrationFailureCode.CALIBRATION_RESULT_INVALID,
            method=method,
            config=config,
            sample_count=sample_count,
        )

    failed_score_paths = tuple(
        path for path in relative_paths if score_records[path].score_status == "failed"
    )
    predicted_anomalous_paths = tuple(
        path
        for path in relative_paths
        if _record_is_anomalous(score_records[path], threshold=threshold)
    )
    predicted_anomalous_count = len(predicted_anomalous_paths)
    realized_false_positive_rate = predicted_anomalous_count / sample_count
    if (
        predicted_anomalous_count > sample_count
        or not math.isfinite(realized_false_positive_rate)
        or realized_false_positive_rate < 0.0
        or realized_false_positive_rate > 1.0
    ):
        return _failed(
            ThresholdCalibrationFailureCode.CALIBRATION_RESULT_INVALID,
            method=method,
            config=config,
            sample_count=sample_count,
        )

    result = NormalThresholdCalibrationResult(
        status="ok",
        failure_code=None,
        method=method,
        quantile=config.threshold_calibration.quantile,
        sample_count=sample_count,
        rank=rank,
        threshold=threshold,
        threshold_source_path=threshold_source_path,
        score_order_paths=ordered,
        failed_score_count=len(failed_score_paths),
        failed_score_paths=failed_score_paths,
        predicted_anomalous_count=predicted_anomalous_count,
        predicted_anomalous_paths=predicted_anomalous_paths,
        realized_false_positive_rate=realized_false_positive_rate,
        invalid_score_path=None,
    )
    if not normal_threshold_calibration_result_is_valid(result, config=config):
        return _failed(
            ThresholdCalibrationFailureCode.CALIBRATION_RESULT_INVALID,
            method=method,
            config=config,
            sample_count=sample_count,
        )
    return result
