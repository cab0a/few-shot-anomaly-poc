"""Select one fixed normal-only threshold per v0.1 method."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.ecc_residual import ECCResidualScoreResult
from few_shot_anomaly_poc.errors import ThresholdCalibrationFailureCode
from few_shot_anomaly_poc.hog_scoring import PatchHOGScoreResult


class CalibrationMethod(StrEnum):
    """Method identifiers whose score scales must remain separate."""

    ECC_RESIDUAL = "ecc_residual"
    PATCH_HOG_ONE_CLASS_SVM = "patch_hog_one_class_svm"


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
        if (
            score_records[path].score_status == "failed"
            or score_records[path].anomaly_score > threshold
        )
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

    return NormalThresholdCalibrationResult(
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
