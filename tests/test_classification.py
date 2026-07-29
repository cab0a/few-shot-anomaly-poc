from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from few_shot_anomaly_poc.calibration import (
    CalibrationMethod,
    NormalThresholdCalibrationResult,
    calibrate_normal_threshold,
    classify_fixed_threshold,
    normal_threshold_calibration_result_is_valid,
)
from few_shot_anomaly_poc.config import ProjectConfig, load_config
from few_shot_anomaly_poc.ecc_residual import ECCResidualScoreResult
from few_shot_anomaly_poc.errors import (
    HOGScoringFailureCode,
    PreprocessingFailureCode,
    ThresholdCalibrationFailureCode,
    ThresholdClassificationFailureCode,
)
from few_shot_anomaly_poc.hog_scoring import PatchHOGScoreResult


@pytest.fixture(scope="module")
def project_config() -> ProjectConfig:
    return load_config(Path("configs/v0.1.yaml"))


def _path(index: int) -> str:
    return f"pcb1/Data/Images/Normal/{index:04d}.JPG"


def _ecc_success(score: float) -> ECCResidualScoreResult:
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


def _ecc_failure(score: float = 1.0) -> ECCResidualScoreResult:
    return ECCResidualScoreResult(
        score_status="failed",
        failure_code=PreprocessingFailureCode.INVALID_PREPROCESSED_IMAGE,
        anomaly_score=score,
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


def _hog_success(score: float) -> PatchHOGScoreResult:
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


def _hog_failure(score: float = 1e12) -> PatchHOGScoreResult:
    return PatchHOGScoreResult(
        score_status="failed",
        failure_code=HOGScoringFailureCode.HOG_SCORE_DECISION_FAILED,
        anomaly_score=score,
        patch_anomaly_scores=None,
        top_patch_count=None,
        top_patch_indices=(),
        successful_patch_count=3,
        failed_patch_index=3,
    )


def _calibrated_ecc(project_config: ProjectConfig) -> NormalThresholdCalibrationResult:
    result = calibrate_normal_threshold(
        {_path(index): _ecc_success(index / 100) for index in reversed(range(20))},
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )
    assert result.succeeded
    assert result.threshold == pytest.approx(0.18)
    return result


def _calibrated_hog(project_config: ProjectConfig) -> NormalThresholdCalibrationResult:
    result = calibrate_normal_threshold(
        {_path(index): _hog_success(float(index - 10)) for index in reversed(range(20))},
        method=CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM,
        config=project_config,
    )
    assert result.succeeded
    assert result.threshold == 8.0
    return result


@pytest.mark.parametrize(
    ("score", "expected_class", "expected_reason", "expected_margin"),
    [
        (0.17, "normal", "score_at_or_below_threshold", -0.01),
        (0.18, "normal", "score_at_or_below_threshold", 0.0),
        (0.19, "anomalous", "score_above_threshold", 0.01),
    ],
)
def test_classify_ecc_uses_strict_greater_boundary(
    project_config: ProjectConfig,
    score: float,
    expected_class: str,
    expected_reason: str,
    expected_margin: float,
) -> None:
    calibration = _calibrated_ecc(project_config)

    result = classify_fixed_threshold(
        "pcb1/Data/Images/Test/candidate.JPG",
        _ecc_success(score),
        calibration=calibration,
        config=project_config,
    )

    assert result.succeeded
    assert result.status == "ok"
    assert result.failure_code is None
    assert result.method is CalibrationMethod.ECC_RESIDUAL
    assert result.relative_path == "pcb1/Data/Images/Test/candidate.JPG"
    assert result.score_status == "ok"
    assert result.score_failure_code is None
    assert result.anomaly_score == score
    assert result.threshold == pytest.approx(0.18)
    assert result.threshold_source_path == _path(18)
    assert result.calibration_sample_count == 20
    assert result.calibration_rank == 19
    assert result.predicted_class == expected_class
    assert result.is_anomalous is (expected_class == "anomalous")
    assert result.decision_reason == expected_reason
    assert result.score_margin == pytest.approx(expected_margin)


@pytest.mark.parametrize(
    ("score", "expected_class"),
    [
        (7.0, "normal"),
        (8.0, "normal"),
        (9.0, "anomalous"),
    ],
)
def test_classify_hog_uses_its_own_calibrated_scale(
    project_config: ProjectConfig,
    score: float,
    expected_class: str,
) -> None:
    calibration = _calibrated_hog(project_config)

    result = classify_fixed_threshold(
        "pcb1/Data/Images/Test/candidate.JPG",
        _hog_success(score),
        calibration=calibration,
        config=project_config,
    )

    assert result.succeeded
    assert result.method is CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM
    assert result.threshold == 8.0
    assert result.predicted_class == expected_class
    assert result.is_anomalous is (expected_class == "anomalous")


def test_classify_failure_as_anomalous_even_when_score_equals_threshold(
    project_config: ProjectConfig,
) -> None:
    scores = {_path(index): _ecc_success(0.0) for index in range(18)}
    scores[_path(18)] = _ecc_success(1.0)
    scores[_path(19)] = _ecc_failure()
    calibration = calibrate_normal_threshold(
        scores,
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )
    assert calibration.succeeded
    assert calibration.threshold == 1.0

    result = classify_fixed_threshold(
        "pcb1/Data/Images/Test/failed.JPG",
        _ecc_failure(),
        calibration=calibration,
        config=project_config,
    )

    assert result.succeeded
    assert result.score_status == "failed"
    assert result.score_failure_code == PreprocessingFailureCode.INVALID_PREPROCESSED_IMAGE
    assert result.anomaly_score == 1.0
    assert result.threshold == 1.0
    assert result.score_margin == 0.0
    assert result.predicted_class == "anomalous"
    assert result.is_anomalous is True
    assert result.decision_reason == "score_failure"


def test_classify_hog_failure_preserves_source_code(
    project_config: ProjectConfig,
) -> None:
    calibration = _calibrated_hog(project_config)

    result = classify_fixed_threshold(
        "pcb1/Data/Images/Test/failed.JPG",
        _hog_failure(),
        calibration=calibration,
        config=project_config,
    )

    assert result.succeeded
    assert result.predicted_class == "anomalous"
    assert result.decision_reason == "score_failure"
    assert result.score_failure_code == HOGScoringFailureCode.HOG_SCORE_DECISION_FAILED


def test_classify_rejects_score_from_other_method(
    project_config: ProjectConfig,
) -> None:
    calibration = _calibrated_ecc(project_config)

    result = classify_fixed_threshold(
        "pcb1/Data/Images/Test/candidate.JPG",
        _hog_success(0.1),
        calibration=calibration,
        config=project_config,
    )

    assert (
        result.failure_code is ThresholdClassificationFailureCode.CLASSIFICATION_SCORE_TYPE_MISMATCH
    )
    assert not result.succeeded
    assert result.method is CalibrationMethod.ECC_RESIDUAL
    assert result.predicted_class is None
    assert result.is_anomalous is None
    assert result.threshold is None


@pytest.mark.parametrize(
    "invalid_score",
    [
        _ecc_success(float("nan")),
        replace(
            _ecc_success(0.1),
            failure_code=PreprocessingFailureCode.INVALID_PREPROCESSED_IMAGE,
        ),
        _ecc_failure(0.9),
    ],
)
def test_classify_rejects_invalid_score_record(
    project_config: ProjectConfig,
    invalid_score: ECCResidualScoreResult,
) -> None:
    calibration = _calibrated_ecc(project_config)

    result = classify_fixed_threshold(
        "pcb1/Data/Images/Test/candidate.JPG",
        invalid_score,
        calibration=calibration,
        config=project_config,
    )

    assert (
        result.failure_code
        is ThresholdClassificationFailureCode.CLASSIFICATION_SCORE_RECORD_INVALID
    )
    assert result.predicted_class is None
    assert result.score_margin is None


@pytest.mark.parametrize(
    "invalid_path",
    ["", "/absolute.JPG", "../parent.JPG", "pcb1/../escape.JPG", "windows\\path.JPG"],
)
def test_classify_rejects_invalid_relative_path(
    project_config: ProjectConfig,
    invalid_path: str,
) -> None:
    calibration = _calibrated_ecc(project_config)

    result = classify_fixed_threshold(
        invalid_path,
        _ecc_success(0.1),
        calibration=calibration,
        config=project_config,
    )

    assert result.failure_code is ThresholdClassificationFailureCode.CLASSIFICATION_PATH_INVALID
    assert result.relative_path == invalid_path
    assert result.method is CalibrationMethod.ECC_RESIDUAL
    assert result.predicted_class is None


@pytest.mark.parametrize(
    "changed_fields",
    [
        {"rank": 18},
        {"threshold": float("nan")},
        {"threshold_source_path": "wrong/source.JPG"},
        {"score_order_paths": (_path(0),) * 20},
        {"realized_false_positive_rate": 0.5},
        {
            "failed_score_count": 1,
            "failed_score_paths": (_path(0),),
        },
    ],
)
def test_classify_rejects_corrupted_calibration_result(
    project_config: ProjectConfig,
    changed_fields: dict[str, object],
) -> None:
    calibration = _calibrated_ecc(project_config)
    corrupted = replace(calibration, **changed_fields)
    assert not normal_threshold_calibration_result_is_valid(
        corrupted,
        config=project_config,
    )

    result = classify_fixed_threshold(
        "pcb1/Data/Images/Test/candidate.JPG",
        _ecc_success(0.1),
        calibration=corrupted,
        config=project_config,
    )

    assert (
        result.failure_code is ThresholdClassificationFailureCode.CLASSIFICATION_CALIBRATION_INVALID
    )
    assert result.predicted_class is None
    assert result.threshold is None


def test_classify_rejects_failed_calibration_result(
    project_config: ProjectConfig,
) -> None:
    calibration = _calibrated_ecc(project_config)
    failed_calibration = replace(
        calibration,
        status="CALIBRATION_FAILED",
        failure_code=ThresholdCalibrationFailureCode.CALIBRATION_RESULT_INVALID,
        rank=None,
        threshold=None,
        threshold_source_path=None,
    )

    result = classify_fixed_threshold(
        "pcb1/Data/Images/Test/candidate.JPG",
        _ecc_success(0.1),
        calibration=failed_calibration,
        config=project_config,
    )

    assert (
        result.failure_code is ThresholdClassificationFailureCode.CLASSIFICATION_CALIBRATION_INVALID
    )
    assert result.predicted_class is None


def test_classify_is_repeatable_and_does_not_mutate_inputs(
    project_config: ProjectConfig,
) -> None:
    calibration = _calibrated_hog(project_config)
    score = _hog_success(9.0)

    first = classify_fixed_threshold(
        "pcb1/Data/Images/Test/candidate.JPG",
        score,
        calibration=calibration,
        config=project_config,
    )
    second = classify_fixed_threshold(
        "pcb1/Data/Images/Test/candidate.JPG",
        score,
        calibration=calibration,
        config=project_config,
    )

    assert first == second
    assert score.anomaly_score == 9.0
    assert calibration.threshold == 8.0
