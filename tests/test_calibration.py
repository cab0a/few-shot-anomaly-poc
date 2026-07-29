from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from few_shot_anomaly_poc.calibration import (
    CalibrationMethod,
    calibrate_normal_threshold,
)
from few_shot_anomaly_poc.config import load_config
from few_shot_anomaly_poc.ecc_residual import ECCResidualScoreResult
from few_shot_anomaly_poc.errors import (
    HOGScoringFailureCode,
    PreprocessingFailureCode,
    ThresholdCalibrationFailureCode,
)
from few_shot_anomaly_poc.hog_scoring import PatchHOGScoreResult


@pytest.fixture
def project_config():
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


def test_calibrate_ecc_selects_nearest_rank_threshold_and_strict_prediction(
    project_config,
) -> None:
    scores = {_path(index): _ecc_success(index / 100) for index in reversed(range(20))}

    result = calibrate_normal_threshold(
        scores,
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )

    assert result.succeeded
    assert result.status == "ok"
    assert result.failure_code is None
    assert result.method is CalibrationMethod.ECC_RESIDUAL
    assert result.quantile == 0.95
    assert result.sample_count == 20
    assert result.rank == 19
    assert result.threshold == pytest.approx(0.18)
    assert result.threshold_source_path == _path(18)
    assert result.score_order_paths == tuple(_path(index) for index in range(20))
    assert result.failed_score_count == 0
    assert not result.failed_score_paths
    assert result.predicted_anomalous_count == 1
    assert result.predicted_anomalous_paths == (_path(19),)
    assert result.realized_false_positive_rate == pytest.approx(0.05)
    assert result.invalid_score_path is None


def test_calibrate_hog_keeps_its_score_scale_separate(project_config) -> None:
    scores = {_path(index): _hog_success(float(index - 10)) for index in reversed(range(20))}

    result = calibrate_normal_threshold(
        scores,
        method=CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM,
        config=project_config,
    )

    assert result.succeeded
    assert result.method is CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM
    assert result.rank == 19
    assert result.threshold == 8.0
    assert result.threshold_source_path == _path(18)
    assert result.predicted_anomalous_paths == (_path(19),)
    assert result.realized_false_positive_rate == pytest.approx(0.05)


def test_calibrate_fixed_v01_sample_count_uses_rank_840(project_config) -> None:
    sample_count = 884
    scores = {
        _path(index): _ecc_success(index / sample_count) for index in reversed(range(sample_count))
    }

    result = calibrate_normal_threshold(
        scores,
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )

    assert result.succeeded
    assert result.sample_count == 884
    assert result.rank == 840
    assert result.threshold == pytest.approx(839 / 884)
    assert result.threshold_source_path == _path(839)
    assert result.predicted_anomalous_count == 44
    assert result.realized_false_positive_rate == pytest.approx(44 / 884)
    assert result.realized_false_positive_rate <= 0.05


def test_calibrate_breaks_score_ties_by_relative_path(project_config) -> None:
    scores = {_path(index): _ecc_success(0.5) for index in reversed(range(20))}

    result = calibrate_normal_threshold(
        scores,
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )

    assert result.succeeded
    assert result.score_order_paths == tuple(_path(index) for index in range(20))
    assert result.threshold_source_path == _path(18)
    assert result.threshold == 0.5
    assert result.predicted_anomalous_count == 0
    assert result.realized_false_positive_rate == 0.0


def test_calibrate_retains_failure_score_and_always_predicts_failure_as_anomalous(
    project_config,
) -> None:
    scores = {_path(index): _ecc_success(0.0) for index in range(18)}
    scores[_path(18)] = _ecc_success(1.0)
    scores[_path(19)] = _ecc_failure()

    result = calibrate_normal_threshold(
        scores,
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )

    assert result.succeeded
    assert result.rank == 19
    assert result.threshold == 1.0
    assert result.threshold_source_path == _path(18)
    assert result.failed_score_count == 1
    assert result.failed_score_paths == (_path(19),)
    assert result.predicted_anomalous_paths == (_path(19),)
    assert result.realized_false_positive_rate == pytest.approx(0.05)


def test_calibrate_is_repeatable_and_does_not_mutate_input(project_config) -> None:
    scores = {_path(index): _hog_success(float((index * 7) % 11)) for index in reversed(range(20))}
    original_items = tuple(scores.items())

    first = calibrate_normal_threshold(
        scores,
        method=CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM,
        config=project_config,
    )
    second = calibrate_normal_threshold(
        scores,
        method=CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM,
        config=project_config,
    )

    assert first == second
    assert tuple(scores.items()) == original_items


def test_calibrate_rejects_empty_input(project_config) -> None:
    result = calibrate_normal_threshold(
        {},
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )

    assert not result.succeeded
    assert result.failure_code is ThresholdCalibrationFailureCode.CALIBRATION_EMPTY
    assert result.method is CalibrationMethod.ECC_RESIDUAL
    assert result.sample_count == 0
    assert result.rank is None
    assert result.threshold is None
    assert not result.score_order_paths
    assert result.failed_score_count is None
    assert result.predicted_anomalous_count is None
    assert result.realized_false_positive_rate is None


def test_calibrate_rejects_string_method_identifier(project_config) -> None:
    result = calibrate_normal_threshold(
        {_path(0): _ecc_success(0.1)},
        method="ecc_residual",
        config=project_config,
    )

    assert result.failure_code is ThresholdCalibrationFailureCode.CALIBRATION_METHOD_INVALID
    assert result.method is None
    assert result.threshold is None


@pytest.mark.parametrize(
    "invalid_path",
    ["", "/absolute.JPG", "../parent.JPG", "pcb1/../escape.JPG", "windows\\path.JPG", None],
)
def test_calibrate_rejects_invalid_relative_path(
    project_config,
    invalid_path: object,
) -> None:
    result = calibrate_normal_threshold(
        {invalid_path: _ecc_success(0.1)},
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )

    assert result.failure_code is ThresholdCalibrationFailureCode.CALIBRATION_PATH_INVALID
    assert result.sample_count == 1
    assert result.invalid_score_path == (invalid_path if isinstance(invalid_path, str) else None)
    assert result.threshold is None


def test_calibrate_rejects_score_type_from_other_method(project_config) -> None:
    result = calibrate_normal_threshold(
        {_path(0): _hog_success(0.1)},
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )

    assert result.failure_code is ThresholdCalibrationFailureCode.CALIBRATION_SCORE_TYPE_MISMATCH
    assert result.invalid_score_path == _path(0)
    assert result.threshold is None


@pytest.mark.parametrize(
    ("method", "invalid_record"),
    [
        (CalibrationMethod.ECC_RESIDUAL, _ecc_success(float("nan"))),
        (
            CalibrationMethod.ECC_RESIDUAL,
            replace(
                _ecc_success(0.1),
                failure_code=PreprocessingFailureCode.INVALID_PREPROCESSED_IMAGE,
            ),
        ),
        (CalibrationMethod.ECC_RESIDUAL, _ecc_failure(0.9)),
        (CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM, _hog_success(1e12)),
        (CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM, _hog_failure(0.0)),
    ],
)
def test_calibrate_rejects_invalid_score_record(
    project_config,
    method: CalibrationMethod,
    invalid_record: ECCResidualScoreResult | PatchHOGScoreResult,
) -> None:
    result = calibrate_normal_threshold(
        {_path(0): invalid_record},
        method=method,
        config=project_config,
    )

    assert result.failure_code is ThresholdCalibrationFailureCode.CALIBRATION_SCORE_RECORD_INVALID
    assert result.invalid_score_path == _path(0)
    assert result.threshold is None


@pytest.mark.parametrize("invalid_quantile", [0.0, 1.1])
def test_calibrate_rejects_impossible_rank(project_config, invalid_quantile: float) -> None:
    changed_calibration = replace(
        project_config.threshold_calibration,
        quantile=invalid_quantile,
    )
    changed_config = replace(
        project_config,
        threshold_calibration=changed_calibration,
    )

    result = calibrate_normal_threshold(
        {_path(0): _ecc_success(0.1)},
        method=CalibrationMethod.ECC_RESIDUAL,
        config=changed_config,
    )

    assert result.failure_code is ThresholdCalibrationFailureCode.CALIBRATION_RESULT_INVALID
    assert result.threshold is None
