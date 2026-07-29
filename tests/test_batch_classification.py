from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import few_shot_anomaly_poc.calibration as calibration_module
from few_shot_anomaly_poc.calibration import (
    CalibrationMethod,
    FixedThresholdClassificationResult,
    NormalThresholdCalibrationResult,
    calibrate_normal_threshold,
    classify_fixed_threshold_batch,
)
from few_shot_anomaly_poc.config import ProjectConfig, load_config
from few_shot_anomaly_poc.ecc_residual import ECCResidualScoreResult
from few_shot_anomaly_poc.errors import (
    BatchClassificationFailureCode,
    HOGScoringFailureCode,
    PreprocessingFailureCode,
    ThresholdCalibrationFailureCode,
    ThresholdClassificationFailureCode,
)
from few_shot_anomaly_poc.hog_scoring import PatchHOGScoreResult


@pytest.fixture(scope="module")
def project_config() -> ProjectConfig:
    return load_config(Path("configs/v0.1.yaml"))


def _calibration_path(index: int) -> str:
    return f"pcb1/Data/Images/Normal/{index:04d}.JPG"


def _test_path(index: int) -> str:
    return f"pcb1/Data/Images/Test/{index:04d}.JPG"


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
        {_calibration_path(index): _ecc_success(index / 100) for index in reversed(range(20))},
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )
    assert result.succeeded
    assert result.threshold == pytest.approx(0.18)
    return result


def _calibrated_hog(project_config: ProjectConfig) -> NormalThresholdCalibrationResult:
    result = calibrate_normal_threshold(
        {
            _calibration_path(index): _hog_success(float(index - 10))
            for index in reversed(range(20))
        },
        method=CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM,
        config=project_config,
    )
    assert result.succeeded
    assert result.threshold == 8.0
    return result


def test_batch_interface_exposes_no_label_or_threshold_override() -> None:
    parameters = inspect.signature(classify_fixed_threshold_batch).parameters

    assert tuple(parameters) == ("score_results", "calibration", "config")
    assert all("label" not in name for name in parameters)
    assert "method" not in parameters
    assert "threshold" not in parameters


def test_batch_classifies_every_ecc_score_in_relative_path_order(
    project_config: ProjectConfig,
) -> None:
    calibration = _calibrated_ecc(project_config)
    scores = {
        _test_path(3): _ecc_failure(),
        _test_path(2): _ecc_success(0.19),
        _test_path(1): _ecc_success(0.18),
        _test_path(0): _ecc_success(0.17),
    }

    result = classify_fixed_threshold_batch(
        scores,
        calibration=calibration,
        config=project_config,
    )

    assert result.succeeded
    assert result.status == "ok"
    assert result.failure_code is None
    assert result.method is CalibrationMethod.ECC_RESIDUAL
    assert result.item_count == 4
    assert result.successful_item_count == 4
    assert result.classifications is not None
    assert result.ordered_paths == tuple(_test_path(index) for index in range(4))
    assert tuple(item.relative_path for item in result.classifications) == result.ordered_paths
    assert tuple(item.predicted_class for item in result.classifications) == (
        "normal",
        "normal",
        "anomalous",
        "anomalous",
    )
    assert result.threshold == pytest.approx(0.18)
    assert result.threshold_source_path == _calibration_path(18)
    assert result.normal_count == 2
    assert result.normal_paths == (_test_path(0), _test_path(1))
    assert result.anomalous_count == 2
    assert result.anomalous_paths == (_test_path(2), _test_path(3))
    assert result.score_failure_count == 1
    assert result.score_failure_paths == (_test_path(3),)
    assert result.failed_path is None
    assert result.item_failure_code is None


def test_batch_uses_unicode_code_point_path_order(
    project_config: ProjectConfig,
) -> None:
    paths = (
        "pcb1/Data/Images/Test/A.JPG",
        "pcb1/Data/Images/Test/a.JPG",
        "pcb1/Data/Images/Test/é.JPG",
        "pcb1/Data/Images/Test/あ.JPG",
    )
    scores = {path: _ecc_success(0.1) for path in reversed(paths)}

    result = classify_fixed_threshold_batch(
        scores,
        calibration=_calibrated_ecc(project_config),
        config=project_config,
    )

    assert result.succeeded
    assert result.ordered_paths == paths
    assert result.normal_paths == paths


def test_batch_classifies_hog_scores_on_hog_scale(
    project_config: ProjectConfig,
) -> None:
    calibration = _calibrated_hog(project_config)
    scores = {
        _test_path(3): _hog_failure(),
        _test_path(2): _hog_success(9.0),
        _test_path(1): _hog_success(8.0),
        _test_path(0): _hog_success(7.0),
    }

    result = classify_fixed_threshold_batch(
        scores,
        calibration=calibration,
        config=project_config,
    )

    assert result.succeeded
    assert result.method is CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM
    assert result.threshold == 8.0
    assert result.normal_paths == (_test_path(0), _test_path(1))
    assert result.anomalous_paths == (_test_path(2), _test_path(3))
    assert result.score_failure_paths == (_test_path(3),)


def test_batch_validates_calibration_once(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
) -> None:
    calibration = _calibrated_ecc(project_config)
    original_validator = calibration_module.normal_threshold_calibration_result_is_valid
    call_count = 0

    def count_validation(result, *, config):
        nonlocal call_count
        call_count += 1
        return original_validator(result, config=config)

    monkeypatch.setattr(
        calibration_module,
        "normal_threshold_calibration_result_is_valid",
        count_validation,
    )

    result = classify_fixed_threshold_batch(
        {_test_path(index): _ecc_success(0.1) for index in reversed(range(10))},
        calibration=calibration,
        config=project_config,
    )

    assert result.succeeded
    assert call_count == 1


def test_batch_is_repeatable_and_does_not_mutate_input(
    project_config: ProjectConfig,
) -> None:
    calibration = _calibrated_hog(project_config)
    scores = {_test_path(index): _hog_success(float(index + 6)) for index in reversed(range(4))}
    original_items = tuple(scores.items())

    first = classify_fixed_threshold_batch(
        scores,
        calibration=calibration,
        config=project_config,
    )
    second = classify_fixed_threshold_batch(
        scores,
        calibration=calibration,
        config=project_config,
    )

    assert first == second
    assert tuple(scores.items()) == original_items
    assert calibration.threshold == 8.0


def test_batch_rejects_empty_input_without_partial_output(
    project_config: ProjectConfig,
) -> None:
    result = classify_fixed_threshold_batch(
        {},
        calibration=_calibrated_ecc(project_config),
        config=project_config,
    )

    assert not result.succeeded
    assert result.failure_code is BatchClassificationFailureCode.BATCH_CLASSIFICATION_EMPTY
    assert result.item_count == 0
    assert result.successful_item_count == 0
    assert result.classifications is None
    assert not result.ordered_paths
    assert result.threshold is None
    assert result.normal_count is None
    assert result.anomalous_count is None
    assert result.score_failure_count is None
    assert result.failed_path is None
    assert result.item_failure_code is None


@pytest.mark.parametrize(
    "invalid_path",
    ["", "/absolute.JPG", "../parent.JPG", "pcb1/../escape.JPG", "windows\\path.JPG"],
)
def test_batch_rejects_invalid_path_before_classification(
    project_config: ProjectConfig,
    invalid_path: str,
) -> None:
    result = classify_fixed_threshold_batch(
        {
            _test_path(0): _ecc_success(0.1),
            invalid_path: _ecc_success(0.1),
        },
        calibration=_calibrated_ecc(project_config),
        config=project_config,
    )

    assert result.failure_code is BatchClassificationFailureCode.BATCH_CLASSIFICATION_PATH_INVALID
    assert result.item_count == 2
    assert result.successful_item_count == 0
    assert result.classifications is None
    assert result.failed_path == invalid_path
    assert (
        result.item_failure_code is ThresholdClassificationFailureCode.CLASSIFICATION_PATH_INVALID
    )


def test_batch_reports_first_invalid_path_in_unicode_order(
    project_config: ProjectConfig,
) -> None:
    scores = {
        "windows\\path.JPG": _ecc_success(0.1),
        _test_path(0): _ecc_success(0.1),
        "/absolute.JPG": _ecc_success(0.1),
        "../parent.JPG": _ecc_success(0.1),
    }

    first = classify_fixed_threshold_batch(
        scores,
        calibration=_calibrated_ecc(project_config),
        config=project_config,
    )
    second = classify_fixed_threshold_batch(
        dict(reversed(tuple(scores.items()))),
        calibration=_calibrated_ecc(project_config),
        config=project_config,
    )

    assert first == second
    assert first.failed_path == "../parent.JPG"
    assert first.successful_item_count == 0


def test_batch_rejects_failed_calibration_before_classification(
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

    result = classify_fixed_threshold_batch(
        {_test_path(0): _ecc_success(0.1)},
        calibration=failed_calibration,
        config=project_config,
    )

    assert (
        result.failure_code
        is BatchClassificationFailureCode.BATCH_CLASSIFICATION_CALIBRATION_INVALID
    )
    assert result.successful_item_count == 0
    assert result.classifications is None
    assert result.failed_path is None
    assert (
        result.item_failure_code
        is ThresholdClassificationFailureCode.CLASSIFICATION_CALIBRATION_INVALID
    )


def test_batch_discards_prior_decisions_after_method_mismatch(
    project_config: ProjectConfig,
) -> None:
    scores = {
        _test_path(2): _hog_success(0.1),
        _test_path(1): _ecc_success(0.1),
        _test_path(0): _ecc_success(0.1),
    }

    result = classify_fixed_threshold_batch(
        scores,
        calibration=_calibrated_ecc(project_config),
        config=project_config,
    )

    assert result.failure_code is BatchClassificationFailureCode.BATCH_CLASSIFICATION_ITEM_FAILED
    assert result.item_count == 3
    assert result.successful_item_count == 2
    assert result.classifications is None
    assert not result.ordered_paths
    assert result.failed_path == _test_path(2)
    assert (
        result.item_failure_code
        is ThresholdClassificationFailureCode.CLASSIFICATION_SCORE_TYPE_MISMATCH
    )
    assert result.normal_count is None
    assert result.anomalous_count is None


def test_batch_discards_prior_decisions_after_invalid_score_record(
    project_config: ProjectConfig,
) -> None:
    scores = {
        _test_path(1): _ecc_failure(0.9),
        _test_path(0): _ecc_success(0.1),
    }

    result = classify_fixed_threshold_batch(
        scores,
        calibration=_calibrated_ecc(project_config),
        config=project_config,
    )

    assert result.failure_code is BatchClassificationFailureCode.BATCH_CLASSIFICATION_ITEM_FAILED
    assert result.successful_item_count == 1
    assert result.classifications is None
    assert result.failed_path == _test_path(1)
    assert (
        result.item_failure_code
        is ThresholdClassificationFailureCode.CLASSIFICATION_SCORE_RECORD_INVALID
    )


def test_batch_rejects_internally_incomplete_result(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
) -> None:
    calibration = _calibrated_ecc(project_config)

    def incomplete_classification(relative_path, score_result, *, calibration, config):
        return FixedThresholdClassificationResult(
            status="ok",
            failure_code=None,
            method=CalibrationMethod.ECC_RESIDUAL,
            relative_path=relative_path,
            score_status="ok",
            score_failure_code=None,
            anomaly_score=0.1,
            threshold=0.18,
            threshold_source_path=_calibration_path(18),
            calibration_sample_count=20,
            calibration_rank=19,
            predicted_class=None,
            is_anomalous=None,
            decision_reason=None,
            score_margin=-0.08,
        )

    monkeypatch.setattr(
        calibration_module,
        "_classify_fixed_threshold_validated",
        incomplete_classification,
    )

    result = classify_fixed_threshold_batch(
        {_test_path(0): _ecc_success(0.1)},
        calibration=calibration,
        config=project_config,
    )

    assert result.failure_code is BatchClassificationFailureCode.BATCH_CLASSIFICATION_RESULT_INVALID
    assert result.successful_item_count == 0
    assert result.classifications is None
    assert result.failed_path == _test_path(0)
    assert (
        result.item_failure_code is ThresholdClassificationFailureCode.CLASSIFICATION_RESULT_INVALID
    )
    assert result.normal_count is None
    assert result.anomalous_count is None
