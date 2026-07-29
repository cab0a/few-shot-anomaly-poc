from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import few_shot_anomaly_poc.hog_scoring as scoring_module
from few_shot_anomaly_poc.config import ProjectConfig, load_config
from few_shot_anomaly_poc.errors import (
    HOGFeatureFailureCode,
    HOGScoringFailureCode,
    HOGScoringStateError,
    PreprocessingFailureCode,
)
from few_shot_anomaly_poc.hog_features import (
    PatchHOGFeatureResult,
    fixed_patch_positions,
)
from few_shot_anomaly_poc.hog_models import (
    PatchHOGModelFitResult,
    fit_position_one_class_svms,
)
from few_shot_anomaly_poc.hog_scalers import (
    PatchHOGScalerFitResult,
    fit_position_scalers,
)
from few_shot_anomaly_poc.hog_scoring import score_patch_hog


@pytest.fixture(scope="module")
def project_config() -> ProjectConfig:
    return load_config(Path("configs/v0.1.yaml"))


def _synthetic_image() -> np.ndarray:
    image = np.zeros((512, 512), dtype=np.float32)
    cv2.rectangle(image, (45, 60), (260, 310), color=0.75, thickness=-1)
    cv2.circle(image, (385, 350), 58, color=0.25, thickness=-1)
    cv2.line(image, (8, 500), (500, 8), color=1.0, thickness=5)
    return image


def _reference_features(
    project_config: ProjectConfig,
) -> dict[str, PatchHOGFeatureResult]:
    positions = fixed_patch_positions(project_config)
    assert positions is not None
    references = {}
    for index in reversed(range(project_config.selection.reference_count)):
        generator = np.random.default_rng(700 + index)
        references[f"reference-{index:02d}.png"] = PatchHOGFeatureResult(
            status="ok",
            failure_code=None,
            features=generator.random(
                (
                    project_config.patch_hog.patch_count,
                    project_config.patch_hog.descriptor_length,
                ),
                dtype=np.float32,
            ),
            positions=positions,
            failed_patch_index=None,
        )
    return references


@pytest.fixture(scope="module")
def fitted_states(
    project_config: ProjectConfig,
) -> tuple[PatchHOGScalerFitResult, PatchHOGModelFitResult]:
    references = _reference_features(project_config)
    scaler_fit = fit_position_scalers(references, config=project_config)
    assert scaler_fit.succeeded
    model_fit = fit_position_one_class_svms(
        references,
        scaler_fit=scaler_fit,
        config=project_config,
    )
    assert model_fit.succeeded
    return scaler_fit, model_fit


def test_score_integrates_hog_scalers_and_models_repeatably_without_mutation(
    project_config: ProjectConfig,
    fitted_states: tuple[PatchHOGScalerFitResult, PatchHOGModelFitResult],
) -> None:
    scaler_fit, model_fit = fitted_states
    image = _synthetic_image()
    original_image = image.copy()
    assert scaler_fit.scalers is not None
    assert model_fit.models is not None
    original_mean = scaler_fit.scalers[0].mean_.copy()
    original_support = model_fit.models[0].support_.copy()

    first = score_patch_hog(
        image,
        scaler_fit=scaler_fit,
        model_fit=model_fit,
        config=project_config,
    )
    second = score_patch_hog(
        image,
        scaler_fit=scaler_fit,
        model_fit=model_fit,
        config=project_config,
    )

    assert first.succeeded and second.succeeded
    assert first.score_status == "ok"
    assert first.failure_code is None
    assert first.anomaly_score == second.anomaly_score
    assert first.patch_anomaly_scores == second.patch_anomaly_scores
    assert first.top_patch_indices == second.top_patch_indices
    assert first.patch_anomaly_scores is not None
    assert len(first.patch_anomaly_scores) == 225
    assert all(math_score < 1e12 for math_score in map(abs, first.patch_anomaly_scores))
    assert first.top_patch_count == 12
    assert len(first.top_patch_indices) == 12
    assert len(set(first.top_patch_indices)) == 12
    assert first.successful_patch_count == 225
    assert first.failed_patch_index is None
    assert np.array_equal(image, original_image)
    assert np.array_equal(scaler_fit.scalers[0].mean_, original_mean)
    assert np.array_equal(model_fit.models[0].support_, original_support)


def test_score_negates_decisions_and_averages_fixed_top_twelve(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    fitted_states: tuple[PatchHOGScalerFitResult, PatchHOGModelFitResult],
) -> None:
    scaler_fit, model_fit = fitted_states
    call_count = 0

    def indexed_decision(self, samples):
        nonlocal call_count
        position_index = call_count
        call_count += 1
        return np.asarray([-float(position_index)], dtype=np.float64)

    monkeypatch.setattr(OneClassSVM, "decision_function", indexed_decision)

    result = score_patch_hog(
        _synthetic_image(),
        scaler_fit=scaler_fit,
        model_fit=model_fit,
        config=project_config,
    )

    assert result.succeeded
    assert result.patch_anomaly_scores == tuple(float(index) for index in range(225))
    assert result.top_patch_count == 12
    assert result.top_patch_indices == tuple(range(224, 212, -1))
    assert result.anomaly_score == pytest.approx(218.5)
    assert call_count == 225


def test_score_breaks_equal_patch_score_ties_by_position(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    fitted_states: tuple[PatchHOGScalerFitResult, PatchHOGModelFitResult],
) -> None:
    scaler_fit, model_fit = fitted_states
    monkeypatch.setattr(
        OneClassSVM,
        "decision_function",
        lambda *args, **kwargs: np.asarray([0.0], dtype=np.float64),
    )

    result = score_patch_hog(
        _synthetic_image(),
        scaler_fit=scaler_fit,
        model_fit=model_fit,
        config=project_config,
    )

    assert result.succeeded
    assert result.top_patch_indices == tuple(range(12))
    assert result.anomaly_score == 0.0


def test_score_converts_preprocessing_failure_to_fixed_score(
    project_config: ProjectConfig,
    fitted_states: tuple[PatchHOGScalerFitResult, PatchHOGModelFitResult],
) -> None:
    scaler_fit, model_fit = fitted_states

    result = score_patch_hog(
        _synthetic_image().astype(np.float64),
        scaler_fit=scaler_fit,
        model_fit=model_fit,
        config=project_config,
    )

    assert not result.succeeded
    assert result.failure_code is PreprocessingFailureCode.INVALID_PREPROCESSED_IMAGE
    assert result.anomaly_score == 1e12
    assert result.patch_anomaly_scores is None
    assert result.top_patch_count is None
    assert not result.top_patch_indices
    assert result.successful_patch_count == 0
    assert result.failed_patch_index is None


def test_score_preserves_feature_extraction_failure(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    fitted_states: tuple[PatchHOGScalerFitResult, PatchHOGModelFitResult],
) -> None:
    scaler_fit, model_fit = fitted_states
    positions = fixed_patch_positions(project_config)
    assert positions is not None
    monkeypatch.setattr(
        scoring_module,
        "extract_patch_hog_features",
        lambda *args, **kwargs: PatchHOGFeatureResult(
            status="failed",
            failure_code=HOGFeatureFailureCode.HOG_EXTRACTION_FAILED,
            features=None,
            positions=positions,
            failed_patch_index=7,
        ),
    )

    result = score_patch_hog(
        _synthetic_image(),
        scaler_fit=scaler_fit,
        model_fit=model_fit,
        config=project_config,
    )

    assert result.failure_code is HOGFeatureFailureCode.HOG_EXTRACTION_FAILED
    assert result.failed_patch_index == 7
    assert result.anomaly_score == 1e12


def test_score_rejects_invalid_successful_feature_result(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    fitted_states: tuple[PatchHOGScalerFitResult, PatchHOGModelFitResult],
) -> None:
    scaler_fit, model_fit = fitted_states
    positions = fixed_patch_positions(project_config)
    assert positions is not None
    monkeypatch.setattr(
        scoring_module,
        "extract_patch_hog_features",
        lambda *args, **kwargs: PatchHOGFeatureResult(
            status="ok",
            failure_code=None,
            features=np.zeros((224, 324), dtype=np.float32),
            positions=positions,
            failed_patch_index=None,
        ),
    )

    result = score_patch_hog(
        _synthetic_image(),
        scaler_fit=scaler_fit,
        model_fit=model_fit,
        config=project_config,
    )

    assert result.failure_code is HOGScoringFailureCode.HOG_SCORE_FEATURE_RESULT_INVALID
    assert result.anomaly_score == 1e12


def test_score_rejects_invalid_fitted_scaler_before_extraction(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    fitted_states: tuple[PatchHOGScalerFitResult, PatchHOGModelFitResult],
) -> None:
    scaler_fit, model_fit = fitted_states
    assert scaler_fit.scalers is not None
    invalid_scale = scaler_fit.scalers[4].scale_.copy()
    invalid_scale[0] = 0.0
    monkeypatch.setattr(scaler_fit.scalers[4], "scale_", invalid_scale)

    with pytest.raises(HOGScoringStateError, match="position 4"):
        score_patch_hog(
            _synthetic_image(),
            scaler_fit=scaler_fit,
            model_fit=model_fit,
            config=project_config,
        )


def test_score_rejects_non_scaler_fitted_state(
    project_config: ProjectConfig,
    fitted_states: tuple[PatchHOGScalerFitResult, PatchHOGModelFitResult],
) -> None:
    scaler_fit, model_fit = fitted_states
    assert scaler_fit.scalers is not None
    invalid_scaler_fit = replace(
        scaler_fit,
        scalers=(object(), *scaler_fit.scalers[1:]),
    )

    with pytest.raises(HOGScoringStateError, match="position 0"):
        score_patch_hog(
            _synthetic_image(),
            scaler_fit=invalid_scaler_fit,
            model_fit=model_fit,
            config=project_config,
        )


def test_score_rejects_invalid_fitted_model_before_extraction(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    fitted_states: tuple[PatchHOGScalerFitResult, PatchHOGModelFitResult],
) -> None:
    scaler_fit, model_fit = fitted_states
    assert model_fit.models is not None
    monkeypatch.setattr(model_fit.models[5], "fit_status_", 1)

    with pytest.raises(HOGScoringStateError, match="position 5"):
        score_patch_hog(
            _synthetic_image(),
            scaler_fit=scaler_fit,
            model_fit=model_fit,
            config=project_config,
        )


def test_score_rejects_mismatched_fitted_reference_sets(
    project_config: ProjectConfig,
    fitted_states: tuple[PatchHOGScalerFitResult, PatchHOGModelFitResult],
) -> None:
    scaler_fit, model_fit = fitted_states
    changed_paths = (*model_fit.reference_paths[:-1], "replacement.png")
    mismatched_model_fit = replace(model_fit, reference_paths=changed_paths)

    with pytest.raises(HOGScoringStateError, match="model state"):
        score_patch_hog(
            _synthetic_image(),
            scaler_fit=scaler_fit,
            model_fit=mismatched_model_fit,
            config=project_config,
        )


def test_score_discards_partial_scores_after_transform_failure(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    fitted_states: tuple[PatchHOGScalerFitResult, PatchHOGModelFitResult],
) -> None:
    scaler_fit, model_fit = fitted_states
    original_transform = StandardScaler.transform
    call_count = 0

    def fail_fourth_transform(self, samples, *args, **kwargs):
        nonlocal call_count
        current = call_count
        call_count += 1
        if current == 3:
            raise ValueError("synthetic transform failure")
        return original_transform(self, samples, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "transform", fail_fourth_transform)

    result = score_patch_hog(
        _synthetic_image(),
        scaler_fit=scaler_fit,
        model_fit=model_fit,
        config=project_config,
    )

    assert result.failure_code is HOGScoringFailureCode.HOG_SCORE_TRANSFORM_FAILED
    assert result.failed_patch_index == 3
    assert result.successful_patch_count == 3
    assert result.patch_anomaly_scores is None
    assert result.anomaly_score == 1e12
    assert call_count == 4


def test_score_rejects_invalid_transformed_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    fitted_states: tuple[PatchHOGScalerFitResult, PatchHOGModelFitResult],
) -> None:
    scaler_fit, model_fit = fitted_states
    monkeypatch.setattr(
        StandardScaler,
        "transform",
        lambda *args, **kwargs: np.zeros((1, 324), dtype=np.float64),
    )

    result = score_patch_hog(
        _synthetic_image(),
        scaler_fit=scaler_fit,
        model_fit=model_fit,
        config=project_config,
    )

    assert result.failure_code is HOGScoringFailureCode.HOG_SCORE_TRANSFORM_INVALID
    assert result.failed_patch_index == 0
    assert result.successful_patch_count == 0


def test_score_discards_partial_scores_after_decision_failure(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    fitted_states: tuple[PatchHOGScalerFitResult, PatchHOGModelFitResult],
) -> None:
    scaler_fit, model_fit = fitted_states
    original_decision = OneClassSVM.decision_function
    call_count = 0

    def fail_fourth_decision(self, samples, *args, **kwargs):
        nonlocal call_count
        current = call_count
        call_count += 1
        if current == 3:
            raise ValueError("synthetic decision failure")
        return original_decision(self, samples, *args, **kwargs)

    monkeypatch.setattr(OneClassSVM, "decision_function", fail_fourth_decision)

    result = score_patch_hog(
        _synthetic_image(),
        scaler_fit=scaler_fit,
        model_fit=model_fit,
        config=project_config,
    )

    assert result.failure_code is HOGScoringFailureCode.HOG_SCORE_DECISION_FAILED
    assert result.failed_patch_index == 3
    assert result.successful_patch_count == 3
    assert result.patch_anomaly_scores is None
    assert call_count == 4


@pytest.mark.parametrize(
    "invalid_decision",
    [
        np.zeros((1, 1), dtype=np.float64),
        np.zeros((1,), dtype=np.float32),
        np.asarray([np.nan], dtype=np.float64),
        np.asarray([-1e12], dtype=np.float64),
    ],
)
def test_score_rejects_invalid_patch_decision(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    fitted_states: tuple[PatchHOGScalerFitResult, PatchHOGModelFitResult],
    invalid_decision: np.ndarray,
) -> None:
    scaler_fit, model_fit = fitted_states
    monkeypatch.setattr(
        OneClassSVM,
        "decision_function",
        lambda *args, **kwargs: invalid_decision,
    )

    result = score_patch_hog(
        _synthetic_image(),
        scaler_fit=scaler_fit,
        model_fit=model_fit,
        config=project_config,
    )

    assert result.failure_code is HOGScoringFailureCode.HOG_SCORE_PATCH_INVALID
    assert result.failed_patch_index == 0
    assert result.successful_patch_count == 0
    assert result.anomaly_score == 1e12


def test_score_converts_invalid_aggregate_to_fixed_failure_score(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    fitted_states: tuple[PatchHOGScalerFitResult, PatchHOGModelFitResult],
) -> None:
    scaler_fit, model_fit = fitted_states
    monkeypatch.setattr(
        scoring_module,
        "_mean_top_patch_scores",
        lambda *args, **kwargs: float("nan"),
    )

    result = score_patch_hog(
        _synthetic_image(),
        scaler_fit=scaler_fit,
        model_fit=model_fit,
        config=project_config,
    )

    assert result.failure_code is HOGScoringFailureCode.HOG_SCORE_AGGREGATION_INVALID
    assert result.successful_patch_count == 225
    assert result.failed_patch_index is None
    assert result.patch_anomaly_scores is None
    assert result.anomaly_score == 1e12


def test_score_converts_aggregation_exception_to_fixed_failure_score(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
    fitted_states: tuple[PatchHOGScalerFitResult, PatchHOGModelFitResult],
) -> None:
    scaler_fit, model_fit = fitted_states

    def fail_aggregation(*args, **kwargs):
        raise ValueError("synthetic aggregation failure")

    monkeypatch.setattr(
        scoring_module,
        "_mean_top_patch_scores",
        fail_aggregation,
    )

    result = score_patch_hog(
        _synthetic_image(),
        scaler_fit=scaler_fit,
        model_fit=model_fit,
        config=project_config,
    )

    assert result.failure_code is HOGScoringFailureCode.HOG_SCORE_AGGREGATION_INVALID
    assert result.successful_patch_count == 225
    assert result.patch_anomaly_scores is None
    assert result.anomaly_score == 1e12
