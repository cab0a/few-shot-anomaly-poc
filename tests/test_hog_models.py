from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

import few_shot_anomaly_poc.hog_models as model_module
from few_shot_anomaly_poc.config import load_config
from few_shot_anomaly_poc.errors import (
    HOGFeatureFailureCode,
    HOGModelFailureCode,
    HOGScalerFailureCode,
)
from few_shot_anomaly_poc.hog_features import (
    PatchHOGFeatureResult,
    extract_patch_hog_features,
    fixed_patch_positions,
)
from few_shot_anomaly_poc.hog_models import (
    fit_position_one_class_svms,
    position_one_class_svm_state_is_valid,
)
from few_shot_anomaly_poc.hog_scalers import fit_position_scalers


@pytest.fixture
def project_config():
    return load_config(Path("configs/v0.1.yaml"))


def _synthetic_image() -> np.ndarray:
    image = np.zeros((512, 512), dtype=np.float32)
    cv2.rectangle(image, (40, 55), (255, 305), color=0.8, thickness=-1)
    cv2.circle(image, (380, 345), 60, color=0.2, thickness=-1)
    cv2.line(image, (5, 500), (500, 5), color=1.0, thickness=5)
    return image


def _feature_result(
    features: np.ndarray,
    *,
    project_config,
) -> PatchHOGFeatureResult:
    positions = fixed_patch_positions(project_config)
    assert positions is not None
    return PatchHOGFeatureResult(
        status="ok",
        failure_code=None,
        features=features,
        positions=positions,
        failed_patch_index=None,
    )


def _varied_reference_features(project_config) -> dict[str, PatchHOGFeatureResult]:
    references = {}
    for index in reversed(range(20)):
        generator = np.random.default_rng(100 + index)
        features = generator.random((225, 324), dtype=np.float32)
        references[f"reference-{index:02d}.png"] = _feature_result(
            features,
            project_config=project_config,
        )
    return references


def test_fit_integrates_real_hog_features_and_scalers(project_config) -> None:
    extracted = extract_patch_hog_features(
        _synthetic_image(),
        config=project_config,
    )
    assert extracted.succeeded
    references = {f"reference-{index:02d}.png": extracted for index in reversed(range(20))}
    scaler_fit = fit_position_scalers(references, config=project_config)
    assert scaler_fit.succeeded

    result = fit_position_one_class_svms(
        references,
        scaler_fit=scaler_fit,
        config=project_config,
    )

    assert result.succeeded
    assert result.status == "ok"
    assert result.failure_code is None
    assert result.reference_count == 20
    assert result.reference_paths[0] == "reference-00.png"
    assert result.reference_paths[-1] == "reference-19.png"
    assert result.models is not None
    assert len(result.models) == 225
    assert result.successful_position_count == 225
    assert result.failed_reference_path is None
    assert result.reference_failure_code is None
    assert result.scaler_failure_code is None
    assert result.failed_position_index is None
    assert all(
        position_one_class_svm_state_is_valid(model, config=project_config)
        for model in result.models
    )


def test_fit_is_position_specific_repeatable_and_does_not_mutate_inputs(
    project_config,
) -> None:
    references = _varied_reference_features(project_config)
    scaler_fit = fit_position_scalers(references, config=project_config)
    assert scaler_fit.succeeded
    original_features = references["reference-00.png"].features
    assert original_features is not None
    original_features = original_features.copy()
    assert scaler_fit.scalers is not None
    original_scales = [scaler.scale_.copy() for scaler in scaler_fit.scalers]

    first = fit_position_one_class_svms(
        references,
        scaler_fit=scaler_fit,
        config=project_config,
    )
    second = fit_position_one_class_svms(
        references,
        scaler_fit=scaler_fit,
        config=project_config,
    )

    assert first.succeeded and second.succeeded
    assert first.reference_paths == tuple(f"reference-{index:02d}.png" for index in range(20))
    assert first.models is not None
    assert second.models is not None
    for first_model, second_model in zip(
        first.models,
        second.models,
        strict=True,
    ):
        assert np.array_equal(first_model.support_, second_model.support_)
        assert np.array_equal(
            first_model.support_vectors_,
            second_model.support_vectors_,
        )
        assert np.array_equal(first_model.dual_coef_, second_model.dual_coef_)
        assert np.array_equal(first_model.intercept_, second_model.intercept_)
        assert np.array_equal(first_model.offset_, second_model.offset_)
        assert first_model.fit_status_ == 0
        assert first_model.n_features_in_ == 324
        assert first_model.shape_fit_ == (20, 324)
        assert first_model.kernel == "rbf"
        assert first_model.gamma == "scale"
        assert first_model.nu == 0.05
        assert first_model.tol == 0.001
        assert first_model.shrinking is True
        assert first_model.cache_size == 200.0
        assert first_model.max_iter == -1
        assert first_model.verbose is False

    current_features = references["reference-00.png"].features
    assert current_features is not None
    assert np.array_equal(current_features, original_features)
    assert scaler_fit.scalers is not None
    assert all(
        np.array_equal(scaler.scale_, original)
        for scaler, original in zip(
            scaler_fit.scalers,
            original_scales,
            strict=True,
        )
    )


def test_fit_requires_exact_reference_count(project_config) -> None:
    references = _varied_reference_features(project_config)
    scaler_fit = fit_position_scalers(references, config=project_config)
    references.pop("reference-19.png")

    result = fit_position_one_class_svms(
        references,
        scaler_fit=scaler_fit,
        config=project_config,
    )

    assert result.failure_code is HOGModelFailureCode.HOG_MODEL_REFERENCE_COUNT_INVALID
    assert result.reference_count == 19
    assert result.models is None
    assert result.successful_position_count == 0


def test_fit_rejects_invalid_reference_path(project_config) -> None:
    references = _varied_reference_features(project_config)
    references[""] = references.pop("reference-19.png")
    scaler_fit = fit_position_scalers(references, config=project_config)

    result = fit_position_one_class_svms(
        references,
        scaler_fit=scaler_fit,
        config=project_config,
    )

    assert result.failure_code is HOGModelFailureCode.HOG_MODEL_REFERENCE_SET_INVALID
    assert result.reference_count == 20
    assert not result.reference_paths
    assert result.models is None


def test_fit_preserves_reference_feature_failure(project_config) -> None:
    references = _varied_reference_features(project_config)
    scaler_fit = fit_position_scalers(references, config=project_config)
    positions = fixed_patch_positions(project_config)
    assert positions is not None
    references["reference-08.png"] = PatchHOGFeatureResult(
        status="failed",
        failure_code=HOGFeatureFailureCode.HOG_EXTRACTION_FAILED,
        features=None,
        positions=positions,
        failed_patch_index=40,
    )

    result = fit_position_one_class_svms(
        references,
        scaler_fit=scaler_fit,
        config=project_config,
    )

    assert result.failure_code is HOGModelFailureCode.HOG_MODEL_REFERENCE_FEATURES_INVALID
    assert result.failed_reference_path == "reference-08.png"
    assert result.reference_failure_code is HOGFeatureFailureCode.HOG_EXTRACTION_FAILED
    assert result.models is None


def test_fit_preserves_failed_scaler_state(project_config) -> None:
    references = _varied_reference_features(project_config)
    scaler_fit = fit_position_scalers(references, config=project_config)
    failed_scaler_fit = replace(
        scaler_fit,
        status="FIT_FAILED",
        failure_code=HOGScalerFailureCode.HOG_FIT_SCALER_FAILED,
        scalers=None,
        successful_position_count=10,
        failed_position_index=10,
    )

    result = fit_position_one_class_svms(
        references,
        scaler_fit=failed_scaler_fit,
        config=project_config,
    )

    assert result.failure_code is HOGModelFailureCode.HOG_MODEL_SCALER_STATE_INVALID
    assert result.scaler_failure_code is HOGScalerFailureCode.HOG_FIT_SCALER_FAILED
    assert result.failed_position_index is None
    assert result.models is None


def test_fit_rejects_scalers_from_different_reference_set(project_config) -> None:
    references = _varied_reference_features(project_config)
    scaler_fit = fit_position_scalers(references, config=project_config)
    references["different-reference.png"] = references.pop("reference-19.png")

    result = fit_position_one_class_svms(
        references,
        scaler_fit=scaler_fit,
        config=project_config,
    )

    assert result.failure_code is HOGModelFailureCode.HOG_MODEL_SCALER_STATE_INVALID
    assert result.failed_position_index is None
    assert result.models is None


def test_fit_rejects_corrupted_scaler_state(project_config) -> None:
    references = _varied_reference_features(project_config)
    scaler_fit = fit_position_scalers(references, config=project_config)
    assert scaler_fit.scalers is not None
    scaler_fit.scalers[4].var_[0] = 0.0
    scaler_fit.scalers[4].scale_[0] = 2.0

    result = fit_position_one_class_svms(
        references,
        scaler_fit=scaler_fit,
        config=project_config,
    )

    assert result.failure_code is HOGModelFailureCode.HOG_MODEL_SCALER_STATE_INVALID
    assert result.failed_position_index == 4
    assert result.models is None


def test_fit_discards_partial_models_after_transform_failure(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    references = _varied_reference_features(project_config)
    scaler_fit = fit_position_scalers(references, config=project_config)
    original_transform = StandardScaler.transform
    call_count = 0

    def fail_fourth_transform(self, samples, *args, **kwargs):
        nonlocal call_count
        current = call_count
        call_count += 1
        if current == 3:
            raise ValueError("synthetic transform failure")
        return original_transform(self, samples, *args, **kwargs)

    monkeypatch.setattr(
        StandardScaler,
        "transform",
        fail_fourth_transform,
    )

    result = fit_position_one_class_svms(
        references,
        scaler_fit=scaler_fit,
        config=project_config,
    )

    assert result.failure_code is HOGModelFailureCode.HOG_MODEL_TRANSFORM_FAILED
    assert result.failed_position_index == 3
    assert result.successful_position_count == 3
    assert result.models is None
    assert call_count == 4


def test_fit_rejects_invalid_transformed_samples(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    references = _varied_reference_features(project_config)
    scaler_fit = fit_position_scalers(references, config=project_config)
    monkeypatch.setattr(
        StandardScaler,
        "transform",
        lambda *args, **kwargs: np.full((20, 324), np.nan, dtype=np.float32),
    )

    result = fit_position_one_class_svms(
        references,
        scaler_fit=scaler_fit,
        config=project_config,
    )

    assert result.failure_code is HOGModelFailureCode.HOG_MODEL_TRANSFORM_INVALID
    assert result.failed_position_index == 0
    assert result.successful_position_count == 0
    assert result.models is None


def test_fit_discards_partial_models_after_library_failure(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    references = _varied_reference_features(project_config)
    scaler_fit = fit_position_scalers(references, config=project_config)
    original_fit = model_module.OneClassSVM.fit
    call_count = 0

    def fail_fourth_model(self, samples, *args, **kwargs):
        nonlocal call_count
        current = call_count
        call_count += 1
        if current == 3:
            raise ValueError("synthetic model failure")
        return original_fit(self, samples, *args, **kwargs)

    monkeypatch.setattr(model_module.OneClassSVM, "fit", fail_fourth_model)

    result = fit_position_one_class_svms(
        references,
        scaler_fit=scaler_fit,
        config=project_config,
    )

    assert result.failure_code is HOGModelFailureCode.HOG_MODEL_FIT_FAILED
    assert result.failed_position_index == 3
    assert result.successful_position_count == 3
    assert result.models is None
    assert call_count == 4


def test_fit_rejects_invalid_model_state(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    references = _varied_reference_features(project_config)
    scaler_fit = fit_position_scalers(references, config=project_config)
    original_fit = model_module.OneClassSVM.fit

    def corrupt_fit_status(self, samples, *args, **kwargs):
        fitted = original_fit(self, samples, *args, **kwargs)
        fitted.fit_status_ = 1
        return fitted

    monkeypatch.setattr(model_module.OneClassSVM, "fit", corrupt_fit_status)

    result = fit_position_one_class_svms(
        references,
        scaler_fit=scaler_fit,
        config=project_config,
    )

    assert result.failure_code is HOGModelFailureCode.HOG_MODEL_STATE_INVALID
    assert result.failed_position_index == 0
    assert result.successful_position_count == 0
    assert result.models is None
