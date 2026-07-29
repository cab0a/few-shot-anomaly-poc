from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

import few_shot_anomaly_poc.hog_scalers as scaler_module
from few_shot_anomaly_poc.config import load_config
from few_shot_anomaly_poc.errors import (
    HOGFeatureFailureCode,
    HOGScalerFailureCode,
)
from few_shot_anomaly_poc.hog_features import (
    PatchHOGFeatureResult,
    extract_patch_hog_features,
    fixed_patch_positions,
)
from few_shot_anomaly_poc.hog_scalers import fit_position_scalers


@pytest.fixture
def project_config():
    return load_config(Path("configs/v0.1.yaml"))


def _synthetic_image() -> np.ndarray:
    image = np.zeros((512, 512), dtype=np.float32)
    cv2.rectangle(image, (45, 60), (250, 300), color=0.75, thickness=-1)
    cv2.circle(image, (375, 340), 65, color=0.25, thickness=-1)
    cv2.line(image, (0, 500), (500, 0), color=1.0, thickness=7)
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
        generator = np.random.default_rng(index)
        features = generator.random((225, 324), dtype=np.float32)
        references[f"reference-{index:02d}.png"] = _feature_result(
            features,
            project_config=project_config,
        )
    return references


def test_fit_integrates_real_features_and_accepts_constant_dimensions(project_config) -> None:
    extracted = extract_patch_hog_features(
        _synthetic_image(),
        config=project_config,
    )
    assert extracted.succeeded
    references = {f"reference-{index:02d}.png": extracted for index in reversed(range(20))}

    result = fit_position_scalers(references, config=project_config)

    assert result.succeeded
    assert result.status == "ok"
    assert result.failure_code is None
    assert result.reference_count == 20
    assert result.reference_paths[0] == "reference-00.png"
    assert result.reference_paths[-1] == "reference-19.png"
    assert result.successful_position_count == 225
    assert result.failed_reference_path is None
    assert result.reference_failure_code is None
    assert result.failed_position_index is None
    assert result.scalers is not None
    assert len(result.scalers) == 225
    assert all(np.all(scaler.var_ == 0.0) for scaler in result.scalers)
    assert all(np.all(scaler.scale_ == 1.0) for scaler in result.scalers)


def test_fit_is_position_specific_repeatable_and_does_not_mutate_features(
    project_config,
) -> None:
    references = _varied_reference_features(project_config)
    original = references["reference-00.png"].features
    assert original is not None
    original = original.copy()

    first = fit_position_scalers(references, config=project_config)
    second = fit_position_scalers(references, config=project_config)

    assert first.succeeded and second.succeeded
    assert first.reference_paths == tuple(f"reference-{index:02d}.png" for index in range(20))
    assert first.scalers is not None
    assert second.scalers is not None
    first_means = np.stack([scaler.mean_ for scaler in first.scalers])
    second_means = np.stack([scaler.mean_ for scaler in second.scalers])
    first_variances = np.stack([scaler.var_ for scaler in first.scalers])
    second_variances = np.stack([scaler.var_ for scaler in second.scalers])
    first_scales = np.stack([scaler.scale_ for scaler in first.scalers])
    second_scales = np.stack([scaler.scale_ for scaler in second.scalers])
    assert np.array_equal(first_means, second_means)
    assert np.array_equal(first_variances, second_variances)
    assert np.array_equal(first_scales, second_scales)

    sorted_features = []
    for path in first.reference_paths:
        matrix = references[path].features
        assert matrix is not None
        sorted_features.append(matrix)
    feature_tensor = np.stack(sorted_features)
    expected_means = np.mean(feature_tensor, axis=0, dtype=np.float64)
    assert first_means == pytest.approx(expected_means)
    assert first_means.shape == (225, 324)
    assert first_variances.shape == (225, 324)
    assert first_scales.shape == (225, 324)
    assert np.isfinite(first_means).all()
    assert np.all(first_variances >= 0.0)
    assert np.all(first_scales > 0.0)
    assert all(scaler.copy is True for scaler in first.scalers)
    assert all(scaler.with_mean is True for scaler in first.scalers)
    assert all(scaler.with_std is True for scaler in first.scalers)
    assert all(scaler.n_features_in_ == 324 for scaler in first.scalers)
    assert all(float(scaler.n_samples_seen_) == 20.0 for scaler in first.scalers)
    current = references["reference-00.png"].features
    assert current is not None
    assert np.array_equal(current, original)


def test_fit_requires_exact_reference_count(project_config) -> None:
    references = _varied_reference_features(project_config)
    references.pop("reference-19.png")

    result = fit_position_scalers(references, config=project_config)

    assert result.failure_code is HOGScalerFailureCode.HOG_FIT_REFERENCE_COUNT_INVALID
    assert result.reference_count == 19
    assert result.scalers is None
    assert result.successful_position_count == 0


def test_fit_rejects_invalid_reference_path(project_config) -> None:
    references = _varied_reference_features(project_config)
    references[""] = references.pop("reference-19.png")

    result = fit_position_scalers(references, config=project_config)

    assert result.failure_code is HOGScalerFailureCode.HOG_FIT_REFERENCE_SET_INVALID
    assert result.reference_count == 20
    assert not result.reference_paths
    assert result.scalers is None


def test_fit_preserves_reference_feature_failure(project_config) -> None:
    references = _varied_reference_features(project_config)
    positions = fixed_patch_positions(project_config)
    assert positions is not None
    references["reference-07.png"] = PatchHOGFeatureResult(
        status="failed",
        failure_code=HOGFeatureFailureCode.HOG_DESCRIPTOR_INVALID,
        features=None,
        positions=positions,
        failed_patch_index=12,
    )

    result = fit_position_scalers(references, config=project_config)

    assert result.failure_code is HOGScalerFailureCode.HOG_FIT_REFERENCE_FEATURES_INVALID
    assert result.failed_reference_path == "reference-07.png"
    assert result.reference_failure_code is HOGFeatureFailureCode.HOG_DESCRIPTOR_INVALID
    assert result.successful_position_count == 0
    assert result.scalers is None


@pytest.mark.parametrize(
    "invalid_features",
    [
        np.zeros((224, 324), dtype=np.float32),
        np.zeros((225, 324), dtype=np.float64),
        np.full((225, 324), np.nan, dtype=np.float32),
    ],
)
def test_fit_rejects_invalid_reference_feature_matrix(
    project_config,
    invalid_features: np.ndarray,
) -> None:
    references = _varied_reference_features(project_config)
    references["reference-03.png"] = _feature_result(
        invalid_features,
        project_config=project_config,
    )

    result = fit_position_scalers(references, config=project_config)

    assert result.failure_code is HOGScalerFailureCode.HOG_FIT_REFERENCE_FEATURES_INVALID
    assert result.failed_reference_path == "reference-03.png"
    assert result.reference_failure_code is None
    assert result.scalers is None


def test_fit_rejects_mismatched_patch_positions(project_config) -> None:
    references = _varied_reference_features(project_config)
    original = references["reference-04.png"]
    references["reference-04.png"] = replace(
        original,
        positions=original.positions[:-1],
    )

    result = fit_position_scalers(references, config=project_config)

    assert result.failure_code is HOGScalerFailureCode.HOG_FIT_REFERENCE_FEATURES_INVALID
    assert result.failed_reference_path == "reference-04.png"
    assert result.scalers is None


def test_fit_discards_partial_scalers_after_library_failure(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    references = _varied_reference_features(project_config)
    original_fit = scaler_module.StandardScaler.fit
    call_count = 0

    def fail_fourth_position(self, samples, *args, **kwargs):
        nonlocal call_count
        current = call_count
        call_count += 1
        if current == 3:
            raise ValueError("synthetic scaler failure")
        return original_fit(self, samples, *args, **kwargs)

    monkeypatch.setattr(scaler_module.StandardScaler, "fit", fail_fourth_position)

    result = fit_position_scalers(references, config=project_config)

    assert result.failure_code is HOGScalerFailureCode.HOG_FIT_SCALER_FAILED
    assert result.failed_position_index == 3
    assert result.successful_position_count == 3
    assert result.scalers is None
    assert call_count == 4


def test_fit_rejects_invalid_scaler_state(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    references = _varied_reference_features(project_config)
    original_fit = scaler_module.StandardScaler.fit

    def corrupt_scale(self, samples, *args, **kwargs):
        fitted = original_fit(self, samples, *args, **kwargs)
        fitted.var_[0] = 0.0
        fitted.scale_[0] = 2.0
        return fitted

    monkeypatch.setattr(scaler_module.StandardScaler, "fit", corrupt_scale)

    result = fit_position_scalers(references, config=project_config)

    assert result.failure_code is HOGScalerFailureCode.HOG_FIT_SCALER_STATE_INVALID
    assert result.failed_position_index == 0
    assert result.successful_position_count == 0
    assert result.scalers is None
