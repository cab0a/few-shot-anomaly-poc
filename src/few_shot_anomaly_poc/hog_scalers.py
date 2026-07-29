"""Fit the fixed v0.1 position-wise StandardScaler collection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.preprocessing import StandardScaler

from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.errors import HOGScalerFailureCode
from few_shot_anomaly_poc.hog_features import (
    PatchHOGFailureCode,
    PatchHOGFeatureResult,
    patch_hog_feature_result_is_valid,
)


@dataclass(frozen=True)
class PatchHOGScalerFitResult:
    """Method fitting state after all position-wise scalers pass validation."""

    status: Literal["ok", "FIT_FAILED"]
    failure_code: HOGScalerFailureCode | None
    reference_paths: tuple[str, ...]
    reference_count: int
    scalers: tuple[StandardScaler, ...] | None
    successful_position_count: int
    failed_reference_path: str | None
    reference_failure_code: PatchHOGFailureCode | None
    failed_position_index: int | None

    @property
    def succeeded(self) -> bool:
        """Return whether all fixed position-wise scalers are available."""
        return self.status == "ok"


def _fit_failed(
    code: HOGScalerFailureCode,
    *,
    reference_paths: tuple[str, ...] = (),
    reference_count: int,
    successful_position_count: int = 0,
    failed_reference_path: str | None = None,
    reference_failure_code: PatchHOGFailureCode | None = None,
    failed_position_index: int | None = None,
) -> PatchHOGScalerFitResult:
    return PatchHOGScalerFitResult(
        status="FIT_FAILED",
        failure_code=code,
        reference_paths=reference_paths,
        reference_count=reference_count,
        scalers=None,
        successful_position_count=successful_position_count,
        failed_reference_path=failed_reference_path,
        reference_failure_code=reference_failure_code,
        failed_position_index=failed_position_index,
    )


def position_scaler_state_is_valid(
    scaler: StandardScaler,
    *,
    config: ProjectConfig,
) -> bool:
    if not isinstance(scaler, StandardScaler):
        return False

    descriptor_length = config.patch_hog.descriptor_length
    expected_shape = (descriptor_length,)
    arrays = (
        getattr(scaler, "mean_", None),
        getattr(scaler, "var_", None),
        getattr(scaler, "scale_", None),
    )
    if any(
        not isinstance(array, np.ndarray)
        or array.shape != expected_shape
        or array.dtype != np.float64
        or not np.isfinite(array).all()
        for array in arrays
    ):
        return False

    _, variance, scale = arrays
    if (
        np.any(variance < 0.0)
        or np.any(scale <= 0.0)
        or np.any(scale[variance == 0.0] != 1.0)
        or getattr(scaler, "n_features_in_", None) != descriptor_length
        or scaler.copy is not config.patch_hog_scaler.copy
        or scaler.with_mean is not config.patch_hog_scaler.with_mean
        or scaler.with_std is not config.patch_hog_scaler.with_std
    ):
        return False

    sample_count = np.asarray(getattr(scaler, "n_samples_seen_", None))
    return (
        sample_count.size == 1
        and np.issubdtype(sample_count.dtype, np.number)
        and np.isfinite(sample_count).all()
        and float(sample_count.reshape(-1)[0]) == config.selection.reference_count
    )


def fit_position_scalers(
    reference_features: Mapping[str, PatchHOGFeatureResult],
    *,
    config: ProjectConfig,
) -> PatchHOGScalerFitResult:
    """Fit one scaler per patch position from exactly 20 reference images."""
    reference_count = len(reference_features)
    if reference_count != config.selection.reference_count:
        return _fit_failed(
            HOGScalerFailureCode.HOG_FIT_REFERENCE_COUNT_INVALID,
            reference_count=reference_count,
        )
    if any(not isinstance(path, str) or not path for path in reference_features):
        return _fit_failed(
            HOGScalerFailureCode.HOG_FIT_REFERENCE_SET_INVALID,
            reference_count=reference_count,
        )

    reference_paths = tuple(sorted(reference_features))
    feature_matrices = []
    for relative_path in reference_paths:
        result = reference_features[relative_path]
        if not patch_hog_feature_result_is_valid(result, config=config):
            source_code = result.failure_code if isinstance(result, PatchHOGFeatureResult) else None
            return _fit_failed(
                HOGScalerFailureCode.HOG_FIT_REFERENCE_FEATURES_INVALID,
                reference_paths=reference_paths,
                reference_count=reference_count,
                failed_reference_path=relative_path,
                reference_failure_code=source_code,
            )
        assert result.features is not None
        feature_matrices.append(result.features)

    feature_tensor = np.stack(feature_matrices, axis=0)
    expected_tensor_shape = (
        config.selection.reference_count,
        config.patch_hog.patch_count,
        config.patch_hog.descriptor_length,
    )
    if (
        feature_tensor.shape != expected_tensor_shape
        or feature_tensor.dtype != np.float32
        or not np.isfinite(feature_tensor).all()
    ):
        return _fit_failed(
            HOGScalerFailureCode.HOG_FIT_REFERENCE_FEATURES_INVALID,
            reference_paths=reference_paths,
            reference_count=reference_count,
        )

    scalers = []
    for position_index in range(config.patch_hog.patch_count):
        samples = feature_tensor[:, position_index, :]
        scaler = StandardScaler(
            copy=config.patch_hog_scaler.copy,
            with_mean=config.patch_hog_scaler.with_mean,
            with_std=config.patch_hog_scaler.with_std,
        )
        try:
            scaler.fit(samples)
        except (FloatingPointError, RuntimeError, TypeError, ValueError):
            return _fit_failed(
                HOGScalerFailureCode.HOG_FIT_SCALER_FAILED,
                reference_paths=reference_paths,
                reference_count=reference_count,
                successful_position_count=len(scalers),
                failed_position_index=position_index,
            )
        if not position_scaler_state_is_valid(scaler, config=config):
            return _fit_failed(
                HOGScalerFailureCode.HOG_FIT_SCALER_STATE_INVALID,
                reference_paths=reference_paths,
                reference_count=reference_count,
                successful_position_count=len(scalers),
                failed_position_index=position_index,
            )
        scalers.append(scaler)

    return PatchHOGScalerFitResult(
        status="ok",
        failure_code=None,
        reference_paths=reference_paths,
        reference_count=reference_count,
        scalers=tuple(scalers),
        successful_position_count=len(scalers),
        failed_reference_path=None,
        reference_failure_code=None,
        failed_position_index=None,
    )
