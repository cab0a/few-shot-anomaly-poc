"""Fit the fixed v0.1 position-wise Patch HOG One-Class SVM collection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.svm import OneClassSVM

from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.errors import (
    HOGModelFailureCode,
    HOGScalerFailureCode,
)
from few_shot_anomaly_poc.hog_features import (
    PatchHOGFailureCode,
    PatchHOGFeatureResult,
    patch_hog_feature_result_is_valid,
)
from few_shot_anomaly_poc.hog_scalers import (
    PatchHOGScalerFitResult,
    position_scaler_state_is_valid,
)


@dataclass(frozen=True)
class PatchHOGModelFitResult:
    """Method fitting state after all position-wise SVMs pass validation."""

    status: Literal["ok", "FIT_FAILED"]
    failure_code: HOGModelFailureCode | None
    reference_paths: tuple[str, ...]
    reference_count: int
    models: tuple[OneClassSVM, ...] | None
    successful_position_count: int
    failed_reference_path: str | None
    reference_failure_code: PatchHOGFailureCode | None
    scaler_failure_code: HOGScalerFailureCode | None
    failed_position_index: int | None

    @property
    def succeeded(self) -> bool:
        """Return whether all fixed position-wise SVMs are available."""
        return self.status == "ok"


def _fit_failed(
    code: HOGModelFailureCode,
    *,
    reference_paths: tuple[str, ...] = (),
    reference_count: int,
    successful_position_count: int = 0,
    failed_reference_path: str | None = None,
    reference_failure_code: PatchHOGFailureCode | None = None,
    scaler_failure_code: HOGScalerFailureCode | None = None,
    failed_position_index: int | None = None,
) -> PatchHOGModelFitResult:
    return PatchHOGModelFitResult(
        status="FIT_FAILED",
        failure_code=code,
        reference_paths=reference_paths,
        reference_count=reference_count,
        models=None,
        successful_position_count=successful_position_count,
        failed_reference_path=failed_reference_path,
        reference_failure_code=reference_failure_code,
        scaler_failure_code=scaler_failure_code,
        failed_position_index=failed_position_index,
    )


def position_one_class_svm_state_is_valid(
    model: OneClassSVM,
    *,
    config: ProjectConfig,
) -> bool:
    """Return whether one fitted model satisfies the fixed state contract."""
    if not isinstance(model, OneClassSVM):
        return False

    svm_config = config.patch_hog_one_class_svm
    descriptor_length = config.patch_hog.descriptor_length
    reference_count = config.selection.reference_count
    support = getattr(model, "support_", None)
    support_vectors = getattr(model, "support_vectors_", None)
    dual_coef = getattr(model, "dual_coef_", None)
    intercept = getattr(model, "intercept_", None)
    offset = getattr(model, "offset_", None)
    if (
        not isinstance(support, np.ndarray)
        or support.ndim != 1
        or not np.issubdtype(support.dtype, np.integer)
        or support.size < 1
        or support.size > reference_count
        or np.any(support < 0)
        or np.any(support >= reference_count)
        or np.unique(support).size != support.size
        or not isinstance(support_vectors, np.ndarray)
        or support_vectors.shape != (support.size, descriptor_length)
        or support_vectors.dtype != np.float64
        or not np.isfinite(support_vectors).all()
        or not isinstance(dual_coef, np.ndarray)
        or dual_coef.shape != (1, support.size)
        or dual_coef.dtype != np.float64
        or not np.isfinite(dual_coef).all()
        or not isinstance(intercept, np.ndarray)
        or intercept.shape != (1,)
        or intercept.dtype != np.float64
        or not np.isfinite(intercept).all()
        or not isinstance(offset, np.ndarray)
        or offset.shape != (1,)
        or offset.dtype != np.float64
        or not np.isfinite(offset).all()
    ):
        return False

    try:
        fitted_gamma = float(getattr(model, "_gamma", math.nan))
    except (TypeError, ValueError):
        return False
    return (
        getattr(model, "fit_status_", None) == 0
        and getattr(model, "n_features_in_", None) == descriptor_length
        and getattr(model, "shape_fit_", None) == (reference_count, descriptor_length)
        and isinstance(getattr(model, "n_iter_", None), int)
        and model.n_iter_ >= 0
        and math.isfinite(fitted_gamma)
        and fitted_gamma > 0.0
        and model.kernel == svm_config.kernel
        and model.gamma == svm_config.gamma
        and model.nu == svm_config.nu
        and model.tol == svm_config.tolerance
        and model.shrinking is svm_config.shrinking
        and model.cache_size == svm_config.cache_size_mb
        and model.max_iter == svm_config.max_iterations
        and model.verbose is svm_config.verbose
    )


def _invalid_scaler_position(
    scaler_fit: object,
    *,
    reference_paths: tuple[str, ...],
    config: ProjectConfig,
) -> int | None:
    if (
        not isinstance(scaler_fit, PatchHOGScalerFitResult)
        or not scaler_fit.succeeded
        or scaler_fit.failure_code is not None
        or scaler_fit.reference_paths != reference_paths
        or scaler_fit.reference_count != config.selection.reference_count
        or scaler_fit.scalers is None
        or len(scaler_fit.scalers) != config.patch_hog.patch_count
        or scaler_fit.successful_position_count != config.patch_hog.patch_count
        or scaler_fit.failed_reference_path is not None
        or scaler_fit.reference_failure_code is not None
        or scaler_fit.failed_position_index is not None
    ):
        return -1

    for position_index, scaler in enumerate(scaler_fit.scalers):
        if not position_scaler_state_is_valid(scaler, config=config):
            return position_index
    return None


def fit_position_one_class_svms(
    reference_features: Mapping[str, PatchHOGFeatureResult],
    *,
    scaler_fit: PatchHOGScalerFitResult,
    config: ProjectConfig,
) -> PatchHOGModelFitResult:
    """Fit one fixed One-Class SVM per patch position from normal references."""
    reference_count = len(reference_features)
    if reference_count != config.selection.reference_count:
        return _fit_failed(
            HOGModelFailureCode.HOG_MODEL_REFERENCE_COUNT_INVALID,
            reference_count=reference_count,
        )
    if any(not isinstance(path, str) or not path for path in reference_features):
        return _fit_failed(
            HOGModelFailureCode.HOG_MODEL_REFERENCE_SET_INVALID,
            reference_count=reference_count,
        )

    reference_paths = tuple(sorted(reference_features))
    feature_matrices = []
    for relative_path in reference_paths:
        result = reference_features[relative_path]
        if not patch_hog_feature_result_is_valid(result, config=config):
            source_code = result.failure_code if isinstance(result, PatchHOGFeatureResult) else None
            return _fit_failed(
                HOGModelFailureCode.HOG_MODEL_REFERENCE_FEATURES_INVALID,
                reference_paths=reference_paths,
                reference_count=reference_count,
                failed_reference_path=relative_path,
                reference_failure_code=source_code,
            )
        assert result.features is not None
        feature_matrices.append(result.features)

    invalid_scaler_position = _invalid_scaler_position(
        scaler_fit,
        reference_paths=reference_paths,
        config=config,
    )
    if invalid_scaler_position is not None:
        source_code = (
            scaler_fit.failure_code if isinstance(scaler_fit, PatchHOGScalerFitResult) else None
        )
        return _fit_failed(
            HOGModelFailureCode.HOG_MODEL_SCALER_STATE_INVALID,
            reference_paths=reference_paths,
            reference_count=reference_count,
            scaler_failure_code=source_code,
            failed_position_index=(
                invalid_scaler_position if invalid_scaler_position >= 0 else None
            ),
        )
    assert scaler_fit.scalers is not None

    feature_tensor = np.stack(feature_matrices, axis=0)
    models = []
    svm_config = config.patch_hog_one_class_svm
    for position_index, scaler in enumerate(scaler_fit.scalers):
        samples = feature_tensor[:, position_index, :]
        try:
            transformed_raw = scaler.transform(samples)
        except (FloatingPointError, RuntimeError, TypeError, ValueError):
            return _fit_failed(
                HOGModelFailureCode.HOG_MODEL_TRANSFORM_FAILED,
                reference_paths=reference_paths,
                reference_count=reference_count,
                successful_position_count=len(models),
                failed_position_index=position_index,
            )
        try:
            transformed = np.asarray(transformed_raw)
        except (TypeError, ValueError):
            return _fit_failed(
                HOGModelFailureCode.HOG_MODEL_TRANSFORM_INVALID,
                reference_paths=reference_paths,
                reference_count=reference_count,
                successful_position_count=len(models),
                failed_position_index=position_index,
            )
        if (
            transformed.shape
            != (
                config.selection.reference_count,
                config.patch_hog.descriptor_length,
            )
            or transformed.dtype != np.float32
            or not np.isfinite(transformed).all()
        ):
            return _fit_failed(
                HOGModelFailureCode.HOG_MODEL_TRANSFORM_INVALID,
                reference_paths=reference_paths,
                reference_count=reference_count,
                successful_position_count=len(models),
                failed_position_index=position_index,
            )

        model = OneClassSVM(
            kernel=svm_config.kernel,
            gamma=svm_config.gamma,
            nu=svm_config.nu,
            tol=svm_config.tolerance,
            shrinking=svm_config.shrinking,
            cache_size=svm_config.cache_size_mb,
            max_iter=svm_config.max_iterations,
            verbose=svm_config.verbose,
        )
        try:
            model.fit(transformed)
        except (FloatingPointError, RuntimeError, TypeError, ValueError):
            return _fit_failed(
                HOGModelFailureCode.HOG_MODEL_FIT_FAILED,
                reference_paths=reference_paths,
                reference_count=reference_count,
                successful_position_count=len(models),
                failed_position_index=position_index,
            )
        if not position_one_class_svm_state_is_valid(model, config=config):
            return _fit_failed(
                HOGModelFailureCode.HOG_MODEL_STATE_INVALID,
                reference_paths=reference_paths,
                reference_count=reference_count,
                successful_position_count=len(models),
                failed_position_index=position_index,
            )
        models.append(model)

    return PatchHOGModelFitResult(
        status="ok",
        failure_code=None,
        reference_paths=reference_paths,
        reference_count=reference_count,
        models=tuple(models),
        successful_position_count=len(models),
        failed_reference_path=None,
        reference_failure_code=None,
        scaler_failure_code=None,
        failed_position_index=None,
    )
