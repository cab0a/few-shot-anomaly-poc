"""Score one preprocessed image with the fixed v0.1 Patch HOG method."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.errors import (
    HOGScoringFailureCode,
    HOGScoringStateError,
)
from few_shot_anomaly_poc.hog_features import (
    PatchHOGFailureCode,
    PatchHOGFeatureResult,
    extract_patch_hog_features,
    patch_hog_feature_result_is_valid,
)
from few_shot_anomaly_poc.hog_models import (
    PatchHOGModelFitResult,
    position_one_class_svm_state_is_valid,
)
from few_shot_anomaly_poc.hog_scalers import (
    PatchHOGScalerFitResult,
    position_scaler_state_is_valid,
)

type PatchHOGScoreFailureCode = PatchHOGFailureCode | HOGScoringFailureCode


@dataclass(frozen=True)
class PatchHOGScoreResult:
    """Image-level score and fixed patch-level diagnostic evidence."""

    score_status: Literal["ok", "failed"]
    failure_code: PatchHOGScoreFailureCode | None
    anomaly_score: float
    patch_anomaly_scores: tuple[float, ...] | None
    top_patch_count: int | None
    top_patch_indices: tuple[int, ...]
    successful_patch_count: int
    failed_patch_index: int | None

    @property
    def succeeded(self) -> bool:
        """Return whether a valid non-failure score was produced."""
        return self.score_status == "ok"


def _reference_paths_are_valid(
    paths: tuple[str, ...],
    *,
    config: ProjectConfig,
) -> bool:
    return (
        len(paths) == config.selection.reference_count
        and all(isinstance(path, str) and path for path in paths)
        and len(set(paths)) == len(paths)
        and paths == tuple(sorted(paths))
    )


def _validate_fitted_state(
    scaler_fit: PatchHOGScalerFitResult,
    model_fit: PatchHOGModelFitResult,
    *,
    config: ProjectConfig,
) -> tuple[tuple[StandardScaler, ...], tuple[OneClassSVM, ...]]:
    patch_count = config.patch_hog.patch_count
    if (
        not isinstance(scaler_fit, PatchHOGScalerFitResult)
        or not scaler_fit.succeeded
        or scaler_fit.failure_code is not None
        or scaler_fit.scalers is None
        or scaler_fit.reference_count != config.selection.reference_count
        or not _reference_paths_are_valid(scaler_fit.reference_paths, config=config)
        or len(scaler_fit.scalers) != patch_count
        or scaler_fit.successful_position_count != patch_count
        or scaler_fit.failed_reference_path is not None
        or scaler_fit.reference_failure_code is not None
        or scaler_fit.failed_position_index is not None
    ):
        raise HOGScoringStateError(
            "a successful fitted Patch HOG scaler state is required before image scoring"
        )

    if (
        not isinstance(model_fit, PatchHOGModelFitResult)
        or not model_fit.succeeded
        or model_fit.failure_code is not None
        or model_fit.models is None
        or model_fit.reference_paths != scaler_fit.reference_paths
        or model_fit.reference_count != config.selection.reference_count
        or len(model_fit.models) != patch_count
        or model_fit.successful_position_count != patch_count
        or model_fit.failed_reference_path is not None
        or model_fit.reference_failure_code is not None
        or model_fit.scaler_failure_code is not None
        or model_fit.failed_position_index is not None
    ):
        raise HOGScoringStateError(
            "a successful fitted Patch HOG model state is required before image scoring"
        )

    for position_index, (scaler, model) in enumerate(
        zip(scaler_fit.scalers, model_fit.models, strict=True)
    ):
        if not position_scaler_state_is_valid(scaler, config=config):
            raise HOGScoringStateError(
                f"the fitted Patch HOG scaler is invalid at position {position_index}"
            )
        if not position_one_class_svm_state_is_valid(model, config=config):
            raise HOGScoringStateError(
                f"the fitted Patch HOG model is invalid at position {position_index}"
            )

    return scaler_fit.scalers, model_fit.models


def _failed(
    code: PatchHOGScoreFailureCode,
    *,
    config: ProjectConfig,
    successful_patch_count: int = 0,
    failed_patch_index: int | None = None,
) -> PatchHOGScoreResult:
    return PatchHOGScoreResult(
        score_status="failed",
        failure_code=code,
        anomaly_score=config.patch_hog_scoring.failure_score,
        patch_anomaly_scores=None,
        top_patch_count=None,
        top_patch_indices=(),
        successful_patch_count=successful_patch_count,
        failed_patch_index=failed_patch_index,
    )


def _mean_top_patch_scores(
    patch_scores: list[float],
    top_patch_indices: tuple[int, ...],
) -> float:
    return float(
        np.mean(
            np.asarray(
                [patch_scores[index] for index in top_patch_indices],
                dtype=np.float64,
            ),
            dtype=np.float64,
        )
    )


def score_patch_hog(
    image: NDArray[np.float32],
    *,
    scaler_fit: PatchHOGScalerFitResult,
    model_fit: PatchHOGModelFitResult,
    config: ProjectConfig,
) -> PatchHOGScoreResult:
    """Return the fixed image score without selecting or applying a threshold."""
    scalers, models = _validate_fitted_state(
        scaler_fit,
        model_fit,
        config=config,
    )

    feature_result = extract_patch_hog_features(image, config=config)
    if not isinstance(feature_result, PatchHOGFeatureResult):
        return _failed(
            HOGScoringFailureCode.HOG_SCORE_FEATURE_RESULT_INVALID,
            config=config,
        )
    if not feature_result.succeeded:
        return _failed(
            feature_result.failure_code or HOGScoringFailureCode.HOG_SCORE_FEATURE_RESULT_INVALID,
            config=config,
            failed_patch_index=feature_result.failed_patch_index,
        )
    if not patch_hog_feature_result_is_valid(feature_result, config=config):
        return _failed(
            HOGScoringFailureCode.HOG_SCORE_FEATURE_RESULT_INVALID,
            config=config,
            failed_patch_index=feature_result.failed_patch_index,
        )
    assert feature_result.features is not None

    descriptor_length = config.patch_hog.descriptor_length
    patch_scores = []
    for position_index, (descriptor, scaler, model) in enumerate(
        zip(feature_result.features, scalers, models, strict=True)
    ):
        samples = descriptor.reshape(1, descriptor_length)
        try:
            transformed_raw = scaler.transform(samples)
        except (FloatingPointError, RuntimeError, TypeError, ValueError):
            return _failed(
                HOGScoringFailureCode.HOG_SCORE_TRANSFORM_FAILED,
                config=config,
                successful_patch_count=len(patch_scores),
                failed_patch_index=position_index,
            )
        try:
            transformed = np.asarray(transformed_raw)
        except (TypeError, ValueError):
            return _failed(
                HOGScoringFailureCode.HOG_SCORE_TRANSFORM_INVALID,
                config=config,
                successful_patch_count=len(patch_scores),
                failed_patch_index=position_index,
            )
        if (
            transformed.shape != (1, descriptor_length)
            or transformed.dtype != np.float32
            or not np.isfinite(transformed).all()
        ):
            return _failed(
                HOGScoringFailureCode.HOG_SCORE_TRANSFORM_INVALID,
                config=config,
                successful_patch_count=len(patch_scores),
                failed_patch_index=position_index,
            )

        try:
            decision_raw = model.decision_function(transformed)
        except (FloatingPointError, RuntimeError, TypeError, ValueError):
            return _failed(
                HOGScoringFailureCode.HOG_SCORE_DECISION_FAILED,
                config=config,
                successful_patch_count=len(patch_scores),
                failed_patch_index=position_index,
            )
        try:
            decision = np.asarray(decision_raw)
        except (TypeError, ValueError):
            return _failed(
                HOGScoringFailureCode.HOG_SCORE_PATCH_INVALID,
                config=config,
                successful_patch_count=len(patch_scores),
                failed_patch_index=position_index,
            )
        if (
            decision.shape != (1,)
            or decision.dtype != np.float64
            or not np.isfinite(decision).all()
        ):
            return _failed(
                HOGScoringFailureCode.HOG_SCORE_PATCH_INVALID,
                config=config,
                successful_patch_count=len(patch_scores),
                failed_patch_index=position_index,
            )

        patch_score = -float(decision[0])
        if (
            not math.isfinite(patch_score)
            or abs(patch_score) >= config.patch_hog_scoring.maximum_absolute_patch_score_exclusive
        ):
            return _failed(
                HOGScoringFailureCode.HOG_SCORE_PATCH_INVALID,
                config=config,
                successful_patch_count=len(patch_scores),
                failed_patch_index=position_index,
            )
        patch_scores.append(patch_score)

    top_patch_count = max(
        1,
        math.ceil(config.patch_hog_scoring.top_fraction * len(patch_scores)),
    )
    top_patch_indices = tuple(
        sorted(
            range(len(patch_scores)),
            key=lambda index: (-patch_scores[index], index),
        )[:top_patch_count]
    )
    try:
        anomaly_score = _mean_top_patch_scores(patch_scores, top_patch_indices)
    except (FloatingPointError, RuntimeError, TypeError, ValueError):
        return _failed(
            HOGScoringFailureCode.HOG_SCORE_AGGREGATION_INVALID,
            config=config,
            successful_patch_count=len(patch_scores),
        )
    if (
        len(patch_scores) != config.patch_hog.patch_count
        or not math.isfinite(anomaly_score)
        or abs(anomaly_score) >= config.patch_hog_scoring.maximum_absolute_patch_score_exclusive
    ):
        return _failed(
            HOGScoringFailureCode.HOG_SCORE_AGGREGATION_INVALID,
            config=config,
            successful_patch_count=len(patch_scores),
        )

    return PatchHOGScoreResult(
        score_status="ok",
        failure_code=None,
        anomaly_score=anomaly_score,
        patch_anomaly_scores=tuple(patch_scores),
        top_patch_count=top_patch_count,
        top_patch_indices=top_patch_indices,
        successful_patch_count=len(patch_scores),
        failed_patch_index=None,
    )
