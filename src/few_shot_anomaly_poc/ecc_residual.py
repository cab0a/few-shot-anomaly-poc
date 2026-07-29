"""Score one preprocessed image with the fixed v0.1 ECC residual method."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.ecc_template import ECCTemplateFitResult
from few_shot_anomaly_poc.errors import (
    ECCRegistrationFailureCode,
    ECCResidualFailureCode,
    ECCResidualStateError,
    ImagePreprocessingError,
    PreprocessingFailureCode,
)
from few_shot_anomaly_poc.preprocessing import validate_preprocessed_image
from few_shot_anomaly_poc.registration import ECCRegistrationResult, register_ecc

type ECCResidualScoreFailureCode = (
    PreprocessingFailureCode | ECCRegistrationFailureCode | ECCResidualFailureCode
)


@dataclass(frozen=True)
class ECCResidualScoreResult:
    """Image-level score and the preregistered diagnostic evidence."""

    score_status: Literal["ok", "failed"]
    failure_code: ECCResidualScoreFailureCode | None
    anomaly_score: float
    registration_status: Literal["not_run", "ok", "failed"]
    correlation: float | None
    warp_matrix: NDArray[np.float32] | None
    rotation_degrees: float | None
    translation_x_pixels: float | None
    translation_y_pixels: float | None
    registration_valid_fraction: float | None
    effective_support_fraction: float | None
    effective_pixel_count: int | None
    top_pixel_count: int | None

    @property
    def succeeded(self) -> bool:
        """Return whether a valid non-failure score was produced."""
        return self.score_status == "ok"


def _validate_fitted_state(
    fitted: ECCTemplateFitResult,
    *,
    config: ProjectConfig,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    if (
        not isinstance(fitted, ECCTemplateFitResult)
        or not fitted.succeeded
        or fitted.failure_code is not None
        or fitted.template is None
        or fitted.support_mask is None
        or fitted.reference_count != config.selection.reference_count
        or (fitted.successful_reference_count < config.ecc_template.minimum_successful_references)
    ):
        raise ECCResidualStateError(
            "a successful fitted ECC state is required before image scoring"
        )

    try:
        validate_preprocessed_image(
            fitted.template,
            label="fitted ECC template",
            config=config.preprocessing,
        )
    except ImagePreprocessingError as error:
        raise ECCResidualStateError("the fitted ECC template is invalid") from error

    support_mask = fitted.support_mask
    if (
        not isinstance(support_mask, np.ndarray)
        or support_mask.shape != fitted.template.shape
        or support_mask.dtype != np.bool_
    ):
        raise ECCResidualStateError("the fitted ECC support mask is invalid")

    support_fraction = float(np.count_nonzero(support_mask) / support_mask.size)
    if (
        fitted.support_fraction is None
        or not math.isfinite(fitted.support_fraction)
        or not math.isclose(
            fitted.support_fraction,
            support_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or support_fraction < config.ecc_template.minimum_support_fraction
    ):
        raise ECCResidualStateError("the fitted ECC support fraction is invalid")

    return fitted.template, support_mask


def _failed(
    code: ECCResidualScoreFailureCode,
    *,
    config: ProjectConfig,
    registration: ECCRegistrationResult | None = None,
    effective_support_fraction: float | None = None,
    effective_pixel_count: int | None = None,
) -> ECCResidualScoreResult:
    if registration is None:
        registration_status: Literal["not_run", "ok", "failed"] = "not_run"
        correlation = None
        warp_matrix = None
        rotation_degrees = None
        translation_x_pixels = None
        translation_y_pixels = None
        registration_valid_fraction = None
    else:
        registration_status = registration.status
        correlation = registration.correlation
        warp_matrix = registration.warp_matrix
        rotation_degrees = registration.rotation_degrees
        translation_x_pixels = registration.translation_x_pixels
        translation_y_pixels = registration.translation_y_pixels
        registration_valid_fraction = registration.valid_fraction

    return ECCResidualScoreResult(
        score_status="failed",
        failure_code=code,
        anomaly_score=config.ecc_residual_scoring.failure_score,
        registration_status=registration_status,
        correlation=correlation,
        warp_matrix=warp_matrix,
        rotation_degrees=rotation_degrees,
        translation_x_pixels=translation_x_pixels,
        translation_y_pixels=translation_y_pixels,
        registration_valid_fraction=registration_valid_fraction,
        effective_support_fraction=effective_support_fraction,
        effective_pixel_count=effective_pixel_count,
        top_pixel_count=None,
    )


def score_ecc_residual(
    image: NDArray[np.float32],
    *,
    fitted: ECCTemplateFitResult,
    config: ProjectConfig,
) -> ECCResidualScoreResult:
    """Return the fixed image score without selecting or applying a threshold."""
    template, template_support = _validate_fitted_state(fitted, config=config)

    try:
        registration = register_ecc(
            template,
            image,
            preprocessing=config.preprocessing,
            registration=config.ecc_registration,
        )
    except ImagePreprocessingError as error:
        return _failed(error.code, config=config)

    if not registration.succeeded:
        return _failed(
            registration.failure_code or ECCRegistrationFailureCode.ECC_RESULT_INVALID,
            config=config,
            registration=registration,
        )
    if (
        registration.failure_code is not None
        or registration.aligned_image is None
        or registration.valid_mask is None
    ):
        return _failed(
            ECCRegistrationFailureCode.ECC_RESULT_INVALID,
            config=config,
            registration=registration,
        )

    try:
        validate_preprocessed_image(
            registration.aligned_image,
            label="ECC-aligned scoring image",
            config=config.preprocessing,
        )
    except ImagePreprocessingError:
        return _failed(
            ECCRegistrationFailureCode.ECC_RESULT_INVALID,
            config=config,
            registration=registration,
        )
    if (
        not isinstance(registration.valid_mask, np.ndarray)
        or registration.valid_mask.shape != template.shape
        or registration.valid_mask.dtype != np.bool_
    ):
        return _failed(
            ECCRegistrationFailureCode.ECC_RESULT_INVALID,
            config=config,
            registration=registration,
        )

    scoring = config.ecc_residual_scoring
    kernel = np.ones(
        (scoring.validity_erosion_kernel_size,) * 2,
        dtype=np.uint8,
    )
    try:
        eroded_validity_raw = cv2.erode(
            registration.valid_mask.astype(np.uint8),
            kernel,
            iterations=scoring.validity_erosion_iterations,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    except cv2.error:
        return _failed(
            ECCResidualFailureCode.SCORE_MASK_EROSION_FAILED,
            config=config,
            registration=registration,
        )
    if (
        not isinstance(eroded_validity_raw, np.ndarray)
        or eroded_validity_raw.shape != template.shape
    ):
        return _failed(
            ECCResidualFailureCode.SCORE_MASK_EROSION_FAILED,
            config=config,
            registration=registration,
        )

    effective_mask = eroded_validity_raw.astype(bool) & template_support
    template_support_count = int(np.count_nonzero(template_support))
    effective_pixel_count = int(np.count_nonzero(effective_mask))
    effective_support_fraction = effective_pixel_count / template_support_count
    if effective_support_fraction < scoring.minimum_effective_support_fraction:
        return _failed(
            ECCResidualFailureCode.SCORE_EFFECTIVE_SUPPORT_TOO_SMALL,
            config=config,
            registration=registration,
            effective_support_fraction=effective_support_fraction,
            effective_pixel_count=effective_pixel_count,
        )

    residual = np.abs(registration.aligned_image - template)
    try:
        smoothed = cv2.GaussianBlur(
            residual,
            (scoring.gaussian_kernel_size,) * 2,
            sigmaX=scoring.gaussian_sigma,
            sigmaY=scoring.gaussian_sigma,
            borderType=cv2.BORDER_CONSTANT,
        )
    except cv2.error:
        return _failed(
            ECCResidualFailureCode.SCORE_RESIDUAL_FILTER_FAILED,
            config=config,
            registration=registration,
            effective_support_fraction=effective_support_fraction,
            effective_pixel_count=effective_pixel_count,
        )
    if (
        not isinstance(smoothed, np.ndarray)
        or smoothed.shape != template.shape
        or smoothed.dtype != np.float32
        or not np.isfinite(smoothed).all()
        or np.any(smoothed < 0.0)
        or np.any(smoothed > 1.0)
    ):
        return _failed(
            ECCResidualFailureCode.SCORE_RESIDUAL_FILTER_FAILED,
            config=config,
            registration=registration,
            effective_support_fraction=effective_support_fraction,
            effective_pixel_count=effective_pixel_count,
        )

    effective_values = smoothed[effective_mask]
    top_pixel_count = max(
        1,
        math.ceil(scoring.top_fraction * effective_pixel_count),
    )
    top_values = np.partition(
        effective_values,
        effective_pixel_count - top_pixel_count,
    )[-top_pixel_count:]
    anomaly_score = float(np.mean(top_values, dtype=np.float64))
    if not math.isfinite(anomaly_score) or anomaly_score < 0.0 or anomaly_score > 1.0:
        return _failed(
            ECCResidualFailureCode.SCORE_RESULT_INVALID,
            config=config,
            registration=registration,
            effective_support_fraction=effective_support_fraction,
            effective_pixel_count=effective_pixel_count,
        )

    return ECCResidualScoreResult(
        score_status="ok",
        failure_code=None,
        anomaly_score=anomaly_score,
        registration_status="ok",
        correlation=registration.correlation,
        warp_matrix=registration.warp_matrix,
        rotation_degrees=registration.rotation_degrees,
        translation_x_pixels=registration.translation_x_pixels,
        translation_y_pixels=registration.translation_y_pixels,
        registration_valid_fraction=registration.valid_fraction,
        effective_support_fraction=effective_support_fraction,
        effective_pixel_count=effective_pixel_count,
        top_pixel_count=top_pixel_count,
    )
