"""Fixed Euclidean ECC registration for the v0.1 residual method."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from few_shot_anomaly_poc.config import ECCRegistrationConfig, PreprocessingConfig
from few_shot_anomaly_poc.errors import ECCRegistrationFailureCode
from few_shot_anomaly_poc.preprocessing import validate_preprocessed_image


@dataclass(frozen=True)
class ECCRegistrationResult:
    """Registration status and bounded diagnostics for one image pair."""

    status: Literal["ok", "failed"]
    failure_code: ECCRegistrationFailureCode | None
    correlation: float | None
    warp_matrix: NDArray[np.float32] | None
    rotation_degrees: float | None
    translation_x_pixels: float | None
    translation_y_pixels: float | None
    valid_fraction: float | None
    aligned_image: NDArray[np.float32] | None
    valid_mask: NDArray[np.bool_] | None

    @property
    def succeeded(self) -> bool:
        """Return whether every preregistered registration gate passed."""
        return self.status == "ok"


def _failed(
    code: ECCRegistrationFailureCode,
    *,
    correlation: float | None = None,
    warp_matrix: NDArray[np.float32] | None = None,
    rotation_degrees: float | None = None,
    translation_x_pixels: float | None = None,
    translation_y_pixels: float | None = None,
    valid_fraction: float | None = None,
) -> ECCRegistrationResult:
    return ECCRegistrationResult(
        status="failed",
        failure_code=code,
        correlation=correlation,
        warp_matrix=warp_matrix,
        rotation_degrees=rotation_degrees,
        translation_x_pixels=translation_x_pixels,
        translation_y_pixels=translation_y_pixels,
        valid_fraction=valid_fraction,
        aligned_image=None,
        valid_mask=None,
    )


def register_ecc(
    template: NDArray[np.float32],
    moving: NDArray[np.float32],
    *,
    preprocessing: PreprocessingConfig,
    registration: ECCRegistrationConfig,
) -> ECCRegistrationResult:
    """Register one preprocessed moving image to a preprocessed template."""
    validate_preprocessed_image(template, label="template", config=preprocessing)
    validate_preprocessed_image(moving, label="moving image", config=preprocessing)

    initial_warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
        registration.max_iterations,
        registration.epsilon,
    )
    try:
        correlation_raw, fitted_warp_raw = cv2.findTransformECC(
            template,
            moving,
            initial_warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            registration.gaussian_filter_size,
        )
    except cv2.error:
        return _failed(ECCRegistrationFailureCode.ECC_OPTIMIZATION_FAILED)

    fitted_warp = np.asarray(fitted_warp_raw)
    if fitted_warp.shape != (2, 3) or fitted_warp.dtype not in {
        np.dtype(np.float32),
        np.dtype(np.float64),
    }:
        return _failed(ECCRegistrationFailureCode.ECC_RESULT_INVALID)

    correlation = float(correlation_raw)
    fitted_warp = fitted_warp.astype(np.float32, copy=True)
    if not math.isfinite(correlation) or not np.isfinite(fitted_warp).all():
        return _failed(ECCRegistrationFailureCode.ECC_RESULT_NONFINITE)

    rotation_degrees = math.degrees(math.atan2(float(fitted_warp[1, 0]), float(fitted_warp[0, 0])))
    translation_x = float(fitted_warp[0, 2])
    translation_y = float(fitted_warp[1, 2])
    diagnostics = {
        "correlation": correlation,
        "warp_matrix": fitted_warp,
        "rotation_degrees": rotation_degrees,
        "translation_x_pixels": translation_x,
        "translation_y_pixels": translation_y,
    }

    if abs(rotation_degrees) > registration.max_abs_rotation_degrees:
        return _failed(
            ECCRegistrationFailureCode.ECC_ROTATION_LIMIT_EXCEEDED,
            **diagnostics,
        )
    if (
        abs(translation_x) > registration.max_abs_horizontal_translation_pixels
        or abs(translation_y) > registration.max_abs_vertical_translation_pixels
    ):
        return _failed(
            ECCRegistrationFailureCode.ECC_TRANSLATION_LIMIT_EXCEEDED,
            **diagnostics,
        )

    output_size = (preprocessing.output_width, preprocessing.output_height)
    try:
        aligned = cv2.warpAffine(
            moving,
            fitted_warp,
            output_size,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        warped_mask = cv2.warpAffine(
            np.ones(moving.shape, dtype=np.uint8),
            fitted_warp,
            output_size,
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    except cv2.error:
        return _failed(ECCRegistrationFailureCode.ECC_WARP_FAILED, **diagnostics)

    if (
        aligned.shape != template.shape
        or aligned.dtype != np.float32
        or not np.isfinite(aligned).all()
        or warped_mask.shape != template.shape
    ):
        return _failed(ECCRegistrationFailureCode.ECC_WARP_FAILED, **diagnostics)

    valid_mask = warped_mask.astype(bool)
    valid_fraction = float(np.count_nonzero(valid_mask) / valid_mask.size)
    if valid_fraction < registration.min_valid_fraction:
        return _failed(
            ECCRegistrationFailureCode.ECC_VALID_AREA_TOO_SMALL,
            valid_fraction=valid_fraction,
            **diagnostics,
        )

    return ECCRegistrationResult(
        status="ok",
        failure_code=None,
        valid_fraction=valid_fraction,
        aligned_image=aligned,
        valid_mask=valid_mask,
        **diagnostics,
    )
