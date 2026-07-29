"""Deterministic shared input preprocessing for both v0.1 methods."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from few_shot_anomaly_poc.config import PreprocessingConfig
from few_shot_anomaly_poc.errors import (
    ImagePreprocessingError,
    PreprocessingFailureCode,
)

DECODE_FLAGS = cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION


def validate_preprocessed_image(
    image: object,
    *,
    label: str,
    config: PreprocessingConfig,
) -> None:
    """Require the fixed shape, dtype, finite state, and numeric range."""
    expected_shape = (config.output_height, config.output_width)
    if (
        not isinstance(image, np.ndarray)
        or image.shape != expected_shape
        or image.dtype != np.float32
        or not np.isfinite(image).all()
        or np.any(image < 0.0)
        or np.any(image > 1.0)
    ):
        raise ImagePreprocessingError(
            PreprocessingFailureCode.INVALID_PREPROCESSED_IMAGE,
            f"{label} must be a finite float32 array in [0, 1] with shape {expected_shape}",
        )


def preprocess_decoded_image(
    image: NDArray[np.generic] | None,
    config: PreprocessingConfig,
) -> NDArray[np.float32]:
    """Validate, resize, and scale one already decoded grayscale image."""
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 2
        or image.size == 0
        or image.dtype != np.uint8
    ):
        raise ImagePreprocessingError(
            PreprocessingFailureCode.INVALID_DECODED_IMAGE,
            "decoded image must be a non-empty two-dimensional uint8 array",
        )

    try:
        resized = cv2.resize(
            image,
            (config.output_width, config.output_height),
            interpolation=cv2.INTER_AREA,
        )
    except cv2.error as error:
        raise ImagePreprocessingError(
            PreprocessingFailureCode.IMAGE_RESIZE_FAILED,
            f"OpenCV could not resize the decoded image: {error}",
        ) from error

    output = resized.astype(np.float32) / np.float32(config.scale_divisor)
    validate_preprocessed_image(output, label="preprocessed image", config=config)
    return output


def load_and_preprocess_image(
    path: Path,
    config: PreprocessingConfig,
) -> NDArray[np.float32]:
    """Read opaque file bytes, decode deterministically, and preprocess."""
    try:
        encoded = path.read_bytes()
    except FileNotFoundError as error:
        raise ImagePreprocessingError(
            PreprocessingFailureCode.IMAGE_NOT_FOUND,
            f"image file does not exist: {path}",
        ) from error
    except OSError as error:
        raise ImagePreprocessingError(
            PreprocessingFailureCode.IMAGE_READ_FAILED,
            f"image file cannot be read: {path}",
        ) from error

    if not encoded:
        raise ImagePreprocessingError(
            PreprocessingFailureCode.IMAGE_DECODE_FAILED,
            f"image file is empty: {path}",
        )

    try:
        decoded = cv2.imdecode(
            np.frombuffer(encoded, dtype=np.uint8),
            DECODE_FLAGS,
        )
    except cv2.error as error:
        raise ImagePreprocessingError(
            PreprocessingFailureCode.IMAGE_DECODE_FAILED,
            f"OpenCV could not decode image file: {path}",
        ) from error
    if decoded is None:
        raise ImagePreprocessingError(
            PreprocessingFailureCode.IMAGE_DECODE_FAILED,
            f"OpenCV could not decode image file: {path}",
        )
    return preprocess_decoded_image(decoded, config)
