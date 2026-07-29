"""Extract the fixed v0.1 row-major patch HOG feature matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from skimage.feature import hog

from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.errors import (
    HOGFeatureFailureCode,
    ImagePreprocessingError,
    PreprocessingFailureCode,
)
from few_shot_anomaly_poc.preprocessing import validate_preprocessed_image

type PatchHOGFailureCode = PreprocessingFailureCode | HOGFeatureFailureCode


@dataclass(frozen=True)
class PatchPosition:
    """One fixed row-major patch location in the preprocessed image."""

    index: int
    row_index: int
    column_index: int
    top: int
    left: int


@dataclass(frozen=True)
class PatchHOGFeatureResult:
    """Feature extraction outcome without scaling or model inference."""

    status: Literal["ok", "failed"]
    failure_code: PatchHOGFailureCode | None
    features: NDArray[np.float32] | None
    positions: tuple[PatchPosition, ...]
    failed_patch_index: int | None

    @property
    def succeeded(self) -> bool:
        """Return whether the complete fixed feature matrix is available."""
        return self.status == "ok"


def _failed(
    code: PatchHOGFailureCode,
    *,
    positions: tuple[PatchPosition, ...] = (),
    failed_patch_index: int | None = None,
) -> PatchHOGFeatureResult:
    return PatchHOGFeatureResult(
        status="failed",
        failure_code=code,
        features=None,
        positions=positions,
        failed_patch_index=failed_patch_index,
    )


def _fixed_positions(config: ProjectConfig) -> tuple[PatchPosition, ...] | None:
    preprocessing = config.preprocessing
    patch_hog = config.patch_hog
    vertical_starts = tuple(
        range(
            0,
            preprocessing.output_height - patch_hog.patch_height + 1,
            patch_hog.vertical_stride,
        )
    )
    horizontal_starts = tuple(
        range(
            0,
            preprocessing.output_width - patch_hog.patch_width + 1,
            patch_hog.horizontal_stride,
        )
    )
    if (
        len(vertical_starts) != patch_hog.vertical_positions
        or len(horizontal_starts) != patch_hog.horizontal_positions
        or not vertical_starts
        or not horizontal_starts
        or vertical_starts[-1] + patch_hog.patch_height != preprocessing.output_height
        or horizontal_starts[-1] + patch_hog.patch_width != preprocessing.output_width
    ):
        return None

    positions = tuple(
        PatchPosition(
            index=row_index * patch_hog.horizontal_positions + column_index,
            row_index=row_index,
            column_index=column_index,
            top=top,
            left=left,
        )
        for row_index, top in enumerate(vertical_starts)
        for column_index, left in enumerate(horizontal_starts)
    )
    if len(positions) != patch_hog.patch_count:
        return None
    return positions


def extract_patch_hog_features(
    image: NDArray[np.float32],
    *,
    config: ProjectConfig,
) -> PatchHOGFeatureResult:
    """Extract all 225 descriptors without fitting or applying a scaler."""
    try:
        validate_preprocessed_image(
            image,
            label="Patch HOG input image",
            config=config.preprocessing,
        )
    except ImagePreprocessingError as error:
        return _failed(error.code)

    positions = _fixed_positions(config)
    if positions is None:
        return _failed(HOGFeatureFailureCode.HOG_GRID_INVALID)

    patch_hog = config.patch_hog
    features = np.empty(
        (patch_hog.patch_count, patch_hog.descriptor_length),
        dtype=np.float32,
    )
    for position in positions:
        patch = image[
            position.top : position.top + patch_hog.patch_height,
            position.left : position.left + patch_hog.patch_width,
        ]
        if patch.shape != (patch_hog.patch_height, patch_hog.patch_width):
            return _failed(
                HOGFeatureFailureCode.HOG_GRID_INVALID,
                positions=positions,
                failed_patch_index=position.index,
            )

        try:
            descriptor_raw = hog(
                patch,
                orientations=patch_hog.orientations,
                pixels_per_cell=(
                    patch_hog.pixels_per_cell_height,
                    patch_hog.pixels_per_cell_width,
                ),
                cells_per_block=(
                    patch_hog.cells_per_block_height,
                    patch_hog.cells_per_block_width,
                ),
                block_norm=patch_hog.block_norm,
                visualize=patch_hog.visualize,
                transform_sqrt=patch_hog.transform_sqrt,
                feature_vector=patch_hog.feature_vector,
                channel_axis=None if patch_hog.channel_axis == "none" else 0,
            )
        except (FloatingPointError, RuntimeError, TypeError, ValueError):
            return _failed(
                HOGFeatureFailureCode.HOG_EXTRACTION_FAILED,
                positions=positions,
                failed_patch_index=position.index,
            )

        try:
            descriptor = np.asarray(descriptor_raw)
        except (TypeError, ValueError):
            return _failed(
                HOGFeatureFailureCode.HOG_DESCRIPTOR_INVALID,
                positions=positions,
                failed_patch_index=position.index,
            )
        if (
            descriptor.shape != (patch_hog.descriptor_length,)
            or descriptor.dtype != np.float32
            or not np.isfinite(descriptor).all()
        ):
            return _failed(
                HOGFeatureFailureCode.HOG_DESCRIPTOR_INVALID,
                positions=positions,
                failed_patch_index=position.index,
            )
        features[position.index] = descriptor

    return PatchHOGFeatureResult(
        status="ok",
        failure_code=None,
        features=features,
        positions=positions,
        failed_patch_index=None,
    )
