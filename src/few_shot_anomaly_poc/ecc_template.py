"""Fit the fixed v0.1 ECC normal template from reference images."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.errors import (
    ECCRegistrationFailureCode,
    ECCTemplateFailureCode,
    ImagePreprocessingError,
    PreprocessingFailureCode,
)
from few_shot_anomaly_poc.preprocessing import validate_preprocessed_image
from few_shot_anomaly_poc.registration import ECCRegistrationResult, register_ecc

type ReferenceFailureCode = PreprocessingFailureCode | ECCRegistrationFailureCode


@dataclass(frozen=True)
class ReferenceRegistrationDiagnostic:
    """Bounded fitting evidence for one sorted reference path."""

    relative_path: str
    is_anchor: bool
    status: Literal["ok", "failed"]
    failure_code: ReferenceFailureCode | None
    correlation: float | None
    warp_matrix: NDArray[np.float32] | None
    rotation_degrees: float | None
    translation_x_pixels: float | None
    translation_y_pixels: float | None
    valid_fraction: float | None


@dataclass(frozen=True)
class ECCTemplateFitResult:
    """Template fitting outcome and evidence required before image scoring."""

    status: Literal["ok", "FIT_FAILED"]
    failure_code: ECCTemplateFailureCode | None
    anchor_path: str | None
    reference_count: int
    successful_reference_count: int
    failed_reference_count: int
    support_fraction: float | None
    template: NDArray[np.float32] | None
    support_mask: NDArray[np.bool_] | None
    reference_diagnostics: tuple[ReferenceRegistrationDiagnostic, ...]

    @property
    def succeeded(self) -> bool:
        """Return whether the fitted state is available for scoring."""
        return self.status == "ok"


def _fit_failed(
    code: ECCTemplateFailureCode,
    *,
    anchor_path: str | None,
    reference_count: int,
    successful_reference_count: int,
    support_fraction: float | None = None,
    support_mask: NDArray[np.bool_] | None = None,
    diagnostics: tuple[ReferenceRegistrationDiagnostic, ...] = (),
) -> ECCTemplateFitResult:
    return ECCTemplateFitResult(
        status="FIT_FAILED",
        failure_code=code,
        anchor_path=anchor_path,
        reference_count=reference_count,
        successful_reference_count=successful_reference_count,
        failed_reference_count=sum(item.status == "failed" for item in diagnostics),
        support_fraction=support_fraction,
        template=None,
        support_mask=support_mask,
        reference_diagnostics=diagnostics,
    )


def _anchor_diagnostic(relative_path: str) -> ReferenceRegistrationDiagnostic:
    return ReferenceRegistrationDiagnostic(
        relative_path=relative_path,
        is_anchor=True,
        status="ok",
        failure_code=None,
        correlation=None,
        warp_matrix=np.eye(2, 3, dtype=np.float32),
        rotation_degrees=0.0,
        translation_x_pixels=0.0,
        translation_y_pixels=0.0,
        valid_fraction=1.0,
    )


def _preprocessing_failure_diagnostic(
    relative_path: str,
    *,
    is_anchor: bool,
    code: PreprocessingFailureCode,
) -> ReferenceRegistrationDiagnostic:
    return ReferenceRegistrationDiagnostic(
        relative_path=relative_path,
        is_anchor=is_anchor,
        status="failed",
        failure_code=code,
        correlation=None,
        warp_matrix=None,
        rotation_degrees=None,
        translation_x_pixels=None,
        translation_y_pixels=None,
        valid_fraction=None,
    )


def _registration_diagnostic(
    relative_path: str,
    result: ECCRegistrationResult,
) -> ReferenceRegistrationDiagnostic:
    return ReferenceRegistrationDiagnostic(
        relative_path=relative_path,
        is_anchor=False,
        status=result.status,
        failure_code=result.failure_code,
        correlation=result.correlation,
        warp_matrix=result.warp_matrix,
        rotation_degrees=result.rotation_degrees,
        translation_x_pixels=result.translation_x_pixels,
        translation_y_pixels=result.translation_y_pixels,
        valid_fraction=result.valid_fraction,
    )


def fit_ecc_normal_template(
    references: Mapping[str, NDArray[np.float32]],
    *,
    config: ProjectConfig,
) -> ECCTemplateFitResult:
    """Fit the preregistered template without using calibration images."""
    reference_count = len(references)
    if reference_count != config.selection.reference_count:
        return _fit_failed(
            ECCTemplateFailureCode.FIT_REFERENCE_COUNT_INVALID,
            anchor_path=None,
            reference_count=reference_count,
            successful_reference_count=0,
        )
    if any(not isinstance(path, str) or not path for path in references):
        return _fit_failed(
            ECCTemplateFailureCode.FIT_REFERENCE_SET_INVALID,
            anchor_path=None,
            reference_count=reference_count,
            successful_reference_count=0,
        )

    sorted_paths = sorted(references)
    anchor_path = sorted_paths[0]
    anchor = references[anchor_path]
    try:
        validate_preprocessed_image(
            anchor,
            label=f"anchor {anchor_path}",
            config=config.preprocessing,
        )
    except ImagePreprocessingError as error:
        diagnostic = _preprocessing_failure_diagnostic(
            anchor_path,
            is_anchor=True,
            code=error.code,
        )
        return _fit_failed(
            ECCTemplateFailureCode.FIT_ANCHOR_PREPROCESSING_FAILED,
            anchor_path=anchor_path,
            reference_count=reference_count,
            successful_reference_count=0,
            diagnostics=(diagnostic,),
        )

    diagnostics = [_anchor_diagnostic(anchor_path)]
    aligned_images = [anchor.copy()]
    validity_masks = [np.ones(anchor.shape, dtype=bool)]

    for relative_path in sorted_paths[1:]:
        moving = references[relative_path]
        try:
            result = register_ecc(
                anchor,
                moving,
                preprocessing=config.preprocessing,
                registration=config.ecc_registration,
            )
        except ImagePreprocessingError as error:
            diagnostics.append(
                _preprocessing_failure_diagnostic(
                    relative_path,
                    is_anchor=False,
                    code=error.code,
                )
            )
            continue

        if result.succeeded and result.aligned_image is not None and result.valid_mask is not None:
            diagnostics.append(_registration_diagnostic(relative_path, result))
            aligned_images.append(result.aligned_image)
            validity_masks.append(result.valid_mask)
        elif result.succeeded:
            diagnostics.append(
                ReferenceRegistrationDiagnostic(
                    relative_path=relative_path,
                    is_anchor=False,
                    status="failed",
                    failure_code=ECCRegistrationFailureCode.ECC_RESULT_INVALID,
                    correlation=result.correlation,
                    warp_matrix=result.warp_matrix,
                    rotation_degrees=result.rotation_degrees,
                    translation_x_pixels=result.translation_x_pixels,
                    translation_y_pixels=result.translation_y_pixels,
                    valid_fraction=result.valid_fraction,
                )
            )
        else:
            diagnostics.append(_registration_diagnostic(relative_path, result))

    successful_count = len(aligned_images)
    diagnostic_tuple = tuple(diagnostics)
    if successful_count < config.ecc_template.minimum_successful_references:
        return _fit_failed(
            ECCTemplateFailureCode.FIT_INSUFFICIENT_REFERENCES,
            anchor_path=anchor_path,
            reference_count=reference_count,
            successful_reference_count=successful_count,
            diagnostics=diagnostic_tuple,
        )

    support_intersection = np.logical_and.reduce(validity_masks)
    kernel_size = config.ecc_template.support_erosion_kernel_size
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    try:
        eroded_support_raw = cv2.erode(
            support_intersection.astype(np.uint8),
            kernel,
            iterations=config.ecc_template.support_erosion_iterations,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    except cv2.error:
        return _fit_failed(
            ECCTemplateFailureCode.FIT_SUPPORT_EROSION_FAILED,
            anchor_path=anchor_path,
            reference_count=reference_count,
            successful_reference_count=successful_count,
            diagnostics=diagnostic_tuple,
        )
    if not isinstance(eroded_support_raw, np.ndarray) or eroded_support_raw.shape != anchor.shape:
        return _fit_failed(
            ECCTemplateFailureCode.FIT_SUPPORT_EROSION_FAILED,
            anchor_path=anchor_path,
            reference_count=reference_count,
            successful_reference_count=successful_count,
            diagnostics=diagnostic_tuple,
        )

    support_mask = eroded_support_raw.astype(bool)
    support_fraction = float(np.count_nonzero(support_mask) / support_mask.size)
    if support_fraction < config.ecc_template.minimum_support_fraction:
        return _fit_failed(
            ECCTemplateFailureCode.FIT_SUPPORT_TOO_SMALL,
            anchor_path=anchor_path,
            reference_count=reference_count,
            successful_reference_count=successful_count,
            support_fraction=support_fraction,
            support_mask=support_mask,
            diagnostics=diagnostic_tuple,
        )

    image_stack = np.stack(aligned_images)
    mask_stack = np.stack(validity_masks)
    valid_values = np.where(mask_stack, image_stack, np.nan)
    template = np.nanmedian(valid_values, axis=0).astype(np.float32)
    try:
        validate_preprocessed_image(
            template,
            label="fitted ECC template",
            config=config.preprocessing,
        )
    except ImagePreprocessingError:
        return _fit_failed(
            ECCTemplateFailureCode.FIT_TEMPLATE_INVALID,
            anchor_path=anchor_path,
            reference_count=reference_count,
            successful_reference_count=successful_count,
            support_fraction=support_fraction,
            support_mask=support_mask,
            diagnostics=diagnostic_tuple,
        )

    return ECCTemplateFitResult(
        status="ok",
        failure_code=None,
        anchor_path=anchor_path,
        reference_count=reference_count,
        successful_reference_count=successful_count,
        failed_reference_count=sum(item.status == "failed" for item in diagnostics),
        support_fraction=support_fraction,
        template=template,
        support_mask=support_mask,
        reference_diagnostics=diagnostic_tuple,
    )
