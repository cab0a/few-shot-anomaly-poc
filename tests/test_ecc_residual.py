from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

import few_shot_anomaly_poc.ecc_residual as residual_module
from few_shot_anomaly_poc.config import load_config
from few_shot_anomaly_poc.ecc_residual import score_ecc_residual
from few_shot_anomaly_poc.ecc_template import ECCTemplateFitResult
from few_shot_anomaly_poc.errors import (
    ECCRegistrationFailureCode,
    ECCResidualFailureCode,
    ECCResidualStateError,
    ECCTemplateFailureCode,
    PreprocessingFailureCode,
)
from few_shot_anomaly_poc.registration import ECCRegistrationResult


@pytest.fixture
def project_config():
    return load_config(Path("configs/v0.1.yaml"))


def _synthetic_pattern() -> np.ndarray:
    generator = np.random.default_rng(23)
    noise = generator.random((512, 512), dtype=np.float32)
    pattern = cv2.GaussianBlur(noise, (0, 0), sigmaX=4.0)
    cv2.rectangle(pattern, (70, 90), (250, 270), color=0.8, thickness=-1)
    cv2.circle(pattern, (380, 350), 50, color=0.2, thickness=-1)
    return np.asarray(pattern, dtype=np.float32)


def _standard_support() -> np.ndarray:
    support = np.zeros((512, 512), dtype=bool)
    support[2:-2, 2:-2] = True
    return support


def _fitted_state(
    template: np.ndarray,
    *,
    support_mask: np.ndarray | None = None,
) -> ECCTemplateFitResult:
    if support_mask is None:
        support_mask = _standard_support()
    support_fraction = float(np.count_nonzero(support_mask) / support_mask.size)
    return ECCTemplateFitResult(
        status="ok",
        failure_code=None,
        anchor_path="reference-00.png",
        reference_count=20,
        successful_reference_count=20,
        failed_reference_count=0,
        support_fraction=support_fraction,
        template=template,
        support_mask=support_mask,
        reference_diagnostics=(),
    )


def _successful_registration(
    aligned_image: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> ECCRegistrationResult:
    if valid_mask is None:
        valid_mask = np.ones((512, 512), dtype=bool)
    return ECCRegistrationResult(
        status="ok",
        failure_code=None,
        correlation=0.95,
        warp_matrix=np.eye(2, 3, dtype=np.float32),
        rotation_degrees=0.0,
        translation_x_pixels=0.0,
        translation_y_pixels=0.0,
        valid_fraction=float(np.count_nonzero(valid_mask) / valid_mask.size),
        aligned_image=aligned_image,
        valid_mask=valid_mask,
    )


def _failed_registration() -> ECCRegistrationResult:
    return ECCRegistrationResult(
        status="failed",
        failure_code=ECCRegistrationFailureCode.ECC_OPTIMIZATION_FAILED,
        correlation=None,
        warp_matrix=None,
        rotation_degrees=None,
        translation_x_pixels=None,
        translation_y_pixels=None,
        valid_fraction=None,
        aligned_image=None,
        valid_mask=None,
    )


def test_score_integrates_real_registration_for_identical_image(project_config) -> None:
    image = _synthetic_pattern()

    first = score_ecc_residual(
        image.copy(),
        fitted=_fitted_state(image),
        config=project_config,
    )
    second = score_ecc_residual(
        image.copy(),
        fitted=_fitted_state(image),
        config=project_config,
    )

    expected_pixels = 508 * 508
    assert first.succeeded and second.succeeded
    assert first.score_status == "ok"
    assert first.failure_code is None
    assert first.anomaly_score == pytest.approx(0.0, abs=1e-6)
    assert first.registration_status == "ok"
    assert first.correlation == pytest.approx(1.0, abs=1e-4)
    assert first.effective_support_fraction == 1.0
    assert first.effective_pixel_count == expected_pixels
    assert first.top_pixel_count == math.ceil(0.01 * expected_pixels)
    assert first.anomaly_score == second.anomaly_score
    assert first.correlation == second.correlation
    assert np.array_equal(first.warp_matrix, second.warp_matrix)
    assert first.rotation_degrees == second.rotation_degrees
    assert first.translation_x_pixels == second.translation_x_pixels
    assert first.translation_y_pixels == second.translation_y_pixels
    assert first.effective_support_fraction == second.effective_support_fraction
    assert first.effective_pixel_count == second.effective_pixel_count
    assert first.top_pixel_count == second.top_pixel_count


def test_score_uses_fixed_gaussian_and_top_one_percent(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    template = np.zeros((512, 512), dtype=np.float32)
    aligned = template.copy()
    aligned[10:20, 10:20] = 1.0
    monkeypatch.setattr(
        residual_module,
        "register_ecc",
        lambda *args, **kwargs: _successful_registration(aligned),
    )
    blur_arguments = {}

    def identity_blur(source, kernel_size, *, sigmaX, sigmaY, borderType):
        blur_arguments.update(
            {
                "kernel_size": kernel_size,
                "sigmaX": sigmaX,
                "sigmaY": sigmaY,
                "borderType": borderType,
            }
        )
        return source

    monkeypatch.setattr(residual_module.cv2, "GaussianBlur", identity_blur)

    result = score_ecc_residual(
        template.copy(),
        fitted=_fitted_state(template),
        config=project_config,
    )

    expected_top_count = math.ceil(0.01 * (508 * 508))
    assert result.succeeded
    assert result.top_pixel_count == expected_top_count
    assert result.anomaly_score == pytest.approx(100 / expected_top_count)
    assert blur_arguments == {
        "kernel_size": (5, 5),
        "sigmaX": 0.0,
        "sigmaY": 0.0,
        "borderType": cv2.BORDER_CONSTANT,
    }


@pytest.mark.parametrize(
    ("count_offset", "expected_success"),
    [(0, True), (-1, False)],
)
def test_score_applies_ninety_five_percent_effective_support_gate(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
    count_offset: int,
    expected_success: bool,
) -> None:
    template = np.zeros((512, 512), dtype=np.float32)
    support = _standard_support()
    support_indices = np.flatnonzero(support)
    retained_count = math.ceil(0.95 * support_indices.size) + count_offset
    eroded = np.zeros((512, 512), dtype=np.uint8)
    eroded.flat[support_indices[:retained_count]] = 1
    monkeypatch.setattr(
        residual_module,
        "register_ecc",
        lambda *args, **kwargs: _successful_registration(template),
    )
    monkeypatch.setattr(
        residual_module.cv2,
        "erode",
        lambda *args, **kwargs: eroded,
    )

    result = score_ecc_residual(
        template.copy(),
        fitted=_fitted_state(template, support_mask=support),
        config=project_config,
    )

    assert result.effective_pixel_count == retained_count
    assert result.effective_support_fraction == pytest.approx(retained_count / support_indices.size)
    if expected_success:
        assert result.succeeded
        assert result.anomaly_score == 0.0
    else:
        assert not result.succeeded
        assert result.failure_code is ECCResidualFailureCode.SCORE_EFFECTIVE_SUPPORT_TOO_SMALL
        assert result.anomaly_score == 1.0


def test_score_preserves_registration_failure_and_fixed_score(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    template = np.zeros((512, 512), dtype=np.float32)
    monkeypatch.setattr(
        residual_module,
        "register_ecc",
        lambda *args, **kwargs: _failed_registration(),
    )

    result = score_ecc_residual(
        template.copy(),
        fitted=_fitted_state(template),
        config=project_config,
    )

    assert not result.succeeded
    assert result.score_status == "failed"
    assert result.failure_code is ECCRegistrationFailureCode.ECC_OPTIMIZATION_FAILED
    assert result.anomaly_score == 1.0
    assert result.registration_status == "failed"
    assert result.top_pixel_count is None


def test_score_converts_invalid_input_to_fixed_failure_score(project_config) -> None:
    template = np.zeros((512, 512), dtype=np.float32)
    invalid = template.astype(np.float64)

    result = score_ecc_residual(
        invalid,
        fitted=_fitted_state(template),
        config=project_config,
    )

    assert result.failure_code is PreprocessingFailureCode.INVALID_PREPROCESSED_IMAGE
    assert result.anomaly_score == 1.0
    assert result.registration_status == "not_run"


def test_score_rejects_method_level_fit_failure(project_config) -> None:
    unavailable = ECCTemplateFitResult(
        status="FIT_FAILED",
        failure_code=ECCTemplateFailureCode.FIT_INSUFFICIENT_REFERENCES,
        anchor_path="reference-00.png",
        reference_count=20,
        successful_reference_count=15,
        failed_reference_count=5,
        support_fraction=None,
        template=None,
        support_mask=None,
        reference_diagnostics=(),
    )
    image = np.zeros((512, 512), dtype=np.float32)

    with pytest.raises(ECCResidualStateError, match="successful fitted ECC state"):
        score_ecc_residual(image, fitted=unavailable, config=project_config)


def test_score_rejects_invalid_fitted_support_state(project_config) -> None:
    template = np.zeros((512, 512), dtype=np.float32)
    invalid_support = _standard_support().astype(np.uint8)

    with pytest.raises(ECCResidualStateError, match="support mask"):
        score_ecc_residual(
            template,
            fitted=_fitted_state(template, support_mask=invalid_support),
            config=project_config,
        )


def test_score_reports_mask_erosion_failure(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    template = np.zeros((512, 512), dtype=np.float32)
    monkeypatch.setattr(
        residual_module,
        "register_ecc",
        lambda *args, **kwargs: _successful_registration(template),
    )

    def fail_erosion(*args, **kwargs):
        raise cv2.error("synthetic erosion failure")

    monkeypatch.setattr(residual_module.cv2, "erode", fail_erosion)

    result = score_ecc_residual(
        template,
        fitted=_fitted_state(template),
        config=project_config,
    )

    assert result.failure_code is ECCResidualFailureCode.SCORE_MASK_EROSION_FAILED
    assert result.anomaly_score == 1.0
    assert result.registration_status == "ok"


def test_score_reports_residual_filter_failure(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    template = np.zeros((512, 512), dtype=np.float32)
    monkeypatch.setattr(
        residual_module,
        "register_ecc",
        lambda *args, **kwargs: _successful_registration(template),
    )

    def fail_blur(*args, **kwargs):
        raise cv2.error("synthetic blur failure")

    monkeypatch.setattr(residual_module.cv2, "GaussianBlur", fail_blur)

    result = score_ecc_residual(
        template,
        fitted=_fitted_state(template),
        config=project_config,
    )

    assert result.failure_code is ECCResidualFailureCode.SCORE_RESIDUAL_FILTER_FAILED
    assert result.anomaly_score == 1.0
    assert result.effective_support_fraction == 1.0


def test_score_rejects_incomplete_successful_registration(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    template = np.zeros((512, 512), dtype=np.float32)
    incomplete = _successful_registration(template)
    incomplete = ECCRegistrationResult(
        status=incomplete.status,
        failure_code=incomplete.failure_code,
        correlation=incomplete.correlation,
        warp_matrix=incomplete.warp_matrix,
        rotation_degrees=incomplete.rotation_degrees,
        translation_x_pixels=incomplete.translation_x_pixels,
        translation_y_pixels=incomplete.translation_y_pixels,
        valid_fraction=incomplete.valid_fraction,
        aligned_image=None,
        valid_mask=incomplete.valid_mask,
    )
    monkeypatch.setattr(
        residual_module,
        "register_ecc",
        lambda *args, **kwargs: incomplete,
    )

    result = score_ecc_residual(
        template,
        fitted=_fitted_state(template),
        config=project_config,
    )

    assert result.failure_code is ECCRegistrationFailureCode.ECC_RESULT_INVALID
    assert result.anomaly_score == 1.0


def test_score_reports_invalid_final_score(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    template = np.zeros((512, 512), dtype=np.float32)
    monkeypatch.setattr(
        residual_module,
        "register_ecc",
        lambda *args, **kwargs: _successful_registration(template),
    )
    monkeypatch.setattr(
        residual_module.np,
        "mean",
        lambda values, dtype: np.float64(np.nan),
    )

    result = score_ecc_residual(
        template,
        fitted=_fitted_state(template),
        config=project_config,
    )

    assert result.failure_code is ECCResidualFailureCode.SCORE_RESULT_INVALID
    assert result.anomaly_score == 1.0
