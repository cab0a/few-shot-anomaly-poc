from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import few_shot_anomaly_poc.ecc_template as template_module
from few_shot_anomaly_poc.config import load_config
from few_shot_anomaly_poc.ecc_template import fit_ecc_normal_template
from few_shot_anomaly_poc.errors import (
    ECCRegistrationFailureCode,
    ECCTemplateFailureCode,
    PreprocessingFailureCode,
)
from few_shot_anomaly_poc.registration import ECCRegistrationResult


@pytest.fixture
def project_config():
    return load_config(Path("configs/v0.1.yaml"))


def _synthetic_pattern() -> np.ndarray:
    generator = np.random.default_rng(7)
    noise = generator.random((512, 512), dtype=np.float32)
    pattern = cv2.GaussianBlur(noise, (0, 0), sigmaX=4.0)
    cv2.rectangle(pattern, (80, 100), (240, 280), color=0.85, thickness=-1)
    cv2.circle(pattern, (370, 350), 45, color=0.15, thickness=-1)
    return np.asarray(pattern, dtype=np.float32)


def _identical_references(image: np.ndarray, *, count: int = 20) -> dict[str, np.ndarray]:
    return {f"reference-{index:02d}.png": image.copy() for index in reversed(range(count))}


def _constant_references() -> dict[str, np.ndarray]:
    return {
        f"reference-{index:02d}.png": np.full(
            (512, 512),
            index / 19.0,
            dtype=np.float32,
        )
        for index in reversed(range(20))
    }


def _successful_registration(
    moving: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> ECCRegistrationResult:
    if valid_mask is None:
        valid_mask = np.ones((512, 512), dtype=bool)
    return ECCRegistrationResult(
        status="ok",
        failure_code=None,
        correlation=0.9,
        warp_matrix=np.eye(2, 3, dtype=np.float32),
        rotation_degrees=0.0,
        translation_x_pixels=0.0,
        translation_y_pixels=0.0,
        valid_fraction=float(np.count_nonzero(valid_mask) / valid_mask.size),
        aligned_image=moving.copy(),
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


def test_fit_integrates_real_registration_for_identical_references(project_config) -> None:
    image = _synthetic_pattern()

    result = fit_ecc_normal_template(
        _identical_references(image),
        config=project_config,
    )

    assert result.succeeded
    assert result.status == "ok"
    assert result.failure_code is None
    assert result.anchor_path == "reference-00.png"
    assert result.reference_count == 20
    assert result.successful_reference_count == 20
    assert result.failed_reference_count == 0
    assert result.template is not None
    assert result.template.dtype == np.float32
    assert np.allclose(result.template, image, atol=1e-5)
    assert result.support_mask is not None
    assert result.support_mask.dtype == np.bool_
    assert result.support_fraction == pytest.approx((508 * 508) / (512 * 512))
    assert not result.support_mask[:2, :].any()
    assert not result.support_mask[-2:, :].any()
    assert not result.support_mask[:, :2].any()
    assert not result.support_mask[:, -2:].any()
    assert result.support_mask[2:-2, 2:-2].all()


def test_fit_sorts_paths_and_uses_valid_pixelwise_median(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    monkeypatch.setattr(
        template_module,
        "register_ecc",
        lambda template, moving, **kwargs: _successful_registration(moving),
    )

    result = fit_ecc_normal_template(_constant_references(), config=project_config)

    expected_paths = [f"reference-{index:02d}.png" for index in range(20)]
    assert result.succeeded
    assert result.anchor_path == expected_paths[0]
    assert [item.relative_path for item in result.reference_diagnostics] == expected_paths
    assert result.reference_diagnostics[0].is_anchor
    assert result.reference_diagnostics[0].correlation is None
    assert result.template is not None
    assert np.all(result.template == np.float32(0.5))


def test_fit_accepts_exactly_sixteen_successful_references(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    def register_by_value(template, moving, **kwargs):
        index = round(float(moving[0, 0]) * 19)
        if index <= 15:
            return _successful_registration(moving)
        return _failed_registration()

    monkeypatch.setattr(template_module, "register_ecc", register_by_value)

    result = fit_ecc_normal_template(_constant_references(), config=project_config)

    assert result.succeeded
    assert result.successful_reference_count == 16
    assert result.failed_reference_count == 4
    assert sum(item.status == "failed" for item in result.reference_diagnostics) == 4


def test_fit_fails_below_minimum_successful_references(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    def register_by_value(template, moving, **kwargs):
        index = round(float(moving[0, 0]) * 19)
        if index <= 14:
            return _successful_registration(moving)
        return _failed_registration()

    monkeypatch.setattr(template_module, "register_ecc", register_by_value)

    result = fit_ecc_normal_template(_constant_references(), config=project_config)

    assert not result.succeeded
    assert result.status == "FIT_FAILED"
    assert result.failure_code is ECCTemplateFailureCode.FIT_INSUFFICIENT_REFERENCES
    assert result.successful_reference_count == 15
    assert result.failed_reference_count == 5
    assert result.template is None
    assert result.support_mask is None


def test_fit_fails_when_eroded_support_is_too_small(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    narrow_mask = np.zeros((512, 512), dtype=bool)
    narrow_mask[:, :350] = True
    monkeypatch.setattr(
        template_module,
        "register_ecc",
        lambda template, moving, **kwargs: _successful_registration(
            moving,
            valid_mask=narrow_mask,
        ),
    )

    result = fit_ecc_normal_template(_constant_references(), config=project_config)

    assert result.failure_code is ECCTemplateFailureCode.FIT_SUPPORT_TOO_SMALL
    assert result.successful_reference_count == 20
    assert result.support_fraction is not None and result.support_fraction < 0.75
    assert result.support_mask is not None
    assert result.template is None


def test_fit_fails_when_anchor_is_not_preprocessed(project_config) -> None:
    references = _identical_references(_synthetic_pattern())
    references["reference-00.png"] = references["reference-00.png"].astype(np.float64)

    result = fit_ecc_normal_template(references, config=project_config)

    assert result.failure_code is ECCTemplateFailureCode.FIT_ANCHOR_PREPROCESSING_FAILED
    assert result.anchor_path == "reference-00.png"
    assert result.successful_reference_count == 0
    assert result.failed_reference_count == 1
    assert len(result.reference_diagnostics) == 1
    diagnostic = result.reference_diagnostics[0]
    assert diagnostic.is_anchor
    assert diagnostic.failure_code is PreprocessingFailureCode.INVALID_PREPROCESSED_IMAGE


def test_fit_records_invalid_non_anchor_and_continues(project_config) -> None:
    references = _identical_references(_synthetic_pattern())
    references["reference-19.png"] = references["reference-19.png"].astype(np.float64)

    result = fit_ecc_normal_template(references, config=project_config)

    assert result.succeeded
    assert result.successful_reference_count == 19
    assert result.failed_reference_count == 1
    diagnostic = result.reference_diagnostics[-1]
    assert diagnostic.relative_path == "reference-19.png"
    assert diagnostic.failure_code is PreprocessingFailureCode.INVALID_PREPROCESSED_IMAGE


def test_fit_rejects_wrong_reference_count(project_config) -> None:
    references = _identical_references(_synthetic_pattern(), count=19)

    result = fit_ecc_normal_template(references, config=project_config)

    assert result.failure_code is ECCTemplateFailureCode.FIT_REFERENCE_COUNT_INVALID
    assert result.reference_count == 19
    assert result.successful_reference_count == 0
    assert not result.reference_diagnostics


def test_fit_reports_support_erosion_failure(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    monkeypatch.setattr(
        template_module,
        "register_ecc",
        lambda template, moving, **kwargs: _successful_registration(moving),
    )

    def fail_erosion(*args, **kwargs):
        raise cv2.error("synthetic erosion failure")

    monkeypatch.setattr(cv2, "erode", fail_erosion)

    result = fit_ecc_normal_template(_constant_references(), config=project_config)

    assert result.failure_code is ECCTemplateFailureCode.FIT_SUPPORT_EROSION_FAILED
    assert result.successful_reference_count == 20


def test_fit_reports_invalid_aggregated_template(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    monkeypatch.setattr(
        template_module,
        "register_ecc",
        lambda template, moving, **kwargs: _successful_registration(moving),
    )
    monkeypatch.setattr(
        template_module.np,
        "nanmedian",
        lambda values, axis: np.full((512, 512), np.nan, dtype=np.float32),
    )

    result = fit_ecc_normal_template(_constant_references(), config=project_config)

    assert result.failure_code is ECCTemplateFailureCode.FIT_TEMPLATE_INVALID
    assert result.successful_reference_count == 20
    assert result.template is None
