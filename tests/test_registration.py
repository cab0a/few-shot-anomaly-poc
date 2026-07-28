from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from few_shot_anomaly_poc.config import load_config
from few_shot_anomaly_poc.errors import (
    ECCRegistrationFailureCode,
    ImagePreprocessingError,
    PreprocessingFailureCode,
)
from few_shot_anomaly_poc.registration import register_ecc


@pytest.fixture
def project_config():
    return load_config(Path("configs/v0.1.yaml"))


def _synthetic_pattern() -> np.ndarray:
    generator = np.random.default_rng(42)
    noise = generator.random((512, 512), dtype=np.float32)
    pattern = cv2.GaussianBlur(noise, (0, 0), sigmaX=3.0)
    cv2.rectangle(pattern, (90, 120), (220, 260), color=0.9, thickness=-1)
    cv2.circle(pattern, (360, 330), 55, color=0.1, thickness=-1)
    return np.asarray(pattern, dtype=np.float32)


def _run(template: np.ndarray, moving: np.ndarray, project_config):
    return register_ecc(
        template,
        moving,
        preprocessing=project_config.preprocessing,
        registration=project_config.ecc_registration,
    )


def test_identity_registration_succeeds(project_config) -> None:
    template = _synthetic_pattern()

    result = _run(template, template.copy(), project_config)

    assert result.succeeded
    assert result.status == "ok"
    assert result.failure_code is None
    assert result.correlation == pytest.approx(1.0, abs=1e-4)
    assert result.warp_matrix == pytest.approx(
        np.eye(2, 3, dtype=np.float32),
        abs=1e-4,
    )
    assert result.rotation_degrees == pytest.approx(0.0, abs=1e-3)
    assert result.translation_x_pixels == pytest.approx(0.0, abs=1e-3)
    assert result.translation_y_pixels == pytest.approx(0.0, abs=1e-3)
    assert result.valid_fraction == 1.0
    assert result.aligned_image is not None
    assert result.valid_mask is not None
    assert result.aligned_image.dtype == np.float32
    assert result.valid_mask.dtype == np.bool_
    assert result.valid_mask.all()


def test_registration_recovers_bounded_euclidean_motion(project_config) -> None:
    template = _synthetic_pattern()
    known_warp = cv2.getRotationMatrix2D((256.0, 256.0), 3.0, 1.0).astype(np.float32)
    known_warp[:, 2] += np.array([7.0, -5.0], dtype=np.float32)
    moving = cv2.warpAffine(
        template,
        known_warp,
        (512, 512),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    result = _run(template, moving, project_config)

    assert result.succeeded
    assert result.correlation is not None and result.correlation > 0.99
    assert result.rotation_degrees == pytest.approx(-3.0, abs=0.2)
    assert result.translation_x_pixels is not None
    assert result.translation_y_pixels is not None
    assert result.valid_fraction is not None and result.valid_fraction >= 0.80
    assert result.aligned_image is not None
    valid = result.valid_mask
    assert valid is not None
    before = np.mean(np.abs(template[valid] - moving[valid]))
    after = np.mean(np.abs(template[valid] - result.aligned_image[valid]))
    assert after < before * 0.2


def test_registration_is_repeatable(project_config) -> None:
    template = _synthetic_pattern()
    known_warp = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, -3.0]], dtype=np.float32)
    moving = cv2.warpAffine(template, known_warp, (512, 512))

    first = _run(template, moving, project_config)
    second = _run(template, moving, project_config)

    assert first.succeeded and second.succeeded
    assert first.correlation == second.correlation
    assert np.array_equal(first.warp_matrix, second.warp_matrix)
    assert np.array_equal(first.aligned_image, second.aligned_image)
    assert np.array_equal(first.valid_mask, second.valid_mask)


def test_low_finite_correlation_is_not_an_independent_failure(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    def fixed_ecc(
        template,
        moving,
        initial_warp,
        motion_model,
        criteria,
        input_mask,
        gaussian_filter_size,
    ):
        assert template.shape == (512, 512)
        assert moving.shape == (512, 512)
        assert initial_warp == pytest.approx(np.eye(2, 3, dtype=np.float32))
        assert motion_model == cv2.MOTION_EUCLIDEAN
        assert criteria == (
            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
            100,
            1e-6,
        )
        assert input_mask is None
        assert gaussian_filter_size == 5
        return -0.25, initial_warp

    monkeypatch.setattr(cv2, "findTransformECC", fixed_ecc)
    image = _synthetic_pattern()

    result = _run(image, image, project_config)

    assert result.succeeded
    assert result.correlation == -0.25


def test_ecc_exception_returns_optimization_failure(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    def fail_ecc(*args, **kwargs):
        raise cv2.error("synthetic non-convergence")

    monkeypatch.setattr(cv2, "findTransformECC", fail_ecc)
    image = _synthetic_pattern()

    result = _run(image, image, project_config)

    assert not result.succeeded
    assert result.failure_code is ECCRegistrationFailureCode.ECC_OPTIMIZATION_FAILED


def test_nonfinite_result_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    warp = np.eye(2, 3, dtype=np.float32)
    warp[0, 2] = np.nan
    monkeypatch.setattr(
        cv2,
        "findTransformECC",
        lambda *args, **kwargs: (math.nan, warp),
    )
    image = _synthetic_pattern()

    result = _run(image, image, project_config)

    assert result.failure_code is ECCRegistrationFailureCode.ECC_RESULT_NONFINITE


def test_invalid_result_shape_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    monkeypatch.setattr(
        cv2,
        "findTransformECC",
        lambda *args, **kwargs: (0.8, np.eye(3, dtype=np.float32)),
    )
    image = _synthetic_pattern()

    result = _run(image, image, project_config)

    assert result.failure_code is ECCRegistrationFailureCode.ECC_RESULT_INVALID


@pytest.mark.parametrize(
    ("warp", "expected_code"),
    [
        (
            np.array(
                [
                    [math.cos(math.radians(10.1)), -math.sin(math.radians(10.1)), 0.0],
                    [math.sin(math.radians(10.1)), math.cos(math.radians(10.1)), 0.0],
                ],
                dtype=np.float32,
            ),
            ECCRegistrationFailureCode.ECC_ROTATION_LIMIT_EXCEEDED,
        ),
        (
            np.array([[1.0, 0.0, 64.1], [0.0, 1.0, 0.0]], dtype=np.float32),
            ECCRegistrationFailureCode.ECC_TRANSLATION_LIMIT_EXCEEDED,
        ),
    ],
)
def test_registration_rejects_warp_outside_fixed_limits(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
    warp: np.ndarray,
    expected_code: ECCRegistrationFailureCode,
) -> None:
    monkeypatch.setattr(
        cv2,
        "findTransformECC",
        lambda *args, **kwargs: (0.8, warp),
    )
    image = _synthetic_pattern()

    result = _run(image, image, project_config)

    assert result.failure_code is expected_code
    assert result.warp_matrix is not None


def test_registration_rejects_small_valid_area(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    angle = math.radians(9.9)
    warp = np.array(
        [
            [math.cos(angle), -math.sin(angle), 64.0],
            [math.sin(angle), math.cos(angle), 64.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        cv2,
        "findTransformECC",
        lambda *args, **kwargs: (0.8, warp),
    )
    image = _synthetic_pattern()

    result = _run(image, image, project_config)

    assert result.failure_code is ECCRegistrationFailureCode.ECC_VALID_AREA_TOO_SMALL
    assert result.valid_fraction is not None and result.valid_fraction < 0.80


def test_warp_exception_is_reported(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    identity = np.eye(2, 3, dtype=np.float32)
    monkeypatch.setattr(
        cv2,
        "findTransformECC",
        lambda *args, **kwargs: (0.8, identity),
    )

    def fail_warp(*args, **kwargs):
        raise cv2.error("synthetic warp failure")

    monkeypatch.setattr(cv2, "warpAffine", fail_warp)
    image = _synthetic_pattern()

    result = _run(image, image, project_config)

    assert result.failure_code is ECCRegistrationFailureCode.ECC_WARP_FAILED


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((511, 512), dtype=np.float32),
        np.zeros((512, 512), dtype=np.float64),
        np.full((512, 512), np.nan, dtype=np.float32),
        np.full((512, 512), 1.01, dtype=np.float32),
    ],
)
def test_registration_rejects_invalid_preprocessed_input(
    invalid: np.ndarray,
    project_config,
) -> None:
    valid = _synthetic_pattern()

    with pytest.raises(ImagePreprocessingError) as captured:
        _run(valid, invalid, project_config)

    assert captured.value.code is PreprocessingFailureCode.INVALID_PREPROCESSED_IMAGE
