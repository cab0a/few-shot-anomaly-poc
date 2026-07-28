from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from few_shot_anomaly_poc.config import load_config
from few_shot_anomaly_poc.errors import (
    ImagePreprocessingError,
    PreprocessingFailureCode,
)
from few_shot_anomaly_poc.preprocessing import (
    DECODE_FLAGS,
    load_and_preprocess_image,
    preprocess_decoded_image,
)


@pytest.fixture
def preprocessing_config():
    return load_config(Path("configs/v0.1.yaml")).preprocessing


def test_load_and_preprocess_returns_fixed_float32_image(
    tmp_path: Path,
    preprocessing_config,
) -> None:
    source = np.full((19, 31), 128, dtype=np.uint8)
    path = tmp_path / "constant.png"
    assert cv2.imwrite(str(path), source)

    output = load_and_preprocess_image(path, preprocessing_config)

    assert output.shape == (512, 512)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert np.all(output == np.float32(128.0 / 255.0))


def test_load_and_preprocess_decodes_color_as_grayscale(
    tmp_path: Path,
    preprocessing_config,
) -> None:
    source = np.zeros((8, 12, 3), dtype=np.uint8)
    source[:, :] = (0, 0, 255)
    path = tmp_path / "red.png"
    assert cv2.imwrite(str(path), source)

    output = load_and_preprocess_image(path, preprocessing_config)

    assert np.all(output == np.float32(76.0 / 255.0))


def test_decode_flags_fix_grayscale_and_ignore_orientation() -> None:
    assert DECODE_FLAGS & cv2.IMREAD_IGNORE_ORIENTATION
    assert DECODE_FLAGS & 7 == cv2.IMREAD_GRAYSCALE


@pytest.mark.parametrize(
    "image",
    [
        None,
        np.empty((0, 3), dtype=np.uint8),
        np.zeros((3, 4, 1), dtype=np.uint8),
        np.zeros((3, 4), dtype=np.float32),
    ],
)
def test_preprocess_rejects_invalid_decoded_image(
    image,
    preprocessing_config,
) -> None:
    with pytest.raises(ImagePreprocessingError) as captured:
        preprocess_decoded_image(image, preprocessing_config)

    assert captured.value.code is PreprocessingFailureCode.INVALID_DECODED_IMAGE


def test_load_rejects_missing_image(
    tmp_path: Path,
    preprocessing_config,
) -> None:
    with pytest.raises(ImagePreprocessingError) as captured:
        load_and_preprocess_image(tmp_path / "missing.png", preprocessing_config)

    assert captured.value.code is PreprocessingFailureCode.IMAGE_NOT_FOUND


def test_load_rejects_undecodable_bytes(
    tmp_path: Path,
    preprocessing_config,
) -> None:
    path = tmp_path / "invalid.png"
    path.write_bytes(b"not an image")

    with pytest.raises(ImagePreprocessingError) as captured:
        load_and_preprocess_image(path, preprocessing_config)

    assert captured.value.code is PreprocessingFailureCode.IMAGE_DECODE_FAILED
