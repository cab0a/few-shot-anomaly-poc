from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

import few_shot_anomaly_poc.hog_features as hog_module
from few_shot_anomaly_poc.config import load_config
from few_shot_anomaly_poc.errors import (
    HOGFeatureFailureCode,
    PreprocessingFailureCode,
)
from few_shot_anomaly_poc.hog_features import extract_patch_hog_features


@pytest.fixture
def project_config():
    return load_config(Path("configs/v0.1.yaml"))


def _synthetic_pattern() -> np.ndarray:
    image = np.zeros((512, 512), dtype=np.float32)
    cv2.rectangle(image, (30, 40), (230, 280), color=0.8, thickness=-1)
    cv2.circle(image, (370, 330), 70, color=0.3, thickness=-1)
    cv2.line(image, (0, 511), (511, 0), color=1.0, thickness=5)
    return image


def test_extracts_repeatable_row_major_feature_matrix(project_config) -> None:
    image = _synthetic_pattern()
    original = image.copy()

    first = extract_patch_hog_features(image, config=project_config)
    second = extract_patch_hog_features(image, config=project_config)

    assert first.succeeded and second.succeeded
    assert first.status == "ok"
    assert first.failure_code is None
    assert first.failed_patch_index is None
    assert first.features is not None
    assert first.features.shape == (225, 324)
    assert first.features.dtype == np.float32
    assert np.isfinite(first.features).all()
    assert np.any(first.features > 0.0)
    assert second.features is not None
    assert np.array_equal(first.features, second.features)
    assert np.array_equal(image, original)

    assert len(first.positions) == 225
    assert first.positions[0] == hog_module.PatchPosition(0, 0, 0, 0, 0)
    assert first.positions[1] == hog_module.PatchPosition(1, 0, 1, 0, 32)
    assert first.positions[14] == hog_module.PatchPosition(14, 0, 14, 0, 448)
    assert first.positions[15] == hog_module.PatchPosition(15, 1, 0, 32, 0)
    assert first.positions[-1] == hog_module.PatchPosition(224, 14, 14, 448, 448)
    assert len({(item.top, item.left) for item in first.positions}) == 225


def test_extractor_passes_only_fixed_hog_arguments_in_patch_order(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    image = np.arange(512 * 512, dtype=np.float32).reshape(512, 512)
    image /= np.float32(image.size - 1)
    calls = []

    def fixed_hog(patch, **kwargs):
        call_index = len(calls)
        calls.append(
            {
                "shape": patch.shape,
                "top_left_value": float(patch[0, 0]),
                "kwargs": kwargs,
            }
        )
        return np.full((324,), call_index, dtype=np.float32)

    monkeypatch.setattr(hog_module, "hog", fixed_hog)

    result = extract_patch_hog_features(image, config=project_config)

    assert result.succeeded
    assert result.features is not None
    assert len(calls) == 225
    assert calls[0]["top_left_value"] == pytest.approx(float(image[0, 0]))
    assert calls[1]["top_left_value"] == pytest.approx(float(image[0, 32]))
    assert calls[15]["top_left_value"] == pytest.approx(float(image[32, 0]))
    assert calls[-1]["top_left_value"] == pytest.approx(float(image[448, 448]))
    assert all(item["shape"] == (64, 64) for item in calls)
    assert calls[0]["kwargs"] == {
        "orientations": 9,
        "pixels_per_cell": (16, 16),
        "cells_per_block": (2, 2),
        "block_norm": "L2-Hys",
        "visualize": False,
        "transform_sqrt": True,
        "feature_vector": True,
        "channel_axis": None,
    }
    assert np.all(result.features[0] == np.float32(0.0))
    assert np.all(result.features[15] == np.float32(15.0))
    assert np.all(result.features[224] == np.float32(224.0))


def test_extractor_rejects_invalid_preprocessed_input(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    called = False

    def unexpected_hog(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("HOG must not run for invalid input")

    monkeypatch.setattr(hog_module, "hog", unexpected_hog)
    invalid = np.zeros((512, 512), dtype=np.float64)

    result = extract_patch_hog_features(invalid, config=project_config)

    assert not result.succeeded
    assert result.failure_code is PreprocessingFailureCode.INVALID_PREPROCESSED_IMAGE
    assert result.features is None
    assert not result.positions
    assert result.failed_patch_index is None
    assert not called


def test_extractor_reports_grid_inconsistency(project_config) -> None:
    changed_hog = replace(project_config.patch_hog, horizontal_positions=14)
    changed_config = replace(project_config, patch_hog=changed_hog)

    result = extract_patch_hog_features(
        _synthetic_pattern(),
        config=changed_config,
    )

    assert result.failure_code is HOGFeatureFailureCode.HOG_GRID_INVALID
    assert result.features is None
    assert not result.positions


def test_extractor_discards_partial_matrix_after_library_failure(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
) -> None:
    call_count = 0

    def fail_fourth_patch(patch, **kwargs):
        nonlocal call_count
        current = call_count
        call_count += 1
        if current == 3:
            raise ValueError("synthetic HOG failure")
        return np.zeros((324,), dtype=np.float32)

    monkeypatch.setattr(hog_module, "hog", fail_fourth_patch)

    result = extract_patch_hog_features(
        _synthetic_pattern(),
        config=project_config,
    )

    assert result.failure_code is HOGFeatureFailureCode.HOG_EXTRACTION_FAILED
    assert result.features is None
    assert len(result.positions) == 225
    assert result.failed_patch_index == 3
    assert call_count == 4


@pytest.mark.parametrize(
    "invalid_descriptor",
    [
        np.zeros((323,), dtype=np.float32),
        np.zeros((324,), dtype=np.float64),
        np.full((324,), np.nan, dtype=np.float32),
    ],
)
def test_extractor_rejects_invalid_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    project_config,
    invalid_descriptor: np.ndarray,
) -> None:
    monkeypatch.setattr(
        hog_module,
        "hog",
        lambda *args, **kwargs: invalid_descriptor,
    )

    result = extract_patch_hog_features(
        _synthetic_pattern(),
        config=project_config,
    )

    assert result.failure_code is HOGFeatureFailureCode.HOG_DESCRIPTOR_INVALID
    assert result.features is None
    assert len(result.positions) == 225
    assert result.failed_patch_index == 0
