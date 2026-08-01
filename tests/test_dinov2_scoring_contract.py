from __future__ import annotations

import numpy as np
import pytest

from few_shot_anomaly_poc.dinov2_errors import (
    DINOv2ScoringError,
    DINOv2ScoringFailureCode,
)
from few_shot_anomaly_poc.dinov2_scoring import (
    ALLOWED_RESOLUTIONS,
    EMBEDDING_DIMENSION,
    INPUT_SHAPE,
    MEMORY_BLOCK_SIZE,
    REFERENCE_COUNT,
    expected_patch_count,
    top_patch_count,
    validate_input_image,
    validate_reference_count,
    validate_resolution,
)


def test_preregistered_scoring_constants_are_fixed() -> None:
    assert INPUT_SHAPE == (512, 512, 3)
    assert ALLOWED_RESOLUTIONS == (224, 448)
    assert EMBEDDING_DIMENSION == 384
    assert REFERENCE_COUNT == 20
    assert MEMORY_BLOCK_SIZE == 2_048


@pytest.mark.parametrize(
    ("resolution", "patch_count", "selected_count"),
    [(224, 256, 2), (448, 1_024, 10)],
)
def test_resolution_fixes_patch_and_top_fraction_counts(
    resolution: int,
    patch_count: int,
    selected_count: int,
) -> None:
    assert validate_resolution(resolution) == resolution
    assert expected_patch_count(resolution) == patch_count
    assert top_patch_count(patch_count) == selected_count


@pytest.mark.parametrize("resolution", [True, 0, 225, 896, 224.0, "224", None])
def test_unregistered_resolution_is_rejected(resolution: object) -> None:
    with pytest.raises(DINOv2ScoringError) as caught:
        validate_resolution(resolution)

    assert caught.value.code is DINOv2ScoringFailureCode.DINO_RESOLUTION_INVALID


def test_top_fraction_count_uses_floor_with_a_minimum_of_one() -> None:
    assert top_patch_count(1) == 1
    assert top_patch_count(99) == 1
    assert top_patch_count(100) == 1
    assert top_patch_count(199) == 1
    assert top_patch_count(200) == 2


@pytest.mark.parametrize("patch_count", [True, 0, -1, 256.0, "256", None])
def test_invalid_patch_count_is_rejected(patch_count: object) -> None:
    with pytest.raises(DINOv2ScoringError) as caught:
        top_patch_count(patch_count)

    assert caught.value.code is DINOv2ScoringFailureCode.DINO_SCORE_INVALID


def test_fixed_input_validation_returns_the_original_contiguous_array() -> None:
    image = np.zeros(INPUT_SHAPE, dtype=np.uint8)

    result = validate_input_image(image)

    assert result is image


@pytest.mark.parametrize(
    ("image", "expected_code"),
    [
        (
            [[[0, 0, 0]]],
            DINOv2ScoringFailureCode.DINO_IMAGE_TYPE_INVALID,
        ),
        (
            np.zeros((512, 512), dtype=np.uint8),
            DINOv2ScoringFailureCode.DINO_IMAGE_SHAPE_INVALID,
        ),
        (
            np.zeros(INPUT_SHAPE, dtype=np.float32),
            DINOv2ScoringFailureCode.DINO_IMAGE_DTYPE_INVALID,
        ),
        (
            np.zeros((512, 512, 6), dtype=np.uint8)[:, :, ::2],
            DINOv2ScoringFailureCode.DINO_IMAGE_CONTIGUITY_INVALID,
        ),
    ],
)
def test_fixed_input_validation_rejects_each_boundary_violation(
    image: object,
    expected_code: DINOv2ScoringFailureCode,
) -> None:
    with pytest.raises(DINOv2ScoringError) as caught:
        validate_input_image(image)

    assert caught.value.code is expected_code


@pytest.mark.parametrize("reference_count", [True, 0, 1, 19, 21, 20.0, "20", None])
def test_reference_budget_rejects_any_count_other_than_twenty(
    reference_count: object,
) -> None:
    with pytest.raises(DINOv2ScoringError) as caught:
        validate_reference_count(reference_count)

    assert caught.value.code is DINOv2ScoringFailureCode.DINO_REFERENCE_COUNT_INVALID


def test_reference_budget_accepts_exactly_twenty() -> None:
    assert validate_reference_count(20) == 20
