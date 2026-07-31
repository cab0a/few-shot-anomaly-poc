from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from few_shot_anomaly_poc.dinov2_scoring import (  # noqa: E402
    EMBEDDING_DIMENSION,
    IMAGENET_MEAN,
    IMAGENET_STANDARD_DEVIATION,
    INPUT_SHAPE,
    MEMORY_BLOCK_SIZE,
    REFERENCE_COUNT,
    aggregate_top_fraction_score,
    build_dinov2_memory_bank,
    create_dinov2_memory_bank,
    exact_cosine_min_distances,
    expected_patch_count,
    extract_dinov2_patch_features,
    preprocess_dinov2_image,
    score_dinov2_image,
)
from few_shot_anomaly_poc.errors import (  # noqa: E402
    DINOv2ScoringError,
    DINOv2ScoringFailureCode,
)


class _FixedTokenModel:
    def __init__(self, tokens: Any) -> None:
        self.training = False
        self.tokens = tokens
        self.calls: list[dict[str, object]] = []

    def get_intermediate_layers(self, tensor: Any, **kwargs: object) -> tuple[Any]:
        self.calls.append({"input": tensor, **kwargs})
        return (self.tokens,)


def _unit_features(row_count: int, *, seed: int) -> Any:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    features = torch.randn(
        (row_count, EMBEDDING_DIMENSION),
        generator=generator,
        dtype=torch.float32,
    )
    return torch.nn.functional.normalize(features, p=2, dim=1, eps=1e-12)


@pytest.mark.parametrize("resolution", [224, 448])
def test_preprocessing_uses_fixed_cpu_float32_normalization(resolution: int) -> None:
    image = np.zeros(INPUT_SHAPE, dtype=np.uint8)

    result = preprocess_dinov2_image(
        image,
        resolution=resolution,
        torch_module=torch,
    )

    expected_channels = torch.tensor(
        [
            -IMAGENET_MEAN[index] / IMAGENET_STANDARD_DEVIATION[index]
            for index in range(3)
        ],
        dtype=torch.float32,
    )
    assert tuple(result.shape) == (1, 3, resolution, resolution)
    assert result.dtype is torch.float32
    assert result.device.type == "cpu"
    assert result.is_contiguous()
    assert torch.allclose(result[0, :, 0, 0], expected_channels)
    assert torch.allclose(result[0, :, -1, -1], expected_channels)


def test_feature_extraction_requests_only_fixed_final_patch_tokens() -> None:
    patch_count = expected_patch_count(224)
    tokens = torch.full(
        (1, patch_count, EMBEDDING_DIMENSION),
        2.0,
        dtype=torch.float32,
    )
    model = _FixedTokenModel(tokens)

    result = extract_dinov2_patch_features(
        np.zeros(INPUT_SHAPE, dtype=np.uint8),
        model=model,
        resolution=224,
        torch_module=torch,
    )

    assert tuple(result.shape) == (patch_count, EMBEDDING_DIMENSION)
    assert torch.allclose(
        torch.linalg.vector_norm(result, ord=2, dim=1),
        torch.ones(patch_count),
    )
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["n"] == 1
    assert call["reshape"] is False
    assert call["return_class_token"] is False
    assert call["norm"] is True
    assert tuple(call["input"].shape) == (1, 3, 224, 224)


def test_memory_bank_build_requires_and_concatenates_twenty_references() -> None:
    patch_count = expected_patch_count(224)
    tokens = _unit_features(patch_count, seed=11).unsqueeze(0)
    model = _FixedTokenModel(tokens)
    images = [np.zeros(INPUT_SHAPE, dtype=np.uint8) for _ in range(REFERENCE_COUNT)]

    result = build_dinov2_memory_bank(
        images,
        model=model,
        resolution=224,
        torch_module=torch,
    )

    assert result.resolution == 224
    assert result.reference_count == REFERENCE_COUNT
    assert result.patch_count_per_reference == patch_count
    assert result.embedding_dimension == EMBEDDING_DIMENSION
    assert tuple(result.features.shape) == (
        REFERENCE_COUNT * patch_count,
        EMBEDDING_DIMENSION,
    )
    assert len(model.calls) == REFERENCE_COUNT


def test_exact_distance_matches_unblocked_reference_across_three_blocks() -> None:
    patch_count = expected_patch_count(224)
    query = _unit_features(patch_count, seed=21)
    memory = _unit_features(REFERENCE_COUNT * patch_count, seed=22)
    assert memory.shape[0] > 2 * MEMORY_BLOCK_SIZE
    bank = create_dinov2_memory_bank(
        memory,
        resolution=224,
        reference_count=REFERENCE_COUNT,
        torch_module=torch,
    )

    result = exact_cosine_min_distances(
        query,
        memory_bank=bank,
        torch_module=torch,
    )
    expected = torch.clamp(1.0 - query @ memory.T, min=0.0, max=2.0).min(dim=1).values

    assert tuple(result.shape) == (patch_count,)
    assert torch.equal(result, expected)


def test_aggregation_sorts_descending_and_averages_fixed_top_two() -> None:
    distances = torch.arange(256, dtype=torch.float32) / 255.0

    result = aggregate_top_fraction_score(distances, torch_module=torch)

    assert result == pytest.approx((1.0 + 254.0 / 255.0) / 2.0)


def test_complete_image_path_matches_feature_level_score() -> None:
    patch_count = expected_patch_count(224)
    query = _unit_features(patch_count, seed=31)
    memory = _unit_features(REFERENCE_COUNT * patch_count, seed=32)
    model = _FixedTokenModel(query.unsqueeze(0))
    bank = create_dinov2_memory_bank(
        memory,
        resolution=224,
        reference_count=REFERENCE_COUNT,
        torch_module=torch,
    )
    expected_distances = torch.clamp(
        1.0 - query @ memory.T,
        min=0.0,
        max=2.0,
    ).min(dim=1).values
    expected_score = float(torch.sort(expected_distances, descending=True).values[:2].mean())

    result = score_dinov2_image(
        np.zeros(INPUT_SHAPE, dtype=np.uint8),
        model=model,
        memory_bank=bank,
        resolution=224,
        torch_module=torch,
    )

    assert result == expected_score


def test_memory_bank_rejects_non_normalized_features() -> None:
    patch_count = expected_patch_count(224)
    invalid = torch.ones(
        (REFERENCE_COUNT * patch_count, EMBEDDING_DIMENSION),
        dtype=torch.float32,
    )

    with pytest.raises(DINOv2ScoringError) as caught:
        create_dinov2_memory_bank(
            invalid,
            resolution=224,
            reference_count=REFERENCE_COUNT,
            torch_module=torch,
        )

    assert caught.value.code is DINOv2ScoringFailureCode.DINO_MEMORY_BANK_INVALID


def test_scoring_rejects_memory_bank_from_another_resolution() -> None:
    patch_count = expected_patch_count(224)
    memory = _unit_features(REFERENCE_COUNT * patch_count, seed=41)
    bank = create_dinov2_memory_bank(
        memory,
        resolution=224,
        reference_count=REFERENCE_COUNT,
        torch_module=torch,
    )
    model = _FixedTokenModel(_unit_features(patch_count, seed=42).unsqueeze(0))

    with pytest.raises(DINOv2ScoringError) as caught:
        score_dinov2_image(
            np.zeros(INPUT_SHAPE, dtype=np.uint8),
            model=model,
            memory_bank=bank,
            resolution=448,
            torch_module=torch,
        )

    assert caught.value.code is DINOv2ScoringFailureCode.DINO_MEMORY_BANK_INVALID
