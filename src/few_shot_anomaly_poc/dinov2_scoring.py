"""Implement the preregistered v0.2 DINOv2 image-scoring path."""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from few_shot_anomaly_poc.dinov2_errors import (
    DINOv2ScoringError,
    DINOv2ScoringFailureCode,
)

INPUT_SHAPE = (512, 512, 3)
ALLOWED_RESOLUTIONS = (224, 448)
PATCH_SIZE = 14
EMBEDDING_DIMENSION = 384
REFERENCE_COUNT = 20
MEMORY_BLOCK_SIZE = 2_048
TOP_FRACTION = 0.01
L2_EPSILON = 1e-12
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STANDARD_DEVIATION = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DINOv2MemoryBank:
    """Hold one validated, fixed-resolution normal-reference memory bank."""

    features: Any
    resolution: int
    reference_count: int
    patch_count_per_reference: int
    embedding_dimension: int


def _raise(code: DINOv2ScoringFailureCode, message: str) -> None:
    raise DINOv2ScoringError(code, message)


def validate_resolution(resolution: object) -> int:
    """Return a preregistered square resolution or reject it."""
    if type(resolution) is not int or resolution not in ALLOWED_RESOLUTIONS:
        _raise(
            DINOv2ScoringFailureCode.DINO_RESOLUTION_INVALID,
            f"resolution must be one of {ALLOWED_RESOLUTIONS}",
        )
    return resolution


def expected_patch_count(resolution: object) -> int:
    """Return the fixed ViT-S/14 patch count for one accepted resolution."""
    validated_resolution = validate_resolution(resolution)
    return (validated_resolution // PATCH_SIZE) ** 2


def top_patch_count(patch_count: object) -> int:
    """Return max(1, floor(0.01 * patch_count)) without result-driven tuning."""
    if type(patch_count) is not int or patch_count <= 0:
        _raise(
            DINOv2ScoringFailureCode.DINO_SCORE_INVALID,
            "patch_count must be a positive integer",
        )
    return max(1, math.floor(TOP_FRACTION * patch_count))


def validate_input_image(image: object) -> NDArray[np.uint8]:
    """Require the fixed decoded RGB array boundary without copying it."""
    if not isinstance(image, np.ndarray):
        _raise(
            DINOv2ScoringFailureCode.DINO_IMAGE_TYPE_INVALID,
            "image must be a NumPy array",
        )
    if image.shape != INPUT_SHAPE:
        _raise(
            DINOv2ScoringFailureCode.DINO_IMAGE_SHAPE_INVALID,
            f"image shape must be {INPUT_SHAPE}",
        )
    if image.dtype != np.uint8:
        _raise(
            DINOv2ScoringFailureCode.DINO_IMAGE_DTYPE_INVALID,
            "image dtype must be uint8",
        )
    if not image.flags.c_contiguous:
        _raise(
            DINOv2ScoringFailureCode.DINO_IMAGE_CONTIGUITY_INVALID,
            "image must be C-contiguous",
        )
    return image


def validate_reference_count(reference_count: object) -> int:
    """Require the preregistered normal-reference budget."""
    if type(reference_count) is not int or reference_count != REFERENCE_COUNT:
        _raise(
            DINOv2ScoringFailureCode.DINO_REFERENCE_COUNT_INVALID,
            f"reference count must be exactly {REFERENCE_COUNT}",
        )
    return reference_count


def _load_torch(torch_module: ModuleType | None) -> ModuleType:
    if torch_module is not None:
        return torch_module
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise DINOv2ScoringError(
            DINOv2ScoringFailureCode.DINO_TORCH_UNAVAILABLE,
            "PyTorch is available only in the isolated v0.2 environment",
        ) from error


def _tensor_is_finite(torch: ModuleType, tensor: Any) -> bool:
    try:
        return bool(torch.isfinite(tensor).all().item())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _validate_feature_tensor(
    torch: ModuleType,
    features: object,
    *,
    expected_rows: int,
    failure_code: DINOv2ScoringFailureCode,
    verify_values: bool,
) -> Any:
    if (
        not isinstance(features, torch.Tensor)
        or features.layout is not torch.strided
        or features.device.type != "cpu"
        or features.dtype is not torch.float32
        or tuple(features.shape) != (expected_rows, EMBEDDING_DIMENSION)
    ):
        _raise(
            failure_code,
            "feature tensor must be a CPU float32 strided matrix with the fixed shape",
        )
    if verify_values:
        if not _tensor_is_finite(torch, features):
            _raise(failure_code, "feature tensor must contain only finite values")
        try:
            norms = torch.linalg.vector_norm(features, ord=2, dim=1)
            normalized = torch.allclose(
                norms,
                torch.ones_like(norms),
                rtol=1e-5,
                atol=1e-6,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise DINOv2ScoringError(
                failure_code,
                "feature normalization validation failed",
            ) from error
        if not normalized:
            _raise(failure_code, "every feature vector must have unit L2 norm")
    return features


def preprocess_dinov2_image(
    image: object,
    *,
    resolution: int,
    torch_module: ModuleType | None = None,
) -> Any:
    """Convert one fixed RGB uint8 array to the normalized DINOv2 input tensor."""
    validated_image = validate_input_image(image)
    validated_resolution = validate_resolution(resolution)
    torch = _load_torch(torch_module)
    try:
        tensor = (
            torch.from_numpy(validated_image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(dtype=torch.float32)
            .div(255.0)
        )
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(validated_resolution, validated_resolution),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        standard_deviation = torch.tensor(
            IMAGENET_STANDARD_DEVIATION,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        tensor = tensor.sub(mean).div(standard_deviation).contiguous()
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise DINOv2ScoringError(
            DINOv2ScoringFailureCode.DINO_PREPROCESSING_FAILED,
            "fixed DINOv2 preprocessing failed",
        ) from error

    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.layout is not torch.strided
        or tensor.device.type != "cpu"
        or tensor.dtype is not torch.float32
        or tuple(tensor.shape)
        != (1, 3, validated_resolution, validated_resolution)
        or not tensor.is_contiguous()
        or not _tensor_is_finite(torch, tensor)
    ):
        _raise(
            DINOv2ScoringFailureCode.DINO_PREPROCESSING_RESULT_INVALID,
            "preprocessed image tensor is outside the fixed CPU float32 boundary",
        )
    return tensor


def extract_dinov2_patch_features(
    image: object,
    *,
    model: object,
    resolution: int,
    torch_module: ModuleType | None = None,
) -> Any:
    """Extract and L2-normalize final-layer non-class patch tokens."""
    validated_resolution = validate_resolution(resolution)
    torch = _load_torch(torch_module)
    if (
        getattr(model, "training", None) is not False
        or not callable(getattr(model, "get_intermediate_layers", None))
    ):
        _raise(
            DINOv2ScoringFailureCode.DINO_MODEL_STATE_INVALID,
            "model must be in eval mode and expose get_intermediate_layers",
        )

    try:
        with torch.inference_mode():
            tensor = preprocess_dinov2_image(
                image,
                resolution=validated_resolution,
                torch_module=torch,
            )
            outputs = model.get_intermediate_layers(
                tensor,
                n=1,
                reshape=False,
                return_class_token=False,
                norm=True,
            )
            if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
                _raise(
                    DINOv2ScoringFailureCode.DINO_FEATURE_RESULT_INVALID,
                    "model must return exactly one final-layer token tensor",
                )
            tokens = outputs[0]
            expected_rows = expected_patch_count(validated_resolution)
            if (
                not isinstance(tokens, torch.Tensor)
                or tuple(tokens.shape)
                != (1, expected_rows, EMBEDDING_DIMENSION)
            ):
                _raise(
                    DINOv2ScoringFailureCode.DINO_FEATURE_RESULT_INVALID,
                    "final-layer token tensor has an unexpected shape",
                )
            features = torch.nn.functional.normalize(
                tokens[0],
                p=2,
                dim=1,
                eps=L2_EPSILON,
            ).contiguous()
    except DINOv2ScoringError:
        raise
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise DINOv2ScoringError(
            DINOv2ScoringFailureCode.DINO_FEATURE_EXTRACTION_FAILED,
            "DINOv2 patch feature extraction failed",
        ) from error

    return _validate_feature_tensor(
        torch,
        features,
        expected_rows=expected_patch_count(validated_resolution),
        failure_code=DINOv2ScoringFailureCode.DINO_FEATURE_RESULT_INVALID,
        verify_values=True,
    )


def create_dinov2_memory_bank(
    features: object,
    *,
    resolution: int,
    reference_count: int,
    torch_module: ModuleType | None = None,
) -> DINOv2MemoryBank:
    """Validate concatenated normalized features for exactly 20 references."""
    validated_resolution = validate_resolution(resolution)
    validated_reference_count = validate_reference_count(reference_count)
    patch_count = expected_patch_count(validated_resolution)
    torch = _load_torch(torch_module)
    validated_features = _validate_feature_tensor(
        torch,
        features,
        expected_rows=patch_count * validated_reference_count,
        failure_code=DINOv2ScoringFailureCode.DINO_MEMORY_BANK_INVALID,
        verify_values=True,
    )
    return DINOv2MemoryBank(
        features=validated_features.detach().contiguous(),
        resolution=validated_resolution,
        reference_count=validated_reference_count,
        patch_count_per_reference=patch_count,
        embedding_dimension=EMBEDDING_DIMENSION,
    )


def build_dinov2_memory_bank(
    reference_images: Sequence[object],
    *,
    model: object,
    resolution: int,
    torch_module: ModuleType | None = None,
) -> DINOv2MemoryBank:
    """Extract and concatenate the fixed set of 20 normal-reference images."""
    if not isinstance(reference_images, Sequence):
        _raise(
            DINOv2ScoringFailureCode.DINO_REFERENCE_COUNT_INVALID,
            "reference_images must be a sized sequence",
        )
    reference_count = validate_reference_count(len(reference_images))
    torch = _load_torch(torch_module)
    features = [
        extract_dinov2_patch_features(
            image,
            model=model,
            resolution=resolution,
            torch_module=torch,
        )
        for image in reference_images
    ]
    try:
        concatenated = torch.cat(features, dim=0)
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise DINOv2ScoringError(
            DINOv2ScoringFailureCode.DINO_MEMORY_BANK_INVALID,
            "normal-reference feature concatenation failed",
        ) from error
    return create_dinov2_memory_bank(
        concatenated,
        resolution=resolution,
        reference_count=reference_count,
        torch_module=torch,
    )


def exact_cosine_min_distances(
    query_features: object,
    *,
    memory_bank: DINOv2MemoryBank,
    torch_module: ModuleType | None = None,
) -> Any:
    """Return exact nearest-neighbor cosine distance for every query patch."""
    torch = _load_torch(torch_module)
    if not isinstance(memory_bank, DINOv2MemoryBank):
        _raise(
            DINOv2ScoringFailureCode.DINO_MEMORY_BANK_INVALID,
            "a validated DINOv2MemoryBank is required",
        )
    patch_count = expected_patch_count(memory_bank.resolution)
    if (
        memory_bank.reference_count != REFERENCE_COUNT
        or memory_bank.patch_count_per_reference != patch_count
        or memory_bank.embedding_dimension != EMBEDDING_DIMENSION
    ):
        _raise(
            DINOv2ScoringFailureCode.DINO_MEMORY_BANK_INVALID,
            "memory-bank metadata differs from the fixed scoring contract",
        )
    query = _validate_feature_tensor(
        torch,
        query_features,
        expected_rows=patch_count,
        failure_code=DINOv2ScoringFailureCode.DINO_FEATURE_RESULT_INVALID,
        verify_values=True,
    )
    memory = _validate_feature_tensor(
        torch,
        memory_bank.features,
        expected_rows=patch_count * REFERENCE_COUNT,
        failure_code=DINOv2ScoringFailureCode.DINO_MEMORY_BANK_INVALID,
        verify_values=False,
    )

    try:
        minimum_distances = torch.full(
            (patch_count,),
            2.0,
            dtype=torch.float32,
            device="cpu",
        )
        for start in range(0, memory.shape[0], MEMORY_BLOCK_SIZE):
            block = memory[start : start + MEMORY_BLOCK_SIZE]
            distances = torch.clamp(1.0 - query @ block.T, min=0.0, max=2.0)
            block_minimum = torch.min(distances, dim=1).values
            minimum_distances = torch.minimum(minimum_distances, block_minimum)
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise DINOv2ScoringError(
            DINOv2ScoringFailureCode.DINO_DISTANCE_COMPUTATION_FAILED,
            "exact blocked cosine-distance computation failed",
        ) from error

    if (
        not isinstance(minimum_distances, torch.Tensor)
        or minimum_distances.device.type != "cpu"
        or minimum_distances.dtype is not torch.float32
        or tuple(minimum_distances.shape) != (patch_count,)
        or not _tensor_is_finite(torch, minimum_distances)
        or not bool(
            torch.logical_and(
                minimum_distances >= 0.0,
                minimum_distances <= 2.0,
            )
            .all()
            .item()
        )
    ):
        _raise(
            DINOv2ScoringFailureCode.DINO_DISTANCE_RESULT_INVALID,
            "patch distances must be finite float32 values in [0, 2]",
        )
    return minimum_distances


def aggregate_top_fraction_score(
    patch_distances: object,
    *,
    torch_module: ModuleType | None = None,
) -> float:
    """Sort descending and average the fixed top one percent of patch distances."""
    torch = _load_torch(torch_module)
    if (
        not isinstance(patch_distances, torch.Tensor)
        or patch_distances.layout is not torch.strided
        or patch_distances.device.type != "cpu"
        or patch_distances.dtype is not torch.float32
        or patch_distances.ndim != 1
        or patch_distances.numel() <= 0
        or not _tensor_is_finite(torch, patch_distances)
    ):
        _raise(
            DINOv2ScoringFailureCode.DINO_DISTANCE_RESULT_INVALID,
            "patch distances must be a non-empty finite CPU float32 vector",
        )
    count = top_patch_count(patch_distances.numel())
    try:
        sorted_distances = torch.sort(
            patch_distances,
            descending=True,
            stable=True,
        ).values
        score = float(sorted_distances[:count].mean().item())
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise DINOv2ScoringError(
            DINOv2ScoringFailureCode.DINO_SCORE_INVALID,
            "top-one-percent score aggregation failed",
        ) from error
    if not math.isfinite(score) or score < 0.0 or score > 2.0:
        _raise(
            DINOv2ScoringFailureCode.DINO_SCORE_INVALID,
            "image score must be finite and within [0, 2]",
        )
    return score


def score_dinov2_patch_features(
    query_features: object,
    *,
    memory_bank: DINOv2MemoryBank,
    torch_module: ModuleType | None = None,
) -> float:
    """Return the fixed image score from normalized query patch features."""
    distances = exact_cosine_min_distances(
        query_features,
        memory_bank=memory_bank,
        torch_module=torch_module,
    )
    return aggregate_top_fraction_score(
        distances,
        torch_module=torch_module,
    )


def score_dinov2_image(
    image: object,
    *,
    model: object,
    memory_bank: DINOv2MemoryBank,
    resolution: int,
    torch_module: ModuleType | None = None,
) -> float:
    """Execute the fixed decoded-RGB-to-scalar-score path for one image."""
    validated_resolution = validate_resolution(resolution)
    if (
        not isinstance(memory_bank, DINOv2MemoryBank)
        or memory_bank.resolution != validated_resolution
    ):
        _raise(
            DINOv2ScoringFailureCode.DINO_MEMORY_BANK_INVALID,
            "memory-bank resolution must match the query resolution",
        )
    features = extract_dinov2_patch_features(
        image,
        model=model,
        resolution=validated_resolution,
        torch_module=torch_module,
    )
    return score_dinov2_patch_features(
        features,
        memory_bank=memory_bank,
        torch_module=torch_module,
    )
