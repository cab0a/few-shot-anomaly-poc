from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from few_shot_anomaly_poc.dinov2_scoring import (  # noqa: E402
    EMBEDDING_DIMENSION,
    INPUT_SHAPE,
    REFERENCE_COUNT,
    expected_patch_count,
)
from few_shot_anomaly_poc.dinov2_timing import (  # noqa: E402
    build_memory_bank_one_at_a_time,
)


class _FixedTokenModel:
    def __init__(self, tokens: Any) -> None:
        self.training = False
        self.tokens = tokens
        self.calls = 0

    def get_intermediate_layers(self, tensor: Any, **_: object) -> tuple[Any]:
        self.calls += 1
        return (self.tokens,)


def test_memory_bank_copies_references_one_at_a_time() -> None:
    patch_count = expected_patch_count(224)
    tokens = torch.ones(
        (1, patch_count, EMBEDDING_DIMENSION),
        dtype=torch.float32,
    )
    model = _FixedTokenModel(tokens)
    copied: list[int] = []

    def copy_reference(index: int) -> np.ndarray:
        copied.append(index)
        return np.full(INPUT_SHAPE, index, dtype=np.uint8)

    bank = build_memory_bank_one_at_a_time(
        copy_reference=copy_reference,
        model=model,
        resolution=224,
        torch_module=torch,
    )

    assert copied == list(range(REFERENCE_COUNT))
    assert model.calls == REFERENCE_COUNT
    assert tuple(bank.features.shape) == (
        REFERENCE_COUNT * patch_count,
        EMBEDDING_DIMENSION,
    )
    assert bank.reference_count == REFERENCE_COUNT
