from __future__ import annotations

import hashlib

import numpy as np

from few_shot_anomaly_poc.dinov2_scoring_smoke import (
    _independent_numpy_result,
    generate_fixed_synthetic_images,
)


def test_fixed_synthetic_inputs_are_repeatable_and_hash_addressable() -> None:
    first_reference, first_query = generate_fixed_synthetic_images()
    second_reference, second_query = generate_fixed_synthetic_images()

    assert np.array_equal(first_reference, second_reference)
    assert np.array_equal(first_query, second_query)
    assert first_reference.shape == (512, 512, 3)
    assert first_query.shape == (512, 512, 3)
    assert first_reference.dtype == np.uint8
    assert first_query.dtype == np.uint8
    assert first_reference.flags.c_contiguous
    assert first_query.flags.c_contiguous
    assert hashlib.sha256(first_reference.tobytes()).hexdigest() == (
        "2177c217fa47c84eac86530410e60ebbf9e7c8ea35c9ffed8ce36d2ce172a550"
    )
    assert hashlib.sha256(first_query.tobytes()).hexdigest() == (
        "ea629e99f050dea9639f24dfc11e9c74f2f329f4c4fbee332cc268a201e5e8d2"
    )


def test_independent_numpy_check_uses_exact_cosine_distance_and_top_fraction() -> None:
    query = np.zeros((100, 2), dtype=np.float32)
    query[:, 0] = 1.0
    query[-1] = np.asarray([0.0, 1.0], dtype=np.float32)
    memory = np.asarray([[1.0, 0.0]], dtype=np.float32)

    distances, score = _independent_numpy_result(query, memory)

    assert distances.dtype == np.float32
    assert np.array_equal(
        distances,
        np.asarray([0.0] * 99 + [1.0], dtype=np.float32),
    )
    assert score == 1.0
