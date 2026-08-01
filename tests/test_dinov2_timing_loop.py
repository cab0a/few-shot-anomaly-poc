from __future__ import annotations

from collections.abc import Callable

import numpy as np

from few_shot_anomaly_poc.dinov2_scoring import INPUT_SHAPE
from few_shot_anomaly_poc.dinov2_timing import (
    LATENCY_GATE_NS,
    QUERY_COUNT,
    TIMED_INVOCATION_COUNT,
    WARMUP_COUNT,
    execute_fixed_timing_loop,
    peak_rss_bytes,
    summarize_timing_observations,
)


def _clock(step: int) -> Callable[[], int]:
    value = -step

    def tick() -> int:
        nonlocal value
        value += step
        return value

    return tick


def test_fixed_loop_executes_exact_warmup_and_three_query_passes() -> None:
    copied: list[int] = []
    scored = 0

    def copy_query(index: int) -> np.ndarray:
        copied.append(index)
        return np.full(INPUT_SHAPE, index % 256, dtype=np.uint8)

    def score_image(image: np.ndarray) -> float:
        nonlocal scored
        scored += 1
        return float(image[0, 0, 0]) / 255.0

    result = execute_fixed_timing_loop(
        copy_query=copy_query,
        score_image=score_image,
        clock_ns=_clock(10),
    )

    assert copied[:WARMUP_COUNT] == [0] * WARMUP_COUNT
    assert copied[WARMUP_COUNT:] == list(range(QUERY_COUNT)) * 3
    assert scored == WARMUP_COUNT + TIMED_INVOCATION_COUNT
    assert result["warmup"] == {
        "completed_count": 25,
        "failure": None,
        "query_id": "synthetic/query/000",
        "required_count": 25,
        "status": "pass",
    }
    assert [item["invocation_index"] for item in result["observations"]] == list(
        range(TIMED_INVOCATION_COUNT)
    )
    assert [item["asset_id"] for item in result["observations"][:100]] == [
        f"synthetic/query/{index:03d}" for index in range(100)
    ]
    assert result["summary"] == {
        "attempted_invocation_count": 300,
        "complete_observation_set": True,
        "failure_count": 0,
        "latency_gate_ns": LATENCY_GATE_NS,
        "latency_gate_passed": True,
        "median_ns": 10.0,
        "missing_invocation_count": 0,
        "p95_method": "nearest-rank",
        "p95_nearest_rank": 285,
        "p95_ns": 10,
        "successful_invocation_count": 300,
        "timed_invocation_count_required": 300,
    }


def test_summary_uses_middle_pair_and_nearest_rank_285() -> None:
    observations = [
        {
            "duration_ns": duration,
            "status": "success",
        }
        for duration in range(1, 301)
    ]

    result = summarize_timing_observations(observations)

    assert result["median_ns"] == 150.5
    assert result["p95_ns"] == 285
    assert result["latency_gate_passed"] is True


def test_timed_memory_failure_is_recorded_once_without_retry() -> None:
    scored = 0

    def copy_query(index: int) -> np.ndarray:
        return np.full(INPUT_SHAPE, index % 256, dtype=np.uint8)

    def score_image(image: np.ndarray) -> float:
        nonlocal scored
        scored += 1
        if scored == WARMUP_COUNT + 8:
            raise MemoryError("synthetic test failure")
        return float(image[0, 0, 0]) / 255.0

    result = execute_fixed_timing_loop(
        copy_query=copy_query,
        score_image=score_image,
        clock_ns=_clock(10),
    )

    assert scored == WARMUP_COUNT + 8
    assert len(result["observations"]) == 8
    assert result["observations"][-1]["status"] == "failure"
    assert result["observations"][-1]["failure"]["category"] == "memory_error"
    assert result["summary"]["attempted_invocation_count"] == 8
    assert result["summary"]["successful_invocation_count"] == 7
    assert result["summary"]["failure_count"] == 1
    assert result["summary"]["missing_invocation_count"] == 292
    assert result["summary"]["p95_ns"] is None
    assert result["summary"]["latency_gate_passed"] is False


def test_warmup_failure_prevents_timed_invocations() -> None:
    def fail(_: np.ndarray) -> float:
        raise RuntimeError("cannot allocate memory")

    result = execute_fixed_timing_loop(
        copy_query=lambda _: np.zeros(INPUT_SHAPE, dtype=np.uint8),
        score_image=fail,
    )

    assert result["warmup"]["completed_count"] == 0
    assert result["warmup"]["failure"]["category"] == "framework_out_of_memory"
    assert result["observations"] == []
    assert result["summary"]["missing_invocation_count"] == 300


def test_peak_rss_is_reported_as_non_negative_bytes() -> None:
    assert peak_rss_bytes() >= 0
