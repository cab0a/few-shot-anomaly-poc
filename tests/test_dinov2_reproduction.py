from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from few_shot_anomaly_poc.dinov2_reproduction import (
    EXPECTED_CONFIGURATION_SHA256,
    REPRODUCTION_QUERY_COUNT,
    SCORE_TOLERANCE,
    TIMING_EXECUTION_COMMIT,
    configuration_sha256,
    execute_score_reproduction,
    fixed_reproduction_configuration,
    reproduction_worker_command,
    validate_timing_baseline,
)
from few_shot_anomaly_poc.dinov2_scoring import INPUT_SHAPE

ROOT = Path(__file__).resolve().parents[1]


def _expected_scores() -> list[dict[str, object]]:
    return [
        {
            "asset_id": f"synthetic/query/{index:03d}",
            "query_index": index,
            "score": float(index) / 100,
        }
        for index in range(REPRODUCTION_QUERY_COUNT)
    ]


def test_fixed_configuration_has_preregistered_identity() -> None:
    configuration = fixed_reproduction_configuration()

    assert configuration["execution"]["resolution"] == 224
    assert configuration["inputs"]["generator_seed"] == 42
    assert configuration["reproduction"] == {
        "absolute_score_tolerance": 1e-6,
        "query_count": 10,
        "timing_measurement": False,
    }
    assert configuration_sha256(configuration) == EXPECTED_CONFIGURATION_SHA256


def test_committed_timing_baseline_selects_only_first_ten_224_scores() -> None:
    baseline = validate_timing_baseline(ROOT)

    assert TIMING_EXECUTION_COMMIT == "d02da60f622090746c8348704e550dccf57358d5"
    assert baseline["source_revision"] == "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    assert [item["asset_id"] for item in baseline["expected_scores"]] == [
        f"synthetic/query/{index:03d}" for index in range(10)
    ]
    assert baseline["expected_scores"][0]["score"] == 0.11556071043014526
    assert baseline["expected_scores"][-1]["score"] == 0.08250007033348083


def test_score_reproduction_passes_at_fixed_tolerance_without_timing() -> None:
    expected = _expected_scores()
    copied: list[int] = []

    def copy_query(index: int) -> np.ndarray:
        copied.append(index)
        return np.full(INPUT_SHAPE, index, dtype=np.uint8)

    result = execute_score_reproduction(
        copy_query=copy_query,
        score_image=lambda image: float(image[0, 0, 0]) / 100 + SCORE_TOLERANCE / 2,
        expected_scores=expected,
    )

    assert copied == list(range(10))
    assert result["summary"]["status"] == "pass"
    assert result["summary"]["maximum_absolute_difference"] <= SCORE_TOLERANCE
    assert len(result["comparisons"]) == 10


def test_score_reproduction_preserves_numerical_mismatch() -> None:
    result = execute_score_reproduction(
        copy_query=lambda index: np.full(INPUT_SHAPE, index, dtype=np.uint8),
        score_image=lambda image: float(image[0, 0, 0]) / 100 + 2 * SCORE_TOLERANCE,
        expected_scores=_expected_scores(),
    )

    assert result["summary"]["status"] == "fail"
    assert result["summary"]["complete_observation_set"] is True
    assert result["summary"]["failure_count"] == 0
    assert all(not item["within_tolerance"] for item in result["comparisons"])


def test_score_reproduction_stops_once_without_retry_on_failure() -> None:
    copied: list[int] = []

    def copy_query(index: int) -> np.ndarray:
        copied.append(index)
        return np.full(INPUT_SHAPE, index, dtype=np.uint8)

    def score_image(image: np.ndarray) -> float:
        if int(image[0, 0, 0]) == 3:
            raise MemoryError("test failure")
        return float(image[0, 0, 0]) / 100

    result = execute_score_reproduction(
        copy_query=copy_query,
        score_image=score_image,
        expected_scores=_expected_scores(),
    )

    assert copied == [0, 1, 2, 3]
    assert result["summary"]["status"] == "fail"
    assert result["summary"]["attempted_count"] == 4
    assert result["summary"]["missing_count"] == 7
    assert result["failure"]["category"] == "memory_error"


def test_score_reproduction_rejects_nonfinite_score() -> None:
    result = execute_score_reproduction(
        copy_query=lambda index: np.full(INPUT_SHAPE, index, dtype=np.uint8),
        score_image=lambda _: math.inf,
        expected_scores=_expected_scores(),
    )

    assert result["summary"]["status"] == "fail"
    assert result["summary"]["attempted_count"] == 1
    assert result["comparisons"] == []


def test_worker_command_fixes_fresh_isolated_224_process() -> None:
    command = reproduction_worker_command(
        python_executable=Path(sys.executable),
        worker_script=ROOT / "scripts/run_v0_2_offline_reproduction_worker.py",
        project_root=ROOT,
        artifact_dir=ROOT / "data/external/v0.2/model-assets",
        source_root=ROOT / "data/external/v0.2/source",
        environment_root=ROOT / "environments/v0.2-preflight/.venv",
        execution_commit="a" * 40,
        verification_date="2026-08-01",
        output_root=ROOT / "work/v0.2-offline-reproduction/test",
    )

    assert command[1:3] == ["-I", "-B"]
    assert "--resolution" not in command
    assert command[command.index("--execution-commit") + 1] == "a" * 40
    assert command[command.index("--output-root") + 1].endswith(
        "work/v0.2-offline-reproduction/test"
    )
