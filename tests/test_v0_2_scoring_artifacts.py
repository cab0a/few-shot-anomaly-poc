from __future__ import annotations

import csv
from pathlib import Path

import pytest

from few_shot_anomaly_poc.v0_2_evaluation_contract import load_v0_2_artifact_schema
from few_shot_anomaly_poc.v0_2_scoring_artifacts import (
    ASSET_COUNT,
    ScoreEvidence,
    TimedScoreEvidence,
    V0_2ScoringArtifactError,
    build_method_scoring_artifacts,
    latency_summary,
    read_method_scoring_artifacts,
    write_method_scoring_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "visa-pcb2-v0-2-final"


def _scores() -> list[ScoreEvidence]:
    return [
        ScoreEvidence(
            asset_id=f"asset-{index:06d}",
            score_status="ok",
            score_failure_code=None,
            anomaly_score=0.1 + index / 1_000,
            diagnostics={"synthetic_index": index},
        )
        for index in range(ASSET_COUNT)
    ]


def _timed(*, dino: bool = False) -> list[TimedScoreEvidence]:
    return [
        TimedScoreEvidence(
            pass_index=pass_index,
            asset_id=f"asset-{index:06d}",
            adapter_duration_ns=10 + index if dino else None,
            scorer_duration_ns=100 + pass_index + index,
            score_status="ok",
            score_failure_code=None,
            anomaly_score=0.1 + index / 1_000,
        )
        for pass_index in range(3)
        for index in range(ASSET_COUNT)
    ]


def test_builds_complete_classical_label_free_bundle() -> None:
    schema = load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json")

    artifacts = build_method_scoring_artifacts(
        run_id=RUN_ID,
        method="ecc_residual",
        threshold=0.2,
        scores=_scores(),
        timed_scores=_timed(),
        schema=schema,
    )

    assert len(artifacts.score_records) == 200
    assert len(artifacts.classification_records) == 200
    assert len(artifacts.latency_records) == 600
    assert artifacts.classification_records[100]["anomaly_score"] == 0.2
    assert artifacts.classification_records[100]["is_anomalous"] is False
    assert artifacts.classification_records[101]["is_anomalous"] is True
    assert all(record["adapter_duration_ns"] is None for record in artifacts.latency_records)
    assert latency_summary(artifacts.latency_records)["p95_rank"] == 570


def test_builds_dinov2_latency_as_adapter_plus_scorer() -> None:
    schema = load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json")

    artifacts = build_method_scoring_artifacts(
        run_id=RUN_ID,
        method="dinov2_vits14_224_nn",
        threshold=0.3,
        scores=_scores(),
        timed_scores=_timed(dino=True),
        schema=schema,
    )

    first = artifacts.latency_records[0]
    assert first["duration_ns"] == first["adapter_duration_ns"] + first["scorer_duration_ns"]


def test_rejects_timed_score_that_differs_from_canonical() -> None:
    schema = load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json")
    timed = _timed()
    timed[-1] = TimedScoreEvidence(
        pass_index=2,
        asset_id="asset-000199",
        adapter_duration_ns=None,
        scorer_duration_ns=1,
        score_status="ok",
        score_failure_code=None,
        anomaly_score=0.5,
    )

    with pytest.raises(V0_2ScoringArtifactError, match="repetition"):
        build_method_scoring_artifacts(
            run_id=RUN_ID,
            method="ecc_residual",
            threshold=0.2,
            scores=_scores(),
            timed_scores=timed,
            schema=schema,
        )


def test_writes_fixed_csv_inventory_without_overwrite(tmp_path: Path) -> None:
    schema = load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json")
    artifacts = build_method_scoring_artifacts(
        run_id=RUN_ID,
        method="ecc_residual",
        threshold=0.2,
        scores=_scores(),
        timed_scores=_timed(),
        schema=schema,
    )
    method_root = tmp_path / "ecc_residual"

    paths = write_method_scoring_artifacts(method_root, artifacts)

    assert {path.name for path in paths} == {
        "scores.csv",
        "classifications.csv",
        "latency-observations.csv",
    }
    with (method_root / "classifications.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 200
    assert rows[0]["is_anomalous"] == "false"
    serialized = read_method_scoring_artifacts(method_root, schema=schema)
    assert serialized == artifacts
    with pytest.raises(FileExistsError):
        write_method_scoring_artifacts(method_root, artifacts)
