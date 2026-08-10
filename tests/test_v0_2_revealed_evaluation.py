from __future__ import annotations

from pathlib import Path

import pytest

from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    METHODS,
    load_v0_2_artifact_schema,
    load_v0_2_config,
)
from few_shot_anomaly_poc.v0_2_revealed_evaluation import (
    ASSET_COUNT,
    V0_2RevealedEvaluationError,
    build_label_reveal_records,
    build_revealed_evaluation_artifacts,
    read_failure_cases_csv,
    read_label_reveal_csv,
    read_metrics_json,
    write_revealed_evaluation_artifacts,
)
from few_shot_anomaly_poc.v0_2_scoring_artifacts import MethodScoringArtifacts

ROOT = Path(__file__).resolve().parents[1]


def _contract() -> tuple[dict, dict]:
    return (
        load_v0_2_config(ROOT / "configs/v0.2.yaml"),
        load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json"),
    )


def _sealed_records() -> list[dict]:
    return [
        {
            "asset_id": f"asset-{index:06d}",
            "class_label": "normal" if index < 100 else "anomaly",
            "source_path": f"pcb2/example-{index:06d}.JPG",
        }
        for index in range(ASSET_COUNT)
    ]


def _scoring(method: str) -> MethodScoringArtifacts:
    scores = []
    classifications = []
    for index in range(ASSET_COUNT):
        asset_id = f"asset-{index:06d}"
        score = 0.9 if index % 10 == 0 else 0.1
        anomalous = score > 0.5
        scores.append(
            {
                "contract_version": "evaluation-artifacts/v0.2",
                "run_id": "visa-pcb2-v0-2-final",
                "run_kind": "final_test",
                "method": method,
                "asset_id": asset_id,
                "score_status": "ok",
                "score_failure_code": None,
                "anomaly_score": score,
                "diagnostics_json": "{}",
            }
        )
        classifications.append(
            {
                "contract_version": "evaluation-artifacts/v0.2",
                "run_id": "visa-pcb2-v0-2-final",
                "run_kind": "final_test",
                "method": method,
                "asset_id": asset_id,
                "score_status": "ok",
                "score_failure_code": None,
                "anomaly_score": score,
                "threshold": 0.5,
                "predicted_class": "anomalous" if anomalous else "normal",
                "is_anomalous": anomalous,
                "decision_reason": (
                    "score_strictly_greater_than_threshold"
                    if anomalous
                    else "score_not_greater_than_threshold"
                ),
                "score_margin": score - 0.5,
            }
        )
    return MethodScoringArtifacts(
        score_records=tuple(scores),
        classification_records=tuple(classifications),
        latency_records=(),
    )


def test_exact_reveal_builds_metrics_and_mechanical_failures() -> None:
    config, schema = _contract()
    labels = build_label_reveal_records(
        _sealed_records(),
        expected_asset_ids=[f"asset-{index:06d}" for index in range(ASSET_COUNT)],
        schema=schema,
    )

    artifacts = build_revealed_evaluation_artifacts(
        label_records=labels,
        scoring_by_method={method: _scoring(method) for method in METHODS},
        config=config,
        schema=schema,
    )

    assert len(artifacts.label_records) == ASSET_COUNT
    for method in METHODS:
        metrics = artifacts.metrics_by_method[method]
        failures = artifacts.failures_by_method[method]
        assert metrics["normal_count"] == metrics["anomaly_count"] == 100
        assert metrics["false_positive_count"] == 10
        assert metrics["false_negative_count"] == 90
        assert metrics["normal_false_positive_rate"] == 0.1
        assert metrics["anomaly_recall"] == 0.1
        assert len(failures) == 10
        assert [record["asset_id"] for record in failures[:5]] == [
            f"asset-{index:06d}" for index in (0, 10, 20, 30, 40)
        ]
        assert [record["asset_id"] for record in failures[5:]] == [
            f"asset-{index:06d}" for index in (101, 102, 103, 104, 105)
        ]


def test_reveal_rejects_missing_duplicate_and_out_of_order_ids() -> None:
    _, schema = _contract()
    expected = [f"asset-{index:06d}" for index in range(ASSET_COUNT)]
    missing = _sealed_records()[:-1]
    duplicate = _sealed_records()
    duplicate[1]["asset_id"] = duplicate[0]["asset_id"]
    out_of_order = _sealed_records()
    out_of_order[0], out_of_order[1] = out_of_order[1], out_of_order[0]

    for changed in (missing, duplicate, out_of_order):
        with pytest.raises(V0_2RevealedEvaluationError):
            build_label_reveal_records(
                changed,
                expected_asset_ids=expected,
                schema=schema,
            )


def test_revealed_artifacts_round_trip_without_image_content(tmp_path: Path) -> None:
    config, schema = _contract()
    labels = build_label_reveal_records(
        _sealed_records(),
        expected_asset_ids=[f"asset-{index:06d}" for index in range(ASSET_COUNT)],
        schema=schema,
    )
    artifacts = build_revealed_evaluation_artifacts(
        label_records=labels,
        scoring_by_method={method: _scoring(method) for method in METHODS},
        config=config,
        schema=schema,
    )
    output = tmp_path / "revealed"

    paths = write_revealed_evaluation_artifacts(output, artifacts)

    assert len(paths) == 7
    assert len(read_label_reveal_csv(output / "revealed-labels.csv", schema=schema)) == 200
    for method in METHODS:
        metrics = read_metrics_json(
            output / method / "metrics.json",
            config=config,
            schema=schema,
        )
        failures = read_failure_cases_csv(
            output / method / "failure-cases.csv",
            schema=schema,
        )
        assert metrics["item_count"] == 200
        assert len(failures) == 10
    assert {path.suffix for path in output.rglob("*") if path.is_file()} == {".csv", ".json"}
