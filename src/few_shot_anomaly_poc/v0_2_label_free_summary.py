"""Summarize committed v0.2.5 label-free evidence without using labels."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from few_shot_anomaly_poc.v0_2_boundary_preparation import RUN_ID
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    METHODS,
    load_v0_2_artifact_schema,
)
from few_shot_anomaly_poc.v0_2_scoring_artifacts import (
    SCORE_TOLERANCE,
    latency_summary,
    read_method_scoring_artifacts,
)

SUMMARY_SCHEMA = "v0.2.5-label-free-summary-v1"


class V0_2LabelFreeSummaryError(Exception):
    """Reject incomplete or cross-boundary label-free summary inputs."""


def _method_summary(
    *,
    method_root: Path,
    method: str,
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = read_method_scoring_artifacts(method_root, schema=schema)
    canonical = {record["asset_id"]: record for record in bundle.score_records}
    max_difference = 0.0
    for observation in bundle.latency_records:
        score = canonical[observation["asset_id"]]
        if (
            observation["method"] != method
            or observation["run_id"] != RUN_ID
            or observation["score_status"] != score["score_status"]
            or observation["score_failure_code"] != score["score_failure_code"]
        ):
            raise V0_2LabelFreeSummaryError("score repetition identity changed")
        difference = abs(float(observation["anomaly_score"]) - float(score["anomaly_score"]))
        if difference > SCORE_TOLERANCE:
            raise V0_2LabelFreeSummaryError("score repetition exceeds the fixed tolerance")
        max_difference = max(max_difference, difference)

    classification_by_asset = {
        record["asset_id"]: record for record in bundle.classification_records
    }
    for asset_id, score in canonical.items():
        classification = classification_by_asset[asset_id]
        if (
            classification["method"] != method
            or classification["run_id"] != RUN_ID
            or classification["score_status"] != score["score_status"]
            or classification["score_failure_code"] != score["score_failure_code"]
            or classification["anomaly_score"] != score["anomaly_score"]
        ):
            raise V0_2LabelFreeSummaryError("classification differs from its canonical score")

    latency = latency_summary(bundle.latency_records)
    scorer_durations = [int(record["scorer_duration_ns"]) for record in bundle.latency_records]
    adapter_durations = [
        int(record["adapter_duration_ns"])
        for record in bundle.latency_records
        if record["adapter_duration_ns"] is not None
    ]
    import statistics

    return {
        "score_count": len(bundle.score_records),
        "score_failure_count": sum(
            record["score_status"] == "failed" for record in bundle.score_records
        ),
        "predicted_anomalous_count": sum(
            record["is_anomalous"] for record in bundle.classification_records
        ),
        "predicted_normal_count": sum(
            not record["is_anomalous"] for record in bundle.classification_records
        ),
        "timed_observation_count": len(bundle.latency_records),
        "median_latency_ns": latency["median_latency_ns"],
        "p95_latency_ns": latency["p95_latency_ns"],
        "p95_rank": latency["p95_rank"],
        "median_scorer_duration_ns": float(statistics.median(scorer_durations)),
        "median_adapter_duration_ns": (
            float(statistics.median(adapter_durations)) if adapter_durations else None
        ),
        "max_score_repetition_absolute_difference": max_difference,
    }


def build_v0_2_label_free_summary(project_root: Path) -> dict[str, Any]:
    """Build a deterministic clone-only summary of the committed label-free CSV files."""
    project_root = project_root.resolve()
    artifact_root = project_root / "artifacts/v0.2/evaluation" / RUN_ID
    schema = load_v0_2_artifact_schema(
        project_root / "schemas/v0.2/evaluation-artifacts.json"
    )
    return {
        "schema_version": SUMMARY_SCHEMA,
        "run_id": RUN_ID,
        "boundary": {
            "labels_accessed": False,
            "metrics_computed": False,
            "failure_cases_selected": False,
            "decision_computed": False,
        },
        "methods": {
            method: _method_summary(
                method_root=artifact_root / method,
                method=method,
                schema=schema,
            )
            for method in METHODS
        },
    }
