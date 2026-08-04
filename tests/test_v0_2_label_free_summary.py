from __future__ import annotations

from pathlib import Path

from few_shot_anomaly_poc.v0_2_label_free_summary import build_v0_2_label_free_summary

ROOT = Path(__file__).resolve().parents[1]


def test_committed_label_free_summary_is_exact_and_keeps_labels_unread() -> None:
    summary = build_v0_2_label_free_summary(ROOT)

    assert summary == {
        "schema_version": "v0.2.5-label-free-summary-v1",
        "run_id": "visa-pcb2-v0-2-final",
        "boundary": {
            "labels_accessed": False,
            "metrics_computed": False,
            "failure_cases_selected": False,
            "decision_computed": False,
        },
        "methods": {
            "ecc_residual": {
                "score_count": 200,
                "score_failure_count": 0,
                "predicted_anomalous_count": 17,
                "predicted_normal_count": 183,
                "timed_observation_count": 600,
                "median_latency_ns": 290_899_561.5,
                "p95_latency_ns": 663_888_755,
                "p95_rank": 570,
                "median_scorer_duration_ns": 290_899_561.5,
                "median_adapter_duration_ns": None,
                "max_score_repetition_absolute_difference": 0.0,
            },
            "patch_hog_ocsvm": {
                "score_count": 200,
                "score_failure_count": 0,
                "predicted_anomalous_count": 21,
                "predicted_normal_count": 179,
                "timed_observation_count": 600,
                "median_latency_ns": 424_095_778.5,
                "p95_latency_ns": 545_039_527,
                "p95_rank": 570,
                "median_scorer_duration_ns": 424_095_778.5,
                "median_adapter_duration_ns": None,
                "max_score_repetition_absolute_difference": 0.0,
            },
            "dinov2_vits14_224_nn": {
                "score_count": 200,
                "score_failure_count": 0,
                "predicted_anomalous_count": 41,
                "predicted_normal_count": 159,
                "timed_observation_count": 600,
                "median_latency_ns": 388_141_686.0,
                "p95_latency_ns": 461_050_665,
                "p95_rank": 570,
                "median_scorer_duration_ns": 380_955_993.5,
                "median_adapter_duration_ns": 6_789_472.0,
                "max_score_repetition_absolute_difference": 0.0,
            },
        },
    }
