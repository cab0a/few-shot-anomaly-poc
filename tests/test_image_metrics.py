from __future__ import annotations

import inspect
import math
from dataclasses import fields, replace
from pathlib import Path

import pytest

import few_shot_anomaly_poc.image_metrics as metrics_module
from few_shot_anomaly_poc.calibration import (
    CalibrationMethod,
    FixedThresholdClassificationResult,
)
from few_shot_anomaly_poc.config import ProjectConfig, load_config
from few_shot_anomaly_poc.errors import (
    ImageMetricsFailureCode,
    LabelRevealFailureCode,
)
from few_shot_anomaly_poc.image_metrics import (
    ImageLevelMetricsResult,
    calculate_image_level_metrics,
)
from few_shot_anomaly_poc.label_reveal import (
    FinalTestLabelRevealResult,
    LabeledFinalTestClassification,
)

THRESHOLD_SOURCE_PATH = "pcb1/Data/Images/Normal/0018.JPG"


@pytest.fixture(scope="module")
def project_config() -> ProjectConfig:
    return load_config(Path("configs/v0.1.yaml"))


def _revealed(
    labels: tuple[str, ...],
    scores: tuple[float, ...],
    *,
    method: CalibrationMethod = CalibrationMethod.ECC_RESIDUAL,
    failed_indices: frozenset[int] = frozenset(),
) -> FinalTestLabelRevealResult:
    assert len(labels) == len(scores)
    threshold = 0.5 if method is CalibrationMethod.ECC_RESIDUAL else 0.0
    paths = tuple(f"pcb1/Data/Images/Test/{index:04d}.JPG" for index in range(len(labels)))
    records = []
    for index, (path, label, score) in enumerate(zip(paths, labels, scores, strict=True)):
        failed = index in failed_indices
        is_anomalous = failed or score > threshold
        classification = FixedThresholdClassificationResult(
            status="ok",
            failure_code=None,
            method=method,
            relative_path=path,
            score_status="failed" if failed else "ok",
            score_failure_code="SYNTHETIC_SCORE_FAILURE" if failed else None,
            anomaly_score=score,
            threshold=threshold,
            threshold_source_path=THRESHOLD_SOURCE_PATH,
            calibration_sample_count=20,
            calibration_rank=19,
            predicted_class="anomalous" if is_anomalous else "normal",
            is_anomalous=is_anomalous,
            decision_reason=(
                "score_failure"
                if failed
                else ("score_above_threshold" if is_anomalous else "score_at_or_below_threshold")
            ),
            score_margin=score - threshold,
        )
        records.append(
            LabeledFinalTestClassification(
                relative_path=path,
                label=label,
                classification=classification,
            )
        )
    return FinalTestLabelRevealResult(
        status="ok",
        failure_code=None,
        method=method,
        batch_item_count=len(records),
        label_record_count=len(records),
        records=tuple(records),
        ordered_paths=paths,
        missing_paths=(),
        extra_paths=(),
        duplicate_path=None,
        invalid_label_index=None,
        order_mismatch_index=None,
        expected_path=None,
        observed_path=None,
    )


def test_metrics_interface_exposes_no_threshold_gate_or_decision_override() -> None:
    parameters = inspect.signature(calculate_image_level_metrics).parameters
    result_fields = {field.name for field in fields(ImageLevelMetricsResult)}

    assert tuple(parameters) == ("revealed", "config")
    assert "threshold" not in parameters
    assert "method" not in parameters
    assert {"gate", "passes_gate", "decision", "adoption_status"}.isdisjoint(result_fields)


@pytest.mark.parametrize(
    ("method", "scores"),
    [
        (CalibrationMethod.ECC_RESIDUAL, (0.1, 0.2, 0.8, 0.9)),
        (
            CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM,
            (-2.0, -1.0, 1.0, 2.0),
        ),
    ],
)
def test_metrics_reports_perfect_generated_example(
    project_config: ProjectConfig,
    method: CalibrationMethod,
    scores: tuple[float, ...],
) -> None:
    revealed = _revealed(
        ("normal", "normal", "anomaly", "anomaly"),
        scores,
        method=method,
    )

    result = calculate_image_level_metrics(
        revealed,
        config=project_config,
    )

    assert result.succeeded
    assert result.status == "ok"
    assert result.failure_code is None
    assert result.method is method
    assert result.positive_class == "anomaly"
    assert result.item_count == 4
    assert result.normal_count == 2
    assert result.anomaly_count == 2
    assert result.true_positive_count == 2
    assert result.false_negative_count == 0
    assert result.true_negative_count == 2
    assert result.false_positive_count == 0
    assert result.score_failure_count == 0
    assert result.image_level_auroc == 1.0
    assert result.image_level_auprc == 1.0
    assert result.normal_false_positive_rate == 0.0
    assert result.anomaly_recall == 1.0
    assert result.threshold == (0.5 if method is CalibrationMethod.ECC_RESIDUAL else 0.0)
    assert result.threshold_source_path == THRESHOLD_SOURCE_PATH


def test_metrics_reports_known_mixed_example(
    project_config: ProjectConfig,
) -> None:
    revealed = _revealed(
        ("normal", "normal", "anomaly", "anomaly"),
        (0.1, 0.7, 0.8, 0.4),
    )

    result = calculate_image_level_metrics(
        revealed,
        config=project_config,
    )

    assert result.succeeded
    assert result.true_positive_count == 1
    assert result.false_negative_count == 1
    assert result.true_negative_count == 1
    assert result.false_positive_count == 1
    assert result.image_level_auroc == pytest.approx(0.75)
    assert result.image_level_auprc == pytest.approx(5 / 6)
    assert result.normal_false_positive_rate == 0.5
    assert result.anomaly_recall == 0.5


def test_metrics_includes_failed_score_in_all_evidence(
    project_config: ProjectConfig,
) -> None:
    revealed = _revealed(
        ("normal", "normal", "anomaly", "anomaly"),
        (1.0, 0.1, 0.8, 0.7),
        failed_indices=frozenset({0}),
    )

    result = calculate_image_level_metrics(
        revealed,
        config=project_config,
    )

    assert result.succeeded
    assert result.item_count == 4
    assert result.normal_count == 2
    assert result.anomaly_count == 2
    assert result.score_failure_count == 1
    assert result.false_positive_count == 1
    assert result.true_negative_count == 1
    assert result.true_positive_count == 2
    assert result.false_negative_count == 0
    assert result.image_level_auroc == pytest.approx(0.5)
    assert result.image_level_auprc == pytest.approx(7 / 12)


def test_metrics_uses_standard_tie_handling(
    project_config: ProjectConfig,
) -> None:
    revealed = _revealed(
        ("normal", "normal", "anomaly", "anomaly"),
        (0.5, 0.5, 0.5, 0.5),
    )

    result = calculate_image_level_metrics(
        revealed,
        config=project_config,
    )

    assert result.succeeded
    assert result.image_level_auroc == 0.5
    assert result.image_level_auprc == 0.5
    assert result.normal_false_positive_rate == 0.0
    assert result.anomaly_recall == 0.0


def test_metrics_is_repeatable_and_does_not_mutate_input(
    project_config: ProjectConfig,
) -> None:
    revealed = _revealed(
        ("normal", "normal", "anomaly", "anomaly"),
        (0.1, 0.7, 0.8, 0.4),
    )

    first = calculate_image_level_metrics(revealed, config=project_config)
    second = calculate_image_level_metrics(revealed, config=project_config)

    assert first == second
    assert revealed == _revealed(
        ("normal", "normal", "anomaly", "anomaly"),
        (0.1, 0.7, 0.8, 0.4),
    )


@pytest.mark.parametrize(
    "labels",
    [
        ("normal", "normal"),
        ("anomaly", "anomaly"),
    ],
)
def test_metrics_rejects_missing_class_support(
    project_config: ProjectConfig,
    labels: tuple[str, ...],
) -> None:
    revealed = _revealed(labels, (0.1, 0.2))

    result = calculate_image_level_metrics(
        revealed,
        config=project_config,
    )

    assert not result.succeeded
    assert result.failure_code is ImageMetricsFailureCode.METRICS_CLASS_SUPPORT_INVALID
    assert result.item_count == 2
    assert result.normal_count is None
    assert result.anomaly_count is None
    assert result.image_level_auroc is None
    assert result.image_level_auprc is None
    assert result.normal_false_positive_rate is None
    assert result.anomaly_recall is None
    assert result.threshold is None


def test_metrics_rejects_failed_label_reveal(
    project_config: ProjectConfig,
) -> None:
    failed_reveal = replace(
        _revealed(("normal", "anomaly"), (0.1, 0.9)),
        status="LABEL_REVEAL_FAILED",
        failure_code=LabelRevealFailureCode.LABEL_REVEAL_ORDER_MISMATCH,
        records=None,
        ordered_paths=(),
        order_mismatch_index=0,
    )

    result = calculate_image_level_metrics(
        failed_reveal,
        config=project_config,
    )

    assert result.failure_code is ImageMetricsFailureCode.METRICS_LABEL_REVEAL_INVALID
    assert result.item_count == 2
    assert result.true_positive_count is None
    assert result.false_positive_count is None


def test_metrics_rejects_nonfinite_score_in_revealed_result(
    project_config: ProjectConfig,
) -> None:
    revealed = _revealed(("normal", "anomaly"), (0.1, 0.9))
    assert revealed.records is not None
    corrupted_classification = replace(
        revealed.records[1].classification,
        anomaly_score=math.nan,
    )
    corrupted_record = replace(
        revealed.records[1],
        classification=corrupted_classification,
    )
    corrupted = replace(
        revealed,
        records=(revealed.records[0], corrupted_record),
    )

    result = calculate_image_level_metrics(
        corrupted,
        config=project_config,
    )

    assert result.failure_code is ImageMetricsFailureCode.METRICS_LABEL_REVEAL_INVALID
    assert result.image_level_auroc is None


def test_metrics_converts_expected_library_failure_to_stable_code(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
) -> None:
    def fail_auroc(y_true, y_score):
        raise ValueError("synthetic metric failure")

    monkeypatch.setattr(metrics_module, "roc_auc_score", fail_auroc)

    result = calculate_image_level_metrics(
        _revealed(("normal", "anomaly"), (0.1, 0.9)),
        config=project_config,
    )

    assert result.failure_code is ImageMetricsFailureCode.METRICS_COMPUTATION_FAILED
    assert result.image_level_auroc is None
    assert result.true_positive_count is None


def test_metrics_rejects_nonfinite_library_result(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
) -> None:
    monkeypatch.setattr(metrics_module, "average_precision_score", lambda *_: math.nan)

    result = calculate_image_level_metrics(
        _revealed(("normal", "anomaly"), (0.1, 0.9)),
        config=project_config,
    )

    assert result.failure_code is ImageMetricsFailureCode.METRICS_RESULT_INVALID
    assert result.image_level_auroc is None
    assert result.image_level_auprc is None
