"""Calculate preregistered image-level metrics without applying decision gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from sklearn.metrics import average_precision_score, roc_auc_score

from few_shot_anomaly_poc.calibration import CalibrationMethod
from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.errors import ImageMetricsFailureCode
from few_shot_anomaly_poc.label_reveal import (
    FinalTestLabelRevealResult,
    final_test_label_reveal_result_is_valid,
)


@dataclass(frozen=True)
class ImageLevelMetricsResult:
    """Preregistered ranking and fixed-threshold image-level measurements."""

    status: Literal["ok", "METRICS_FAILED"]
    failure_code: ImageMetricsFailureCode | None
    method: CalibrationMethod | None
    positive_class: Literal["anomaly"]
    item_count: int
    normal_count: int | None
    anomaly_count: int | None
    true_positive_count: int | None
    false_negative_count: int | None
    true_negative_count: int | None
    false_positive_count: int | None
    score_failure_count: int | None
    image_level_auroc: float | None
    image_level_auprc: float | None
    normal_false_positive_rate: float | None
    anomaly_recall: float | None
    threshold: float | None
    threshold_source_path: str | None

    @property
    def succeeded(self) -> bool:
        """Return whether every preregistered image-level metric was calculated."""
        return self.status == "ok"


def _failed(
    code: ImageMetricsFailureCode,
    *,
    revealed: object,
) -> ImageLevelMetricsResult:
    method = (
        revealed.method
        if (
            isinstance(revealed, FinalTestLabelRevealResult)
            and isinstance(revealed.method, CalibrationMethod)
        )
        else None
    )
    item_count = (
        revealed.batch_item_count
        if (
            isinstance(revealed, FinalTestLabelRevealResult)
            and isinstance(revealed.batch_item_count, int)
            and not isinstance(revealed.batch_item_count, bool)
            and revealed.batch_item_count >= 0
        )
        else 0
    )
    return ImageLevelMetricsResult(
        status="METRICS_FAILED",
        failure_code=code,
        method=method,
        positive_class="anomaly",
        item_count=item_count,
        normal_count=None,
        anomaly_count=None,
        true_positive_count=None,
        false_negative_count=None,
        true_negative_count=None,
        false_positive_count=None,
        score_failure_count=None,
        image_level_auroc=None,
        image_level_auprc=None,
        normal_false_positive_rate=None,
        anomaly_recall=None,
        threshold=None,
        threshold_source_path=None,
    )


def image_level_metrics_result_is_valid(result: ImageLevelMetricsResult) -> bool:
    """Return whether a successful metric result is internally consistent."""
    counts = (
        result.normal_count,
        result.anomaly_count,
        result.true_positive_count,
        result.false_negative_count,
        result.true_negative_count,
        result.false_positive_count,
        result.score_failure_count,
    )
    metrics = (
        result.image_level_auroc,
        result.image_level_auprc,
        result.normal_false_positive_rate,
        result.anomaly_recall,
    )
    if (
        not result.succeeded
        or result.failure_code is not None
        or not isinstance(result.method, CalibrationMethod)
        or result.positive_class != "anomaly"
        or not isinstance(result.item_count, int)
        or isinstance(result.item_count, bool)
        or result.item_count < 2
        or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts
        )
        or any(
            not isinstance(metric, float) or not math.isfinite(metric) or not 0.0 <= metric <= 1.0
            for metric in metrics
        )
        or not isinstance(result.threshold, float)
        or not math.isfinite(result.threshold)
        or not isinstance(result.threshold_source_path, str)
        or not result.threshold_source_path
    ):
        return False

    assert result.normal_count is not None
    assert result.anomaly_count is not None
    assert result.true_positive_count is not None
    assert result.false_negative_count is not None
    assert result.true_negative_count is not None
    assert result.false_positive_count is not None
    assert result.score_failure_count is not None
    assert result.normal_false_positive_rate is not None
    assert result.anomaly_recall is not None
    return (
        result.normal_count > 0
        and result.anomaly_count > 0
        and result.normal_count + result.anomaly_count == result.item_count
        and result.true_positive_count + result.false_negative_count == result.anomaly_count
        and result.true_negative_count + result.false_positive_count == result.normal_count
        and result.score_failure_count <= result.item_count
        and result.normal_false_positive_rate == result.false_positive_count / result.normal_count
        and result.anomaly_recall == result.true_positive_count / result.anomaly_count
    )


def calculate_image_level_metrics(
    revealed: FinalTestLabelRevealResult,
    *,
    config: ProjectConfig,
) -> ImageLevelMetricsResult:
    """Calculate ranking and fixed-threshold metrics with anomaly as positive."""
    if not final_test_label_reveal_result_is_valid(
        revealed,
        config=config,
    ):
        return _failed(
            ImageMetricsFailureCode.METRICS_LABEL_REVEAL_INVALID,
            revealed=revealed,
        )
    assert revealed.records is not None

    normal_count = sum(record.label == "normal" for record in revealed.records)
    anomaly_count = sum(record.label == "anomaly" for record in revealed.records)
    if normal_count == 0 or anomaly_count == 0:
        return _failed(
            ImageMetricsFailureCode.METRICS_CLASS_SUPPORT_INVALID,
            revealed=revealed,
        )

    y_true = tuple(1 if record.label == "anomaly" else 0 for record in revealed.records)
    y_score = tuple(record.classification.anomaly_score for record in revealed.records)
    if any(not isinstance(score, float) or not math.isfinite(score) for score in y_score):
        return _failed(
            ImageMetricsFailureCode.METRICS_LABEL_REVEAL_INVALID,
            revealed=revealed,
        )

    try:
        image_level_auroc = float(roc_auc_score(y_true, y_score))
        image_level_auprc = float(average_precision_score(y_true, y_score))
    except (TypeError, ValueError, FloatingPointError):
        return _failed(
            ImageMetricsFailureCode.METRICS_COMPUTATION_FAILED,
            revealed=revealed,
        )

    true_positive_count = sum(
        record.label == "anomaly" and record.classification.is_anomalous is True
        for record in revealed.records
    )
    false_negative_count = sum(
        record.label == "anomaly" and record.classification.is_anomalous is False
        for record in revealed.records
    )
    true_negative_count = sum(
        record.label == "normal" and record.classification.is_anomalous is False
        for record in revealed.records
    )
    false_positive_count = sum(
        record.label == "normal" and record.classification.is_anomalous is True
        for record in revealed.records
    )
    score_failure_count = sum(
        record.classification.score_status == "failed" for record in revealed.records
    )
    first_classification = revealed.records[0].classification
    result = ImageLevelMetricsResult(
        status="ok",
        failure_code=None,
        method=revealed.method,
        positive_class="anomaly",
        item_count=revealed.batch_item_count,
        normal_count=normal_count,
        anomaly_count=anomaly_count,
        true_positive_count=true_positive_count,
        false_negative_count=false_negative_count,
        true_negative_count=true_negative_count,
        false_positive_count=false_positive_count,
        score_failure_count=score_failure_count,
        image_level_auroc=image_level_auroc,
        image_level_auprc=image_level_auprc,
        normal_false_positive_rate=false_positive_count / normal_count,
        anomaly_recall=true_positive_count / anomaly_count,
        threshold=first_classification.threshold,
        threshold_source_path=first_classification.threshold_source_path,
    )
    if not image_level_metrics_result_is_valid(result):
        return _failed(
            ImageMetricsFailureCode.METRICS_RESULT_INVALID,
            revealed=revealed,
        )
    return result
