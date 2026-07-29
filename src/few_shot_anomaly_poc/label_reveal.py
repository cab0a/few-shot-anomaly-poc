"""Join final-test labels only after label-free batch classification."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from few_shot_anomaly_poc.calibration import (
    CalibrationMethod,
    FixedThresholdBatchClassificationResult,
    FixedThresholdClassificationResult,
)
from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.errors import LabelRevealFailureCode, ManifestIntegrityError
from few_shot_anomaly_poc.manifests import normalize_relative_path

type FinalTestClassLabel = Literal["normal", "anomaly"]


@dataclass(frozen=True)
class FinalTestLabelRecord:
    """One ordered final-test path and its ground-truth class."""

    relative_path: str
    label: FinalTestClassLabel


@dataclass(frozen=True)
class LabeledFinalTestClassification:
    """One unchanged classification paired with its revealed label."""

    relative_path: str
    label: FinalTestClassLabel
    classification: FixedThresholdClassificationResult


@dataclass(frozen=True)
class FinalTestLabelRevealResult:
    """Complete result of the one-way final-test label reveal boundary."""

    status: Literal["ok", "LABEL_REVEAL_FAILED"]
    failure_code: LabelRevealFailureCode | None
    method: CalibrationMethod | None
    batch_item_count: int
    label_record_count: int
    records: tuple[LabeledFinalTestClassification, ...] | None
    ordered_paths: tuple[str, ...]
    missing_paths: tuple[str, ...]
    extra_paths: tuple[str, ...]
    duplicate_path: str | None
    invalid_label_index: int | None
    order_mismatch_index: int | None
    expected_path: str | None
    observed_path: str | None

    @property
    def succeeded(self) -> bool:
        """Return whether every classification received exactly one label."""
        return self.status == "ok"


def _relative_path_is_valid(path: object) -> bool:
    if not isinstance(path, str):
        return False
    try:
        return normalize_relative_path(path) == path
    except ManifestIntegrityError:
        return False


def _classification_is_valid(
    result: object,
    *,
    method: CalibrationMethod,
    relative_path: str,
    threshold: float,
    threshold_source_path: str,
    sample_count: int,
    rank: int,
    config: ProjectConfig,
) -> bool:
    if (
        not isinstance(result, FixedThresholdClassificationResult)
        or not result.succeeded
        or result.failure_code is not None
        or result.method is not method
        or result.relative_path != relative_path
        or result.score_status not in {"ok", "failed"}
        or not isinstance(result.anomaly_score, float)
        or not math.isfinite(result.anomaly_score)
        or result.threshold != threshold
        or result.threshold_source_path != threshold_source_path
        or result.calibration_sample_count != sample_count
        or result.calibration_rank != rank
        or not isinstance(result.score_margin, float)
        or not math.isfinite(result.score_margin)
        or result.score_margin != result.anomaly_score - threshold
    ):
        return False

    if method is CalibrationMethod.ECC_RESIDUAL:
        successful_score_is_valid = 0.0 <= result.anomaly_score <= 1.0
        failure_score = config.ecc_residual_scoring.failure_score
    else:
        limit = config.patch_hog_scoring.maximum_absolute_patch_score_exclusive
        successful_score_is_valid = abs(result.anomaly_score) < limit
        failure_score = config.patch_hog_scoring.failure_score

    if result.score_status == "failed":
        return (
            isinstance(result.score_failure_code, str)
            and bool(result.score_failure_code)
            and result.anomaly_score == failure_score
            and result.predicted_class == "anomalous"
            and result.is_anomalous is True
            and result.decision_reason == "score_failure"
        )

    expected_is_anomalous = result.anomaly_score > threshold
    return (
        result.score_failure_code is None
        and successful_score_is_valid
        and result.predicted_class == ("anomalous" if expected_is_anomalous else "normal")
        and result.is_anomalous is expected_is_anomalous
        and result.decision_reason
        == ("score_above_threshold" if expected_is_anomalous else "score_at_or_below_threshold")
    )


def _batch_is_valid(
    result: object,
    *,
    config: ProjectConfig,
) -> bool:
    if (
        not isinstance(result, FixedThresholdBatchClassificationResult)
        or not result.succeeded
        or result.failure_code is not None
        or not isinstance(result.method, CalibrationMethod)
        or not isinstance(result.item_count, int)
        or isinstance(result.item_count, bool)
        or result.item_count < 1
        or not isinstance(result.successful_item_count, int)
        or isinstance(result.successful_item_count, bool)
        or result.successful_item_count != result.item_count
        or not isinstance(result.classifications, tuple)
        or len(result.classifications) != result.item_count
        or not isinstance(result.ordered_paths, tuple)
        or len(result.ordered_paths) != result.item_count
        or any(not _relative_path_is_valid(path) for path in result.ordered_paths)
        or tuple(sorted(result.ordered_paths)) != result.ordered_paths
        or len(set(result.ordered_paths)) != result.item_count
        or not isinstance(result.threshold, float)
        or not math.isfinite(result.threshold)
        or not _relative_path_is_valid(result.threshold_source_path)
        or not isinstance(result.normal_count, int)
        or isinstance(result.normal_count, bool)
        or not isinstance(result.normal_paths, tuple)
        or result.normal_count != len(result.normal_paths)
        or not isinstance(result.anomalous_count, int)
        or isinstance(result.anomalous_count, bool)
        or not isinstance(result.anomalous_paths, tuple)
        or result.anomalous_count != len(result.anomalous_paths)
        or result.normal_count + result.anomalous_count != result.item_count
        or not isinstance(result.score_failure_count, int)
        or isinstance(result.score_failure_count, bool)
        or not isinstance(result.score_failure_paths, tuple)
        or result.score_failure_count != len(result.score_failure_paths)
        or result.failed_path is not None
        or result.item_failure_code is not None
        or any(
            not isinstance(item, FixedThresholdClassificationResult)
            for item in result.classifications
        )
    ):
        return False

    if any(
        not isinstance(item.calibration_sample_count, int)
        or isinstance(item.calibration_sample_count, bool)
        or not isinstance(item.calibration_rank, int)
        or isinstance(item.calibration_rank, bool)
        for item in result.classifications
    ):
        return False
    sample_counts = {item.calibration_sample_count for item in result.classifications}
    ranks = {item.calibration_rank for item in result.classifications}
    if len(sample_counts) != 1 or len(ranks) != 1:
        return False
    sample_count = next(iter(sample_counts))
    rank = next(iter(ranks))
    assert isinstance(sample_count, int)
    assert isinstance(rank, int)
    if (
        sample_count < 1
        or rank != math.ceil(config.threshold_calibration.quantile * sample_count)
        or rank < 1
        or rank > sample_count
    ):
        return False

    if result.method is CalibrationMethod.ECC_RESIDUAL:
        threshold_is_valid = 0.0 <= result.threshold <= config.ecc_residual_scoring.failure_score
    else:
        limit = config.patch_hog_scoring.maximum_absolute_patch_score_exclusive
        threshold_is_valid = -limit < result.threshold <= config.patch_hog_scoring.failure_score
    if not threshold_is_valid:
        return False

    if any(
        not _classification_is_valid(
            item,
            method=result.method,
            relative_path=path,
            threshold=result.threshold,
            threshold_source_path=result.threshold_source_path,
            sample_count=sample_count,
            rank=rank,
            config=config,
        )
        for path, item in zip(
            result.ordered_paths,
            result.classifications,
            strict=True,
        )
    ):
        return False

    normal_paths = tuple(
        item.relative_path for item in result.classifications if item.predicted_class == "normal"
    )
    anomalous_paths = tuple(
        item.relative_path for item in result.classifications if item.predicted_class == "anomalous"
    )
    score_failure_paths = tuple(
        item.relative_path for item in result.classifications if item.score_status == "failed"
    )
    return (
        result.normal_paths == normal_paths
        and result.anomalous_paths == anomalous_paths
        and result.score_failure_paths == score_failure_paths
        and set(score_failure_paths).issubset(anomalous_paths)
    )


def _failed(
    code: LabelRevealFailureCode,
    *,
    batch: object,
    label_record_count: int,
    missing_paths: tuple[str, ...] = (),
    extra_paths: tuple[str, ...] = (),
    duplicate_path: str | None = None,
    invalid_label_index: int | None = None,
    order_mismatch_index: int | None = None,
    expected_path: str | None = None,
    observed_path: str | None = None,
) -> FinalTestLabelRevealResult:
    method = (
        batch.method
        if (
            isinstance(batch, FixedThresholdBatchClassificationResult)
            and isinstance(batch.method, CalibrationMethod)
        )
        else None
    )
    batch_item_count = (
        batch.item_count
        if (
            isinstance(batch, FixedThresholdBatchClassificationResult)
            and isinstance(batch.item_count, int)
            and not isinstance(batch.item_count, bool)
            and batch.item_count >= 0
        )
        else 0
    )
    return FinalTestLabelRevealResult(
        status="LABEL_REVEAL_FAILED",
        failure_code=code,
        method=method,
        batch_item_count=batch_item_count,
        label_record_count=label_record_count,
        records=None,
        ordered_paths=(),
        missing_paths=missing_paths,
        extra_paths=extra_paths,
        duplicate_path=duplicate_path,
        invalid_label_index=invalid_label_index,
        order_mismatch_index=order_mismatch_index,
        expected_path=expected_path,
        observed_path=observed_path,
    )


def final_test_label_reveal_result_is_valid(
    result: object,
    *,
    config: ProjectConfig,
) -> bool:
    """Return whether a revealed result can enter final-test evaluation."""
    if (
        not isinstance(result, FinalTestLabelRevealResult)
        or not result.succeeded
        or result.failure_code is not None
        or not isinstance(result.method, CalibrationMethod)
        or not isinstance(result.batch_item_count, int)
        or isinstance(result.batch_item_count, bool)
        or result.batch_item_count < 1
        or not isinstance(result.label_record_count, int)
        or isinstance(result.label_record_count, bool)
        or result.label_record_count != result.batch_item_count
        or not isinstance(result.records, tuple)
        or len(result.records) != result.batch_item_count
        or not isinstance(result.ordered_paths, tuple)
        or len(result.ordered_paths) != result.batch_item_count
        or any(not _relative_path_is_valid(path) for path in result.ordered_paths)
        or tuple(sorted(result.ordered_paths)) != result.ordered_paths
        or len(set(result.ordered_paths)) != result.batch_item_count
        or result.missing_paths != ()
        or result.extra_paths != ()
        or result.duplicate_path is not None
        or result.invalid_label_index is not None
        or result.order_mismatch_index is not None
        or result.expected_path is not None
        or result.observed_path is not None
        or any(not isinstance(record, LabeledFinalTestClassification) for record in result.records)
    ):
        return False

    first_classification = result.records[0].classification
    if (
        not isinstance(first_classification, FixedThresholdClassificationResult)
        or not isinstance(first_classification.threshold, float)
        or not math.isfinite(first_classification.threshold)
        or not _relative_path_is_valid(first_classification.threshold_source_path)
        or not isinstance(first_classification.calibration_sample_count, int)
        or isinstance(first_classification.calibration_sample_count, bool)
        or first_classification.calibration_sample_count < 1
        or not isinstance(first_classification.calibration_rank, int)
        or isinstance(first_classification.calibration_rank, bool)
        or first_classification.calibration_rank
        != math.ceil(
            config.threshold_calibration.quantile * first_classification.calibration_sample_count
        )
    ):
        return False

    if result.method is CalibrationMethod.ECC_RESIDUAL:
        threshold_is_valid = (
            0.0 <= first_classification.threshold <= config.ecc_residual_scoring.failure_score
        )
    else:
        limit = config.patch_hog_scoring.maximum_absolute_patch_score_exclusive
        threshold_is_valid = (
            -limit < first_classification.threshold <= config.patch_hog_scoring.failure_score
        )
    if not threshold_is_valid:
        return False

    return all(
        record.relative_path == path
        and record.label in ("normal", "anomaly")
        and _classification_is_valid(
            record.classification,
            method=result.method,
            relative_path=path,
            threshold=first_classification.threshold,
            threshold_source_path=first_classification.threshold_source_path,
            sample_count=first_classification.calibration_sample_count,
            rank=first_classification.calibration_rank,
            config=config,
        )
        for path, record in zip(
            result.ordered_paths,
            result.records,
            strict=True,
        )
    )


def reveal_final_test_labels(
    batch: FixedThresholdBatchClassificationResult,
    label_records: Sequence[FinalTestLabelRecord],
    *,
    config: ProjectConfig,
) -> FinalTestLabelRevealResult:
    """Pair labels with a complete batch without recalculating decisions."""
    label_record_count = len(label_records)
    if not _batch_is_valid(batch, config=config):
        return _failed(
            LabelRevealFailureCode.LABEL_REVEAL_BATCH_INVALID,
            batch=batch,
            label_record_count=label_record_count,
        )
    if label_record_count == 0:
        return _failed(
            LabelRevealFailureCode.LABEL_REVEAL_LABELS_EMPTY,
            batch=batch,
            label_record_count=0,
        )

    for index, record in enumerate(label_records):
        if (
            not isinstance(record, FinalTestLabelRecord)
            or not _relative_path_is_valid(record.relative_path)
            or record.label not in ("normal", "anomaly")
        ):
            return _failed(
                LabelRevealFailureCode.LABEL_REVEAL_LABEL_RECORD_INVALID,
                batch=batch,
                label_record_count=label_record_count,
                invalid_label_index=index,
            )

    label_paths = tuple(record.relative_path for record in label_records)
    seen_paths: set[str] = set()
    for path in label_paths:
        if path in seen_paths:
            return _failed(
                LabelRevealFailureCode.LABEL_REVEAL_LABEL_DUPLICATE_PATH,
                batch=batch,
                label_record_count=label_record_count,
                duplicate_path=path,
            )
        seen_paths.add(path)

    batch_paths = batch.ordered_paths
    missing_paths = tuple(sorted(set(batch_paths) - seen_paths))
    extra_paths = tuple(sorted(seen_paths - set(batch_paths)))
    if missing_paths:
        return _failed(
            LabelRevealFailureCode.LABEL_REVEAL_PATH_MISSING,
            batch=batch,
            label_record_count=label_record_count,
            missing_paths=missing_paths,
            extra_paths=extra_paths,
        )
    if extra_paths:
        return _failed(
            LabelRevealFailureCode.LABEL_REVEAL_PATH_EXTRA,
            batch=batch,
            label_record_count=label_record_count,
            extra_paths=extra_paths,
        )

    if label_paths != batch_paths:
        mismatch_index = next(
            index
            for index, (expected, observed) in enumerate(zip(batch_paths, label_paths, strict=True))
            if expected != observed
        )
        return _failed(
            LabelRevealFailureCode.LABEL_REVEAL_ORDER_MISMATCH,
            batch=batch,
            label_record_count=label_record_count,
            order_mismatch_index=mismatch_index,
            expected_path=batch_paths[mismatch_index],
            observed_path=label_paths[mismatch_index],
        )

    assert batch.classifications is not None
    records = tuple(
        LabeledFinalTestClassification(
            relative_path=label.relative_path,
            label=label.label,
            classification=classification,
        )
        for label, classification in zip(
            label_records,
            batch.classifications,
            strict=True,
        )
    )
    return FinalTestLabelRevealResult(
        status="ok",
        failure_code=None,
        method=batch.method,
        batch_item_count=batch.item_count,
        label_record_count=label_record_count,
        records=records,
        ordered_paths=batch_paths,
        missing_paths=(),
        extra_paths=(),
        duplicate_path=None,
        invalid_label_index=None,
        order_mismatch_index=None,
        expected_path=None,
        observed_path=None,
    )
