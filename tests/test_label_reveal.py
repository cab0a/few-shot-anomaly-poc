from __future__ import annotations

import inspect
from dataclasses import fields, replace
from pathlib import Path

import pytest

from few_shot_anomaly_poc.calibration import (
    CalibrationMethod,
    FixedThresholdBatchClassificationResult,
    FixedThresholdClassificationResult,
)
from few_shot_anomaly_poc.config import ProjectConfig, load_config
from few_shot_anomaly_poc.errors import (
    BatchClassificationFailureCode,
    LabelRevealFailureCode,
)
from few_shot_anomaly_poc.label_reveal import (
    FinalTestLabelRecord,
    FinalTestLabelRevealResult,
    reveal_final_test_labels,
)

PATHS = tuple(f"pcb1/Data/Images/Test/{index:04d}.JPG" for index in range(4))
THRESHOLD_SOURCE_PATH = "pcb1/Data/Images/Normal/0018.JPG"


@pytest.fixture(scope="module")
def project_config() -> ProjectConfig:
    return load_config(Path("configs/v0.1.yaml"))


def _classification(
    path: str,
    *,
    method: CalibrationMethod,
    score: float,
    threshold: float,
    failed: bool = False,
) -> FixedThresholdClassificationResult:
    is_anomalous = failed or score > threshold
    return FixedThresholdClassificationResult(
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


def _batch(
    method: CalibrationMethod = CalibrationMethod.ECC_RESIDUAL,
) -> FixedThresholdBatchClassificationResult:
    if method is CalibrationMethod.ECC_RESIDUAL:
        threshold = 0.5
        scores = (0.1, 0.5, 0.6, 1.0)
    else:
        threshold = 0.0
        scores = (-1.0, 0.0, 1.0, 1e12)
    classifications = tuple(
        _classification(
            path,
            method=method,
            score=score,
            threshold=threshold,
            failed=index == 3,
        )
        for index, (path, score) in enumerate(zip(PATHS, scores, strict=True))
    )
    return FixedThresholdBatchClassificationResult(
        status="ok",
        failure_code=None,
        method=method,
        item_count=4,
        successful_item_count=4,
        classifications=classifications,
        ordered_paths=PATHS,
        threshold=threshold,
        threshold_source_path=THRESHOLD_SOURCE_PATH,
        normal_count=2,
        normal_paths=PATHS[:2],
        anomalous_count=2,
        anomalous_paths=PATHS[2:],
        score_failure_count=1,
        score_failure_paths=(PATHS[3],),
        failed_path=None,
        item_failure_code=None,
    )


def _labels() -> tuple[FinalTestLabelRecord, ...]:
    return (
        FinalTestLabelRecord(PATHS[0], "normal"),
        FinalTestLabelRecord(PATHS[1], "normal"),
        FinalTestLabelRecord(PATHS[2], "anomaly"),
        FinalTestLabelRecord(PATHS[3], "anomaly"),
    )


def test_label_reveal_interface_has_no_metric_or_decision_override() -> None:
    parameters = inspect.signature(reveal_final_test_labels).parameters
    result_fields = {field.name for field in fields(FinalTestLabelRevealResult)}

    assert tuple(parameters) == ("batch", "label_records", "config")
    assert "threshold" not in parameters
    assert "method" not in parameters
    assert {
        "auroc",
        "auprc",
        "false_positive_count",
        "false_negative_count",
        "decision",
    }.isdisjoint(result_fields)


@pytest.mark.parametrize(
    "method",
    [
        CalibrationMethod.ECC_RESIDUAL,
        CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM,
    ],
)
def test_reveal_pairs_every_label_without_changing_classification(
    project_config: ProjectConfig,
    method: CalibrationMethod,
) -> None:
    batch = _batch(method)
    labels = _labels()

    result = reveal_final_test_labels(
        batch,
        labels,
        config=project_config,
    )

    assert result.succeeded
    assert result.status == "ok"
    assert result.failure_code is None
    assert result.method is method
    assert result.batch_item_count == 4
    assert result.label_record_count == 4
    assert result.records is not None
    assert result.ordered_paths == PATHS
    assert tuple(record.relative_path for record in result.records) == PATHS
    assert tuple(record.label for record in result.records) == (
        "normal",
        "normal",
        "anomaly",
        "anomaly",
    )
    assert all(
        labeled.classification is original
        for labeled, original in zip(
            result.records,
            batch.classifications,
            strict=True,
        )
    )
    assert not result.missing_paths
    assert not result.extra_paths
    assert result.duplicate_path is None
    assert result.invalid_label_index is None
    assert result.order_mismatch_index is None
    assert result.expected_path is None
    assert result.observed_path is None


def test_label_reveal_is_repeatable_and_does_not_mutate_inputs(
    project_config: ProjectConfig,
) -> None:
    batch = _batch()
    labels = _labels()

    first = reveal_final_test_labels(batch, labels, config=project_config)
    second = reveal_final_test_labels(batch, labels, config=project_config)

    assert first == second
    assert batch == _batch()
    assert labels == _labels()


def test_label_values_do_not_change_existing_decisions(
    project_config: ProjectConfig,
) -> None:
    batch = _batch()
    all_normal = tuple(FinalTestLabelRecord(path, "normal") for path in PATHS)

    result = reveal_final_test_labels(
        batch,
        all_normal,
        config=project_config,
    )

    assert result.succeeded
    assert result.records is not None
    assert tuple(record.label for record in result.records) == ("normal",) * 4
    assert tuple(record.classification for record in result.records) == batch.classifications


def test_label_reveal_rejects_failed_batch(
    project_config: ProjectConfig,
) -> None:
    failed_batch = replace(
        _batch(),
        status="BATCH_CLASSIFICATION_FAILED",
        failure_code=BatchClassificationFailureCode.BATCH_CLASSIFICATION_ITEM_FAILED,
        successful_item_count=2,
        classifications=None,
        ordered_paths=(),
        threshold=None,
        threshold_source_path=None,
        normal_count=None,
        normal_paths=(),
        anomalous_count=None,
        anomalous_paths=(),
        score_failure_count=None,
        score_failure_paths=(),
        failed_path=PATHS[2],
    )

    result = reveal_final_test_labels(
        failed_batch,
        _labels(),
        config=project_config,
    )

    assert not result.succeeded
    assert result.failure_code is LabelRevealFailureCode.LABEL_REVEAL_BATCH_INVALID
    assert result.records is None
    assert result.batch_item_count == 4
    assert result.label_record_count == 4


def test_label_reveal_rejects_internally_inconsistent_batch(
    project_config: ProjectConfig,
) -> None:
    batch = _batch()
    assert batch.classifications is not None
    corrupted = replace(
        batch.classifications[0],
        score_margin=999.0,
    )
    corrupted_batch = replace(
        batch,
        classifications=(corrupted, *batch.classifications[1:]),
    )

    result = reveal_final_test_labels(
        corrupted_batch,
        _labels(),
        config=project_config,
    )

    assert result.failure_code is LabelRevealFailureCode.LABEL_REVEAL_BATCH_INVALID
    assert result.records is None


def test_label_reveal_rejects_wrong_typed_batch_path_without_raising(
    project_config: ProjectConfig,
) -> None:
    corrupted_batch = replace(
        _batch(),
        ordered_paths=(PATHS[0], object(), *PATHS[2:]),
    )

    result = reveal_final_test_labels(
        corrupted_batch,
        _labels(),
        config=project_config,
    )

    assert result.failure_code is LabelRevealFailureCode.LABEL_REVEAL_BATCH_INVALID
    assert result.records is None


def test_label_reveal_rejects_empty_labels(
    project_config: ProjectConfig,
) -> None:
    result = reveal_final_test_labels(
        _batch(),
        (),
        config=project_config,
    )

    assert result.failure_code is LabelRevealFailureCode.LABEL_REVEAL_LABELS_EMPTY
    assert result.label_record_count == 0
    assert result.records is None


@pytest.mark.parametrize(
    "invalid_record",
    [
        object(),
        FinalTestLabelRecord("", "normal"),
        FinalTestLabelRecord("../escape.JPG", "normal"),
        FinalTestLabelRecord(PATHS[0], "defect"),
    ],
)
def test_label_reveal_rejects_invalid_label_record(
    project_config: ProjectConfig,
    invalid_record: object,
) -> None:
    labels: tuple[object, ...] = (_labels()[0], invalid_record, *_labels()[2:])

    result = reveal_final_test_labels(
        _batch(),
        labels,
        config=project_config,
    )

    assert result.failure_code is LabelRevealFailureCode.LABEL_REVEAL_LABEL_RECORD_INVALID
    assert result.invalid_label_index == 1
    assert result.records is None


def test_label_reveal_rejects_duplicate_path_before_set_comparison(
    project_config: ProjectConfig,
) -> None:
    labels = (*_labels(), FinalTestLabelRecord(PATHS[1], "normal"))

    result = reveal_final_test_labels(
        _batch(),
        labels,
        config=project_config,
    )

    assert result.failure_code is LabelRevealFailureCode.LABEL_REVEAL_LABEL_DUPLICATE_PATH
    assert result.duplicate_path == PATHS[1]
    assert result.records is None
    assert not result.missing_paths
    assert not result.extra_paths


def test_label_reveal_rejects_missing_path(
    project_config: ProjectConfig,
) -> None:
    result = reveal_final_test_labels(
        _batch(),
        _labels()[:-1],
        config=project_config,
    )

    assert result.failure_code is LabelRevealFailureCode.LABEL_REVEAL_PATH_MISSING
    assert result.missing_paths == (PATHS[3],)
    assert not result.extra_paths
    assert result.records is None


def test_label_reveal_rejects_extra_path(
    project_config: ProjectConfig,
) -> None:
    extra_path = "pcb1/Data/Images/Test/9999.JPG"
    labels = (*_labels(), FinalTestLabelRecord(extra_path, "normal"))

    result = reveal_final_test_labels(
        _batch(),
        labels,
        config=project_config,
    )

    assert result.failure_code is LabelRevealFailureCode.LABEL_REVEAL_PATH_EXTRA
    assert not result.missing_paths
    assert result.extra_paths == (extra_path,)
    assert result.records is None


def test_label_reveal_reports_missing_and_extra_paths_together(
    project_config: ProjectConfig,
) -> None:
    extra_path = "pcb1/Data/Images/Test/9999.JPG"
    labels = (*_labels()[:-1], FinalTestLabelRecord(extra_path, "normal"))

    result = reveal_final_test_labels(
        _batch(),
        labels,
        config=project_config,
    )

    assert result.failure_code is LabelRevealFailureCode.LABEL_REVEAL_PATH_MISSING
    assert result.missing_paths == (PATHS[3],)
    assert result.extra_paths == (extra_path,)
    assert result.records is None


def test_label_reveal_rejects_order_mismatch_after_exact_set_match(
    project_config: ProjectConfig,
) -> None:
    labels = (_labels()[1], _labels()[0], *_labels()[2:])

    result = reveal_final_test_labels(
        _batch(),
        labels,
        config=project_config,
    )

    assert result.failure_code is LabelRevealFailureCode.LABEL_REVEAL_ORDER_MISMATCH
    assert result.order_mismatch_index == 0
    assert result.expected_path == PATHS[0]
    assert result.observed_path == PATHS[1]
    assert result.records is None
