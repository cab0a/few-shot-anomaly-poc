"""Select fixed-threshold error cases without reading image content."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from few_shot_anomaly_poc.calibration import CalibrationMethod
from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.errors import FailureCaseSelectionFailureCode
from few_shot_anomaly_poc.label_reveal import (
    FinalTestClassLabel,
    FinalTestLabelRevealResult,
    final_test_label_reveal_result_is_valid,
)

type FailureCaseType = Literal["false_positive", "false_negative"]


@dataclass(frozen=True)
class SelectedFailureCase:
    """One mechanically ranked fixed-threshold classification error."""

    case_type: FailureCaseType
    rank: int
    relative_path: str
    true_class: FinalTestClassLabel
    predicted_class: Literal["normal", "anomalous"]
    anomaly_score: float
    threshold: float
    score_margin: float
    score_status: Literal["ok", "failed"]
    score_failure_code: str | None


@dataclass(frozen=True)
class FailureCaseSelectionResult:
    """Complete bounded false-positive and false-negative selections."""

    status: Literal["ok", "FAILURE_SELECTION_FAILED"]
    failure_code: FailureCaseSelectionFailureCode | None
    method: CalibrationMethod | None
    item_count: int
    max_cases_per_type: int
    false_positive_count: int | None
    false_negative_count: int | None
    selected_false_positives: tuple[SelectedFailureCase, ...] | None
    selected_false_negatives: tuple[SelectedFailureCase, ...] | None

    @property
    def succeeded(self) -> bool:
        """Return whether both bounded selections were produced."""
        return self.status == "ok"


def _failed(
    code: FailureCaseSelectionFailureCode,
    *,
    revealed: object,
    config: ProjectConfig,
) -> FailureCaseSelectionResult:
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
    return FailureCaseSelectionResult(
        status="FAILURE_SELECTION_FAILED",
        failure_code=code,
        method=method,
        item_count=item_count,
        max_cases_per_type=config.failure_case_selection.max_cases_per_type,
        false_positive_count=None,
        false_negative_count=None,
        selected_false_positives=None,
        selected_false_negatives=None,
    )


def _selected_case(
    record,
    *,
    case_type: FailureCaseType,
    rank: int,
) -> SelectedFailureCase:
    classification = record.classification
    assert classification.predicted_class is not None
    assert classification.anomaly_score is not None
    assert classification.threshold is not None
    assert classification.score_margin is not None
    assert classification.score_status is not None
    return SelectedFailureCase(
        case_type=case_type,
        rank=rank,
        relative_path=record.relative_path,
        true_class=record.label,
        predicted_class=classification.predicted_class,
        anomaly_score=classification.anomaly_score,
        threshold=classification.threshold,
        score_margin=classification.score_margin,
        score_status=classification.score_status,
        score_failure_code=classification.score_failure_code,
    )


def _result_is_valid(
    result: FailureCaseSelectionResult,
    *,
    false_positive_paths: tuple[str, ...],
    false_negative_paths: tuple[str, ...],
    config: ProjectConfig,
) -> bool:
    if (
        not result.succeeded
        or result.failure_code is not None
        or not isinstance(result.method, CalibrationMethod)
        or not isinstance(result.item_count, int)
        or isinstance(result.item_count, bool)
        or result.item_count < 1
        or result.max_cases_per_type != config.failure_case_selection.max_cases_per_type
        or result.false_positive_count != len(false_positive_paths)
        or result.false_negative_count != len(false_negative_paths)
        or not isinstance(result.selected_false_positives, tuple)
        or not isinstance(result.selected_false_negatives, tuple)
        or len(result.selected_false_positives)
        != min(len(false_positive_paths), result.max_cases_per_type)
        or len(result.selected_false_negatives)
        != min(len(false_negative_paths), result.max_cases_per_type)
        or tuple(item.relative_path for item in result.selected_false_positives)
        != false_positive_paths[: result.max_cases_per_type]
        or tuple(item.relative_path for item in result.selected_false_negatives)
        != false_negative_paths[: result.max_cases_per_type]
    ):
        return False

    for expected_type, selected in (
        ("false_positive", result.selected_false_positives),
        ("false_negative", result.selected_false_negatives),
    ):
        for rank, item in enumerate(selected, start=1):
            if (
                not isinstance(item, SelectedFailureCase)
                or item.case_type != expected_type
                or item.rank != rank
                or not isinstance(item.anomaly_score, float)
                or not math.isfinite(item.anomaly_score)
                or not isinstance(item.threshold, float)
                or not math.isfinite(item.threshold)
                or not isinstance(item.score_margin, float)
                or not math.isfinite(item.score_margin)
                or item.score_margin != item.anomaly_score - item.threshold
            ):
                return False
            if expected_type == "false_positive":
                if item.true_class != "normal" or item.predicted_class != "anomalous":
                    return False
            elif item.true_class != "anomaly" or item.predicted_class != "normal":
                return False
    return True


def select_failure_cases(
    revealed: FinalTestLabelRevealResult,
    *,
    config: ProjectConfig,
) -> FailureCaseSelectionResult:
    """Select up to five highest-score FPs and lowest-score FNs."""
    if not final_test_label_reveal_result_is_valid(
        revealed,
        config=config,
    ):
        return _failed(
            FailureCaseSelectionFailureCode.FAILURE_SELECTION_REVEAL_INVALID,
            revealed=revealed,
            config=config,
        )
    assert revealed.records is not None

    false_positives = tuple(
        sorted(
            (
                record
                for record in revealed.records
                if record.label == "normal" and record.classification.predicted_class == "anomalous"
            ),
            key=lambda record: (
                -record.classification.anomaly_score,
                record.relative_path,
            ),
        )
    )
    false_negatives = tuple(
        sorted(
            (
                record
                for record in revealed.records
                if record.label == "anomaly" and record.classification.predicted_class == "normal"
            ),
            key=lambda record: (
                record.classification.anomaly_score,
                record.relative_path,
            ),
        )
    )
    max_cases = config.failure_case_selection.max_cases_per_type
    selected_false_positives = tuple(
        _selected_case(record, case_type="false_positive", rank=rank)
        for rank, record in enumerate(false_positives[:max_cases], start=1)
    )
    selected_false_negatives = tuple(
        _selected_case(record, case_type="false_negative", rank=rank)
        for rank, record in enumerate(false_negatives[:max_cases], start=1)
    )
    false_positive_paths = tuple(record.relative_path for record in false_positives)
    false_negative_paths = tuple(record.relative_path for record in false_negatives)
    result = FailureCaseSelectionResult(
        status="ok",
        failure_code=None,
        method=revealed.method,
        item_count=revealed.batch_item_count,
        max_cases_per_type=max_cases,
        false_positive_count=len(false_positives),
        false_negative_count=len(false_negatives),
        selected_false_positives=selected_false_positives,
        selected_false_negatives=selected_false_negatives,
    )
    if not _result_is_valid(
        result,
        false_positive_paths=false_positive_paths,
        false_negative_paths=false_negative_paths,
        config=config,
    ):
        return _failed(
            FailureCaseSelectionFailureCode.FAILURE_SELECTION_RESULT_INVALID,
            revealed=revealed,
            config=config,
        )
    return result
