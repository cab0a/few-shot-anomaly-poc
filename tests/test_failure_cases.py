from __future__ import annotations

import inspect
from dataclasses import fields, replace
from pathlib import Path

import pytest

import few_shot_anomaly_poc.failure_cases as failure_module
from few_shot_anomaly_poc.calibration import (
    CalibrationMethod,
    FixedThresholdClassificationResult,
)
from few_shot_anomaly_poc.config import ProjectConfig, load_config
from few_shot_anomaly_poc.errors import (
    FailureCaseSelectionFailureCode,
    LabelRevealFailureCode,
)
from few_shot_anomaly_poc.failure_cases import (
    FailureCaseSelectionResult,
    SelectedFailureCase,
    select_failure_cases,
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
    failed_indices: frozenset[int] = frozenset(),
) -> FinalTestLabelRevealResult:
    paths = tuple(f"pcb1/Data/Images/Test/{index:04d}.JPG" for index in range(len(labels)))
    records = []
    for index, (path, label, score) in enumerate(zip(paths, labels, scores, strict=True)):
        failed = index in failed_indices
        is_anomalous = failed or score > 0.5
        classification = FixedThresholdClassificationResult(
            status="ok",
            failure_code=None,
            method=CalibrationMethod.ECC_RESIDUAL,
            relative_path=path,
            score_status="failed" if failed else "ok",
            score_failure_code="SYNTHETIC_SCORE_FAILURE" if failed else None,
            anomaly_score=score,
            threshold=0.5,
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
            score_margin=score - 0.5,
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
        method=CalibrationMethod.ECC_RESIDUAL,
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


def test_failure_selection_interface_has_no_limit_or_image_input() -> None:
    parameters = inspect.signature(select_failure_cases).parameters
    case_fields = {field.name for field in fields(SelectedFailureCase)}
    result_fields = {field.name for field in fields(FailureCaseSelectionResult)}

    assert tuple(parameters) == ("revealed", "config")
    assert "limit" not in parameters
    assert all("image" not in name for name in parameters)
    assert all("image" not in name for name in case_fields | result_fields)
    assert "technical_observation" not in case_fields


def test_failure_selection_applies_fixed_orders_and_limit(
    project_config: ProjectConfig,
) -> None:
    revealed = _revealed(
        ("normal",) * 6 + ("anomaly",) * 6,
        (0.9, 0.8, 0.8, 0.7, 0.6, 0.55, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5),
    )

    result = select_failure_cases(
        revealed,
        config=project_config,
    )

    assert result.succeeded
    assert result.status == "ok"
    assert result.failure_code is None
    assert result.method is CalibrationMethod.ECC_RESIDUAL
    assert result.item_count == 12
    assert result.max_cases_per_type == 5
    assert result.false_positive_count == 6
    assert result.false_negative_count == 6
    assert result.selected_false_positives is not None
    assert result.selected_false_negatives is not None
    assert tuple(item.rank for item in result.selected_false_positives) == (
        1,
        2,
        3,
        4,
        5,
    )
    assert tuple(item.relative_path for item in result.selected_false_positives) == (
        "pcb1/Data/Images/Test/0000.JPG",
        "pcb1/Data/Images/Test/0001.JPG",
        "pcb1/Data/Images/Test/0002.JPG",
        "pcb1/Data/Images/Test/0003.JPG",
        "pcb1/Data/Images/Test/0004.JPG",
    )
    assert tuple(item.anomaly_score for item in result.selected_false_positives) == (
        0.9,
        0.8,
        0.8,
        0.7,
        0.6,
    )
    assert tuple(item.relative_path for item in result.selected_false_negatives) == (
        "pcb1/Data/Images/Test/0006.JPG",
        "pcb1/Data/Images/Test/0007.JPG",
        "pcb1/Data/Images/Test/0008.JPG",
        "pcb1/Data/Images/Test/0009.JPG",
        "pcb1/Data/Images/Test/0010.JPG",
    )
    assert tuple(item.anomaly_score for item in result.selected_false_negatives) == (
        0.1,
        0.2,
        0.2,
        0.3,
        0.4,
    )
    assert all(
        item.case_type == "false_positive"
        and item.true_class == "normal"
        and item.predicted_class == "anomalous"
        for item in result.selected_false_positives
    )
    assert all(
        item.case_type == "false_negative"
        and item.true_class == "anomaly"
        and item.predicted_class == "normal"
        for item in result.selected_false_negatives
    )


def test_failure_selection_returns_empty_complete_selections_without_errors(
    project_config: ProjectConfig,
) -> None:
    result = select_failure_cases(
        _revealed(
            ("normal", "normal", "anomaly", "anomaly"),
            (0.1, 0.2, 0.8, 0.9),
        ),
        config=project_config,
    )

    assert result.succeeded
    assert result.false_positive_count == 0
    assert result.false_negative_count == 0
    assert result.selected_false_positives == ()
    assert result.selected_false_negatives == ()


def test_failure_selection_retains_failed_normal_score_as_false_positive(
    project_config: ProjectConfig,
) -> None:
    result = select_failure_cases(
        _revealed(
            ("normal", "normal", "anomaly"),
            (1.0, 0.8, 0.1),
            failed_indices=frozenset({0}),
        ),
        config=project_config,
    )

    assert result.succeeded
    assert result.selected_false_positives is not None
    first = result.selected_false_positives[0]
    assert first.relative_path == "pcb1/Data/Images/Test/0000.JPG"
    assert first.score_status == "failed"
    assert first.score_failure_code == "SYNTHETIC_SCORE_FAILURE"
    assert first.anomaly_score == 1.0
    assert first.score_margin == 0.5


def test_failure_selection_ignores_true_predictions(
    project_config: ProjectConfig,
) -> None:
    result = select_failure_cases(
        _revealed(
            ("normal", "normal", "anomaly", "anomaly"),
            (0.9, 0.1, 0.8, 0.2),
        ),
        config=project_config,
    )

    assert result.succeeded
    assert result.false_positive_count == 1
    assert result.false_negative_count == 1
    assert result.selected_false_positives is not None
    assert result.selected_false_negatives is not None
    assert result.selected_false_positives[0].relative_path.endswith("0000.JPG")
    assert result.selected_false_negatives[0].relative_path.endswith("0003.JPG")


def test_failure_selection_is_repeatable_and_does_not_mutate_input(
    project_config: ProjectConfig,
) -> None:
    revealed = _revealed(
        ("normal", "normal", "anomaly", "anomaly"),
        (0.9, 0.8, 0.2, 0.1),
    )

    first = select_failure_cases(revealed, config=project_config)
    second = select_failure_cases(revealed, config=project_config)

    assert first == second
    assert revealed == _revealed(
        ("normal", "normal", "anomaly", "anomaly"),
        (0.9, 0.8, 0.2, 0.1),
    )


def test_failure_selection_rejects_failed_reveal(
    project_config: ProjectConfig,
) -> None:
    failed_reveal = replace(
        _revealed(("normal", "anomaly"), (0.9, 0.1)),
        status="LABEL_REVEAL_FAILED",
        failure_code=LabelRevealFailureCode.LABEL_REVEAL_ORDER_MISMATCH,
        records=None,
        ordered_paths=(),
        order_mismatch_index=0,
    )

    result = select_failure_cases(
        failed_reveal,
        config=project_config,
    )

    assert not result.succeeded
    assert result.failure_code is FailureCaseSelectionFailureCode.FAILURE_SELECTION_REVEAL_INVALID
    assert result.item_count == 2
    assert result.false_positive_count is None
    assert result.false_negative_count is None
    assert result.selected_false_positives is None
    assert result.selected_false_negatives is None


def test_failure_selection_rejects_invalid_internal_result(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
) -> None:
    original_builder = failure_module._selected_case

    def wrong_rank(record, *, case_type, rank):
        return replace(
            original_builder(record, case_type=case_type, rank=rank),
            rank=99,
        )

    monkeypatch.setattr(failure_module, "_selected_case", wrong_rank)

    result = select_failure_cases(
        _revealed(("normal", "anomaly"), (0.9, 0.1)),
        config=project_config,
    )

    assert result.failure_code is FailureCaseSelectionFailureCode.FAILURE_SELECTION_RESULT_INVALID
    assert result.selected_false_positives is None
    assert result.selected_false_negatives is None
