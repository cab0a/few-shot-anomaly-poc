"""Build and serialize the fixed v0.2.7 revealed evaluation artifacts."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.metrics import average_precision_score, roc_auc_score

from few_shot_anomaly_poc.jsonio import write_json_atomic
from few_shot_anomaly_poc.v0_2_boundary_preparation import RUN_ID
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    ARTIFACT_CONTRACT_VERSION,
    METHODS,
    validate_json_artifact,
    validate_tabular_records,
)
from few_shot_anomaly_poc.v0_2_scoring_artifacts import MethodScoringArtifacts

RUN_KIND = "final_test"
ASSET_COUNT = 200
MAX_CASES_PER_TYPE = 5
LABEL_REVEAL_COLUMNS = (
    "contract_version",
    "run_id",
    "run_kind",
    "asset_id",
    "source_path",
    "true_class",
)
FAILURE_CASE_COLUMNS = (
    "contract_version",
    "run_id",
    "run_kind",
    "method",
    "case_type",
    "rank",
    "asset_id",
    "source_path",
    "true_class",
    "predicted_class",
    "anomaly_score",
    "threshold",
    "score_margin",
    "score_status",
    "score_failure_code",
)


class V0_2RevealedEvaluationError(Exception):
    """Reject incomplete joins, changed scores, or inconsistent revealed evidence."""


@dataclass(frozen=True)
class RevealedEvaluationArtifacts:
    """Hold one complete label artifact plus per-method metrics and failures."""

    label_records: tuple[dict[str, Any], ...]
    metrics_by_method: dict[str, dict[str, Any]]
    failures_by_method: dict[str, tuple[dict[str, Any], ...]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V0_2RevealedEvaluationError(message)


def build_label_reveal_records(
    sealed_records: Sequence[Mapping[str, Any]],
    *,
    expected_asset_ids: Sequence[str],
    schema: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Convert one exact ordered sealed mapping into the public reveal contract."""
    _require(
        len(sealed_records) == len(expected_asset_ids) == ASSET_COUNT,
        "reveal input count differs from the fixed final-test count",
    )
    records: list[dict[str, Any]] = []
    seen_assets: set[str] = set()
    seen_sources: set[str] = set()
    for index, (sealed, expected_id) in enumerate(
        zip(sealed_records, expected_asset_ids, strict=True)
    ):
        _require(
            set(sealed) == {"asset_id", "class_label", "source_path"},
            "sealed record fields changed",
        )
        asset_id = sealed["asset_id"]
        source_path = sealed["source_path"]
        true_class = sealed["class_label"]
        _require(
            asset_id == expected_id == f"asset-{index:06d}",
            "reveal asset ID is missing, extra, duplicate, or out of order",
        )
        _require(asset_id not in seen_assets, "reveal asset ID is duplicated")
        _require(
            isinstance(source_path, str)
            and source_path not in seen_sources
            and true_class in {"normal", "anomaly"},
            "reveal source path or class is invalid",
        )
        seen_assets.add(asset_id)
        seen_sources.add(source_path)
        records.append(
            {
                "contract_version": ARTIFACT_CONTRACT_VERSION,
                "run_id": RUN_ID,
                "run_kind": RUN_KIND,
                "asset_id": asset_id,
                "source_path": source_path,
                "true_class": true_class,
            }
        )
    return validate_tabular_records("label_reveal", records, schema=schema)


def _validate_method_join(
    *,
    method: str,
    scoring: MethodScoringArtifacts,
    labels: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]:
    _require(
        len(scoring.score_records)
        == len(scoring.classification_records)
        == len(labels)
        == ASSET_COUNT,
        f"{method} join count changed",
    )
    joined = []
    for score, classification, label in zip(
        scoring.score_records,
        scoring.classification_records,
        labels,
        strict=True,
    ):
        _require(
            score["method"] == classification["method"] == method
            and score["asset_id"] == classification["asset_id"] == label["asset_id"]
            and score["score_status"] == classification["score_status"]
            and score["score_failure_code"] == classification["score_failure_code"]
            and score["anomaly_score"] == classification["anomaly_score"],
            f"{method} score, classification, or reveal join changed",
        )
        joined.append((classification, label))
    return tuple(joined)


def _metrics(
    *,
    method: str,
    joined: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    y_true = tuple(1 if label["true_class"] == "anomaly" else 0 for _, label in joined)
    y_score = tuple(float(classification["anomaly_score"]) for classification, _ in joined)
    _require(
        all(math.isfinite(score) for score in y_score),
        f"{method} contains a non-finite revealed score",
    )
    normal_count = y_true.count(0)
    anomaly_count = y_true.count(1)
    _require(normal_count > 0 and anomaly_count > 0, "both revealed classes are required")
    try:
        auroc = float(roc_auc_score(y_true, y_score))
        auprc = float(average_precision_score(y_true, y_score))
    except (TypeError, ValueError, FloatingPointError) as error:
        raise V0_2RevealedEvaluationError(f"{method} ranking metrics failed") from error

    true_positive_count = sum(
        label["true_class"] == "anomaly" and classification["is_anomalous"] is True
        for classification, label in joined
    )
    false_negative_count = sum(
        label["true_class"] == "anomaly" and classification["is_anomalous"] is False
        for classification, label in joined
    )
    true_negative_count = sum(
        label["true_class"] == "normal" and classification["is_anomalous"] is False
        for classification, label in joined
    )
    false_positive_count = sum(
        label["true_class"] == "normal" and classification["is_anomalous"] is True
        for classification, label in joined
    )
    thresholds = {float(classification["threshold"]) for classification, _ in joined}
    _require(len(thresholds) == 1, f"{method} classification thresholds changed")
    record = {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "run_id": RUN_ID,
        "run_kind": RUN_KIND,
        "method": method,
        "positive_class": "anomaly",
        "item_count": ASSET_COUNT,
        "normal_count": normal_count,
        "anomaly_count": anomaly_count,
        "true_positive_count": true_positive_count,
        "false_negative_count": false_negative_count,
        "true_negative_count": true_negative_count,
        "false_positive_count": false_positive_count,
        "score_failure_count": sum(
            classification["score_status"] == "failed" for classification, _ in joined
        ),
        "image_level_auroc": auroc,
        "image_level_auprc": auprc,
        "normal_false_positive_rate": false_positive_count / normal_count,
        "anomaly_recall": true_positive_count / anomaly_count,
        "threshold": thresholds.pop(),
    }
    return validate_json_artifact(
        "metrics",
        record,
        config=config,
        schema=schema,
    )


def _failure_record(
    *,
    method: str,
    case_type: str,
    rank: int,
    classification: Mapping[str, Any],
    label: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "run_id": RUN_ID,
        "run_kind": RUN_KIND,
        "method": method,
        "case_type": case_type,
        "rank": rank,
        "asset_id": label["asset_id"],
        "source_path": label["source_path"],
        "true_class": label["true_class"],
        "predicted_class": classification["predicted_class"],
        "anomaly_score": float(classification["anomaly_score"]),
        "threshold": float(classification["threshold"]),
        "score_margin": float(classification["score_margin"]),
        "score_status": classification["score_status"],
        "score_failure_code": classification["score_failure_code"],
    }


def _failure_cases(
    *,
    method: str,
    joined: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    schema: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    false_positives = sorted(
        (
            (classification, label)
            for classification, label in joined
            if label["true_class"] == "normal" and classification["is_anomalous"] is True
        ),
        key=lambda item: (-float(item[0]["anomaly_score"]), item[1]["asset_id"]),
    )
    false_negatives = sorted(
        (
            (classification, label)
            for classification, label in joined
            if label["true_class"] == "anomaly" and classification["is_anomalous"] is False
        ),
        key=lambda item: (float(item[0]["anomaly_score"]), item[1]["asset_id"]),
    )
    records = [
        _failure_record(
            method=method,
            case_type="false_positive",
            rank=rank,
            classification=classification,
            label=label,
        )
        for rank, (classification, label) in enumerate(
            false_positives[:MAX_CASES_PER_TYPE], start=1
        )
    ]
    records.extend(
        _failure_record(
            method=method,
            case_type="false_negative",
            rank=rank,
            classification=classification,
            label=label,
        )
        for rank, (classification, label) in enumerate(
            false_negatives[:MAX_CASES_PER_TYPE], start=1
        )
    )
    _require(records, f"{method} produced no selectable fixed-threshold error")
    return validate_tabular_records("failure_case", records, schema=schema)


def build_revealed_evaluation_artifacts(
    *,
    label_records: Sequence[Mapping[str, Any]],
    scoring_by_method: Mapping[str, MethodScoringArtifacts],
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> RevealedEvaluationArtifacts:
    """Join unchanged label-free evidence and calculate only the v0.2.7 outputs."""
    validated_labels = validate_tabular_records("label_reveal", label_records, schema=schema)
    _require(set(scoring_by_method) == set(METHODS), "revealed method inventory changed")
    metrics_by_method: dict[str, dict[str, Any]] = {}
    failures_by_method: dict[str, tuple[dict[str, Any], ...]] = {}
    for method in METHODS:
        joined = _validate_method_join(
            method=method,
            scoring=scoring_by_method[method],
            labels=validated_labels,
        )
        metrics_by_method[method] = _metrics(
            method=method,
            joined=joined,
            config=config,
            schema=schema,
        )
        failures_by_method[method] = _failure_cases(
            method=method,
            joined=joined,
            schema=schema,
        )
    return RevealedEvaluationArtifacts(
        label_records=validated_labels,
        metrics_by_method=metrics_by_method,
        failures_by_method=failures_by_method,
    )


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _write_csv(
    path: Path,
    *,
    columns: Sequence[str],
    records: Sequence[Mapping[str, Any]],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for record in records:
                writer.writerow({key: _csv_value(record[key]) for key in columns})
    except Exception:
        path.unlink(missing_ok=True)
        raise


def write_revealed_evaluation_artifacts(
    root: Path,
    artifacts: RevealedEvaluationArtifacts,
) -> tuple[Path, ...]:
    """Write one complete v0.2.7 stage into a new staging root."""
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    root.mkdir(parents=True)
    paths = [root / "revealed-labels.csv"]
    try:
        _write_csv(
            paths[0],
            columns=LABEL_REVEAL_COLUMNS,
            records=artifacts.label_records,
        )
        for method in METHODS:
            method_root = root / method
            metrics_path = method_root / "metrics.json"
            failures_path = method_root / "failure-cases.csv"
            write_json_atomic(metrics_path, artifacts.metrics_by_method[method])
            _write_csv(
                failures_path,
                columns=FAILURE_CASE_COLUMNS,
                records=artifacts.failures_by_method[method],
            )
            paths.extend((metrics_path, failures_path))
    except Exception:
        for path in paths:
            path.unlink(missing_ok=True)
        raise
    return tuple(paths)


def _read_csv(path: Path, *, columns: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            _require(reader.fieldnames == list(columns), f"{path.name} columns changed")
            rows = list(reader)
    except (csv.Error, OSError, UnicodeError) as error:
        raise V0_2RevealedEvaluationError(f"cannot read {path.name}") from error
    _require(rows and all(None not in row for row in rows), f"{path.name} rows are invalid")
    return rows


def read_label_reveal_csv(path: Path, *, schema: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Read and validate one serialized v0.2 label reveal."""
    return validate_tabular_records(
        "label_reveal",
        _read_csv(path, columns=LABEL_REVEAL_COLUMNS),
        schema=schema,
    )


def read_failure_cases_csv(path: Path, *, schema: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Read and validate one serialized v0.2 failure selection."""
    try:
        records = [
            {
                **row,
                "rank": int(row["rank"]),
                "anomaly_score": float(row["anomaly_score"]),
                "threshold": float(row["threshold"]),
                "score_margin": float(row["score_margin"]),
                "score_failure_code": row["score_failure_code"] or None,
            }
            for row in _read_csv(path, columns=FAILURE_CASE_COLUMNS)
        ]
    except ValueError as error:
        raise V0_2RevealedEvaluationError("failure-case numeric value is invalid") from error
    return validate_tabular_records("failure_case", records, schema=schema)


def read_metrics_json(
    path: Path,
    *,
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Read and validate one serialized v0.2 method metrics artifact."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V0_2RevealedEvaluationError("cannot read metrics JSON") from error
    return validate_json_artifact("metrics", value, config=config, schema=schema)
