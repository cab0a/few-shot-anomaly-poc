"""Serialize validated evaluation evidence under the fixed v0.1 contract."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np

from few_shot_anomaly_poc.calibration import (
    CalibrationMethod,
    FixedThresholdBatchClassificationResult,
    calibrate_normal_threshold,
    classify_fixed_threshold_batch,
)
from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.cpu_latency import (
    CPULatencyResult,
    cpu_latency_result_is_valid,
)
from few_shot_anomaly_poc.ecc_residual import ECCResidualScoreResult
from few_shot_anomaly_poc.errors import EvaluationArtifactError
from few_shot_anomaly_poc.failure_cases import (
    FailureCaseSelectionResult,
    select_failure_cases,
)
from few_shot_anomaly_poc.hard_gate_decision import (
    DecisionProcessEvidence,
    HardGateDecisionResult,
    apply_hard_gate_decision,
)
from few_shot_anomaly_poc.hog_scoring import PatchHOGScoreResult
from few_shot_anomaly_poc.image_metrics import (
    ImageLevelMetricsResult,
    calculate_image_level_metrics,
)
from few_shot_anomaly_poc.label_reveal import (
    FinalTestLabelRecord,
    FinalTestLabelRevealResult,
    reveal_final_test_labels,
)

CONTRACT_VERSION = "evaluation-artifacts/v0.1"
RUN_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

type ScoreResult = ECCResidualScoreResult | PatchHOGScoreResult
type RunKind = Literal["synthetic", "final_test"]

SCORE_COLUMNS = (
    "contract_version",
    "run_id",
    "run_kind",
    "method",
    "partition",
    "relative_path",
    "score_status",
    "score_failure_code",
    "anomaly_score",
    "diagnostics_json",
)
CLASSIFICATION_COLUMNS = (
    "contract_version",
    "run_id",
    "run_kind",
    "method",
    "relative_path",
    "score_status",
    "score_failure_code",
    "anomaly_score",
    "threshold",
    "threshold_source_path",
    "calibration_sample_count",
    "calibration_rank",
    "predicted_class",
    "is_anomalous",
    "decision_reason",
    "score_margin",
)
LABEL_COLUMNS = (
    "contract_version",
    "run_id",
    "run_kind",
    "method",
    "relative_path",
    "true_class",
)
LATENCY_OBSERVATION_COLUMNS = (
    "contract_version",
    "run_id",
    "run_kind",
    "method",
    "pass_index",
    "relative_path",
    "duration_ns",
    "score_status",
    "score_failure_code",
)
FAILURE_COLUMNS = (
    "contract_version",
    "run_id",
    "run_kind",
    "method",
    "case_type",
    "rank",
    "relative_path",
    "true_class",
    "predicted_class",
    "anomaly_score",
    "threshold",
    "score_margin",
    "score_status",
    "score_failure_code",
)


@dataclass(frozen=True)
class MethodEvaluationArtifacts:
    """One method's complete primitive outputs and their source scores."""

    method: CalibrationMethod
    calibration_scores: Mapping[str, ScoreResult]
    final_test_scores: Mapping[str, ScoreResult]
    classifications: FixedThresholdBatchClassificationResult
    revealed: FinalTestLabelRevealResult
    metrics: ImageLevelMetricsResult
    latency: CPULatencyResult
    failures: FailureCaseSelectionResult
    process_evidence: DecisionProcessEvidence
    decision: HardGateDecisionResult


@dataclass(frozen=True)
class EvaluationArtifactBundle:
    """Immutable inputs for one synthetic or final-test artifact directory."""

    run_id: str
    run_kind: RunKind
    dataset: str
    category: str
    source_commit: str
    partition_manifest_sha256: str
    methods: tuple[MethodEvaluationArtifacts, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, (str, int, float)):
        return str(value)
    raise EvaluationArtifactError(f"unsupported CSV value type: {type(value).__name__}")


def _json_ready(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    raise EvaluationArtifactError(f"unsupported JSON value type: {type(value).__name__}")


def _canonical_json_cell(value: object) -> str:
    try:
        return json.dumps(
            _json_ready(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise EvaluationArtifactError("cannot serialize canonical JSON cell") from error


def _write_json(path: Path, value: object) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(
                _json_ready(value),
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
    except (OSError, TypeError, ValueError) as error:
        raise EvaluationArtifactError(f"cannot write JSON artifact {path.name}") from error


def _write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=columns,
                dialect="excel",
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            for row in rows:
                if set(row) != set(columns):
                    raise EvaluationArtifactError(f"CSV row keys do not match {path.name} contract")
                writer.writerow({key: _text(row[key]) for key in columns})
    except EvaluationArtifactError:
        raise
    except (OSError, csv.Error, TypeError, ValueError) as error:
        raise EvaluationArtifactError(f"cannot write CSV artifact {path.name}") from error


def _score_diagnostics(score: ScoreResult) -> dict:
    if isinstance(score, ECCResidualScoreResult):
        return {
            "correlation": score.correlation,
            "effective_pixel_count": score.effective_pixel_count,
            "effective_support_fraction": score.effective_support_fraction,
            "registration_status": score.registration_status,
            "registration_valid_fraction": score.registration_valid_fraction,
            "rotation_degrees": score.rotation_degrees,
            "top_pixel_count": score.top_pixel_count,
            "translation_x_pixels": score.translation_x_pixels,
            "translation_y_pixels": score.translation_y_pixels,
            "warp_matrix": score.warp_matrix,
        }
    return {
        "failed_patch_index": score.failed_patch_index,
        "patch_anomaly_scores": score.patch_anomaly_scores,
        "successful_patch_count": score.successful_patch_count,
        "top_patch_count": score.top_patch_count,
        "top_patch_indices": score.top_patch_indices,
    }


def _score_rows(
    item: MethodEvaluationArtifacts,
    *,
    run_id: str,
    run_kind: RunKind,
) -> list[dict[str, object]]:
    rows = []
    for partition, scores in (
        ("calibration", item.calibration_scores),
        ("final_test", item.final_test_scores),
    ):
        for path in sorted(scores):
            score = scores[path]
            rows.append(
                {
                    "contract_version": CONTRACT_VERSION,
                    "run_id": run_id,
                    "run_kind": run_kind,
                    "method": str(item.method),
                    "partition": partition,
                    "relative_path": path,
                    "score_status": score.score_status,
                    "score_failure_code": score.failure_code,
                    "anomaly_score": score.anomaly_score,
                    "diagnostics_json": _canonical_json_cell(_score_diagnostics(score)),
                }
            )
    return rows


def _classification_rows(
    item: MethodEvaluationArtifacts,
    *,
    run_id: str,
    run_kind: RunKind,
) -> list[dict[str, object]]:
    assert item.classifications.classifications is not None
    return [
        {
            "contract_version": CONTRACT_VERSION,
            "run_id": run_id,
            "run_kind": run_kind,
            "method": str(item.method),
            "relative_path": record.relative_path,
            "score_status": record.score_status,
            "score_failure_code": record.score_failure_code,
            "anomaly_score": record.anomaly_score,
            "threshold": record.threshold,
            "threshold_source_path": record.threshold_source_path,
            "calibration_sample_count": record.calibration_sample_count,
            "calibration_rank": record.calibration_rank,
            "predicted_class": record.predicted_class,
            "is_anomalous": record.is_anomalous,
            "decision_reason": record.decision_reason,
            "score_margin": record.score_margin,
        }
        for record in item.classifications.classifications
    ]


def _label_rows(
    item: MethodEvaluationArtifacts,
    *,
    run_id: str,
    run_kind: RunKind,
) -> list[dict[str, object]]:
    assert item.revealed.records is not None
    return [
        {
            "contract_version": CONTRACT_VERSION,
            "run_id": run_id,
            "run_kind": run_kind,
            "method": str(item.method),
            "relative_path": record.relative_path,
            "true_class": record.label,
        }
        for record in item.revealed.records
    ]


def _metrics_json(
    item: MethodEvaluationArtifacts,
    *,
    run_id: str,
    run_kind: RunKind,
) -> dict:
    result = item.metrics
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "run_kind": run_kind,
        "method": str(item.method),
        "positive_class": result.positive_class,
        "item_count": result.item_count,
        "normal_count": result.normal_count,
        "anomaly_count": result.anomaly_count,
        "true_positive_count": result.true_positive_count,
        "false_negative_count": result.false_negative_count,
        "true_negative_count": result.true_negative_count,
        "false_positive_count": result.false_positive_count,
        "score_failure_count": result.score_failure_count,
        "image_level_auroc": result.image_level_auroc,
        "image_level_auprc": result.image_level_auprc,
        "normal_false_positive_rate": result.normal_false_positive_rate,
        "anomaly_recall": result.anomaly_recall,
        "threshold": result.threshold,
        "threshold_source_path": result.threshold_source_path,
    }


def _latency_json(
    item: MethodEvaluationArtifacts,
    *,
    run_id: str,
    run_kind: RunKind,
) -> dict:
    result = item.latency
    assert result.environment is not None
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "run_kind": run_kind,
        "method": str(item.method),
        "measurement_boundary": result.measurement_boundary,
        "timer": result.timer,
        "warmup_passes": result.warmup_passes,
        "timed_passes": result.timed_passes,
        "item_count": result.item_count,
        "sample_count": result.sample_count,
        "score_failure_timing_count": result.score_failure_timing_count,
        "score_failure_paths": result.score_failure_paths,
        "median_latency_ns": result.median_latency_ns,
        "p95_latency_ns": result.p95_latency_ns,
        "median_latency_seconds": result.median_latency_seconds,
        "p95_latency_seconds": result.p95_latency_seconds,
        "p95_rank": result.p95_rank,
        "environment": {
            "cpu_model": result.environment.cpu_model,
            "logical_core_count": result.environment.logical_core_count,
            "physical_core_count": result.environment.physical_core_count,
            "ram_bytes": result.environment.ram_bytes,
            "operating_system": result.environment.operating_system,
            "machine": result.environment.machine,
            "python_version": result.environment.python_version,
            "opencv_thread_count": result.environment.opencv_thread_count,
            "dependency_versions": result.environment.dependency_versions,
            "thread_environment": result.environment.thread_environment,
        },
    }


def _latency_rows(
    item: MethodEvaluationArtifacts,
    *,
    run_id: str,
    run_kind: RunKind,
) -> list[dict[str, object]]:
    assert item.latency.observations is not None
    return [
        {
            "contract_version": CONTRACT_VERSION,
            "run_id": run_id,
            "run_kind": run_kind,
            "method": str(item.method),
            "pass_index": record.pass_index,
            "relative_path": record.relative_path,
            "duration_ns": record.duration_ns,
            "score_status": record.score_status,
            "score_failure_code": record.score_failure_code,
        }
        for record in item.latency.observations
    ]


def _failure_rows(
    item: MethodEvaluationArtifacts,
    *,
    run_id: str,
    run_kind: RunKind,
) -> list[dict[str, object]]:
    assert item.failures.selected_false_positives is not None
    assert item.failures.selected_false_negatives is not None
    selected = (
        *item.failures.selected_false_positives,
        *item.failures.selected_false_negatives,
    )
    return [
        {
            "contract_version": CONTRACT_VERSION,
            "run_id": run_id,
            "run_kind": run_kind,
            "method": str(item.method),
            "case_type": record.case_type,
            "rank": record.rank,
            "relative_path": record.relative_path,
            "true_class": record.true_class,
            "predicted_class": record.predicted_class,
            "anomaly_score": record.anomaly_score,
            "threshold": record.threshold,
            "score_margin": record.score_margin,
            "score_status": record.score_status,
            "score_failure_code": record.score_failure_code,
        }
        for record in selected
    ]


def _decision_json(
    item: MethodEvaluationArtifacts,
    *,
    run_id: str,
    run_kind: RunKind,
) -> dict:
    result = item.decision
    assert result.gate_outcomes is not None
    return {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "run_kind": run_kind,
        "method": str(item.method),
        "decision": result.decision,
        "gate_outcomes": [
            {
                "gate_id": outcome.gate_id,
                "order": outcome.order,
                "operator": outcome.operator,
                "required_value": outcome.required_value,
                "observed_value": outcome.observed_value,
                "passed": outcome.passed,
            }
            for outcome in result.gate_outcomes
        ],
        "all_hard_gates_passed": result.all_hard_gates_passed,
        "first_failed_gate": result.first_failed_gate,
        "test_leakage_detected": result.test_leakage_detected,
        "failure_review_disposition": result.failure_review_disposition,
        "failure_review_rationale": result.failure_review_rationale,
        "condition": result.condition,
        "decision_reason": result.decision_reason,
    }


def _validate_method(item: MethodEvaluationArtifacts, *, config: ProjectConfig) -> None:
    calibration = calibrate_normal_threshold(
        item.calibration_scores,
        method=item.method,
        config=config,
    )
    if not calibration.succeeded:
        raise EvaluationArtifactError("calibration scores do not produce a valid threshold")
    classifications = classify_fixed_threshold_batch(
        item.final_test_scores,
        calibration=calibration,
        config=config,
    )
    if classifications != item.classifications:
        raise EvaluationArtifactError("classification artifact does not match source scores")
    if item.revealed.records is None:
        raise EvaluationArtifactError("revealed labels are unavailable")
    label_records = tuple(
        FinalTestLabelRecord(
            relative_path=record.relative_path,
            label=record.label,
        )
        for record in item.revealed.records
    )
    revealed = reveal_final_test_labels(
        classifications,
        label_records,
        config=config,
    )
    if revealed != item.revealed:
        raise EvaluationArtifactError("label reveal artifact does not match classifications")
    metrics = calculate_image_level_metrics(revealed, config=config)
    if metrics != item.metrics:
        raise EvaluationArtifactError("metric artifact does not match revealed records")
    failures = select_failure_cases(revealed, config=config)
    if failures != item.failures:
        raise EvaluationArtifactError("failure artifact does not match revealed records")
    if not cpu_latency_result_is_valid(item.latency, config=config):
        raise EvaluationArtifactError("latency artifact is invalid")
    decision = apply_hard_gate_decision(
        metrics,
        item.latency,
        item.process_evidence,
        config=config,
    )
    if decision != item.decision:
        raise EvaluationArtifactError("decision artifact does not match fixed hard gates")


def _validate_bundle(bundle: EvaluationArtifactBundle, *, config: ProjectConfig) -> None:
    if (
        not isinstance(bundle, EvaluationArtifactBundle)
        or not RUN_ID_PATTERN.fullmatch(bundle.run_id)
        or bundle.run_kind not in ("synthetic", "final_test")
        or not isinstance(bundle.dataset, str)
        or not bundle.dataset
        or not isinstance(bundle.category, str)
        or not bundle.category
        or not COMMIT_PATTERN.fullmatch(bundle.source_commit)
        or not SHA256_PATTERN.fullmatch(bundle.partition_manifest_sha256)
        or not bundle.methods
        or any(not isinstance(item, MethodEvaluationArtifacts) for item in bundle.methods)
        or tuple(item.method for item in bundle.methods)
        != tuple(sorted((item.method for item in bundle.methods), key=str))
        or len({item.method for item in bundle.methods}) != len(bundle.methods)
    ):
        raise EvaluationArtifactError("bundle metadata is invalid")
    for item in bundle.methods:
        _validate_method(item, config=config)


def write_evaluation_artifact_bundle(
    bundle: EvaluationArtifactBundle,
    *,
    output_root: Path,
    config_path: Path,
    config: ProjectConfig,
) -> Path:
    """Validate and atomically create one non-overwritable artifact bundle."""
    _validate_bundle(bundle, config=config)
    try:
        config_sha256 = _sha256_file(config_path)
    except OSError as error:
        raise EvaluationArtifactError("cannot hash the fixed configuration") from error

    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / bundle.run_id
    if target.exists():
        raise EvaluationArtifactError(f"refusing to overwrite artifact bundle {target}")
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{bundle.run_id}.",
            dir=output_root,
        )
    )
    file_entries = []
    try:
        for item in bundle.methods:
            method_dir = temporary / str(item.method)
            method_dir.mkdir()
            files = (
                (
                    "score",
                    method_dir / "scores.csv",
                    SCORE_COLUMNS,
                    _score_rows(item, run_id=bundle.run_id, run_kind=bundle.run_kind),
                ),
                (
                    "classification",
                    method_dir / "classifications.csv",
                    CLASSIFICATION_COLUMNS,
                    _classification_rows(
                        item,
                        run_id=bundle.run_id,
                        run_kind=bundle.run_kind,
                    ),
                ),
                (
                    "label_reveal",
                    method_dir / "revealed-labels.csv",
                    LABEL_COLUMNS,
                    _label_rows(item, run_id=bundle.run_id, run_kind=bundle.run_kind),
                ),
                (
                    "latency_observations",
                    method_dir / "latency-observations.csv",
                    LATENCY_OBSERVATION_COLUMNS,
                    _latency_rows(item, run_id=bundle.run_id, run_kind=bundle.run_kind),
                ),
                (
                    "failure",
                    method_dir / "failure-cases.csv",
                    FAILURE_COLUMNS,
                    _failure_rows(item, run_id=bundle.run_id, run_kind=bundle.run_kind),
                ),
            )
            for artifact_type, path, columns, rows in files:
                _write_csv(path, columns, rows)
                file_entries.append(
                    {
                        "artifact_type": artifact_type,
                        "method": str(item.method),
                        "relative_path": path.relative_to(temporary).as_posix(),
                        "sha256": _sha256_file(path),
                        "record_count": len(rows),
                    }
                )
            json_files = (
                (
                    "metrics",
                    method_dir / "metrics.json",
                    _metrics_json(item, run_id=bundle.run_id, run_kind=bundle.run_kind),
                ),
                (
                    "latency",
                    method_dir / "latency.json",
                    _latency_json(item, run_id=bundle.run_id, run_kind=bundle.run_kind),
                ),
                (
                    "decision",
                    method_dir / "decision.json",
                    _decision_json(item, run_id=bundle.run_id, run_kind=bundle.run_kind),
                ),
            )
            for artifact_type, path, value in json_files:
                _write_json(path, value)
                file_entries.append(
                    {
                        "artifact_type": artifact_type,
                        "method": str(item.method),
                        "relative_path": path.relative_to(temporary).as_posix(),
                        "sha256": _sha256_file(path),
                        "record_count": 1,
                    }
                )

        manifest = {
            "contract_version": CONTRACT_VERSION,
            "run_id": bundle.run_id,
            "run_kind": bundle.run_kind,
            "dataset": bundle.dataset,
            "category": bundle.category,
            "source_commit": bundle.source_commit,
            "config_sha256": config_sha256,
            "partition_manifest_sha256": bundle.partition_manifest_sha256,
            "files": sorted(file_entries, key=lambda entry: entry["relative_path"]),
        }
        _write_json(temporary / "artifact-manifest.json", manifest)
        os.replace(temporary, target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target
