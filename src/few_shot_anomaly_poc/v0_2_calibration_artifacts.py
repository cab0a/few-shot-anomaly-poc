"""Build exact v0.2 normal-only fit and calibration artifacts."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from few_shot_anomaly_poc.jsonio import write_json_atomic
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    ARTIFACT_CONTRACT_VERSION,
    validate_json_artifact,
    validate_tabular_records,
)

RUN_KIND = "final_test"
CALIBRATION_QUANTILE = 0.95
V0_2_4_MILESTONE = "v0.2.4"
NORMAL_REFERENCE_COUNT = 20
NORMAL_CALIBRATION_COUNT = 881
TOTAL_NORMAL_COUNT = NORMAL_REFERENCE_COUNT + NORMAL_CALIBRATION_COUNT
RGB_INPUT_SHAPE = (512, 512, 3)
RGB_STORE_SCHEMA = "v0.2-normal-rgb512-store-v1"
CALIBRATION_COLUMNS = (
    "contract_version",
    "run_id",
    "run_kind",
    "method",
    "source_path",
    "score_status",
    "score_failure_code",
    "anomaly_score",
)


class V0_2CalibrationArtifactError(Exception):
    """Reject incomplete, inconsistent, or overwriting calibration evidence."""


@dataclass(frozen=True)
class CalibrationScore:
    """Represent one normal-only score before threshold calibration."""

    source_path: str
    score_status: str
    score_failure_code: str | None
    anomaly_score: float


@dataclass(frozen=True)
class CalibrationResult:
    """Preserve validated score rows and their fixed threshold summary."""

    records: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def build_fit_record(
    *,
    run_id: str,
    method: str,
    status: str,
    successful_reference_count: int,
    failed_reference_count: int,
    reference_manifest_sha256: str,
    fitted_state_sha256: str | None,
    failure_code: str | None,
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one exact per-method fit artifact."""
    record = {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "run_id": run_id,
        "run_kind": RUN_KIND,
        "method": method,
        "status": status,
        "reference_count": 20,
        "successful_reference_count": successful_reference_count,
        "failed_reference_count": failed_reference_count,
        "reference_manifest_sha256": reference_manifest_sha256,
        "fitted_state_sha256": fitted_state_sha256,
        "failure_code": failure_code,
    }
    return validate_json_artifact(
        "fit",
        record,
        config=config,
        schema=schema,
    )


def calibrate_normal_scores(
    scores: Sequence[CalibrationScore],
    *,
    run_id: str,
    method: str,
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> CalibrationResult:
    """Apply the fixed nearest-rank normal-only threshold rule."""
    if not scores:
        raise V0_2CalibrationArtifactError("normal calibration scores are empty")
    paths = [score.source_path for score in scores]
    if len(paths) != len(set(paths)):
        raise V0_2CalibrationArtifactError("normal calibration paths are not unique")

    records = tuple(
        {
            "contract_version": ARTIFACT_CONTRACT_VERSION,
            "run_id": run_id,
            "run_kind": RUN_KIND,
            "method": method,
            "source_path": score.source_path,
            "score_status": score.score_status,
            "score_failure_code": score.score_failure_code,
            "anomaly_score": float(score.anomaly_score),
        }
        for score in sorted(scores, key=lambda item: item.source_path)
    )
    validated_records = validate_tabular_records(
        "calibration_score",
        records,
        schema=schema,
    )
    score_order = sorted(
        validated_records,
        key=lambda row: (float(row["anomaly_score"]), row["source_path"]),
    )
    sample_count = len(score_order)
    rank = math.ceil(CALIBRATION_QUANTILE * sample_count)
    threshold_source = score_order[rank - 1]
    threshold = float(threshold_source["anomaly_score"])
    predicted_anomalous_count = sum(
        row["score_status"] == "failed" or float(row["anomaly_score"]) > threshold
        for row in score_order
    )
    failure_count = sum(row["score_status"] == "failed" for row in score_order)
    summary = validate_json_artifact(
        "calibration_summary",
        {
            "contract_version": ARTIFACT_CONTRACT_VERSION,
            "run_id": run_id,
            "run_kind": RUN_KIND,
            "method": method,
            "sample_count": sample_count,
            "rank": rank,
            "quantile": CALIBRATION_QUANTILE,
            "threshold": threshold,
            "threshold_source_path": threshold_source["source_path"],
            "predicted_anomalous_count": predicted_anomalous_count,
            "score_failure_count": failure_count,
            "realized_normal_fpr": predicted_anomalous_count / sample_count,
        },
        config=config,
        schema=schema,
    )
    return CalibrationResult(records=validated_records, summary=summary)


def write_calibration_scores(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write validated score rows with fixed columns and no overwrite."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=CALIBRATION_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            for record in records:
                writer.writerow(record)
    except (csv.Error, OSError, ValueError) as error:
        path.unlink(missing_ok=True)
        raise V0_2CalibrationArtifactError("cannot write calibration scores") from error


def write_method_artifacts(
    *,
    method_root: Path,
    fit_record: Mapping[str, Any],
    calibration: CalibrationResult | None,
) -> tuple[Path, ...]:
    """Write one method's complete fixed stage without overwriting."""
    if method_root.exists():
        raise FileExistsError(f"refusing to overwrite {method_root}")
    method_root.mkdir(parents=True)
    paths = [method_root / "fit.json"]
    try:
        write_json_atomic(paths[0], dict(fit_record))
        if fit_record["status"] == "fit_ok":
            if calibration is None:
                raise V0_2CalibrationArtifactError("successful fit requires calibration artifacts")
            score_path = method_root / "calibration-scores.csv"
            summary_path = method_root / "calibration-summary.json"
            paths.append(score_path)
            write_calibration_scores(score_path, calibration.records)
            paths.append(summary_path)
            write_json_atomic(summary_path, calibration.summary)
        elif calibration is not None:
            raise V0_2CalibrationArtifactError("failed fit cannot contain calibration artifacts")
    except Exception:
        for path in reversed(paths):
            path.unlink(missing_ok=True)
        method_root.rmdir()
        raise
    return tuple(paths)
