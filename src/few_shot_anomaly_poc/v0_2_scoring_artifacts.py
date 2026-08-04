"""Build and validate label-free v0.2.5 scoring artifacts."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    ARTIFACT_CONTRACT_VERSION,
    validate_tabular_records,
)

RUN_KIND = "final_test"
ASSET_COUNT = 200
TIMED_PASS_COUNT = 3
SCORE_TOLERANCE = 1e-6
SCORE_COLUMNS = (
    "contract_version",
    "run_id",
    "run_kind",
    "method",
    "asset_id",
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
    "asset_id",
    "score_status",
    "score_failure_code",
    "anomaly_score",
    "threshold",
    "predicted_class",
    "is_anomalous",
    "decision_reason",
    "score_margin",
)
LATENCY_COLUMNS = (
    "contract_version",
    "run_id",
    "run_kind",
    "method",
    "pass_index",
    "asset_id",
    "adapter_duration_ns",
    "scorer_duration_ns",
    "duration_ns",
    "score_status",
    "score_failure_code",
    "anomaly_score",
)


class V0_2ScoringArtifactError(Exception):
    """Reject incomplete, inconsistent, or overwriting scoring evidence."""


@dataclass(frozen=True)
class ScoreEvidence:
    """Hold one canonical label-free score and non-semantic diagnostics."""

    asset_id: str
    score_status: str
    score_failure_code: str | None
    anomaly_score: float
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class TimedScoreEvidence:
    """Hold one timed score observation at the fixed CPU boundary."""

    pass_index: int
    asset_id: str
    adapter_duration_ns: int | None
    scorer_duration_ns: int
    score_status: str
    score_failure_code: str | None
    anomaly_score: float


@dataclass(frozen=True)
class MethodScoringArtifacts:
    """Hold one method's complete validated v0.2.5 public bundle."""

    score_records: tuple[dict[str, Any], ...]
    classification_records: tuple[dict[str, Any], ...]
    latency_records: tuple[dict[str, Any], ...]


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise V0_2ScoringArtifactError("score diagnostics are not canonical JSON data") from error


def _expected_asset_ids() -> tuple[str, ...]:
    return tuple(f"asset-{index:06d}" for index in range(ASSET_COUNT))


def build_method_scoring_artifacts(
    *,
    run_id: str,
    method: str,
    threshold: float,
    scores: Sequence[ScoreEvidence],
    timed_scores: Sequence[TimedScoreEvidence],
    schema: Mapping[str, Any],
) -> MethodScoringArtifacts:
    """Build one complete method bundle and enforce score repetition identity."""
    expected_assets = _expected_asset_ids()
    ordered_scores = tuple(sorted(scores, key=lambda item: item.asset_id))
    if tuple(item.asset_id for item in ordered_scores) != expected_assets:
        raise V0_2ScoringArtifactError("canonical scores do not cover the fixed opaque assets")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
    ):
        raise V0_2ScoringArtifactError("threshold must be finite")

    score_records = tuple(
        {
            "contract_version": ARTIFACT_CONTRACT_VERSION,
            "run_id": run_id,
            "run_kind": RUN_KIND,
            "method": method,
            "asset_id": item.asset_id,
            "score_status": item.score_status,
            "score_failure_code": item.score_failure_code,
            "anomaly_score": float(item.anomaly_score),
            "diagnostics_json": _canonical_json(item.diagnostics),
        }
        for item in ordered_scores
    )
    validated_scores = validate_tabular_records("score", score_records, schema=schema)

    classification_records = []
    for record in validated_scores:
        score = float(record["anomaly_score"])
        failed = record["score_status"] == "failed"
        is_anomalous = failed or score > float(threshold)
        reason = (
            "score_failed"
            if failed
            else "score_strictly_greater_than_threshold"
            if score > float(threshold)
            else "score_not_greater_than_threshold"
        )
        classification_records.append(
            {
                "contract_version": ARTIFACT_CONTRACT_VERSION,
                "run_id": run_id,
                "run_kind": RUN_KIND,
                "method": method,
                "asset_id": record["asset_id"],
                "score_status": record["score_status"],
                "score_failure_code": record["score_failure_code"],
                "anomaly_score": score,
                "threshold": float(threshold),
                "predicted_class": "anomalous" if is_anomalous else "normal",
                "is_anomalous": is_anomalous,
                "decision_reason": reason,
                "score_margin": score - float(threshold),
            }
        )
    validated_classifications = validate_tabular_records(
        "classification",
        classification_records,
        schema=schema,
    )

    ordered_timed = tuple(sorted(timed_scores, key=lambda item: (item.pass_index, item.asset_id)))
    expected_timed_keys = tuple(
        (pass_index, asset_id)
        for pass_index in range(TIMED_PASS_COUNT)
        for asset_id in expected_assets
    )
    if tuple((item.pass_index, item.asset_id) for item in ordered_timed) != expected_timed_keys:
        raise V0_2ScoringArtifactError("timed scores do not cover the fixed passes and assets")

    canonical_by_asset = {record["asset_id"]: record for record in validated_scores}
    latency_records = []
    for item in ordered_timed:
        canonical = canonical_by_asset[item.asset_id]
        if (
            item.score_status != canonical["score_status"]
            or item.score_failure_code != canonical["score_failure_code"]
            or abs(float(item.anomaly_score) - float(canonical["anomaly_score"])) > SCORE_TOLERANCE
        ):
            raise V0_2ScoringArtifactError("timed score repetition differs from canonical pass 0")
        if (
            not isinstance(item.scorer_duration_ns, int)
            or isinstance(item.scorer_duration_ns, bool)
            or item.scorer_duration_ns < 1
        ):
            raise V0_2ScoringArtifactError("scorer duration must be a positive integer")
        if method == "dinov2_vits14_224_nn":
            if (
                not isinstance(item.adapter_duration_ns, int)
                or isinstance(item.adapter_duration_ns, bool)
                or item.adapter_duration_ns < 1
            ):
                raise V0_2ScoringArtifactError("DINOv2 adapter duration must be positive")
            duration_ns = item.adapter_duration_ns + item.scorer_duration_ns
        else:
            if item.adapter_duration_ns is not None:
                raise V0_2ScoringArtifactError("classical timing cannot include adapter duration")
            duration_ns = item.scorer_duration_ns
        latency_records.append(
            {
                "contract_version": ARTIFACT_CONTRACT_VERSION,
                "run_id": run_id,
                "run_kind": RUN_KIND,
                "method": method,
                "pass_index": item.pass_index,
                "asset_id": item.asset_id,
                "adapter_duration_ns": item.adapter_duration_ns,
                "scorer_duration_ns": item.scorer_duration_ns,
                "duration_ns": duration_ns,
                "score_status": item.score_status,
                "score_failure_code": item.score_failure_code,
                "anomaly_score": float(item.anomaly_score),
            }
        )
    validated_latency = validate_tabular_records(
        "latency_observation",
        latency_records,
        schema=schema,
    )
    return MethodScoringArtifacts(
        score_records=validated_scores,
        classification_records=validated_classifications,
        latency_records=validated_latency,
    )


def latency_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    """Derive the fixed all-observation median and nearest-rank p95."""
    durations = [int(record["duration_ns"]) for record in records]
    if len(durations) != ASSET_COUNT * TIMED_PASS_COUNT or any(value < 1 for value in durations):
        raise V0_2ScoringArtifactError("latency observations are incomplete")
    p95_rank = math.ceil(0.95 * len(durations))
    return {
        "sample_count": len(durations),
        "median_latency_ns": float(statistics.median(durations)),
        "p95_latency_ns": sorted(durations)[p95_rank - 1],
        "p95_rank": p95_rank,
    }


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
    except (csv.Error, OSError, ValueError) as error:
        path.unlink(missing_ok=True)
        raise V0_2ScoringArtifactError("cannot write scoring CSV") from error


def write_method_scoring_artifacts(
    method_root: Path,
    artifacts: MethodScoringArtifacts,
) -> tuple[Path, Path, Path]:
    """Write the three fixed v0.2.5 CSV files without overwriting."""
    if method_root.exists():
        raise FileExistsError(f"refusing to overwrite {method_root}")
    paths = (
        method_root / "scores.csv",
        method_root / "classifications.csv",
        method_root / "latency-observations.csv",
    )
    method_root.mkdir(parents=True)
    try:
        _write_csv(paths[0], columns=SCORE_COLUMNS, records=artifacts.score_records)
        _write_csv(
            paths[1],
            columns=CLASSIFICATION_COLUMNS,
            records=artifacts.classification_records,
        )
        _write_csv(paths[2], columns=LATENCY_COLUMNS, records=artifacts.latency_records)
    except Exception:
        for path in paths:
            path.unlink(missing_ok=True)
        method_root.rmdir()
        raise
    return paths


def _read_csv(path: Path, *, columns: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(columns):
                raise V0_2ScoringArtifactError("scoring CSV columns changed")
            rows = list(reader)
    except (csv.Error, OSError, UnicodeError) as error:
        raise V0_2ScoringArtifactError("cannot read scoring CSV") from error
    if not rows or any(None in row for row in rows):
        raise V0_2ScoringArtifactError("scoring CSV rows are invalid")
    return rows


def _nullable(value: str) -> str | None:
    return None if value == "" else value


def _boolean(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise V0_2ScoringArtifactError("scoring CSV boolean is invalid")


def read_method_scoring_artifacts(
    method_root: Path,
    *,
    schema: Mapping[str, Any],
) -> MethodScoringArtifacts:
    """Read back and contract-validate one serialized method bundle."""
    required_names = {"scores.csv", "classifications.csv", "latency-observations.csv"}
    if not method_root.is_dir() or method_root.is_symlink():
        raise V0_2ScoringArtifactError("serialized method artifact inventory is invalid")
    for name in required_names:
        path = method_root / name
        if not path.is_file() or path.is_symlink():
            raise V0_2ScoringArtifactError("serialized method artifact inventory is invalid")
    score_records = [
        {
            **row,
            "score_failure_code": _nullable(row["score_failure_code"]),
            "anomaly_score": float(row["anomaly_score"]),
        }
        for row in _read_csv(method_root / "scores.csv", columns=SCORE_COLUMNS)
    ]
    classification_records = [
        {
            **row,
            "score_failure_code": _nullable(row["score_failure_code"]),
            "anomaly_score": float(row["anomaly_score"]),
            "threshold": float(row["threshold"]),
            "is_anomalous": _boolean(row["is_anomalous"]),
            "score_margin": float(row["score_margin"]),
        }
        for row in _read_csv(
            method_root / "classifications.csv",
            columns=CLASSIFICATION_COLUMNS,
        )
    ]
    latency_records = [
        {
            **row,
            "pass_index": int(row["pass_index"]),
            "adapter_duration_ns": (
                None if row["adapter_duration_ns"] == "" else int(row["adapter_duration_ns"])
            ),
            "scorer_duration_ns": int(row["scorer_duration_ns"]),
            "duration_ns": int(row["duration_ns"]),
            "score_failure_code": _nullable(row["score_failure_code"]),
            "anomaly_score": float(row["anomaly_score"]),
        }
        for row in _read_csv(
            method_root / "latency-observations.csv",
            columns=LATENCY_COLUMNS,
        )
    ]
    try:
        validated_scores = validate_tabular_records("score", score_records, schema=schema)
        validated_classifications = validate_tabular_records(
            "classification",
            classification_records,
            schema=schema,
        )
        validated_latency = validate_tabular_records(
            "latency_observation",
            latency_records,
            schema=schema,
        )
    except (TypeError, ValueError) as error:
        raise V0_2ScoringArtifactError("scoring CSV numeric value is invalid") from error
    if (
        len(validated_scores) != ASSET_COUNT
        or len(validated_classifications) != ASSET_COUNT
        or len(validated_latency) != ASSET_COUNT * TIMED_PASS_COUNT
    ):
        raise V0_2ScoringArtifactError("serialized method artifact counts changed")
    return MethodScoringArtifacts(
        score_records=validated_scores,
        classification_records=validated_classifications,
        latency_records=validated_latency,
    )
