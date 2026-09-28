"""Pure metadata validation and paired summaries for the fixed normal probe."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from few_shot_anomaly_poc.v0_3_diagnostic_contract import (
    ARTIFACT_CONTRACT_VERSION,
    CONDITIONS,
    DIAGNOSTIC_ID,
    METHODS,
    validate_tabular_record,
)

SCORE_NAME = "controlled-scores.csv"
SUMMARY_NAME = "condition-summaries.csv"
BASE_COUNT = 60
RECORD_COUNT = BASE_COUNT * len(CONDITIONS) * len(METHODS)
STORE_SHAPE = (BASE_COUNT * len(CONDITIONS), 512, 512, 3)


class ProbeError(Exception):
    """Stop and preserve an experiment outside the fixed diagnostic contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    ).encode()


def write_once(path: Path, content: bytes) -> None:
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlinked output path")
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def append_audit(path: Path, value: Mapping) -> None:
    content = json.dumps(value, allow_nan=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def validate_partition(records: Sequence[Mapping], *, schema: Mapping) -> None:
    require(len(records) == BASE_COUNT, "normal partition must contain 60 records")
    paths = []
    for rank, row in enumerate(records, 1):
        validate_tabular_record("normal_partition", row, schema=schema)
        require(row["selection_rank"] == rank, "normal partition rank/order changed")
        require(row["relative_path"].startswith("pcb2/Data/Images/Normal/"), "non-normal input")
        require(row["byte_count"] > 0, "empty normal input identity")
        paths.append(row["relative_path"])
    require(len(set(paths)) == BASE_COUNT, "duplicate normal path")


def score_record(
    *,
    method: str,
    selection_rank: int,
    condition: str,
    base_sha256: str,
    status: str,
    failure_code: str | None,
    score: float,
    config: Mapping,
    schema: Mapping,
) -> dict:
    require(method in METHODS and condition in CONDITIONS, "unknown method or condition")
    row = {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "method": method,
        "selection_rank": selection_rank,
        "condition": condition,
        "base_image_sha256": base_sha256,
        "score_status": status,
        "score_failure_code": failure_code,
        "anomaly_score": score,
        "threshold": config["methods"][method]["threshold"],
        "above_threshold": status == "failed" or score > config["methods"][method]["threshold"],
    }
    validate_score(row, config=config, schema=schema)
    return row


def validate_score(row: Mapping, *, config: Mapping, schema: Mapping) -> None:
    validate_tabular_record("diagnostic_score", row, schema=schema)
    require(1 <= row["selection_rank"] <= BASE_COUNT, "score selection rank changed")
    method = config["methods"][row["method"]]
    value = row["anomaly_score"]
    require(type(value) in (float, int) and math.isfinite(value), "non-finite or invalid score")
    require(row["threshold"] == method["threshold"], "fixed threshold changed")
    failed = row["score_status"] == "failed"
    if failed:
        require(
            isinstance(row["score_failure_code"], str)
            and bool(row["score_failure_code"])
            and value == method["failure_score"],
            "invalid failed-score evidence",
        )
    else:
        require(row["score_failure_code"] is None, "successful score has a failure code")
        low, high = method["success_score_minimum"], method["success_score_maximum"]
        require(
            (value >= low if method["success_score_minimum_inclusive"] else value > low)
            and (value <= high if method["success_score_maximum_inclusive"] else value < high),
            "score outside the unchanged method range",
        )
    require(
        row["above_threshold"] == (failed or value > method["threshold"]), "classification changed"
    )


def validate_scores(
    rows: Sequence[Mapping],
    partition: Sequence[Mapping],
    *,
    config: Mapping,
    schema: Mapping,
    methods: Sequence[str] = METHODS,
) -> None:
    validate_partition(partition, schema=schema)
    require(tuple(methods) in (METHODS, (METHODS[2],)), "invalid score-method boundary")
    expected = [
        (method, rank, condition)
        for method in methods
        for rank in range(1, BASE_COUNT + 1)
        for condition in CONDITIONS
    ]
    observed = [(row["method"], row["selection_rank"], row["condition"]) for row in rows]
    require(observed == expected, "score count, duplicate key, or ordering mismatch")
    for row in rows:
        validate_score(row, config=config, schema=schema)
        require(
            row["base_image_sha256"] == partition[row["selection_rank"] - 1]["sha256"],
            "score base-image identity changed",
        )


def summarize_scores(
    rows: Sequence[Mapping], partition: Sequence[Mapping], *, config, schema
) -> list[dict]:
    validate_scores(rows, partition, config=config, schema=schema)
    require(
        all(row["score_status"] == "ok" for row in rows), "failed scores: preserve incomplete run"
    )
    keyed = {(row["method"], row["selection_rank"], row["condition"]): row for row in rows}
    summaries = []
    for method in METHODS:
        controls = [keyed[method, rank, "control"] for rank in range(1, BASE_COUNT + 1)]
        for condition in CONDITIONS:
            selected = [keyed[method, rank, condition] for rank in range(1, BASE_COUNT + 1)]
            deltas = [
                row["anomaly_score"] - control["anomaly_score"]
                for row, control in zip(selected, controls, strict=True)
            ]
            absolute = sorted(abs(delta) for delta in deltas)
            is_control = condition == "control"
            above = sum(row["above_threshold"] for row in selected)
            summary = {
                "contract_version": ARTIFACT_CONTRACT_VERSION,
                "diagnostic_id": DIAGNOSTIC_ID,
                "method": method,
                "condition": condition,
                "attempted_count": BASE_COUNT,
                "successful_count": BASE_COUNT,
                "failed_count": 0,
                "median_signed_delta": None if is_control else statistics.median(deltas),
                "median_absolute_delta": None if is_control else statistics.median(absolute),
                "nearest_rank_p95_absolute_delta": None
                if is_control
                else absolute[math.ceil(0.95 * BASE_COUNT) - 1],
                "above_threshold_count": above,
                "above_threshold_rate": above / BASE_COUNT,
                "normal_to_anomalous_crossing_count": sum(
                    not control["above_threshold"] and row["above_threshold"]
                    for row, control in zip(selected, controls, strict=True)
                ),
                "anomalous_to_normal_crossing_count": sum(
                    control["above_threshold"] and not row["above_threshold"]
                    for row, control in zip(selected, controls, strict=True)
                ),
            }
            validate_tabular_record("condition_summary", summary, schema=schema)
            summaries.append(summary)
    return summaries


def serialize_table(name: str, rows: Sequence[Mapping], *, schema: Mapping) -> bytes:
    fields = [column["name"] for column in schema["contracts"][name]["columns"]]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        validate_tabular_record(name, row, schema=schema)
        writer.writerow(
            {
                key: str(value).lower() if type(value) is bool else value
                for key, value in row.items()
            }
        )
    return stream.getvalue().encode("utf-8")


def read_table(path: Path, name: str, *, schema: Mapping) -> list[dict]:
    columns = schema["contracts"][name]["columns"]
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == [column["name"] for column in columns], "CSV header changed")
        result = []
        for raw in reader:
            require(set(raw) == set(reader.fieldnames), "CSV fields changed")
            row = {}
            for column in columns:
                value = raw[column["name"]]
                if value == "" and column["nullable"]:
                    value = None
                elif column["type"] == "integer":
                    value = int(value)
                elif column["type"] == "number":
                    value = float(value)
                elif column["type"] == "boolean":
                    require(value in {"true", "false"}, "invalid CSV boolean")
                    value = value == "true"
                row[column["name"]] = value
            validate_tabular_record(name, row, schema=schema)
            result.append(row)
    return result


def verify_probe_outputs(output: Path, partition: Sequence[Mapping], *, config, schema) -> None:
    rows = read_table(output / SCORE_NAME, "diagnostic_score", schema=schema)
    expected = summarize_scores(rows, partition, config=config, schema=schema)
    require(
        (output / SCORE_NAME).read_bytes()
        == serialize_table("diagnostic_score", rows, schema=schema),
        "score serialization changed",
    )
    require(
        (output / SUMMARY_NAME).read_bytes()
        == serialize_table("condition_summary", expected, schema=schema),
        "paired summary differs from the complete score evidence",
    )


def bytes_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
