"""Deterministic metadata-only join after the immutable v0.3.4 first pass."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from few_shot_anomaly_poc.v0_3_blinded_review import (
    read_blind_observations_csv,
    validate_review_completion_checkpoint,
)
from few_shot_anomaly_poc.v0_3_diagnostic_contract import (
    EXPECTED_ASSET_IDS,
    EXPECTED_METHOD_CASES,
    METHODS,
    load_v0_3_config,
    load_v0_3_schema,
    sha256_file,
    validate_blind_observation_records,
    validate_tabular_record,
)
from few_shot_anomaly_poc.v0_3_first_observation import (
    ARTIFACT_ROOT,
    INVENTORY_HASHES,
    PARENT_ROOT,
)
from few_shot_anomaly_poc.v0_3_inventory import REVIEW_CASE_FIELDS, serialize_csv

OBSERVATION_COMMIT = "73462a529809705ba555dacecd8a10c671936537"
OBSERVATION_HASHES = {
    "blind-observations.csv": ("93a6303824dc9b24819b4027f409b1acb3752086ca72921ac30a2cebbe6783f3"),
    "review-completion-checkpoint.json": (
        "3c9b56ecd9761ca2a35b3bfbb17fefff5021776b33cc082f8cf1c15e72767dbd"
    ),
}
JOIN_NAME = "observation-case-join.csv"


class V0_3ObservationJoinError(Exception):
    """Reject changed evidence, incomplete observations, or an existing output."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V0_3ObservationJoinError(message)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def check_pushed_source(root: Path) -> str:
    """Require clean pushed code descending from the completed observations."""
    _require(not _git(root, "status", "--porcelain"), "worktree must be clean")
    source = _git(root, "rev-parse", "HEAD")
    _require(source == _git(root, "rev-parse", "origin/main"), "HEAD != origin/main")
    remote = _git(root, "ls-remote", "origin", "refs/heads/main").split()
    _require(bool(remote) and remote[0] == source, "implementation must be pushed to origin/main")
    _git(root, "merge-base", "--is-ancestor", OBSERVATION_COMMIT, source)
    return source


def _verify_file(path: Path, expected: str) -> None:
    _require(
        path.is_file() and not any(part.is_symlink() for part in (path, *path.parents)),
        f"required regular metadata file is missing: {path}",
    )
    _require(sha256_file(path) == expected, f"fixed metadata SHA-256 changed: {path}")


def _validate_links(links: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]) -> None:
    for record in links:
        validate_tabular_record("review_case_link", record, schema=schema)
    identities = tuple(
        (row["method"], row["case_type"], row["rank"], row["asset_id"]) for row in links
    )
    _require(identities == EXPECTED_METHOD_CASES, "method-case identities or order changed")


def load_join_inputs(root: Path) -> tuple[dict, dict, list[dict], list[dict]]:
    """Verify only committed metadata; no external image or fitted-state access."""
    config = load_v0_3_config(root / "configs/v0.3.yaml")
    schema = load_v0_3_schema(root / "schemas/v0.3/diagnostic-artifacts.json")
    for name, digest in {**INVENTORY_HASHES, **OBSERVATION_HASHES}.items():
        _verify_file(root / ARTIFACT_ROOT / name, digest)
    parent = config["parent_evidence"]
    for relative, digest in (
        (config["preregistration"]["document"], config["preregistration"]["document_sha256"]),
        ("configs/v0.2.yaml", parent["v0_2_config_sha256"]),
        ("schemas/v0.2/evaluation-artifacts.json", parent["v0_2_schema_sha256"]),
        ("uv.lock", config["dependencies"]["root_lock_sha256"]),
        ("environments/v0.2-preflight/uv.lock", config["dependencies"]["dinov2_lock_sha256"]),
        (PARENT_ROOT / "artifact-manifest.json", parent["artifact_manifest_sha256"]),
    ):
        _verify_file(root / relative, digest)
    manifest = json.loads((root / PARENT_ROOT / "artifact-manifest.json").read_text("utf-8"))
    for entry in manifest["files"]:
        _verify_file(root / PARENT_ROOT / entry["relative_path"], entry["sha256"])
    checkpoint = validate_review_completion_checkpoint(
        json.loads((root / ARTIFACT_ROOT / "review-completion-checkpoint.json").read_text("utf-8"))
    )
    _require(
        checkpoint["review_assets_sha256"] == INVENTORY_HASHES["review-assets.csv"]
        and checkpoint["blind_observations_sha256"] == OBSERVATION_HASHES["blind-observations.csv"],
        "completion checkpoint does not bind the fixed inputs",
    )
    observations = read_blind_observations_csv(
        root / ARTIFACT_ROOT / "blind-observations.csv",
        expected_sha256=checkpoint["blind_observations_sha256"],
        schema=schema,
    )
    with (root / ARTIFACT_ROOT / "review-case-linkage.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        _require(tuple(reader.fieldnames or ()) == REVIEW_CASE_FIELDS, "linkage header changed")
        links = [{**row, "rank": int(row["rank"])} for row in reader]
    _validate_links(links, schema)
    for link in links:
        _require(
            link["parent_failure_sha256"] == parent["failure_case_sha256"][link["method"]],
            "linkage parent identity changed",
        )
    return config, schema, observations, links


def join_observations(
    observations: Sequence[Mapping[str, Any]],
    links: Sequence[Mapping[str, Any]],
    *,
    schema: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Copy every form verbatim to its exact fixed method-case rows."""
    validated = validate_blind_observation_records(
        observations, expected_asset_ids=EXPECTED_ASSET_IDS, schema=schema, require_complete=True
    )
    _validate_links(links, schema)
    by_asset = {row["asset_id"]: row for row in validated}
    fields = [column["name"] for column in schema["contracts"]["observation_join"]["columns"]]
    result = []
    for link in links:
        combined = {**by_asset[link["asset_id"]], **link}
        result.append(
            validate_tabular_record(
                "observation_join", {name: combined[name] for name in fields}, schema=schema
            )
        )
    return result


def serialize_observation_join(records: Sequence[Mapping[str, Any]], *, schema: Mapping) -> bytes:
    fields = [column["name"] for column in schema["contracts"]["observation_join"]["columns"]]
    identities = tuple(
        (row["method"], row["case_type"], row["rank"], row["asset_id"]) for row in records
    )
    _require(identities == EXPECTED_METHOD_CASES, "joined identities or order changed")
    for record in records:
        validate_tabular_record("observation_join", record, schema=schema)
    return serialize_csv(fields, records)


def write_observation_join(path: Path, content: bytes) -> str:
    """Write exclusively and preserve partial storage failures for inspection."""
    _require(not path.exists() and not path.is_symlink(), "join output already exists")
    _require(
        not any(parent.is_symlink() for parent in path.parents), "join output ancestor is symlinked"
    )
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_file(path)


def visual_yes_counts(records: Sequence[Mapping[str, Any]], *, config: Mapping) -> dict:
    """Count unique assets per method-family, without deciding diagnostic support."""
    families = config["signal_rule"]["families"]
    return {
        method: {
            family: len(
                {
                    row["asset_id"]
                    for row in records
                    if row["method"] == method
                    and any(row[field] == "yes" for field in definition["visual_fields"])
                }
            )
            for family, definition in families.items()
        }
        for method in METHODS
    }
