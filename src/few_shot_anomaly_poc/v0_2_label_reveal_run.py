"""Execute the irreversible v0.2.7 label reveal, metrics, and failure selection."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.jsonio import write_json_atomic
from few_shot_anomaly_poc.opaque_boundary import (
    load_scoring_manifest,
    load_sealed_mapping,
)
from few_shot_anomaly_poc.v0_2_boundary_preparation import (
    RUN_ID,
    validate_boundary_execution_identity,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    METHODS,
    load_v0_2_artifact_schema,
    load_v0_2_config,
    validate_json_artifact,
)
from few_shot_anomaly_poc.v0_2_offline_reproduction_run import (
    read_reproduction_csv,
)
from few_shot_anomaly_poc.v0_2_pre_reveal_checkpoint import (
    CHECKPOINT_NAME,
    FIXED_HASHES,
    label_free_bundle_sha256,
)
from few_shot_anomaly_poc.v0_2_revealed_evaluation import (
    ASSET_COUNT,
    RevealedEvaluationArtifacts,
    build_label_reveal_records,
    build_revealed_evaluation_artifacts,
    read_failure_cases_csv,
    read_label_reveal_csv,
    read_metrics_json,
    write_revealed_evaluation_artifacts,
)
from few_shot_anomaly_poc.v0_2_scoring_artifacts import (
    MethodScoringArtifacts,
    read_method_scoring_artifacts,
)

MILESTONE = "v0.2.7"
EVIDENCE_COMMIT = "30e830e3f805f2410c669c0b8c2da6cb27894ca5"
SCORING_SOURCE_COMMIT = "ba23a2fe12a715161b420bc7d73d42f4de3bfc8c"
OPAQUE_SCORING_MANIFEST_SHA256 = "32ea52ed1b9872f39ae27f5d58a353ea84b8b143642e3a7f0fabe940184705e8"
SEALED_MAPPING_SHA256 = "0e40e3777e797fd5099fd7ea8fd307547e396785dc33219f0bc0b3120e3dfe39"
PRE_REVEAL_CHECKPOINT_SHA256 = "4bf7d63da308868933a8967a6922fc439f95a524bf54cd8a38ff39f894219b07"
LABEL_FREE_BUNDLE_SHA256 = "b3fb2a1283117350cf3c016ae284d3558aa7ff9ea1cf1c6beb65eba54c5cb389"
REPRODUCTION_HASHES = {
    "ecc_residual/offline-reproduction.csv": (
        "beb6a4211885e12623b53739aca23dabe62ad6968296cc6364dbb019b1341833"
    ),
    "patch_hog_ocsvm/offline-reproduction.csv": (
        "7fa501f12c3ab76bbfc1bf2d88ca79d74df8eb82b0b8e61e9544017122578f0d"
    ),
    "dinov2_vits14_224_nn/offline-reproduction.csv": (
        "f75976fb3ef3c17db6f5ee4150bcba4a9c3bb79d6172920e22345f9e59c93dd5"
    ),
}
PRE_REVEAL_HASHES = {
    **FIXED_HASHES,
    **REPRODUCTION_HASHES,
    CHECKPOINT_NAME: PRE_REVEAL_CHECKPOINT_SHA256,
}


class V0_2LabelRevealRunError(Exception):
    """Reject an unsafe reveal, incomplete join, overwrite, or post-reveal retry."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V0_2LabelRevealRunError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V0_2LabelRevealRunError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise V0_2LabelRevealRunError(f"{label} must contain one JSON object")
    return value


def _require_ancestor(project_root: Path, ancestor: str, descendant: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=project_root,
        check=False,
        capture_output=True,
    )
    _require(completed.returncode == 0, "execution commit lacks fixed pre-reveal lineage")


def _require_tracked_bytes(
    project_root: Path,
    *,
    repository_relative_path: str,
    expected_sha256: str,
) -> None:
    completed = subprocess.run(
        ["git", "show", f"HEAD:{repository_relative_path}"],
        cwd=project_root,
        check=False,
        capture_output=True,
    )
    _require(
        completed.returncode == 0
        and hashlib.sha256(completed.stdout).hexdigest() == expected_sha256,
        f"pre-reveal artifact is not fixed at HEAD: {repository_relative_path}",
    )


def _validate_pre_reveal_public_root(
    *,
    project_root: Path,
    public_root: Path,
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, MethodScoringArtifacts]]:
    observed_paths = {
        path.relative_to(public_root).as_posix()
        for path in public_root.rglob("*")
        if path.is_file()
    }
    _require(
        observed_paths == set(PRE_REVEAL_HASHES),
        "pre-reveal public artifact inventory changed",
    )
    for relative_path, expected_sha256 in PRE_REVEAL_HASHES.items():
        path = public_root / relative_path
        _require(
            path.is_file() and not path.is_symlink() and sha256_file(path) == expected_sha256,
            f"pre-reveal artifact identity changed: {relative_path}",
        )
        _require_tracked_bytes(
            project_root,
            repository_relative_path=path.relative_to(project_root).as_posix(),
            expected_sha256=expected_sha256,
        )

    checkpoint = validate_json_artifact(
        "pre_reveal_checkpoint",
        _read_json(public_root / CHECKPOINT_NAME, label="pre-reveal checkpoint"),
        config=config,
        schema=schema,
    )
    _require(
        checkpoint["source_commit"] == SCORING_SOURCE_COMMIT
        and checkpoint["git_commit"] == EVIDENCE_COMMIT
        and checkpoint["git_push_verified"] is True
        and checkpoint["labels_accessed"] is False
        and checkpoint["method_order"] == list(METHODS)
        and checkpoint["method_score_counts"] == {method: ASSET_COUNT for method in METHODS}
        and checkpoint["reproduction_status"] == {method: "pass" for method in METHODS}
        and checkpoint["label_free_bundle_sha256"] == LABEL_FREE_BUNDLE_SHA256,
        "pre-reveal checkpoint fields changed",
    )
    bundle_paths = sorted(path for path in PRE_REVEAL_HASHES if path != CHECKPOINT_NAME)
    _require(
        len(bundle_paths) == 23
        and label_free_bundle_sha256(public_root, bundle_paths) == LABEL_FREE_BUNDLE_SHA256,
        "label-free bundle identity changed",
    )
    scoring_by_method: dict[str, MethodScoringArtifacts] = {}
    for method in METHODS:
        scoring_by_method[method] = read_method_scoring_artifacts(
            public_root / method,
            schema=schema,
        )
        reproduction = read_reproduction_csv(
            public_root / method / "offline-reproduction.csv",
            schema=schema,
        )
        _require(
            all(record["within_tolerance"] for record in reproduction),
            f"{method} reproduction did not pass",
        )
    return checkpoint, scoring_by_method


def _validate_external_pre_reveal_state(state: Mapping[str, Any]) -> None:
    checkpoint = state.get("pre_reveal_checkpoint")
    reproduction = state.get("offline_reproduction")
    _require(
        state.get("schema_version") == "v0.2-boundary-state-v1"
        and state.get("run_id") == RUN_ID
        and state.get("boundary", {}).get("anomaly_score_computed") is True
        and state.get("boundary", {}).get("final_test_label_revealed") is False
        and state.get("opaque_final_test", {}).get("asset_count") == ASSET_COUNT
        and state.get("opaque_final_test", {}).get("scoring_manifest_sha256")
        == OPAQUE_SCORING_MANIFEST_SHA256
        and state.get("opaque_final_test", {}).get("sealed_mapping_sha256") == SEALED_MAPPING_SHA256
        and isinstance(reproduction, dict)
        and reproduction.get("reproduction_status") == {method: "pass" for method in METHODS}
        and reproduction.get("labels_accessed") is False
        and isinstance(checkpoint, dict)
        and checkpoint.get("label_free_evidence_commit") == EVIDENCE_COMMIT
        and checkpoint.get("label_free_bundle_sha256") == LABEL_FREE_BUNDLE_SHA256
        and checkpoint.get("git_push_verified") is True
        and checkpoint.get("labels_accessed") is False
        and "label_reveal_evaluation" not in state,
        "external state is not the fixed unrevealed checkpoint state",
    )


def _preflight_inputs(
    *,
    project_root: Path,
    execution_commit: str,
    external_root: Path,
    public_root: Path,
    work_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, MethodScoringArtifacts],
    tuple[str, ...],
    Path,
]:
    project_root = project_root.resolve()
    external_root = external_root.resolve()
    public_root = public_root.resolve()
    work_root = work_root.resolve()
    _require(
        external_root == (project_root / f"data/external/v0.2/evaluation/{RUN_ID}").resolve()
        and public_root == (project_root / f"artifacts/v0.2/evaluation/{RUN_ID}").resolve()
        and work_root == (project_root / f"work/v0.2/evaluation/{RUN_ID}").resolve(),
        "evaluation roots differ from the fixed contract",
    )
    validate_boundary_execution_identity(
        project_root=project_root, execution_commit=execution_commit
    )
    _require_ancestor(project_root, EVIDENCE_COMMIT, execution_commit)
    config = load_v0_2_config(project_root / "configs/v0.2.yaml")
    schema = load_v0_2_artifact_schema(project_root / "schemas/v0.2/evaluation-artifacts.json")
    _checkpoint, scoring_by_method = _validate_pre_reveal_public_root(
        project_root=project_root,
        public_root=public_root,
        config=config,
        schema=schema,
    )
    state_path = external_root / "boundary-state.json"
    state = _read_json(state_path, label="boundary state")
    _validate_external_pre_reveal_state(state)
    _require(
        not (work_root / "label-reveal-evaluation-state.json").exists(),
        "local label-reveal state already exists",
    )

    scoring_manifest_path = external_root / "scorer/scoring-manifest.json"
    sealed_mapping_path = external_root / "sealed/mapping.json"
    _require(
        sha256_file(scoring_manifest_path) == OPAQUE_SCORING_MANIFEST_SHA256
        and sha256_file(sealed_mapping_path) == SEALED_MAPPING_SHA256,
        "scoring or sealed mapping identity changed",
    )
    scoring_manifest = load_scoring_manifest(scoring_manifest_path)
    scoring_records = scoring_manifest["records"]
    _require(
        len(scoring_records) == ASSET_COUNT,
        "opaque scoring manifest count changed",
    )
    expected_ids = tuple(record["asset_id"] for record in scoring_records)
    _require(
        expected_ids == tuple(f"asset-{index:06d}" for index in range(ASSET_COUNT)),
        "opaque scoring manifest order changed",
    )
    return config, schema, state, scoring_by_method, expected_ids, sealed_mapping_path


def _mark_reveal_started(
    *,
    state_path: Path,
    state: Mapping[str, Any],
    execution_commit: str,
) -> dict[str, Any]:
    updated = dict(state)
    boundary = dict(updated["boundary"])
    boundary["final_test_label_revealed"] = True
    updated["boundary"] = boundary
    updated["label_reveal_evaluation"] = {
        "milestone": MILESTONE,
        "execution_commit": execution_commit,
        "status": "in_progress",
        "sealed_mapping_sha256": SEALED_MAPPING_SHA256,
        "labels_accessed": True,
        "image_content_accessed": False,
        "image_content_displayed": False,
        "official_split_accessed": False,
        "score_or_threshold_modified": False,
        "decision_generated": False,
    }
    write_json_atomic(state_path, updated, overwrite=True)
    return updated


def _artifact_identities(public_root: Path) -> list[dict[str, Any]]:
    paths = [public_root / "revealed-labels.csv"]
    for method in METHODS:
        paths.extend(
            (
                public_root / method / "metrics.json",
                public_root / method / "failure-cases.csv",
            )
        )
    return [
        {
            "relative_path": path.relative_to(public_root).as_posix(),
            "byte_count": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def _validate_staged(
    *,
    stage_root: Path,
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> None:
    labels = read_label_reveal_csv(stage_root / "revealed-labels.csv", schema=schema)
    _require(len(labels) == ASSET_COUNT, "serialized reveal count changed")
    for method in METHODS:
        metrics = read_metrics_json(
            stage_root / method / "metrics.json",
            config=config,
            schema=schema,
        )
        failures = read_failure_cases_csv(
            stage_root / method / "failure-cases.csv",
            schema=schema,
        )
        _require(
            metrics["method"] == method
            and len(failures)
            == min(metrics["false_positive_count"], 5) + min(metrics["false_negative_count"], 5),
            f"{method} serialized failure selection is incomplete",
        )


def run_v0_2_label_reveal_evaluation(
    *,
    project_root: Path,
    execution_commit: str,
    external_root: Path,
    public_root: Path,
    work_root: Path,
) -> dict[str, int]:
    """Cross the reveal boundary once and publish no decision or image pixels."""
    project_root = project_root.resolve()
    external_root = external_root.resolve()
    public_root = public_root.resolve()
    work_root = work_root.resolve()
    (
        config,
        schema,
        state,
        scoring_by_method,
        expected_ids,
        sealed_mapping_path,
    ) = _preflight_inputs(
        project_root=project_root,
        execution_commit=execution_commit,
        external_root=external_root,
        public_root=public_root,
        work_root=work_root,
    )
    stage_root = Path(tempfile.mkdtemp(dir=work_root, prefix=".v0.2.7-reveal-"))
    public_stage = stage_root / "public"
    state_path = external_root / "boundary-state.json"
    reveal_started = False
    moved: list[Path] = []
    try:
        sealed_mapping = load_sealed_mapping(sealed_mapping_path)
        reveal_started = True
        state = _mark_reveal_started(
            state_path=state_path,
            state=state,
            execution_commit=execution_commit,
        )
        label_records = build_label_reveal_records(
            sealed_mapping["records"],
            expected_asset_ids=expected_ids,
            schema=schema,
        )
        artifacts: RevealedEvaluationArtifacts = build_revealed_evaluation_artifacts(
            label_records=label_records,
            scoring_by_method=scoring_by_method,
            config=config,
            schema=schema,
        )
        write_revealed_evaluation_artifacts(public_stage, artifacts)
        _validate_staged(stage_root=public_stage, config=config, schema=schema)

        source_paths = [public_stage / "revealed-labels.csv"]
        destinations = [public_root / "revealed-labels.csv"]
        for method in METHODS:
            source_paths.extend(
                (
                    public_stage / method / "metrics.json",
                    public_stage / method / "failure-cases.csv",
                )
            )
            destinations.extend(
                (
                    public_root / method / "metrics.json",
                    public_root / method / "failure-cases.csv",
                )
            )
        _require(
            all(not destination.exists() for destination in destinations),
            "v0.2.7 public output already exists",
        )
        for source, destination in zip(source_paths, destinations, strict=True):
            os.replace(source, destination)
            moved.append(destination)
        identities = _artifact_identities(public_root)
        record_counts = {
            "revealed_labels": ASSET_COUNT,
            **{
                method: {
                    "metrics": 1,
                    "failure_cases": len(artifacts.failures_by_method[method]),
                }
                for method in METHODS
            },
        }
        local_state = {
            "schema_version": "v0.2.7-label-reveal-evaluation-state-v1",
            "milestone": MILESTONE,
            "run_id": RUN_ID,
            "execution_commit": execution_commit,
            "source_evidence_commit": EVIDENCE_COMMIT,
            "pre_reveal_checkpoint_sha256": PRE_REVEAL_CHECKPOINT_SHA256,
            "sealed_mapping_sha256": SEALED_MAPPING_SHA256,
            "artifact_identities": identities,
            "record_counts": record_counts,
            "labels_accessed": True,
            "image_content_accessed": False,
            "image_content_displayed": False,
            "official_split_accessed": False,
            "score_or_threshold_modified": False,
            "decision_generated": False,
        }
        write_json_atomic(work_root / "label-reveal-evaluation-state.json", local_state)
        completed_state = dict(state)
        stage = dict(completed_state["label_reveal_evaluation"])
        stage.update(
            {
                "status": "completed",
                "artifact_identities": identities,
                "record_counts": record_counts,
            }
        )
        completed_state["label_reveal_evaluation"] = stage
        opaque = dict(completed_state["opaque_final_test"])
        opaque["class_counts_published"] = True
        completed_state["opaque_final_test"] = opaque
        write_json_atomic(state_path, completed_state, overwrite=True)
        return {
            "revealed_labels": ASSET_COUNT,
            **{method: len(artifacts.failures_by_method[method]) for method in METHODS},
        }
    except Exception as error:
        for path in moved:
            path.unlink(missing_ok=True)
        if reveal_started:
            failed_state = _read_json(state_path, label="revealed boundary state")
            stage = dict(failed_state.get("label_reveal_evaluation", {}))
            stage.update(
                {
                    "milestone": MILESTONE,
                    "execution_commit": execution_commit,
                    "status": "failed",
                    "labels_accessed": True,
                    "image_content_accessed": False,
                    "image_content_displayed": False,
                    "official_split_accessed": False,
                    "score_or_threshold_modified": False,
                    "decision_generated": False,
                    "failure_type": type(error).__name__,
                }
            )
            failed_state["label_reveal_evaluation"] = stage
            write_json_atomic(state_path, failed_state, overwrite=True)
        raise
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
