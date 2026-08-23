"""Execute v0.2.8 decisions from committed numerical and process evidence."""

from __future__ import annotations

import csv
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
from few_shot_anomaly_poc.v0_2_boundary_preparation import RUN_ID
from few_shot_anomaly_poc.v0_2_decision import (
    MethodDecisionEvidence,
    build_method_decision,
    build_project_decision,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    ARTIFACT_CONTRACT_VERSION,
    METHODS,
    load_v0_2_artifact_schema,
    load_v0_2_config,
    validate_json_artifact,
)
from few_shot_anomaly_poc.v0_2_label_reveal_run import PRE_REVEAL_HASHES
from few_shot_anomaly_poc.v0_2_offline_reproduction_run import read_reproduction_csv
from few_shot_anomaly_poc.v0_2_revealed_evaluation import (
    read_failure_cases_csv,
    read_label_reveal_csv,
    read_metrics_json,
)
from few_shot_anomaly_poc.v0_2_scoring_artifacts import (
    latency_summary,
    read_method_scoring_artifacts,
)

MILESTONE = "v0.2.8"
RUN_KIND = "final_test"
REVEALED_EVIDENCE_COMMIT = "3044c6d8fa16705819674010b566880e4a295940"
REVEALED_HASHES = {
    "revealed-labels.csv": (
        "f1602ebf21241e7579f9be9092b90db8e9ca1fc9822b984130eb464d804d595c"
    ),
    "ecc_residual/metrics.json": (
        "c7ff3c98e9c350d33406d647c1ea5100e25addb128c9baf39a78e02a25a286a7"
    ),
    "ecc_residual/failure-cases.csv": (
        "1bd73cf3b0050b3eff934467625ba118d662e442ceaa5829f8a47344b4f390e8"
    ),
    "patch_hog_ocsvm/metrics.json": (
        "fd8c61a99e00109e8f5ec8c68fbbb8138042d908cda286b38ec06b67bf79d233"
    ),
    "patch_hog_ocsvm/failure-cases.csv": (
        "7edde3c32fd23e9d53db0c665595bb3c6789a1ce52bf039810b9d7646d3387ff"
    ),
    "dinov2_vits14_224_nn/metrics.json": (
        "22f5cca3187d974d69efde8f495c70cbcef45c193123704010c10534f1a2fb29"
    ),
    "dinov2_vits14_224_nn/failure-cases.csv": (
        "ad8f05152f936b8b9da5a6d94f2f8dc9eab43035373836f949dbb3a7887c22e9"
    ),
}
FIXED_INPUT_HASHES = {**PRE_REVEAL_HASHES, **REVEALED_HASHES}


class V0_2DecisionRunError(Exception):
    """Reject changed inputs, unsafe state, overwrite, or an unpushed runner."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V0_2DecisionRunError(message)


def _git(project_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise V0_2DecisionRunError(f"git {' '.join(arguments)} failed")
    return completed.stdout.strip()


def _require_execution_identity(project_root: Path, execution_commit: str) -> None:
    _require(
        len(execution_commit) == 40
        and all(character in "0123456789abcdef" for character in execution_commit),
        "execution commit must be one full lowercase Git SHA",
    )
    _require(_git(project_root, "rev-parse", "HEAD") == execution_commit, "HEAD changed")
    _require(
        not _git(project_root, "status", "--porcelain", "--untracked-files=all"),
        "worktree must be clean before decision generation",
    )
    _require(
        _git(project_root, "rev-parse", "origin/main") == execution_commit,
        "decision runner commit must be pushed to origin/main",
    )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", REVEALED_EVIDENCE_COMMIT, execution_commit],
        cwd=project_root,
        check=False,
        capture_output=True,
    )
    _require(ancestor.returncode == 0, "execution commit lacks revealed-evidence lineage")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V0_2DecisionRunError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise V0_2DecisionRunError(f"{label} must contain one object")
    return value


def _tracked_at_head(
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
        f"decision input is not fixed at HEAD: {repository_relative_path}",
    )


def _validate_state(state: Mapping[str, Any]) -> None:
    scoring = state.get("label_free_scoring")
    calibration = state.get("normal_only_fit_calibration")
    reproduction = state.get("offline_reproduction")
    checkpoint = state.get("pre_reveal_checkpoint")
    reveal = state.get("label_reveal_evaluation")
    _require(
        state.get("schema_version") == "v0.2-boundary-state-v1"
        and state.get("run_id") == RUN_ID
        and state.get("boundary", {}).get("final_test_label_revealed") is True
        and isinstance(scoring, dict)
        and scoring.get("anomaly_labels_used") is False
        and scoring.get("semantic_paths_accessed") is False
        and scoring.get("sealed_mapping_accessed") is False
        and scoring.get("official_split_accessed") is False
        and isinstance(calibration, dict)
        and calibration.get("anomaly_labels_used") is False
        and calibration.get("final_test_accessed") is False
        and isinstance(reproduction, dict)
        and reproduction.get("labels_accessed") is False
        and reproduction.get("semantic_paths_accessed") is False
        and reproduction.get("sealed_mapping_accessed") is False
        and reproduction.get("official_split_accessed") is False
        and isinstance(checkpoint, dict)
        and checkpoint.get("git_push_verified") is True
        and checkpoint.get("labels_accessed") is False
        and isinstance(reveal, dict)
        and reveal.get("milestone") == "v0.2.7"
        and reveal.get("status") == "completed"
        and reveal.get("labels_accessed") is True
        and reveal.get("image_content_accessed") is False
        and reveal.get("image_content_displayed") is False
        and reveal.get("official_split_accessed") is False
        and reveal.get("score_or_threshold_modified") is False
        and reveal.get("decision_generated") is False
        and "hard_gate_decision" not in state,
        "boundary state is not the fixed post-v0.2.7 pre-decision state",
    )


def _validate_inputs(
    *,
    project_root: Path,
    public_root: Path,
    external_root: Path,
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> tuple[
    dict[str, MethodDecisionEvidence],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    observed = {
        path.relative_to(public_root).as_posix()
        for path in public_root.rglob("*")
        if path.is_file()
    }
    _require(observed == set(FIXED_INPUT_HASHES), "pre-decision artifact inventory changed")
    for relative_path, expected_sha256 in FIXED_INPUT_HASHES.items():
        path = public_root / relative_path
        _require(
            path.is_file() and not path.is_symlink() and sha256_file(path) == expected_sha256,
            f"fixed decision evidence changed: {relative_path}",
        )
        _tracked_at_head(
            project_root,
            repository_relative_path=path.relative_to(project_root).as_posix(),
            expected_sha256=expected_sha256,
        )

    state = _read_json(external_root / "boundary-state.json", label="boundary state")
    _validate_state(state)
    labels = read_label_reveal_csv(public_root / "revealed-labels.csv", schema=schema)
    _require(len(labels) == 200, "revealed label count changed")

    evidence_by_method: dict[str, MethodDecisionEvidence] = {}
    metrics_by_method: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        fit = validate_json_artifact(
            "fit",
            _read_json(public_root / method / "fit.json", label=f"{method} fit"),
            config=config,
            schema=schema,
        )
        scoring = read_method_scoring_artifacts(public_root / method, schema=schema)
        metrics = read_metrics_json(
            public_root / method / "metrics.json",
            config=config,
            schema=schema,
        )
        failures = read_failure_cases_csv(
            public_root / method / "failure-cases.csv",
            schema=schema,
        )
        _require(bool(failures), f"{method} failure selection is empty")
        reproduction = read_reproduction_csv(
            public_root / method / "offline-reproduction.csv",
            schema=schema,
        )
        latency = latency_summary(scoring.latency_records)
        evidence_by_method[method] = MethodDecisionEvidence(
            fit_status=fit["status"],
            test_leakage_detected=False,
            normal_false_positive_rate=float(metrics["normal_false_positive_rate"]),
            anomaly_recall=float(metrics["anomaly_recall"]),
            cpu_median_latency_seconds=float(latency["median_latency_ns"]) / 1_000_000_000,
            cpu_p95_latency_seconds=float(latency["p95_latency_ns"]) / 1_000_000_000,
            normal_reference_count=int(fit["reference_count"]),
            anomaly_training_labels_used=False,
            reproducibility_verified=all(row["within_tolerance"] for row in reproduction),
        )
        metrics_by_method[method] = metrics
    return evidence_by_method, metrics_by_method, state


def _artifact_type(relative_path: str) -> str:
    name = Path(relative_path).name
    return {
        "boundary-record.json": "boundary_record",
        "pre-evaluation-freeze.json": "pre_evaluation_freeze",
        "fit.json": "fit",
        "calibration-scores.csv": "calibration_score",
        "calibration-summary.json": "calibration_summary",
        "scores.csv": "score",
        "classifications.csv": "classification",
        "latency-observations.csv": "latency_observation",
        "offline-reproduction.csv": "reproduction",
        "pre-reveal-checkpoint.json": "pre_reveal_checkpoint",
        "revealed-labels.csv": "label_reveal",
        "metrics.json": "metrics",
        "failure-cases.csv": "failure_case",
        "decision.json": "method_decision",
        "project-decision.json": "project_decision",
    }[name]


def _record_count(path: Path) -> int:
    if path.suffix == ".json":
        return 1
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
    except (csv.Error, OSError, UnicodeError) as error:
        raise V0_2DecisionRunError(f"cannot count {path.name}") from error
    _require(bool(rows), f"{path.name} is empty")
    return len(rows) - 1


def _build_manifest(
    *,
    project_root: Path,
    public_root: Path,
    stage_root: Path,
    execution_commit: str,
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    existing = {
        path.relative_to(public_root).as_posix(): path
        for path in public_root.rglob("*")
        if path.is_file()
    }
    staged = {
        path.relative_to(stage_root).as_posix(): path
        for path in stage_root.rglob("*")
        if path.is_file()
    }
    _require(not set(existing).intersection(staged), "staged decision output would overwrite")
    paths = {**existing, **staged}
    files = []
    for relative_path in sorted(paths):
        path = paths[relative_path]
        first_component = relative_path.split("/", maxsplit=1)[0]
        files.append(
            {
                "artifact_type": _artifact_type(relative_path),
                "method": first_component if first_component in METHODS else None,
                "record_count": _record_count(path),
                "relative_path": relative_path,
                "sha256": sha256_file(path),
            }
        )
    return validate_json_artifact(
        "bundle_manifest",
        {
            "contract_version": ARTIFACT_CONTRACT_VERSION,
            "run_id": RUN_ID,
            "run_kind": RUN_KIND,
            "source_commit": execution_commit,
            "config_sha256": sha256_file(project_root / "configs/v0.2.yaml"),
            "schema_sha256": sha256_file(
                project_root / "schemas/v0.2/evaluation-artifacts.json"
            ),
            "preregistration_commit": config["preregistration"]["commit"],
            "preregistration_document_sha256": config["preregistration"][
                "document_sha256"
            ],
            "scoring_manifest_sha256": state["opaque_final_test"][
                "scoring_manifest_sha256"
            ],
            "sealed_mapping_sha256": state["opaque_final_test"]["sealed_mapping_sha256"],
            "files": files,
        },
        config=config,
        schema=schema,
    )


def run_v0_2_decision(
    *,
    project_root: Path,
    execution_commit: str,
    public_root: Path,
    external_root: Path,
    work_root: Path,
) -> dict[str, Any]:
    """Create the one fixed decision bundle without image or raw-data access."""
    project_root = project_root.resolve()
    public_root = public_root.resolve()
    external_root = external_root.resolve()
    work_root = work_root.resolve()
    _require(
        public_root == (project_root / f"artifacts/v0.2/evaluation/{RUN_ID}").resolve()
        and external_root
        == (project_root / f"data/external/v0.2/evaluation/{RUN_ID}").resolve()
        and work_root == (project_root / f"work/v0.2/evaluation/{RUN_ID}").resolve(),
        "decision roots differ from the fixed contract",
    )
    _require_execution_identity(project_root, execution_commit)
    config = load_v0_2_config(project_root / "configs/v0.2.yaml")
    schema = load_v0_2_artifact_schema(
        project_root / "schemas/v0.2/evaluation-artifacts.json"
    )
    evidence, _metrics, state = _validate_inputs(
        project_root=project_root,
        public_root=public_root,
        external_root=external_root,
        config=config,
        schema=schema,
    )
    output_paths = [public_root / method / "decision.json" for method in METHODS]
    output_paths.extend(
        (public_root / "project-decision.json", public_root / "artifact-manifest.json")
    )
    _require(all(not path.exists() for path in output_paths), "v0.2.8 output already exists")
    local_state_path = work_root / "hard-gate-decision-state.json"
    _require(not local_state_path.exists(), "local decision state already exists")

    method_decisions = {
        method: build_method_decision(
            method,
            evidence[method],
            config=config,
            schema=schema,
        )
        for method in METHODS
    }
    project_decision = build_project_decision(
        method_decisions,
        evidence,
        config=config,
        schema=schema,
    )

    stage_root = Path(tempfile.mkdtemp(dir=work_root, prefix=".v0.2.8-decision-"))
    moved: list[Path] = []
    try:
        for method in METHODS:
            write_json_atomic(stage_root / method / "decision.json", method_decisions[method])
        write_json_atomic(stage_root / "project-decision.json", project_decision)
        manifest = _build_manifest(
            project_root=project_root,
            public_root=public_root,
            stage_root=stage_root,
            execution_commit=execution_commit,
            config=config,
            schema=schema,
            state=state,
        )
        write_json_atomic(stage_root / "artifact-manifest.json", manifest)
        for destination in output_paths:
            source = stage_root / destination.relative_to(public_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            moved.append(destination)

        identities = [
            {
                "relative_path": path.relative_to(public_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in output_paths
        ]
        local_state = {
            "schema_version": "v0.2.8-hard-gate-decision-state-v1",
            "milestone": MILESTONE,
            "run_id": RUN_ID,
            "execution_commit": execution_commit,
            "method_decisions": {
                method: method_decisions[method]["decision"] for method in METHODS
            },
            "project_decision": project_decision["decision"],
            "selected_method": project_decision["selected_method"],
            "artifact_identities": identities,
            "image_content_accessed": False,
            "image_content_displayed": False,
            "weighted_score_used": False,
            "hard_gate_waiver_used": False,
        }
        write_json_atomic(local_state_path, local_state)
        updated_state = dict(state)
        boundary = dict(updated_state["boundary"])
        boundary["decision_generated"] = True
        updated_state["boundary"] = boundary
        updated_state["hard_gate_decision"] = {
            "milestone": MILESTONE,
            "execution_commit": execution_commit,
            "status": "completed",
            "method_decisions": local_state["method_decisions"],
            "project_decision": project_decision["decision"],
            "selected_method": project_decision["selected_method"],
            "artifact_identities": identities,
            "image_content_accessed": False,
            "image_content_displayed": False,
            "score_or_threshold_modified": False,
            "weighted_score_used": False,
            "hard_gate_waiver_used": False,
        }
        write_json_atomic(
            external_root / "boundary-state.json",
            updated_state,
            overwrite=True,
        )
        return {
            "method_decisions": local_state["method_decisions"],
            "project_decision": project_decision["decision"],
            "selected_method": project_decision["selected_method"],
            "manifest_file_count": len(manifest["files"]),
        }
    except Exception:
        for path in moved:
            path.unlink(missing_ok=True)
        local_state_path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
