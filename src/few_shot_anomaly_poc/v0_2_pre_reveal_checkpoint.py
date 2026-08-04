"""Create the v0.2.6 checkpoint over pushed label-free evaluation evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.jsonio import write_json_atomic
from few_shot_anomaly_poc.v0_2_boundary_preparation import (
    RUN_ID,
    validate_boundary_execution_identity,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    ARTIFACT_CONTRACT_VERSION,
    METHODS,
    load_v0_2_artifact_schema,
    load_v0_2_config,
    validate_json_artifact,
)
from few_shot_anomaly_poc.v0_2_offline_reproduction_run import (
    SCORING_ARTIFACT_HASHES,
    SCORING_EXECUTION_COMMIT,
    read_reproduction_csv,
)
from few_shot_anomaly_poc.v0_2_scoring_artifacts import read_method_scoring_artifacts

MILESTONE = "v0.2.6"
RUN_KIND = "final_test"
CHECKPOINT_NAME = "pre-reveal-checkpoint.json"
PRE_SCORING_HASHES = {
    "boundary/boundary-record.json": (
        "e122bfa51ce618e0588a580f2cf66447c44a2cf801f08f851cce9d5271a4c698"
    ),
    "freeze/pre-evaluation-freeze.json": (
        "ae552d805dd9648163a48683bad828c7e1b7ecc4f1d69f1fa28511363b08ce3b"
    ),
    "ecc_residual/calibration-scores.csv": (
        "b0b513f16b6ce0d068658d43a34f474106095bbd52e57cbc388927cde6f21c26"
    ),
    "ecc_residual/calibration-summary.json": (
        "e68f519abaa6f1f08a1376eec444cae5777736e3414b882ed93bf73f93345735"
    ),
    "ecc_residual/fit.json": ("fee3b96824987a6d03f279f5fefc8cfe15a0c807050ec6a35700d4c2e567ae3d"),
    "patch_hog_ocsvm/calibration-scores.csv": (
        "f6c40c3e4af1979f67ddf8916889887e022b58da0cd9fa04561c00f171a84d8e"
    ),
    "patch_hog_ocsvm/calibration-summary.json": (
        "9e687e80437de5f38f848677ad9372104c1bc78275ae536a7c58f4b02bdee356"
    ),
    "patch_hog_ocsvm/fit.json": (
        "d8da4c0704736e45495501618fcaa57d654e97acf4d3d10e12084e112e29b432"
    ),
    "dinov2_vits14_224_nn/calibration-scores.csv": (
        "dc99f12c1a76c1421a9dd6258d14231afa82e75ebf1d2d22a20a20a4b09ba1bc"
    ),
    "dinov2_vits14_224_nn/calibration-summary.json": (
        "0a89bcb834f5e38a601bb956d9b099c613786187adbbd2ef8ba75226e2a75da7"
    ),
    "dinov2_vits14_224_nn/fit.json": (
        "ba2bc1b5d2b974cf6d700d775ac6c94a4b14d407e9e7fd6a8832a7b9877ef4e6"
    ),
}
FIXED_HASHES = {**PRE_SCORING_HASHES, **SCORING_ARTIFACT_HASHES}


class V0_2PreRevealCheckpointError(Exception):
    """Reject incomplete, mutable, unpushed, or label-exposed checkpoint inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V0_2PreRevealCheckpointError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V0_2PreRevealCheckpointError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise V0_2PreRevealCheckpointError(f"{label} must contain one JSON object")
    return value


def label_free_bundle_sha256(public_root: Path, relative_paths: list[str]) -> str:
    """Hash sorted path/hash pairs for every pre-checkpoint public evidence file."""
    _require(
        relative_paths == sorted(relative_paths)
        and len(relative_paths) == len(set(relative_paths)),
        "label-free bundle paths must be unique and sorted",
    )
    digest = hashlib.sha256()
    for relative_path in relative_paths:
        file_sha256 = sha256_file(public_root / relative_path)
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _tracked_at_head(project_root: Path, relative_path: str, expected_sha256: str) -> None:
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
        cwd=project_root,
        check=False,
        capture_output=True,
    )
    _require(
        completed.returncode == 0
        and hashlib.sha256(completed.stdout).hexdigest() == expected_sha256,
        f"label-free artifact is not fixed at HEAD: {relative_path}",
    )


def create_pre_reveal_checkpoint(
    *, project_root: Path, evidence_commit: str, public_root: Path, external_root: Path
) -> dict[str, Any]:
    """Bind the complete pushed score-side bundle before any label access."""
    project_root = project_root.resolve()
    public_root = public_root.resolve()
    external_root = external_root.resolve()
    _require(
        public_root == (project_root / f"artifacts/v0.2/evaluation/{RUN_ID}").resolve()
        and external_root == (project_root / f"data/external/v0.2/evaluation/{RUN_ID}").resolve(),
        "checkpoint roots differ from the fixed contract",
    )
    checkpoint_path = public_root / CHECKPOINT_NAME
    _require(not checkpoint_path.exists(), "pre-reveal checkpoint already exists")
    validate_boundary_execution_identity(
        project_root=project_root, execution_commit=evidence_commit
    )
    config = load_v0_2_config(project_root / "configs/v0.2.yaml")
    schema = load_v0_2_artifact_schema(project_root / "schemas/v0.2/evaluation-artifacts.json")

    expected_paths = set(FIXED_HASHES)
    expected_paths.update(f"{method}/offline-reproduction.csv" for method in METHODS)
    observed_paths = {
        path.relative_to(public_root).as_posix()
        for path in public_root.rglob("*")
        if path.is_file()
    }
    _require(observed_paths == expected_paths, "pre-reveal public artifact inventory changed")
    for relative_path, expected_sha256 in FIXED_HASHES.items():
        _require(
            sha256_file(public_root / relative_path) == expected_sha256,
            f"fixed evidence changed: {relative_path}",
        )

    method_score_counts: dict[str, int] = {}
    reproduction_status: dict[str, str] = {}
    for method in METHODS:
        scoring = read_method_scoring_artifacts(public_root / method, schema=schema)
        reproduction_path = public_root / method / "offline-reproduction.csv"
        reproduction = read_reproduction_csv(reproduction_path, schema=schema)
        method_score_counts[method] = len(scoring.score_records)
        reproduction_status[method] = (
            "pass" if all(record["within_tolerance"] for record in reproduction) else "fail"
        )

    identities = {
        relative_path: sha256_file(public_root / relative_path)
        for relative_path in sorted(expected_paths)
    }
    for relative_path, identity in identities.items():
        repository_relative = (public_root / relative_path).relative_to(project_root).as_posix()
        _tracked_at_head(project_root, repository_relative, identity)
    state = _read_json(external_root / "boundary-state.json", label="boundary state")
    reproduction_state = state.get("offline_reproduction")
    _require(
        state.get("boundary", {}).get("final_test_label_revealed") is False
        and isinstance(reproduction_state, dict)
        and reproduction_state.get("milestone") == MILESTONE
        and reproduction_state.get("reproduction_status") == reproduction_status
        and reproduction_state.get("labels_accessed") is False
        and reproduction_state.get("semantic_paths_accessed") is False
        and reproduction_state.get("sealed_mapping_accessed") is False
        and reproduction_state.get("official_split_accessed") is False
        and "pre_reveal_checkpoint" not in state,
        "external state is not the fixed post-reproduction pre-reveal state",
    )
    relative_paths = sorted(expected_paths)
    checkpoint = validate_json_artifact(
        "pre_reveal_checkpoint",
        {
            "contract_version": ARTIFACT_CONTRACT_VERSION,
            "run_id": RUN_ID,
            "run_kind": RUN_KIND,
            "source_commit": SCORING_EXECUTION_COMMIT,
            "label_free_bundle_sha256": label_free_bundle_sha256(public_root, relative_paths),
            "method_order": list(METHODS),
            "method_score_counts": method_score_counts,
            "reproduction_status": reproduction_status,
            "git_commit": evidence_commit,
            "git_push_verified": True,
            "labels_accessed": False,
        },
        config=config,
        schema=schema,
    )
    write_json_atomic(checkpoint_path, checkpoint)
    state["pre_reveal_checkpoint"] = {
        "milestone": MILESTONE,
        "source_commit": SCORING_EXECUTION_COMMIT,
        "label_free_evidence_commit": evidence_commit,
        "label_free_bundle_sha256": checkpoint["label_free_bundle_sha256"],
        "checkpoint_artifact_sha256": sha256_file(checkpoint_path),
        "git_push_verified": True,
        "labels_accessed": False,
        "semantic_paths_accessed": False,
        "sealed_mapping_accessed": False,
        "official_split_accessed": False,
    }
    write_json_atomic(external_root / "boundary-state.json", state, overwrite=True)
    return checkpoint
