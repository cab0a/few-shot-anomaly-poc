"""Create and verify the v0.1 pre-evaluation freeze checkpoint."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import tomllib
from dataclasses import asdict
from pathlib import Path

from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.jsonio import write_json_atomic

FREEZE_SCHEMA_VERSION = 1
FREEZE_CHECKPOINT_ID = "v0.1-pre-evaluation-freeze"
FREEZE_STATUS = "FROZEN_BEFORE_FINAL_TEST"
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
CI_URL_PATTERN = re.compile(
    r"^https://github\.com/cab0a/few-shot-anomaly-poc/actions/runs/[1-9][0-9]*$"
)


class FreezeCheckpointError(Exception):
    """Reject a missing, changed, or internally inconsistent checkpoint."""


FROZEN_PATHS = (
    ".github/workflows/ci.yml",
    "LICENSE",
    "NOTICE.md",
    "artifacts/v0.1/data/dataset-record.json",
    "artifacts/v0.1/data/pcb1-normal-partitions.csv",
    "configs/v0.1.yaml",
    "docs/data-acquisition-record.md",
    "docs/dependencies-and-licenses.md",
    "docs/evaluation-artifact-schema.md",
    "docs/evaluation-plan.md",
    "docs/method-specification.md",
    "docs/problem-and-requirements.md",
    "docs/research-and-method-selection.md",
    "pyproject.toml",
    "schemas/v0.1/evaluation-artifacts.json",
    "src/few_shot_anomaly_poc/calibration.py",
    "src/few_shot_anomaly_poc/config.py",
    "src/few_shot_anomaly_poc/cpu_latency.py",
    "src/few_shot_anomaly_poc/ecc_residual.py",
    "src/few_shot_anomaly_poc/ecc_template.py",
    "src/few_shot_anomaly_poc/evaluation_artifacts.py",
    "src/few_shot_anomaly_poc/errors.py",
    "src/few_shot_anomaly_poc/failure_cases.py",
    "src/few_shot_anomaly_poc/hard_gate_decision.py",
    "src/few_shot_anomaly_poc/hog_features.py",
    "src/few_shot_anomaly_poc/hog_models.py",
    "src/few_shot_anomaly_poc/hog_scalers.py",
    "src/few_shot_anomaly_poc/hog_scoring.py",
    "src/few_shot_anomaly_poc/image_metrics.py",
    "src/few_shot_anomaly_poc/label_reveal.py",
    "src/few_shot_anomaly_poc/manifests.py",
    "src/few_shot_anomaly_poc/preprocessing.py",
    "src/few_shot_anomaly_poc/registration.py",
    "uv.lock",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise FreezeCheckpointError(f"cannot hash frozen file {path}") from error
    return digest.hexdigest()


def _frozen_file_records(project_root: Path) -> list[dict[str, str]]:
    return [
        {
            "relative_path": relative_path,
            "sha256": _sha256_file(project_root / relative_path),
        }
        for relative_path in FROZEN_PATHS
    ]


def _tree_sha256(records: list[dict[str, str]]) -> str:
    canonical = "".join(
        f"{record['relative_path']}\t{record['sha256']}\n" for record in records
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _json_compatible(value):
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    return value


def _partition_records(project_root: Path) -> tuple[list[str], int]:
    path = project_root / "artifacts/v0.1/data/pcb1-normal-partitions.csv"
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = tuple(csv.DictReader(stream))
    except (OSError, csv.Error) as error:
        raise FreezeCheckpointError("cannot read the normal partition manifest") from error
    expected_columns = {
        "partition",
        "selection_rank",
        "relative_path",
        "selection_sha256",
    }
    if (
        not rows
        or set(rows[0]) != expected_columns
        or len({row["relative_path"] for row in rows}) != len(rows)
    ):
        raise FreezeCheckpointError("normal partition manifest is invalid")
    reference_rows = tuple(row for row in rows if row["partition"] == "reference")
    calibration_rows = tuple(row for row in rows if row["partition"] == "calibration")
    if len(reference_rows) != 20 or len(calibration_rows) != 884:
        raise FreezeCheckpointError("normal partition counts are invalid")
    reference_rows = tuple(sorted(reference_rows, key=lambda row: int(row["selection_rank"])))
    return ([row["relative_path"] for row in reference_rows], len(calibration_rows))


def _dependency_record(project_root: Path) -> dict:
    pyproject_path = project_root / "pyproject.toml"
    try:
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise FreezeCheckpointError("cannot read pyproject.toml") from error
    project = pyproject["project"]
    return {
        "python_requirement": project["requires-python"],
        "direct_runtime_dependencies": project["dependencies"],
        "direct_development_dependencies": pyproject["dependency-groups"]["dev"],
        "uv_required_version": pyproject["tool"]["uv"]["required-version"],
        "pyproject_sha256": _sha256_file(pyproject_path),
        "uv_lock_sha256": _sha256_file(project_root / "uv.lock"),
    }


def _git_source_is_clean(
    project_root: Path,
    source_commit: str,
) -> bool:
    completed = subprocess.run(
        ["git", "diff", "--quiet", source_commit, "--", *FROZEN_PATHS],
        cwd=project_root,
        check=False,
    )
    return completed.returncode == 0


def build_pre_evaluation_freeze(
    *,
    project_root: Path,
    source_commit: str,
    ci_run_id: int,
    ci_run_url: str,
    config: ProjectConfig,
) -> dict:
    """Build a complete freeze record from the CI-verified source state."""
    if (
        not COMMIT_PATTERN.fullmatch(source_commit)
        or not isinstance(ci_run_id, int)
        or isinstance(ci_run_id, bool)
        or ci_run_id < 1
        or not CI_URL_PATTERN.fullmatch(ci_run_url)
        or not ci_run_url.endswith(f"/{ci_run_id}")
        or not _git_source_is_clean(project_root, source_commit)
    ):
        raise FreezeCheckpointError("source commit or CI evidence is invalid")

    reference_ids, calibration_count = _partition_records(project_root)
    frozen_files = _frozen_file_records(project_root)
    return {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "checkpoint_id": FREEZE_CHECKPOINT_ID,
        "status": FREEZE_STATUS,
        "freeze_date": "2026-07-29",
        "evaluation_source_commit": source_commit,
        "ci": {
            "provider": "GitHub Actions",
            "workflow": "CI",
            "run_id": ci_run_id,
            "run_url": ci_run_url,
            "conclusion": "success",
        },
        "dataset_scope": {
            "name": config.dataset_name,
            "category": config.category,
            "license": config.dataset_license,
            "raw_data_committed_to_git": False,
            "dataset_record_sha256": _sha256_file(
                project_root / "artifacts/v0.1/data/dataset-record.json"
            ),
        },
        "selection": {
            "seed": config.selection.seed,
            "reference_count": config.selection.reference_count,
            "procedure_version": config.selection.procedure_version,
            "namespace": config.selection.namespace,
            "reference_ids": reference_ids,
        },
        "partitions": {
            "reference_count": len(reference_ids),
            "calibration_count": calibration_count,
            "final_test_source": "pinned_official_one_class_test_split",
            "normal_partition_manifest": ("artifacts/v0.1/data/pcb1-normal-partitions.csv"),
            "normal_partition_manifest_sha256": _sha256_file(
                project_root / "artifacts/v0.1/data/pcb1-normal-partitions.csv"
            ),
            "official_split_revision": config.split.revision,
            "official_split_sha256": config.split.sha256,
            "overlap_allowed": False,
        },
        "methods": [
            "ecc_residual",
            "patch_hog_one_class_svm",
        ],
        "evaluation_rules": {
            "threshold_calibration": _json_compatible(asdict(config.threshold_calibration)),
            "latency_measurement": _json_compatible(asdict(config.latency_measurement)),
            "failure_case_selection": _json_compatible(asdict(config.failure_case_selection)),
            "hard_gate_decision": _json_compatible(asdict(config.hard_gate_decision)),
            "artifact_contract": "evaluation-artifacts/v0.1",
            "artifact_contract_sha256": _sha256_file(
                project_root / "schemas/v0.1/evaluation-artifacts.json"
            ),
            "weighted_aggregate_score_allowed": False,
            "hard_gate_waiver_allowed": False,
        },
        "dependencies": _dependency_record(project_root),
        "frozen_files": frozen_files,
        "frozen_tree_sha256": _tree_sha256(frozen_files),
        "boundary_state": {
            "final_test_scoring_started": False,
            "final_test_label_join_performed": False,
            "final_test_metrics_computed": False,
            "final_test_decision_recorded": False,
        },
        "change_policy": {
            "frozen_file_change_invalidates_checkpoint": True,
            "final_test_result_cannot_change_gate_or_threshold_rules": True,
            "implementation_defect_requires_documented_new_checkpoint": True,
            "post_freeze_result_and_report_files_may_be_added": True,
        },
    }


def verify_pre_evaluation_freeze(
    record: object,
    *,
    project_root: Path,
) -> None:
    """Raise when the record or any frozen file differs from the checkpoint."""
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != FREEZE_SCHEMA_VERSION
        or record.get("checkpoint_id") != FREEZE_CHECKPOINT_ID
        or record.get("status") != FREEZE_STATUS
        or not COMMIT_PATTERN.fullmatch(record.get("evaluation_source_commit", ""))
        or record.get("ci", {}).get("conclusion") != "success"
        or record.get("boundary_state")
        != {
            "final_test_scoring_started": False,
            "final_test_label_join_performed": False,
            "final_test_metrics_computed": False,
            "final_test_decision_recorded": False,
        }
        or record.get("change_policy", {}).get("frozen_file_change_invalidates_checkpoint")
        is not True
        or not isinstance(record.get("frozen_files"), list)
    ):
        raise FreezeCheckpointError("freeze record metadata is invalid")
    frozen_files = record["frozen_files"]
    if tuple(item.get("relative_path") for item in frozen_files) != FROZEN_PATHS:
        raise FreezeCheckpointError("freeze record path set is invalid")
    observed_files = _frozen_file_records(project_root)
    if observed_files != frozen_files:
        raise FreezeCheckpointError("a frozen file has changed")
    if record.get("frozen_tree_sha256") != _tree_sha256(observed_files):
        raise FreezeCheckpointError("frozen tree digest is invalid")

    config_record = next(
        item for item in observed_files if item["relative_path"] == "configs/v0.1.yaml"
    )
    if record.get("evaluation_rules", {}).get("artifact_contract") != (
        "evaluation-artifacts/v0.1"
    ) or record.get("dependencies", {}).get("uv_lock_sha256") != next(
        item["sha256"] for item in observed_files if item["relative_path"] == "uv.lock"
    ):
        raise FreezeCheckpointError("frozen rule or dependency identity is invalid")
    if config_record["sha256"] != _sha256_file(project_root / "configs/v0.1.yaml"):
        raise FreezeCheckpointError("fixed configuration identity is invalid")


def read_and_verify_pre_evaluation_freeze(
    path: Path,
    *,
    project_root: Path,
) -> dict:
    """Read and verify one committed checkpoint record."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FreezeCheckpointError("cannot read freeze record") from error
    verify_pre_evaluation_freeze(record, project_root=project_root)
    return record


def write_pre_evaluation_freeze(
    record: dict,
    *,
    path: Path,
    project_root: Path,
) -> None:
    """Verify and write one non-overwritable checkpoint record."""
    verify_pre_evaluation_freeze(record, project_root=project_root)
    write_json_atomic(path, record)
