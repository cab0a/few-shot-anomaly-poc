"""Verify fixed v0.2 evidence and apply the final ordered preflight gate."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.jsonio import write_json_atomic
from few_shot_anomaly_poc.opaque_boundary import run_synthetic_boundary_feasibility

FINAL_DECISION_SCHEMA = "v0.2-final-preflight-decision-v1"
ARTIFACT_MANIFEST_SCHEMA = "v0.2-final-preflight-artifact-manifest-v1"
PREREGISTRATION_ID = "v0.2-dinov2-cpu-preflight-2"
BASE_PREREGISTRATION_ID = "v0.2-dinov2-cpu-preflight-1"
BASE_PREREGISTRATION_COMMIT = "e9330be10742947e4227ced4c99acafe4d098566"
MEMORY_PREREGISTRATION_COMMIT = "a177b5648c450b1e33ca3bbf5c16a051410ef756"
SOURCE_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
CHECKPOINT_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
ENVIRONMENT_LOCK_SHA256 = "28a0878f3425bf6ea6fcc1631b168d67e6ddcc747d71f9c92c85b3aa9c706ec8"

FIXED_EVIDENCE: dict[str, tuple[Path, str, str | None]] = {
    "base_preregistration": (
        Path("docs/v0.2-preflight-preregistration.md"),
        "19d4cf4079c6df7c9042be464859ccf98d41108656ba0259c8940ace740ebf42",
        None,
    ),
    "memory_preregistration": (
        Path("docs/v0.2-memory-bounded-cpu-preflight.md"),
        "8d2d055d6f311719e28f52fb7e8f2f87fb3202c04414b56440d0a420832658ba",
        None,
    ),
    "dependency_license_inventory": (
        Path("docs/v0.2-dependencies-and-licenses.md"),
        "ee44271a23ca396eebffe82a73fe96242d811aea6385a6bf608a4a703e06cb2a",
        None,
    ),
    "environment_lock": (
        Path("environments/v0.2-preflight/uv.lock"),
        ENVIRONMENT_LOCK_SHA256,
        None,
    ),
    "model_acquisition": (
        Path("artifacts/v0.2/model-assets/acquisition.json"),
        "ba976ed08369fd80423d241129b8a86b05fcef650a39befa4ee67c8314233dac",
        "v0.2-model-asset-acquisition-v1",
    ),
    "dependency_inspection": (
        Path("artifacts/v0.2/dependencies/wheel-inspection.json"),
        "402e35c32a7c31e2fd2470877f8047685a372e933d48167046366553eea1d0ad",
        "v0.2-dependency-artifact-inspection-v1",
    ),
    "import_smoke": (
        Path("artifacts/v0.2/environment/import-smoke.json"),
        "b0f38afb103f7084a0e5e09e8fd00e4cf2e0e5825d7a3fe8d5e3b48afd7b1f74",
        "v0.2-isolated-import-smoke-v1",
    ),
    "strict_load": (
        Path("artifacts/v0.2/model-compatibility/strict-load.json"),
        "4491f2fb472df813642d296d92d396e62476a2fd257d6b9da431c3a90b6aa604",
        "v0.2-weights-only-strict-load-v1",
    ),
    "scoring_smoke": (
        Path("artifacts/v0.2/scoring-path/synthetic-smoke.json"),
        "56b5f342c3b8875df6c9baec61fdd8339c0f40d1f69c83245bdbf580ac23f7b8",
        "v0.2-fixed-dinov2-scoring-smoke-v1",
    ),
    "timing_preconditions": (
        Path("artifacts/v0.2/cpu-preflight/attempt-002-memory-bounded-pass.json"),
        "f9befbd3df1c980f1dc0a8dc48563fd4dffbecd24530b2a4e2b413b4e688715d",
        "v0.2-cpu-timing-preconditions-v2",
    ),
    "timing_summary": (
        Path("artifacts/v0.2/cpu-timing/first-fixed-memory-bounded-run/summary.json"),
        "64f8ca7b56ad861f8b3dd821f1b0d07dac3cff6809fa331e1042dbae7a9dfd71",
        "v0.2-dinov2-timing-parent-v1",
    ),
    "timing_224": (
        Path("artifacts/v0.2/cpu-timing/first-fixed-memory-bounded-run/resolution-224.json"),
        "2c0820771cc435f23b79c053d1c705f2bf1fcd875a7bb46a9d12706fddc4c3d4",
        "v0.2-dinov2-timing-resolution-v1",
    ),
    "timing_448": (
        Path("artifacts/v0.2/cpu-timing/first-fixed-memory-bounded-run/resolution-448.json"),
        "7988270f82eb2a6e34b7f429fe755dd55d8b1e287fefbd2353191e50966c875a",
        "v0.2-dinov2-timing-resolution-v1",
    ),
    "reproduction_summary": (
        Path("artifacts/v0.2/offline-reproduction/first-fixed-run/summary.json"),
        "c6036f0214bf55cb0d7b208f51d54304aef9a3fe9688e4afffaf2dcbc0eafdd3",
        "v0.2-dinov2-offline-reproduction-parent-v1",
    ),
    "reproduction_detail": (
        Path("artifacts/v0.2/offline-reproduction/first-fixed-run/reproduction.json"),
        "3b53ad4370377fa86394bb757b204f9f6dc05c38448e203d16d77ace8bf89aeb",
        "v0.2-dinov2-offline-reproduction-v1",
    ),
}

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class V0_2PreflightDecisionError(Exception):
    """Reject an invalid final preflight execution request."""


def _git(project_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise V0_2PreflightDecisionError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def validate_clean_execution_identity(
    *, project_root: Path, execution_commit: str
) -> dict[str, Any]:
    """Require a clean exact commit containing both controlling preregistrations."""
    if not _COMMIT_PATTERN.fullmatch(execution_commit):
        raise V0_2PreflightDecisionError("execution_commit must be a full Git commit")
    observed_commit = _git(project_root, "rev-parse", "HEAD")
    if observed_commit != execution_commit:
        raise V0_2PreflightDecisionError("execution_commit does not match HEAD")
    if _git(project_root, "status", "--porcelain", "--untracked-files=all"):
        raise V0_2PreflightDecisionError("worktree must be clean before the checkpoint")
    for controlling_commit in (
        BASE_PREREGISTRATION_COMMIT,
        MEMORY_PREREGISTRATION_COMMIT,
    ):
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", controlling_commit, execution_commit],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise V0_2PreflightDecisionError(
                "execution commit does not contain every controlling preregistration"
            )
    return {
        "execution_commit": execution_commit,
        "worktree_clean": True,
    }


def _inspect_fixed_evidence(project_root: Path) -> tuple[
    dict[str, dict[str, Any]], dict[str, dict[str, Any] | None]
]:
    records: dict[str, dict[str, Any]] = {}
    values: dict[str, dict[str, Any] | None] = {}
    for name, (relative_path, expected_sha256, expected_schema) in FIXED_EVIDENCE.items():
        path = project_root / relative_path
        observed_sha256 = sha256_file(path) if path.is_file() else None
        identity_match = observed_sha256 == expected_sha256
        value: dict[str, Any] | None = None
        schema_match = expected_schema is None
        if identity_match and expected_schema is not None:
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                loaded = None
            if isinstance(loaded, dict):
                value = loaded
                schema_match = loaded.get("schema_version") == expected_schema
        verification = identity_match and schema_match
        records[name] = {
            "expected_sha256": expected_sha256,
            "observed_sha256": observed_sha256,
            "path": relative_path.as_posix(),
            "verification": "pass" if verification else "fail",
        }
        values[name] = value
    return records, values


def _evidence(records: dict[str, dict[str, Any]], *names: str) -> list[dict[str, Any]]:
    return [records[name] for name in names]


def _all_verified(records: dict[str, dict[str, Any]], *names: str) -> bool:
    return all(records[name]["verification"] == "pass" for name in names)


def evaluate_fixed_conditions(
    *,
    project_root: Path,
    execution_identity: dict[str, Any],
    boundary_feasibility: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate the ten preregistered conditions from immutable evidence."""
    records, values = _inspect_fixed_evidence(project_root)
    acquisition = values["model_acquisition"] or {}
    dependency = values["dependency_inspection"] or {}
    import_smoke = values["import_smoke"] or {}
    strict_load = values["strict_load"] or {}
    scoring_smoke = values["scoring_smoke"] or {}
    preconditions = values["timing_preconditions"] or {}
    timing_summary = values["timing_summary"] or {}
    timing_224 = values["timing_224"] or {}
    timing_448 = values["timing_448"] or {}
    reproduction_summary = values["reproduction_summary"] or {}
    reproduction_detail = values["reproduction_detail"] or {}

    preregistration_pass = _all_verified(
        records, "base_preregistration", "memory_preregistration"
    ) and execution_identity.get("worktree_clean") is True
    source_license_pass = (
        _all_verified(records, "model_acquisition")
        and acquisition.get("source", {}).get("identity", {}).get("revision")
        == SOURCE_REVISION
        and acquisition.get("source", {}).get("license", {}).get("identifier")
        == "Apache-2.0"
        and acquisition.get("checkpoint", {}).get("identity", {}).get("model_identifier")
        == "dinov2_vits14"
        and acquisition.get("checkpoint", {}).get("identity", {}).get("register_tokens")
        is False
    )
    dependency_pass = (
        _all_verified(
            records, "dependency_inspection", "environment_lock", "import_smoke"
        )
        and dependency.get("environment", {}).get("lock_sha256")
        == ENVIRONMENT_LOCK_SHA256
        and dependency.get("decision", {}).get("installation") == "INSTALL"
        and import_smoke.get("boundary", {}).get("network_access_during_verification")
        is False
    )
    checkpoint_pass = (
        _all_verified(records, "model_acquisition", "strict_load")
        and acquisition.get("checkpoint", {}).get("artifact", {}).get("observed_sha256")
        == CHECKPOINT_SHA256
        and strict_load.get("checkpoint", {}).get("identity", {}).get("sha256")
        == CHECKPOINT_SHA256
        and strict_load.get("model", {}).get("strict_load") == "pass"
        and strict_load.get("boundary", {}).get("model_inference_performed") is False
    )
    packages = dependency.get("packages")
    third_party_pass = (
        _all_verified(
            records, "dependency_inspection", "dependency_license_inventory"
        )
        and isinstance(packages, list)
        and bool(packages)
        and all(
            isinstance(package, dict)
            and package.get("license", {}).get("material_count", 0) >= 1
            for package in packages
        )
    )
    target_machine_pass = (
        _all_verified(records, "timing_preconditions")
        and preconditions.get("decision", {}).get("status") == "pass"
        and preconditions.get("target_machine", {}).get("evaluation", {}).get("status")
        == "pass"
    )
    timing_workers = timing_summary.get("worker_processes")
    execution_integrity_pass = (
        _all_verified(records, "scoring_smoke", "timing_summary", "timing_224", "timing_448")
        and scoring_smoke.get("decision", {}).get("status") == "PASS"
        and scoring_smoke.get("boundary", {}).get("dataset_access") is False
        and isinstance(timing_workers, list)
        and len(timing_workers) == 2
        and all(worker.get("validation_failure") is None for worker in timing_workers)
        and timing_224.get("boundary", {}).get("dataset_access") is False
        and timing_448.get("boundary", {}).get("dataset_access") is False
    )
    resolution_pass = timing_summary.get("decision", {}).get("resolution_pass")
    cpu_result_pass = (
        _all_verified(records, "timing_summary", "timing_224", "timing_448")
        and timing_summary.get("decision", {}).get("status") == "pass"
        and isinstance(resolution_pass, dict)
        and any(value is True for value in resolution_pass.values())
    )
    reproduction_pass = (
        _all_verified(records, "reproduction_summary", "reproduction_detail")
        and reproduction_summary.get("decision", {}).get("status") == "pass"
        and reproduction_summary.get("boundary", {}).get("network_access") is False
        and reproduction_detail.get("decision", {}).get("status") == "pass"
        and reproduction_detail.get("reproduction", {}).get("summary", {}).get("status")
        == "pass"
        and reproduction_detail.get("reproduction", {}).get("summary", {}).get(
            "maximum_absolute_difference"
        )
        <= 1e-6
    )
    boundary_checks = boundary_feasibility.get("checks")
    boundary_pass = (
        boundary_feasibility.get("decision", {}).get("status") == "pass"
        and isinstance(boundary_checks, dict)
        and bool(boundary_checks)
        and all(value is True for value in boundary_checks.values())
        and boundary_feasibility.get("boundary", {}).get("dataset_access") is False
        and boundary_feasibility.get("boundary", {}).get("official_split_access") is False
        and boundary_feasibility.get("boundary", {}).get("dataset_labels_accessed") is False
    )

    return [
        {
            "condition": 1,
            "evidence": _evidence(records, "base_preregistration", "memory_preregistration"),
            "name": "preregistration_identity",
            "passed": preregistration_pass,
            "reason": (
                "Both controlling preregistrations are exact and the execution commit is clean."
            ),
        },
        {
            "condition": 2,
            "evidence": _evidence(records, "model_acquisition"),
            "name": "source_and_license",
            "passed": source_license_pass,
            "reason": "The fixed non-register DINOv2 source and Apache-2.0 boundary are verified.",
        },
        {
            "condition": 3,
            "evidence": _evidence(
                records, "dependency_inspection", "environment_lock", "import_smoke"
            ),
            "name": "dependency_resolution",
            "passed": dependency_pass,
            "reason": (
                "The exact CPU-only lock, inspected wheels, and isolated imports are verified."
            ),
        },
        {
            "condition": 4,
            "evidence": _evidence(records, "model_acquisition", "strict_load"),
            "name": "checkpoint_acquisition",
            "passed": checkpoint_pass,
            "reason": "The observed checkpoint identity and weights-only strict load are verified.",
        },
        {
            "condition": 5,
            "evidence": _evidence(
                records, "dependency_inspection", "dependency_license_inventory"
            ),
            "name": "third_party_separation",
            "passed": third_party_pass,
            "reason": "Every locked wheel has inventoried license material under separate terms.",
        },
        {
            "condition": 6,
            "evidence": _evidence(records, "timing_preconditions"),
            "name": "target_machine",
            "passed": target_machine_pass,
            "reason": "The fixed CPU boundary passed the memory-bounded precondition check.",
        },
        {
            "condition": 7,
            "evidence": _evidence(
                records, "scoring_smoke", "timing_summary", "timing_224", "timing_448"
            ),
            "name": "execution_integrity",
            "passed": execution_integrity_pass,
            "reason": (
                "The fixed scorer and both fresh timing workers retained the registered controls."
            ),
        },
        {
            "condition": 8,
            "evidence": _evidence(records, "timing_summary", "timing_224", "timing_448"),
            "name": "cpu_result",
            "passed": cpu_result_pass,
            "reason": "At least one preregistered resolution passed; 224 passed and 448 failed.",
        },
        {
            "condition": 9,
            "evidence": _evidence(records, "reproduction_summary", "reproduction_detail"),
            "name": "reproducibility",
            "passed": reproduction_pass,
            "reason": (
                "The selected 224 scores were regenerated offline within the fixed tolerance."
            ),
        },
        {
            "condition": 10,
            "evidence": [
                {
                    "schema_version": boundary_feasibility.get("schema_version"),
                    "verification": "pass" if boundary_pass else "fail",
                }
            ],
            "name": "evaluation_boundary",
            "passed": boundary_pass,
            "reason": (
                "A synthetic no-data checkpoint verified opaque ordering, manifest separation, "
                "copy identity, scorer isolation, and non-overwrite behavior."
            ),
        },
    ]


def ordered_preflight_decision(
    condition_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the ten hard gates in order without a weighted score."""
    if len(condition_results) != 10:
        raise V0_2PreflightDecisionError("exactly ten condition results are required")
    first_failed: int | None = None
    output: list[dict[str, Any]] = []
    for expected_condition, result in enumerate(condition_results, start=1):
        if result.get("condition") != expected_condition or type(result.get("passed")) is not bool:
            raise V0_2PreflightDecisionError("condition results must be ordered and boolean")
        if first_failed is None and not result["passed"]:
            first_failed = expected_condition
        status = (
            "pass"
            if first_failed is None
            else "fail"
            if first_failed == expected_condition
            else "not_evaluated"
        )
        output.append(
            {
                "condition": expected_condition,
                "evidence": result.get("evidence", []),
                "name": result.get("name"),
                "reason": result.get("reason"),
                "status": status,
            }
        )
    proceed = first_failed is None
    return {
        "conditions": output,
        "decision": {
            "first_failed_condition": first_failed,
            "next_step": (
                "CREATE_SEPARATE_V0_2_METHOD_AND_EVALUATION_PREREGISTRATION"
                if proceed
                else "STOP_V0_2_PREFLIGHT"
            ),
            "outcome": "PROCEED" if proceed else "DO_NOT_PROCEED",
            "weighted_score_used": False,
        },
    }


def run_final_preflight_checkpoint(
    *,
    project_root: Path,
    execution_commit: str,
    verification_date: str,
    output_root: Path,
) -> dict[str, Any]:
    """Run the no-data boundary checkpoint and write final preflight artifacts."""
    project_root = project_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    execution_identity = validate_clean_execution_identity(
        project_root=project_root, execution_commit=execution_commit
    )
    with tempfile.TemporaryDirectory(prefix="v0-2-boundary-feasibility-") as temporary:
        feasibility = run_synthetic_boundary_feasibility(Path(temporary) / "checkpoint")

    feasibility["execution"] = {
        **execution_identity,
        "verification_date": verification_date,
    }
    feasibility["future_boundary"] = {
        "archive_sha256": "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362",
        "category": "pcb2",
        "dataset": "VisA",
        "official_split_revision": "2a692ab575001cbde74d402d897a7286086c6199",
        "official_split_sha256": "a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995",
        "preparation_authorized_by_this_run": False,
        "reference_budget": 20,
        "reference_seed": 42,
        "selection_namespace": "few-shot-anomaly-poc:v0.2:pcb2",
    }
    condition_results = evaluate_fixed_conditions(
        project_root=project_root,
        execution_identity=execution_identity,
        boundary_feasibility=feasibility,
    )
    gated = ordered_preflight_decision(condition_results)
    final_decision = {
        "boundary": {
            "dataset_access": False,
            "dataset_labels_accessed": False,
            "final_test_scoring_performed": False,
            "image_decode_performed": False,
            "official_split_access": False,
            "synthetic_boundary_fixture_only": True,
        },
        "conditions": gated["conditions"],
        "decision": {
            **gated["decision"],
            "scope": (
                "Authorizes only a separate v0.2 method-and-evaluation preregistration; "
                "it does not adopt DINOv2 or authorize unregistered performance claims."
            ),
        },
        "execution": feasibility["execution"],
        "preregistration": {
            "base_id": BASE_PREREGISTRATION_ID,
            "id": PREREGISTRATION_ID,
            "ordered_hard_gate": True,
        },
        "schema_version": FINAL_DECISION_SCHEMA,
    }

    output_root.mkdir(parents=True)
    feasibility_path = output_root / "boundary-feasibility.json"
    decision_path = output_root / "final-decision.json"
    write_json_atomic(feasibility_path, feasibility)
    write_json_atomic(decision_path, final_decision)
    artifact_records = []
    for path in (feasibility_path, decision_path):
        artifact_records.append(
            {
                "byte_count": path.stat().st_size,
                "path": path.relative_to(output_root).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    artifact_manifest = {
        "artifacts": artifact_records,
        "execution_commit": execution_commit,
        "schema_version": ARTIFACT_MANIFEST_SCHEMA,
    }
    write_json_atomic(output_root / "artifact-manifest.json", artifact_manifest)
    return final_decision
