"""Reproduce the first fixed DINOv2 scores in a fresh offline process."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from few_shot_anomaly_poc.dinov2_scoring import (
    EMBEDDING_DIMENSION,
    IMAGENET_MEAN,
    IMAGENET_STANDARD_DEVIATION,
    INPUT_SHAPE,
    L2_EPSILON,
    MEMORY_BLOCK_SIZE,
    REFERENCE_COUNT,
    TOP_FRACTION,
    score_dinov2_image,
)
from few_shot_anomaly_poc.dinov2_timing import (
    GENERATOR_SEED,
    LOGICAL_STORE_ID,
    PREREGISTRATION_ID,
    QUERY_IDS,
    REFERENCE_IDS,
    RESOLUTION_WORKER_SCHEMA,
    DINOv2TimingError,
    _exception_record,
    _load_fixed_runtime,
    _load_json,
    _resolve_project_path,
    _validate_execution_identity,
    _worker_environment,
    build_memory_bank_one_at_a_time,
    copy_store_image,
    create_synthetic_input_store,
    open_verified_synthetic_input_store,
    peak_rss_bytes,
    validate_synthetic_input_manifest,
)
from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.jsonio import write_json_atomic
from few_shot_anomaly_poc.model_assets import SOURCE_REVISION, SOURCE_ROOT
from few_shot_anomaly_poc.model_compatibility import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_SOURCE_SHA256,
    _module_origins,
)

REPRODUCTION_RESOLUTION = 224
REPRODUCTION_QUERY_COUNT = 10
SCORE_TOLERANCE = 1e-6
REPRODUCTION_SCHEMA = "v0.2-dinov2-offline-reproduction-v1"
REPRODUCTION_PARENT_SCHEMA = "v0.2-dinov2-offline-reproduction-parent-v1"
TIMING_EXECUTION_COMMIT = "d02da60f622090746c8348704e550dccf57358d5"
TIMING_ARTIFACT_DIR = Path("artifacts/v0.2/cpu-timing/first-fixed-memory-bounded-run")
TIMING_SUMMARY_SHA256 = "64f8ca7b56ad861f8b3dd821f1b0d07dac3cff6809fa331e1042dbae7a9dfd71"
TIMING_RESOLUTION_SHA256 = "2c0820771cc435f23b79c053d1c705f2bf1fcd875a7bb46a9d12706fddc4c3d4"
TIMING_INPUT_MANIFEST_SHA256 = "0760e26a49e11130b66bae57e287d8593dea666feda0a6c08041baf77e2c7dec"
TIMING_RAW_STORE_SHA256 = "b57319a8aa9fc8c27d1daa22acf8640a31cf366074a2c42e14e65ff55f4501b7"
SCORING_SMOKE_PATH = Path("artifacts/v0.2/scoring-path/synthetic-smoke.json")
SCORING_SMOKE_SHA256 = "56b5f342c3b8875df6c9baec61fdd8339c0f40d1f69c83245bdbf580ac23f7b8"
ACQUISITION_PATH = Path("artifacts/v0.2/model-assets/acquisition.json")
ACQUISITION_SHA256 = "ba976ed08369fd80423d241129b8a86b05fcef650a39befa4ee67c8314233dac"
WHEEL_INSPECTION_PATH = Path("artifacts/v0.2/dependencies/wheel-inspection.json")
WHEEL_INSPECTION_SHA256 = "402e35c32a7c31e2fd2470877f8047685a372e933d48167046366553eea1d0ad"
ENVIRONMENT_LOCK_PATH = Path("environments/v0.2-preflight/uv.lock")
ENVIRONMENT_LOCK_SHA256 = "28a0878f3425bf6ea6fcc1631b168d67e6ddcc747d71f9c92c85b3aa9c706ec8"
SCORING_SOURCE_PATH = Path("src/few_shot_anomaly_poc/dinov2_scoring.py")
SCORING_SOURCE_SHA256 = "944dc4c349aa011157ecc2e147bbcdd27da8b8c1fa1084fc0ae0e4641cec4cee"
TIMING_SOURCE_PATH = Path("src/few_shot_anomaly_poc/dinov2_timing.py")
TIMING_SOURCE_SHA256 = "4ff3a9839127d9c11e0e16dbb94965e85cebeff20d605dbbcfd32b2052ace916"
EXPECTED_CONFIGURATION_SHA256 = "f4de068c34c9ca222c5de9a454c80d3c41c1e41b478ca583621d01c694aa5ab0"


class DINOv2ReproductionError(Exception):
    """Reject an operation outside the fixed offline reproduction contract."""


def fixed_reproduction_configuration() -> dict[str, Any]:
    """Return the preregistered scoring and reproduction configuration."""
    return {
        "execution": {
            "batch_size": 1,
            "deterministic_algorithms": True,
            "interop_threads": 1,
            "intraop_threads": 4,
            "network_required": False,
            "resolution": REPRODUCTION_RESOLUTION,
        },
        "inputs": {
            "generator": "numpy.random.Generator(numpy.random.PCG64(42))",
            "generator_seed": GENERATOR_SEED,
            "input_shape": list(INPUT_SHAPE),
            "query_ids": list(QUERY_IDS[:REPRODUCTION_QUERY_COUNT]),
            "reference_ids": list(REFERENCE_IDS),
        },
        "model": {
            "embedding_dimension": EMBEDDING_DIMENSION,
            "entry_point": "dinov2.hub.backbones.dinov2_vits14",
            "register_tokens": 0,
        },
        "preprocessing": {
            "image_net_mean": list(IMAGENET_MEAN),
            "image_net_standard_deviation": list(IMAGENET_STANDARD_DEVIATION),
            "resize": "bicubic_antialias_true",
        },
        "reproduction": {
            "absolute_score_tolerance": SCORE_TOLERANCE,
            "query_count": REPRODUCTION_QUERY_COUNT,
            "timing_measurement": False,
        },
        "scoring": {
            "distance": "exact_cosine_clamp_0_2",
            "l2_epsilon": L2_EPSILON,
            "memory_block_size": MEMORY_BLOCK_SIZE,
            "reference_count": REFERENCE_COUNT,
            "top_fraction": TOP_FRACTION,
        },
    }


def configuration_sha256(configuration: object) -> str:
    """Hash one canonical JSON configuration without machine-specific data."""
    serialized = json.dumps(
        configuration,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _require_sha256(path: Path, expected: str, *, field: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise DINOv2ReproductionError(f"{field} SHA-256 changed")


def validate_timing_baseline(project_root: Path) -> dict[str, Any]:
    """Validate the immutable timing evidence and return its first ten scores."""
    project_root = project_root.resolve()
    paths = {
        "summary": project_root / TIMING_ARTIFACT_DIR / "summary.json",
        "resolution": project_root / TIMING_ARTIFACT_DIR / "resolution-224.json",
        "input_manifest": project_root / TIMING_ARTIFACT_DIR / "input-manifest.json",
        "scoring_smoke": project_root / SCORING_SMOKE_PATH,
        "acquisition": project_root / ACQUISITION_PATH,
        "wheel_inspection": project_root / WHEEL_INSPECTION_PATH,
        "environment_lock": project_root / ENVIRONMENT_LOCK_PATH,
        "scoring_source": project_root / SCORING_SOURCE_PATH,
        "timing_source": project_root / TIMING_SOURCE_PATH,
    }
    expected_hashes = {
        "summary": TIMING_SUMMARY_SHA256,
        "resolution": TIMING_RESOLUTION_SHA256,
        "input_manifest": TIMING_INPUT_MANIFEST_SHA256,
        "scoring_smoke": SCORING_SMOKE_SHA256,
        "acquisition": ACQUISITION_SHA256,
        "wheel_inspection": WHEEL_INSPECTION_SHA256,
        "environment_lock": ENVIRONMENT_LOCK_SHA256,
        "scoring_source": SCORING_SOURCE_SHA256,
        "timing_source": TIMING_SOURCE_SHA256,
    }
    for name, expected in expected_hashes.items():
        _require_sha256(paths[name], expected, field=name)

    summary = _load_json(paths["summary"], field="fixed timing summary")
    resolution = _load_json(paths["resolution"], field="fixed 224 timing artifact")
    input_manifest = validate_synthetic_input_manifest(
        _load_json(paths["input_manifest"], field="fixed timing input manifest")
    )
    scoring_smoke = _load_json(paths["scoring_smoke"], field="fixed scoring smoke")
    acquisition = _load_json(paths["acquisition"], field="fixed model acquisition")
    wheel_inspection = _load_json(paths["wheel_inspection"], field="fixed wheel inspection")
    observations = resolution.get("loop", {}).get("observations")
    if (
        summary.get("schema_version") != "v0.2-dinov2-timing-parent-v1"
        or summary.get("execution", {}).get("execution_commit") != TIMING_EXECUTION_COMMIT
        or summary.get("decision", {}).get("selected_resolution_candidate")
        != REPRODUCTION_RESOLUTION
        or summary.get("decision", {}).get("resolution_pass") != {"224": True, "448": False}
        or summary.get("decision", {}).get("next_step") != "PROCEED_TO_OFFLINE_REPRODUCTION"
        or resolution.get("schema_version") != RESOLUTION_WORKER_SCHEMA
        or resolution.get("resolution") != REPRODUCTION_RESOLUTION
        or resolution.get("decision", {}).get("status") != "pass"
        or not isinstance(observations, list)
        or len(observations) != 300
        or input_manifest["logical_store"]["file_sha256"] != TIMING_RAW_STORE_SHA256
        or input_manifest["logical_store"]["logical_id"] != LOGICAL_STORE_ID
    ):
        raise DINOv2ReproductionError("fixed timing baseline contract is invalid")

    first_ten: list[dict[str, Any]] = []
    for query_index, item in enumerate(observations[:REPRODUCTION_QUERY_COUNT]):
        if (
            not isinstance(item, dict)
            or item.get("invocation_index") != query_index
            or item.get("pass_index") != 0
            or item.get("query_index") != query_index
            or item.get("asset_id") != QUERY_IDS[query_index]
            or item.get("status") != "success"
            or type(item.get("score")) is not float
            or not math.isfinite(item["score"])
        ):
            raise DINOv2ReproductionError("fixed timing score order is invalid")
        first_ten.append(
            {
                "asset_id": item["asset_id"],
                "query_index": query_index,
                "score": item["score"],
            }
        )

    fixed_contract = scoring_smoke.get("fixed_contract")
    expected_contract = {
        "allowed_resolutions": [224, 448],
        "embedding_dimension": EMBEDDING_DIMENSION,
        "l2_epsilon": L2_EPSILON,
        "memory_block_size": MEMORY_BLOCK_SIZE,
        "model_entry_point": "dinov2.hub.backbones.dinov2_vits14",
        "reference_count": REFERENCE_COUNT,
        "top_fraction": TOP_FRACTION,
    }
    if (
        fixed_contract != expected_contract
        or wheel_inspection.get("environment", {}).get("lock_sha256") != ENVIRONMENT_LOCK_SHA256
        or configuration_sha256(fixed_reproduction_configuration()) != EXPECTED_CONFIGURATION_SHA256
        or acquisition.get("source", {}).get("identity", {}).get("revision") != SOURCE_REVISION
    ):
        raise DINOv2ReproductionError("fixed reproduction configuration changed")

    fixed_runtime = resolution["fixed_runtime"]
    if (
        fixed_runtime["source_identity"]["sha256"] != EXPECTED_SOURCE_SHA256
        or fixed_runtime["checkpoint_identity"]["sha256"] != EXPECTED_CHECKPOINT_SHA256
    ):
        raise DINOv2ReproductionError("fixed model asset identity changed")
    return {
        "configuration": fixed_reproduction_configuration(),
        "configuration_sha256": EXPECTED_CONFIGURATION_SHA256,
        "expected_scores": first_ten,
        "fixed_runtime": fixed_runtime,
        "input_manifest": input_manifest,
        "source_revision": acquisition["source"]["identity"]["revision"],
    }


def execute_score_reproduction(
    *,
    copy_query: Callable[[int], NDArray[np.uint8]],
    score_image: Callable[[NDArray[np.uint8]], float],
    expected_scores: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Regenerate ten ordered scores once and preserve mismatch or failure."""
    if len(expected_scores) != REPRODUCTION_QUERY_COUNT:
        raise DINOv2ReproductionError("expected_scores must contain exactly 10 rows")
    comparisons: list[dict[str, Any]] = []
    failure: dict[str, str] | None = None
    for query_index, expected in enumerate(expected_scores):
        asset_id = QUERY_IDS[query_index]
        if (
            expected.get("asset_id") != asset_id
            or expected.get("query_index") != query_index
            or type(expected.get("score")) is not float
            or not math.isfinite(expected["score"])
        ):
            raise DINOv2ReproductionError("expected score identity is invalid")
        try:
            image = copy_query(query_index)
            observed_score = float(score_image(image))
            if not math.isfinite(observed_score):
                raise DINOv2ReproductionError("reproduced score is not finite")
            absolute_difference = abs(observed_score - expected["score"])
            comparisons.append(
                {
                    "absolute_difference": absolute_difference,
                    "asset_id": asset_id,
                    "expected_score": expected["score"],
                    "query_index": query_index,
                    "reproduced_score": observed_score,
                    "within_tolerance": absolute_difference <= SCORE_TOLERANCE,
                }
            )
        except Exception as error:
            failure = _exception_record(error, phase=f"query_{query_index:03d}")
            break
    differences = [item["absolute_difference"] for item in comparisons]
    complete = len(comparisons) == REPRODUCTION_QUERY_COUNT and failure is None
    passed = complete and all(item["within_tolerance"] for item in comparisons)
    return {
        "comparisons": comparisons,
        "failure": failure,
        "summary": {
            "all_scores_finite": complete,
            "asset_id_order_match": complete,
            "attempted_count": len(comparisons) + (1 if failure is not None else 0),
            "complete_observation_set": complete,
            "failure_count": 0 if failure is None else 1,
            "maximum_absolute_difference": max(differences) if differences else None,
            "missing_count": REPRODUCTION_QUERY_COUNT - len(comparisons),
            "required_count": REPRODUCTION_QUERY_COUNT,
            "status": "pass" if passed else "fail",
            "tolerance": SCORE_TOLERANCE,
        },
    }


def _identity_record(*, expected: str, observed: str) -> dict[str, Any]:
    return {
        "expected": expected,
        "match": observed == expected,
        "observed": observed,
    }


def run_reproduction_worker(
    *,
    acquisition_path: Path,
    import_smoke_path: Path,
    strict_load_path: Path,
    artifact_dir: Path,
    source_root: Path,
    environment_root: Path,
    project_root: Path,
    execution_commit: str,
    verification_date: str,
    output_root: Path,
) -> dict[str, Any]:
    """Run the fixed 224 score reproduction inside one fresh process."""
    project_root = project_root.resolve()
    resolved = {
        name: _resolve_project_path(path, project_root=project_root, field=name)
        for name, path in {
            "acquisition_path": acquisition_path,
            "artifact_dir": artifact_dir,
            "environment_root": environment_root,
            "import_smoke_path": import_smoke_path,
            "output_root": output_root,
            "source_root": source_root,
            "strict_load_path": strict_load_path,
        }.items()
    }
    output_root = resolved["output_root"]
    if not output_root.is_dir():
        raise DINOv2ReproductionError("output_root must be created by the parent")
    report_path = output_root / "reproduction.json"
    input_store_path = output_root / "synthetic-inputs.npy"
    input_manifest_path = output_root / "input-manifest.json"
    if any(path.exists() for path in (report_path, input_store_path, input_manifest_path)):
        raise FileExistsError("refusing to overwrite reproduction output")
    _validate_execution_identity(
        project_root=project_root,
        execution_commit=execution_commit,
        verification_date=verification_date,
    )
    baseline = validate_timing_baseline(project_root)
    phase = "input_regeneration"
    state = {"model_constructed": False, "model_inference_performed": False}
    memory = {"process_start_peak_rss_bytes": peak_rss_bytes()}
    result: dict[str, Any] | None = None
    worker_failure: dict[str, str] | None = None
    generated_manifest: dict[str, Any] | None = None
    fixed_runtime: dict[str, Any] | None = None
    network_guard: Any | None = None
    previous_sys_path: list[str] | None = None
    try:
        generated_manifest = create_synthetic_input_store(
            store_path=input_store_path,
            manifest_path=input_manifest_path,
        )
        if sha256_file(input_manifest_path) != TIMING_INPUT_MANIFEST_SHA256:
            raise DINOv2ReproductionError("regenerated input manifest changed")
        store = open_verified_synthetic_input_store(
            store_path=input_store_path,
            manifest=generated_manifest,
        )
        phase = "runtime_setup"
        torch, model, fixed_runtime, network_guard, previous_sys_path = _load_fixed_runtime(
            acquisition_path=resolved["acquisition_path"],
            import_smoke_path=resolved["import_smoke_path"],
            strict_load_path=resolved["strict_load_path"],
            artifact_dir=resolved["artifact_dir"],
            source_root=resolved["source_root"],
            environment_root=resolved["environment_root"],
        )
        state["model_constructed"] = True
        memory["after_model_load_peak_rss_bytes"] = peak_rss_bytes()
        if fixed_runtime != {
            key: value
            for key, value in baseline["fixed_runtime"].items()
            if key != "module_origins"
        }:
            raise DINOv2ReproductionError("reproduced fixed runtime identity changed")

        def copy_reference(index: int) -> NDArray[np.uint8]:
            return copy_store_image(
                store,
                index=index,
                expected_sha256=generated_manifest["references"][index]["sha256"],
            )

        phase = "reference_fitting"
        state["model_inference_performed"] = True
        memory_bank = build_memory_bank_one_at_a_time(
            copy_reference=copy_reference,
            model=model,
            resolution=REPRODUCTION_RESOLUTION,
            torch_module=torch,
        )
        memory["after_memory_bank_peak_rss_bytes"] = peak_rss_bytes()

        def copy_query(index: int) -> NDArray[np.uint8]:
            return copy_store_image(
                store,
                index=REFERENCE_COUNT + index,
                expected_sha256=generated_manifest["queries"][index]["sha256"],
            )

        def score_image(image: NDArray[np.uint8]) -> float:
            return score_dinov2_image(
                image,
                model=model,
                memory_bank=memory_bank,
                resolution=REPRODUCTION_RESOLUTION,
                torch_module=torch,
            )

        phase = "score_reproduction"
        result = execute_score_reproduction(
            copy_query=copy_query,
            score_image=score_image,
            expected_scores=baseline["expected_scores"],
        )
        memory["after_reproduction_peak_rss_bytes"] = peak_rss_bytes()
        fixed_runtime["module_origins"] = _module_origins(resolved["source_root"])
        if fixed_runtime != baseline["fixed_runtime"]:
            raise DINOv2ReproductionError("reproduced module identity changed")
        if "xformers" in sys.modules or any(name.startswith("xformers.") for name in sys.modules):
            raise DINOv2ReproductionError("xformers was imported unexpectedly")
    except Exception as error:
        worker_failure = _exception_record(error, phase=phase)
        memory["failure_peak_rss_bytes"] = peak_rss_bytes()
    finally:
        if previous_sys_path is not None:
            sys.path[:] = previous_sys_path
        if network_guard is not None:
            network_guard.__exit__(None, None, None)
            if network_guard.attempts and worker_failure is None:
                worker_failure = {
                    "category": "execution_error",
                    "exception_type": "network_operation_attempted",
                    "phase": "network_boundary",
                }

    reproduction_passed = (
        worker_failure is None and result is not None and result["summary"]["status"] == "pass"
    )
    identities = {
        "checkpoint_sha256": _identity_record(
            expected=EXPECTED_CHECKPOINT_SHA256,
            observed=(
                fixed_runtime["checkpoint_identity"]["sha256"]
                if fixed_runtime is not None
                else "unavailable"
            ),
        ),
        "configuration_sha256": _identity_record(
            expected=EXPECTED_CONFIGURATION_SHA256,
            observed=configuration_sha256(fixed_reproduction_configuration()),
        ),
        "environment_lock_sha256": _identity_record(
            expected=ENVIRONMENT_LOCK_SHA256,
            observed=sha256_file(project_root / ENVIRONMENT_LOCK_PATH),
        ),
        "generated_input_manifest_sha256": _identity_record(
            expected=TIMING_INPUT_MANIFEST_SHA256,
            observed=(
                sha256_file(input_manifest_path) if input_manifest_path.is_file() else "unavailable"
            ),
        ),
        "raw_input_store_sha256": _identity_record(
            expected=TIMING_RAW_STORE_SHA256,
            observed=(
                generated_manifest["logical_store"]["file_sha256"]
                if generated_manifest is not None
                else "unavailable"
            ),
        ),
        "source_archive_sha256": _identity_record(
            expected=EXPECTED_SOURCE_SHA256,
            observed=(
                fixed_runtime["source_identity"]["sha256"]
                if fixed_runtime is not None
                else "unavailable"
            ),
        ),
        "source_revision": _identity_record(
            expected=SOURCE_REVISION,
            observed=baseline["source_revision"],
        ),
    }
    all_identities_match = all(item["match"] for item in identities.values())
    passed = reproduction_passed and all_identities_match
    report = {
        "boundary": {
            "dataset_access": False,
            "labels_accessed": False,
            "latency_measurement_performed": False,
            "model_constructed": state["model_constructed"],
            "model_inference_performed": state["model_inference_performed"],
            "network_access": False,
            "synthetic_inputs_only": True,
            "threshold_calibration_performed": False,
        },
        "decision": {
            "next_step": ("PROCEED_TO_EVALUATION_BOUNDARY_CHECK" if passed else "DO_NOT_PROCEED"),
            "status": "pass" if passed else "fail",
        },
        "execution": {
            "execution_commit": execution_commit,
            "fresh_process_required": True,
            "process_id_recorded": False,
            "resolution": REPRODUCTION_RESOLUTION,
            "verification_date": verification_date,
        },
        "failure": worker_failure
        if worker_failure is not None
        else (result["failure"] if result is not None else None),
        "fixed_runtime": fixed_runtime,
        "identities": {
            "all_match": all_identities_match,
            "records": identities,
        },
        "inputs": {
            "logical_store": (
                generated_manifest["logical_store"] if generated_manifest is not None else None
            ),
            "preregistration_id": PREREGISTRATION_ID,
            "raw_store_in_git": False,
            "timing_artifact": (TIMING_ARTIFACT_DIR / "resolution-224.json").as_posix(),
            "timing_artifact_sha256": TIMING_RESOLUTION_SHA256,
        },
        "memory": {
            **memory,
            "peak_rss_is_gating": False,
            "units": "bytes",
        },
        "reproduction": result,
        "schema_version": REPRODUCTION_SCHEMA,
    }
    write_json_atomic(report_path, report)
    return report


def reproduction_worker_command(
    *,
    python_executable: Path,
    worker_script: Path,
    project_root: Path,
    artifact_dir: Path,
    source_root: Path,
    environment_root: Path,
    execution_commit: str,
    verification_date: str,
    output_root: Path,
) -> list[str]:
    """Build the fixed isolated-worker command without shell interpolation."""
    return [
        str(python_executable),
        "-I",
        "-B",
        str(worker_script),
        "--artifact-dir",
        str(artifact_dir),
        "--source-root",
        str(source_root),
        "--environment-root",
        str(environment_root),
        "--project-root",
        str(project_root),
        "--execution-commit",
        execution_commit,
        "--verification-date",
        verification_date,
        "--output-root",
        str(output_root),
    ]


def run_reproduction_parent(
    *,
    artifact_dir: Path,
    environment_root: Path,
    project_root: Path,
    execution_commit: str,
    verification_date: str,
    output_root: Path,
) -> dict[str, Any]:
    """Start the fixed 224 reproduction worker and validate its result."""
    project_root = project_root.resolve()
    resolved = {
        name: _resolve_project_path(path, project_root=project_root, field=name)
        for name, path in {
            "artifact_dir": artifact_dir,
            "environment_root": environment_root,
            "output_root": output_root,
        }.items()
    }
    output_root = resolved["output_root"]
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    _validate_execution_identity(
        project_root=project_root,
        execution_commit=execution_commit,
        verification_date=verification_date,
    )
    validate_timing_baseline(project_root)
    python_executable = resolved["environment_root"] / "bin/python"
    worker_script = project_root / "scripts/run_v0_2_offline_reproduction_worker.py"
    source_root = (
        resolved["artifact_dir"] / f"dinov2-source-sha256-{EXPECTED_SOURCE_SHA256}" / SOURCE_ROOT
    )
    if not python_executable.is_file() or not worker_script.is_file() or not source_root.is_dir():
        raise DINOv2ReproductionError("isolated worker runtime is unavailable")
    output_root.mkdir(parents=True)
    command = reproduction_worker_command(
        python_executable=python_executable,
        worker_script=worker_script,
        project_root=project_root,
        artifact_dir=resolved["artifact_dir"],
        source_root=source_root,
        environment_root=resolved["environment_root"],
        execution_commit=execution_commit,
        verification_date=verification_date,
        output_root=output_root,
    )
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=_worker_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    (output_root / "worker.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_root / "worker.stderr.log").write_text(completed.stderr, encoding="utf-8")
    report_path = output_root / "reproduction.json"
    validation_failure: str | None = None
    report: dict[str, Any]
    try:
        report = _load_json(report_path, field="offline reproduction artifact")
        if (
            report.get("schema_version") != REPRODUCTION_SCHEMA
            or report.get("execution", {}).get("execution_commit") != execution_commit
            or report.get("execution", {}).get("resolution") != REPRODUCTION_RESOLUTION
        ):
            raise DINOv2ReproductionError("offline reproduction artifact is invalid")
    except (DINOv2ReproductionError, DINOv2TimingError, OSError):
        report = {
            "decision": {"next_step": "DO_NOT_PROCEED", "status": "fail"},
            "schema_version": REPRODUCTION_SCHEMA,
        }
        validation_failure = "missing_or_invalid_worker_artifact"
    report_passed = report.get("decision", {}).get("status") == "pass"
    if validation_failure is None and (
        (report_passed and completed.returncode != 0)
        or (not report_passed and completed.returncode not in (1, 2))
    ):
        validation_failure = "worker_exit_and_artifact_disagree"
        report["decision"] = {"next_step": "DO_NOT_PROCEED", "status": "fail"}
    passed = validation_failure is None and report["decision"]["status"] == "pass"
    summary = {
        "boundary": {
            "dataset_access": False,
            "labels_accessed": False,
            "latency_measurement_performed": False,
            "network_access": False,
            "synthetic_inputs_only": True,
            "threshold_calibration_performed": False,
        },
        "decision": {
            "next_step": ("PROCEED_TO_EVALUATION_BOUNDARY_CHECK" if passed else "DO_NOT_PROCEED"),
            "status": "pass" if passed else "fail",
        },
        "execution": {
            "execution_commit": execution_commit,
            "resolution": REPRODUCTION_RESOLUTION,
            "verification_date": verification_date,
        },
        "schema_version": REPRODUCTION_PARENT_SCHEMA,
        "worker": {
            "artifact": report_path.name if report_path.is_file() else None,
            "artifact_sha256": sha256_file(report_path) if report_path.is_file() else None,
            "fresh_process": True,
            "return_code": completed.returncode,
            "validation_failure": validation_failure,
        },
    }
    write_json_atomic(output_root / "summary.json", summary)
    return summary
