"""Run a bounded synthetic smoke check of the fixed DINOv2 scoring path."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from few_shot_anomaly_poc.dinov2_scoring import (
    ALLOWED_RESOLUTIONS,
    EMBEDDING_DIMENSION,
    INPUT_SHAPE,
    L2_EPSILON,
    MEMORY_BLOCK_SIZE,
    REFERENCE_COUNT,
    TOP_FRACTION,
    aggregate_top_fraction_score,
    create_dinov2_memory_bank,
    exact_cosine_min_distances,
    expected_patch_count,
    extract_dinov2_patch_features,
    top_patch_count,
)
from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.jsonio import write_json_atomic
from few_shot_anomaly_poc.model_assets import SOURCE_ROOT
from few_shot_anomaly_poc.model_compatibility import (
    EXPECTED_CHECKPOINT_BYTES,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_MODEL_ENTRY_POINT,
    EXPECTED_SOURCE_SHA256,
    EXPECTED_TORCH_VERSION,
    NetworkGuard,
    _load_acquisition_record,
    _load_import_smoke_record,
    _model_summary,
    _module_origins,
    _summarize_state_dict,
    _validate_environment,
    _verify_asset,
)

OUTPUT_SCHEMA = "v0.2-fixed-dinov2-scoring-smoke-v1"
PREREGISTRATION_ID = "v0.2-dinov2-cpu-preflight-1"
STRICT_LOAD_SCHEMA = "v0.2-weights-only-strict-load-v1"
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
NUMPY_SCORE_TOLERANCE = 1e-6


class DINOv2ScoringSmokeError(Exception):
    """Reject a smoke run outside its fixed, non-performance boundary."""


def _load_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DINOv2ScoringSmokeError(f"cannot read {field}: {error}") from error
    if not isinstance(value, dict):
        raise DINOv2ScoringSmokeError(f"{field} must contain a JSON object")
    return value


def _validate_execution_identity(
    *,
    project_root: Path,
    execution_commit: str,
    verification_date: str,
) -> None:
    if not COMMIT_PATTERN.fullmatch(execution_commit):
        raise DINOv2ScoringSmokeError("execution_commit must be a full lowercase Git commit")
    if not DATE_PATTERN.fullmatch(verification_date):
        raise DINOv2ScoringSmokeError("verification_date must use YYYY-MM-DD")
    try:
        observed_commit = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        worktree_status = subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise DINOv2ScoringSmokeError("cannot verify Git execution identity") from error
    if observed_commit != execution_commit:
        raise DINOv2ScoringSmokeError("execution_commit is not the checked-out commit")
    if worktree_status:
        raise DINOv2ScoringSmokeError("worktree must be clean before the smoke run")


def generate_fixed_synthetic_images() -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic reference and query RGB arrays without a dataset."""
    y_coordinates, x_coordinates = np.indices(INPUT_SHAPE[:2], dtype=np.uint16)
    reference = np.empty(INPUT_SHAPE, dtype=np.uint8)
    reference[:, :, 0] = ((3 * x_coordinates + 5 * y_coordinates) % 256).astype(np.uint8)
    reference[:, :, 1] = ((7 * x_coordinates + 2 * y_coordinates + 31) % 256).astype(np.uint8)
    reference[:, :, 2] = ((x_coordinates + 11 * y_coordinates + 73) % 256).astype(np.uint8)
    query = np.roll(reference, shift=(9, -13), axis=(0, 1)).copy()
    query[176:336, 208:304, 0] = 255 - query[176:336, 208:304, 0]
    query[176:336, 208:304, 2] = query[176:336, 208:304, 2] // 3
    if not reference.flags.c_contiguous or not query.flags.c_contiguous:
        raise DINOv2ScoringSmokeError("synthetic inputs must be C-contiguous")
    return reference, query


def _image_identity(image: np.ndarray) -> dict[str, Any]:
    return {
        "byte_count": image.nbytes,
        "c_contiguous": bool(image.flags.c_contiguous),
        "dtype": str(image.dtype),
        "sha256": hashlib.sha256(image.tobytes(order="C")).hexdigest(),
        "shape": list(image.shape),
    }


def _independent_numpy_result(
    query_features: np.ndarray,
    memory_features: np.ndarray,
) -> tuple[np.ndarray, float]:
    similarities = query_features @ memory_features.T
    distances = np.clip(1.0 - similarities, 0.0, 2.0).min(axis=1)
    selected_count = top_patch_count(int(distances.size))
    score = float(
        np.mean(
            np.sort(distances)[::-1][:selected_count],
            dtype=np.float32,
        )
    )
    return distances, score


def _run_resolution(
    *,
    torch: Any,
    model: Any,
    reference_image: np.ndarray,
    query_image: np.ndarray,
    resolution: int,
) -> dict[str, Any]:
    reference_features = extract_dinov2_patch_features(
        reference_image,
        model=model,
        resolution=resolution,
        torch_module=torch,
    )
    query_features = extract_dinov2_patch_features(
        query_image,
        model=model,
        resolution=resolution,
        torch_module=torch,
    )
    repeated_reference_features = torch.cat(
        [reference_features] * REFERENCE_COUNT,
        dim=0,
    )
    memory_bank = create_dinov2_memory_bank(
        repeated_reference_features,
        resolution=resolution,
        reference_count=REFERENCE_COUNT,
        torch_module=torch,
    )
    patch_distances = exact_cosine_min_distances(
        query_features,
        memory_bank=memory_bank,
        torch_module=torch,
    )
    score = aggregate_top_fraction_score(
        patch_distances,
        torch_module=torch,
    )

    query_numpy = query_features.numpy()
    memory_numpy = memory_bank.features.numpy()
    independent_distances, independent_score = _independent_numpy_result(
        query_numpy,
        memory_numpy,
    )
    maximum_distance_difference = float(
        np.max(
            np.abs(
                patch_distances.numpy().astype(np.float64)
                - independent_distances.astype(np.float64)
            )
        )
    )
    score_difference = abs(score - independent_score)
    if score_difference > NUMPY_SCORE_TOLERANCE:
        raise DINOv2ScoringSmokeError(
            "blocked PyTorch scalar score differs from the independent NumPy calculation: "
            f"score_difference={score_difference:.12g}, "
            f"tolerance={NUMPY_SCORE_TOLERANCE:.12g}"
        )

    query_norms = torch.linalg.vector_norm(query_features, ord=2, dim=1)
    reference_norms = torch.linalg.vector_norm(reference_features, ord=2, dim=1)
    return {
        "embedding_dimension": query_features.shape[1],
        "finite": {
            "patch_distances": bool(torch.isfinite(patch_distances).all().item()),
            "query_features": bool(torch.isfinite(query_features).all().item()),
            "reference_features": bool(torch.isfinite(reference_features).all().item()),
            "score": bool(np.isfinite(score)),
        },
        "independent_numpy_check": {
            "maximum_absolute_patch_distance_difference": maximum_distance_difference,
            "patch_distance_difference_is_gating": False,
            "score": independent_score,
            "score_absolute_difference": score_difference,
            "score_tolerance": NUMPY_SCORE_TOLERANCE,
            "score_verification": "pass",
        },
        "memory_bank": {
            "construction": (
                "one synthetic reference embedding repeated twenty times for "
                "implementation smoke only"
            ),
            "patch_count": memory_bank.features.shape[0],
            "reference_count_contract": memory_bank.reference_count,
            "unique_reference_image_count": 1,
        },
        "patch_count": query_features.shape[0],
        "patch_distance_range": {
            "maximum": float(patch_distances.max().item()),
            "minimum": float(patch_distances.min().item()),
        },
        "query_feature_norm_range": {
            "maximum": float(query_norms.max().item()),
            "minimum": float(query_norms.min().item()),
        },
        "reference_feature_norm_range": {
            "maximum": float(reference_norms.max().item()),
            "minimum": float(reference_norms.min().item()),
        },
        "resolution": resolution,
        "score": score,
        "top_patch_count": top_patch_count(query_features.shape[0]),
    }


def run_dinov2_scoring_smoke(
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
    output_path: Path,
) -> dict[str, Any]:
    """Run two-resolution inference without data, labels, or timing."""
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    project_root = project_root.resolve()
    _validate_execution_identity(
        project_root=project_root,
        execution_commit=execution_commit,
        verification_date=verification_date,
    )
    acquisition_record = _load_acquisition_record(acquisition_path)
    import_smoke_record = _load_import_smoke_record(import_smoke_path)
    strict_load_record = _load_json(strict_load_path, field="strict-load record")
    if (
        strict_load_record.get("schema_version") != STRICT_LOAD_SCHEMA
        or strict_load_record.get("decision", {}).get("next_step")
        != "PROCEED_TO_FIXED_DINOV2_SCORING_PATH_IMPLEMENTATION"
    ):
        raise DINOv2ScoringSmokeError(
            "strict-load record does not authorize scoring-path implementation"
        )
    environment = _validate_environment(
        environment_root=environment_root,
        import_smoke_record=import_smoke_record,
    )

    source_artifact = acquisition_record["source"]["artifact"]
    checkpoint_artifact = acquisition_record["checkpoint"]["artifact"]
    source_filename = source_artifact["filename"]
    checkpoint_filename = checkpoint_artifact["filename"]
    source_archive_path = artifact_dir / source_filename
    checkpoint_path = artifact_dir / checkpoint_filename
    source_identity = _verify_asset(
        path=source_archive_path,
        expected_sha256=EXPECTED_SOURCE_SHA256,
        expected_bytes=source_artifact["byte_count"],
        field="source archive",
    )
    checkpoint_identity = _verify_asset(
        path=checkpoint_path,
        expected_sha256=EXPECTED_CHECKPOINT_SHA256,
        expected_bytes=EXPECTED_CHECKPOINT_BYTES,
        field="checkpoint",
    )
    expected_source_root = (
        artifact_dir / f"dinov2-source-sha256-{EXPECTED_SOURCE_SHA256}" / SOURCE_ROOT
    ).resolve()
    if source_root.resolve() != expected_source_root or not source_root.is_dir():
        raise DINOv2ScoringSmokeError("source_root is not the verified hash-addressed extraction")

    reference_image, query_image = generate_fixed_synthetic_images()
    previous_sys_path = list(sys.path)
    for name in tuple(sys.modules):
        if name == "dinov2" or name.startswith("dinov2."):
            raise DINOv2ScoringSmokeError("DINOv2 was imported before the network guard")

    with NetworkGuard() as network_guard:
        try:
            sys.path.insert(0, str(source_root))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                torch = importlib.import_module("torch")
                backbones = importlib.import_module("dinov2.hub.backbones")
            if (
                torch.__version__ != EXPECTED_TORCH_VERSION
                or torch.version.cuda is not None
                or torch.version.hip is not None
            ):
                raise DINOv2ScoringSmokeError("PyTorch is not the fixed CPU build")
            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            state = torch.load(
                checkpoint_path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
            state_summary = _summarize_state_dict(torch, state)
            torch.manual_seed(42)
            model = backbones.dinov2_vits14(pretrained=False)
            model_summary = _model_summary(torch, model, state)
            with torch.inference_mode():
                resolution_results = [
                    _run_resolution(
                        torch=torch,
                        model=model,
                        reference_image=reference_image,
                        query_image=query_image,
                        resolution=resolution,
                    )
                    for resolution in ALLOWED_RESOLUTIONS
                ]
            module_origins = _module_origins(source_root)
            if "xformers" in sys.modules or any(
                name.startswith("xformers.") for name in sys.modules
            ):
                raise DINOv2ScoringSmokeError("xformers was imported unexpectedly")
        finally:
            sys.path[:] = previous_sys_path
    if network_guard.attempts:
        raise DINOv2ScoringSmokeError("a network operation was attempted")

    report = {
        "boundary": {
            "accelerator_runtime_probe_performed": False,
            "dataset_access": False,
            "formal_latency_measurement_performed": False,
            "labels_accessed": False,
            "model_inference_performed": True,
            "network_access": False,
            "performance_claim": False,
            "synthetic_inputs_only": True,
            "threshold_calibration_performed": False,
        },
        "decision": {
            "next_step": "PROCEED_TO_PREREGISTERED_CPU_TIMING_WORKLOAD",
            "reason": (
                "The fixed scoring path completed for both resolutions and "
                "matched an independent exact NumPy calculation within tolerance."
            ),
            "status": "PASS",
        },
        "environment": {
            **environment,
            "accelerator_distribution_count": 0,
            "cuda_build_version": None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "hip_build_version": None,
            "interop_threads": torch.get_num_interop_threads(),
            "intraop_threads": torch.get_num_threads(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "xformers_imported": False,
        },
        "execution": {
            "execution_commit": execution_commit,
            "verification_date": verification_date,
        },
        "fixed_contract": {
            "allowed_resolutions": list(ALLOWED_RESOLUTIONS),
            "embedding_dimension": EMBEDDING_DIMENSION,
            "l2_epsilon": L2_EPSILON,
            "memory_block_size": MEMORY_BLOCK_SIZE,
            "model_entry_point": EXPECTED_MODEL_ENTRY_POINT,
            "reference_count": REFERENCE_COUNT,
            "top_fraction": TOP_FRACTION,
        },
        "inputs": {
            "acquisition_record": acquisition_path.relative_to(project_root).as_posix(),
            "acquisition_record_sha256": sha256_file(acquisition_path),
            "import_smoke_record": import_smoke_path.relative_to(project_root).as_posix(),
            "import_smoke_record_sha256": sha256_file(import_smoke_path),
            "preregistration_id": PREREGISTRATION_ID,
            "strict_load_record": strict_load_path.relative_to(project_root).as_posix(),
            "strict_load_record_sha256": sha256_file(strict_load_path),
            "synthetic_generator": (
                "fixed coordinate formulas, deterministic roll, and fixed rectangle edit"
            ),
            "synthetic_query": _image_identity(query_image),
            "synthetic_reference": _image_identity(reference_image),
        },
        "model": {
            "embedding_dimension": model_summary["embedding_dimension"],
            "entry_point": model_summary["entry_point"],
            "eval_mode": model_summary["eval_mode"],
            "num_register_tokens": model_summary["num_register_tokens"],
            "parameter_count": model_summary["parameter_count"],
            "patch_size": model_summary["patch_size"],
            "strict_load": model_summary["strict_load"],
        },
        "resolutions": resolution_results,
        "schema_version": OUTPUT_SCHEMA,
        "source": {
            "checkpoint_identity": checkpoint_identity,
            "module_import_count": len(module_origins),
            "module_origins": module_origins,
            "source_identity": source_identity,
            "state_key_count": state_summary["key_count"],
        },
        "system": {
            "machine": platform.machine(),
            "platform": sys.platform,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "python_version": platform.python_version(),
        },
    }
    if any(
        result["patch_count"] != expected_patch_count(result["resolution"])
        or result["embedding_dimension"] != EMBEDDING_DIMENSION
        or result["top_patch_count"] != top_patch_count(expected_patch_count(result["resolution"]))
        or not all(result["finite"].values())
        for result in resolution_results
    ):
        raise DINOv2ScoringSmokeError("scoring-path output violated the fixed contract")
    write_json_atomic(output_path, report)
    return report
