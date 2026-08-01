"""Run the preregistered memory-bounded DINOv2 CPU timing workload."""

from __future__ import annotations

import hashlib
import math
import os
import resource
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from few_shot_anomaly_poc.dinov2_scoring import INPUT_SHAPE, REFERENCE_COUNT
from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.jsonio import write_json_atomic

PREREGISTRATION_ID = "v0.2-dinov2-cpu-preflight-2"
INPUT_STORE_SCHEMA = "v0.2-dinov2-timing-input-store-v1"
GENERATOR_SEED = 42
QUERY_COUNT = 100
TOTAL_IMAGE_COUNT = REFERENCE_COUNT + QUERY_COUNT
STORE_SHAPE = (TOTAL_IMAGE_COUNT, *INPUT_SHAPE)
LOGICAL_STORE_ID = "synthetic-pcg64-42-memory-bounded-v1"
REFERENCE_IDS = tuple(f"synthetic/reference/{index:03d}" for index in range(REFERENCE_COUNT))
QUERY_IDS = tuple(f"synthetic/query/{index:03d}" for index in range(QUERY_COUNT))
WARMUP_COUNT = 25
TIMED_PASS_COUNT = 3
TIMED_INVOCATION_COUNT = QUERY_COUNT * TIMED_PASS_COUNT
P95_NEAREST_RANK = 285
LATENCY_GATE_NS = 1_000_000_000


class DINOv2TimingError(Exception):
    """Reject a timing operation outside the fixed memory-bounded contract."""


def _image_sha256(image: NDArray[np.uint8]) -> str:
    return hashlib.sha256(image.tobytes(order="C")).hexdigest()


def _validate_store_paths(*, store_path: Path, manifest_path: Path) -> None:
    if store_path == manifest_path:
        raise DINOv2TimingError("store_path and manifest_path must differ")
    if store_path.suffix != ".npy":
        raise DINOv2TimingError("store_path must use the .npy suffix")
    if store_path.exists():
        raise FileExistsError(f"refusing to overwrite {store_path}")
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")


def _input_record(*, asset_id: str, index: int, image: NDArray[np.uint8]) -> dict[str, Any]:
    return {
        "asset_id": asset_id,
        "byte_count": int(image.nbytes),
        "dtype": str(image.dtype),
        "index": index,
        "sha256": _image_sha256(image),
        "shape": list(image.shape),
    }


def create_synthetic_input_store(
    *,
    store_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Generate the fixed PCG64 arrays into a new local memory-mapped store."""
    store_path = store_path.resolve()
    manifest_path = manifest_path.resolve()
    _validate_store_paths(store_path=store_path, manifest_path=manifest_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = store_path.with_name(f".{store_path.name}.partial")
    if partial_path.exists():
        raise FileExistsError(f"refusing to overwrite {partial_path}")

    references: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    generator = np.random.Generator(np.random.PCG64(GENERATOR_SEED))
    try:
        store = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.uint8,
            shape=STORE_SHAPE,
        )
        for index, asset_id in enumerate((*REFERENCE_IDS, *QUERY_IDS)):
            image = generator.integers(
                0,
                256,
                size=INPUT_SHAPE,
                dtype=np.uint8,
            )
            store[index] = image
            record = _input_record(asset_id=asset_id, index=index, image=image)
            if index < REFERENCE_COUNT:
                references.append(record)
            else:
                queries.append(record)
            del image
        store.flush()
        del store
        os.replace(partial_path, store_path)
        manifest = {
            "boundary": {
                "dataset_access": False,
                "labels_accessed": False,
                "model_inference_performed": False,
                "network_access": False,
                "synthetic_inputs_only": True,
                "timing_invocation_count": 0,
            },
            "generation": {
                "bit_generator": "numpy.random.PCG64",
                "generation_order": "20 references followed by 100 queries",
                "high_exclusive": 256,
                "low_inclusive": 0,
                "numpy_version": np.__version__,
                "seed": GENERATOR_SEED,
            },
            "logical_store": {
                "byte_count": store_path.stat().st_size,
                "dtype": "uint8",
                "file_sha256": sha256_file(store_path),
                "format": "NumPy .npy memory map",
                "logical_id": LOGICAL_STORE_ID,
                "path_recorded": False,
                "shape": list(STORE_SHAPE),
            },
            "preregistration_id": PREREGISTRATION_ID,
            "queries": queries,
            "references": references,
            "resident_policy": {
                "all_source_images_retained_in_process_memory": False,
                "current_source_image_count": 1,
                "memory_map_to_contiguous_copy_outside_timer": True,
            },
            "schema_version": INPUT_STORE_SCHEMA,
        }
        write_json_atomic(manifest_path, manifest)
    except Exception:
        partial_path.unlink(missing_ok=True)
        if not manifest_path.exists():
            store_path.unlink(missing_ok=True)
        raise
    return manifest


def validate_synthetic_input_manifest(manifest: object) -> dict[str, Any]:
    """Validate the fixed logical IDs, shapes, ordering, and record counts."""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != INPUT_STORE_SCHEMA:
        raise DINOv2TimingError("input manifest schema is invalid")
    if manifest.get("preregistration_id") != PREREGISTRATION_ID:
        raise DINOv2TimingError("input manifest preregistration identity is invalid")
    logical_store = manifest.get("logical_store")
    generation = manifest.get("generation")
    references = manifest.get("references")
    queries = manifest.get("queries")
    if (
        not isinstance(logical_store, dict)
        or logical_store.get("logical_id") != LOGICAL_STORE_ID
        or logical_store.get("shape") != list(STORE_SHAPE)
        or logical_store.get("dtype") != "uint8"
        or logical_store.get("path_recorded") is not False
        or not isinstance(generation, dict)
        or generation.get("bit_generator") != "numpy.random.PCG64"
        or generation.get("seed") != GENERATOR_SEED
        or not isinstance(references, list)
        or not isinstance(queries, list)
        or len(references) != REFERENCE_COUNT
        or len(queries) != QUERY_COUNT
    ):
        raise DINOv2TimingError("input manifest fixed contract is invalid")
    expected = [
        (asset_id, index)
        for index, asset_id in enumerate((*REFERENCE_IDS, *QUERY_IDS))
    ]
    observed = [
        (record.get("asset_id"), record.get("index"))
        for record in (*references, *queries)
        if isinstance(record, dict)
    ]
    if observed != expected:
        raise DINOv2TimingError("input manifest IDs or order are invalid")
    for record in (*references, *queries):
        if (
            not isinstance(record, dict)
            or record.get("byte_count") != int(np.prod(INPUT_SHAPE))
            or record.get("dtype") != "uint8"
            or record.get("shape") != list(INPUT_SHAPE)
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
        ):
            raise DINOv2TimingError("input manifest image identity is invalid")
    return manifest


def open_verified_synthetic_input_store(
    *,
    store_path: Path,
    manifest: object,
) -> np.memmap:
    """Open and verify the fixed store without retaining its images in RAM."""
    validated_manifest = validate_synthetic_input_manifest(manifest)
    store_path = store_path.resolve()
    logical_store = validated_manifest["logical_store"]
    if not store_path.is_file():
        raise DINOv2TimingError("input store is unavailable")
    if store_path.stat().st_size != logical_store.get("byte_count"):
        raise DINOv2TimingError("input store byte count changed")
    if sha256_file(store_path) != logical_store.get("file_sha256"):
        raise DINOv2TimingError("input store SHA-256 changed")
    try:
        store = np.load(store_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise DINOv2TimingError("input store cannot be opened safely") from error
    if (
        not isinstance(store, np.memmap)
        or store.shape != STORE_SHAPE
        or store.dtype != np.uint8
        or not store.flags.c_contiguous
    ):
        raise DINOv2TimingError("input store array contract is invalid")
    return store


def copy_store_image(
    store: object,
    *,
    index: int,
    expected_sha256: str,
) -> NDArray[np.uint8]:
    """Copy one store item into the fixed contiguous decoded-image boundary."""
    if not isinstance(store, np.memmap) or store.shape != STORE_SHAPE or store.dtype != np.uint8:
        raise DINOv2TimingError("a validated input memory map is required")
    if type(index) is not int or not 0 <= index < TOTAL_IMAGE_COUNT:
        raise DINOv2TimingError("input store index is invalid")
    image = np.array(store[index], dtype=np.uint8, order="C", copy=True)
    if image.shape != INPUT_SHAPE or not image.flags.c_contiguous:
        raise DINOv2TimingError("copied input image contract is invalid")
    if _image_sha256(image) != expected_sha256:
        raise DINOv2TimingError("copied input image SHA-256 changed")
    return image


def peak_rss_bytes() -> int:
    """Return the Linux process high-water resident set in bytes."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DINOv2TimingError("process peak RSS is unavailable")
    return value * 1_024


def _exception_record(error: BaseException, *, phase: str) -> dict[str, str]:
    current: BaseException | None = error
    observed_text: list[str] = []
    while current is not None:
        observed_text.append(str(current).lower())
        current = current.__cause__ or current.__context__
    combined = " ".join(observed_text)
    if isinstance(error, MemoryError):
        category = "memory_error"
    elif any(
        marker in combined
        for marker in (
            "out of memory",
            "cannot allocate memory",
            "can't allocate memory",
            "bad allocation",
            "std::bad_alloc",
        )
    ):
        category = "framework_out_of_memory"
    else:
        category = "execution_error"
    return {
        "category": category,
        "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "phase": phase,
    }


def _validate_score(score: object) -> float:
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
        or not 0.0 <= float(score) <= 2.0
    ):
        raise DINOv2TimingError("image score must be finite and within [0, 2]")
    return float(score)


def summarize_timing_observations(observations: object) -> dict[str, Any]:
    """Apply the fixed 300-observation median, p95, and completion gate."""
    if not isinstance(observations, list) or len(observations) > TIMED_INVOCATION_COUNT:
        raise DINOv2TimingError("timing observations must be a bounded list")
    successful = [
        item
        for item in observations
        if isinstance(item, dict) and item.get("status") == "success"
    ]
    failures = [
        item
        for item in observations
        if isinstance(item, dict) and item.get("status") == "failure"
    ]
    if len(successful) + len(failures) != len(observations):
        raise DINOv2TimingError("timing observation status is invalid")
    durations = [item.get("duration_ns") for item in successful]
    if any(type(value) is not int or value < 0 for value in durations):
        raise DINOv2TimingError("successful duration_ns must be a non-negative integer")
    complete = (
        len(observations) == TIMED_INVOCATION_COUNT
        and len(successful) == TIMED_INVOCATION_COUNT
        and not failures
    )
    median_ns: float | None = None
    p95_ns: int | None = None
    if complete:
        ordered = sorted(durations)
        median_ns = (ordered[149] + ordered[150]) / 2
        p95_ns = ordered[P95_NEAREST_RANK - 1]
    return {
        "attempted_invocation_count": len(observations),
        "complete_observation_set": complete,
        "failure_count": len(failures),
        "latency_gate_ns": LATENCY_GATE_NS,
        "latency_gate_passed": complete and p95_ns is not None and p95_ns <= LATENCY_GATE_NS,
        "median_ns": median_ns,
        "missing_invocation_count": TIMED_INVOCATION_COUNT - len(observations),
        "p95_method": "nearest-rank",
        "p95_nearest_rank": P95_NEAREST_RANK,
        "p95_ns": p95_ns,
        "successful_invocation_count": len(successful),
        "timed_invocation_count_required": TIMED_INVOCATION_COUNT,
    }


def execute_fixed_timing_loop(
    *,
    copy_query: Callable[[int], NDArray[np.uint8]],
    score_image: Callable[[NDArray[np.uint8]], float],
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Execute the fixed warm-up and timed query order without retries."""
    warmup_completed = 0
    warmup_failure: dict[str, str] | None = None
    for _ in range(WARMUP_COUNT):
        try:
            image = copy_query(0)
            _validate_score(score_image(image))
            warmup_completed += 1
        except Exception as error:
            warmup_failure = _exception_record(error, phase="warmup")
            break

    observations: list[dict[str, Any]] = []
    if warmup_failure is None:
        stop = False
        for pass_index in range(TIMED_PASS_COUNT):
            for query_index, asset_id in enumerate(QUERY_IDS):
                invocation_index = pass_index * QUERY_COUNT + query_index
                try:
                    image = copy_query(query_index)
                except Exception as error:
                    observations.append(
                        {
                            "asset_id": asset_id,
                            "duration_ns": None,
                            "failure": _exception_record(error, phase="input_copy"),
                            "invocation_index": invocation_index,
                            "pass_index": pass_index,
                            "query_index": query_index,
                            "score": None,
                            "status": "failure",
                        }
                    )
                    stop = True
                    break
                try:
                    started_ns = clock_ns()
                    score = _validate_score(score_image(image))
                    finished_ns = clock_ns()
                    if type(started_ns) is not int or type(finished_ns) is not int:
                        raise DINOv2TimingError("clock values must be integers")
                    duration_ns = finished_ns - started_ns
                    if duration_ns < 0:
                        raise DINOv2TimingError("clock duration must not be negative")
                except Exception as error:
                    try:
                        failure_finished_ns = clock_ns()
                        failure_duration_ns = (
                            failure_finished_ns - started_ns
                            if type(failure_finished_ns) is int
                            and type(started_ns) is int
                            and failure_finished_ns >= started_ns
                            else None
                        )
                    except Exception:
                        failure_duration_ns = None
                    observations.append(
                        {
                            "asset_id": asset_id,
                            "duration_ns": failure_duration_ns,
                            "failure": _exception_record(error, phase="timed_scoring"),
                            "invocation_index": invocation_index,
                            "pass_index": pass_index,
                            "query_index": query_index,
                            "score": None,
                            "status": "failure",
                        }
                    )
                    stop = True
                    break
                observations.append(
                    {
                        "asset_id": asset_id,
                        "duration_ns": duration_ns,
                        "failure": None,
                        "invocation_index": invocation_index,
                        "pass_index": pass_index,
                        "query_index": query_index,
                        "score": score,
                        "status": "success",
                    }
                )
            if stop:
                break

    return {
        "observations": observations,
        "summary": summarize_timing_observations(observations),
        "warmup": {
            "completed_count": warmup_completed,
            "failure": warmup_failure,
            "query_id": QUERY_IDS[0],
            "required_count": WARMUP_COUNT,
            "status": "pass" if warmup_failure is None else "failure",
        },
    }
