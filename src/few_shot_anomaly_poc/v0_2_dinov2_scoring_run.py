"""Run the isolated v0.2.5 DINOv2 label-free scoring worker."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from few_shot_anomaly_poc.dinov2_errors import DINOv2ScoringError
from few_shot_anomaly_poc.dinov2_scoring import (
    EMBEDDING_DIMENSION,
    REFERENCE_COUNT,
    create_dinov2_memory_bank,
    expected_patch_count,
    score_dinov2_image,
)
from few_shot_anomaly_poc.dinov2_timing import _load_fixed_runtime
from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.jsonio import write_json_atomic
from few_shot_anomaly_poc.model_compatibility import NetworkGuard
from few_shot_anomaly_poc.v0_2_boundary_preparation import (
    RUN_ID,
    validate_boundary_execution_identity,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    EXPECTED_CONFIG_SHA256,
    load_v0_2_artifact_schema,
    load_v0_2_config,
)
from few_shot_anomaly_poc.v0_2_scoring_artifacts import (
    ASSET_COUNT,
    TIMED_PASS_COUNT,
    ScoreEvidence,
    TimedScoreEvidence,
    build_method_scoring_artifacts,
    latency_summary,
    read_method_scoring_artifacts,
    write_method_scoring_artifacts,
)

METHOD = "dinov2_vits14_224_nn"
RESOLUTION = 224
FAILURE_SCORE = 2.0
WORKER_REPORT_SCHEMA = "v0.2.5-dinov2-worker-report-v1"
DINO_STORE_SCHEMA = "v0.2.5-dinov2-label-free-rgb512-store-v1"
DINO_INPUT_SHAPE = (512, 512, 3)
DINO_STORE_SHAPE = (ASSET_COUNT, *DINO_INPUT_SHAPE)
OPAQUE_SCORING_MANIFEST_SHA256 = "32ea52ed1b9872f39ae27f5d58a353ea84b8b143642e3a7f0fabe940184705e8"
CALIBRATION_EXECUTION_COMMIT = "6548bfa97e7e834bd88b0efbbd1b557ae85e242c"
FREEZE_RECORD_SHA256 = "ae552d805dd9648163a48683bad828c7e1b7ecc4f1d69f1fa28511363b08ce3b"
REFERENCE_MANIFEST_SHA256 = "e587f1808262480261ae8a7b940faff0d9ef5f83cf215028b31490ba48369b99"


class V0_2DINOv2ScoringRunError(Exception):
    """Reject invalid isolated inputs or incomplete DINOv2 scoring evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V0_2DINOv2ScoringRunError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V0_2DINOv2ScoringRunError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise V0_2DINOv2ScoringRunError(f"{label} must contain one JSON object")
    return value


def _resolve(path: Path, *, project_root: Path, label: str) -> Path:
    candidate = path if path.is_absolute() else project_root / path
    resolved = candidate.resolve()
    if not resolved.is_relative_to(project_root):
        raise V0_2DINOv2ScoringRunError(f"{label} must remain within project_root")
    return resolved


def _open_store(
    *,
    store_path: Path,
    manifest_path: Path,
) -> tuple[np.memmap, list[dict[str, Any]]]:
    manifest = _read_json(manifest_path, label="DINOv2 scoring input manifest")
    records = manifest.get("records")
    expected_keys = {
        "schema_version",
        "run_id",
        "opaque_scoring_manifest_sha256",
        "shape",
        "dtype",
        "store_byte_count",
        "store_sha256",
        "records",
        "labels_accessed",
        "semantic_paths_accessed",
        "sealed_mapping_accessed",
    }
    _require(
        set(manifest) == expected_keys
        and manifest.get("schema_version") == DINO_STORE_SCHEMA
        and manifest.get("run_id") == RUN_ID
        and manifest.get("opaque_scoring_manifest_sha256") == OPAQUE_SCORING_MANIFEST_SHA256
        and manifest.get("shape") == list(DINO_STORE_SHAPE)
        and manifest.get("dtype") == "uint8"
        and manifest.get("labels_accessed") is False
        and manifest.get("semantic_paths_accessed") is False
        and manifest.get("sealed_mapping_accessed") is False
        and isinstance(records, list)
        and len(records) == ASSET_COUNT
        and store_path.is_file()
        and store_path.stat().st_size == manifest.get("store_byte_count")
        and sha256_file(store_path) == manifest.get("store_sha256"),
        "DINOv2 scoring input store identity changed",
    )
    for index, record in enumerate(records):
        durations = record.get("adapter_duration_ns") if isinstance(record, dict) else None
        _require(
            isinstance(record, dict)
            and set(record) == {"asset_id", "index", "rgb_sha256", "adapter_duration_ns"}
            and record.get("asset_id") == f"asset-{index:06d}"
            and record.get("index") == index
            and isinstance(record.get("rgb_sha256"), str)
            and len(record["rgb_sha256"]) == 64
            and isinstance(durations, list)
            and len(durations) == TIMED_PASS_COUNT
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in durations
            ),
            "DINOv2 scoring input record changed",
        )
    try:
        store = np.load(store_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise V0_2DINOv2ScoringRunError("cannot open DINOv2 scoring input store") from error
    _require(
        isinstance(store, np.memmap)
        and store.shape == DINO_STORE_SHAPE
        and store.dtype == np.uint8
        and store.flags.c_contiguous,
        "DINOv2 scoring input array changed",
    )
    return store, records


def _copy_image(
    store: np.memmap,
    *,
    record: dict[str, Any],
) -> NDArray[np.uint8]:
    image = np.array(store[record["index"]], dtype=np.uint8, order="C", copy=True)
    import hashlib

    _require(
        image.shape == DINO_INPUT_SHAPE
        and image.flags.c_contiguous
        and hashlib.sha256(image.tobytes(order="C")).hexdigest() == record["rgb_sha256"],
        "DINOv2 scoring input bytes changed",
    )
    return image


def _load_memory_bank(
    path: Path,
    *,
    expected_sha256: str,
    torch: Any,
) -> Any:
    _require(
        path.is_file() and sha256_file(path) == expected_sha256,
        "DINOv2 fitted-state identity changed",
    )
    try:
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception as error:
        raise V0_2DINOv2ScoringRunError("cannot load DINOv2 fitted state") from error
    expected_keys = {
        "milestone",
        "run_id",
        "method",
        "execution_commit",
        "freeze_record_sha256",
        "v0_2_config_sha256",
        "reference_manifest_sha256",
        "resolution",
        "reference_count",
        "patch_count_per_reference",
        "embedding_dimension",
        "features",
    }
    _require(
        isinstance(state, dict)
        and set(state) == expected_keys
        and state.get("milestone") == "v0.2.4"
        and state.get("run_id") == RUN_ID
        and state.get("method") == METHOD
        and state.get("execution_commit") == CALIBRATION_EXECUTION_COMMIT
        and state.get("freeze_record_sha256") == FREEZE_RECORD_SHA256
        and state.get("v0_2_config_sha256") == EXPECTED_CONFIG_SHA256
        and state.get("reference_manifest_sha256") == REFERENCE_MANIFEST_SHA256
        and state.get("resolution") == RESOLUTION
        and state.get("reference_count") == REFERENCE_COUNT
        and state.get("patch_count_per_reference") == expected_patch_count(RESOLUTION)
        and state.get("embedding_dimension") == EMBEDDING_DIMENSION,
        "DINOv2 fitted-state metadata changed",
    )
    return create_dinov2_memory_bank(
        state["features"],
        resolution=RESOLUTION,
        reference_count=REFERENCE_COUNT,
        torch_module=torch,
    )


def _score(
    image: NDArray[np.uint8],
    *,
    asset_id: str,
    model: object,
    memory_bank: object,
    torch: Any,
) -> ScoreEvidence:
    try:
        value = score_dinov2_image(
            image,
            model=model,
            memory_bank=memory_bank,
            resolution=RESOLUTION,
            torch_module=torch,
        )
    except DINOv2ScoringError as error:
        return ScoreEvidence(asset_id, "failed", str(error.code), FAILURE_SCORE, {})
    except Exception:
        return ScoreEvidence(asset_id, "failed", "DINO_SCORE_EXECUTION_FAILED", FAILURE_SCORE, {})
    return ScoreEvidence(asset_id, "ok", None, float(value), {})


def run_dinov2_label_free_scoring(
    *,
    project_root: Path,
    execution_commit: str,
    input_store_path: Path,
    input_manifest_path: Path,
    fitted_state_path: Path,
    fitted_state_sha256: str,
    threshold: float,
    acquisition_path: Path,
    import_smoke_path: Path,
    strict_load_path: Path,
    artifact_dir: Path,
    source_root: Path,
    environment_root: Path,
    output_dir: Path,
    report_path: Path,
) -> None:
    """Run one complete isolated DINOv2 bundle with no label-bearing input."""
    project_root = project_root.resolve()
    resolved = {
        name: _resolve(path, project_root=project_root, label=name)
        for name, path in {
            "input_store_path": input_store_path,
            "input_manifest_path": input_manifest_path,
            "fitted_state_path": fitted_state_path,
            "acquisition_path": acquisition_path,
            "import_smoke_path": import_smoke_path,
            "strict_load_path": strict_load_path,
            "artifact_dir": artifact_dir,
            "source_root": source_root,
            "environment_root": environment_root,
            "output_dir": output_dir,
            "report_path": report_path,
        }.items()
    }
    _require(not resolved["output_dir"].exists(), "DINOv2 scoring output already exists")
    _require(not resolved["report_path"].exists(), "DINOv2 worker report already exists")
    validate_boundary_execution_identity(
        project_root=project_root,
        execution_commit=execution_commit,
    )
    load_v0_2_config(project_root / "configs/v0.2.yaml")
    schema = load_v0_2_artifact_schema(project_root / "schemas/v0.2/evaluation-artifacts.json")
    store, records = _open_store(
        store_path=resolved["input_store_path"],
        manifest_path=resolved["input_manifest_path"],
    )
    network_guard: NetworkGuard | None = None
    previous_sys_path: list[str] | None = None
    try:
        torch, model, fixed_runtime, network_guard, previous_sys_path = _load_fixed_runtime(
            acquisition_path=resolved["acquisition_path"],
            import_smoke_path=resolved["import_smoke_path"],
            strict_load_path=resolved["strict_load_path"],
            artifact_dir=resolved["artifact_dir"],
            source_root=resolved["source_root"],
            environment_root=resolved["environment_root"],
        )
        memory_bank = _load_memory_bank(
            resolved["fitted_state_path"],
            expected_sha256=fitted_state_sha256,
            torch=torch,
        )
        for index, record in enumerate(records, start=1):
            image = _copy_image(store, record=record)
            _score(
                image,
                asset_id=record["asset_id"],
                model=model,
                memory_bank=memory_bank,
                torch=torch,
            )
            if index % 50 == 0 or index == ASSET_COUNT:
                print(f"DINOv2 scorer warm-up: {index}/{ASSET_COUNT}", flush=True)

        canonical: list[ScoreEvidence] = []
        timed: list[TimedScoreEvidence] = []
        for pass_index in range(TIMED_PASS_COUNT):
            for index, record in enumerate(records, start=1):
                image = _copy_image(store, record=record)
                started = time.perf_counter_ns()
                evidence = _score(
                    image,
                    asset_id=record["asset_id"],
                    model=model,
                    memory_bank=memory_bank,
                    torch=torch,
                )
                scorer_duration = max(1, time.perf_counter_ns() - started)
                if pass_index == 0:
                    canonical.append(evidence)
                timed.append(
                    TimedScoreEvidence(
                        pass_index=pass_index,
                        asset_id=record["asset_id"],
                        adapter_duration_ns=record["adapter_duration_ns"][pass_index],
                        scorer_duration_ns=scorer_duration,
                        score_status=evidence.score_status,
                        score_failure_code=evidence.score_failure_code,
                        anomaly_score=evidence.anomaly_score,
                    )
                )
                if index % 50 == 0 or index == ASSET_COUNT:
                    print(
                        f"DINOv2 scorer timed pass {pass_index + 1}/{TIMED_PASS_COUNT}: "
                        f"{index}/{ASSET_COUNT}",
                        flush=True,
                    )
        artifacts = build_method_scoring_artifacts(
            run_id=RUN_ID,
            method=METHOD,
            threshold=threshold,
            scores=canonical,
            timed_scores=timed,
            schema=schema,
        )
        write_method_scoring_artifacts(resolved["output_dir"], artifacts)
        serialized = read_method_scoring_artifacts(resolved["output_dir"], schema=schema)
        write_json_atomic(
            resolved["report_path"],
            {
                "schema_version": WORKER_REPORT_SCHEMA,
                "run_id": RUN_ID,
                "execution_commit": execution_commit,
                "method": METHOD,
                "record_counts": {
                    "scores": ASSET_COUNT,
                    "classifications": ASSET_COUNT,
                    "latency": ASSET_COUNT * TIMED_PASS_COUNT,
                },
                "latency_summary": latency_summary(serialized.latency_records),
                "environment": fixed_runtime["environment"],
                "fitted_state_sha256": fitted_state_sha256,
                "network_attempted": False,
                "labels_accessed": False,
                "semantic_paths_accessed": False,
                "sealed_mapping_accessed": False,
            },
        )
    finally:
        if previous_sys_path is not None:
            sys.path[:] = previous_sys_path
        if network_guard is not None:
            network_guard.__exit__(None, None, None)
            if network_guard.attempts:
                raise V0_2DINOv2ScoringRunError("network operation was attempted")
