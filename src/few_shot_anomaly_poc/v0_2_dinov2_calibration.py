"""Run the fixed v0.2.4 DINOv2 normal-only worker in isolation."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Mapping
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
    extract_dinov2_patch_features,
    score_dinov2_image,
)
from few_shot_anomaly_poc.dinov2_timing import _load_fixed_runtime
from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.model_compatibility import NetworkGuard
from few_shot_anomaly_poc.v0_2_boundary_preparation import (
    RUN_ID,
    validate_boundary_execution_identity,
)
from few_shot_anomaly_poc.v0_2_calibration_artifacts import (
    NORMAL_CALIBRATION_COUNT,
    RGB_INPUT_SHAPE,
    RGB_STORE_SCHEMA,
    TOTAL_NORMAL_COUNT,
    V0_2_4_MILESTONE,
    CalibrationScore,
    build_fit_record,
    calibrate_normal_scores,
    write_method_artifacts,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    EXPECTED_CONFIG_SHA256,
    load_v0_2_artifact_schema,
    load_v0_2_config,
)

METHOD = "dinov2_vits14_224_nn"
RESOLUTION = 224
FAILURE_SCORE = 2.0
MILESTONE_LABEL = V0_2_4_MILESTONE
CALIBRATION_COUNT = NORMAL_CALIBRATION_COUNT
FREEZE_RECORD_SHA256 = "ae552d805dd9648163a48683bad828c7e1b7ecc4f1d69f1fa28511363b08ce3b"
REFERENCE_MANIFEST_SHA256 = "e587f1808262480261ae8a7b940faff0d9ef5f83cf215028b31490ba48369b99"
CALIBRATION_MANIFEST_SHA256 = "77d5adb588e7d463e7fcab1c10b841b9ad23d827b51138c31f42dac35bd99ca3"


class V0_2DINOv2CalibrationError(Exception):
    """Reject an invalid isolated worker input or incomplete worker output."""


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V0_2DINOv2CalibrationError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise V0_2DINOv2CalibrationError(f"{label} must contain one JSON object")
    return value


def _resolve_project_path(path: Path, *, project_root: Path, label: str) -> Path:
    candidate = path if path.is_absolute() else project_root / path
    resolved = candidate.resolve()
    if not resolved.is_relative_to(project_root):
        raise V0_2DINOv2CalibrationError(f"{label} must remain within project_root")
    return resolved


def open_normal_rgb_store(
    *,
    store_path: Path,
    manifest_path: Path,
) -> tuple[np.memmap, dict[str, Any]]:
    """Hash-check the ignored normal RGB store and its label-free manifest."""
    manifest = _read_json(manifest_path, label="normal RGB manifest")
    records = manifest.get("records")
    if (
        manifest.get("schema_version") != RGB_STORE_SCHEMA
        or manifest.get("run_id") != RUN_ID
        or manifest.get("shape") != [TOTAL_NORMAL_COUNT, *RGB_INPUT_SHAPE]
        or manifest.get("dtype") != "uint8"
        or manifest.get("reference_count") != REFERENCE_COUNT
        or manifest.get("calibration_count") != CALIBRATION_COUNT
        or manifest.get("reference_manifest_sha256") != REFERENCE_MANIFEST_SHA256
        or manifest.get("calibration_manifest_sha256") != CALIBRATION_MANIFEST_SHA256
        or manifest.get("labels_accessed") is not False
        or manifest.get("final_test_accessed") is not False
        or not isinstance(records, list)
        or len(records) != TOTAL_NORMAL_COUNT
        or not store_path.is_file()
        or store_path.stat().st_size != manifest.get("store_byte_count")
        or sha256_file(store_path) != manifest.get("store_sha256")
    ):
        raise V0_2DINOv2CalibrationError("normal RGB store identity changed")
    expected_paths: set[str] = set()
    for index, record in enumerate(records):
        expected_partition = "reference" if index < REFERENCE_COUNT else "calibration"
        if (
            not isinstance(record, dict)
            or record.get("index") != index
            or record.get("selection_rank") != index + 1
            or record.get("partition") != expected_partition
            or record.get("adapter_status") not in {"ok", "failed"}
            or not isinstance(record.get("source_path"), str)
            or record["source_path"] in expected_paths
            or not isinstance(record.get("source_sha256"), str)
            or len(record["source_sha256"]) != 64
        ):
            raise V0_2DINOv2CalibrationError("normal RGB manifest record changed")
        expected_paths.add(record["source_path"])
        if record["adapter_status"] == "ok":
            if (
                record.get("adapter_failure_code") is not None
                or not isinstance(record.get("rgb_sha256"), str)
                or len(record["rgb_sha256"]) != 64
            ):
                raise V0_2DINOv2CalibrationError("successful adapter record is invalid")
        elif (
            not isinstance(record.get("adapter_failure_code"), str)
            or record.get("rgb_sha256") is not None
        ):
            raise V0_2DINOv2CalibrationError("failed adapter record is invalid")
    try:
        store = np.load(store_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise V0_2DINOv2CalibrationError("normal RGB store cannot be opened") from error
    if (
        not isinstance(store, np.memmap)
        or store.shape != (TOTAL_NORMAL_COUNT, *RGB_INPUT_SHAPE)
        or store.dtype != np.uint8
        or not store.flags.c_contiguous
    ):
        raise V0_2DINOv2CalibrationError("normal RGB store array changed")
    return store, manifest


def _copy_image(
    store: np.memmap,
    *,
    record: Mapping[str, Any],
) -> NDArray[np.uint8]:
    index = record["index"]
    image = np.array(store[index], dtype=np.uint8, order="C", copy=True)
    if (
        image.shape != RGB_INPUT_SHAPE
        or not image.flags.c_contiguous
        or sha256_file_bytes(image.tobytes(order="C")) != record["rgb_sha256"]
    ):
        raise V0_2DINOv2CalibrationError("normal RGB input changed")
    return image


def sha256_file_bytes(value: bytes) -> str:
    """Hash one in-memory fixed input without importing a second utility."""
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _write_torch_state_atomic(
    path: Path,
    *,
    torch: Any,
    execution_commit: str,
    memory_bank: Any,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary_path: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary_path = Path(name)
        torch.save(
            {
                "milestone": MILESTONE_LABEL,
                "run_id": RUN_ID,
                "method": METHOD,
                "execution_commit": execution_commit,
                "freeze_record_sha256": FREEZE_RECORD_SHA256,
                "v0_2_config_sha256": EXPECTED_CONFIG_SHA256,
                "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
                "resolution": RESOLUTION,
                "reference_count": REFERENCE_COUNT,
                "patch_count_per_reference": memory_bank.patch_count_per_reference,
                "embedding_dimension": memory_bank.embedding_dimension,
                "features": memory_bank.features.detach().contiguous(),
            },
            temporary_path,
        )
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return sha256_file(path)


def _fit_failed_record(
    *,
    successful: int,
    failure_code: str,
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    return build_fit_record(
        run_id=RUN_ID,
        method=METHOD,
        status="fit_failed",
        successful_reference_count=successful,
        failed_reference_count=REFERENCE_COUNT - successful,
        reference_manifest_sha256=REFERENCE_MANIFEST_SHA256,
        fitted_state_sha256=None,
        failure_code=failure_code,
        config=config,
        schema=schema,
    )


def _fit_memory_bank(
    *,
    store: np.memmap,
    records: list[dict[str, Any]],
    model: object,
    torch: Any,
) -> tuple[Any | None, int, str | None]:
    reference_records = sorted(
        records[:REFERENCE_COUNT],
        key=lambda record: record["source_path"],
    )
    patch_count = expected_patch_count(RESOLUTION)
    features = torch.empty(
        (REFERENCE_COUNT * patch_count, EMBEDDING_DIMENSION),
        dtype=torch.float32,
        device="cpu",
    )
    successful = 0
    for record in reference_records:
        if record["adapter_status"] != "ok":
            return None, successful, record["adapter_failure_code"]
        try:
            image = _copy_image(store, record=record)
            extracted = extract_dinov2_patch_features(
                image,
                model=model,
                resolution=RESOLUTION,
                torch_module=torch,
            )
            start = successful * patch_count
            features[start : start + patch_count].copy_(extracted)
        except DINOv2ScoringError as error:
            return None, successful, str(error.code)
        except Exception:
            return None, successful, "DINO_REFERENCE_EXECUTION_FAILED"
        successful += 1
        del image
        del extracted
    try:
        memory_bank = create_dinov2_memory_bank(
            features,
            resolution=RESOLUTION,
            reference_count=REFERENCE_COUNT,
            torch_module=torch,
        )
    except DINOv2ScoringError as error:
        return None, successful, str(error.code)
    return memory_bank, successful, None


def _score_calibration(
    *,
    store: np.memmap,
    records: list[dict[str, Any]],
    model: object,
    memory_bank: Any,
    torch: Any,
) -> list[CalibrationScore]:
    scores: list[CalibrationScore] = []
    calibration_records = sorted(
        records[REFERENCE_COUNT:],
        key=lambda record: record["source_path"],
    )
    for index, record in enumerate(calibration_records, start=1):
        if record["adapter_status"] != "ok":
            score = CalibrationScore(
                source_path=record["source_path"],
                score_status="failed",
                score_failure_code=record["adapter_failure_code"],
                anomaly_score=FAILURE_SCORE,
            )
        else:
            try:
                image = _copy_image(store, record=record)
                anomaly_score = score_dinov2_image(
                    image,
                    model=model,
                    memory_bank=memory_bank,
                    resolution=RESOLUTION,
                    torch_module=torch,
                )
            except DINOv2ScoringError as error:
                score = CalibrationScore(
                    source_path=record["source_path"],
                    score_status="failed",
                    score_failure_code=str(error.code),
                    anomaly_score=FAILURE_SCORE,
                )
            except Exception:
                score = CalibrationScore(
                    source_path=record["source_path"],
                    score_status="failed",
                    score_failure_code="DINO_SCORE_EXECUTION_FAILED",
                    anomaly_score=FAILURE_SCORE,
                )
            else:
                score = CalibrationScore(
                    source_path=record["source_path"],
                    score_status="ok",
                    score_failure_code=None,
                    anomaly_score=float(anomaly_score),
                )
                del image
        scores.append(score)
        if index % 25 == 0 or index == CALIBRATION_COUNT:
            print(
                f"DINOv2 normal-only scoring: {index}/{CALIBRATION_COUNT}",
                flush=True,
            )
    return scores


def run_dinov2_normal_fit_and_calibration(
    *,
    project_root: Path,
    execution_commit: str,
    input_store_path: Path,
    input_manifest_path: Path,
    acquisition_path: Path,
    import_smoke_path: Path,
    strict_load_path: Path,
    artifact_dir: Path,
    source_root: Path,
    environment_root: Path,
    output_dir: Path,
    state_path: Path,
) -> dict[str, Any]:
    """Fit and calibrate DINOv2 once without any final-test input."""
    project_root = project_root.resolve()
    resolved = {
        name: _resolve_project_path(path, project_root=project_root, label=name)
        for name, path in {
            "input_store_path": input_store_path,
            "input_manifest_path": input_manifest_path,
            "acquisition_path": acquisition_path,
            "import_smoke_path": import_smoke_path,
            "strict_load_path": strict_load_path,
            "artifact_dir": artifact_dir,
            "source_root": source_root,
            "environment_root": environment_root,
            "output_dir": output_dir,
            "state_path": state_path,
        }.items()
    }
    if resolved["output_dir"].exists() or resolved["state_path"].exists():
        raise FileExistsError("refusing to overwrite DINOv2 calibration output")
    validate_boundary_execution_identity(
        project_root=project_root,
        execution_commit=execution_commit,
    )
    config = load_v0_2_config(project_root / "configs/v0.2.yaml")
    schema = load_v0_2_artifact_schema(project_root / "schemas/v0.2/evaluation-artifacts.json")
    store, manifest = open_normal_rgb_store(
        store_path=resolved["input_store_path"],
        manifest_path=resolved["input_manifest_path"],
    )
    records = manifest["records"]
    failed_reference_adapters = [
        record for record in records[:REFERENCE_COUNT] if record["adapter_status"] != "ok"
    ]
    if failed_reference_adapters:
        fit = _fit_failed_record(
            successful=REFERENCE_COUNT - len(failed_reference_adapters),
            failure_code=failed_reference_adapters[0]["adapter_failure_code"],
            config=config,
            schema=schema,
        )
        write_method_artifacts(
            method_root=resolved["output_dir"],
            fit_record=fit,
            calibration=None,
        )
        return fit

    torch: Any | None = None
    model: object | None = None
    network_guard: NetworkGuard | None = None
    previous_sys_path: list[str] | None = None
    try:
        torch, model, _, network_guard, previous_sys_path = _load_fixed_runtime(
            acquisition_path=resolved["acquisition_path"],
            import_smoke_path=resolved["import_smoke_path"],
            strict_load_path=resolved["strict_load_path"],
            artifact_dir=resolved["artifact_dir"],
            source_root=resolved["source_root"],
            environment_root=resolved["environment_root"],
        )

        memory_bank, successful, failure_code = _fit_memory_bank(
            store=store,
            records=records,
            model=model,
            torch=torch,
        )
        if memory_bank is None:
            fit = _fit_failed_record(
                successful=successful,
                failure_code=failure_code or "DINO_FIT_FAILED",
                config=config,
                schema=schema,
            )
            write_method_artifacts(
                method_root=resolved["output_dir"],
                fit_record=fit,
                calibration=None,
            )
            return fit
        try:
            state_sha256 = _write_torch_state_atomic(
                resolved["state_path"],
                torch=torch,
                execution_commit=execution_commit,
                memory_bank=memory_bank,
            )
        except Exception:
            resolved["state_path"].unlink(missing_ok=True)
            fit = _fit_failed_record(
                successful=REFERENCE_COUNT,
                failure_code="DINO_STATE_PERSISTENCE_FAILED",
                config=config,
                schema=schema,
            )
            write_method_artifacts(
                method_root=resolved["output_dir"],
                fit_record=fit,
                calibration=None,
            )
            return fit
        fit = build_fit_record(
            run_id=RUN_ID,
            method=METHOD,
            status="fit_ok",
            successful_reference_count=REFERENCE_COUNT,
            failed_reference_count=0,
            reference_manifest_sha256=REFERENCE_MANIFEST_SHA256,
            fitted_state_sha256=state_sha256,
            failure_code=None,
            config=config,
            schema=schema,
        )
        scores = _score_calibration(
            store=store,
            records=records,
            model=model,
            memory_bank=memory_bank,
            torch=torch,
        )
        calibration = calibrate_normal_scores(
            scores,
            run_id=RUN_ID,
            method=METHOD,
            config=config,
            schema=schema,
        )
        write_method_artifacts(
            method_root=resolved["output_dir"],
            fit_record=fit,
            calibration=calibration,
        )
        return fit
    finally:
        if previous_sys_path is not None:
            sys.path[:] = previous_sys_path
        if network_guard is not None:
            network_guard.__exit__(None, None, None)
            if network_guard.attempts:
                raise V0_2DINOv2CalibrationError("network operation was attempted")
