"""Run v0.2.4 fitting and threshold calibration from normal inputs only."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from few_shot_anomaly_poc.config import ProjectConfig, load_config
from few_shot_anomaly_poc.ecc_residual import score_ecc_residual
from few_shot_anomaly_poc.ecc_template import ECCTemplateFitResult, fit_ecc_normal_template
from few_shot_anomaly_poc.errors import ImagePreprocessingError
from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.hog_features import PatchHOGFeatureResult, extract_patch_hog_features
from few_shot_anomaly_poc.hog_models import (
    PatchHOGModelFitResult,
    fit_position_one_class_svms,
)
from few_shot_anomaly_poc.hog_scalers import PatchHOGScalerFitResult, fit_position_scalers
from few_shot_anomaly_poc.hog_scoring import score_patch_hog
from few_shot_anomaly_poc.jsonio import write_json_atomic
from few_shot_anomaly_poc.model_assets import SOURCE_ROOT
from few_shot_anomaly_poc.model_compatibility import EXPECTED_SOURCE_SHA256
from few_shot_anomaly_poc.preprocessing import load_and_preprocess_image
from few_shot_anomaly_poc.v0_2_boundary_preparation import (
    BOUNDARY_STATE_SCHEMA,
    NORMAL_MANIFEST_SCHEMA,
    RUN_ID,
    validate_boundary_execution_identity,
)
from few_shot_anomaly_poc.v0_2_calibration_artifacts import (
    NORMAL_CALIBRATION_COUNT,
    NORMAL_REFERENCE_COUNT,
    RGB_INPUT_SHAPE,
    RGB_STORE_SCHEMA,
    TOTAL_NORMAL_COUNT,
    V0_2_4_MILESTONE,
    CalibrationResult,
    CalibrationScore,
    build_fit_record,
    calibrate_normal_scores,
    write_method_artifacts,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_SCHEMA_SHA256,
    METHODS,
    load_v0_2_artifact_schema,
    load_v0_2_config,
    validate_json_artifact,
)

MILESTONE_LABEL = V0_2_4_MILESTONE
FREEZE_RECORD_COMMIT = "a6ae39651bb12fdd4e561d808a77b8e268c498a1"
FREEZE_RECORD_SHA256 = "ae552d805dd9648163a48683bad828c7e1b7ecc4f1d69f1fa28511363b08ce3b"
BOUNDARY_RECORD_SHA256 = "e122bfa51ce618e0588a580f2cf66447c44a2cf801f08f851cce9d5271a4c698"
REFERENCE_MANIFEST_SHA256 = "e587f1808262480261ae8a7b940faff0d9ef5f83cf215028b31490ba48369b99"
CALIBRATION_MANIFEST_SHA256 = "77d5adb588e7d463e7fcab1c10b841b9ad23d827b51138c31f42dac35bd99ca3"
MANIFEST_SET_SHA256 = "bb6b633d7c7645f767276f47ec74f709dd20afc83e0f444c4df3579418185e4b"
CLASSICAL_CONFIG_SHA256 = "9a8149c0d37f86a4474089ae251bd4954071d7171f7110073e82c9796b52d917"
DINO_SCORING_SOURCE_SHA256 = "944dc4c349aa011157ecc2e147bbcdd27da8b8c1fa1084fc0ae0e4641cec4cee"
ISOLATED_LOCK_SHA256 = "28a0878f3425bf6ea6fcc1631b168d67e6ddcc747d71f9c92c85b3aa9c706ec8"
REFERENCE_COUNT = NORMAL_REFERENCE_COUNT
CALIBRATION_COUNT = NORMAL_CALIBRATION_COUNT
SELECTION_NAMESPACE = "few-shot-anomaly-poc:v0.2:pcb2"
SELECTION_SEED = 42
EXPECTED_PUBLIC_INPUTS = {
    "boundary/boundary-record.json",
    "freeze/pre-evaluation-freeze.json",
}
EXPECTED_EXTERNAL_TOP_LEVEL = {
    "archive-identity.json",
    "boundary-state.json",
    "extraction.json",
    "normal-manifests",
    "scorer",
    "sealed",
    "source",
}
PICKLE_PROTOCOL = 5
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
type ProgressCallback = Callable[[str], None]


class V0_2NormalCalibrationError(Exception):
    """Reject changed inputs, leakage, overwrite, or incomplete stage execution."""


class DINOv2InputAdapterError(Exception):
    """Carry a stable normal-image adapter failure code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class NormalInputRecord:
    """Identify one verified normal-only source without loading a class label."""

    byte_count: int
    partition: str
    relative_path: str
    selection_rank: int
    selection_sha256: str
    sha256: str


@dataclass(frozen=True)
class V0_2CalibrationInputs:
    """Hold fixed normal-only inputs and contract objects."""

    boundary_state: dict[str, Any]
    calibration_records: tuple[NormalInputRecord, ...]
    classical_config: ProjectConfig
    config: dict[str, Any]
    reference_records: tuple[NormalInputRecord, ...]
    schema: dict[str, Any]


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V0_2NormalCalibrationError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise V0_2NormalCalibrationError(f"{label} must contain one JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V0_2NormalCalibrationError(message)


def _selection_sha256(relative_path: str) -> str:
    value = f"{SELECTION_NAMESPACE}:{SELECTION_SEED}:{relative_path}".encode()
    return hashlib.sha256(value).hexdigest()


def _safe_source_path(source_root: Path, relative_path: str) -> Path:
    pure_path = PurePosixPath(relative_path)
    if (
        pure_path.is_absolute()
        or pure_path.as_posix() != relative_path
        or any(part in {"", ".", ".."} for part in pure_path.parts)
        or not relative_path.startswith("pcb2/Data/Images/Normal/")
    ):
        raise V0_2NormalCalibrationError("normal manifest contains an unsafe source path")
    path = source_root.joinpath(*pure_path.parts)
    if (
        not source_root.is_dir()
        or source_root.is_symlink()
        or not path.is_file()
        or path.is_symlink()
    ):
        raise V0_2NormalCalibrationError("normal source must be a regular non-symlink file")
    current = path.parent
    while current != source_root:
        if current.is_symlink():
            raise V0_2NormalCalibrationError("normal source path contains a symlink")
        current = current.parent
    try:
        path.resolve().relative_to(source_root.resolve())
    except ValueError as error:
        raise V0_2NormalCalibrationError("normal source escapes the fixed source root") from error
    return path


def load_verified_normal_manifest(
    path: Path,
    *,
    partition: str,
    expected_count: int,
    first_rank: int,
    source_root: Path,
) -> tuple[NormalInputRecord, ...]:
    """Load one exact normal partition and verify every source byte identity."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise V0_2NormalCalibrationError("cannot read normal manifest") from error
    if len(lines) != expected_count or any(not line for line in lines):
        raise V0_2NormalCalibrationError("normal manifest count changed")

    records: list[NormalInputRecord] = []
    expected_keys = {
        "byte_count",
        "partition",
        "relative_path",
        "schema_version",
        "selection_rank",
        "selection_sha256",
        "sha256",
    }
    for offset, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise V0_2NormalCalibrationError("normal manifest JSON is invalid") from error
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise V0_2NormalCalibrationError("normal manifest fields changed")
        expected_rank = first_rank + offset
        relative_path = value.get("relative_path")
        if (
            value.get("schema_version") != NORMAL_MANIFEST_SCHEMA
            or value.get("partition") != partition
            or value.get("selection_rank") != expected_rank
            or not isinstance(relative_path, str)
            or value.get("selection_sha256") != _selection_sha256(relative_path)
            or not isinstance(value.get("byte_count"), int)
            or isinstance(value.get("byte_count"), bool)
            or value["byte_count"] <= 0
            or not isinstance(value.get("sha256"), str)
            or SHA256_PATTERN.fullmatch(value["sha256"]) is None
        ):
            raise V0_2NormalCalibrationError("normal manifest record changed")
        source_path = _safe_source_path(source_root, relative_path)
        if (
            source_path.stat().st_size != value["byte_count"]
            or sha256_file(source_path) != value["sha256"]
        ):
            raise V0_2NormalCalibrationError("normal source byte identity changed")
        records.append(
            NormalInputRecord(
                byte_count=value["byte_count"],
                partition=partition,
                relative_path=relative_path,
                selection_rank=expected_rank,
                selection_sha256=value["selection_sha256"],
                sha256=value["sha256"],
            )
        )
    paths = [record.relative_path for record in records]
    if len(paths) != len(set(paths)):
        raise V0_2NormalCalibrationError("normal manifest paths are not unique")
    return tuple(records)


def _validate_ancestor(project_root: Path, ancestor: str, descendant: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise V0_2NormalCalibrationError("execution source does not contain the freeze record")


def validate_v0_2_calibration_inputs(
    *,
    project_root: Path,
    execution_commit: str,
    external_root: Path,
    public_artifact_root: Path,
) -> V0_2CalibrationInputs:
    """Validate the frozen boundary without reading final-test assets or labels."""
    project_root = project_root.resolve()
    external_root = external_root.resolve()
    public_artifact_root = public_artifact_root.resolve()
    expected_external = (project_root / "data/external/v0.2/evaluation" / RUN_ID).resolve()
    expected_public = (project_root / "artifacts/v0.2/evaluation" / RUN_ID).resolve()
    if external_root != expected_external or public_artifact_root != expected_public:
        raise V0_2NormalCalibrationError("evaluation roots differ from the fixed contract")
    if (
        not external_root.is_dir()
        or external_root.is_symlink()
        or not public_artifact_root.is_dir()
        or public_artifact_root.is_symlink()
    ):
        raise V0_2NormalCalibrationError("evaluation roots are invalid")
    validate_boundary_execution_identity(
        project_root=project_root,
        execution_commit=execution_commit,
    )
    _validate_ancestor(project_root, FREEZE_RECORD_COMMIT, execution_commit)

    observed_public = {
        path.relative_to(public_artifact_root).as_posix()
        for path in public_artifact_root.rglob("*")
        if path.is_file()
    }
    _require(observed_public == EXPECTED_PUBLIC_INPUTS, "public evaluation stage already advanced")
    _require(
        {path.name for path in external_root.iterdir()} == EXPECTED_EXTERNAL_TOP_LEVEL,
        "external boundary inventory changed",
    )

    config = load_v0_2_config(project_root / "configs/v0.2.yaml")
    schema = load_v0_2_artifact_schema(project_root / "schemas/v0.2/evaluation-artifacts.json")
    boundary_path = public_artifact_root / "boundary/boundary-record.json"
    freeze_path = public_artifact_root / "freeze/pre-evaluation-freeze.json"
    _require(
        sha256_file(boundary_path) == BOUNDARY_RECORD_SHA256,
        "public boundary record changed",
    )
    _require(sha256_file(freeze_path) == FREEZE_RECORD_SHA256, "freeze record changed")
    boundary = validate_json_artifact(
        "boundary_record",
        _read_json(boundary_path, label="boundary record"),
        config=config,
        schema=schema,
    )
    freeze = validate_json_artifact(
        "pre_evaluation_freeze",
        _read_json(freeze_path, label="freeze record"),
        config=config,
        schema=schema,
    )
    _require(
        boundary["reference_count"] == REFERENCE_COUNT
        and boundary["calibration_count"] == CALIBRATION_COUNT
        and freeze["reference_manifest_sha256"] == REFERENCE_MANIFEST_SHA256
        and freeze["calibration_manifest_sha256"] == CALIBRATION_MANIFEST_SHA256
        and freeze["method_order"] == list(METHODS)
        and freeze["label_reveal_completed"] is False,
        "frozen normal-only identities changed",
    )

    state_path = external_root / "boundary-state.json"
    state = _read_json(state_path, label="external boundary state")
    _require(
        state.get("schema_version") == BOUNDARY_STATE_SCHEMA
        and state.get("run_id") == RUN_ID
        and state.get("contract", {}).get("config_sha256") == EXPECTED_CONFIG_SHA256
        and state.get("contract", {}).get("schema_sha256") == EXPECTED_SCHEMA_SHA256,
        "external boundary state identity changed",
    )
    _require(
        state.get("boundary")
        == {
            "anomaly_score_computed": False,
            "final_test_label_revealed": False,
            "image_content_decoded": False,
            "image_content_displayed": False,
            "method_fit_performed": False,
            "raw_data_in_git": False,
            "threshold_calibrated": False,
        },
        "evaluation advanced before normal-only fitting",
    )
    _require("normal_only_fit_calibration" not in state, "normal-only stage already recorded")

    manifest_root = external_root / "normal-manifests"
    reference_path = manifest_root / "reference.jsonl"
    calibration_path = manifest_root / "calibration.jsonl"
    manifest_set_path = manifest_root / "manifest-set.json"
    _require(
        sha256_file(reference_path) == REFERENCE_MANIFEST_SHA256
        and sha256_file(calibration_path) == CALIBRATION_MANIFEST_SHA256
        and sha256_file(manifest_set_path) == MANIFEST_SET_SHA256,
        "normal manifest identity changed",
    )
    source_root = external_root / "source"
    reference_records = load_verified_normal_manifest(
        reference_path,
        partition="reference",
        expected_count=REFERENCE_COUNT,
        first_rank=1,
        source_root=source_root,
    )
    calibration_records = load_verified_normal_manifest(
        calibration_path,
        partition="calibration",
        expected_count=CALIBRATION_COUNT,
        first_rank=REFERENCE_COUNT + 1,
        source_root=source_root,
    )
    _require(
        not (
            {record.relative_path for record in reference_records}
            & {record.relative_path for record in calibration_records}
        ),
        "normal partitions overlap",
    )

    classical_config_path = project_root / "configs/v0.1.yaml"
    dino_source_path = project_root / "src/few_shot_anomaly_poc/dinov2_scoring.py"
    isolated_lock_path = project_root / "environments/v0.2-preflight/uv.lock"
    _require(
        sha256_file(classical_config_path) == CLASSICAL_CONFIG_SHA256
        and sha256_file(dino_source_path) == DINO_SCORING_SOURCE_SHA256
        and sha256_file(isolated_lock_path) == ISOLATED_LOCK_SHA256,
        "method or dependency identity changed",
    )
    return V0_2CalibrationInputs(
        boundary_state=state,
        calibration_records=calibration_records,
        classical_config=load_config(classical_config_path),
        config=config,
        reference_records=reference_records,
        schema=schema,
    )


def adapt_dinov2_source_image(path: Path) -> NDArray[np.uint8]:
    """Apply the fixed OpenCV BGR-to-RGB 512-pixel DINOv2 adapter."""
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise DINOv2InputAdapterError(
            "DINO_IMAGE_READ_FAILED", "cannot read encoded normal image"
        ) from error
    if not encoded:
        raise DINOv2InputAdapterError("DINO_IMAGE_DECODE_FAILED", "normal image is empty")
    try:
        decoded = cv2.imdecode(
            np.frombuffer(encoded, dtype=np.uint8),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
    except cv2.error as error:
        raise DINOv2InputAdapterError(
            "DINO_IMAGE_DECODE_FAILED", "OpenCV could not decode normal image"
        ) from error
    if (
        not isinstance(decoded, np.ndarray)
        or decoded.ndim != 3
        or decoded.shape[2] != 3
        or decoded.dtype != np.uint8
        or decoded.size == 0
    ):
        raise DINOv2InputAdapterError(
            "DINO_IMAGE_DECODE_FAILED", "decoded normal image is outside the BGR boundary"
        )
    try:
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
    except cv2.error as error:
        raise DINOv2InputAdapterError(
            "DINO_INPUT_ADAPTER_FAILED", "OpenCV DINOv2 adapter failed"
        ) from error
    output = np.ascontiguousarray(resized, dtype=np.uint8)
    if output.shape != RGB_INPUT_SHAPE or output.dtype != np.uint8:
        raise DINOv2InputAdapterError(
            "DINO_INPUT_ADAPTER_FAILED", "DINOv2 adapter output is invalid"
        )
    return output


def _image_sha256(image: NDArray[np.uint8]) -> str:
    return hashlib.sha256(image.tobytes(order="C")).hexdigest()


def create_normal_rgb_input_store(
    *,
    store_path: Path,
    manifest_path: Path,
    source_root: Path,
    reference_records: Sequence[NormalInputRecord],
    calibration_records: Sequence[NormalInputRecord],
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Write one memory-bounded, ignored RGB store for the isolated worker."""
    if store_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite normal RGB input store")
    records = (*reference_records, *calibration_records)
    if len(records) != TOTAL_NORMAL_COUNT:
        raise V0_2NormalCalibrationError("normal RGB store record count changed")
    store_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = store_path.with_name(f".{store_path.name}.partial")
    if partial_path.exists():
        raise FileExistsError(f"refusing to overwrite {partial_path}")
    manifest_records: list[dict[str, Any]] = []
    try:
        store = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.uint8,
            shape=(TOTAL_NORMAL_COUNT, *RGB_INPUT_SHAPE),
        )
        for index, record in enumerate(records):
            source_path = _safe_source_path(source_root, record.relative_path)
            try:
                image = adapt_dinov2_source_image(source_path)
            except DINOv2InputAdapterError as error:
                store[index].fill(0)
                adapter_status = "failed"
                failure_code = error.code
                image_sha256 = None
            else:
                store[index] = image
                adapter_status = "ok"
                failure_code = None
                image_sha256 = _image_sha256(image)
                del image
            manifest_records.append(
                {
                    "adapter_failure_code": failure_code,
                    "adapter_status": adapter_status,
                    "index": index,
                    "partition": record.partition,
                    "selection_rank": record.selection_rank,
                    "source_path": record.relative_path,
                    "source_sha256": record.sha256,
                    "rgb_sha256": image_sha256,
                }
            )
            if (index + 1) % 100 == 0 or index + 1 == len(records):
                _emit(progress, f"DINOv2 normal RGB adapter: {index + 1}/{len(records)}")
        store.flush()
        del store
        os.replace(partial_path, store_path)
        manifest = {
            "schema_version": RGB_STORE_SCHEMA,
            "run_id": RUN_ID,
            "shape": [TOTAL_NORMAL_COUNT, *RGB_INPUT_SHAPE],
            "dtype": "uint8",
            "reference_count": REFERENCE_COUNT,
            "calibration_count": CALIBRATION_COUNT,
            "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
            "calibration_manifest_sha256": CALIBRATION_MANIFEST_SHA256,
            "store_byte_count": store_path.stat().st_size,
            "store_sha256": sha256_file(store_path),
            "labels_accessed": False,
            "final_test_accessed": False,
            "records": manifest_records,
        }
        write_json_atomic(manifest_path, manifest)
        return manifest
    except Exception:
        partial_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        store_path.unlink(missing_ok=True)
        raise


def _write_pickle_state(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            pickle.dump(value, stream, protocol=PICKLE_PROTOCOL)
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return sha256_file(path)


def _failed_fit(
    *,
    method: str,
    successful: int,
    failure_code: str,
    inputs: V0_2CalibrationInputs,
) -> dict[str, Any]:
    return build_fit_record(
        run_id=RUN_ID,
        method=method,
        status="fit_failed",
        successful_reference_count=successful,
        failed_reference_count=REFERENCE_COUNT - successful,
        reference_manifest_sha256=REFERENCE_MANIFEST_SHA256,
        fitted_state_sha256=None,
        failure_code=failure_code,
        config=inputs.config,
        schema=inputs.schema,
    )


def _successful_fit(
    *,
    method: str,
    successful: int,
    failed: int,
    state_sha256: str,
    inputs: V0_2CalibrationInputs,
) -> dict[str, Any]:
    return build_fit_record(
        run_id=RUN_ID,
        method=method,
        status="fit_ok",
        successful_reference_count=successful,
        failed_reference_count=failed,
        reference_manifest_sha256=REFERENCE_MANIFEST_SHA256,
        fitted_state_sha256=state_sha256,
        failure_code=None,
        config=inputs.config,
        schema=inputs.schema,
    )


def _fit_classical_methods(
    *,
    project_root: Path,
    external_root: Path,
    execution_commit: str,
    inputs: V0_2CalibrationInputs,
    state_root: Path,
    progress: ProgressCallback | None,
) -> tuple[
    dict[str, Any],
    ECCTemplateFitResult | None,
    dict[str, Any],
    PatchHOGScalerFitResult | None,
    PatchHOGModelFitResult | None,
]:
    source_root = external_root / "source"
    references: dict[str, NDArray[np.float32]] = {}
    for record in inputs.reference_records:
        try:
            references[record.relative_path] = load_and_preprocess_image(
                source_root / record.relative_path,
                inputs.classical_config.preprocessing,
            )
        except ImagePreprocessingError:
            continue
    _emit(progress, f"classical reference preprocessing: {len(references)}/{REFERENCE_COUNT}")
    if len(references) != REFERENCE_COUNT:
        successful = len(references)
        return (
            _failed_fit(
                method="ecc_residual",
                successful=successful,
                failure_code="REFERENCE_PREPROCESSING_FAILED",
                inputs=inputs,
            ),
            None,
            _failed_fit(
                method="patch_hog_ocsvm",
                successful=successful,
                failure_code="REFERENCE_PREPROCESSING_FAILED",
                inputs=inputs,
            ),
            None,
            None,
        )

    try:
        ecc_fit = fit_ecc_normal_template(references, config=inputs.classical_config)
    except Exception:
        ecc_fit = None
        ecc_record = _failed_fit(
            method="ecc_residual",
            successful=0,
            failure_code="ECC_FIT_EXECUTION_FAILED",
            inputs=inputs,
        )
    else:
        if ecc_fit.succeeded:
            ecc_state_sha256 = _write_pickle_state(
                state_root / "ecc_residual.pkl",
                {
                    "milestone": MILESTONE_LABEL,
                    "run_id": RUN_ID,
                    "method": "ecc_residual",
                    "execution_commit": execution_commit,
                    "freeze_record_sha256": FREEZE_RECORD_SHA256,
                    "v0_2_config_sha256": EXPECTED_CONFIG_SHA256,
                    "config_sha256": CLASSICAL_CONFIG_SHA256,
                    "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
                    "fit": ecc_fit,
                },
            )
            ecc_record = _successful_fit(
                method="ecc_residual",
                successful=ecc_fit.successful_reference_count,
                failed=ecc_fit.failed_reference_count,
                state_sha256=ecc_state_sha256,
                inputs=inputs,
            )
        else:
            ecc_record = _failed_fit(
                method="ecc_residual",
                successful=ecc_fit.successful_reference_count,
                failure_code=str(ecc_fit.failure_code or "ECC_FIT_FAILED"),
                inputs=inputs,
            )
            ecc_fit = None

    reference_features: dict[str, PatchHOGFeatureResult] = {}
    feature_failure_code: str | None = None
    for relative_path in sorted(references):
        try:
            result = extract_patch_hog_features(
                references[relative_path],
                config=inputs.classical_config,
            )
        except Exception:
            feature_failure_code = "PATCH_HOG_FEATURE_EXECUTION_FAILED"
            break
        if not result.succeeded:
            feature_failure_code = str(result.failure_code or "PATCH_HOG_FEATURE_FAILED")
            break
        reference_features[relative_path] = result
    scaler_fit: PatchHOGScalerFitResult | None = None
    model_fit: PatchHOGModelFitResult | None = None
    if feature_failure_code is not None or len(reference_features) != REFERENCE_COUNT:
        hog_record = _failed_fit(
            method="patch_hog_ocsvm",
            successful=len(reference_features),
            failure_code=feature_failure_code or "PATCH_HOG_FEATURE_FAILED",
            inputs=inputs,
        )
    else:
        try:
            scaler_fit = fit_position_scalers(
                reference_features,
                config=inputs.classical_config,
            )
            if scaler_fit.succeeded:
                model_fit = fit_position_one_class_svms(
                    reference_features,
                    scaler_fit=scaler_fit,
                    config=inputs.classical_config,
                )
        except Exception:
            scaler_fit = None
            model_fit = None
            hog_record = _failed_fit(
                method="patch_hog_ocsvm",
                successful=0,
                failure_code="PATCH_HOG_FIT_EXECUTION_FAILED",
                inputs=inputs,
            )
        else:
            if scaler_fit.succeeded and model_fit is not None and model_fit.succeeded:
                hog_state_sha256 = _write_pickle_state(
                    state_root / "patch_hog_ocsvm.pkl",
                    {
                        "milestone": MILESTONE_LABEL,
                        "run_id": RUN_ID,
                        "method": "patch_hog_ocsvm",
                        "execution_commit": execution_commit,
                        "freeze_record_sha256": FREEZE_RECORD_SHA256,
                        "v0_2_config_sha256": EXPECTED_CONFIG_SHA256,
                        "config_sha256": CLASSICAL_CONFIG_SHA256,
                        "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
                        "scaler_fit": scaler_fit,
                        "model_fit": model_fit,
                    },
                )
                hog_record = _successful_fit(
                    method="patch_hog_ocsvm",
                    successful=REFERENCE_COUNT,
                    failed=0,
                    state_sha256=hog_state_sha256,
                    inputs=inputs,
                )
            else:
                if not scaler_fit.succeeded:
                    failure_code = str(scaler_fit.failure_code or "PATCH_HOG_SCALER_FIT_FAILED")
                elif model_fit is None:
                    failure_code = "PATCH_HOG_MODEL_FIT_FAILED"
                else:
                    failure_code = str(model_fit.failure_code or "PATCH_HOG_MODEL_FIT_FAILED")
                hog_record = _failed_fit(
                    method="patch_hog_ocsvm",
                    successful=0,
                    failure_code=failure_code or "PATCH_HOG_FIT_FAILED",
                    inputs=inputs,
                )
                scaler_fit = None
                model_fit = None
    del references
    del reference_features
    gc.collect()
    return ecc_record, ecc_fit, hog_record, scaler_fit, model_fit


def _classical_calibration_scores(
    *,
    external_root: Path,
    inputs: V0_2CalibrationInputs,
    ecc_fit: ECCTemplateFitResult | None,
    scaler_fit: PatchHOGScalerFitResult | None,
    model_fit: PatchHOGModelFitResult | None,
    progress: ProgressCallback | None,
) -> tuple[list[CalibrationScore], list[CalibrationScore]]:
    source_root = external_root / "source"
    ecc_scores: list[CalibrationScore] = []
    hog_scores: list[CalibrationScore] = []
    for index, record in enumerate(inputs.calibration_records, start=1):
        try:
            image = load_and_preprocess_image(
                source_root / record.relative_path,
                inputs.classical_config.preprocessing,
            )
        except ImagePreprocessingError as error:
            if ecc_fit is not None:
                ecc_scores.append(
                    CalibrationScore(
                        source_path=record.relative_path,
                        score_status="failed",
                        score_failure_code=str(error.code),
                        anomaly_score=inputs.config["methods"]["ecc_residual"]["failure_score"],
                    )
                )
            if scaler_fit is not None and model_fit is not None:
                hog_scores.append(
                    CalibrationScore(
                        source_path=record.relative_path,
                        score_status="failed",
                        score_failure_code=str(error.code),
                        anomaly_score=inputs.config["methods"]["patch_hog_ocsvm"]["failure_score"],
                    )
                )
        else:
            if ecc_fit is not None:
                try:
                    result = score_ecc_residual(
                        image,
                        fitted=ecc_fit,
                        config=inputs.classical_config,
                    )
                    ecc_scores.append(
                        CalibrationScore(
                            source_path=record.relative_path,
                            score_status=result.score_status,
                            score_failure_code=(
                                None if result.failure_code is None else str(result.failure_code)
                            ),
                            anomaly_score=float(result.anomaly_score),
                        )
                    )
                except Exception:
                    ecc_scores.append(
                        CalibrationScore(
                            source_path=record.relative_path,
                            score_status="failed",
                            score_failure_code="ECC_SCORE_EXECUTION_FAILED",
                            anomaly_score=inputs.config["methods"]["ecc_residual"]["failure_score"],
                        )
                    )
            if scaler_fit is not None and model_fit is not None:
                try:
                    result = score_patch_hog(
                        image,
                        scaler_fit=scaler_fit,
                        model_fit=model_fit,
                        config=inputs.classical_config,
                    )
                    hog_scores.append(
                        CalibrationScore(
                            source_path=record.relative_path,
                            score_status=result.score_status,
                            score_failure_code=(
                                None if result.failure_code is None else str(result.failure_code)
                            ),
                            anomaly_score=float(result.anomaly_score),
                        )
                    )
                except Exception:
                    hog_scores.append(
                        CalibrationScore(
                            source_path=record.relative_path,
                            score_status="failed",
                            score_failure_code="PATCH_HOG_SCORE_EXECUTION_FAILED",
                            anomaly_score=inputs.config["methods"]["patch_hog_ocsvm"][
                                "failure_score"
                            ],
                        )
                    )
            del image
        if index % 50 == 0 or index == CALIBRATION_COUNT:
            _emit(progress, f"classical normal-only scoring: {index}/{CALIBRATION_COUNT}")
    return ecc_scores, hog_scores


def run_classical_fit_and_calibration(
    *,
    project_root: Path,
    external_root: Path,
    execution_commit: str,
    inputs: V0_2CalibrationInputs,
    public_stage_root: Path,
    state_stage_root: Path,
    progress: ProgressCallback | None = None,
) -> dict[str, str]:
    """Run the two frozen classical methods and write staged evidence."""
    ecc_record, ecc_fit, hog_record, scaler_fit, model_fit = _fit_classical_methods(
        project_root=project_root,
        external_root=external_root,
        execution_commit=execution_commit,
        inputs=inputs,
        state_root=state_stage_root,
        progress=progress,
    )
    ecc_scores, hog_scores = _classical_calibration_scores(
        external_root=external_root,
        inputs=inputs,
        ecc_fit=ecc_fit,
        scaler_fit=scaler_fit,
        model_fit=model_fit,
        progress=progress,
    )
    calibrations: dict[str, CalibrationResult | None] = {
        "ecc_residual": (
            calibrate_normal_scores(
                ecc_scores,
                run_id=RUN_ID,
                method="ecc_residual",
                config=inputs.config,
                schema=inputs.schema,
            )
            if ecc_record["status"] == "fit_ok"
            else None
        ),
        "patch_hog_ocsvm": (
            calibrate_normal_scores(
                hog_scores,
                run_id=RUN_ID,
                method="patch_hog_ocsvm",
                config=inputs.config,
                schema=inputs.schema,
            )
            if hog_record["status"] == "fit_ok"
            else None
        ),
    }
    for method, fit_record in (
        ("ecc_residual", ecc_record),
        ("patch_hog_ocsvm", hog_record),
    ):
        write_method_artifacts(
            method_root=public_stage_root / method,
            fit_record=fit_record,
            calibration=calibrations[method],
        )
        _emit(progress, f"{method} stage complete: {fit_record['status']}")
    return {
        "ecc_residual": ecc_record["status"],
        "patch_hog_ocsvm": hog_record["status"],
    }


def _isolated_worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "MKL_NUM_THREADS": "4",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "42",
            "PYTHONNOUSERSITE": "1",
            "XFORMERS_DISABLED": "1",
        }
    )
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    return environment


def run_dinov2_worker(
    *,
    project_root: Path,
    execution_commit: str,
    input_store_path: Path,
    input_manifest_path: Path,
    public_stage_root: Path,
    state_stage_root: Path,
    progress: ProgressCallback | None,
) -> None:
    """Run DINOv2 in the exact isolated environment and preserve its output."""
    artifact_dir = project_root / "data/external/v0.2/model-assets"
    source_root = artifact_dir / f"dinov2-source-sha256-{EXPECTED_SOURCE_SHA256}" / SOURCE_ROOT
    command = [
        str(project_root / "environments/v0.2-preflight/.venv/bin/python"),
        "-I",
        "-B",
        str(project_root / "scripts/run_v0_2_4_dinov2_calibration_worker.py"),
        "--project-root",
        str(project_root),
        "--execution-commit",
        execution_commit,
        "--input-store",
        str(input_store_path),
        "--input-manifest",
        str(input_manifest_path),
        "--artifact-dir",
        str(artifact_dir),
        "--source-root",
        str(source_root),
        "--environment-root",
        str(project_root / "environments/v0.2-preflight/.venv"),
        "--output-dir",
        str(public_stage_root / "dinov2_vits14_224_nn"),
        "--state-path",
        str(state_stage_root / "dinov2_vits14_224_nn.pt"),
    ]
    _emit(progress, "starting isolated DINOv2 normal-only worker")
    completed = subprocess.run(
        command,
        cwd=project_root,
        check=False,
        env=_isolated_worker_environment(),
        text=True,
    )
    if completed.returncode != 0:
        raise V0_2NormalCalibrationError(
            f"isolated DINOv2 worker failed with exit code {completed.returncode}"
        )
    _emit(progress, "isolated DINOv2 normal-only worker complete")


def _load_fit_status(method_root: Path, *, inputs: V0_2CalibrationInputs) -> str:
    fit = validate_json_artifact(
        "fit",
        _read_json(method_root / "fit.json", label="fit artifact"),
        config=inputs.config,
        schema=inputs.schema,
    )
    expected_files = {"fit.json"}
    if fit["status"] == "fit_ok":
        expected_files.update({"calibration-scores.csv", "calibration-summary.json"})
    observed_files = {path.name for path in method_root.iterdir() if path.is_file()}
    _require(observed_files == expected_files, "method stage artifact inventory is invalid")
    return fit["status"]


def _validate_staged_state_identities(
    *,
    public_stage_root: Path,
    state_stage_root: Path,
    inputs: V0_2CalibrationInputs,
) -> None:
    suffixes = {
        "ecc_residual": ".pkl",
        "patch_hog_ocsvm": ".pkl",
        "dinov2_vits14_224_nn": ".pt",
    }
    expected_state_files: set[str] = set()
    for method in METHODS:
        fit = validate_json_artifact(
            "fit",
            _read_json(public_stage_root / method / "fit.json", label="fit artifact"),
            config=inputs.config,
            schema=inputs.schema,
        )
        state_name = f"{method}{suffixes[method]}"
        state_path = state_stage_root / state_name
        if fit["status"] == "fit_ok":
            _require(
                state_path.is_file() and sha256_file(state_path) == fit["fitted_state_sha256"],
                "fitted-state identity differs from the fit artifact",
            )
            expected_state_files.add(state_name)
        else:
            _require(not state_path.exists(), "failed fit retained a fitted state")
    observed_state_files = {path.name for path in state_stage_root.iterdir() if path.is_file()}
    _require(
        observed_state_files == expected_state_files,
        "fitted-state inventory is invalid",
    )


def _update_external_stage_state(
    *,
    state_path: Path,
    original_state: Mapping[str, Any],
    execution_commit: str,
    public_artifact_root: Path,
    method_statuses: Mapping[str, str],
) -> None:
    state = dict(original_state)
    boundary = dict(state["boundary"])
    boundary.update(
        {
            "image_content_decoded": True,
            "method_fit_performed": True,
            "threshold_calibrated": all(status == "fit_ok" for status in method_statuses.values()),
        }
    )
    identities = []
    for method in METHODS:
        method_root = public_artifact_root / method
        for path in sorted(item for item in method_root.iterdir() if item.is_file()):
            identities.append(
                {
                    "relative_path": path.relative_to(public_artifact_root).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
    state["boundary"] = boundary
    state["normal_only_fit_calibration"] = {
        "milestone": MILESTONE_LABEL,
        "execution_commit": execution_commit,
        "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "calibration_manifest_sha256": CALIBRATION_MANIFEST_SHA256,
        "method_statuses": dict(method_statuses),
        "artifact_identities": identities,
        "anomaly_labels_used": False,
        "final_test_accessed": False,
        "sealed_mapping_accessed": False,
    }
    write_json_atomic(state_path, state, overwrite=True)


def run_v0_2_normal_fit_and_calibration(
    *,
    project_root: Path,
    execution_commit: str,
    external_root: Path,
    public_artifact_root: Path,
    work_root: Path,
    progress: ProgressCallback | None = None,
) -> dict[str, str]:
    """Execute the fixed v0.2.4 stage once without final-test access."""
    project_root = project_root.resolve()
    external_root = external_root.resolve()
    public_artifact_root = public_artifact_root.resolve()
    work_root = work_root.resolve()
    expected_work_root = (project_root / "work/v0.2/evaluation" / RUN_ID).resolve()
    if work_root != expected_work_root:
        raise V0_2NormalCalibrationError("work_root differs from the fixed local path")
    final_state_root = work_root / "fitted-state"
    if final_state_root.exists() or any(
        (public_artifact_root / method).exists() for method in METHODS
    ):
        raise FileExistsError("refusing to overwrite v0.2.4 outputs")
    inputs = validate_v0_2_calibration_inputs(
        project_root=project_root,
        execution_commit=execution_commit,
        external_root=external_root,
        public_artifact_root=public_artifact_root,
    )
    _emit(progress, "fixed normal manifests and every normal source byte matched")
    work_root.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(dir=work_root, prefix=".v0.2.4-stage-"))
    public_stage_root = stage_root / "public"
    state_stage_root = stage_root / "fitted-state"
    input_store_path = stage_root / "normal-rgb512.npy"
    input_manifest_path = stage_root / "normal-rgb512-manifest.json"
    try:
        public_stage_root.mkdir()
        state_stage_root.mkdir()
        create_normal_rgb_input_store(
            store_path=input_store_path,
            manifest_path=input_manifest_path,
            source_root=external_root / "source",
            reference_records=inputs.reference_records,
            calibration_records=inputs.calibration_records,
            progress=progress,
        )
        method_statuses = run_classical_fit_and_calibration(
            project_root=project_root,
            external_root=external_root,
            execution_commit=execution_commit,
            inputs=inputs,
            public_stage_root=public_stage_root,
            state_stage_root=state_stage_root,
            progress=progress,
        )
        run_dinov2_worker(
            project_root=project_root,
            execution_commit=execution_commit,
            input_store_path=input_store_path,
            input_manifest_path=input_manifest_path,
            public_stage_root=public_stage_root,
            state_stage_root=state_stage_root,
            progress=progress,
        )
        method_statuses["dinov2_vits14_224_nn"] = _load_fit_status(
            public_stage_root / "dinov2_vits14_224_nn",
            inputs=inputs,
        )
        _require(set(method_statuses) == set(METHODS), "method stage is incomplete")
        for method in METHODS:
            _load_fit_status(public_stage_root / method, inputs=inputs)
        _validate_staged_state_identities(
            public_stage_root=public_stage_root,
            state_stage_root=state_stage_root,
            inputs=inputs,
        )

        input_store_path.unlink()
        input_manifest_path.unlink()
        state_moved = False
        moved_methods: list[str] = []
        try:
            os.replace(state_stage_root, final_state_root)
            state_moved = True
            for method in METHODS:
                os.replace(public_stage_root / method, public_artifact_root / method)
                moved_methods.append(method)
            _update_external_stage_state(
                state_path=external_root / "boundary-state.json",
                original_state=inputs.boundary_state,
                execution_commit=execution_commit,
                public_artifact_root=public_artifact_root,
                method_statuses=method_statuses,
            )
        except Exception:
            for method in reversed(moved_methods):
                os.replace(public_artifact_root / method, public_stage_root / method)
            if state_moved:
                os.replace(final_state_root, state_stage_root)
            raise
        shutil.rmtree(stage_root, ignore_errors=True)
        return method_statuses
    except Exception:
        if stage_root.exists():
            shutil.rmtree(stage_root)
        raise
