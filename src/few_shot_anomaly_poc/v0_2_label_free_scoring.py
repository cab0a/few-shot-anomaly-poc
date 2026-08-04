"""Execute v0.2.5 label-free final-test scoring and CPU latency measurement."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from few_shot_anomaly_poc.config import ProjectConfig, load_config
from few_shot_anomaly_poc.cpu_latency import _capture_cpu_environment
from few_shot_anomaly_poc.dinov2_timing_preflight import (
    capture_target_machine,
    evaluate_target_machine,
)
from few_shot_anomaly_poc.ecc_residual import ECCResidualScoreResult, score_ecc_residual
from few_shot_anomaly_poc.errors import ImagePreprocessingError
from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.hog_scoring import PatchHOGScoreResult, score_patch_hog
from few_shot_anomaly_poc.jsonio import write_json_atomic
from few_shot_anomaly_poc.model_assets import SOURCE_ROOT
from few_shot_anomaly_poc.model_compatibility import EXPECTED_SOURCE_SHA256
from few_shot_anomaly_poc.preprocessing import DECODE_FLAGS, preprocess_decoded_image
from few_shot_anomaly_poc.v0_2_boundary_preparation import (
    BOUNDARY_STATE_SCHEMA,
    RUN_ID,
    validate_boundary_execution_identity,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    EXPECTED_CONFIG_SHA256,
    EXPECTED_SCHEMA_SHA256,
    METHODS,
    load_v0_2_artifact_schema,
    load_v0_2_config,
    validate_json_artifact,
)
from few_shot_anomaly_poc.v0_2_normal_calibration import (
    CLASSICAL_CONFIG_SHA256,
    DINO_SCORING_SOURCE_SHA256,
    FREEZE_RECORD_COMMIT,
    FREEZE_RECORD_SHA256,
    ISOLATED_LOCK_SHA256,
    REFERENCE_MANIFEST_SHA256,
    _isolated_worker_environment,
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

MILESTONE = "v0.2.5"
OPAQUE_MANIFEST_SCHEMA = "v0.2-opaque-scoring-manifest-v1"
OPAQUE_SCORING_MANIFEST_SHA256 = "32ea52ed1b9872f39ae27f5d58a353ea84b8b143642e3a7f0fabe940184705e8"
CALIBRATION_EXECUTION_COMMIT = "6548bfa97e7e834bd88b0efbbd1b557ae85e242c"
CALIBRATION_RECORD_COMMIT = "4ed4595ff185c09e5e8c965b8d68081d4c00ae02"
DINO_STORE_SCHEMA = "v0.2.5-dinov2-label-free-rgb512-store-v1"
DINO_INPUT_SHAPE = (512, 512, 3)
DINO_STORE_SHAPE = (ASSET_COUNT, *DINO_INPUT_SHAPE)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
EXPECTED_EXTERNAL_TOP_LEVEL = {
    "archive-identity.json",
    "boundary-state.json",
    "extraction.json",
    "normal-manifests",
    "scorer",
    "sealed",
    "source",
}
EXPECTED_PUBLIC_HASHES = {
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
STATE_HASHES = {
    "ecc_residual": "f796dcef8fb7b6197f656c2a57800766c0398ac4842fd65f85372781063b800b",
    "patch_hog_ocsvm": "ba7e1d47e8ff6fd7873edfce84c027c6ada37f6aab85d1b26cf92426463056f9",
    "dinov2_vits14_224_nn": "11ac0a0a4b0c082e2450fcf708e1c96a804ec738967aff392b727959ec425f8d",
}
type ProgressCallback = Callable[[str], None]


class V0_2LabelFreeScoringError(Exception):
    """Reject changed inputs, leakage, overwrite, or incomplete v0.2.5 evidence."""


@dataclass(frozen=True)
class OpaqueAsset:
    """Identify one byte-verified final-test asset without semantic metadata."""

    asset_id: str
    byte_count: int
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class V0_2ScoringInputs:
    """Hold fixed label-free inputs and previously calibrated method state."""

    assets: tuple[OpaqueAsset, ...]
    boundary_state: dict[str, Any]
    classical_config: ProjectConfig
    config: dict[str, Any]
    schema: dict[str, Any]
    thresholds: dict[str, float]


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V0_2LabelFreeScoringError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V0_2LabelFreeScoringError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise V0_2LabelFreeScoringError(f"{label} must contain one JSON object")
    return value


def _validate_ancestor(project_root: Path, ancestor: str, descendant: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise V0_2LabelFreeScoringError("execution source does not contain the fixed prior stage")


def _validate_public_inputs(
    *,
    public_root: Path,
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, float]:
    observed = {
        path.relative_to(public_root).as_posix()
        for path in public_root.rglob("*")
        if path.is_file()
    }
    _require(observed == set(EXPECTED_PUBLIC_HASHES), "public evaluation inventory changed")
    for relative_path, expected_hash in EXPECTED_PUBLIC_HASHES.items():
        _require(
            sha256_file(public_root / relative_path) == expected_hash,
            f"public input hash changed: {relative_path}",
        )

    thresholds: dict[str, float] = {}
    for method in METHODS:
        fit = validate_json_artifact(
            "fit",
            _read_json(public_root / method / "fit.json", label=f"{method} fit"),
            config=config,
            schema=schema,
        )
        summary = validate_json_artifact(
            "calibration_summary",
            _read_json(
                public_root / method / "calibration-summary.json",
                label=f"{method} calibration summary",
            ),
            config=config,
            schema=schema,
        )
        _require(
            fit["status"] == "fit_ok"
            and fit["fitted_state_sha256"] == STATE_HASHES[method]
            and summary["sample_count"] == 881
            and summary["score_failure_count"] == 0,
            f"{method} is not a fixed successful normal-only calibration",
        )
        thresholds[method] = float(summary["threshold"])
    return thresholds


def _validate_opaque_assets(scorer_root: Path) -> tuple[OpaqueAsset, ...]:
    manifest_path = scorer_root / "scoring-manifest.json"
    _require(
        scorer_root.is_dir()
        and not scorer_root.is_symlink()
        and manifest_path.is_file()
        and not manifest_path.is_symlink()
        and sha256_file(manifest_path) == OPAQUE_SCORING_MANIFEST_SHA256,
        "opaque scoring manifest identity changed",
    )
    manifest = _read_json(manifest_path, label="opaque scoring manifest")
    records = manifest.get("records")
    _require(
        set(manifest) == {"records", "schema_version"}
        and manifest.get("schema_version") == OPAQUE_MANIFEST_SCHEMA
        and isinstance(records, list)
        and len(records) == ASSET_COUNT,
        "opaque scoring manifest contract changed",
    )
    assets_root = scorer_root / "assets"
    _require(assets_root.is_dir() and not assets_root.is_symlink(), "opaque asset root is invalid")
    assets: list[OpaqueAsset] = []
    for index, record in enumerate(records):
        expected_id = f"asset-{index:06d}"
        expected_relative_path = f"assets/{expected_id}.jpg"
        if (
            not isinstance(record, dict)
            or set(record) != {"asset_id", "byte_count", "relative_path", "sha256"}
            or record.get("asset_id") != expected_id
            or record.get("relative_path") != expected_relative_path
            or not isinstance(record.get("byte_count"), int)
            or isinstance(record.get("byte_count"), bool)
            or record["byte_count"] < 1
            or not isinstance(record.get("sha256"), str)
            or SHA256_PATTERN.fullmatch(record["sha256"]) is None
        ):
            raise V0_2LabelFreeScoringError("opaque scoring manifest record changed")
        relative_path = PurePosixPath(expected_relative_path)
        path = scorer_root.joinpath(*relative_path.parts)
        _require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == record["byte_count"]
            and sha256_file(path) == record["sha256"],
            "opaque asset byte identity changed",
        )
        assets.append(
            OpaqueAsset(
                asset_id=expected_id,
                byte_count=record["byte_count"],
                relative_path=expected_relative_path,
                sha256=record["sha256"],
            )
        )
    expected_files = {f"{asset.asset_id}.jpg" for asset in assets}
    observed_files = {path.name for path in assets_root.iterdir() if path.is_file()}
    _require(
        observed_files == expected_files
        and all(not path.is_symlink() for path in assets_root.iterdir()),
        "opaque asset inventory changed",
    )
    _require(
        {path.name for path in scorer_root.iterdir()} == {"assets", "scoring-manifest.json"},
        "opaque scorer inventory changed",
    )
    return tuple(assets)


def _validate_external_state(state: Mapping[str, Any]) -> None:
    expected_boundary = {
        "anomaly_score_computed": False,
        "final_test_label_revealed": False,
        "image_content_decoded": True,
        "image_content_displayed": False,
        "method_fit_performed": True,
        "raw_data_in_git": False,
        "threshold_calibrated": True,
    }
    normal = state.get("normal_only_fit_calibration")
    _require(
        state.get("schema_version") == BOUNDARY_STATE_SCHEMA
        and state.get("run_id") == RUN_ID
        and state.get("contract", {}).get("config_sha256") == EXPECTED_CONFIG_SHA256
        and state.get("contract", {}).get("schema_sha256") == EXPECTED_SCHEMA_SHA256
        and state.get("boundary") == expected_boundary
        and isinstance(normal, dict)
        and normal.get("milestone") == "v0.2.4"
        and normal.get("execution_commit") == CALIBRATION_EXECUTION_COMMIT
        and normal.get("method_statuses") == {method: "fit_ok" for method in METHODS}
        and normal.get("anomaly_labels_used") is False
        and normal.get("final_test_accessed") is False
        and normal.get("sealed_mapping_accessed") is False
        and "label_free_scoring" not in state,
        "external boundary state is not the fixed post-v0.2.4 state",
    )


def validate_v0_2_scoring_inputs(
    *,
    project_root: Path,
    execution_commit: str,
    external_root: Path,
    public_root: Path,
    work_root: Path,
) -> V0_2ScoringInputs:
    """Validate the frozen scorer boundary without opening labels or semantic mappings."""
    project_root = project_root.resolve()
    external_root = external_root.resolve()
    public_root = public_root.resolve()
    work_root = work_root.resolve()
    _require(
        external_root == (project_root / "data/external/v0.2/evaluation" / RUN_ID).resolve()
        and public_root == (project_root / "artifacts/v0.2/evaluation" / RUN_ID).resolve()
        and work_root == (project_root / "work/v0.2/evaluation" / RUN_ID).resolve(),
        "evaluation roots differ from the fixed contract",
    )
    validate_boundary_execution_identity(
        project_root=project_root,
        execution_commit=execution_commit,
    )
    _validate_ancestor(project_root, FREEZE_RECORD_COMMIT, execution_commit)
    _validate_ancestor(project_root, CALIBRATION_RECORD_COMMIT, execution_commit)
    _require(
        external_root.is_dir()
        and not external_root.is_symlink()
        and public_root.is_dir()
        and not public_root.is_symlink()
        and {path.name for path in external_root.iterdir()} == EXPECTED_EXTERNAL_TOP_LEVEL,
        "evaluation boundary inventory changed",
    )

    config = load_v0_2_config(project_root / "configs/v0.2.yaml")
    schema = load_v0_2_artifact_schema(project_root / "schemas/v0.2/evaluation-artifacts.json")
    _require(
        sha256_file(project_root / "configs/v0.1.yaml") == CLASSICAL_CONFIG_SHA256
        and sha256_file(project_root / "src/few_shot_anomaly_poc/dinov2_scoring.py")
        == DINO_SCORING_SOURCE_SHA256
        and sha256_file(project_root / "environments/v0.2-preflight/uv.lock")
        == ISOLATED_LOCK_SHA256,
        "method or dependency identity changed",
    )
    state = _read_json(external_root / "boundary-state.json", label="external boundary state")
    _validate_external_state(state)
    thresholds = _validate_public_inputs(public_root=public_root, config=config, schema=schema)
    assets = _validate_opaque_assets(external_root / "scorer")

    state_root = work_root / "fitted-state"
    suffixes = {
        "ecc_residual": ".pkl",
        "patch_hog_ocsvm": ".pkl",
        "dinov2_vits14_224_nn": ".pt",
    }
    _require(state_root.is_dir() and not state_root.is_symlink(), "fitted state is unavailable")
    for method, expected_hash in STATE_HASHES.items():
        path = state_root / f"{method}{suffixes[method]}"
        _require(
            path.is_file() and not path.is_symlink() and sha256_file(path) == expected_hash,
            f"{method} fitted-state identity changed",
        )
    _require(
        {path.name for path in state_root.iterdir()}
        == {f"{method}{suffixes[method]}" for method in METHODS},
        "fitted-state inventory changed",
    )
    for method in METHODS:
        for name in ("scores.csv", "classifications.csv", "latency-observations.csv"):
            _require(
                not (public_root / method / name).exists(), "v0.2.5 public output already exists"
            )
    _require(
        not (work_root / "label-free-scoring-state.json").exists(),
        "v0.2.5 local state already exists",
    )
    return V0_2ScoringInputs(
        assets=assets,
        boundary_state=state,
        classical_config=load_config(project_root / "configs/v0.1.yaml"),
        config=config,
        schema=schema,
        thresholds=thresholds,
    )


def _load_classical_state(path: Path, *, method: str) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            state = pickle.load(stream)
    except (OSError, pickle.UnpicklingError) as error:
        raise V0_2LabelFreeScoringError(f"cannot load {method} fitted state") from error
    expected_keys = (
        {
            "milestone",
            "run_id",
            "method",
            "execution_commit",
            "freeze_record_sha256",
            "v0_2_config_sha256",
            "config_sha256",
            "reference_manifest_sha256",
            "fit",
        }
        if method == "ecc_residual"
        else {
            "milestone",
            "run_id",
            "method",
            "execution_commit",
            "freeze_record_sha256",
            "v0_2_config_sha256",
            "config_sha256",
            "reference_manifest_sha256",
            "scaler_fit",
            "model_fit",
        }
    )
    _require(
        isinstance(state, dict)
        and set(state) == expected_keys
        and state.get("milestone") == "v0.2.4"
        and state.get("run_id") == RUN_ID
        and state.get("method") == method
        and state.get("execution_commit") == CALIBRATION_EXECUTION_COMMIT
        and state.get("freeze_record_sha256") == FREEZE_RECORD_SHA256
        and state.get("v0_2_config_sha256") == EXPECTED_CONFIG_SHA256
        and state.get("config_sha256") == CLASSICAL_CONFIG_SHA256
        and state.get("reference_manifest_sha256") == REFERENCE_MANIFEST_SHA256,
        f"{method} fitted-state metadata changed",
    )
    return state


def _decode(path: Path, *, color: bool) -> NDArray[np.uint8]:
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise V0_2LabelFreeScoringError("cannot read opaque asset") from error
    if not encoded:
        raise V0_2LabelFreeScoringError("opaque asset is empty")
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION if color else DECODE_FLAGS
    try:
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), flags)
    except cv2.error as error:
        raise V0_2LabelFreeScoringError("cannot decode opaque asset") from error
    expected_ndim = 3 if color else 2
    if (
        not isinstance(decoded, np.ndarray)
        or decoded.ndim != expected_ndim
        or decoded.dtype != np.uint8
        or decoded.size == 0
        or (color and decoded.shape[2] != 3)
    ):
        raise V0_2LabelFreeScoringError("opaque asset decoded outside the fixed boundary")
    return decoded


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _classical_evidence(
    decoded: NDArray[np.uint8],
    *,
    asset_id: str,
    method: str,
    state: Mapping[str, Any],
    config: ProjectConfig,
) -> ScoreEvidence:
    try:
        image = preprocess_decoded_image(decoded, config.preprocessing)
        if method == "ecc_residual":
            result: ECCResidualScoreResult | PatchHOGScoreResult = score_ecc_residual(
                image,
                fitted=state["fit"],
                config=config,
            )
        else:
            result = score_patch_hog(
                image,
                scaler_fit=state["scaler_fit"],
                model_fit=state["model_fit"],
                config=config,
            )
    except ImagePreprocessingError as error:
        failure_score = (
            config.ecc_residual_scoring.failure_score
            if method == "ecc_residual"
            else config.patch_hog_scoring.failure_score
        )
        return ScoreEvidence(asset_id, "failed", str(error.code), float(failure_score), {})
    except Exception:
        failure_score = (
            config.ecc_residual_scoring.failure_score
            if method == "ecc_residual"
            else config.patch_hog_scoring.failure_score
        )
        code = (
            "ECC_SCORE_EXECUTION_FAILED"
            if method == "ecc_residual"
            else "PATCH_HOG_SCORE_EXECUTION_FAILED"
        )
        return ScoreEvidence(asset_id, "failed", code, float(failure_score), {})

    if isinstance(result, ECCResidualScoreResult):
        diagnostics = {
            "correlation": result.correlation,
            "effective_pixel_count": result.effective_pixel_count,
            "effective_support_fraction": result.effective_support_fraction,
            "registration_status": result.registration_status,
            "registration_valid_fraction": result.registration_valid_fraction,
            "rotation_degrees": result.rotation_degrees,
            "top_pixel_count": result.top_pixel_count,
            "translation_x_pixels": result.translation_x_pixels,
            "translation_y_pixels": result.translation_y_pixels,
            "warp_matrix": result.warp_matrix,
        }
    else:
        diagnostics = {
            "failed_patch_index": result.failed_patch_index,
            "patch_anomaly_scores": result.patch_anomaly_scores,
            "successful_patch_count": result.successful_patch_count,
            "top_patch_count": result.top_patch_count,
            "top_patch_indices": result.top_patch_indices,
        }
    return ScoreEvidence(
        asset_id=asset_id,
        score_status=result.score_status,
        score_failure_code=None if result.failure_code is None else str(result.failure_code),
        anomaly_score=float(result.anomaly_score),
        diagnostics=_json_safe(diagnostics),
    )


def _run_classical_method(
    *,
    scorer_root: Path,
    assets: tuple[OpaqueAsset, ...],
    method: str,
    state: Mapping[str, Any],
    config: ProjectConfig,
    threshold: float,
    schema: Mapping[str, Any],
    output_root: Path,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    for index, asset in enumerate(assets, start=1):
        decoded = _decode(scorer_root / asset.relative_path, color=False)
        _classical_evidence(
            decoded,
            asset_id=asset.asset_id,
            method=method,
            state=state,
            config=config,
        )
        if index % 50 == 0 or index == ASSET_COUNT:
            _emit(progress, f"{method} warm-up: {index}/{ASSET_COUNT}")

    canonical: list[ScoreEvidence] = []
    timed: list[TimedScoreEvidence] = []
    for pass_index in range(TIMED_PASS_COUNT):
        for index, asset in enumerate(assets, start=1):
            decoded = _decode(scorer_root / asset.relative_path, color=False)
            started = time.perf_counter_ns()
            evidence = _classical_evidence(
                decoded,
                asset_id=asset.asset_id,
                method=method,
                state=state,
                config=config,
            )
            duration = max(1, time.perf_counter_ns() - started)
            if pass_index == 0:
                canonical.append(evidence)
            timed.append(
                TimedScoreEvidence(
                    pass_index=pass_index,
                    asset_id=asset.asset_id,
                    adapter_duration_ns=None,
                    scorer_duration_ns=duration,
                    score_status=evidence.score_status,
                    score_failure_code=evidence.score_failure_code,
                    anomaly_score=evidence.anomaly_score,
                )
            )
            if index % 50 == 0 or index == ASSET_COUNT:
                _emit(
                    progress,
                    f"{method} timed pass {pass_index + 1}/{TIMED_PASS_COUNT}: "
                    f"{index}/{ASSET_COUNT}",
                )
    artifacts = build_method_scoring_artifacts(
        run_id=RUN_ID,
        method=method,
        threshold=threshold,
        scores=canonical,
        timed_scores=timed,
        schema=schema,
    )
    write_method_scoring_artifacts(output_root / method, artifacts)
    serialized = read_method_scoring_artifacts(output_root / method, schema=schema)
    return latency_summary(serialized.latency_records)


def _adapt_dinov2(decoded_bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    try:
        rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
    except cv2.error as error:
        raise V0_2LabelFreeScoringError("DINOv2 input adapter failed") from error
    output = np.ascontiguousarray(resized, dtype=np.uint8)
    if output.shape != DINO_INPUT_SHAPE or output.dtype != np.uint8:
        raise V0_2LabelFreeScoringError("DINOv2 input adapter output changed")
    return output


def _array_sha256(value: NDArray[np.uint8]) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _create_dinov2_input_store(
    *,
    scorer_root: Path,
    assets: tuple[OpaqueAsset, ...],
    store_path: Path,
    manifest_path: Path,
    progress: ProgressCallback | None,
) -> None:
    if store_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite DINOv2 scoring input store")
    store_path.parent.mkdir(parents=True, exist_ok=True)
    for index, asset in enumerate(assets, start=1):
        decoded = _decode(scorer_root / asset.relative_path, color=True)
        _adapt_dinov2(decoded)
        if index % 50 == 0 or index == ASSET_COUNT:
            _emit(progress, f"dinov2 input-adapter warm-up: {index}/{ASSET_COUNT}")

    partial_path = store_path.with_name(f".{store_path.name}.partial")
    records = [
        {
            "asset_id": asset.asset_id,
            "index": index,
            "rgb_sha256": None,
            "adapter_duration_ns": [],
        }
        for index, asset in enumerate(assets)
    ]
    try:
        for pass_index in range(TIMED_PASS_COUNT):
            for index, asset in enumerate(assets):
                decoded = _decode(scorer_root / asset.relative_path, color=True)
                started = time.perf_counter_ns()
                adapted = _adapt_dinov2(decoded)
                duration = max(1, time.perf_counter_ns() - started)
                observed_hash = _array_sha256(adapted)
                if pass_index == 0:
                    records[index]["rgb_sha256"] = observed_hash
                elif records[index]["rgb_sha256"] != observed_hash:
                    raise V0_2LabelFreeScoringError("DINOv2 adapter output changed across passes")
                records[index]["adapter_duration_ns"].append(duration)
                if (index + 1) % 50 == 0 or index + 1 == ASSET_COUNT:
                    _emit(
                        progress,
                        f"dinov2 input-adapter timed pass {pass_index + 1}/{TIMED_PASS_COUNT}: "
                        f"{index + 1}/{ASSET_COUNT}",
                    )
        store = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.uint8,
            shape=DINO_STORE_SHAPE,
        )
        for index, asset in enumerate(assets):
            decoded = _decode(scorer_root / asset.relative_path, color=True)
            adapted = _adapt_dinov2(decoded)
            if _array_sha256(adapted) != records[index]["rgb_sha256"]:
                raise V0_2LabelFreeScoringError("untimed DINOv2 transfer input changed")
            store[index] = adapted
            if (index + 1) % 50 == 0 or index + 1 == ASSET_COUNT:
                _emit(progress, f"dinov2 isolated input transfer: {index + 1}/{ASSET_COUNT}")
        store.flush()
        del store
        os.replace(partial_path, store_path)
        write_json_atomic(
            manifest_path,
            {
                "schema_version": DINO_STORE_SCHEMA,
                "run_id": RUN_ID,
                "opaque_scoring_manifest_sha256": OPAQUE_SCORING_MANIFEST_SHA256,
                "shape": list(DINO_STORE_SHAPE),
                "dtype": "uint8",
                "store_byte_count": store_path.stat().st_size,
                "store_sha256": sha256_file(store_path),
                "records": records,
                "labels_accessed": False,
                "semantic_paths_accessed": False,
                "sealed_mapping_accessed": False,
            },
        )
    except Exception:
        partial_path.unlink(missing_ok=True)
        store_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise


def _concurrent_benchmark_count() -> int:
    markers = (
        "run_v0_2_cpu_timing.py",
        "run_v0_2_cpu_timing_resolution.py",
        "run_v0_2_5_label_free_scoring.py",
        "run_v0_2_5_dinov2_scoring_worker.py",
    )
    count = 0
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            pid = int(path.parent.name)
            if pid == os.getpid():
                continue
            process_name = (path.parent / "comm").read_text(encoding="utf-8").strip().lower()
            command = path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, ValueError):
            continue
        if "python" in process_name and any(marker in command for marker in markers):
            count += 1
    return count


def _run_dinov2_worker(
    *,
    project_root: Path,
    execution_commit: str,
    store_path: Path,
    manifest_path: Path,
    state_path: Path,
    threshold: float,
    output_root: Path,
    report_path: Path,
    progress: ProgressCallback | None,
) -> None:
    artifact_dir = project_root / "data/external/v0.2/model-assets"
    source_root = artifact_dir / f"dinov2-source-sha256-{EXPECTED_SOURCE_SHA256}" / SOURCE_ROOT
    command = [
        str(project_root / "environments/v0.2-preflight/.venv/bin/python"),
        "-I",
        "-B",
        str(project_root / "scripts/run_v0_2_5_dinov2_scoring_worker.py"),
        "--project-root",
        str(project_root),
        "--execution-commit",
        execution_commit,
        "--input-store",
        str(store_path),
        "--input-manifest",
        str(manifest_path),
        "--fitted-state",
        str(state_path),
        "--fitted-state-sha256",
        STATE_HASHES["dinov2_vits14_224_nn"],
        "--threshold",
        repr(threshold),
        "--artifact-dir",
        str(artifact_dir),
        "--source-root",
        str(source_root),
        "--environment-root",
        str(project_root / "environments/v0.2-preflight/.venv"),
        "--output-dir",
        str(output_root / "dinov2_vits14_224_nn"),
        "--report",
        str(report_path),
    ]
    serialized = " ".join(command).lower()
    _require(
        "sealed" not in serialized
        and "boundary-state" not in serialized
        and "source_path" not in serialized
        and "hmac" not in serialized,
        "isolated worker command exposes a protected boundary input",
    )
    _emit(progress, "starting isolated DINOv2 label-free scoring worker")
    completed = subprocess.run(
        command,
        cwd=project_root,
        check=False,
        env=_isolated_worker_environment(),
        text=True,
    )
    if completed.returncode != 0:
        raise V0_2LabelFreeScoringError(
            f"isolated DINOv2 worker failed with exit code {completed.returncode}"
        )
    _emit(progress, "isolated DINOv2 label-free scoring worker complete")


def _artifact_identities(public_root: Path) -> list[dict[str, str]]:
    identities = []
    for method in METHODS:
        for name in ("scores.csv", "classifications.csv", "latency-observations.csv"):
            path = public_root / method / name
            identities.append(
                {
                    "relative_path": path.relative_to(public_root).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
    return identities


def run_v0_2_label_free_scoring(
    *,
    project_root: Path,
    execution_commit: str,
    external_root: Path,
    public_root: Path,
    work_root: Path,
    confirm_no_concurrent_project_benchmark: bool,
    progress: ProgressCallback | None = None,
) -> dict[str, int]:
    """Execute the fixed label-free run once and publish only complete CSV bundles."""
    if confirm_no_concurrent_project_benchmark is not True:
        raise V0_2LabelFreeScoringError("explicit no-concurrent-benchmark confirmation is required")
    project_root = project_root.resolve()
    external_root = external_root.resolve()
    public_root = public_root.resolve()
    work_root = work_root.resolve()
    inputs = validate_v0_2_scoring_inputs(
        project_root=project_root,
        execution_commit=execution_commit,
        external_root=external_root,
        public_root=public_root,
        work_root=work_root,
    )
    machine = capture_target_machine()
    machine_evaluation = evaluate_target_machine(machine)
    _require(machine_evaluation["status"] == "pass", "fixed target-machine check failed")
    _require(_concurrent_benchmark_count() == 0, "another project benchmark is running")
    _emit(
        progress, "fixed identities, opaque assets, target machine, and benchmark isolation passed"
    )

    state_root = work_root / "fitted-state"
    stage_root = Path(tempfile.mkdtemp(dir=work_root, prefix=".v0.2.5-stage-"))
    public_stage = stage_root / "public"
    dino_store = stage_root / "dinov2-rgb512.npy"
    dino_manifest = stage_root / "dinov2-rgb512-manifest.json"
    dino_report = stage_root / "dinov2-worker-report.json"
    local_state_stage = stage_root / "label-free-scoring-state.json"
    scorer_root = external_root / "scorer"
    latency_summaries: dict[str, Any] = {}
    try:
        public_stage.mkdir()
        for method in ("ecc_residual", "patch_hog_ocsvm"):
            suffix = "ecc_residual.pkl" if method == "ecc_residual" else "patch_hog_ocsvm.pkl"
            state = _load_classical_state(state_root / suffix, method=method)
            latency_summaries[method] = _run_classical_method(
                scorer_root=scorer_root,
                assets=inputs.assets,
                method=method,
                state=state,
                config=inputs.classical_config,
                threshold=inputs.thresholds[method],
                schema=inputs.schema,
                output_root=public_stage,
                progress=progress,
            )
        _create_dinov2_input_store(
            scorer_root=scorer_root,
            assets=inputs.assets,
            store_path=dino_store,
            manifest_path=dino_manifest,
            progress=progress,
        )
        _run_dinov2_worker(
            project_root=project_root,
            execution_commit=execution_commit,
            store_path=dino_store,
            manifest_path=dino_manifest,
            state_path=state_root / "dinov2_vits14_224_nn.pt",
            threshold=inputs.thresholds["dinov2_vits14_224_nn"],
            output_root=public_stage,
            report_path=dino_report,
            progress=progress,
        )
        worker_report = _read_json(dino_report, label="DINOv2 worker report")
        _require(
            worker_report.get("schema_version") == "v0.2.5-dinov2-worker-report-v1"
            and worker_report.get("record_counts")
            == {"scores": ASSET_COUNT, "classifications": ASSET_COUNT, "latency": 600}
            and worker_report.get("network_attempted") is False,
            "DINOv2 worker report is incomplete",
        )
        latency_summaries["dinov2_vits14_224_nn"] = worker_report["latency_summary"]
        _validate_opaque_assets(scorer_root)

        staged_identities = []
        for method in METHODS:
            method_root = public_stage / method
            read_method_scoring_artifacts(method_root, schema=inputs.schema)
            _require(
                {path.name for path in method_root.iterdir() if path.is_file()}
                == {"scores.csv", "classifications.csv", "latency-observations.csv"},
                "staged method artifact inventory is incomplete",
            )
            for path in sorted(method_root.iterdir()):
                staged_identities.append(
                    {
                        "relative_path": path.relative_to(public_stage).as_posix(),
                        "sha256": sha256_file(path),
                    }
                )
        local_state = {
            "schema_version": "v0.2.5-label-free-scoring-state-v1",
            "milestone": MILESTONE,
            "run_id": RUN_ID,
            "execution_commit": execution_commit,
            "opaque_scoring_manifest_sha256": OPAQUE_SCORING_MANIFEST_SHA256,
            "artifact_identities": staged_identities,
            "record_counts": {
                method: {"scores": ASSET_COUNT, "classifications": ASSET_COUNT, "latency": 600}
                for method in METHODS
            },
            "latency_summaries": latency_summaries,
            "machine": asdict(machine),
            "machine_evaluation": machine_evaluation,
            "classical_environment": asdict(_capture_cpu_environment()),
            "dinov2_environment": worker_report["environment"],
            "benchmark_isolation": {
                "explicit_confirmation": True,
                "observed_concurrent_project_benchmarks": 0,
            },
            "boundary": {
                "labels_accessed": False,
                "semantic_paths_accessed": False,
                "sealed_mapping_accessed": False,
                "official_split_accessed": False,
                "image_content_displayed": False,
            },
        }
        write_json_atomic(local_state_stage, local_state)

        moved: list[tuple[Path, Path]] = []
        local_state_destination = work_root / "label-free-scoring-state.json"
        try:
            for method in METHODS:
                for name in ("scores.csv", "classifications.csv", "latency-observations.csv"):
                    source = public_stage / method / name
                    destination = public_root / method / name
                    os.replace(source, destination)
                    moved.append((destination, source))
            os.replace(local_state_stage, local_state_destination)
            updated_state = dict(inputs.boundary_state)
            boundary = dict(updated_state["boundary"])
            boundary["anomaly_score_computed"] = True
            boundary["image_content_decoded"] = True
            updated_state["boundary"] = boundary
            updated_state["label_free_scoring"] = {
                "milestone": MILESTONE,
                "execution_commit": execution_commit,
                "opaque_scoring_manifest_sha256": OPAQUE_SCORING_MANIFEST_SHA256,
                "artifact_identities": _artifact_identities(public_root),
                "record_counts": local_state["record_counts"],
                "anomaly_labels_used": False,
                "semantic_paths_accessed": False,
                "sealed_mapping_accessed": False,
                "official_split_accessed": False,
                "image_content_displayed": False,
            }
            write_json_atomic(external_root / "boundary-state.json", updated_state, overwrite=True)
        except Exception:
            local_state_destination.unlink(missing_ok=True)
            for destination, source in reversed(moved):
                source.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, source)
            raise
        shutil.rmtree(stage_root, ignore_errors=True)
        return {method: ASSET_COUNT for method in METHODS}
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise
