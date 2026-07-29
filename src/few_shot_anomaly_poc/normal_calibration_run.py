"""Run and preserve the fixed normal-reference fit and calibration checkpoint."""

from __future__ import annotations

import csv
import json
import os
import pickle
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from few_shot_anomaly_poc.calibration import (
    CalibrationMethod,
    NormalThresholdCalibrationResult,
    calibrate_normal_threshold,
    normal_threshold_calibration_result_is_valid,
)
from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.dataset_integrity import verify_visa_pcb1_integrity
from few_shot_anomaly_poc.ecc_residual import (
    ECCResidualScoreResult,
    score_ecc_residual,
)
from few_shot_anomaly_poc.ecc_template import (
    ECCTemplateFitResult,
    fit_ecc_normal_template,
)
from few_shot_anomaly_poc.errors import ImagePreprocessingError
from few_shot_anomaly_poc.freeze_checkpoint import (
    read_and_verify_pre_evaluation_freeze,
)
from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.hog_features import (
    PatchHOGFeatureResult,
    extract_patch_hog_features,
)
from few_shot_anomaly_poc.hog_models import (
    PatchHOGModelFitResult,
    fit_position_one_class_svms,
    position_one_class_svm_state_is_valid,
)
from few_shot_anomaly_poc.hog_scalers import (
    PatchHOGScalerFitResult,
    fit_position_scalers,
    position_scaler_state_is_valid,
)
from few_shot_anomaly_poc.hog_scoring import (
    PatchHOGScoreResult,
    score_patch_hog,
)
from few_shot_anomaly_poc.jsonio import write_json_atomic
from few_shot_anomaly_poc.preprocessing import load_and_preprocess_image

CALIBRATION_CONTRACT_VERSION = "normal-only-calibration/v0.1"
CALIBRATION_CHECKPOINT_ID = "v0.1-normal-reference-fit-and-calibration"
CALIBRATION_STATUS = "THRESHOLDS_FIXED_BEFORE_FINAL_TEST"
LOCAL_STATE_LOGICAL_PATH = "work/v0.1/calibration/normal-only-state.pkl"
PICKLE_PROTOCOL = 5
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SCORE_COLUMNS = (
    "contract_version",
    "checkpoint_id",
    "method",
    "partition",
    "relative_path",
    "score_status",
    "score_failure_code",
    "anomaly_score",
    "diagnostics_json",
)

type CalibrationScoreResult = ECCResidualScoreResult | PatchHOGScoreResult
type ProgressCallback = Callable[[str], None]


class NormalCalibrationRunError(Exception):
    """Reject invalid inputs, failed fits, or changed checkpoint state."""


@dataclass(frozen=True)
class NormalCalibrationRunState:
    """Local fitted objects and exact normal-only scores for later scoring."""

    contract_version: str
    checkpoint_id: str
    source_commit: str
    freeze_checkpoint_sha256: str
    config_sha256: str
    partition_manifest_sha256: str
    dataset_integrity_sha256: str
    reference_paths: tuple[str, ...]
    calibration_paths: tuple[str, ...]
    ecc_fit: ECCTemplateFitResult
    hog_scaler_fit: PatchHOGScalerFitResult
    hog_model_fit: PatchHOGModelFitResult
    ecc_calibration_scores: dict[str, ECCResidualScoreResult]
    hog_calibration_scores: dict[str, PatchHOGScoreResult]
    ecc_calibration: NormalThresholdCalibrationResult
    hog_calibration: NormalThresholdCalibrationResult


def _emit(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _json_ready(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    raise NormalCalibrationRunError(
        f"cannot serialize calibration diagnostic type: {type(value).__name__}"
    )


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            _json_ready(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise NormalCalibrationRunError("calibration diagnostic is not finite JSON") from error


def load_fixed_normal_partitions(
    path: Path,
    *,
    expected_reference_paths: tuple[str, ...],
    expected_calibration_count: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read the frozen reference and calibration path lists in selection-rank order."""
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = tuple(reader)
            fieldnames = tuple(reader.fieldnames or ())
    except (OSError, csv.Error) as error:
        raise NormalCalibrationRunError("cannot read the normal partition manifest") from error
    if fieldnames != (
        "partition",
        "selection_rank",
        "relative_path",
        "selection_sha256",
    ):
        raise NormalCalibrationRunError("normal partition columns are invalid")

    parsed: list[tuple[int, str, str]] = []
    seen_paths: set[str] = set()
    seen_ranks: set[int] = set()
    for row in rows:
        try:
            rank = int(row["selection_rank"])
        except (KeyError, TypeError, ValueError) as error:
            raise NormalCalibrationRunError("normal partition rank is invalid") from error
        partition = row.get("partition", "")
        relative_path = row.get("relative_path", "")
        digest = row.get("selection_sha256", "")
        if (
            partition not in {"reference", "calibration"}
            or rank < 1
            or not relative_path.startswith("pcb1/Data/Images/Normal/")
            or "\\" in relative_path
            or not SHA256_PATTERN.fullmatch(digest)
            or relative_path in seen_paths
            or rank in seen_ranks
        ):
            raise NormalCalibrationRunError("normal partition row is invalid")
        parsed.append((rank, partition, relative_path))
        seen_paths.add(relative_path)
        seen_ranks.add(rank)

    parsed.sort()
    if [rank for rank, _, _ in parsed] != list(range(1, len(parsed) + 1)):
        raise NormalCalibrationRunError("normal partition ranks are not contiguous")
    reference_paths = tuple(path for _, partition, path in parsed if partition == "reference")
    calibration_paths = tuple(
        path for _, partition, path in parsed if partition == "calibration"
    )
    if (
        reference_paths != expected_reference_paths
        or len(calibration_paths) != expected_calibration_count
        or len(parsed) != len(reference_paths) + len(calibration_paths)
    ):
        raise NormalCalibrationRunError("normal partition does not match the freeze record")
    return reference_paths, calibration_paths


def _ecc_preprocessing_failure(
    error: ImagePreprocessingError,
    *,
    config: ProjectConfig,
) -> ECCResidualScoreResult:
    return ECCResidualScoreResult(
        score_status="failed",
        failure_code=error.code,
        anomaly_score=config.ecc_residual_scoring.failure_score,
        registration_status="not_run",
        correlation=None,
        warp_matrix=None,
        rotation_degrees=None,
        translation_x_pixels=None,
        translation_y_pixels=None,
        registration_valid_fraction=None,
        effective_support_fraction=None,
        effective_pixel_count=None,
        top_pixel_count=None,
    )


def _hog_preprocessing_failure(
    error: ImagePreprocessingError,
    *,
    config: ProjectConfig,
) -> PatchHOGScoreResult:
    return PatchHOGScoreResult(
        score_status="failed",
        failure_code=error.code,
        anomaly_score=config.patch_hog_scoring.failure_score,
        patch_anomaly_scores=None,
        top_patch_count=None,
        top_patch_indices=(),
        successful_patch_count=0,
        failed_patch_index=None,
    )


def _fit_methods(
    reference_paths: tuple[str, ...],
    *,
    dataset_root: Path,
    config: ProjectConfig,
    progress: ProgressCallback | None,
) -> tuple[ECCTemplateFitResult, PatchHOGScalerFitResult, PatchHOGModelFitResult]:
    references = {}
    for relative_path in reference_paths:
        try:
            references[relative_path] = load_and_preprocess_image(
                dataset_root / relative_path,
                config.preprocessing,
            )
        except ImagePreprocessingError as error:
            raise NormalCalibrationRunError(
                f"reference preprocessing failed for {relative_path}: {error.code}"
            ) from error
    _emit(progress, f"reference preprocessing complete: {len(references)}/{len(reference_paths)}")

    ecc_fit = fit_ecc_normal_template(references, config=config)
    if not ecc_fit.succeeded:
        raise NormalCalibrationRunError(
            f"ECC reference fitting failed: {ecc_fit.failure_code}"
        )
    _emit(
        progress,
        "ECC reference fitting complete: "
        f"successful={ecc_fit.successful_reference_count}, "
        f"failed={ecc_fit.failed_reference_count}",
    )

    reference_features: dict[str, PatchHOGFeatureResult] = {}
    for relative_path in reference_paths:
        feature_result = extract_patch_hog_features(
            references[relative_path],
            config=config,
        )
        if not feature_result.succeeded:
            raise NormalCalibrationRunError(
                f"Patch HOG reference extraction failed for {relative_path}: "
                f"{feature_result.failure_code}"
            )
        reference_features[relative_path] = feature_result
    scaler_fit = fit_position_scalers(reference_features, config=config)
    if not scaler_fit.succeeded:
        raise NormalCalibrationRunError(
            f"Patch HOG scaler fitting failed: {scaler_fit.failure_code}"
        )
    model_fit = fit_position_one_class_svms(
        reference_features,
        scaler_fit=scaler_fit,
        config=config,
    )
    if not model_fit.succeeded:
        raise NormalCalibrationRunError(
            f"Patch HOG model fitting failed: {model_fit.failure_code}"
        )
    _emit(
        progress,
        "Patch HOG fitting complete: "
        f"scalers={scaler_fit.successful_position_count}, "
        f"models={model_fit.successful_position_count}",
    )
    return ecc_fit, scaler_fit, model_fit


def _score_calibration(
    calibration_paths: tuple[str, ...],
    *,
    dataset_root: Path,
    ecc_fit: ECCTemplateFitResult,
    scaler_fit: PatchHOGScalerFitResult,
    model_fit: PatchHOGModelFitResult,
    config: ProjectConfig,
    progress: ProgressCallback | None,
) -> tuple[
    dict[str, ECCResidualScoreResult],
    dict[str, PatchHOGScoreResult],
]:
    ecc_scores = {}
    hog_scores = {}
    for index, relative_path in enumerate(calibration_paths, start=1):
        try:
            image = load_and_preprocess_image(
                dataset_root / relative_path,
                config.preprocessing,
            )
        except ImagePreprocessingError as error:
            ecc_scores[relative_path] = _ecc_preprocessing_failure(error, config=config)
            hog_scores[relative_path] = _hog_preprocessing_failure(error, config=config)
        else:
            ecc_scores[relative_path] = score_ecc_residual(
                image,
                fitted=ecc_fit,
                config=config,
            )
            hog_scores[relative_path] = score_patch_hog(
                image,
                scaler_fit=scaler_fit,
                model_fit=model_fit,
                config=config,
            )
        if index % 50 == 0 or index == len(calibration_paths):
            _emit(
                progress,
                f"normal-only calibration scoring: {index}/{len(calibration_paths)}",
            )
    return ecc_scores, hog_scores


def _validate_fitted_objects(
    state: NormalCalibrationRunState,
    *,
    config: ProjectConfig,
) -> None:
    if (
        not state.ecc_fit.succeeded
        or state.ecc_fit.reference_count != len(state.reference_paths)
        or state.ecc_fit.anchor_path != min(state.reference_paths)
        or state.ecc_fit.template is None
        or state.ecc_fit.support_mask is None
        or state.ecc_fit.successful_reference_count
        < config.ecc_template.minimum_successful_references
    ):
        raise NormalCalibrationRunError("stored ECC fitted state is invalid")
    if (
        not state.hog_scaler_fit.succeeded
        or state.hog_scaler_fit.scalers is None
        or state.hog_scaler_fit.reference_paths != tuple(sorted(state.reference_paths))
        or len(state.hog_scaler_fit.scalers) != config.patch_hog.patch_count
        or not all(
            position_scaler_state_is_valid(scaler, config=config)
            for scaler in state.hog_scaler_fit.scalers
        )
    ):
        raise NormalCalibrationRunError("stored Patch HOG scaler state is invalid")
    if (
        not state.hog_model_fit.succeeded
        or state.hog_model_fit.models is None
        or state.hog_model_fit.reference_paths != state.hog_scaler_fit.reference_paths
        or len(state.hog_model_fit.models) != config.patch_hog.patch_count
        or not all(
            position_one_class_svm_state_is_valid(model, config=config)
            for model in state.hog_model_fit.models
        )
    ):
        raise NormalCalibrationRunError("stored Patch HOG model state is invalid")


def validate_normal_calibration_state(
    state: object,
    *,
    config: ProjectConfig,
) -> None:
    """Reject state that cannot reproduce the fixed calibration threshold."""
    if (
        not isinstance(state, NormalCalibrationRunState)
        or state.contract_version != CALIBRATION_CONTRACT_VERSION
        or state.checkpoint_id != CALIBRATION_CHECKPOINT_ID
        or not COMMIT_PATTERN.fullmatch(state.source_commit)
        or any(
            not SHA256_PATTERN.fullmatch(value)
            for value in (
                state.freeze_checkpoint_sha256,
                state.config_sha256,
                state.partition_manifest_sha256,
                state.dataset_integrity_sha256,
            )
        )
        or len(state.reference_paths) != config.selection.reference_count
        or len(set(state.reference_paths)) != len(state.reference_paths)
        or len(state.calibration_paths) != 884
        or len(set(state.calibration_paths)) != len(state.calibration_paths)
        or set(state.reference_paths) & set(state.calibration_paths)
        or tuple(state.ecc_calibration_scores) != state.calibration_paths
        or tuple(state.hog_calibration_scores) != state.calibration_paths
    ):
        raise NormalCalibrationRunError("normal calibration state metadata is invalid")
    _validate_fitted_objects(state, config=config)

    calibrations = (
        (
            state.ecc_calibration_scores,
            CalibrationMethod.ECC_RESIDUAL,
            state.ecc_calibration,
        ),
        (
            state.hog_calibration_scores,
            CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM,
            state.hog_calibration,
        ),
    )
    for scores, method, stored in calibrations:
        if (
            not normal_threshold_calibration_result_is_valid(stored, config=config)
            or calibrate_normal_threshold(scores, method=method, config=config) != stored
        ):
            raise NormalCalibrationRunError(
                f"stored calibration is invalid for {method}"
            )


def build_normal_calibration_state(
    *,
    source_commit: str,
    freeze_checkpoint_sha256: str,
    config_sha256: str,
    partition_manifest_sha256: str,
    dataset_integrity_sha256: str,
    reference_paths: tuple[str, ...],
    calibration_paths: tuple[str, ...],
    dataset_root: Path,
    config: ProjectConfig,
    progress: ProgressCallback | None = None,
) -> NormalCalibrationRunState:
    """Fit both fixed methods and calibrate thresholds from normal images only."""
    ecc_fit, scaler_fit, model_fit = _fit_methods(
        reference_paths,
        dataset_root=dataset_root,
        config=config,
        progress=progress,
    )
    ecc_scores, hog_scores = _score_calibration(
        calibration_paths,
        dataset_root=dataset_root,
        ecc_fit=ecc_fit,
        scaler_fit=scaler_fit,
        model_fit=model_fit,
        config=config,
        progress=progress,
    )
    ecc_calibration = calibrate_normal_threshold(
        ecc_scores,
        method=CalibrationMethod.ECC_RESIDUAL,
        config=config,
    )
    hog_calibration = calibrate_normal_threshold(
        hog_scores,
        method=CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM,
        config=config,
    )
    if not ecc_calibration.succeeded or not hog_calibration.succeeded:
        raise NormalCalibrationRunError("normal-only threshold calibration failed")

    state = NormalCalibrationRunState(
        contract_version=CALIBRATION_CONTRACT_VERSION,
        checkpoint_id=CALIBRATION_CHECKPOINT_ID,
        source_commit=source_commit,
        freeze_checkpoint_sha256=freeze_checkpoint_sha256,
        config_sha256=config_sha256,
        partition_manifest_sha256=partition_manifest_sha256,
        dataset_integrity_sha256=dataset_integrity_sha256,
        reference_paths=reference_paths,
        calibration_paths=calibration_paths,
        ecc_fit=ecc_fit,
        hog_scaler_fit=scaler_fit,
        hog_model_fit=model_fit,
        ecc_calibration_scores=ecc_scores,
        hog_calibration_scores=hog_scores,
        ecc_calibration=ecc_calibration,
        hog_calibration=hog_calibration,
    )
    validate_normal_calibration_state(state, config=config)
    return state


def _score_diagnostics(result: CalibrationScoreResult) -> dict:
    values = asdict(result)
    for field in ("score_status", "failure_code", "anomaly_score"):
        values.pop(field)
    return values


def _write_scores_csv(
    path: Path,
    *,
    method: CalibrationMethod,
    scores: Mapping[str, CalibrationScoreResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=SCORE_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for relative_path in sorted(scores):
            result = scores[relative_path]
            writer.writerow(
                {
                    "contract_version": CALIBRATION_CONTRACT_VERSION,
                    "checkpoint_id": CALIBRATION_CHECKPOINT_ID,
                    "method": str(method),
                    "partition": "calibration",
                    "relative_path": relative_path,
                    "score_status": result.score_status,
                    "score_failure_code": (
                        str(result.failure_code) if result.failure_code is not None else ""
                    ),
                    "anomaly_score": result.anomaly_score,
                    "diagnostics_json": _canonical_json(_score_diagnostics(result)),
                }
            )


def _ecc_fit_summary(fit: ECCTemplateFitResult) -> dict:
    return {
        "status": fit.status,
        "failure_code": str(fit.failure_code) if fit.failure_code is not None else None,
        "anchor_path": fit.anchor_path,
        "reference_count": fit.reference_count,
        "successful_reference_count": fit.successful_reference_count,
        "failed_reference_count": fit.failed_reference_count,
        "support_fraction": fit.support_fraction,
        "reference_diagnostics": [
            _json_ready(asdict(diagnostic)) for diagnostic in fit.reference_diagnostics
        ],
    }


def _hog_fit_summary(
    scaler_fit: PatchHOGScalerFitResult,
    model_fit: PatchHOGModelFitResult,
) -> dict:
    assert model_fit.models is not None
    support_counts = tuple(int(model.support_.size) for model in model_fit.models)
    iteration_counts = tuple(int(model.n_iter_) for model in model_fit.models)
    return {
        "scaler_status": scaler_fit.status,
        "model_status": model_fit.status,
        "reference_count": model_fit.reference_count,
        "successful_scaler_position_count": scaler_fit.successful_position_count,
        "successful_model_position_count": model_fit.successful_position_count,
        "support_vector_count_min": min(support_counts),
        "support_vector_count_max": max(support_counts),
        "support_vector_count_total": sum(support_counts),
        "iteration_count_min": min(iteration_counts),
        "iteration_count_max": max(iteration_counts),
    }


def _calibration_summary(result: NormalThresholdCalibrationResult) -> dict:
    return {
        "status": result.status,
        "failure_code": (
            str(result.failure_code) if result.failure_code is not None else None
        ),
        "quantile": result.quantile,
        "sample_count": result.sample_count,
        "rank": result.rank,
        "threshold": result.threshold,
        "threshold_source_path": result.threshold_source_path,
        "failed_score_count": result.failed_score_count,
        "failed_score_paths": list(result.failed_score_paths),
        "predicted_anomalous_count": result.predicted_anomalous_count,
        "realized_normal_false_positive_rate": result.realized_false_positive_rate,
        "prediction_rule": "failed_or_strictly_greater",
    }


def _write_pickle_atomic(path: Path, state: NormalCalibrationRunState) -> None:
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
            pickle.dump(state, stream, protocol=PICKLE_PROTOCOL)
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def write_normal_calibration_outputs(
    state: NormalCalibrationRunState,
    *,
    output_dir: Path,
    state_path: Path,
    config: ProjectConfig,
) -> dict:
    """Atomically write local fitted state and public normal-only artifacts."""
    validate_normal_calibration_state(state, config=config)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if state_path.exists():
        raise FileExistsError(f"refusing to overwrite {state_path}")

    _write_pickle_atomic(state_path, state)
    state_written = True
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=output_dir.parent,
            prefix=f".{output_dir.name}.",
        )
    )
    try:
        score_files = (
            (
                CalibrationMethod.ECC_RESIDUAL,
                state.ecc_calibration_scores,
            ),
            (
                CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM,
                state.hog_calibration_scores,
            ),
        )
        score_artifacts = {}
        for method, scores in score_files:
            relative_path = f"{method}/scores.csv"
            path = staging / relative_path
            _write_scores_csv(path, method=method, scores=scores)
            score_artifacts[str(method)] = {
                "relative_path": relative_path,
                "record_count": len(scores),
                "sha256": sha256_file(path),
            }

        checkpoint = {
            "schema_version": 1,
            "contract_version": CALIBRATION_CONTRACT_VERSION,
            "checkpoint_id": CALIBRATION_CHECKPOINT_ID,
            "status": CALIBRATION_STATUS,
            "source_commit": state.source_commit,
            "freeze_checkpoint_sha256": state.freeze_checkpoint_sha256,
            "config_sha256": state.config_sha256,
            "normal_partition_manifest_sha256": state.partition_manifest_sha256,
            "dataset_integrity_record_sha256": state.dataset_integrity_sha256,
            "reference": {
                "count": len(state.reference_paths),
                "paths": list(state.reference_paths),
            },
            "calibration": {
                "count": len(state.calibration_paths),
                "normal_only": True,
                "anomaly_labels_used": False,
                "final_test_paths_used": False,
            },
            "methods": {
                str(CalibrationMethod.ECC_RESIDUAL): {
                    "fit": _ecc_fit_summary(state.ecc_fit),
                    "threshold_calibration": _calibration_summary(
                        state.ecc_calibration
                    ),
                    "score_artifact": score_artifacts[
                        str(CalibrationMethod.ECC_RESIDUAL)
                    ],
                },
                str(CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM): {
                    "fit": _hog_fit_summary(
                        state.hog_scaler_fit,
                        state.hog_model_fit,
                    ),
                    "threshold_calibration": _calibration_summary(
                        state.hog_calibration
                    ),
                    "score_artifact": score_artifacts[
                        str(CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM)
                    ],
                },
            },
            "local_state": {
                "logical_path": LOCAL_STATE_LOGICAL_PATH,
                "sha256": sha256_file(state_path),
                "format": f"Python pickle protocol {PICKLE_PROTOCOL}",
                "committed_to_git": False,
                "trusted_local_generation_only": True,
            },
            "evaluation_boundary": {
                "final_test_image_read": False,
                "final_test_scoring_started": False,
                "per_path_final_test_label_read": False,
                "final_test_label_join_performed": False,
                "metric_computed": False,
                "latency_measured": False,
                "decision_recorded": False,
                "threshold_rule_changed": False,
                "hard_gate_changed": False,
            },
        }
        write_json_atomic(staging / "normal-only-calibration.json", checkpoint)
        os.replace(staging, output_dir)
        return checkpoint
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if state_written:
            state_path.unlink(missing_ok=True)
        raise


def read_normal_calibration_checkpoint(path: Path) -> dict:
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NormalCalibrationRunError("cannot read normal calibration checkpoint") from error
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema_version") != 1
        or checkpoint.get("contract_version") != CALIBRATION_CONTRACT_VERSION
        or checkpoint.get("checkpoint_id") != CALIBRATION_CHECKPOINT_ID
        or checkpoint.get("status") != CALIBRATION_STATUS
        or not COMMIT_PATTERN.fullmatch(checkpoint.get("source_commit", ""))
    ):
        raise NormalCalibrationRunError("normal calibration checkpoint is invalid")
    return checkpoint


def load_normal_calibration_state(
    state_path: Path,
    *,
    checkpoint_path: Path,
    config: ProjectConfig,
) -> NormalCalibrationRunState:
    """Hash-check and load only a trusted locally generated state file."""
    checkpoint = read_normal_calibration_checkpoint(checkpoint_path)
    expected_sha256 = checkpoint.get("local_state", {}).get("sha256")
    if (
        not SHA256_PATTERN.fullmatch(expected_sha256 or "")
        or sha256_file(state_path) != expected_sha256
    ):
        raise NormalCalibrationRunError("local fitted state SHA-256 is invalid")
    try:
        with state_path.open("rb") as stream:
            state = pickle.load(stream)
    except (OSError, pickle.PickleError, AttributeError, EOFError) as error:
        raise NormalCalibrationRunError("cannot load local fitted state") from error
    validate_normal_calibration_state(state, config=config)
    assert isinstance(state, NormalCalibrationRunState)
    if (
        state.source_commit != checkpoint["source_commit"]
        or state.freeze_checkpoint_sha256 != checkpoint["freeze_checkpoint_sha256"]
        or state.config_sha256 != checkpoint["config_sha256"]
        or state.partition_manifest_sha256
        != checkpoint["normal_partition_manifest_sha256"]
        or state.dataset_integrity_sha256
        != checkpoint["dataset_integrity_record_sha256"]
        or state.ecc_calibration.threshold
        != checkpoint["methods"]["ecc_residual"]["threshold_calibration"]["threshold"]
        or state.hog_calibration.threshold
        != checkpoint["methods"]["patch_hog_one_class_svm"]["threshold_calibration"][
            "threshold"
        ]
    ):
        raise NormalCalibrationRunError("local fitted state differs from its checkpoint")
    return state


def run_normal_reference_fit_and_calibration(
    *,
    source_commit: str,
    archive_path: Path,
    dataset_root: Path,
    split_csv: Path,
    config_path: Path,
    freeze_checkpoint_path: Path,
    partition_manifest_path: Path,
    dataset_record_path: Path,
    dataset_integrity_path: Path,
    output_dir: Path,
    state_path: Path,
    config: ProjectConfig,
    progress: ProgressCallback | None = None,
) -> dict:
    """Verify fixed inputs, fit both methods, and freeze normal-only thresholds."""
    if not COMMIT_PATTERN.fullmatch(source_commit):
        raise NormalCalibrationRunError("source commit must be a full Git SHA")
    freeze = read_and_verify_pre_evaluation_freeze(
        freeze_checkpoint_path,
        project_root=config.project_root,
    )
    current_integrity = verify_visa_pcb1_integrity(
        archive_path=archive_path,
        dataset_root=dataset_root,
        split_csv=split_csv,
        dataset_record_path=dataset_record_path,
        category=config.category,
    )
    try:
        committed_integrity = json.loads(
            dataset_integrity_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise NormalCalibrationRunError("cannot read committed integrity record") from error
    if current_integrity != committed_integrity:
        raise NormalCalibrationRunError("current local asset differs from integrity checkpoint")
    _emit(progress, "local dataset integrity checkpoint matched")

    reference_paths, calibration_paths = load_fixed_normal_partitions(
        partition_manifest_path,
        expected_reference_paths=tuple(freeze["selection"]["reference_ids"]),
        expected_calibration_count=freeze["partitions"]["calibration_count"],
    )
    state = build_normal_calibration_state(
        source_commit=source_commit,
        freeze_checkpoint_sha256=sha256_file(freeze_checkpoint_path),
        config_sha256=sha256_file(config_path),
        partition_manifest_sha256=sha256_file(partition_manifest_path),
        dataset_integrity_sha256=sha256_file(dataset_integrity_path),
        reference_paths=reference_paths,
        calibration_paths=calibration_paths,
        dataset_root=dataset_root,
        config=config,
        progress=progress,
    )
    checkpoint = write_normal_calibration_outputs(
        state,
        output_dir=output_dir,
        state_path=state_path,
        config=config,
    )
    _emit(progress, "normal-only calibration checkpoint written")
    return checkpoint
