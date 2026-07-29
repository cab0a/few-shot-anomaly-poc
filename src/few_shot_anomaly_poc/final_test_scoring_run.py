"""Run the first fixed final-test scoring pass without class metadata."""

from __future__ import annotations

import csv
import hashlib
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

import cv2
import numpy as np

from few_shot_anomaly_poc.calibration import (
    CalibrationMethod,
    FixedThresholdBatchClassificationResult,
    NormalThresholdCalibrationResult,
    classify_fixed_threshold_batch,
)
from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.cpu_latency import (
    CPULatencyResult,
    cpu_latency_result_is_valid,
    measure_cpu_latency,
)
from few_shot_anomaly_poc.ecc_residual import (
    ECCResidualScoreResult,
    score_ecc_residual,
)
from few_shot_anomaly_poc.errors import ImagePreprocessingError
from few_shot_anomaly_poc.freeze_checkpoint import (
    read_and_verify_pre_evaluation_freeze,
)
from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.hog_scoring import (
    PatchHOGScoreResult,
    score_patch_hog,
)
from few_shot_anomaly_poc.jsonio import write_json_atomic
from few_shot_anomaly_poc.manifests import normalize_relative_path
from few_shot_anomaly_poc.normal_calibration_run import (
    NormalCalibrationRunState,
    load_normal_calibration_state,
)
from few_shot_anomaly_poc.preprocessing import (
    DECODE_FLAGS,
    preprocess_decoded_image,
)

SCORING_CONTRACT_VERSION = "first-fixed-final-test-scoring/v0.1"
SCORING_RUN_ID = "visa-pcb1-v0-1-first-fixed"
SCORING_STATUS = "SCORES_CLASSIFICATIONS_AND_LATENCY_FIXED"
EXPECTED_FINAL_TEST_COUNT = 200
LOCAL_STATE_LOGICAL_PATH = "work/v0.1/final-test/first-fixed-scoring-state.pkl"
PICKLE_PROTOCOL = 5
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FINAL_TEST_RECORD_KEYS = {
    "id",
    "schema_version",
    "partition",
    "category",
    "relative_path",
    "source_split",
    "source_row",
}
FINAL_TEST_ACCESS_POLICY = {
    "stage": "manifest_only",
    "image_content_reading": False,
    "image_display": False,
    "class_label_exposure": False,
    "score_calculation": False,
    "statistics": False,
    "parameter_selection": False,
}
SCORE_COLUMNS = (
    "contract_version",
    "run_id",
    "method",
    "partition",
    "relative_path",
    "score_status",
    "score_failure_code",
    "anomaly_score",
    "diagnostics_json",
)
CLASSIFICATION_COLUMNS = (
    "contract_version",
    "run_id",
    "method",
    "relative_path",
    "score_status",
    "score_failure_code",
    "anomaly_score",
    "threshold",
    "threshold_source_path",
    "calibration_sample_count",
    "calibration_rank",
    "predicted_class",
    "is_anomalous",
    "decision_reason",
    "score_margin",
)
LATENCY_OBSERVATION_COLUMNS = (
    "contract_version",
    "run_id",
    "method",
    "pass_index",
    "relative_path",
    "duration_ns",
    "score_status",
    "score_failure_code",
)

type FinalTestScore = ECCResidualScoreResult | PatchHOGScoreResult
type ProgressCallback = Callable[[str], None]


class FirstFixedScoringRunError(Exception):
    """Reject a changed input boundary or an incomplete scoring run."""


@dataclass(frozen=True)
class UnlabeledFinalTestManifest:
    """Validated final-test paths with no class or pixel metadata."""

    manifest_set_sha256: str
    final_test_manifest_sha256: str
    archive_sha256: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class FirstFixedScoringRunState:
    """Local score, classification, and timing objects for final evaluation."""

    contract_version: str
    run_id: str
    source_commit: str
    freeze_checkpoint_sha256: str
    config_sha256: str
    dataset_integrity_sha256: str
    calibration_checkpoint_sha256: str
    calibration_state_sha256: str
    manifest_set_sha256: str
    final_test_manifest_sha256: str
    final_test_paths: tuple[str, ...]
    ecc_scores: dict[str, ECCResidualScoreResult]
    hog_scores: dict[str, PatchHOGScoreResult]
    ecc_classifications: FixedThresholdBatchClassificationResult
    hog_classifications: FixedThresholdBatchClassificationResult
    ecc_latency: CPULatencyResult
    hog_latency: CPULatencyResult


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _record_id(category: str, relative_path: str) -> str:
    suffix = hashlib.sha256(relative_path.encode()).hexdigest()[:16]
    return f"visa-{category}-{suffix}"


def _read_json_object(path: Path, *, description: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FirstFixedScoringRunError(f"cannot read {description}") from error
    if not isinstance(value, dict):
        raise FirstFixedScoringRunError(f"{description} must contain one JSON object")
    return value


def load_unlabeled_final_test_manifest(
    manifest_set_path: Path,
    final_test_manifest_path: Path,
    *,
    config: ProjectConfig,
    expected_count: int = EXPECTED_FINAL_TEST_COUNT,
) -> UnlabeledFinalTestManifest:
    """Load only a hash-bound final-test path list that contains no class metadata."""
    metadata = _read_json_object(manifest_set_path, description="manifest set")
    dataset = metadata.get("dataset")
    split = metadata.get("official_split")
    inventory = metadata.get("manifests")
    if (
        metadata.get("schema_version") != 1
        or not isinstance(dataset, dict)
        or dataset.get("name") != config.dataset_name
        or dataset.get("category") != config.category
        or dataset.get("license") != config.dataset_license
        or dataset.get("archive_identifier") != config.archive.identifier
        or not SHA256_PATTERN.fullmatch(dataset.get("archive_sha256", ""))
        or not isinstance(split, dict)
        or split.get("repository") != config.split.repository
        or split.get("revision") != config.split.revision
        or split.get("path") != config.split.path
        or split.get("sha256") != config.split.sha256
        or metadata.get("selection") != asdict(config.selection)
        or metadata.get("final_test_access_policy") != FINAL_TEST_ACCESS_POLICY
        or not isinstance(inventory, dict)
    ):
        raise FirstFixedScoringRunError("manifest set does not match the fixed input")

    final_inventory = inventory.get("final-test")
    if (
        not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count < 1
        or not isinstance(final_inventory, dict)
        or final_inventory.get("file") != final_test_manifest_path.name
        or final_inventory.get("record_count") != expected_count
        or not SHA256_PATTERN.fullmatch(final_inventory.get("sha256", ""))
        or sha256_file(final_test_manifest_path) != final_inventory.get("sha256")
    ):
        raise FirstFixedScoringRunError("final-test manifest inventory is invalid")

    records = []
    try:
        with final_test_manifest_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise FirstFixedScoringRunError(
                        f"final-test manifest line {line_number} is invalid"
                    ) from error
                if not isinstance(record, dict) or set(record) != FINAL_TEST_RECORD_KEYS:
                    raise FirstFixedScoringRunError(
                        "final-test manifest exposes unexpected metadata"
                    )
                records.append(record)
    except OSError as error:
        raise FirstFixedScoringRunError("cannot read final-test manifest") from error

    paths = []
    seen_ids: set[str] = set()
    seen_rows: set[int] = set()
    for record in records:
        relative_path = record.get("relative_path")
        source_row = record.get("source_row")
        try:
            normalized_path = normalize_relative_path(relative_path)
        except Exception as error:
            raise FirstFixedScoringRunError(
                "final-test manifest contains an invalid path"
            ) from error
        if (
            normalized_path != relative_path
            or not relative_path.startswith(f"{config.category}/Data/Images/")
            or record.get("schema_version") != 1
            or record.get("partition") != "final-test"
            or record.get("category") != config.category
            or record.get("source_split") != "test"
            or not isinstance(source_row, int)
            or isinstance(source_row, bool)
            or source_row < 2
            or record.get("id") != _record_id(config.category, relative_path)
            or record["id"] in seen_ids
            or source_row in seen_rows
        ):
            raise FirstFixedScoringRunError("final-test manifest record is invalid")
        paths.append(relative_path)
        seen_ids.add(record["id"])
        seen_rows.add(source_row)

    ordered_paths = tuple(paths)
    if (
        len(ordered_paths) != expected_count
        or len(set(ordered_paths)) != expected_count
        or ordered_paths != tuple(sorted(ordered_paths))
    ):
        raise FirstFixedScoringRunError(
            "final-test manifest paths are incomplete, duplicated, or out of order"
        )
    return UnlabeledFinalTestManifest(
        manifest_set_sha256=sha256_file(manifest_set_path),
        final_test_manifest_sha256=sha256_file(final_test_manifest_path),
        archive_sha256=dataset["archive_sha256"],
        paths=ordered_paths,
    )


def _decode_grayscale(path: Path) -> np.ndarray:
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise FirstFixedScoringRunError(f"cannot read final-test asset: {path.name}") from error
    if not encoded:
        raise FirstFixedScoringRunError(f"final-test asset is empty: {path.name}")
    try:
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), DECODE_FLAGS)
    except cv2.error as error:
        raise FirstFixedScoringRunError(
            f"cannot decode final-test asset: {path.name}"
        ) from error
    if (
        not isinstance(decoded, np.ndarray)
        or decoded.ndim != 2
        or decoded.size == 0
        or decoded.dtype != np.uint8
    ):
        raise FirstFixedScoringRunError(
            f"final-test asset did not decode to grayscale uint8: {path.name}"
        )
    return decoded


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


def _ecc_scorer(
    calibration_state: NormalCalibrationRunState,
    *,
    config: ProjectConfig,
) -> Callable[[object], ECCResidualScoreResult]:
    def score_one(decoded: object) -> ECCResidualScoreResult:
        try:
            image = preprocess_decoded_image(decoded, config.preprocessing)
        except ImagePreprocessingError as error:
            return _ecc_preprocessing_failure(error, config=config)
        return score_ecc_residual(image, fitted=calibration_state.ecc_fit, config=config)

    return score_one


def _hog_scorer(
    calibration_state: NormalCalibrationRunState,
    *,
    config: ProjectConfig,
) -> Callable[[object], PatchHOGScoreResult]:
    def score_one(decoded: object) -> PatchHOGScoreResult:
        try:
            image = preprocess_decoded_image(decoded, config.preprocessing)
        except ImagePreprocessingError as error:
            return _hog_preprocessing_failure(error, config=config)
        return score_patch_hog(
            image,
            scaler_fit=calibration_state.hog_scaler_fit,
            model_fit=calibration_state.hog_model_fit,
            config=config,
        )

    return score_one


def _score_paths(
    decoded_images: Mapping[str, object],
    score_one: Callable[[object], FinalTestScore],
    *,
    progress: ProgressCallback | None,
    method: CalibrationMethod,
) -> dict[str, FinalTestScore]:
    results = {}
    for index, path in enumerate(sorted(decoded_images), start=1):
        results[path] = score_one(decoded_images[path])
        if index % 25 == 0 or index == len(decoded_images):
            _emit(progress, f"{method} fixed scoring: {index}/{len(decoded_images)}")
    return results


def validate_first_fixed_scoring_state(
    state: object,
    *,
    ecc_calibration: NormalThresholdCalibrationResult,
    hog_calibration: NormalThresholdCalibrationResult,
    config: ProjectConfig,
    expected_count: int = EXPECTED_FINAL_TEST_COUNT,
) -> None:
    """Reject incomplete, changed, or internally inconsistent scoring evidence."""
    if (
        not isinstance(state, FirstFixedScoringRunState)
        or state.contract_version != SCORING_CONTRACT_VERSION
        or state.run_id != SCORING_RUN_ID
        or not COMMIT_PATTERN.fullmatch(state.source_commit)
        or any(
            not SHA256_PATTERN.fullmatch(value)
            for value in (
                state.freeze_checkpoint_sha256,
                state.config_sha256,
                state.dataset_integrity_sha256,
                state.calibration_checkpoint_sha256,
                state.calibration_state_sha256,
                state.manifest_set_sha256,
                state.final_test_manifest_sha256,
            )
        )
        or len(state.final_test_paths) != expected_count
        or state.final_test_paths != tuple(sorted(state.final_test_paths))
        or len(set(state.final_test_paths)) != expected_count
        or tuple(state.ecc_scores) != state.final_test_paths
        or tuple(state.hog_scores) != state.final_test_paths
    ):
        raise FirstFixedScoringRunError("first fixed scoring state metadata is invalid")

    expected_ecc = classify_fixed_threshold_batch(
        state.ecc_scores,
        calibration=ecc_calibration,
        config=config,
    )
    expected_hog = classify_fixed_threshold_batch(
        state.hog_scores,
        calibration=hog_calibration,
        config=config,
    )
    if (
        not expected_ecc.succeeded
        or not expected_hog.succeeded
        or state.ecc_classifications != expected_ecc
        or state.hog_classifications != expected_hog
        or not cpu_latency_result_is_valid(state.ecc_latency, config=config)
        or not cpu_latency_result_is_valid(state.hog_latency, config=config)
        or state.ecc_latency.method is not CalibrationMethod.ECC_RESIDUAL
        or state.hog_latency.method is not CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM
        or state.ecc_latency.ordered_paths != state.final_test_paths
        or state.hog_latency.ordered_paths != state.final_test_paths
    ):
        raise FirstFixedScoringRunError("first fixed scoring outputs are invalid")


def build_first_fixed_scoring_state(
    *,
    source_commit: str,
    freeze_checkpoint_sha256: str,
    config_sha256: str,
    dataset_integrity_sha256: str,
    calibration_checkpoint_sha256: str,
    calibration_state_sha256: str,
    manifest: UnlabeledFinalTestManifest,
    dataset_root: Path,
    calibration_state: NormalCalibrationRunState,
    config: ProjectConfig,
    progress: ProgressCallback | None = None,
    expected_count: int = EXPECTED_FINAL_TEST_COUNT,
) -> FirstFixedScoringRunState:
    """Decode, score, classify, and time the fixed test assets without class metadata."""
    if not COMMIT_PATTERN.fullmatch(source_commit):
        raise FirstFixedScoringRunError("source commit must be a full Git SHA")

    decoded_images = {}
    for index, relative_path in enumerate(manifest.paths, start=1):
        decoded_images[relative_path] = _decode_grayscale(dataset_root / relative_path)
        if index % 25 == 0 or index == len(manifest.paths):
            _emit(progress, f"final-test decode: {index}/{len(manifest.paths)}")

    ecc_score_one = _ecc_scorer(calibration_state, config=config)
    hog_score_one = _hog_scorer(calibration_state, config=config)
    ecc_scores = _score_paths(
        decoded_images,
        ecc_score_one,
        progress=progress,
        method=CalibrationMethod.ECC_RESIDUAL,
    )
    hog_scores = _score_paths(
        decoded_images,
        hog_score_one,
        progress=progress,
        method=CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM,
    )
    ecc_classifications = classify_fixed_threshold_batch(
        ecc_scores,
        calibration=calibration_state.ecc_calibration,
        config=config,
    )
    hog_classifications = classify_fixed_threshold_batch(
        hog_scores,
        calibration=calibration_state.hog_calibration,
        config=config,
    )
    if not ecc_classifications.succeeded or not hog_classifications.succeeded:
        raise FirstFixedScoringRunError("fixed-threshold batch classification failed")

    _emit(progress, "ECC CPU latency measurement started")
    ecc_latency = measure_cpu_latency(
        decoded_images,
        ecc_score_one,
        method=CalibrationMethod.ECC_RESIDUAL,
        config=config,
    )
    if not ecc_latency.succeeded:
        raise FirstFixedScoringRunError(
            f"ECC CPU latency measurement failed: {ecc_latency.failure_code}"
        )
    _emit(progress, "ECC CPU latency measurement complete")

    _emit(progress, "Patch HOG CPU latency measurement started")
    hog_latency = measure_cpu_latency(
        decoded_images,
        hog_score_one,
        method=CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM,
        config=config,
    )
    if not hog_latency.succeeded:
        raise FirstFixedScoringRunError(
            f"Patch HOG CPU latency measurement failed: {hog_latency.failure_code}"
        )
    _emit(progress, "Patch HOG CPU latency measurement complete")

    state = FirstFixedScoringRunState(
        contract_version=SCORING_CONTRACT_VERSION,
        run_id=SCORING_RUN_ID,
        source_commit=source_commit,
        freeze_checkpoint_sha256=freeze_checkpoint_sha256,
        config_sha256=config_sha256,
        dataset_integrity_sha256=dataset_integrity_sha256,
        calibration_checkpoint_sha256=calibration_checkpoint_sha256,
        calibration_state_sha256=calibration_state_sha256,
        manifest_set_sha256=manifest.manifest_set_sha256,
        final_test_manifest_sha256=manifest.final_test_manifest_sha256,
        final_test_paths=manifest.paths,
        ecc_scores=ecc_scores,
        hog_scores=hog_scores,
        ecc_classifications=ecc_classifications,
        hog_classifications=hog_classifications,
        ecc_latency=ecc_latency,
        hog_latency=hog_latency,
    )
    validate_first_fixed_scoring_state(
        state,
        ecc_calibration=calibration_state.ecc_calibration,
        hog_calibration=calibration_state.hog_calibration,
        config=config,
        expected_count=expected_count,
    )
    return state


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
    raise FirstFixedScoringRunError(
        f"unsupported artifact value: {type(value).__name__}"
    )


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            _json_ready(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise FirstFixedScoringRunError("artifact value is not finite JSON") from error


def _score_diagnostics(score: FinalTestScore) -> dict:
    values = asdict(score)
    for field in ("score_status", "failure_code", "anomaly_score"):
        values.pop(field)
    return values


def _write_csv(
    path: Path,
    *,
    columns: tuple[str, ...],
    rows: tuple[dict[str, object], ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_method_artifacts(
    method_dir: Path,
    *,
    method: CalibrationMethod,
    scores: Mapping[str, FinalTestScore],
    classifications: FixedThresholdBatchClassificationResult,
    latency: CPULatencyResult,
) -> dict:
    score_path = method_dir / "scores.csv"
    classification_path = method_dir / "classifications.csv"
    latency_path = method_dir / "latency.json"
    latency_observations_path = method_dir / "latency-observations.csv"
    score_rows = tuple(
        {
            "contract_version": SCORING_CONTRACT_VERSION,
            "run_id": SCORING_RUN_ID,
            "method": str(method),
            "partition": "final_test",
            "relative_path": path,
            "score_status": score.score_status,
            "score_failure_code": (
                str(score.failure_code) if score.failure_code is not None else ""
            ),
            "anomaly_score": score.anomaly_score,
            "diagnostics_json": _canonical_json(_score_diagnostics(score)),
        }
        for path, score in scores.items()
    )
    assert classifications.classifications is not None
    classification_rows = tuple(
        {
            "contract_version": SCORING_CONTRACT_VERSION,
            "run_id": SCORING_RUN_ID,
            "method": str(method),
            "relative_path": item.relative_path,
            "score_status": item.score_status,
            "score_failure_code": item.score_failure_code or "",
            "anomaly_score": item.anomaly_score,
            "threshold": item.threshold,
            "threshold_source_path": item.threshold_source_path,
            "calibration_sample_count": item.calibration_sample_count,
            "calibration_rank": item.calibration_rank,
            "predicted_class": item.predicted_class,
            "is_anomalous": str(item.is_anomalous).lower(),
            "decision_reason": item.decision_reason,
            "score_margin": item.score_margin,
        }
        for item in classifications.classifications
    )
    assert latency.observations is not None
    latency_observation_rows = tuple(
        {
            "contract_version": SCORING_CONTRACT_VERSION,
            "run_id": SCORING_RUN_ID,
            "method": str(method),
            "pass_index": item.pass_index,
            "relative_path": item.relative_path,
            "duration_ns": item.duration_ns,
            "score_status": item.score_status,
            "score_failure_code": item.score_failure_code or "",
        }
        for item in latency.observations
    )
    _write_csv(score_path, columns=SCORE_COLUMNS, rows=score_rows)
    _write_csv(
        classification_path,
        columns=CLASSIFICATION_COLUMNS,
        rows=classification_rows,
    )
    latency_record = asdict(latency)
    latency_record.pop("observations")
    write_json_atomic(latency_path, _json_ready(latency_record))
    _write_csv(
        latency_observations_path,
        columns=LATENCY_OBSERVATION_COLUMNS,
        rows=latency_observation_rows,
    )
    return {
        "score": {
            "relative_path": f"{method}/scores.csv",
            "record_count": len(score_rows),
            "sha256": sha256_file(score_path),
        },
        "classification": {
            "relative_path": f"{method}/classifications.csv",
            "record_count": len(classification_rows),
            "sha256": sha256_file(classification_path),
        },
        "latency": {
            "relative_path": f"{method}/latency.json",
            "sha256": sha256_file(latency_path),
            "observation_relative_path": f"{method}/latency-observations.csv",
            "observation_record_count": len(latency_observation_rows),
            "observation_sha256": sha256_file(latency_observations_path),
        },
    }


def _write_pickle_atomic(path: Path, state: FirstFixedScoringRunState) -> None:
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


def write_first_fixed_scoring_outputs(
    state: FirstFixedScoringRunState,
    *,
    output_dir: Path,
    state_path: Path,
    ecc_calibration: NormalThresholdCalibrationResult,
    hog_calibration: NormalThresholdCalibrationResult,
    config: ProjectConfig,
    expected_count: int = EXPECTED_FINAL_TEST_COUNT,
) -> dict:
    """Write non-overwritable score, classification, latency, and local state."""
    validate_first_fixed_scoring_state(
        state,
        ecc_calibration=ecc_calibration,
        hog_calibration=hog_calibration,
        config=config,
        expected_count=expected_count,
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if state_path.exists():
        raise FileExistsError(f"refusing to overwrite {state_path}")

    _write_pickle_atomic(state_path, state)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}.")
    )
    try:
        methods = {}
        for method, scores, classifications, latency in (
            (
                CalibrationMethod.ECC_RESIDUAL,
                state.ecc_scores,
                state.ecc_classifications,
                state.ecc_latency,
            ),
            (
                CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM,
                state.hog_scores,
                state.hog_classifications,
                state.hog_latency,
            ),
        ):
            artifacts = _write_method_artifacts(
                staging / str(method),
                method=method,
                scores=scores,
                classifications=classifications,
                latency=latency,
            )
            methods[str(method)] = {
                "threshold": classifications.threshold,
                "threshold_source_path": classifications.threshold_source_path,
                "item_count": classifications.item_count,
                "predicted_normal_count": classifications.normal_count,
                "predicted_anomalous_count": classifications.anomalous_count,
                "score_failure_count": classifications.score_failure_count,
                "median_latency_seconds": latency.median_latency_seconds,
                "p95_latency_seconds": latency.p95_latency_seconds,
                "artifacts": artifacts,
            }

        checkpoint = {
            "schema_version": 1,
            "contract_version": SCORING_CONTRACT_VERSION,
            "run_id": SCORING_RUN_ID,
            "status": SCORING_STATUS,
            "source_commit": state.source_commit,
            "freeze_checkpoint_sha256": state.freeze_checkpoint_sha256,
            "config_sha256": state.config_sha256,
            "dataset_integrity_record_sha256": state.dataset_integrity_sha256,
            "normal_calibration_checkpoint_sha256": (
                state.calibration_checkpoint_sha256
            ),
            "normal_calibration_state_sha256": state.calibration_state_sha256,
            "manifest_set_sha256": state.manifest_set_sha256,
            "final_test_manifest_sha256": state.final_test_manifest_sha256,
            "final_test_item_count": len(state.final_test_paths),
            "methods": methods,
            "local_state": {
                "logical_path": LOCAL_STATE_LOGICAL_PATH,
                "sha256": sha256_file(state_path),
                "format": f"Python pickle protocol {PICKLE_PROTOCOL}",
                "committed_to_git": False,
                "trusted_local_generation_only": True,
            },
            "evaluation_boundary": {
                "final_test_images_decoded": True,
                "final_test_scoring_completed": True,
                "fixed_threshold_classification_completed": True,
                "cpu_latency_measured": True,
                "per_path_final_test_class_read": False,
                "final_test_class_join_performed": False,
                "metric_computed": False,
                "failure_case_selected": False,
                "decision_recorded": False,
                "image_displayed": False,
                "threshold_rule_changed": False,
                "hard_gate_changed": False,
            },
        }
        write_json_atomic(staging / "first-fixed-scoring.json", checkpoint)
        os.replace(staging, output_dir)
        return checkpoint
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        state_path.unlink(missing_ok=True)
        raise


def load_first_fixed_scoring_state(
    state_path: Path,
    *,
    checkpoint_path: Path,
    ecc_calibration: NormalThresholdCalibrationResult,
    hog_calibration: NormalThresholdCalibrationResult,
    config: ProjectConfig,
    expected_count: int = EXPECTED_FINAL_TEST_COUNT,
) -> FirstFixedScoringRunState:
    """Hash-check and load only the locally generated first-run state."""
    checkpoint = _read_json_object(
        checkpoint_path,
        description="first fixed scoring checkpoint",
    )
    expected_sha256 = checkpoint.get("local_state", {}).get("sha256")
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("contract_version") != SCORING_CONTRACT_VERSION
        or checkpoint.get("run_id") != SCORING_RUN_ID
        or checkpoint.get("status") != SCORING_STATUS
        or not SHA256_PATTERN.fullmatch(expected_sha256 or "")
        or sha256_file(state_path) != expected_sha256
    ):
        raise FirstFixedScoringRunError("first fixed scoring state SHA-256 is invalid")
    try:
        with state_path.open("rb") as stream:
            state = pickle.load(stream)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError) as error:
        raise FirstFixedScoringRunError("cannot load first fixed scoring state") from error
    validate_first_fixed_scoring_state(
        state,
        ecc_calibration=ecc_calibration,
        hog_calibration=hog_calibration,
        config=config,
        expected_count=expected_count,
    )
    if (
        checkpoint.get("source_commit") != state.source_commit
        or checkpoint.get("freeze_checkpoint_sha256")
        != state.freeze_checkpoint_sha256
        or checkpoint.get("config_sha256") != state.config_sha256
        or checkpoint.get("dataset_integrity_record_sha256")
        != state.dataset_integrity_sha256
        or checkpoint.get("normal_calibration_checkpoint_sha256")
        != state.calibration_checkpoint_sha256
        or checkpoint.get("normal_calibration_state_sha256")
        != state.calibration_state_sha256
        or checkpoint.get("manifest_set_sha256") != state.manifest_set_sha256
        or checkpoint.get("final_test_manifest_sha256")
        != state.final_test_manifest_sha256
        or checkpoint.get("final_test_item_count") != len(state.final_test_paths)
    ):
        raise FirstFixedScoringRunError(
            "first fixed scoring checkpoint does not match local state"
        )
    return state


def run_first_fixed_final_test_scoring(
    *,
    source_commit: str,
    dataset_root: Path,
    config_path: Path,
    freeze_checkpoint_path: Path,
    dataset_integrity_path: Path,
    calibration_checkpoint_path: Path,
    calibration_state_path: Path,
    manifest_set_path: Path,
    final_test_manifest_path: Path,
    output_dir: Path,
    scoring_state_path: Path,
    config: ProjectConfig,
    progress: ProgressCallback | None = None,
) -> dict:
    """Execute the first score-and-latency run without accepting class metadata."""
    freeze = read_and_verify_pre_evaluation_freeze(
        freeze_checkpoint_path,
        project_root=config.project_root,
    )
    if freeze["boundary_state"]["final_test_scoring_started"]:
        raise FirstFixedScoringRunError("freeze boundary already reports scoring started")

    calibration_state = load_normal_calibration_state(
        calibration_state_path,
        checkpoint_path=calibration_checkpoint_path,
        config=config,
    )
    freeze_sha256 = sha256_file(freeze_checkpoint_path)
    config_sha256 = sha256_file(config_path)
    dataset_integrity_sha256 = sha256_file(dataset_integrity_path)
    calibration_state_sha256 = sha256_file(calibration_state_path)
    if (
        calibration_state.freeze_checkpoint_sha256 != freeze_sha256
        or calibration_state.config_sha256 != config_sha256
        or calibration_state.dataset_integrity_sha256 != dataset_integrity_sha256
    ):
        raise FirstFixedScoringRunError(
            "calibration state does not match the frozen scoring inputs"
        )

    manifest = load_unlabeled_final_test_manifest(
        manifest_set_path,
        final_test_manifest_path,
        config=config,
    )
    state = build_first_fixed_scoring_state(
        source_commit=source_commit,
        freeze_checkpoint_sha256=freeze_sha256,
        config_sha256=config_sha256,
        dataset_integrity_sha256=dataset_integrity_sha256,
        calibration_checkpoint_sha256=sha256_file(calibration_checkpoint_path),
        calibration_state_sha256=calibration_state_sha256,
        manifest=manifest,
        dataset_root=dataset_root,
        calibration_state=calibration_state,
        config=config,
        progress=progress,
    )
    checkpoint = write_first_fixed_scoring_outputs(
        state,
        output_dir=output_dir,
        state_path=scoring_state_path,
        ecc_calibration=calibration_state.ecc_calibration,
        hog_calibration=calibration_state.hog_calibration,
        config=config,
    )
    _emit(progress, "first fixed scoring checkpoint written")
    return checkpoint
