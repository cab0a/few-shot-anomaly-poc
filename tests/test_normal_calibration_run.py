from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from few_shot_anomaly_poc.calibration import (
    CalibrationMethod,
    calibrate_normal_threshold,
)
from few_shot_anomaly_poc.config import ProjectConfig, load_config
from few_shot_anomaly_poc.ecc_residual import ECCResidualScoreResult
from few_shot_anomaly_poc.ecc_template import (
    ECCTemplateFitResult,
    ReferenceRegistrationDiagnostic,
)
from few_shot_anomaly_poc.hog_features import (
    PatchHOGFeatureResult,
    fixed_patch_positions,
)
from few_shot_anomaly_poc.hog_models import fit_position_one_class_svms
from few_shot_anomaly_poc.hog_scalers import fit_position_scalers
from few_shot_anomaly_poc.hog_scoring import PatchHOGScoreResult
from few_shot_anomaly_poc.normal_calibration_run import (
    CALIBRATION_CHECKPOINT_ID,
    CALIBRATION_CONTRACT_VERSION,
    NormalCalibrationRunError,
    NormalCalibrationRunState,
    load_fixed_normal_partitions,
    load_normal_calibration_state,
    write_normal_calibration_outputs,
)


@pytest.fixture(scope="module")
def project_config() -> ProjectConfig:
    return load_config(Path("configs/v0.1.yaml"))


def _reference_paths() -> tuple[str, ...]:
    return tuple(f"pcb1/Data/Images/Normal/{index:04d}.JPG" for index in range(20))


def _calibration_paths() -> tuple[str, ...]:
    return tuple(f"pcb1/Data/Images/Normal/{index:04d}.JPG" for index in range(20, 904))


def _ecc_fit() -> ECCTemplateFitResult:
    reference_paths = _reference_paths()
    diagnostics = tuple(
        ReferenceRegistrationDiagnostic(
            relative_path=path,
            is_anchor=index == 0,
            status="ok",
            failure_code=None,
            correlation=None if index == 0 else 1.0,
            warp_matrix=np.eye(2, 3, dtype=np.float32),
            rotation_degrees=0.0,
            translation_x_pixels=0.0,
            translation_y_pixels=0.0,
            valid_fraction=1.0,
        )
        for index, path in enumerate(reference_paths)
    )
    return ECCTemplateFitResult(
        status="ok",
        failure_code=None,
        anchor_path=reference_paths[0],
        reference_count=20,
        successful_reference_count=20,
        failed_reference_count=0,
        support_fraction=1.0,
        template=np.zeros((512, 512), dtype=np.float32),
        support_mask=np.ones((512, 512), dtype=bool),
        reference_diagnostics=diagnostics,
    )


def _hog_fits(project_config: ProjectConfig):
    positions = fixed_patch_positions(project_config)
    assert positions is not None
    references = {}
    for index, path in enumerate(_reference_paths()):
        generator = np.random.default_rng(900 + index)
        references[path] = PatchHOGFeatureResult(
            status="ok",
            failure_code=None,
            features=generator.random(
                (
                    project_config.patch_hog.patch_count,
                    project_config.patch_hog.descriptor_length,
                ),
                dtype=np.float32,
            ),
            positions=positions,
            failed_patch_index=None,
        )
    scaler_fit = fit_position_scalers(references, config=project_config)
    assert scaler_fit.succeeded
    model_fit = fit_position_one_class_svms(
        references,
        scaler_fit=scaler_fit,
        config=project_config,
    )
    assert model_fit.succeeded
    return scaler_fit, model_fit


def _ecc_score(value: float) -> ECCResidualScoreResult:
    return ECCResidualScoreResult(
        score_status="ok",
        failure_code=None,
        anomaly_score=value,
        registration_status="ok",
        correlation=1.0,
        warp_matrix=np.eye(2, 3, dtype=np.float32),
        rotation_degrees=0.0,
        translation_x_pixels=0.0,
        translation_y_pixels=0.0,
        registration_valid_fraction=1.0,
        effective_support_fraction=1.0,
        effective_pixel_count=262_144,
        top_pixel_count=2_622,
    )


def _hog_score(value: float) -> PatchHOGScoreResult:
    return PatchHOGScoreResult(
        score_status="ok",
        failure_code=None,
        anomaly_score=value,
        patch_anomaly_scores=None,
        top_patch_count=None,
        top_patch_indices=(),
        successful_patch_count=225,
        failed_patch_index=None,
    )


@pytest.fixture(scope="module")
def calibration_state(project_config: ProjectConfig) -> NormalCalibrationRunState:
    calibration_paths = _calibration_paths()
    ecc_scores = {
        path: _ecc_score(index / 1000)
        for index, path in enumerate(calibration_paths)
    }
    hog_scores = {
        path: _hog_score(float(index - 442))
        for index, path in enumerate(calibration_paths)
    }
    ecc_calibration = calibrate_normal_threshold(
        ecc_scores,
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )
    hog_calibration = calibrate_normal_threshold(
        hog_scores,
        method=CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM,
        config=project_config,
    )
    assert ecc_calibration.succeeded and hog_calibration.succeeded
    scaler_fit, model_fit = _hog_fits(project_config)
    return NormalCalibrationRunState(
        contract_version=CALIBRATION_CONTRACT_VERSION,
        checkpoint_id=CALIBRATION_CHECKPOINT_ID,
        source_commit="a" * 40,
        freeze_checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
        partition_manifest_sha256="d" * 64,
        dataset_integrity_sha256="e" * 64,
        reference_paths=_reference_paths(),
        calibration_paths=calibration_paths,
        ecc_fit=_ecc_fit(),
        hog_scaler_fit=scaler_fit,
        hog_model_fit=model_fit,
        ecc_calibration_scores=ecc_scores,
        hog_calibration_scores=hog_scores,
        ecc_calibration=ecc_calibration,
        hog_calibration=hog_calibration,
    )


def _write_partition(path: Path, *, duplicate_last: bool = False) -> None:
    paths = (*_reference_paths(), *_calibration_paths())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "partition",
                "selection_rank",
                "relative_path",
                "selection_sha256",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for index, relative_path in enumerate(paths, start=1):
            if duplicate_last and index == len(paths):
                relative_path = paths[-2]
            writer.writerow(
                {
                    "partition": "reference" if index <= 20 else "calibration",
                    "selection_rank": index,
                    "relative_path": relative_path,
                    "selection_sha256": "f" * 64,
                }
            )


def test_fixed_partition_loader_requires_exact_frozen_lists(tmp_path: Path) -> None:
    path = tmp_path / "partitions.csv"
    _write_partition(path)

    reference, calibration = load_fixed_normal_partitions(
        path,
        expected_reference_paths=_reference_paths(),
        expected_calibration_count=884,
    )

    assert reference == _reference_paths()
    assert calibration == _calibration_paths()


def test_fixed_partition_loader_rejects_duplicate_path(tmp_path: Path) -> None:
    path = tmp_path / "partitions.csv"
    _write_partition(path, duplicate_last=True)

    with pytest.raises(NormalCalibrationRunError, match="row is invalid"):
        load_fixed_normal_partitions(
            path,
            expected_reference_paths=_reference_paths(),
            expected_calibration_count=884,
        )


def test_outputs_preserve_scores_and_hash_checked_local_state(
    tmp_path: Path,
    project_config: ProjectConfig,
    calibration_state: NormalCalibrationRunState,
) -> None:
    output_dir = tmp_path / "normal-only"
    state_path = tmp_path / "state.pkl"

    checkpoint = write_normal_calibration_outputs(
        calibration_state,
        output_dir=output_dir,
        state_path=state_path,
        config=project_config,
    )
    loaded = load_normal_calibration_state(
        state_path,
        checkpoint_path=output_dir / "normal-only-calibration.json",
        config=project_config,
    )

    assert loaded.source_commit == calibration_state.source_commit
    assert loaded.ecc_calibration == calibration_state.ecc_calibration
    assert loaded.hog_calibration == calibration_state.hog_calibration
    assert checkpoint["status"] == "THRESHOLDS_FIXED_BEFORE_FINAL_TEST"
    assert checkpoint["calibration"] == {
        "count": 884,
        "normal_only": True,
        "anomaly_labels_used": False,
        "final_test_paths_used": False,
    }
    assert checkpoint["evaluation_boundary"]["final_test_scoring_started"] is False
    assert checkpoint["evaluation_boundary"]["metric_computed"] is False
    for method in ("ecc_residual", "patch_hog_one_class_svm"):
        score_path = output_dir / method / "scores.csv"
        rows = tuple(csv.DictReader(score_path.open(encoding="utf-8", newline="")))
        assert len(rows) == 884
        assert all(row["partition"] == "calibration" for row in rows)
        assert all("label" not in row for row in rows)
        assert checkpoint["methods"][method]["score_artifact"]["record_count"] == 884


def test_local_state_hash_mismatch_is_rejected(
    tmp_path: Path,
    project_config: ProjectConfig,
    calibration_state: NormalCalibrationRunState,
) -> None:
    output_dir = tmp_path / "normal-only"
    state_path = tmp_path / "state.pkl"
    write_normal_calibration_outputs(
        calibration_state,
        output_dir=output_dir,
        state_path=state_path,
        config=project_config,
    )
    with state_path.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(NormalCalibrationRunError, match="SHA-256"):
        load_normal_calibration_state(
            state_path,
            checkpoint_path=output_dir / "normal-only-calibration.json",
            config=project_config,
        )


def test_public_checkpoint_is_deterministic_and_contains_no_absolute_path(
    tmp_path: Path,
    project_config: ProjectConfig,
    calibration_state: NormalCalibrationRunState,
) -> None:
    outputs = []
    for index in range(2):
        output_dir = tmp_path / f"output-{index}"
        state_path = tmp_path / f"state-{index}.pkl"
        write_normal_calibration_outputs(
            calibration_state,
            output_dir=output_dir,
            state_path=state_path,
            config=project_config,
        )
        outputs.append(
            (output_dir / "normal-only-calibration.json").read_bytes()
        )

    assert outputs[0] == outputs[1]
    checkpoint = json.loads(outputs[0])
    assert checkpoint["local_state"]["logical_path"].startswith("work/")
    assert str(tmp_path) not in outputs[0].decode()
