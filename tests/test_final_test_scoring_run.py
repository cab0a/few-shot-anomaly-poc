from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import few_shot_anomaly_poc.final_test_scoring_run as scoring_module
from few_shot_anomaly_poc.calibration import (
    CalibrationMethod,
    calibrate_normal_threshold,
    classify_fixed_threshold_batch,
)
from few_shot_anomaly_poc.config import ProjectConfig, load_config
from few_shot_anomaly_poc.cpu_latency import measure_cpu_latency
from few_shot_anomaly_poc.ecc_residual import ECCResidualScoreResult
from few_shot_anomaly_poc.final_test_scoring_run import (
    FINAL_TEST_ACCESS_POLICY,
    SCORING_CONTRACT_VERSION,
    SCORING_RUN_ID,
    FirstFixedScoringRunError,
    FirstFixedScoringRunState,
    UnlabeledFinalTestManifest,
    build_first_fixed_scoring_state,
    load_first_fixed_scoring_state,
    load_unlabeled_final_test_manifest,
    run_first_fixed_final_test_scoring,
    write_first_fixed_scoring_outputs,
)
from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.hog_scoring import PatchHOGScoreResult


@pytest.fixture(scope="module")
def project_config() -> ProjectConfig:
    return load_config(Path("configs/v0.1.yaml"))


def _record_id(category: str, relative_path: str) -> str:
    suffix = hashlib.sha256(relative_path.encode()).hexdigest()[:16]
    return f"visa-{category}-{suffix}"


def _write_manifest_set(
    root: Path,
    *,
    config: ProjectConfig,
    records: list[dict],
) -> tuple[Path, Path]:
    manifest_path = root / "final-test.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True))
            stream.write("\n")
    metadata = {
        "schema_version": 1,
        "dataset": {
            "name": config.dataset_name,
            "category": config.category,
            "license": config.dataset_license,
            "archive_identifier": config.archive.identifier,
            "archive_sha256": "a" * 64,
        },
        "official_split": {
            "repository": config.split.repository,
            "revision": config.split.revision,
            "path": config.split.path,
            "sha256": config.split.sha256,
        },
        "selection": {
            "reference_count": config.selection.reference_count,
            "seed": config.selection.seed,
            "procedure_version": config.selection.procedure_version,
            "namespace": config.selection.namespace,
        },
        "final_test_access_policy": FINAL_TEST_ACCESS_POLICY,
        "manifests": {
            "final-test": {
                "file": manifest_path.name,
                "record_count": len(records),
                "sha256": sha256_file(manifest_path),
            }
        },
    }
    metadata_path = root / "manifest-set.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata_path, manifest_path


def _manifest_records() -> list[dict]:
    paths = (
        "pcb1/Data/Images/Test/0001.JPG",
        "pcb1/Data/Images/Test/0002.JPG",
    )
    return [
        {
            "schema_version": 1,
            "id": _record_id("pcb1", path),
            "partition": "final-test",
            "category": "pcb1",
            "relative_path": path,
            "source_split": "test",
            "source_row": index + 2,
        }
        for index, path in enumerate(paths)
    ]


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
        patch_anomaly_scores=tuple(value for _ in range(225)),
        top_patch_count=12,
        top_patch_indices=tuple(range(12)),
        successful_patch_count=225,
        failed_patch_index=None,
    )


def _calibrations(project_config: ProjectConfig):
    paths = (
        "pcb1/Data/Images/Normal/0001.JPG",
        "pcb1/Data/Images/Normal/0002.JPG",
    )
    ecc = calibrate_normal_threshold(
        dict(zip(paths, (_ecc_score(0.1), _ecc_score(0.2)), strict=True)),
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )
    hog = calibrate_normal_threshold(
        dict(zip(paths, (_hog_score(-0.2), _hog_score(-0.1)), strict=True)),
        method=CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM,
        config=project_config,
    )
    assert ecc.succeeded and hog.succeeded
    return ecc, hog


def _state(project_config: ProjectConfig) -> FirstFixedScoringRunState:
    paths = tuple(record["relative_path"] for record in _manifest_records())
    ecc_calibration, hog_calibration = _calibrations(project_config)
    ecc_scores = dict(
        zip(paths, (_ecc_score(0.15), _ecc_score(0.25)), strict=True)
    )
    hog_scores = dict(
        zip(paths, (_hog_score(-0.15), _hog_score(0.05)), strict=True)
    )
    ecc_classifications = classify_fixed_threshold_batch(
        ecc_scores,
        calibration=ecc_calibration,
        config=project_config,
    )
    hog_classifications = classify_fixed_threshold_batch(
        hog_scores,
        calibration=hog_calibration,
        config=project_config,
    )
    decoded = {path: object() for path in paths}
    ecc_latency = measure_cpu_latency(
        decoded,
        lambda _: _ecc_score(0.15),
        method=CalibrationMethod.ECC_RESIDUAL,
        config=project_config,
    )
    hog_latency = measure_cpu_latency(
        decoded,
        lambda _: _hog_score(-0.15),
        method=CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM,
        config=project_config,
    )
    assert ecc_classifications.succeeded and hog_classifications.succeeded
    assert ecc_latency.succeeded and hog_latency.succeeded
    return FirstFixedScoringRunState(
        contract_version=SCORING_CONTRACT_VERSION,
        run_id=SCORING_RUN_ID,
        source_commit="a" * 40,
        freeze_checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
        dataset_integrity_sha256="d" * 64,
        calibration_checkpoint_sha256="e" * 64,
        calibration_state_sha256="f" * 64,
        manifest_set_sha256="1" * 64,
        final_test_manifest_sha256="2" * 64,
        final_test_paths=paths,
        ecc_scores=ecc_scores,
        hog_scores=hog_scores,
        ecc_classifications=ecc_classifications,
        hog_classifications=hog_classifications,
        ecc_latency=ecc_latency,
        hog_latency=hog_latency,
    )


def test_scoring_interface_accepts_no_split_class_metric_or_decision() -> None:
    parameters = inspect.signature(run_first_fixed_final_test_scoring).parameters
    state_fields = {field.name for field in fields(FirstFixedScoringRunState)}

    assert "split_csv" not in parameters
    assert all("label" not in name and "class" not in name for name in parameters)
    assert {"metrics", "failures", "decision"}.isdisjoint(state_fields)


def test_unlabeled_manifest_loader_accepts_exact_hash_bound_records(
    tmp_path: Path,
    project_config: ProjectConfig,
) -> None:
    metadata_path, manifest_path = _write_manifest_set(
        tmp_path,
        config=project_config,
        records=_manifest_records(),
    )

    result = load_unlabeled_final_test_manifest(
        metadata_path,
        manifest_path,
        config=project_config,
        expected_count=2,
    )

    assert result.paths == tuple(
        record["relative_path"] for record in _manifest_records()
    )
    assert result.manifest_set_sha256 == sha256_file(metadata_path)
    assert result.final_test_manifest_sha256 == sha256_file(manifest_path)
    assert result.archive_sha256 == "a" * 64


def test_unlabeled_manifest_loader_rejects_class_field_before_image_access(
    tmp_path: Path,
    project_config: ProjectConfig,
) -> None:
    records = _manifest_records()
    records[0]["label"] = "normal"
    metadata_path, manifest_path = _write_manifest_set(
        tmp_path,
        config=project_config,
        records=records,
    )

    with pytest.raises(
        FirstFixedScoringRunError,
        match="unexpected metadata",
    ):
        load_unlabeled_final_test_manifest(
            metadata_path,
            manifest_path,
            config=project_config,
            expected_count=2,
        )


def test_outputs_round_trip_hash_checked_state_without_evaluation_results(
    tmp_path: Path,
    project_config: ProjectConfig,
) -> None:
    state = _state(project_config)
    ecc_calibration, hog_calibration = _calibrations(project_config)
    output_dir = tmp_path / "scoring"
    state_path = tmp_path / "state.pkl"

    checkpoint = write_first_fixed_scoring_outputs(
        state,
        output_dir=output_dir,
        state_path=state_path,
        ecc_calibration=ecc_calibration,
        hog_calibration=hog_calibration,
        config=project_config,
        expected_count=2,
    )
    loaded = load_first_fixed_scoring_state(
        state_path,
        checkpoint_path=output_dir / "first-fixed-scoring.json",
        ecc_calibration=ecc_calibration,
        hog_calibration=hog_calibration,
        config=project_config,
        expected_count=2,
    )

    assert loaded.source_commit == state.source_commit
    assert loaded.final_test_paths == state.final_test_paths
    assert checkpoint["status"] == "SCORES_CLASSIFICATIONS_AND_LATENCY_FIXED"
    assert checkpoint["evaluation_boundary"] == {
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
    }
    artifact_names = {path.name for path in output_dir.rglob("*") if path.is_file()}
    assert artifact_names == {
        "first-fixed-scoring.json",
        "scores.csv",
        "classifications.csv",
        "latency.json",
        "latency-observations.csv",
    }
    all_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output_dir.rglob("*")
        if path.is_file()
    )
    assert "true_class" not in all_text
    assert "auroc" not in all_text.lower()
    assert "auprc" not in all_text.lower()
    assert "ADOPT" not in all_text


def test_build_connects_scoring_classification_and_latency_without_class_input(
    monkeypatch: pytest.MonkeyPatch,
    project_config: ProjectConfig,
) -> None:
    paths = tuple(record["relative_path"] for record in _manifest_records())
    ecc_calibration, hog_calibration = _calibrations(project_config)
    calibration_state = SimpleNamespace(
        ecc_calibration=ecc_calibration,
        hog_calibration=hog_calibration,
    )
    monkeypatch.setattr(
        scoring_module,
        "_decode_grayscale",
        lambda _: np.zeros((8, 8), dtype=np.uint8),
    )
    monkeypatch.setattr(
        scoring_module,
        "_ecc_scorer",
        lambda *_args, **_kwargs: (lambda _decoded: _ecc_score(0.15)),
    )
    monkeypatch.setattr(
        scoring_module,
        "_hog_scorer",
        lambda *_args, **_kwargs: (lambda _decoded: _hog_score(-0.15)),
    )
    manifest = UnlabeledFinalTestManifest(
        manifest_set_sha256="1" * 64,
        final_test_manifest_sha256="2" * 64,
        archive_sha256="3" * 64,
        paths=paths,
    )

    result = build_first_fixed_scoring_state(
        source_commit="a" * 40,
        freeze_checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
        dataset_integrity_sha256="d" * 64,
        calibration_checkpoint_sha256="e" * 64,
        calibration_state_sha256="f" * 64,
        manifest=manifest,
        dataset_root=Path("/unused"),
        calibration_state=calibration_state,
        config=project_config,
        expected_count=2,
    )

    assert result.final_test_paths == paths
    assert result.ecc_classifications.item_count == 2
    assert result.hog_classifications.item_count == 2
    assert result.ecc_latency.sample_count == 6
    assert result.hog_latency.sample_count == 6
