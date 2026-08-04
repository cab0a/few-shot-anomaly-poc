from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from few_shot_anomaly_poc.v0_2_calibration_artifacts import (
    CALIBRATION_COLUMNS,
    CalibrationScore,
    V0_2CalibrationArtifactError,
    build_fit_record,
    calibrate_normal_scores,
    write_method_artifacts,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    load_v0_2_artifact_schema,
    load_v0_2_config,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "synthetic-calibration"
SHA256 = "a" * 64


def _contract() -> tuple[dict, dict]:
    return (
        load_v0_2_config(ROOT / "configs/v0.2.yaml"),
        load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json"),
    )


def _scores(count: int = 20) -> tuple[CalibrationScore, ...]:
    return tuple(
        CalibrationScore(
            source_path=f"pcb2/Data/Images/Normal/{index:04d}.JPG",
            score_status="ok",
            score_failure_code=None,
            anomaly_score=index / 100,
        )
        for index in reversed(range(count))
    )


def test_normal_calibration_uses_fixed_rank_tie_break_and_strict_threshold() -> None:
    config, schema = _contract()
    scores = list(_scores())
    scores[0] = CalibrationScore(
        source_path=scores[0].source_path,
        score_status="failed",
        score_failure_code="synthetic_failure",
        anomaly_score=1.0,
    )

    result = calibrate_normal_scores(
        scores,
        run_id=RUN_ID,
        method="ecc_residual",
        config=config,
        schema=schema,
    )

    assert [record["source_path"] for record in result.records] == sorted(
        score.source_path for score in scores
    )
    assert result.summary["rank"] == 19
    assert result.summary["threshold"] == 0.18
    assert result.summary["predicted_anomalous_count"] == 1
    assert result.summary["score_failure_count"] == 1
    assert result.summary["realized_normal_fpr"] == 0.05


def test_normal_calibration_rejects_duplicate_paths() -> None:
    config, schema = _contract()
    duplicated = (*_scores(), _scores()[0])

    with pytest.raises(V0_2CalibrationArtifactError, match="not unique"):
        calibrate_normal_scores(
            duplicated,
            run_id=RUN_ID,
            method="ecc_residual",
            config=config,
            schema=schema,
        )


def test_method_artifacts_match_the_fixed_schema_and_refuse_overwrite(
    tmp_path: Path,
) -> None:
    config, schema = _contract()
    fit = build_fit_record(
        run_id=RUN_ID,
        method="dinov2_vits14_224_nn",
        status="fit_ok",
        successful_reference_count=20,
        failed_reference_count=0,
        reference_manifest_sha256=SHA256,
        fitted_state_sha256=SHA256,
        failure_code=None,
        config=config,
        schema=schema,
    )
    calibration = calibrate_normal_scores(
        tuple(
            CalibrationScore(
                source_path=score.source_path,
                score_status=score.score_status,
                score_failure_code=score.score_failure_code,
                anomaly_score=score.anomaly_score,
            )
            for score in _scores()
        ),
        run_id=RUN_ID,
        method="dinov2_vits14_224_nn",
        config=config,
        schema=schema,
    )
    method_root = tmp_path / "dinov2_vits14_224_nn"

    paths = write_method_artifacts(
        method_root=method_root,
        fit_record=fit,
        calibration=calibration,
    )

    assert [path.name for path in paths] == [
        "fit.json",
        "calibration-scores.csv",
        "calibration-summary.json",
    ]
    assert json.loads((method_root / "fit.json").read_text(encoding="utf-8")) == fit
    with (method_root / "calibration-scores.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = tuple(reader)
        assert tuple(reader.fieldnames or ()) == CALIBRATION_COLUMNS
    assert len(rows) == 20
    with pytest.raises(FileExistsError, match="overwrite"):
        write_method_artifacts(
            method_root=method_root,
            fit_record=fit,
            calibration=calibration,
        )


def test_failed_fit_writes_no_fabricated_calibration(tmp_path: Path) -> None:
    config, schema = _contract()
    fit = build_fit_record(
        run_id=RUN_ID,
        method="patch_hog_ocsvm",
        status="fit_failed",
        successful_reference_count=19,
        failed_reference_count=1,
        reference_manifest_sha256=SHA256,
        fitted_state_sha256=None,
        failure_code="synthetic_fit_failure",
        config=config,
        schema=schema,
    )

    paths = write_method_artifacts(
        method_root=tmp_path / "patch_hog_ocsvm",
        fit_record=fit,
        calibration=None,
    )

    assert [path.name for path in paths] == ["fit.json"]
