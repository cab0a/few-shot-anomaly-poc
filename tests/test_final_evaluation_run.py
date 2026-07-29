from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from few_shot_anomaly_poc.config import ProjectConfig, load_config
from few_shot_anomaly_poc.evaluation_artifacts import (
    write_evaluation_artifact_bundle,
)
from few_shot_anomaly_poc.final_evaluation_run import (
    FAILURE_REVIEW_CONDITION,
    FAILURE_REVIEW_RATIONALE,
    FINAL_EVALUATION_RUN_ID,
    FinalEvaluationRunError,
    build_final_evaluation_bundle,
    load_final_test_class_records,
)
from few_shot_anomaly_poc.label_reveal import FinalTestLabelRecord
from few_shot_anomaly_poc.synthetic_evaluation import (
    SYNTHETIC_REFERENCE_PATHS,
    build_synthetic_evaluation_bundle,
)
from scripts.run_final_evaluation import _method_summary
from tests.helpers import create_config, final_test_row, normal_train


@pytest.fixture(scope="module")
def project_config() -> ProjectConfig:
    return load_config(Path("configs/v0.1.yaml"))


def test_class_record_loader_reveals_exact_sorted_official_test_coverage(
    tmp_path: Path,
) -> None:
    rows = [
        normal_train(100),
        normal_train(101),
        final_test_row(2, "anomaly"),
        final_test_row(1, "normal"),
    ]
    config = create_config(tmp_path, rows=rows, reference_count=1)
    expected_paths = tuple(sorted(row["image"] for row in rows if row["split"] == "test"))

    records = load_final_test_class_records(
        config.paths.split_csv,
        expected_paths=expected_paths,
        config=config,
    )

    assert tuple(record.relative_path for record in records) == expected_paths
    assert tuple(record.label for record in records) == ("anomaly", "normal")


def test_class_record_loader_rejects_scoring_path_mismatch(tmp_path: Path) -> None:
    rows = [
        normal_train(100),
        normal_train(101),
        final_test_row(1, "normal"),
        final_test_row(2, "anomaly"),
    ]
    config = create_config(tmp_path, rows=rows, reference_count=1)

    with pytest.raises(FinalEvaluationRunError, match="exactly cover"):
        load_final_test_class_records(
            config.paths.split_csv,
            expected_paths=("pcb1/Data/Images/Normal/not-present.JPG",),
            config=config,
        )


def test_final_bundle_connects_preserved_scores_to_fixed_decision_primitives(
    tmp_path: Path,
    project_config: ProjectConfig,
) -> None:
    synthetic = build_synthetic_evaluation_bundle(
        source_commit="a" * 40,
        config=project_config,
    )
    by_method = {item.method: item for item in synthetic.methods}
    ecc = by_method[next(method for method in by_method if str(method) == "ecc_residual")]
    hog = by_method[
        next(method for method in by_method if str(method) == "patch_hog_one_class_svm")
    ]
    calibration_state = SimpleNamespace(
        reference_paths=SYNTHETIC_REFERENCE_PATHS,
        ecc_calibration_scores=ecc.calibration_scores,
        hog_calibration_scores=hog.calibration_scores,
    )
    scoring_state = SimpleNamespace(
        ecc_scores=ecc.final_test_scores,
        hog_scores=hog.final_test_scores,
        ecc_classifications=ecc.classifications,
        hog_classifications=hog.classifications,
        ecc_latency=ecc.latency,
        hog_latency=hog.latency,
    )
    assert ecc.revealed.records is not None
    class_records = tuple(
        FinalTestLabelRecord(
            relative_path=record.relative_path,
            label=record.label,
        )
        for record in ecc.revealed.records
    )

    bundle = build_final_evaluation_bundle(
        source_commit="b" * 40,
        partition_manifest_sha256="c" * 64,
        calibration_state=calibration_state,
        scoring_state=scoring_state,
        class_records=class_records,
        config=project_config,
    )
    output = write_evaluation_artifact_bundle(
        bundle,
        output_root=tmp_path,
        config_path=Path("configs/v0.1.yaml"),
        config=project_config,
    )

    assert output.name == FINAL_EVALUATION_RUN_ID
    manifest = json.loads((output / "artifact-manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_kind"] == "final_test"
    for method in ("ecc_residual", "patch_hog_one_class_svm"):
        decision = json.loads(
            (output / method / "decision.json").read_text(encoding="utf-8")
        )
        assert decision["decision"] == "ADOPT WITH CONDITIONS"
        assert decision["all_hard_gates_passed"] is True
        assert decision["failure_review_disposition"] == "guardrail_required"
        assert decision["failure_review_rationale"] == FAILURE_REVIEW_RATIONALE
        assert decision["condition"] == FAILURE_REVIEW_CONDITION


def test_final_bundle_rejects_invalid_source_commit(
    project_config: ProjectConfig,
) -> None:
    synthetic = build_synthetic_evaluation_bundle(
        source_commit="a" * 40,
        config=project_config,
    )
    ecc, hog = synthetic.methods
    calibration_state = SimpleNamespace(
        reference_paths=SYNTHETIC_REFERENCE_PATHS,
        ecc_calibration_scores=ecc.calibration_scores,
        hog_calibration_scores=hog.calibration_scores,
    )
    scoring_state = SimpleNamespace(
        ecc_scores=ecc.final_test_scores,
        hog_scores=hog.final_test_scores,
        ecc_classifications=ecc.classifications,
        hog_classifications=hog.classifications,
        ecc_latency=ecc.latency,
        hog_latency=hog.latency,
    )
    assert ecc.revealed.records is not None
    class_records = tuple(
        FinalTestLabelRecord(
            relative_path=record.relative_path,
            label=record.label,
        )
        for record in ecc.revealed.records
    )

    with pytest.raises(FinalEvaluationRunError, match="full Git SHA"):
        build_final_evaluation_bundle(
            source_commit="short",
            partition_manifest_sha256="c" * 64,
            calibration_state=calibration_state,
            scoring_state=scoring_state,
            class_records=class_records,
            config=project_config,
        )


def test_cli_summary_uses_written_artifact_metric_keys(tmp_path: Path) -> None:
    method_dir = tmp_path / "ecc_residual"
    method_dir.mkdir()
    (method_dir / "metrics.json").write_text(
        json.dumps(
            {
                "image_level_auroc": 0.8,
                "image_level_auprc": 0.7,
                "normal_false_positive_rate": 0.1,
                "anomaly_recall": 0.2,
            }
        ),
        encoding="utf-8",
    )
    (method_dir / "decision.json").write_text(
        json.dumps({"decision": "REJECT"}),
        encoding="utf-8",
    )

    assert _method_summary(method_dir, "ecc_residual") == (
        "  ecc_residual: AUROC=0.8, AUPRC=0.7, normal_FPR=0.1, "
        "anomaly_recall=0.2, decision=REJECT"
    )
