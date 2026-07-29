"""Reveal final-test classes and produce the fixed v0.1 evaluation decision."""

from __future__ import annotations

import re
from pathlib import Path

from few_shot_anomaly_poc.calibration import CalibrationMethod
from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.evaluation_artifacts import (
    EvaluationArtifactBundle,
    MethodEvaluationArtifacts,
    write_evaluation_artifact_bundle,
)
from few_shot_anomaly_poc.failure_cases import select_failure_cases
from few_shot_anomaly_poc.final_test_scoring_run import (
    FirstFixedScoringRunState,
    load_first_fixed_scoring_state,
)
from few_shot_anomaly_poc.freeze_checkpoint import (
    read_and_verify_pre_evaluation_freeze,
)
from few_shot_anomaly_poc.hard_gate_decision import (
    DecisionProcessEvidence,
    FailureReviewDisposition,
    apply_hard_gate_decision,
)
from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.image_metrics import calculate_image_level_metrics
from few_shot_anomaly_poc.label_reveal import (
    FinalTestLabelRecord,
    reveal_final_test_labels,
)
from few_shot_anomaly_poc.manifests import load_official_rows
from few_shot_anomaly_poc.normal_calibration_run import (
    NormalCalibrationRunState,
    load_normal_calibration_state,
)

FINAL_EVALUATION_RUN_ID = "visa-pcb1-v0-1-final"
FAILURE_REVIEW_RATIONALE = (
    "Mechanical false-positive and false-negative selection is complete, but "
    "image content has not been reviewed at this decision stage; causal or "
    "intended-use boundary claims would be premature."
)
FAILURE_REVIEW_CONDITION = (
    "Review the selected false-positive and false-negative image content and "
    "define an operating guardrail before any follow-up trial."
)
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class FinalEvaluationRunError(Exception):
    """Reject changed lineage, class coverage, or incomplete evaluation output."""


def load_final_test_class_records(
    split_csv: Path,
    *,
    expected_paths: tuple[str, ...],
    config: ProjectConfig,
) -> tuple[FinalTestLabelRecord, ...]:
    """Reveal fixed official test classes only after scoring has completed."""
    rows = load_official_rows(
        split_csv,
        expected_sha256=config.split.sha256,
        category=config.category,
    )
    records = tuple(
        FinalTestLabelRecord(relative_path=row.relative_path, label=row.label)
        for row in sorted(
            (row for row in rows if row.split == "test"),
            key=lambda row: row.relative_path,
        )
    )
    observed_paths = tuple(record.relative_path for record in records)
    if (
        not expected_paths
        or expected_paths != tuple(sorted(expected_paths))
        or observed_paths != expected_paths
        or len(set(observed_paths)) != len(observed_paths)
    ):
        raise FinalEvaluationRunError(
            "official final-test classes do not exactly cover the fixed scoring paths"
        )
    return records


def _process_evidence(reference_count: int) -> DecisionProcessEvidence:
    return DecisionProcessEvidence(
        normal_reference_count=reference_count,
        anomaly_training_labels_used=False,
        reproducibility_verified=True,
        test_leakage_detected=False,
        failure_review_disposition=FailureReviewDisposition.GUARDRAIL_REQUIRED,
        failure_review_rationale=FAILURE_REVIEW_RATIONALE,
        condition=FAILURE_REVIEW_CONDITION,
    )


def _method_artifacts(
    method: CalibrationMethod,
    *,
    calibration_state: NormalCalibrationRunState,
    scoring_state: FirstFixedScoringRunState,
    class_records: tuple[FinalTestLabelRecord, ...],
    config: ProjectConfig,
) -> MethodEvaluationArtifacts:
    if method is CalibrationMethod.ECC_RESIDUAL:
        calibration_scores = calibration_state.ecc_calibration_scores
        final_scores = scoring_state.ecc_scores
        classifications = scoring_state.ecc_classifications
        latency = scoring_state.ecc_latency
    else:
        calibration_scores = calibration_state.hog_calibration_scores
        final_scores = scoring_state.hog_scores
        classifications = scoring_state.hog_classifications
        latency = scoring_state.hog_latency

    revealed = reveal_final_test_labels(
        classifications,
        class_records,
        config=config,
    )
    metrics = calculate_image_level_metrics(revealed, config=config)
    failures = select_failure_cases(revealed, config=config)
    process_evidence = _process_evidence(len(calibration_state.reference_paths))
    decision = apply_hard_gate_decision(
        metrics,
        latency,
        process_evidence,
        config=config,
    )
    if not all(
        result.succeeded
        for result in (
            revealed,
            metrics,
            failures,
            latency,
            decision,
        )
    ):
        raise FinalEvaluationRunError(f"final evaluation failed for {method}")
    return MethodEvaluationArtifacts(
        method=method,
        calibration_scores=calibration_scores,
        final_test_scores=final_scores,
        classifications=classifications,
        revealed=revealed,
        metrics=metrics,
        latency=latency,
        failures=failures,
        process_evidence=process_evidence,
        decision=decision,
    )


def build_final_evaluation_bundle(
    *,
    source_commit: str,
    partition_manifest_sha256: str,
    calibration_state: NormalCalibrationRunState,
    scoring_state: FirstFixedScoringRunState,
    class_records: tuple[FinalTestLabelRecord, ...],
    config: ProjectConfig,
) -> EvaluationArtifactBundle:
    """Connect unchanged scores to classes, metrics, failures, and hard gates."""
    if not COMMIT_PATTERN.fullmatch(source_commit):
        raise FinalEvaluationRunError("source commit must be a full Git SHA")
    methods = tuple(
        _method_artifacts(
            method,
            calibration_state=calibration_state,
            scoring_state=scoring_state,
            class_records=class_records,
            config=config,
        )
        for method in (
            CalibrationMethod.ECC_RESIDUAL,
            CalibrationMethod.PATCH_HOG_ONE_CLASS_SVM,
        )
    )
    return EvaluationArtifactBundle(
        run_id=FINAL_EVALUATION_RUN_ID,
        run_kind="final_test",
        dataset=config.dataset_name,
        category=config.category,
        source_commit=source_commit,
        partition_manifest_sha256=partition_manifest_sha256,
        methods=methods,
    )


def run_final_evaluation(
    *,
    source_commit: str,
    config_path: Path,
    freeze_checkpoint_path: Path,
    dataset_integrity_path: Path,
    calibration_checkpoint_path: Path,
    calibration_state_path: Path,
    scoring_checkpoint_path: Path,
    scoring_state_path: Path,
    manifest_set_path: Path,
    final_test_manifest_path: Path,
    split_csv: Path,
    output_root: Path,
    config: ProjectConfig,
) -> Path:
    """Verify lineage, reveal classes once, and write the immutable final bundle."""
    if not COMMIT_PATTERN.fullmatch(source_commit):
        raise FinalEvaluationRunError("source commit must be a full Git SHA")
    read_and_verify_pre_evaluation_freeze(
        freeze_checkpoint_path,
        project_root=config.project_root,
    )
    calibration_state = load_normal_calibration_state(
        calibration_state_path,
        checkpoint_path=calibration_checkpoint_path,
        config=config,
    )
    scoring_state = load_first_fixed_scoring_state(
        scoring_state_path,
        checkpoint_path=scoring_checkpoint_path,
        ecc_calibration=calibration_state.ecc_calibration,
        hog_calibration=calibration_state.hog_calibration,
        config=config,
    )

    lineage = (
        (scoring_state.freeze_checkpoint_sha256, freeze_checkpoint_path),
        (scoring_state.config_sha256, config_path),
        (scoring_state.dataset_integrity_sha256, dataset_integrity_path),
        (scoring_state.calibration_checkpoint_sha256, calibration_checkpoint_path),
        (scoring_state.calibration_state_sha256, calibration_state_path),
        (scoring_state.manifest_set_sha256, manifest_set_path),
        (scoring_state.final_test_manifest_sha256, final_test_manifest_path),
    )
    if any(sha256_file(path) != expected for expected, path in lineage):
        raise FinalEvaluationRunError("final evaluation lineage differs from fixed scoring")

    class_records = load_final_test_class_records(
        split_csv,
        expected_paths=scoring_state.final_test_paths,
        config=config,
    )
    bundle = build_final_evaluation_bundle(
        source_commit=source_commit,
        partition_manifest_sha256=scoring_state.manifest_set_sha256,
        calibration_state=calibration_state,
        scoring_state=scoring_state,
        class_records=class_records,
        config=config,
    )
    return write_evaluation_artifact_bundle(
        bundle,
        output_root=output_root,
        config_path=config_path,
        config=config,
    )
