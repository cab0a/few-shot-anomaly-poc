"""Reveal VisA pcb1 final-test classes and apply the fixed v0.1 decision rules."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from few_shot_anomaly_poc.config import load_config
from few_shot_anomaly_poc.errors import EvaluationArtifactError
from few_shot_anomaly_poc.final_evaluation_run import (
    FINAL_EVALUATION_RUN_ID,
    FinalEvaluationRunError,
    run_final_evaluation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/v0.1.yaml"
DEFAULT_FREEZE = PROJECT_ROOT / "artifacts/v0.1/freeze/pre-evaluation-freeze.json"
DEFAULT_INTEGRITY = PROJECT_ROOT / "artifacts/v0.1/data/pcb1-local-integrity.json"
DEFAULT_CALIBRATION_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts/v0.1/calibration/normal-only/normal-only-calibration.json"
)
DEFAULT_CALIBRATION_STATE = (
    PROJECT_ROOT / "work/v0.1/calibration/normal-only-state.pkl"
)
DEFAULT_SCORING_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts/v0.1/scoring/first-fixed-final-test/first-fixed-scoring.json"
)
DEFAULT_SCORING_STATE = (
    PROJECT_ROOT / "work/v0.1/final-test/first-fixed-scoring-state.pkl"
)
DEFAULT_MANIFEST_SET = PROJECT_ROOT / "data/manifests/v0.1/manifest-set.json"
DEFAULT_FINAL_TEST_MANIFEST = PROJECT_ROOT / "data/manifests/v0.1/final-test.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/v0.1/evaluation"


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _require_clean_source(source_commit: str) -> None:
    if source_commit != _git_head():
        raise FinalEvaluationRunError(
            "source commit is not the checked-out Git HEAD"
        )
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        raise FinalEvaluationRunError(
            "working tree must be clean before final evaluation"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reveal final-test classes, calculate fixed metrics and failure "
            "cases, and apply the preregistered hard gates."
        )
    )
    parser.add_argument("--source-commit")
    parser.add_argument("--split-csv", type=Path)
    parser.add_argument("--manifest-set", type=Path, default=DEFAULT_MANIFEST_SET)
    parser.add_argument(
        "--final-test-manifest",
        type=Path,
        default=DEFAULT_FINAL_TEST_MANIFEST,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_config(DEFAULT_CONFIG)
    source_commit = args.source_commit or _git_head()
    try:
        _require_clean_source(source_commit)
        output = run_final_evaluation(
            source_commit=source_commit,
            config_path=DEFAULT_CONFIG,
            freeze_checkpoint_path=DEFAULT_FREEZE,
            dataset_integrity_path=DEFAULT_INTEGRITY,
            calibration_checkpoint_path=DEFAULT_CALIBRATION_CHECKPOINT,
            calibration_state_path=DEFAULT_CALIBRATION_STATE,
            scoring_checkpoint_path=DEFAULT_SCORING_CHECKPOINT,
            scoring_state_path=DEFAULT_SCORING_STATE,
            manifest_set_path=args.manifest_set,
            final_test_manifest_path=args.final_test_manifest,
            split_csv=args.split_csv or config.paths.split_csv,
            output_root=args.output_root,
            config=config,
        )
    except (
        EvaluationArtifactError,
        FinalEvaluationRunError,
        FileExistsError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        print(f"error: {error}")
        return 1

    print(f"final evaluation written: {output}")
    for method in ("ecc_residual", "patch_hog_one_class_svm"):
        method_dir = output / method
        metrics = json.loads((method_dir / "metrics.json").read_text(encoding="utf-8"))
        decision = json.loads((method_dir / "decision.json").read_text(encoding="utf-8"))
        print(
            f"  {method}: AUROC={metrics['auroc']}, "
            f"AUPRC={metrics['auprc']}, "
            f"normal_FPR={metrics['normal_false_positive_rate']}, "
            f"anomaly_recall={metrics['anomaly_recall']}, "
            f"decision={decision['decision']}"
        )
    print(f"Run ID: {FINAL_EVALUATION_RUN_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
