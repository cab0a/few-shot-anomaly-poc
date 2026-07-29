"""Run the fixed VisA pcb1 normal-reference fitting and calibration checkpoint."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from few_shot_anomaly_poc.config import load_config
from few_shot_anomaly_poc.normal_calibration_run import (
    NormalCalibrationRunError,
    run_normal_reference_fit_and_calibration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/v0.1.yaml"
DEFAULT_FREEZE = PROJECT_ROOT / "artifacts/v0.1/freeze/pre-evaluation-freeze.json"
DEFAULT_PARTITIONS = PROJECT_ROOT / "artifacts/v0.1/data/pcb1-normal-partitions.csv"
DEFAULT_DATASET_RECORD = PROJECT_ROOT / "artifacts/v0.1/data/dataset-record.json"
DEFAULT_INTEGRITY = PROJECT_ROOT / "artifacts/v0.1/data/pcb1-local-integrity.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/v0.1/calibration/normal-only"
DEFAULT_STATE = PROJECT_ROOT / "work/v0.1/calibration/normal-only-state.pkl"


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
        raise NormalCalibrationRunError("source commit is not the checked-out Git HEAD")
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        raise NormalCalibrationRunError("working tree must be clean before calibration")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit both frozen v0.1 methods from 20 normal references and fix one "
            "threshold per method from the normal calibration partition only."
        )
    )
    parser.add_argument("--source-commit")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_config(DEFAULT_CONFIG)
    source_commit = args.source_commit or _git_head()
    try:
        _require_clean_source(source_commit)
        checkpoint = run_normal_reference_fit_and_calibration(
            source_commit=source_commit,
            archive_path=args.archive or config.paths.archive,
            dataset_root=args.dataset_root or config.paths.extracted,
            split_csv=args.split_csv or config.paths.split_csv,
            config_path=DEFAULT_CONFIG,
            freeze_checkpoint_path=DEFAULT_FREEZE,
            partition_manifest_path=DEFAULT_PARTITIONS,
            dataset_record_path=DEFAULT_DATASET_RECORD,
            dataset_integrity_path=DEFAULT_INTEGRITY,
            output_dir=args.output_dir,
            state_path=args.state_path,
            config=config,
            progress=lambda message: print(message, flush=True),
        )
    except (NormalCalibrationRunError, FileExistsError, OSError) as error:
        print(f"error: {error}")
        return 1

    methods = checkpoint["methods"]
    print("normal-only thresholds fixed before final-test scoring:")
    for method in ("ecc_residual", "patch_hog_one_class_svm"):
        calibration = methods[method]["threshold_calibration"]
        print(
            f"  {method}: threshold={calibration['threshold']}, "
            f"failed_scores={calibration['failed_score_count']}"
        )
    print("No final-test image, per-path final-test label, metric, or decision was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
