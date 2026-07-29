"""Run the first fixed VisA pcb1 final-test scoring pass without class metadata."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from few_shot_anomaly_poc.config import load_config
from few_shot_anomaly_poc.final_test_scoring_run import (
    FirstFixedScoringRunError,
    run_first_fixed_final_test_scoring,
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
DEFAULT_MANIFEST_SET = PROJECT_ROOT / "data/manifests/v0.1/manifest-set.json"
DEFAULT_FINAL_TEST_MANIFEST = PROJECT_ROOT / "data/manifests/v0.1/final-test.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/v0.1/scoring/first-fixed-final-test"
DEFAULT_SCORING_STATE = (
    PROJECT_ROOT / "work/v0.1/final-test/first-fixed-scoring-state.pkl"
)


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
        raise FirstFixedScoringRunError(
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
        raise FirstFixedScoringRunError(
            "working tree must be clean before final-test scoring"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score and time the fixed final-test assets without reading class "
            "metadata or calculating evaluation metrics."
        )
    )
    parser.add_argument("--source-commit")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--manifest-set", type=Path, default=DEFAULT_MANIFEST_SET)
    parser.add_argument(
        "--final-test-manifest",
        type=Path,
        default=DEFAULT_FINAL_TEST_MANIFEST,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_SCORING_STATE)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_config(DEFAULT_CONFIG)
    source_commit = args.source_commit or _git_head()
    try:
        _require_clean_source(source_commit)
        checkpoint = run_first_fixed_final_test_scoring(
            source_commit=source_commit,
            dataset_root=args.dataset_root or config.paths.extracted,
            config_path=DEFAULT_CONFIG,
            freeze_checkpoint_path=DEFAULT_FREEZE,
            dataset_integrity_path=DEFAULT_INTEGRITY,
            calibration_checkpoint_path=DEFAULT_CALIBRATION_CHECKPOINT,
            calibration_state_path=DEFAULT_CALIBRATION_STATE,
            manifest_set_path=args.manifest_set,
            final_test_manifest_path=args.final_test_manifest,
            output_dir=args.output_dir,
            scoring_state_path=args.state_path,
            config=config,
            progress=lambda message: print(message, flush=True),
        )
    except (
        FirstFixedScoringRunError,
        FileExistsError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        print(f"error: {error}")
        return 1

    print("first fixed final-test scoring complete:")
    for method, record in checkpoint["methods"].items():
        print(
            f"  {method}: items={record['item_count']}, "
            f"score_failures={record['score_failure_count']}, "
            f"p95_seconds={record['p95_latency_seconds']}"
        )
    print("No per-path final-test class, metric, failure case, or decision was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
