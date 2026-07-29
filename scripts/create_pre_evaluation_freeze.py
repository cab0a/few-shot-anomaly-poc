"""Create the v0.1 pre-evaluation freeze record."""

from __future__ import annotations

import argparse
from pathlib import Path

from few_shot_anomaly_poc.config import load_config
from few_shot_anomaly_poc.freeze_checkpoint import (
    build_pre_evaluation_freeze,
    write_pre_evaluation_freeze,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/v0.1.yaml"
OUTPUT_PATH = PROJECT_ROOT / "artifacts/v0.1/freeze/pre-evaluation-freeze.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the non-overwritable v0.1 pre-evaluation freeze record."
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--ci-run-id", required=True, type=int)
    parser.add_argument("--ci-run-url", required=True)
    arguments = parser.parse_args()
    config = load_config(CONFIG_PATH)
    record = build_pre_evaluation_freeze(
        project_root=PROJECT_ROOT,
        source_commit=arguments.source_commit,
        ci_run_id=arguments.ci_run_id,
        ci_run_url=arguments.ci_run_url,
        config=config,
    )
    write_pre_evaluation_freeze(
        record,
        path=OUTPUT_PATH,
        project_root=PROJECT_ROOT,
    )
    print(f"Pre-evaluation freeze written to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
