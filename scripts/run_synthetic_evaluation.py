"""Generate the synthetic v0.1 end-to-end evaluation artifact bundle."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from few_shot_anomaly_poc.config import load_config
from few_shot_anomaly_poc.synthetic_evaluation import run_synthetic_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/v0.1.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/v0.1/evaluation"


def _source_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic synthetic evaluation records; "
            "the output is not VisA performance evidence."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory for the immutable synthetic-e2e bundle.",
    )
    parser.add_argument(
        "--source-commit",
        default=None,
        help="40-character source commit; defaults to the current Git HEAD.",
    )
    arguments = parser.parse_args()
    config = load_config(DEFAULT_CONFIG)
    output = run_synthetic_evaluation(
        output_root=arguments.output_root.resolve(),
        source_commit=arguments.source_commit or _source_commit(),
        config_path=DEFAULT_CONFIG,
        config=config,
    )
    print(f"Synthetic evaluation artifacts written to {output}")
    print("These artifacts are plumbing evidence, not VisA performance evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
