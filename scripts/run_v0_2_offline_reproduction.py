"""Start the fixed 224 DINOv2 offline score-reproduction worker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE = PROJECT_ROOT / "src"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

from few_shot_anomaly_poc.dinov2_reproduction import (  # noqa: E402
    DINOv2ReproductionError,
    run_reproduction_parent,
)
from few_shot_anomaly_poc.dinov2_timing import DINOv2TimingError  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Start one fresh isolated process that regenerates the first 10 "
            "fixed 224 DINOv2 scores. This does not access VisA or labels."
        )
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--environment-root",
        type=Path,
        default=Path("environments/v0.2-preflight/.venv"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--verification-date", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = run_reproduction_parent(
            artifact_dir=args.artifact_dir,
            environment_root=args.environment_root,
            project_root=args.project_root,
            execution_commit=args.execution_commit,
            verification_date=args.verification_date,
            output_root=args.output_root,
        )
    except (
        DINOv2ReproductionError,
        DINOv2TimingError,
        FileExistsError,
        OSError,
    ) as error:
        print(f"error: {error}")
        return 1
    print(
        "DINOv2 offline score reproduction parent: "
        f"status={report['decision']['status']}, "
        f"next_step={report['decision']['next_step']}"
    )
    return 0 if report["decision"]["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
