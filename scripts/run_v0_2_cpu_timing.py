"""Run the preregistered memory-bounded DINOv2 CPU timing parent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE = PROJECT_ROOT / "src"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

from few_shot_anomaly_poc.dinov2_timing import (  # noqa: E402
    DINOv2TimingError,
    run_timing_parent,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the fixed memory-mapped synthetic workload and run DINOv2 "
            "at 224 then 448 in separate isolated CPU processes. This does not "
            "access VisA, labels, thresholds, or anomaly-performance metrics."
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
    parser.add_argument(
        "--no-concurrent-project-benchmark-confirmed",
        action="store_true",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = run_timing_parent(
            artifact_dir=args.artifact_dir,
            environment_root=args.environment_root,
            project_root=args.project_root,
            execution_commit=args.execution_commit,
            verification_date=args.verification_date,
            no_concurrent_project_benchmark_confirmed=(
                args.no_concurrent_project_benchmark_confirmed
            ),
            output_root=args.output_root,
        )
    except (DINOv2TimingError, FileExistsError, OSError) as error:
        print(f"error: {error}")
        return 1
    print(
        "DINOv2 CPU timing parent: "
        f"status={report['decision']['status']}, "
        f"selected={report['decision']['selected_resolution_candidate']}, "
        f"next_step={report['decision']['next_step']}, "
        f"output={args.output_root.as_posix()}"
    )
    return 0 if report["decision"]["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
