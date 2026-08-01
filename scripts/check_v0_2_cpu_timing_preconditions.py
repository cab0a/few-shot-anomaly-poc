"""Check ordered prerequisites before the v0.2 CPU timing workload."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE = PROJECT_ROOT / "src"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

from few_shot_anomaly_poc.dinov2_timing_preflight import (  # noqa: E402
    DINOv2TimingPreflightError,
    run_timing_preconditions,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check preregistration identity, prior evidence, AC power, fixed CPU, "
            "core counts, diagnostic memory metadata, WSL2, default affinity, "
            "default priority, and concurrent-project controls before any DINOv2 "
            "timing invocation."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--verification-date", required=True)
    parser.add_argument(
        "--no-concurrent-project-benchmark-confirmed",
        action="store_true",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = run_timing_preconditions(
            project_root=args.project_root,
            execution_commit=args.execution_commit,
            verification_date=args.verification_date,
            no_concurrent_project_benchmark_confirmed=(
                args.no_concurrent_project_benchmark_confirmed
            ),
            output_path=args.output,
        )
    except (DINOv2TimingPreflightError, FileExistsError, OSError) as error:
        print(f"error: {error}")
        return 1
    observed = report["target_machine"]["observed"]
    resource_policy = report["target_machine"]["resource_policy"]
    print(
        "v0.2 CPU timing preconditions: "
        f"status={report['decision']['status']}, "
        f"outcome={report['decision']['outcome']}, "
        f"ram_bytes={observed['ram_bytes']}, "
        f"total_ram_is_gating={resource_policy['total_ram_is_gating']}, "
        f"next_step={report['decision']['next_step']}, "
        f"output={args.output.as_posix()}"
    )
    return 0 if report["decision"]["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
