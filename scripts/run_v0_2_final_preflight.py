"""Run the untouched-boundary feasibility checkpoint and final preflight gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE = PROJECT_ROOT / "src"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

from few_shot_anomaly_poc.opaque_boundary import OpaqueBoundaryError  # noqa: E402
from few_shot_anomaly_poc.v0_2_preflight_decision import (  # noqa: E402
    V0_2PreflightDecisionError,
    run_final_preflight_checkpoint,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify fixed preflight evidence and a synthetic opaque evaluation boundary. "
            "This command does not access VisA, split rows, labels, or images."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--verification-date", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = run_final_preflight_checkpoint(
            project_root=args.project_root,
            execution_commit=args.execution_commit,
            verification_date=args.verification_date,
            output_root=args.output_root,
        )
    except (
        FileExistsError,
        OSError,
        OpaqueBoundaryError,
        V0_2PreflightDecisionError,
    ) as error:
        print(f"error: {error}")
        return 1
    outcome = report["decision"]["outcome"]
    print(
        "v0.2 final preflight: "
        f"outcome={outcome}, next_step={report['decision']['next_step']}"
    )
    return 0 if outcome == "PROCEED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
