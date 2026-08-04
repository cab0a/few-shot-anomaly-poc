"""Run one fixed v0.2.6 method reproduction in a fresh offline process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE = PROJECT_ROOT / "src"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

from few_shot_anomaly_poc.v0_2_boundary_preparation import (  # noqa: E402
    V0_2BoundaryPreparationError,
)
from few_shot_anomaly_poc.v0_2_offline_reproduction_run import (  # noqa: E402
    V0_2OfflineReproductionError,
    run_reproduction_worker,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate the fixed first ten opaque scores for one method offline."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fitted-state", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        run_reproduction_worker(
            project_root=args.project_root,
            execution_commit=args.execution_commit,
            method=args.method,
            expected_path=args.expected,
            input_path=args.input,
            fitted_state_path=args.fitted_state,
            report_path=args.report,
        )
    except (
        FileExistsError,
        OSError,
        V0_2BoundaryPreparationError,
        V0_2OfflineReproductionError,
    ) as error:
        print(f"error: {error}")
        return 1
    print(f"{args.method}: fixed first-ten offline reproduction complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
