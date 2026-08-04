"""Run the fixed v0.2.6 offline reproduction stage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE = PROJECT_ROOT / "src"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

from few_shot_anomaly_poc.v0_2_boundary_preparation import (  # noqa: E402
    RUN_ID,
    V0_2BoundaryPreparationError,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (  # noqa: E402
    V0_2EvaluationContractError,
)
from few_shot_anomaly_poc.v0_2_offline_reproduction_run import (  # noqa: E402
    V0_2OfflineReproductionError,
    run_v0_2_offline_reproduction,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce each method's first ten committed opaque scores offline."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path(f"data/external/v0.2/evaluation/{RUN_ID}"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(f"artifacts/v0.2/evaluation/{RUN_ID}"),
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(f"work/v0.2/evaluation/{RUN_ID}"),
    )
    return parser


def _under(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def main() -> int:
    args = _parser().parse_args()
    project_root = args.project_root.resolve()
    try:
        statuses = run_v0_2_offline_reproduction(
            project_root=project_root,
            execution_commit=args.execution_commit,
            external_root=_under(project_root, args.external_root),
            public_root=_under(project_root, args.output_root),
            work_root=_under(project_root, args.work_root),
        )
    except (
        FileExistsError,
        OSError,
        V0_2BoundaryPreparationError,
        V0_2EvaluationContractError,
        V0_2OfflineReproductionError,
    ) as error:
        print(f"error: {error}")
        return 1
    print("v0.2.6 fixed first-ten offline reproduction complete:")
    for method, status in statuses.items():
        print(f"  {method}: {status}")
    print(
        "No label, semantic path, official split, sealed mapping, metric, or decision was accessed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
