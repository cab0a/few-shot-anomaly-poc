"""Run the fixed v0.2.4 normal-reference fitting and calibration stage."""

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
from few_shot_anomaly_poc.v0_2_calibration_artifacts import (  # noqa: E402
    V0_2CalibrationArtifactError,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (  # noqa: E402
    V0_2EvaluationContractError,
)
from few_shot_anomaly_poc.v0_2_normal_calibration import (  # noqa: E402
    V0_2NormalCalibrationError,
    run_v0_2_normal_fit_and_calibration,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the three fixed v0.2 methods from 20 normal references and fix "
            "one threshold per successfully fitted method from the 881 known-normal "
            "calibration inputs. Final-test assets and labels remain inaccessible."
        )
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


def _under_project(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def main() -> int:
    args = _parser().parse_args()
    project_root = args.project_root.resolve()
    try:
        statuses = run_v0_2_normal_fit_and_calibration(
            project_root=project_root,
            execution_commit=args.execution_commit,
            external_root=_under_project(project_root, args.external_root),
            public_artifact_root=_under_project(project_root, args.output_root),
            work_root=_under_project(project_root, args.work_root),
            progress=lambda message: print(message, flush=True),
        )
    except (
        FileExistsError,
        OSError,
        V0_2BoundaryPreparationError,
        V0_2CalibrationArtifactError,
        V0_2EvaluationContractError,
        V0_2NormalCalibrationError,
    ) as error:
        print(f"error: {error}")
        return 1
    print("v0.2.4 normal-only stage complete:")
    for method, status in statuses.items():
        print(f"  {method}: {status}")
    print("No final-test asset, label, sealed mapping, metric, latency, or decision was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
