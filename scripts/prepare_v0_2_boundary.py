"""Prepare the fixed v0.2 pcb2 evaluation boundary without image decoding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE = PROJECT_ROOT / "src"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

from few_shot_anomaly_poc.errors import DataPreparationError  # noqa: E402
from few_shot_anomaly_poc.opaque_boundary import OpaqueBoundaryError  # noqa: E402
from few_shot_anomaly_poc.v0_2_boundary_preparation import (  # noqa: E402
    RUN_ID,
    V0_2BoundaryPreparationError,
    prepare_v0_2_boundary,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (  # noqa: E402
    V0_2EvaluationContractError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare fixed pcb2 normal partitions and an opaque final-test boundary. "
            "This command reads split labels only into a protected mapping; it does not "
            "decode, display, fit, calibrate, score, or evaluate an image."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/external/archives/VisA_20220922.tar"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("data/external/splits/visa-1cls.csv"),
    )
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
    return parser


def _under_project(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def main() -> int:
    args = _parser().parse_args()
    project_root = args.project_root.resolve()
    try:
        result = prepare_v0_2_boundary(
            project_root=project_root,
            execution_commit=args.execution_commit,
            archive_path=_under_project(project_root, args.archive),
            split_path=_under_project(project_root, args.split),
            external_root=_under_project(project_root, args.external_root),
            public_artifact_root=_under_project(project_root, args.output_root),
        )
    except (
        DataPreparationError,
        FileExistsError,
        OSError,
        OpaqueBoundaryError,
        V0_2BoundaryPreparationError,
        V0_2EvaluationContractError,
    ) as error:
        print(f"error: {error}")
        return 1
    print(
        "v0.2 boundary prepared: "
        f"run_id={RUN_ID}, reference={result.reference_count}, "
        f"calibration={result.calibration_count}, final_test={result.final_test_count}, "
        f"boundary_record_sha256={result.public_boundary_record_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
