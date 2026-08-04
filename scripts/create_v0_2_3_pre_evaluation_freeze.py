"""Create the v0.2.3 pre-evaluation freeze from the prepared boundary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE = PROJECT_ROOT / "src"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

from few_shot_anomaly_poc.opaque_boundary import OpaqueBoundaryError  # noqa: E402
from few_shot_anomaly_poc.v0_2_boundary_preparation import (  # noqa: E402
    RUN_ID,
    V0_2BoundaryPreparationError,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (  # noqa: E402
    V0_2EvaluationContractError,
)
from few_shot_anomaly_poc.v0_2_freeze_checkpoint import (  # noqa: E402
    MILESTONE_LABEL,
    V0_2FreezeCheckpointError,
    prepare_v0_2_freeze,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the v0.2.3 source, contract, normal-manifest, and opaque-boundary "
            "identities before fitting. This command does not read images or sealed "
            "labels and does not fit, calibrate, score, reveal, or evaluate."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--source-commit", required=True)
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
        record, freeze_sha256 = prepare_v0_2_freeze(
            project_root=project_root,
            source_commit=args.source_commit,
            external_root=_under_project(project_root, args.external_root),
            public_artifact_root=_under_project(project_root, args.output_root),
        )
    except (
        FileExistsError,
        OSError,
        OpaqueBoundaryError,
        V0_2BoundaryPreparationError,
        V0_2EvaluationContractError,
        V0_2FreezeCheckpointError,
    ) as error:
        print(f"error: {error}")
        return 1
    print(
        f"{MILESTONE_LABEL} freeze created: run_id={record['run_id']}, "
        f"source_commit={record['source_commit']}, sha256={freeze_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
