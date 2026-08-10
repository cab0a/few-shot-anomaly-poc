"""Run the fixed v0.2.7 one-way label reveal and evaluation stage."""

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
from few_shot_anomaly_poc.v0_2_label_reveal_run import (  # noqa: E402
    V0_2LabelRevealRunError,
    run_v0_2_label_reveal_evaluation,
)
from few_shot_anomaly_poc.v0_2_revealed_evaluation import (  # noqa: E402
    V0_2RevealedEvaluationError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reveal the fixed opaque labels once, calculate method metrics, "
            "and select failure records without reading image pixels."
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


def _under(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def main() -> int:
    args = _parser().parse_args()
    project_root = args.project_root.resolve()
    try:
        counts = run_v0_2_label_reveal_evaluation(
            project_root=project_root,
            execution_commit=args.execution_commit,
            external_root=_under(project_root, args.external_root),
            public_root=_under(project_root, args.output_root),
            work_root=_under(project_root, args.work_root),
        )
    except (
        FileExistsError,
        OSError,
        OpaqueBoundaryError,
        V0_2BoundaryPreparationError,
        V0_2EvaluationContractError,
        V0_2LabelRevealRunError,
        V0_2RevealedEvaluationError,
    ) as error:
        print(f"error: {error}")
        return 1
    print("v0.2.7 label reveal, metrics, and failure selection completed once.")
    print(f"Revealed label records: {counts['revealed_labels']}")
    print("Method metric files: 3")
    print("Method failure-selection files: 3")
    print("No image pixel was read or displayed, and no decision was generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
