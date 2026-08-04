"""Run the fixed v0.2.5 label-free final-test scoring stage."""

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
)
from few_shot_anomaly_poc.v0_2_boundary_preparation import (  # noqa: E402
    RUN_ID,
    V0_2BoundaryPreparationError,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (  # noqa: E402
    V0_2EvaluationContractError,
)
from few_shot_anomaly_poc.v0_2_label_free_scoring import (  # noqa: E402
    V0_2LabelFreeScoringError,
    run_v0_2_label_free_scoring,
)
from few_shot_anomaly_poc.v0_2_scoring_artifacts import (  # noqa: E402
    V0_2ScoringArtifactError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score the 200 opaque final-test assets with all three frozen methods, "
            "record three CPU timing passes, and publish no labels or metrics."
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
    parser.add_argument(
        "--confirm-no-concurrent-project-benchmark",
        action="store_true",
        help="Confirm that no other project benchmark is intentionally running.",
    )
    return parser


def _under_project(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def main() -> int:
    args = _parser().parse_args()
    project_root = args.project_root.resolve()
    try:
        counts = run_v0_2_label_free_scoring(
            project_root=project_root,
            execution_commit=args.execution_commit,
            external_root=_under_project(project_root, args.external_root),
            public_root=_under_project(project_root, args.output_root),
            work_root=_under_project(project_root, args.work_root),
            confirm_no_concurrent_project_benchmark=(args.confirm_no_concurrent_project_benchmark),
            progress=lambda message: print(message, flush=True),
        )
    except (
        DINOv2TimingPreflightError,
        FileExistsError,
        OSError,
        V0_2BoundaryPreparationError,
        V0_2EvaluationContractError,
        V0_2LabelFreeScoringError,
        V0_2ScoringArtifactError,
    ) as error:
        print(f"error: {error}")
        return 1
    print("v0.2.5 label-free scoring stage complete:")
    for method, count in counts.items():
        print(f"  {method}: {count} opaque assets")
    print("No final-test label, semantic path, metric, failure case, or decision was accessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
