"""Run the fixed v0.2.8 ordered hard-gate decision stage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE = PROJECT_ROOT / "src"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

from few_shot_anomaly_poc.v0_2_boundary_preparation import RUN_ID  # noqa: E402
from few_shot_anomaly_poc.v0_2_decision import V0_2DecisionError  # noqa: E402
from few_shot_anomaly_poc.v0_2_decision_run import (  # noqa: E402
    V0_2DecisionRunError,
    run_v0_2_decision,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (  # noqa: E402
    V0_2EvaluationContractError,
)
from few_shot_anomaly_poc.v0_2_offline_reproduction_run import (  # noqa: E402
    V0_2OfflineReproductionError,
)
from few_shot_anomaly_poc.v0_2_revealed_evaluation import (  # noqa: E402
    V0_2RevealedEvaluationError,
)
from few_shot_anomaly_poc.v0_2_scoring_artifacts import (  # noqa: E402
    V0_2ScoringArtifactError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the fixed v0.2 ordered hard gates, create per-method and project "
            "decisions, and seal the complete evaluation artifact manifest."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(f"artifacts/v0.2/evaluation/{RUN_ID}"),
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path(f"data/external/v0.2/evaluation/{RUN_ID}"),
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
        result = run_v0_2_decision(
            project_root=project_root,
            execution_commit=args.execution_commit,
            public_root=_under(project_root, args.output_root),
            external_root=_under(project_root, args.external_root),
            work_root=_under(project_root, args.work_root),
        )
    except (
        FileExistsError,
        OSError,
        V0_2DecisionError,
        V0_2DecisionRunError,
        V0_2EvaluationContractError,
        V0_2OfflineReproductionError,
        V0_2RevealedEvaluationError,
        V0_2ScoringArtifactError,
    ) as error:
        print(f"error: {error}")
        return 1
    print("v0.2.8 ordered hard-gate decision completed once.")
    for method, decision in result["method_decisions"].items():
        print(f"{method}: {decision}")
    print(f"project: {result['project_decision']}")
    print(f"selected method: {result['selected_method']}")
    print(f"manifest entries: {result['manifest_file_count']}")
    print("No image pixel was read or displayed, and no prior artifact was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
