"""Create the fixed v0.2.6 checkpoint before label reveal."""

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
from few_shot_anomaly_poc.v0_2_pre_reveal_checkpoint import (  # noqa: E402
    V0_2PreRevealCheckpointError,
    create_pre_reveal_checkpoint,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bind all committed and pushed v0.2 label-free evidence before reveal."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--evidence-commit", required=True)
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
    return parser


def _under(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def main() -> int:
    args = _parser().parse_args()
    project_root = args.project_root.resolve()
    try:
        checkpoint = create_pre_reveal_checkpoint(
            project_root=project_root,
            evidence_commit=args.evidence_commit,
            public_root=_under(project_root, args.output_root),
            external_root=_under(project_root, args.external_root),
        )
    except (
        FileExistsError,
        OSError,
        V0_2BoundaryPreparationError,
        V0_2EvaluationContractError,
        V0_2PreRevealCheckpointError,
    ) as error:
        print(f"error: {error}")
        return 1
    print("v0.2.6 pre-reveal checkpoint created:")
    print(f"  label-free evidence commit: {checkpoint['git_commit']}")
    print(f"  label-free bundle SHA-256: {checkpoint['label_free_bundle_sha256']}")
    print(
        "No label, semantic path, official split, sealed mapping, metric, or decision was accessed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
