"""Run the bounded synthetic smoke check for the fixed DINOv2 scoring path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE = PROJECT_ROOT / "src"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

from few_shot_anomaly_poc.dinov2_errors import DINOv2ScoringError  # noqa: E402
from few_shot_anomaly_poc.dinov2_scoring_smoke import (  # noqa: E402
    DINOv2ScoringSmokeError,
    run_dinov2_scoring_smoke,
)
from few_shot_anomaly_poc.model_compatibility import (  # noqa: E402
    ModelCompatibilityError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one bounded synthetic reference/query smoke check at 224 and "
            "448 pixels. This performs model inference and fixed scoring but "
            "does not measure latency, access a dataset or labels, calibrate a "
            "threshold, or make an accuracy claim."
        )
    )
    parser.add_argument(
        "--acquisition-record",
        type=Path,
        default=Path("artifacts/v0.2/model-assets/acquisition.json"),
    )
    parser.add_argument(
        "--import-smoke-record",
        type=Path,
        default=Path("artifacts/v0.2/environment/import-smoke.json"),
    )
    parser.add_argument(
        "--strict-load-record",
        type=Path,
        default=Path("artifacts/v0.2/model-compatibility/strict-load.json"),
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--verification-date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = run_dinov2_scoring_smoke(
            acquisition_path=args.acquisition_record,
            import_smoke_path=args.import_smoke_record,
            strict_load_path=args.strict_load_record,
            artifact_dir=args.artifact_dir,
            source_root=args.source_root,
            environment_root=args.environment_root,
            project_root=args.project_root,
            execution_commit=args.execution_commit,
            verification_date=args.verification_date,
            output_path=args.output,
        )
    except (
        DINOv2ScoringError,
        DINOv2ScoringSmokeError,
        FileExistsError,
        ModelCompatibilityError,
        OSError,
    ) as error:
        print(f"error: {error}")
        return 1
    print(
        "fixed DINOv2 scoring-path smoke passed: "
        + ", ".join(
            f"resolution={item['resolution']} "
            f"patches={item['patch_count']} "
            f"top={item['top_patch_count']} "
            f"score={item['score']:.9f}"
            for item in report["resolutions"]
        )
        + f", output={args.output.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
