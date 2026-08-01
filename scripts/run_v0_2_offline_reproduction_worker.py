"""Run the fixed 224 DINOv2 score reproduction in an isolated process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE = PROJECT_ROOT / "src"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

from few_shot_anomaly_poc.dinov2_reproduction import (  # noqa: E402
    DINOv2ReproductionError,
    run_reproduction_worker,
)
from few_shot_anomaly_poc.dinov2_timing import DINOv2TimingError  # noqa: E402
from few_shot_anomaly_poc.model_compatibility import (  # noqa: E402
    ModelCompatibilityError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the first 10 fixed 224 DINOv2 scores from local assets "
            "without timing, VisA, labels, thresholds, or network access."
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
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = run_reproduction_worker(
            acquisition_path=args.acquisition_record,
            import_smoke_path=args.import_smoke_record,
            strict_load_path=args.strict_load_record,
            artifact_dir=args.artifact_dir,
            source_root=args.source_root,
            environment_root=args.environment_root,
            project_root=args.project_root,
            execution_commit=args.execution_commit,
            verification_date=args.verification_date,
            output_root=args.output_root,
        )
    except (
        DINOv2ReproductionError,
        DINOv2TimingError,
        FileExistsError,
        ModelCompatibilityError,
        OSError,
    ) as error:
        print(f"error: {error}")
        return 1
    print(
        "DINOv2 offline score reproduction worker: "
        f"resolution={report['execution']['resolution']}, "
        f"status={report['decision']['status']}, "
        f"next_step={report['decision']['next_step']}"
    )
    return 0 if report["decision"]["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
