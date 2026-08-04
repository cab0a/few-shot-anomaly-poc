"""Run the v0.2.4 DINOv2 normal-only worker in the fixed isolated environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE = PROJECT_ROOT / "src"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

from few_shot_anomaly_poc.dinov2_timing import DINOv2TimingError  # noqa: E402
from few_shot_anomaly_poc.model_compatibility import (  # noqa: E402
    ModelCompatibilityError,
)
from few_shot_anomaly_poc.v0_2_dinov2_calibration import (  # noqa: E402
    V0_2DINOv2CalibrationError,
    run_dinov2_normal_fit_and_calibration,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (  # noqa: E402
    V0_2EvaluationContractError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the fixed DINOv2 224-pixel CPU method from 20 normal references "
            "and calibrate its threshold from normal inputs only. This isolated "
            "worker receives no final-test asset, label, sealed mapping, or HMAC key."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--input-store", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        fit = run_dinov2_normal_fit_and_calibration(
            project_root=args.project_root,
            execution_commit=args.execution_commit,
            input_store_path=args.input_store,
            input_manifest_path=args.input_manifest,
            acquisition_path=args.acquisition_record,
            import_smoke_path=args.import_smoke_record,
            strict_load_path=args.strict_load_record,
            artifact_dir=args.artifact_dir,
            source_root=args.source_root,
            environment_root=args.environment_root,
            output_dir=args.output_dir,
            state_path=args.state_path,
        )
    except (
        DINOv2TimingError,
        FileExistsError,
        ModelCompatibilityError,
        OSError,
        V0_2DINOv2CalibrationError,
        V0_2EvaluationContractError,
    ) as error:
        print(f"error: {error}")
        return 1
    print(
        "DINOv2 normal-only stage complete: "
        f"status={fit['status']}, "
        f"successful_references={fit['successful_reference_count']}, "
        f"failed_references={fit['failed_reference_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
