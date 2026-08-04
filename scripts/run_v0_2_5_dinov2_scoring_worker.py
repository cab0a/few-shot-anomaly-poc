"""Run the isolated v0.2.5 DINOv2 scorer with label-free inputs only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE = PROJECT_ROOT / "src"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

from few_shot_anomaly_poc.dinov2_timing import DINOv2TimingError  # noqa: E402
from few_shot_anomaly_poc.model_compatibility import ModelCompatibilityError  # noqa: E402
from few_shot_anomaly_poc.v0_2_dinov2_scoring_run import (  # noqa: E402
    V0_2DINOv2ScoringRunError,
    run_dinov2_label_free_scoring,
)
from few_shot_anomaly_poc.v0_2_scoring_artifacts import (  # noqa: E402
    V0_2ScoringArtifactError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score opaque RGB arrays with the fixed isolated DINOv2 CPU method."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--input-store", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--fitted-state", type=Path, required=True)
    parser.add_argument("--fitted-state-sha256", required=True)
    parser.add_argument("--threshold", type=float, required=True)
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
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        run_dinov2_label_free_scoring(
            project_root=args.project_root,
            execution_commit=args.execution_commit,
            input_store_path=args.input_store,
            input_manifest_path=args.input_manifest,
            fitted_state_path=args.fitted_state,
            fitted_state_sha256=args.fitted_state_sha256,
            threshold=args.threshold,
            acquisition_path=args.acquisition_record,
            import_smoke_path=args.import_smoke_record,
            strict_load_path=args.strict_load_record,
            artifact_dir=args.artifact_dir,
            source_root=args.source_root,
            environment_root=args.environment_root,
            output_dir=args.output_dir,
            report_path=args.report,
        )
    except (
        DINOv2TimingError,
        FileExistsError,
        ModelCompatibilityError,
        OSError,
        V0_2DINOv2ScoringRunError,
        V0_2ScoringArtifactError,
    ) as error:
        print(f"error: {error}")
        return 1
    print("DINOv2 label-free scoring worker completed the fixed record counts.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
