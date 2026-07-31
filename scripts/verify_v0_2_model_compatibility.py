"""Verify weights-only checkpoint and strict local DINOv2 model compatibility."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE = PROJECT_ROOT / "src"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

from few_shot_anomaly_poc.model_compatibility import (  # noqa: E402
    ModelCompatibilityError,
    verify_model_compatibility,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the fixed checkpoint through weights-only CPU loading and an "
            "exact strict load into the fixed local non-register DINOv2 ViT-S/14. "
            "No forward pass, feature extraction, benchmark, dataset access, or "
            "accelerator availability probe is performed."
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
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--verification-date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = verify_model_compatibility(
            acquisition_path=args.acquisition_record,
            import_smoke_path=args.import_smoke_record,
            artifact_dir=args.artifact_dir,
            extraction_dir=args.extraction_dir,
            environment_root=args.environment_root,
            project_root=args.project_root,
            execution_commit=args.execution_commit,
            verification_date=args.verification_date,
            output_path=args.output,
        )
    except (FileExistsError, ModelCompatibilityError, OSError) as error:
        print(f"error: {error}")
        return 1
    print(
        "weights-only strict-load verification passed: "
        f"state_keys={report['checkpoint']['state_dictionary']['key_count']}, "
        f"tensor_elements="
        f"{report['checkpoint']['state_dictionary']['total_tensor_elements']}, "
        f"parameters={report['model']['parameter_count']}, "
        f"decision={report['decision']['next_step']}, "
        f"output={args.output.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
