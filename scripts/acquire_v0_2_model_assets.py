"""Acquire and inspect the fixed v0.2 DINOv2 source and checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from few_shot_anomaly_poc.model_assets import ModelAssetError, acquire_model_assets


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire the fixed DINOv2 source archive and ViT-S/14 checkpoint "
            "outside Git, then verify their hashes and container structure without "
            "source execution, checkpoint deserialization, model construction, "
            "tensor operations, dataset access, or benchmarking."
        )
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--acquisition-date", required=True)
    parser.add_argument("--acquisition-base-commit", required=True)
    parser.add_argument("--preregistration-commit", required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("docs/v0.2-preflight-preregistration.md"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = acquire_model_assets(
            artifact_dir=args.artifact_dir,
            output_path=args.output,
            project_root=args.project_root,
            acquisition_date=args.acquisition_date,
            acquisition_base_commit=args.acquisition_base_commit,
            preregistration_commit=args.preregistration_commit,
            preregistration_path=args.preregistration,
        )
    except (FileExistsError, ModelAssetError, OSError) as error:
        print(f"error: {error}")
        return 1
    print(
        "controlled model-asset acquisition passed: "
        f"source_sha256={report['source']['artifact']['observed_sha256']}, "
        f"checkpoint_sha256={report['checkpoint']['artifact']['observed_sha256']}, "
        f"decision={report['decision']['next_step']}, "
        f"output={args.output.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
