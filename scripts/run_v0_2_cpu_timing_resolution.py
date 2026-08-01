"""Run one preregistered DINOv2 resolution in an isolated fresh process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE = PROJECT_ROOT / "src"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

from few_shot_anomaly_poc.dinov2_timing import (  # noqa: E402
    DINOv2TimingError,
    run_timing_resolution_worker,
)
from few_shot_anomaly_poc.model_compatibility import (  # noqa: E402
    ModelCompatibilityError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one fixed DINOv2 CPU timing resolution from a verified local "
            "memory-mapped synthetic input store. This worker must be started "
            "by the parent orchestrator in a fresh isolated process."
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
    parser.add_argument(
        "--precondition-record",
        type=Path,
        default=Path("artifacts/v0.2/cpu-preflight/attempt-002-memory-bounded-pass.json"),
    )
    parser.add_argument("--input-store", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--verification-date", required=True)
    parser.add_argument("--resolution", type=int, choices=(224, 448), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = run_timing_resolution_worker(
            acquisition_path=args.acquisition_record,
            import_smoke_path=args.import_smoke_record,
            strict_load_path=args.strict_load_record,
            precondition_path=args.precondition_record,
            input_store_path=args.input_store,
            input_manifest_path=args.input_manifest,
            artifact_dir=args.artifact_dir,
            source_root=args.source_root,
            environment_root=args.environment_root,
            project_root=args.project_root,
            execution_commit=args.execution_commit,
            verification_date=args.verification_date,
            resolution=args.resolution,
            output_path=args.output,
        )
    except (
        DINOv2TimingError,
        FileExistsError,
        ModelCompatibilityError,
        OSError,
    ) as error:
        print(f"error: {error}")
        return 1
    summary = report["loop"]["summary"] if report["loop"] is not None else None
    print(
        "DINOv2 CPU timing resolution: "
        f"resolution={report['resolution']}, "
        f"status={report['decision']['status']}, "
        f"attempted={report['boundary']['timing_invocation_count']}, "
        f"p95_ns={None if summary is None else summary['p95_ns']}, "
        f"output={args.output.as_posix()}"
    )
    return 0 if report["decision"]["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
