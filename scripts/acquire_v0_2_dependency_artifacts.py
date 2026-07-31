"""Acquire and inspect the exact v0.2 dependency wheels without installing them."""

from __future__ import annotations

import argparse
from pathlib import Path

from few_shot_anomaly_poc.dependency_artifacts import (
    INSTALLATION_DECISIONS,
    DependencyArtifactError,
    acquire_and_inspect_locked_wheels,
)

DEFAULT_LOCK = Path("environments/v0.2-preflight/uv.lock")
DEFAULT_ENVIRONMENT = Path("environments/v0.2-preflight/pyproject.toml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire the exact v0.2 wheels outside Git, verify their locked hashes, "
            "and inspect ZIP, RECORD, metadata, license, notice, and native-file "
            "content without installation, extraction, import, or execution."
        )
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--installation-decision",
        choices=sorted(INSTALLATION_DECISIONS),
        required=True,
    )
    parser.add_argument("--decision-reason", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = acquire_and_inspect_locked_wheels(
            lock_path=args.lock,
            environment_path=args.environment,
            artifact_dir=args.artifact_dir,
            output_path=args.output,
            project_root=args.project_root,
            reuse_existing=args.reuse_existing,
            installation_decision=args.installation_decision,
            decision_reason=args.decision_reason,
        )
    except (DependencyArtifactError, FileExistsError, OSError) as error:
        print(f"error: {error}")
        return 1
    summary = report["summary"]
    print(
        "dependency artifact inspection passed: "
        f"distributions={summary['distribution_count']}, "
        f"license_material={summary['license_material_count']}, "
        f"native_files={summary['native_file_count']}, "
        f"decision={report['decision']['installation']}, "
        f"output={args.output.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
