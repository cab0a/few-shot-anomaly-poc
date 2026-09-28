"""Run metadata preflight and the first fixed observation on the data-owning PC."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from few_shot_anomaly_poc.v0_3_blinded_review import V0_3BlindedReviewError
from few_shot_anomaly_poc.v0_3_diagnostic_contract import V0_3DiagnosticContractError
from few_shot_anomaly_poc.v0_3_first_observation import (
    ARTIFACT_ROOT,
    preflight_review,
    run_first_observation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path("data/external/v0.2/evaluation/visa-pcb2-v0-2-final"),
    )
    parser.add_argument(
        "--fitted-state-root",
        type=Path,
        default=Path("work/v0.2/evaluation/visa-pcb2-v0-2-final/fitted-state"),
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Verify metadata/state; open no images"
    )
    parser.add_argument("--reviewer-kind", choices=("human", "isolated_agent"))
    args = parser.parse_args()
    if not args.check_only and not args.reviewer_kind:
        parser.error("--reviewer-kind is required for a real observation")
    root = Path(__file__).resolve().parents[1]
    external_root = args.external_root.absolute()
    fitted_root = args.fitted_state_root.absolute()
    session_root = root / "work/v0.3.4/first-observation"
    try:
        preflight = preflight_review(
            repository_root=root,
            external_root=external_root,
            fitted_state_root=fitted_root,
            session_root=session_root,
        )
        print(f"Preflight passed: source_commit={preflight.source_commit}", flush=True)
        if args.check_only:
            print("No image accessed; no session or observation written.")
            return 0
        print(f"Reviewer exchange: {session_root / 'reviewer'}", flush=True)
        print(
            "Awaiting one fixed response per image. Do not give the reviewer repository context.",
            flush=True,
        )
        run_first_observation(
            preflight=preflight,
            assets_root=external_root / "scorer/assets",
            output_root=root / ARTIFACT_ROOT,
            session_root=session_root,
            reviewer_kind=args.reviewer_kind,
        )
    except (
        OSError,
        ValueError,
        subprocess.CalledProcessError,
        V0_3BlindedReviewError,
        V0_3DiagnosticContractError,
    ) as error:
        print(f"STOP: {error}", file=sys.stderr)
        return 2
    print("28 observations hash-locked. No method join or anomaly scoring performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
