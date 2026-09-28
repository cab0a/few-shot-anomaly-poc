"""Preflight or explicitly execute the fixed 60 x 5 x 3 normal-score probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from few_shot_anomaly_poc.v0_3_controlled_probe import preflight_probe, run_probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Verify prerequisites; no image bytes or model loading",
    )
    mode.add_argument(
        "--execute", action="store_true", help="Begin the one-shot real 900-score probe"
    )
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
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        preflight = preflight_probe(
            root=root,
            external_root=args.external_root.absolute(),
            state_root=args.fitted_state_root.absolute(),
        )
        print(f"Preflight passed: source_commit={preflight.source_commit}", flush=True)
        if args.check_only:
            print("60 normal identities, fixed states and both runtimes verified.")
            print("No image bytes read, model loaded, score computed, or probe session created.")
            return 0
        completed = run_probe(preflight, progress=lambda text: print(text, flush=True))
        print("900 controlled scores and 15 paired summaries completed.")
        print(f"controlled_scores_sha256={completed['controlled_scores_sha256']}")
        print(f"condition_summaries_sha256={completed['condition_summaries_sha256']}")
        print(
            "No final-test rescoring, refitting, threshold change, "
            "latency measurement, or diagnostic decision."
        )
    except Exception as error:
        print(f"STOP: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
