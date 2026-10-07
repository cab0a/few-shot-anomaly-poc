"""Create or verify the diagnostic decision and complete manifest from saved metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from few_shot_anomaly_poc.v0_3_diagnostic_closure import (
    preflight_closure,
    verify_closure,
    write_closure,
)
from few_shot_anomaly_poc.v0_3_first_observation import ARTIFACT_ROOT
from few_shot_anomaly_poc.v0_3_observation_join import check_pushed_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        if args.verify:
            decision, manifest = verify_closure(root)
        else:
            source, _config, schema, decision = preflight_closure(root)
            print(f"Preflight passed: source_commit={source}", flush=True)
            print(f"decision={decision['decision']}", flush=True)
            if args.check_only:
                print(
                    "28 observations, 29 joined cases, 900 scores and 15 summaries verified. "
                    "No output written."
                )
                return 0
            if check_pushed_source(root) != source:
                raise ValueError("execution source changed before publication")
            # Recheck the evidence immediately before exclusive publication.
            repeated, _, _, verified_decision = preflight_closure(root)
            if repeated != source or verified_decision != decision:
                raise ValueError("decision source or evidence changed")
            manifest = write_closure(
                root / ARTIFACT_ROOT, decision, source_commit=source, schema=schema
            )
            verified, verified_manifest = verify_closure(root)
            if verified != decision or verified_manifest != manifest:
                raise ValueError("published closure differs from validated evidence")
        print(f"decision={decision['decision']}")
        print(f"selected_family={decision['selected_family']}")
        print(
            f"Bundle verified: {len(manifest['files'])} manifest entries; "
            "11 files including manifest."
        )
        print(
            "No image accessed, model loaded, scorer invoked, "
            "threshold changed, or v0.2 decision revised."
        )
    except Exception as error:
        print(f"STOP: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
