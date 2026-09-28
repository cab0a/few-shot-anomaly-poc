"""Isolated DINOv2 worker for the fixed development-only normal probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from few_shot_anomaly_poc.v0_3_dinov2_probe import check_runtime, run_worker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--input-manifest-sha256")
    parser.add_argument("--state-root", type=Path)
    args = parser.parse_args()
    try:
        if args.check_runtime:
            print(json.dumps(check_runtime(ROOT), sort_keys=True))
        else:
            if (
                args.input_manifest is None
                or args.input_manifest_sha256 is None
                or args.state_root is None
            ):
                parser.error("the fixed input manifest, hash and state directory are required")
            run_worker(
                ROOT,
                args.input_manifest,
                expected_sha256=args.input_manifest_sha256,
                state_root=args.state_root,
            )
    except Exception as error:
        print(f"STOP: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
