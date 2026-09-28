"""Join the 28 immutable blinded observations to 29 fixed method-case records."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from few_shot_anomaly_poc.v0_3_blinded_review import V0_3BlindedReviewError
from few_shot_anomaly_poc.v0_3_diagnostic_contract import V0_3DiagnosticContractError
from few_shot_anomaly_poc.v0_3_observation_join import (
    ARTIFACT_ROOT,
    INVENTORY_HASHES,
    JOIN_NAME,
    OBSERVATION_HASHES,
    V0_3ObservationJoinError,
    check_pushed_source,
    join_observations,
    load_join_inputs,
    serialize_observation_join,
    visual_yes_counts,
    write_observation_join,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify metadata without writing")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / ARTIFACT_ROOT / JOIN_NAME
    try:
        if output.exists() or output.is_symlink():
            raise V0_3ObservationJoinError(
                "join output already exists; preserve it for verification"
            )
        source = check_pushed_source(root)
        expected_inputs = set(INVENTORY_HASHES) | set(OBSERVATION_HASHES)
        if {path.name for path in output.parent.iterdir()} != expected_inputs:
            raise V0_3ObservationJoinError("unexpected diagnostic output already exists")
        config, schema, observations, links = load_join_inputs(root)
        print(f"Preflight passed: source_commit={source}", flush=True)
        if args.check_only:
            print("28 observations and 29 case links verified. No output written.")
            return 0
        records = join_observations(observations, links, schema=schema)
        content = serialize_observation_join(records, schema=schema)
        load_join_inputs(root)
        if check_pushed_source(root) != source:
            raise V0_3ObservationJoinError("execution source changed before writing")
        digest = write_observation_join(output, content)
    except (
        OSError,
        ValueError,
        subprocess.CalledProcessError,
        V0_3BlindedReviewError,
        V0_3DiagnosticContractError,
        V0_3ObservationJoinError,
    ) as error:
        print(f"STOP: {error}", file=sys.stderr)
        return 2
    print("29 method-case records joined from 28 immutable observations.")
    print(f"output={output.relative_to(root).as_posix()}")
    print(f"sha256={digest}")
    print("Unique visual yes counts (descriptive only):")
    print(json.dumps(visual_yes_counts(records, config=config), indent=2))
    print("No images accessed or scored. Controlled probe and diagnostic decision remain pending.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
