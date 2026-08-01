from __future__ import annotations

import csv
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from few_shot_anomaly_poc.dinov2_timing import (
    QUERY_IDS,
    _worker_command,
    select_resolution_candidate,
    write_timing_observations_csv,
)


def _worker_report(resolution: int, *, passed: bool) -> dict[str, object]:
    return {
        "decision": {"status": "pass" if passed else "fail"},
        "resolution": resolution,
    }


def test_resolution_selection_prefers_448_then_224_without_weighted_score() -> None:
    both = select_resolution_candidate(
        [_worker_report(224, passed=True), _worker_report(448, passed=True)]
    )
    only_224 = select_resolution_candidate(
        [_worker_report(224, passed=True), _worker_report(448, passed=False)]
    )
    neither = select_resolution_candidate(
        [_worker_report(224, passed=False), _worker_report(448, passed=False)]
    )

    assert both["selected_resolution_candidate"] == 448
    assert both["next_step"] == "PROCEED_TO_OFFLINE_REPRODUCTION"
    assert only_224["selected_resolution_candidate"] == 224
    assert only_224["next_step"] == "PROCEED_TO_OFFLINE_REPRODUCTION"
    assert neither["selected_resolution_candidate"] is None
    assert neither["next_step"] == "DO_NOT_PROCEED"
    assert neither["status"] == "fail"


def test_worker_command_fixes_isolation_and_resolution_order_inputs(tmp_path: Path) -> None:
    command = _worker_command(
        python_executable=tmp_path / "environment/bin/python",
        worker_script=tmp_path / "scripts/worker.py",
        project_root=tmp_path,
        artifact_dir=tmp_path / "assets",
        source_root=tmp_path / "source",
        environment_root=tmp_path / "environment",
        execution_commit="a" * 40,
        verification_date="2026-08-01",
        precondition_path=tmp_path / "preconditions.json",
        input_store_path=tmp_path / "inputs.npy",
        input_manifest_path=tmp_path / "manifest.json",
        resolution=224,
        output_path=tmp_path / "resolution-224.json",
        observations_csv_path=tmp_path / "resolution-224.csv",
    )

    assert command[1:3] == ["-I", "-B"]
    assert command[command.index("--resolution") + 1] == "224"
    assert command[command.index("--execution-commit") + 1] == "a" * 40
    assert "--input-store" in command
    assert "--observations-csv" in command


def test_observation_csv_preserves_fixed_columns_and_failure(tmp_path: Path) -> None:
    path = tmp_path / "observations.csv"
    observations = [
        {
            "asset_id": QUERY_IDS[0],
            "duration_ns": 123,
            "failure": None,
            "invocation_index": 0,
            "pass_index": 0,
            "query_index": 0,
            "score": 0.25,
            "status": "success",
        },
        {
            "asset_id": QUERY_IDS[1],
            "duration_ns": None,
            "failure": {
                "category": "memory_error",
                "exception_type": "builtins.MemoryError",
            },
            "invocation_index": 1,
            "pass_index": 0,
            "query_index": 1,
            "score": None,
            "status": "failure",
        },
    ]

    write_timing_observations_csv(path, observations=observations)

    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["asset_id"] == QUERY_IDS[0]
    assert rows[0]["duration_ns"] == "123"
    assert rows[0]["failure_category"] == ""
    assert rows[1]["failure_category"] == "memory_error"
    assert rows[1]["failure_exception_type"] == "builtins.MemoryError"


def test_worker_commands_run_in_distinct_sequential_processes(tmp_path: Path) -> None:
    worker_script = tmp_path / "fake_worker.py"
    worker_script.write_text(
        textwrap.dedent(
            """
            import argparse
            import json
            import os
            from pathlib import Path

            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument("--resolution", type=int, required=True)
            parser.add_argument("--output", type=Path, required=True)
            args, _ = parser.parse_known_args()
            args.output.write_text(
                json.dumps({"pid": os.getpid(), "resolution": args.resolution}),
                encoding="utf-8",
            )
            """
        ).lstrip(),
        encoding="utf-8",
    )
    records: list[dict[str, int]] = []
    for resolution in (224, 448):
        output_path = tmp_path / f"resolution-{resolution}.json"
        command = _worker_command(
            python_executable=Path(sys.executable),
            worker_script=worker_script,
            project_root=tmp_path,
            artifact_dir=tmp_path / "assets",
            source_root=tmp_path / "source",
            environment_root=tmp_path / "environment",
            execution_commit="a" * 40,
            verification_date="2026-08-01",
            precondition_path=tmp_path / "preconditions.json",
            input_store_path=tmp_path / "inputs.npy",
            input_manifest_path=tmp_path / "manifest.json",
            resolution=resolution,
            output_path=output_path,
            observations_csv_path=tmp_path / f"resolution-{resolution}.csv",
        )

        subprocess.run(command, check=True)
        records.append(json.loads(output_path.read_text(encoding="utf-8")))

    assert [record["resolution"] for record in records] == [224, 448]
    assert records[0]["pid"] != records[1]["pid"]
