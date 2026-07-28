from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from few_shot_anomaly_poc.config import load_config
from few_shot_anomaly_poc.hashing import sha256_file

PARTITION_ARTIFACT = Path("artifacts/v0.1/data/pcb1-normal-partitions.csv")
DATASET_RECORD = Path("artifacts/v0.1/data/dataset-record.json")


def test_committed_normal_partitions_match_fixed_selection_rule() -> None:
    config = load_config(Path("configs/v0.1.yaml"))
    dataset_record = json.loads(DATASET_RECORD.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(PARTITION_ARTIFACT.open(encoding="utf-8", newline="")))

    assert sha256_file(PARTITION_ARTIFACT) == dataset_record["partition"]["manifest_sha256"]
    assert len(rows) == 904
    assert len({row["relative_path"] for row in rows}) == len(rows)

    for expected_rank, row in enumerate(rows, start=1):
        expected_partition = (
            "reference"
            if expected_rank <= config.selection.reference_count
            else "calibration"
        )
        expected_digest = hashlib.sha256(
            (
                f"{config.selection.namespace}:{config.selection.seed}:"
                f"{row['relative_path']}"
            ).encode()
        ).hexdigest()

        assert row["partition"] == expected_partition
        assert int(row["selection_rank"]) == expected_rank
        assert row["relative_path"].startswith(f"{config.category}/")
        assert row["selection_sha256"] == expected_digest
