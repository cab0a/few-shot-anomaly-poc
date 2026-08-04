from __future__ import annotations

import json
import subprocess
from pathlib import Path

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    METHODS,
    load_v0_2_artifact_schema,
    load_v0_2_config,
    validate_json_artifact,
)
from few_shot_anomaly_poc.v0_2_pre_reveal_checkpoint import (
    CHECKPOINT_NAME,
    FIXED_HASHES,
    label_free_bundle_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "visa-pcb2-v0-2-final"
ARTIFACT_ROOT = ROOT / "artifacts/v0.2/evaluation" / RUN_ID
CHECKPOINT_PATH = ARTIFACT_ROOT / CHECKPOINT_NAME
CHECKPOINT_SHA256 = "4bf7d63da308868933a8967a6922fc439f95a524bf54cd8a38ff39f894219b07"
LABEL_FREE_BUNDLE_SHA256 = (
    "b3fb2a1283117350cf3c016ae284d3558aa7ff9ea1cf1c6beb65eba54c5cb389"
)
EVIDENCE_COMMIT = "30e830e3f805f2410c669c0b8c2da6cb27894ca5"
SCORING_SOURCE_COMMIT = "ba23a2fe12a715161b420bc7d73d42f4de3bfc8c"


def _checkpoint() -> dict:
    value = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_committed_pre_reveal_checkpoint_matches_the_frozen_contract() -> None:
    config = load_v0_2_config(ROOT / "configs/v0.2.yaml")
    schema = load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json")

    checkpoint = validate_json_artifact(
        "pre_reveal_checkpoint",
        _checkpoint(),
        config=config,
        schema=schema,
    )

    assert sha256_file(CHECKPOINT_PATH) == CHECKPOINT_SHA256
    assert checkpoint["source_commit"] == SCORING_SOURCE_COMMIT
    assert checkpoint["git_commit"] == EVIDENCE_COMMIT
    assert checkpoint["git_push_verified"] is True
    assert checkpoint["labels_accessed"] is False
    assert checkpoint["method_order"] == list(METHODS)
    assert checkpoint["method_score_counts"] == {method: 200 for method in METHODS}
    assert checkpoint["reproduction_status"] == {method: "pass" for method in METHODS}


def test_committed_pre_reveal_bundle_identity_covers_every_prior_artifact() -> None:
    paths = set(FIXED_HASHES)
    paths.update(f"{method}/offline-reproduction.csv" for method in METHODS)

    observed = label_free_bundle_sha256(ARTIFACT_ROOT, sorted(paths))

    assert len(paths) == 23
    assert observed == LABEL_FREE_BUNDLE_SHA256
    assert _checkpoint()["label_free_bundle_sha256"] == observed


def test_label_free_evidence_commit_precedes_the_checkpoint_commit() -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EVIDENCE_COMMIT, "HEAD"],
        cwd=ROOT,
        check=False,
    )

    assert completed.returncode == 0
