from __future__ import annotations

import csv
import json
from pathlib import Path

from few_shot_anomaly_poc.v0_3_blinded_review import (
    read_blind_observations_csv,
    validate_review_completion_checkpoint,
)
from few_shot_anomaly_poc.v0_3_diagnostic_contract import (
    EXPECTED_ASSET_IDS,
    EXPECTED_METHOD_CASES,
    load_v0_3_schema,
    sha256_file,
    validate_checkpoint_record,
    validate_tabular_record,
)
from few_shot_anomaly_poc.v0_3_observation_join import (
    JOIN_NAME,
    join_observations,
    serialize_observation_join,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/v0.3/diagnostics/pcb2-development"
SCHEMA = load_v0_3_schema(ROOT / "schemas/v0.3/diagnostic-artifacts.json")
EXPECTED_HASHES = {
    "review-assets.csv": "399ebef79498917ff0cf5b2bf83467541a5417dce74bdd8e0e462fec70a7fd58",
    "review-case-linkage.csv": ("861cf3428808de3e300278d54be84de95fa42204393050358c9c2b592a638d7e"),
    "normal-diagnostic-partition.csv": (
        "39415c626652c4cbfb856e68d8b559d0f119392f869e0569d47165df5a48e110"
    ),
    "pre-access-checkpoint.json": (
        "a8cade456af699df734007e3b575babd1f624f390dbf19728f912e8a43960df0"
    ),
}
OBSERVATION_HASHES = {
    "blind-observations.csv": (
        "93a6303824dc9b24819b4027f409b1acb3752086ca72921ac30a2cebbe6783f3"
    ),
    "review-completion-checkpoint.json": (
        "3c9b56ecd9761ca2a35b3bfbb17fefff5021776b33cc082f8cf1c15e72767dbd"
    ),
}


def _read_csv(name: str) -> list[dict[str, str]]:
    with (ARTIFACT_ROOT / name).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _typed_review_asset(record: dict[str, str]) -> dict:
    return {
        **record,
        "byte_count": int(record["byte_count"]),
    }


def _typed_review_link(record: dict[str, str]) -> dict:
    return {
        **record,
        "rank": int(record["rank"]),
    }


def _typed_normal(record: dict[str, str]) -> dict:
    return {
        **record,
        "selection_rank": int(record["selection_rank"]),
        "byte_count": int(record["byte_count"]),
    }


def test_committed_v0_3_2_file_inventory_and_hashes_are_exact() -> None:
    files = sorted(path.name for path in ARTIFACT_ROOT.iterdir() if path.is_file())

    required = set(EXPECTED_HASHES) | set(OBSERVATION_HASHES)
    assert set(files) in (required, required | {JOIN_NAME})
    assert {name: sha256_file(ARTIFACT_ROOT / name) for name in EXPECTED_HASHES} == EXPECTED_HASHES
    assert not list(ARTIFACT_ROOT.rglob("*.jpg"))
    assert not list(ARTIFACT_ROOT.rglob("*.png"))


def test_committed_v0_3_4_observations_are_immutable_complete_and_hash_bound() -> None:
    observation_path = ARTIFACT_ROOT / "blind-observations.csv"
    checkpoint_path = ARTIFACT_ROOT / "review-completion-checkpoint.json"
    assert {
        name: sha256_file(ARTIFACT_ROOT / name) for name in OBSERVATION_HASHES
    } == OBSERVATION_HASHES
    checkpoint = validate_review_completion_checkpoint(
        json.loads(checkpoint_path.read_text("utf-8"))
    )
    assert checkpoint["review_assets_sha256"] == EXPECTED_HASHES["review-assets.csv"]
    records = read_blind_observations_csv(
        observation_path,
        expected_sha256=checkpoint["blind_observations_sha256"],
        schema=SCHEMA,
    )
    assert [record["asset_id"] for record in records] == list(EXPECTED_ASSET_IDS)
    assert all(record["review_status"] == "reviewed" for record in records)
    assert all(record["inferred_cause"] is None for record in records)


def test_later_join_if_present_matches_every_immutable_observation() -> None:
    path = ARTIFACT_ROOT / JOIN_NAME
    if not path.exists():
        return
    observations = read_blind_observations_csv(
        ARTIFACT_ROOT / "blind-observations.csv",
        expected_sha256=OBSERVATION_HASHES["blind-observations.csv"],
        schema=SCHEMA,
    )
    links = [_typed_review_link(record) for record in _read_csv("review-case-linkage.csv")]
    expected = join_observations(observations, links, schema=SCHEMA)
    assert path.read_bytes() == serialize_observation_join(expected, schema=SCHEMA)


def test_committed_review_assets_are_exact_safe_ordered_identities() -> None:
    rows = [_typed_review_asset(record) for record in _read_csv("review-assets.csv")]

    assert len(rows) == 28
    assert [record["asset_id"] for record in rows] == list(EXPECTED_ASSET_IDS)
    for record in rows:
        validate_tabular_record("review_asset", record, schema=SCHEMA)
        assert "relative_path" not in record
        assert "method" not in record
        assert "true_class" not in record
        assert "anomaly_score" not in record


def test_committed_later_join_linkage_matches_all_29_parent_cases() -> None:
    rows = [_typed_review_link(record) for record in _read_csv("review-case-linkage.csv")]
    identities = tuple(
        (record["method"], record["case_type"], record["rank"], record["asset_id"])
        for record in rows
    )

    assert identities == EXPECTED_METHOD_CASES
    assert sum(record["asset_id"] == "asset-000112" for record in rows) == 2
    for record in rows:
        validate_tabular_record("review_case_link", record, schema=SCHEMA)


def test_committed_normal_partition_is_complete_ranked_and_unique() -> None:
    rows = [_typed_normal(record) for record in _read_csv("normal-diagnostic-partition.csv")]

    assert len(rows) == 60
    assert [record["selection_rank"] for record in rows] == list(range(1, 61))
    assert len({record["relative_path"] for record in rows}) == 60
    assert rows[0]["selection_sha256"] == (
        "005c964c73458e1f42eeb4add98fc341ba58bb8f3b18b98227c0484e2923d276"
    )
    assert rows[0]["relative_path"] == "pcb2/Data/Images/Normal/0632.JPG"
    assert rows[-1]["selection_sha256"] == (
        "10d1931272017d1bc9e72794f7fe50a532b63b355527e44d2c1be5f8d28e984a"
    )
    assert rows[-1]["relative_path"] == "pcb2/Data/Images/Normal/0770.JPG"
    for record in rows:
        validate_tabular_record("normal_partition", record, schema=SCHEMA)


def test_committed_checkpoint_keeps_every_v0_3_access_flag_closed() -> None:
    checkpoint = json.loads(
        (ARTIFACT_ROOT / "pre-access-checkpoint.json").read_text(encoding="utf-8")
    )

    assert validate_checkpoint_record(checkpoint) == checkpoint
    assert checkpoint["source_commit"] == "98147e8541340f63ea12713caaf310a1395761fa"
    assert checkpoint["review_asset_count"] == 28
    assert checkpoint["method_case_count"] == 29
    assert checkpoint["normal_partition_count"] == 60
    closed_flags = [
        "diagnostic_image_accessed",
        "diagnostic_image_decoded",
        "diagnostic_image_displayed",
        "diagnostic_score_computed",
        "v0_2_final_test_rescored",
        "unselected_final_test_image_accessed",
        "protected_metadata_exposed",
        "v0_2_artifacts_modified",
        "preexisting_output_detected",
        "output_overwrite_performed",
        "raw_data_in_git",
    ]
    assert all(checkpoint[field] is False for field in closed_flags)
