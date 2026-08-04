from __future__ import annotations

from pathlib import Path

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    METHODS,
    load_v0_2_artifact_schema,
)
from few_shot_anomaly_poc.v0_2_offline_reproduction_run import (
    ASSET_COUNT,
    read_reproduction_csv,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "visa-pcb2-v0-2-final"
ARTIFACT_ROOT = ROOT / "artifacts/v0.2/evaluation" / RUN_ID
EXPECTED_HASHES = {
    "ecc_residual": "beb6a4211885e12623b53739aca23dabe62ad6968296cc6364dbb019b1341833",
    "patch_hog_ocsvm": "7fa501f12c3ab76bbfc1bf2d88ca79d74df8eb82b0b8e61e9544017122578f0d",
    "dinov2_vits14_224_nn": (
        "f75976fb3ef3c17db6f5ee4150bcba4a9c3bb79d6172920e22345f9e59c93dd5"
    ),
}
FORBIDDEN_FIELDS = {
    "true_class",
    "class_label",
    "source_path",
    "official_split",
    "sealed_mapping",
    "hmac_key",
}


def test_committed_v0_2_first_ten_reproduction_hashes_are_fixed() -> None:
    observed = {
        method: sha256_file(ARTIFACT_ROOT / method / "offline-reproduction.csv")
        for method in METHODS
    }

    assert observed == EXPECTED_HASHES


def test_committed_v0_2_first_ten_reproduction_passes_exactly() -> None:
    schema = load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json")
    for method in METHODS:
        records = read_reproduction_csv(
            ARTIFACT_ROOT / method / "offline-reproduction.csv",
            schema=schema,
        )

        assert len(records) == ASSET_COUNT
        assert [record["asset_id"] for record in records] == [
            f"asset-{index:06d}" for index in range(ASSET_COUNT)
        ]
        assert all(record["within_tolerance"] for record in records)
        assert all(record["expected_score_status"] == "ok" for record in records)
        assert all(record["reproduced_score_status"] == "ok" for record in records)
        assert max(record["absolute_difference"] for record in records) == 0.0


def test_committed_v0_2_reproduction_headers_remain_label_free() -> None:
    for method in METHODS:
        path = ARTIFACT_ROOT / method / "offline-reproduction.csv"
        header = path.read_text(encoding="utf-8").splitlines()[0]

        assert not FORBIDDEN_FIELDS.intersection(header.split(","))
