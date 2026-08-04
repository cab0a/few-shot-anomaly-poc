from __future__ import annotations

import hashlib
from pathlib import Path

from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    METHODS,
    load_v0_2_artifact_schema,
)
from few_shot_anomaly_poc.v0_2_offline_reproduction_run import (
    ASSET_COUNT,
    _write_reproduction_csv,
    build_reproduction_records,
    read_reproduction_csv,
)
from few_shot_anomaly_poc.v0_2_pre_reveal_checkpoint import label_free_bundle_sha256

ROOT = Path(__file__).resolve().parents[1]


def _scores(method: str, *, offset: float = 0.0) -> list[dict]:
    return [
        {
            "asset_id": f"asset-{index:06d}",
            "score_status": "ok",
            "score_failure_code": None,
            "anomaly_score": 0.1 + index / 100.0 + offset,
            "method": method,
        }
        for index in range(ASSET_COUNT)
    ]


def test_first_ten_reproduction_round_trips_fixed_contract(tmp_path: Path) -> None:
    schema = load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json")
    records = build_reproduction_records(
        method="ecc_residual",
        expected=_scores("ecc_residual"),
        reproduced=_scores("ecc_residual", offset=0.5e-6),
        schema=schema,
    )
    output = tmp_path / "offline-reproduction.csv"

    _write_reproduction_csv(output, records)
    reloaded = read_reproduction_csv(output, schema=schema)

    assert len(reloaded) == ASSET_COUNT
    assert all(record["within_tolerance"] for record in reloaded)
    assert reloaded[0]["asset_id"] == "asset-000000"
    assert reloaded[-1]["asset_id"] == "asset-000009"


def test_reproduction_failure_is_retained_without_rerun_or_tolerance_change() -> None:
    schema = load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json")
    reproduced = _scores("patch_hog_ocsvm")
    reproduced[4]["score_status"] = "failed"
    reproduced[4]["score_failure_code"] = "PATCH_HOG_SCORE_EXECUTION_FAILED"
    reproduced[4]["anomaly_score"] = 1e12

    records = build_reproduction_records(
        method="patch_hog_ocsvm",
        expected=_scores("patch_hog_ocsvm"),
        reproduced=reproduced,
        schema=schema,
    )

    assert records[4]["within_tolerance"] is False
    assert records[4]["reproduced_score_status"] == "failed"
    assert records[4]["reproduced_failure_code"] == "PATCH_HOG_SCORE_EXECUTION_FAILED"
    assert records[4]["absolute_difference"] > 1e-6


def test_label_free_bundle_identity_hashes_sorted_path_and_file_hash_pairs(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_bytes(b"alpha\n")
    (tmp_path / "b.txt").write_bytes(b"beta\n")
    expected = hashlib.sha256()
    for relative_path in ("a.txt", "b.txt"):
        file_hash = hashlib.sha256((tmp_path / relative_path).read_bytes()).hexdigest()
        expected.update(relative_path.encode())
        expected.update(b"\0")
        expected.update(file_hash.encode("ascii"))
        expected.update(b"\n")

    observed = label_free_bundle_sha256(tmp_path, ["a.txt", "b.txt"])

    assert observed == expected.hexdigest()


def test_fixed_method_inventory_remains_three_methods() -> None:
    assert METHODS == (
        "ecc_residual",
        "patch_hog_ocsvm",
        "dinov2_vits14_224_nn",
    )
