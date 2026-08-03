from __future__ import annotations

import hashlib
import json
from pathlib import Path

from few_shot_anomaly_poc.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts/v0.2/preflight/final-decision"
EXPECTED_SHA256 = {
    "artifact-manifest.json": (
        "e678d361fc232f8a421a7391b9a6cae0bfbf848019cbd9a419168eb96e67e5b4"
    ),
    "boundary-feasibility.json": (
        "12960eb2f05d5e5928e6fea14afe59e062bf607db85ab3cd38bb87f0cb5a01c0"
    ),
    "final-decision.json": (
        "04da3c64f181c05411b8fdaec2b56db15a38812978815f4c2a1fcac37a9e0298"
    ),
}


def _json(name: str) -> dict[str, object]:
    return json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))


def test_committed_final_preflight_files_have_fixed_identity() -> None:
    actual_names = {path.name for path in ARTIFACT_DIR.iterdir() if path.is_file()}

    assert actual_names == set(EXPECTED_SHA256)
    assert {
        name: sha256_file(ARTIFACT_DIR / name) for name in sorted(actual_names)
    } == EXPECTED_SHA256


def test_committed_boundary_feasibility_preserves_untouched_boundary() -> None:
    report = _json("boundary-feasibility.json")

    assert report["schema_version"] == "v0.2-opaque-boundary-feasibility-v1"
    assert report["execution"] == {
        "execution_commit": "2c075087dc8313a4e6fcc5c968232a921379e857",
        "verification_date": "2026-08-03",
        "worktree_clean": True,
    }
    assert report["boundary"] == {
        "boundary_prepared": False,
        "dataset_access": False,
        "dataset_labels_accessed": False,
        "image_decode_performed": False,
        "network_access": False,
        "official_split_access": False,
        "scoring_performed": False,
        "synthetic_fixture_only": True,
    }
    assert report["decision"] == {"status": "pass"}
    assert all(report["checks"].values())
    assert report["fixture"] == {
        "asset_count": 3,
        "protected_values_published": False,
        "raw_fixture_bytes_published": False,
    }


def test_committed_final_decision_applies_all_ordered_hard_gates() -> None:
    report = _json("final-decision.json")

    assert report["schema_version"] == "v0.2-final-preflight-decision-v1"
    assert [condition["condition"] for condition in report["conditions"]] == list(
        range(1, 11)
    )
    assert [condition["name"] for condition in report["conditions"]] == [
        "preregistration_identity",
        "source_and_license",
        "dependency_resolution",
        "checkpoint_acquisition",
        "third_party_separation",
        "target_machine",
        "execution_integrity",
        "cpu_result",
        "reproducibility",
        "evaluation_boundary",
    ]
    assert all(condition["status"] == "pass" for condition in report["conditions"])
    assert report["decision"] == {
        "first_failed_condition": None,
        "next_step": "CREATE_SEPARATE_V0_2_METHOD_AND_EVALUATION_PREREGISTRATION",
        "outcome": "PROCEED",
        "scope": (
            "Authorizes only a separate v0.2 method-and-evaluation preregistration; "
            "it does not adopt DINOv2 or authorize unregistered performance claims."
        ),
        "weighted_score_used": False,
    }


def test_committed_manifest_matches_published_artifacts() -> None:
    manifest = _json("artifact-manifest.json")
    records = manifest["artifacts"]

    assert manifest["schema_version"] == "v0.2-final-preflight-artifact-manifest-v1"
    assert manifest["execution_commit"] == "2c075087dc8313a4e6fcc5c968232a921379e857"
    assert [record["path"] for record in records] == [
        "boundary-feasibility.json",
        "final-decision.json",
    ]
    for record in records:
        path = ARTIFACT_DIR / record["path"]
        assert path.stat().st_size == record["byte_count"]
        assert sha256_file(path) == record["sha256"]


def test_committed_final_preflight_excludes_secrets_labels_and_machine_paths() -> None:
    serialized = "\n".join(
        (ARTIFACT_DIR / name).read_text(encoding="utf-8")
        for name in sorted(EXPECTED_SHA256)
    )

    assert "/home/" not in serialized
    assert "\\\\wsl" not in serialized
    assert "fixture-class" not in serialized
    assert "visual-one" not in serialized
    assert "visual-two" not in serialized
    assert "visual-three" not in serialized
    fixture_key = hashlib.sha256(b"synthetic opaque boundary feasibility only").hexdigest()
    assert fixture_key not in serialized
