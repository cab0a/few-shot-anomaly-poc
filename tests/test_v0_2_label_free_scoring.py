from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from few_shot_anomaly_poc.v0_2_label_free_scoring import (
    ASSET_COUNT,
    OPAQUE_MANIFEST_SCHEMA,
    _run_dinov2_worker,
    _validate_opaque_assets,
)

ROOT = Path(__file__).resolve().parents[1]


def test_opaque_asset_validation_accepts_only_contiguous_byte_verified_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer_root = tmp_path / "scorer"
    assets_root = scorer_root / "assets"
    assets_root.mkdir(parents=True)
    records = []
    for index in range(ASSET_COUNT):
        asset_id = f"asset-{index:06d}"
        content = f"opaque-{index}".encode()
        path = assets_root / f"{asset_id}.jpg"
        path.write_bytes(content)
        records.append(
            {
                "asset_id": asset_id,
                "byte_count": len(content),
                "relative_path": f"assets/{asset_id}.jpg",
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    manifest_path = scorer_root / "scoring-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"records": records, "schema_version": OPAQUE_MANIFEST_SCHEMA},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "few_shot_anomaly_poc.v0_2_label_free_scoring.OPAQUE_SCORING_MANIFEST_SHA256",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )

    assets = _validate_opaque_assets(scorer_root)

    assert len(assets) == ASSET_COUNT
    assert assets[0].asset_id == "asset-000000"
    assert assets[-1].asset_id == "asset-000199"


def test_dinov2_worker_command_has_no_label_or_sealed_boundary_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    def fake_run(command, **_kwargs):
        observed.extend(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "few_shot_anomaly_poc.v0_2_label_free_scoring.subprocess.run",
        fake_run,
    )
    _run_dinov2_worker(
        project_root=ROOT,
        execution_commit="a" * 40,
        store_path=tmp_path / "rgb.npy",
        manifest_path=tmp_path / "rgb.json",
        state_path=tmp_path / "state.pt",
        threshold=0.25,
        output_root=tmp_path / "output",
        report_path=tmp_path / "report.json",
        progress=None,
    )

    serialized = " ".join(observed).lower()
    assert "sealed" not in serialized
    assert "boundary-state" not in serialized
    assert "label" not in serialized
    assert "source-path" not in serialized
    assert "hmac" not in serialized
