from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pytest

import few_shot_anomaly_poc.v0_3_first_observation as module
from few_shot_anomaly_poc.v0_3_blinded_review import (
    FORM_FIELDS,
    BlankObservationForm,
    BlindReviewItem,
    ReviewAssetIdentity,
    V0_3BlindedReviewError,
    read_blind_observations_csv,
)
from few_shot_anomaly_poc.v0_3_diagnostic_contract import EXPECTED_ASSET_IDS, load_v0_3_schema
from few_shot_anomaly_poc.v0_3_first_observation import (
    INVENTORY_HASHES,
    FileExchangeReviewer,
    ReviewPreflight,
    run_first_observation,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = load_v0_3_schema(ROOT / "schemas/v0.3/diagnostic-artifacts.json")


def _response() -> dict:
    return {
        "review_status": "reviewed",
        "framing_or_crop_difference_visible": "uncertain",
        "pose_or_registration_offset_visible": "uncertain",
        "global_exposure_difference_visible": "uncertain",
        "blur_visible": "no",
        "localized_change_visible": "yes",
        "diffuse_change_visible": "no",
        "no_obvious_visible_change": "no",
        "visible_observation": "Synthetic alternating colored pixels are visible.",
        "inferred_cause": None,
    }


@pytest.fixture
def synthetic_run(tmp_path: Path) -> dict:
    assets_root = tmp_path / "assets"
    assets_root.mkdir()
    output_root = tmp_path / "output"
    output_root.mkdir()
    fixed_files = {}
    for name in INVENTORY_HASHES:
        path = output_root / name
        path.write_bytes(b"synthetic fixed metadata")
        fixed_files[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    pixels = np.arange(9 * 13 * 3, dtype=np.uint8).reshape(9, 13, 3)
    success, encoded = cv2.imencode(".png", pixels)
    assert success
    payload = encoded.tobytes()
    assets = []
    for asset_id in EXPECTED_ASSET_IDS:
        (assets_root / f"{asset_id}.jpg").write_bytes(payload)
        assets.append(
            ReviewAssetIdentity(asset_id, len(payload), hashlib.sha256(payload).hexdigest())
        )
    return {
        "preflight": ReviewPreflight("a" * 40, tuple(assets), SCHEMA, fixed_files),
        "assets_root": assets_root,
        "output_root": output_root,
        "session_root": tmp_path / "session",
        "reviewer_kind": "isolated_agent",
    }


def test_complete_synthetic_run_decodes_once_and_hash_locks_before_join(
    synthetic_run: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded = []
    observed = []
    original_decode = module.decode_verified_opaque_asset

    def decode(identity, **kwargs):
        decoded.append(identity.asset_id)
        return original_decode(identity, **kwargs)

    def reviewer(item):
        observed.append(item.asset_id)
        assert item.pixels.shape == (9, 13, 3)
        assert not item.pixels.flags.writeable
        assert item.form == BlankObservationForm()
        assert not hasattr(item, "method")
        return _response()

    monkeypatch.setattr(module, "decode_verified_opaque_asset", decode)
    checkpoint = run_first_observation(**synthetic_run, reviewer_factory=lambda _root: reviewer)
    assert decoded == observed == list(EXPECTED_ASSET_IDS)
    assert checkpoint["first_pass_complete"] is True
    assert checkpoint["method_metadata_joined"] is False
    records = read_blind_observations_csv(
        synthetic_run["output_root"] / "blind-observations.csv",
        expected_sha256=checkpoint["blind_observations_sha256"],
        schema=SCHEMA,
    )
    assert len(records) == 28
    assert not list(synthetic_run["output_root"].glob("*.png"))
    operator = synthetic_run["session_root"] / "operator"
    assert len(list(operator.glob("*.attempt.json"))) == 28
    assert len(list(operator.glob("*.response.json"))) == 28
    assert not (operator / "stopped.json").exists()


@pytest.mark.parametrize("failure", ["changed_image", "invalid_form", "interrupt"])
def test_failure_preserves_attempt_and_stops_before_any_completion(
    synthetic_run: dict,
    failure: str,
) -> None:
    seen = []
    if failure == "changed_image":
        (synthetic_run["assets_root"] / f"{EXPECTED_ASSET_IDS[2]}.jpg").write_bytes(b"changed")

    def reviewer(item):
        seen.append(item.asset_id)
        if failure == "interrupt":
            raise KeyboardInterrupt
        if failure == "invalid_form":
            return {**_response(), "method": "protected"}
        return _response()

    expected = KeyboardInterrupt if failure == "interrupt" else V0_3BlindedReviewError
    with pytest.raises(expected):
        run_first_observation(**synthetic_run, reviewer_factory=lambda _root: reviewer)
    assert not (synthetic_run["output_root"] / "blind-observations.csv").exists()
    assert not (synthetic_run["output_root"] / "review-completion-checkpoint.json").exists()
    stopped = json.loads((synthetic_run["session_root"] / "operator/stopped.json").read_text())
    assert stopped["first_pass_complete"] is False
    assert len(stopped["attempted_asset_ids"]) == (3 if failure == "changed_image" else 1)
    assert len(seen) == (2 if failure == "changed_image" else 1)
    with pytest.raises(V0_3BlindedReviewError, match="session already exists"):
        run_first_observation(**synthetic_run, reviewer_factory=lambda _root: reviewer)


def test_existing_public_output_rejects_before_image_decode(
    synthetic_run: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = synthetic_run["output_root"] / "blind-observations.csv"
    existing.write_bytes(b"preserve me")
    monkeypatch.setattr(
        module, "decode_verified_opaque_asset", lambda *_a, **_k: pytest.fail("decode")
    )
    with pytest.raises(V0_3BlindedReviewError, match="already exists"):
        run_first_observation(**synthetic_run)
    assert existing.read_bytes() == b"preserve me"
    assert not synthetic_run["session_root"].exists()


def test_parent_change_during_collection_prevents_completion(synthetic_run: dict) -> None:
    def reviewer(_item):
        next(iter(synthetic_run["preflight"].fixed_files)).write_bytes(b"changed evidence")
        return _response()

    with pytest.raises(V0_3BlindedReviewError, match="identity changed"):
        run_first_observation(**synthetic_run, reviewer_factory=lambda _root: reviewer)
    assert not (synthetic_run["output_root"] / "blind-observations.csv").exists()


def test_audit_storage_failure_never_publishes_completion(
    synthetic_run: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_once = module._write_once

    def fail_collection(path, content):
        if path.name == "collection.json":
            raise OSError("synthetic storage failure")
        write_once(path, content)

    monkeypatch.setattr(module, "_write_once", fail_collection)
    with pytest.raises(OSError, match="storage failure"):
        run_first_observation(
            **synthetic_run,
            reviewer_factory=lambda _root: lambda _item: _response(),
        )
    assert (synthetic_run["output_root"] / "blind-observations.csv").is_file()
    assert not (synthetic_run["output_root"] / "review-completion-checkpoint.json").exists()
    stopped = json.loads((synthetic_run["session_root"] / "operator/stopped.json").read_text())
    assert stopped["first_pass_complete"] is False
    assert len(stopped["response_asset_ids"]) == 28


def test_file_exchange_only_presents_allowlist_and_lossless_pixels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pixels = np.frombuffer(bytes(range(90)), dtype=np.uint8).reshape(5, 6, 3)
    item = BlindReviewItem(EXPECTED_ASSET_IDS[0], pixels, BlankObservationForm())
    directory = tmp_path / item.asset_id

    def respond(_seconds):
        assert (directory / "item.ready").is_file()
        visible = json.loads((directory / "item.json").read_text())
        assert set(visible) == {"asset_id", "pixels", "form"}
        assert visible["asset_id"] == item.asset_id
        assert visible["pixels"] == "pixels.png"
        assert visible["form"] == asdict(BlankObservationForm())
        assert set(visible["form"]) == set(FORM_FIELDS)
        raster = cv2.imread(str(directory / "pixels.png"), cv2.IMREAD_COLOR)
        assert np.array_equal(raster, pixels)
        assert {path.name for path in directory.iterdir()} == {
            "item.json",
            "item.ready",
            "pixels.png",
        }
        (directory / "response.json").write_text(json.dumps(_response()))
        (directory / "response.ready").touch()

    monkeypatch.setattr(module.time, "sleep", respond)
    assert FileExchangeReviewer(tmp_path)(item) == _response()


def test_exchange_timeout_never_fabricates_a_response(tmp_path: Path) -> None:
    item = BlindReviewItem(
        EXPECTED_ASSET_IDS[0],
        np.frombuffer(bytes(12), dtype=np.uint8).reshape(2, 2, 3),
        BlankObservationForm(),
    )
    with pytest.raises(V0_3BlindedReviewError, match="timed out"):
        FileExchangeReviewer(tmp_path, timeout_seconds=0.001)(item)
    assert not (tmp_path / item.asset_id / "response.json").exists()


def test_missing_external_metadata_stops_preflight_without_image_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def git(_root, *arguments):
        if arguments[0] in {"rev-parse", "ls-remote"}:
            return "a" * 40
        return ""

    monkeypatch.setattr(module, "_git", git)
    monkeypatch.setattr(
        module, "decode_verified_opaque_asset", lambda *_a, **_k: pytest.fail("decode")
    )
    session = tmp_path / "session"
    with pytest.raises(V0_3BlindedReviewError, match="required regular file is missing"):
        module.preflight_review(
            repository_root=ROOT,
            external_root=tmp_path / "absent",
            fitted_state_root=tmp_path / "absent-state",
            session_root=session,
        )
    assert not session.exists()
