from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

import few_shot_anomaly_poc.v0_3_observation_join as module
from few_shot_anomaly_poc.v0_3_blinded_review import (
    V0_3BlindedReviewError,
    build_review_completion_checkpoint,
    serialize_blind_observations,
    serialize_review_completion_checkpoint,
)
from few_shot_anomaly_poc.v0_3_diagnostic_contract import (
    ARTIFACT_CONTRACT_VERSION,
    BLIND_OBSERVATION_FIELDS,
    DIAGNOSTIC_ID,
    EXPECTED_ASSET_IDS,
    EXPECTED_METHOD_CASES,
    METHODS,
    V0_3DiagnosticContractError,
    load_v0_3_config,
    load_v0_3_schema,
    sha256_file,
)
from few_shot_anomaly_poc.v0_3_inventory import serialize_csv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_v0_3_config(ROOT / "configs/v0.3.yaml")
SCHEMA = load_v0_3_schema(ROOT / "schemas/v0.3/diagnostic-artifacts.json")


@pytest.fixture
def observations() -> list[dict]:
    return [
        {
            "contract_version": ARTIFACT_CONTRACT_VERSION,
            "diagnostic_id": DIAGNOSTIC_ID,
            "asset_id": asset_id,
            "review_status": "reviewed",
            **{field: "uncertain" for field in BLIND_OBSERVATION_FIELDS[4:11]},
            "visible_observation": "Synthetic observation, preserved verbatim.",
            "inferred_cause": None,
        }
        for asset_id in EXPECTED_ASSET_IDS
    ]


@pytest.fixture
def links() -> list[dict]:
    return [
        {
            "contract_version": ARTIFACT_CONTRACT_VERSION,
            "diagnostic_id": DIAGNOSTIC_ID,
            "method": method,
            "case_type": case_type,
            "rank": rank,
            "asset_id": asset_id,
            "parent_failure_sha256": CONFIG["parent_evidence"]["failure_case_sha256"][method],
        }
        for method, case_type, rank, asset_id in EXPECTED_METHOD_CASES
    ]


@pytest.fixture
def metadata_checkout(tmp_path, observations, monkeypatch) -> Path:
    root = tmp_path / "repository"
    for relative in ("configs", "schemas", module.PARENT_ROOT):
        shutil.copytree(ROOT / relative, root / relative)
    for relative in (
        Path(CONFIG["preregistration"]["document"]),
        Path("uv.lock"),
        Path("environments/v0.2-preflight/uv.lock"),
        *(module.ARTIFACT_ROOT / name for name in module.INVENTORY_HASHES),
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    output = root / module.ARTIFACT_ROOT
    (output / "blind-observations.csv").write_bytes(serialize_blind_observations(observations))
    checkpoint = build_review_completion_checkpoint(
        review_assets_sha256=module.INVENTORY_HASHES["review-assets.csv"],
        blind_observations_sha256=sha256_file(output / "blind-observations.csv"),
    )
    (output / "review-completion-checkpoint.json").write_bytes(
        serialize_review_completion_checkpoint(checkpoint)
    )
    monkeypatch.setattr(
        module,
        "OBSERVATION_HASHES",
        {name: sha256_file(output / name) for name in module.OBSERVATION_HASHES},
    )
    return root


def test_join_preserves_forms_and_shared_asset_in_exact_case_order(observations, links):
    records = module.join_observations(observations, links, schema=SCHEMA)
    assert len(records) == 29
    assert len({row["asset_id"] for row in records}) == 28
    assert sum(row["asset_id"] == "asset-000112" for row in records) == 2
    by_asset = {row["asset_id"]: row for row in observations}
    for row in records:
        assert {field: row[field] for field in BLIND_OBSERVATION_FIELDS} == by_asset[
            row["asset_id"]
        ]
        assert "parent_failure_sha256" not in row
    first = module.serialize_observation_join(records, schema=SCHEMA)
    second = module.serialize_observation_join(
        module.join_observations(observations, links, schema=SCHEMA), schema=SCHEMA
    )
    assert first == second
    assert b'"Synthetic observation, preserved verbatim."' in first
    assert b"\r" not in first


@pytest.mark.parametrize("target", ["observations", "links"])
@pytest.mark.parametrize("change", ["missing", "duplicate", "reverse"])
def test_invalid_input_set_or_order_is_rejected(observations, links, target, change):
    rows = observations if target == "observations" else links
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    else:
        rows.reverse()
    with pytest.raises((V0_3DiagnosticContractError, module.V0_3ObservationJoinError)):
        module.join_observations(observations, links, schema=SCHEMA)


def test_visual_counts_deduplicate_fields_and_assets_within_each_method(observations, links):
    shared = next(row for row in observations if row["asset_id"] == "asset-000112")
    for field in (
        "framing_or_crop_difference_visible",
        "pose_or_registration_offset_visible",
        "blur_visible",
    ):
        shared[field] = "yes"
    records = module.join_observations(observations, links, schema=SCHEMA)
    counts = module.visual_yes_counts(records + records, config=CONFIG)
    for method in METHODS:
        expected = int(method != "patch_hog_ocsvm")
        assert counts[method] == {
            "framing_or_pose": expected,
            "global_exposure": 0,
            "blur": expected,
        }


def test_preflight_needs_only_committed_metadata(metadata_checkout, monkeypatch):
    original_open = Path.open

    def metadata_only(path, *args, **kwargs):
        relative = path.relative_to(metadata_checkout)
        assert relative.parts[0] not in {"data", "work"}
        assert path.suffix not in {".jpg", ".png", ".pkl", ".pt"}
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", metadata_only)
    _, _, observations, links = module.load_join_inputs(metadata_checkout)
    assert len(observations) == 28
    assert len(links) == 29
    assert not (metadata_checkout / module.ARTIFACT_ROOT / module.JOIN_NAME).exists()


@pytest.mark.parametrize(
    "name",
    ["blind-observations.csv", "review-completion-checkpoint.json", "review-case-linkage.csv"],
)
def test_changed_fixed_metadata_is_rejected(metadata_checkout, name):
    path = metadata_checkout / module.ARTIFACT_ROOT / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(module.V0_3ObservationJoinError, match="SHA-256 changed"):
        module.load_join_inputs(metadata_checkout)


@pytest.mark.parametrize(
    "field,value",
    [
        ("first_pass_complete", False),
        ("protected_metadata_exposed", True),
        ("method_metadata_joined", True),
    ],
)
def test_invalid_completion_boundary_rejected_even_with_matching_hash(
    metadata_checkout, field, value, monkeypatch
):
    path = metadata_checkout / module.ARTIFACT_ROOT / "review-completion-checkpoint.json"
    checkpoint = json.loads(path.read_text("utf-8"))
    checkpoint[field] = value
    path.write_text(json.dumps(checkpoint), encoding="utf-8")
    monkeypatch.setitem(module.OBSERVATION_HASHES, path.name, sha256_file(path))
    with pytest.raises(V0_3BlindedReviewError):
        module.load_join_inputs(metadata_checkout)


def test_join_never_overwrites_existing_bytes(tmp_path, observations, links):
    content = module.serialize_observation_join(
        module.join_observations(observations, links, schema=SCHEMA), schema=SCHEMA
    )
    target = tmp_path / module.JOIN_NAME
    digest = module.write_observation_join(target, content)
    assert sha256_file(target) == digest
    with pytest.raises(module.V0_3ObservationJoinError, match="already exists"):
        module.write_observation_join(target, b"replacement")
    assert target.read_bytes() == content


@pytest.mark.parametrize("state", ["pushed", "dirty", "local_ahead", "remote_ahead"])
def test_source_must_be_clean_pushed_and_after_observations(tmp_path, monkeypatch, state):
    calls = []

    def git(_root, *args):
        calls.append(args)
        if args[0] == "status":
            return "dirty" if state == "dirty" else ""
        if args[0] == "rev-parse":
            return "b" * 40 if args[1] == "origin/main" and state == "local_ahead" else "a" * 40
        if args[0] == "ls-remote":
            return ("b" if state == "remote_ahead" else "a") * 40 + " refs/heads/main"
        return ""

    monkeypatch.setattr(module, "_git", git)
    if state == "pushed":
        assert module.check_pushed_source(tmp_path) == "a" * 40
        assert ("merge-base", "--is-ancestor", module.OBSERVATION_COMMIT, "a" * 40) in calls
    else:
        with pytest.raises(module.V0_3ObservationJoinError):
            module.check_pushed_source(tmp_path)


def test_cli_preflight_then_join_ignores_old_review_session(metadata_checkout, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location(
        "join_cli", ROOT / "scripts/run_v0_3_5_observation_join.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "__file__", str(metadata_checkout / "scripts/join.py"))
    monkeypatch.setattr(cli, "check_pushed_source", lambda _root: "a" * 40)
    session = metadata_checkout / "work/v0.3.4/first-observation"
    session.mkdir(parents=True)
    (session / "preserved.txt").write_text("keep", encoding="utf-8")
    output = metadata_checkout / module.ARTIFACT_ROOT / module.JOIN_NAME
    monkeypatch.setattr(cli.sys, "argv", ["join.py", "--check-only"])
    assert cli.main() == 0
    assert not output.exists()
    monkeypatch.setattr(cli.sys, "argv", ["join.py"])
    assert cli.main() == 0
    assert "29 method-case records joined" in capsys.readouterr().out
    original = output.read_bytes()
    assert cli.main() == 2
    assert "join output already exists" in capsys.readouterr().err
    assert output.read_bytes() == original
    assert (session / "preserved.txt").read_text("utf-8") == "keep"


def test_invalid_join_schema_rejected_before_serialization(observations, links):
    records = module.join_observations(observations, links, schema=SCHEMA)
    records[0]["anomaly_score"] = 0.5
    with pytest.raises(V0_3DiagnosticContractError):
        module.serialize_observation_join(records, schema=SCHEMA)


def test_linkage_serialization_matches_fixed_contract(links):
    assert (
        serialize_csv(module.REVIEW_CASE_FIELDS, links)
        == (ROOT / module.ARTIFACT_ROOT / "review-case-linkage.csv").read_bytes()
    )
