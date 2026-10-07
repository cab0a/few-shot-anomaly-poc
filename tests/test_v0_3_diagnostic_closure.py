from __future__ import annotations

import copy
import json
import math
import shutil
from pathlib import Path

import pytest

import few_shot_anomaly_poc.v0_3_diagnostic_closure as closure
from few_shot_anomaly_poc.v0_3_diagnostic_contract import (
    METHODS,
    load_v0_3_config,
    load_v0_3_schema,
)
from few_shot_anomaly_poc.v0_3_first_observation import ARTIFACT_ROOT
from few_shot_anomaly_poc.v0_3_probe_artifacts import ProbeError, read_table

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ARTIFACT_ROOT
CONFIG = load_v0_3_config(ROOT / "configs/v0.3.yaml")
SCHEMA = load_v0_3_schema(ROOT / "schemas/v0.3/diagnostic-artifacts.json")


@pytest.fixture
def evidence():
    joined = read_table(OUTPUT / closure.JOIN_NAME, "observation_join", schema=SCHEMA)
    summaries = read_table(OUTPUT / closure.SUMMARY_NAME, "condition_summary", schema=SCHEMA)
    for row in joined:
        for definition in CONFIG["signal_rule"]["families"].values():
            for field in definition["visual_fields"]:
                row[field] = "uncertain"
    for row in summaries:
        if row["condition"] != "control":
            row["median_absolute_delta"] = 0.0
            row["normal_to_anomalous_crossing_count"] = 0
    return joined, summaries


def visual_yes(joined, method, family, count=3):
    fields = CONFIG["signal_rule"]["families"][family]["visual_fields"]
    selected = [r for r in joined if r["method"] == method][:count]
    for row in selected:
        for field in fields:
            row[field] = "yes"


def summary_for(summaries, method, condition):
    return next(r for r in summaries if (r["method"], r["condition"]) == (method, condition))


def decide(joined, summaries):
    return closure.build_decision(closure.evaluate_families(joined, summaries, config=CONFIG))


@pytest.mark.parametrize("crossings, expected", [(5, False), (6, True)])
def test_crossing_threshold_is_inclusive(evidence, crossings, expected):
    joined, summaries = evidence
    visual_yes(joined, METHODS[0], "framing_or_pose")
    summary_for(summaries, METHODS[0], "translate_4_4")["normal_to_anomalous_crossing_count"] = (
        crossings
    )
    decision = decide(joined, summaries)
    assert (decision["selected_family"] == "framing_or_pose") is expected


@pytest.mark.parametrize("at_boundary", [False, True])
def test_delta_threshold_is_inclusive(evidence, at_boundary):
    joined, summaries = evidence
    visual_yes(joined, METHODS[0], "blur")
    boundary = 0.1 * CONFIG["methods"][METHODS[0]]["threshold"]
    summary_for(summaries, METHODS[0], "gaussian_blur")["median_absolute_delta"] = (
        boundary if at_boundary else math.nextafter(boundary, -math.inf)
    )
    assert (decide(joined, summaries)["selected_family"] == "blur") is at_boundary


def test_same_method_must_satisfy_both_requirements(evidence):
    joined, summaries = evidence
    visual_yes(joined, METHODS[0], "global_exposure")
    summary_for(summaries, METHODS[1], "brightness_115")["normal_to_anomalous_crossing_count"] = 6
    assert decide(joined, summaries)["decision"] == "STOP: NO BOUNDED DIAGNOSTIC SIGNAL"


def test_descriptive_p95_cannot_replace_the_median_or_crossing_rule(evidence):
    joined, summaries = evidence
    visual_yes(joined, METHODS[0], "framing_or_pose")
    summary_for(summaries, METHODS[0], "translate_4_4")["nearest_rank_p95_absolute_delta"] = 1.0
    assert decide(joined, summaries)["selected_family"] is None


@pytest.mark.parametrize("condition", ["brightness_085", "brightness_115"])
def test_either_brightness_condition_can_qualify(evidence, condition):
    joined, summaries = evidence
    visual_yes(joined, METHODS[1], "global_exposure")
    summary_for(summaries, METHODS[1], condition)["normal_to_anomalous_crossing_count"] = 6
    assert decide(joined, summaries)["selected_family"] == "global_exposure"


def test_duplicate_assets_and_two_pose_fields_count_once(evidence):
    joined, summaries = evidence
    visual_yes(joined, METHODS[0], "framing_or_pose", count=2)
    joined.extend(copy.deepcopy(joined[:2]))
    summary_for(summaries, METHODS[0], "translate_4_4")["normal_to_anomalous_crossing_count"] = 60
    evaluations = closure.evaluate_families(joined, summaries, config=CONFIG)
    ecc = evaluations[0]["method_evaluations"][0]
    assert ecc["visual_yes_unique_asset_count"] == 2
    assert not ecc["supported"]


def test_priority_and_visual_family_scope_remain_fixed(evidence):
    joined, summaries = evidence
    for family, condition in (
        ("framing_or_pose", "translate_4_4"),
        ("global_exposure", "brightness_115"),
        ("blur", "gaussian_blur"),
    ):
        visual_yes(joined, METHODS[0], family)
        summary_for(summaries, METHODS[0], condition)["normal_to_anomalous_crossing_count"] = 6
    assert decide(joined, summaries)["selected_family"] == "framing_or_pose"
    # Uncertain is not yes, and localized change is not an eligible signal family.
    for row in joined:
        row["localized_change_visible"] = "yes"
        for family in CONFIG["signal_rule"]["families"].values():
            for field in family["visual_fields"]:
                row[field] = "uncertain"
    assert decide(joined, summaries)["selected_family"] is None


@pytest.mark.parametrize(
    "violated, complete, expected",
    [(True, True, "INVALID"), (True, False, "INVALID"), (False, False, "INCONCLUSIVE")],
)
def test_invalid_and_incomplete_precede_supported_family(evidence, violated, complete, expected):
    joined, summaries = evidence
    visual_yes(joined, METHODS[0], "blur")
    summary_for(summaries, METHODS[0], "gaussian_blur")["normal_to_anomalous_crossing_count"] = 6
    value = closure.build_decision(
        closure.evaluate_families(joined, summaries, config=CONFIG),
        boundary_violated=violated,
        evidence_complete=complete,
    )
    assert value["decision"].startswith(expected)
    assert value["selected_family"] is None


def test_completed_evidence_produces_stop_and_only_ecc_exposure_numeric_pass():
    _, _, joined, summaries = closure.load_closure_inputs(
        ROOT, completed=(OUTPUT / closure.MANIFEST_NAME).exists()
    )
    value = decide(joined, summaries)
    assert value["decision"] == "STOP: NO BOUNDED DIAGNOSTIC SIGNAL"
    assert value["status"] == "complete"
    assert value["selected_family"] is None
    methods = [
        m
        for f in value["family_evaluations"]
        for m in f["method_evaluations"]
        if m["numeric_requirement_met"]
    ]
    assert len(methods) == 1 and methods[0]["method"] == METHODS[0]
    assert methods[0]["visual_yes_unique_asset_count"] == 0
    assert methods[0]["condition_evaluations"][1]["normal_to_anomalous_crossing_count"] == 13
    assert all(not f["supported"] for f in value["family_evaluations"])


@pytest.fixture
def copied_bundle(tmp_path):
    output = tmp_path / ARTIFACT_ROOT
    output.mkdir(parents=True)
    for name in closure.INPUT_HASHES:
        shutil.copyfile(OUTPUT / name, output / name)
    return tmp_path, output


@pytest.mark.parametrize("mutation", ["hash", "count", "path", "missing", "duplicate", "self"])
def test_manifest_roundtrip_covers_every_artifact_except_itself(copied_bundle, mutation):
    root, output = copied_bundle
    joined = read_table(output / closure.JOIN_NAME, "observation_join", schema=SCHEMA)
    summaries = read_table(output / closure.SUMMARY_NAME, "condition_summary", schema=SCHEMA)
    decision = decide(joined, summaries)
    manifest = closure.write_closure(output, decision, source_commit="a" * 40, schema=SCHEMA)
    # load_join_inputs also verifies committed parent evidence; use the real root for those reads.
    shutil.copytree(ROOT / "configs", root / "configs")
    shutil.copytree(ROOT / "schemas", root / "schemas")
    shutil.copytree(
        ROOT / "artifacts/v0.2/evaluation/visa-pcb2-v0-2-final",
        root / "artifacts/v0.2/evaluation/visa-pcb2-v0-2-final",
    )
    document = CONFIG["preregistration"]["document"]
    (root / document).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / document, root / document)
    shutil.copyfile(ROOT / "uv.lock", root / "uv.lock")
    (root / "environments/v0.2-preflight").mkdir(parents=True)
    shutil.copyfile(
        ROOT / "environments/v0.2-preflight/uv.lock", root / "environments/v0.2-preflight/uv.lock"
    )
    assert closure.verify_closure(root) == (decision, manifest)
    assert len(manifest["files"]) == 10
    names = [f["relative_path"] for f in manifest["files"]]
    assert names == sorted(set(closure.INPUT_HASHES) | {closure.DECISION_NAME})
    assert closure.MANIFEST_NAME not in names
    preserved = (output / closure.DECISION_NAME).read_bytes()
    with pytest.raises(ProbeError, match="already exists"):
        closure.write_closure(output, decision, source_commit="b" * 40, schema=SCHEMA)
    assert (output / closure.DECISION_NAME).read_bytes() == preserved
    if mutation == "hash":
        manifest["files"][0]["sha256"] = "f" * 64
    elif mutation == "count":
        manifest["files"][0]["record_count"] += 1
    elif mutation == "path":
        manifest["files"][0]["relative_path"] = "../outside.json"
    elif mutation == "missing":
        manifest["files"].pop()
    elif mutation == "duplicate":
        manifest["files"].append(manifest["files"][0])
    else:
        manifest["files"][0]["relative_path"] = closure.MANIFEST_NAME
    (output / closure.MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(ProbeError, match="manifest content"):
        closure.verify_closure(root)


def test_changed_input_rejected_before_decision_write(copied_bundle):
    _, output = copied_bundle
    (output / closure.SCORE_NAME).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        closure.write_closure(output, {}, source_commit="a" * 40, schema=SCHEMA)
    assert not (output / closure.DECISION_NAME).exists()


def test_failed_manifest_write_preserves_decision_and_prevents_retry(copied_bundle, monkeypatch):
    _, output = copied_bundle
    joined = read_table(output / closure.JOIN_NAME, "observation_join", schema=SCHEMA)
    summaries = read_table(output / closure.SUMMARY_NAME, "condition_summary", schema=SCHEMA)
    real_write = closure.write_once

    def failing_write(path, content):
        if path.name == closure.MANIFEST_NAME:
            raise OSError("synthetic storage failure")
        real_write(path, content)

    monkeypatch.setattr(closure, "write_once", failing_write)
    with pytest.raises(OSError, match="storage failure"):
        closure.write_closure(
            output, decide(joined, summaries), source_commit="a" * 40, schema=SCHEMA
        )
    assert (output / closure.DECISION_NAME).is_file()
    with pytest.raises(ProbeError, match="already exists"):
        closure.verify_inventory(output, completed=False)
