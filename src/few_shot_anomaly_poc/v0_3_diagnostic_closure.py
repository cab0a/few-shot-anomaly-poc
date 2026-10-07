"""Apply the frozen diagnostic rule to saved evidence and close its metadata bundle."""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

from few_shot_anomaly_poc.v0_3_diagnostic_contract import (
    ARTIFACT_CONTRACT_VERSION,
    CONDITIONS,
    DIAGNOSTIC_ID,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_SCHEMA_SHA256,
    METHODS,
)
from few_shot_anomaly_poc.v0_3_first_observation import ARTIFACT_ROOT, INVENTORY_HASHES
from few_shot_anomaly_poc.v0_3_observation_join import (
    JOIN_NAME,
    OBSERVATION_HASHES,
    check_pushed_source,
    join_observations,
    load_join_inputs,
    serialize_observation_join,
    visual_yes_counts,
)
from few_shot_anomaly_poc.v0_3_probe_artifacts import (
    SCORE_NAME,
    SUMMARY_NAME,
    json_bytes,
    read_table,
    require,
    verify_probe_outputs,
    write_once,
)
from few_shot_anomaly_poc.v0_3_probe_runtime import verify_file

PROBE_EVIDENCE_COMMIT = "436d5c2ef96769e28147999daad742cfdba2a59c"
INPUT_HASHES = {
    **INVENTORY_HASHES,
    **OBSERVATION_HASHES,
    JOIN_NAME: "e78117d7affe0c3800911cd0cebe62038b872bec2264cf2fc2c5014fa51fc5b8",
    SCORE_NAME: "58c82b1447332584f9d2c06bf628d1ce4687be27d2ca8364ec1560101902a7ef",
    SUMMARY_NAME: "c08b974a84dffbcf6bcd2a327595e9b69fed2a7310e52baec216e001fb38a9c8",
}
DECISION_NAME = "diagnostic-decision.json"
MANIFEST_NAME = "artifact-manifest.json"
CLOSURE_NAMES = {DECISION_NAME, MANIFEST_NAME}


def verify_inventory(output: Path, *, completed: bool) -> None:
    require(
        output.is_dir() and not any(p.is_symlink() for p in (output, *output.parents)),
        "invalid diagnostic directory",
    )
    expected = set(INPUT_HASHES) | (CLOSURE_NAMES if completed else set())
    require(
        {p.name for p in output.iterdir()} == expected,
        "diagnostic output already exists, is incomplete, or has an unexpected entry",
    )
    for name, digest in INPUT_HASHES.items():
        verify_file(output / name, digest)


def load_closure_inputs(root: Path, *, completed: bool = False) -> tuple[dict, dict, list, list]:
    """Read committed metadata only; no external data, model, or private session is required."""
    output = root / ARTIFACT_ROOT
    verify_inventory(output, completed=completed)
    config, schema, observations, links = load_join_inputs(root)
    expected_join = join_observations(observations, links, schema=schema)
    require(
        (output / JOIN_NAME).read_bytes()
        == serialize_observation_join(expected_join, schema=schema),
        "join differs from the immutable first-pass observations",
    )
    partition = read_table(
        output / "normal-diagnostic-partition.csv", "normal_partition", schema=schema
    )
    verify_probe_outputs(output, partition, config=config, schema=schema)
    summaries = read_table(output / SUMMARY_NAME, "condition_summary", schema=schema)
    return config, schema, expected_join, summaries


def evaluate_families(joined: list[dict], summaries: list[dict], *, config: dict) -> list[dict]:
    """Require both visual and numeric evidence for the same method, then use fixed priority."""
    keyed = {(r["method"], r["condition"]): r for r in summaries}
    require(
        len(summaries) == len(keyed) == len(METHODS) * len(CONDITIONS)
        and set(keyed) == {(m, c) for m in METHODS for c in CONDITIONS},
        "summary keys are missing, duplicated, or unexpected",
    )
    counts = visual_yes_counts(joined, config=config)
    rule = config["signal_rule"]
    evaluations = []
    for priority, family in enumerate(rule["priority"], 1):
        definition = rule["families"][family]
        methods = []
        for method in METHODS:
            visual_count = counts[method][family]
            visual_met = visual_count >= rule["minimum_visual_yes_unique_assets"]
            delta_minimum = (
                rule["minimum_median_absolute_delta_threshold_fraction"]
                * config["methods"][method]["threshold"]
            )
            conditions = []
            for condition in definition["conditions"]:
                row = keyed[method, condition]
                delta = row["median_absolute_delta"]
                crossing = row["normal_to_anomalous_crossing_count"]
                require(
                    type(delta) in (float, int) and math.isfinite(delta) and delta >= 0,
                    "invalid median absolute delta",
                )
                require(type(crossing) is int and 0 <= crossing <= 60, "invalid crossing count")
                crossing_met = crossing >= rule["minimum_normal_to_anomalous_crossings"]
                delta_met = delta >= delta_minimum
                conditions.append(
                    {
                        "condition": condition,
                        "normal_to_anomalous_crossing_count": crossing,
                        "minimum_crossing_count": rule["minimum_normal_to_anomalous_crossings"],
                        "crossing_requirement_met": crossing_met,
                        "median_absolute_delta": delta,
                        "minimum_median_absolute_delta": delta_minimum,
                        "delta_requirement_met": delta_met,
                        "numeric_requirement_met": crossing_met or delta_met,
                    }
                )
            numeric_met = any(r["numeric_requirement_met"] for r in conditions)
            methods.append(
                {
                    "method": method,
                    "visual_yes_unique_asset_count": visual_count,
                    "minimum_visual_yes_unique_asset_count": rule[
                        "minimum_visual_yes_unique_assets"
                    ],
                    "visual_requirement_met": visual_met,
                    "condition_evaluations": conditions,
                    "numeric_requirement_met": numeric_met,
                    "supported": visual_met and numeric_met,
                }
            )
        evaluations.append(
            {
                "family": family,
                "priority": priority,
                "method_evaluations": methods,
                "supporting_methods": [r["method"] for r in methods if r["supported"]],
                "supported": any(r["supported"] for r in methods),
            }
        )
    return evaluations


def build_decision(
    evaluations: list[dict],
    *,
    boundary_violated: bool = False,
    evidence_complete: bool = True,
) -> dict:
    """Apply invalid/incomplete/proceed/stop precedence without a weighted score."""
    require(
        type(boundary_violated) is bool and type(evidence_complete) is bool,
        "invalid evidence flags",
    )
    selected = None
    if boundary_violated:
        status, decision = "invalid", "INVALID: BOUNDARY VIOLATION"
        reason = "A diagnostic access or evidence boundary was violated."
    elif not evidence_complete:
        status, decision = "incomplete", "INCONCLUSIVE: REQUIRED EVIDENCE INCOMPLETE"
        reason = "Required first-pass observation or controlled-score evidence is incomplete."
    else:
        supported = [r for r in evaluations if r["supported"]]
        status = "complete"
        if supported:
            decision = "PROCEED TO ONE BOUNDED FOLLOW-UP PREREGISTRATION"
            selected = supported[0]["family"]
            reason = (
                "The first supported family in the preregistered priority qualifies "
                "for one bounded follow-up preregistration."
            )
        else:
            decision = "STOP: NO BOUNDED DIAGNOSTIC SIGNAL"
            reason = (
                "All required evidence is complete. No family meets both the visual "
                "and controlled-score requirements for the same method."
            )
    return {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "status": status,
        "decision": decision,
        "selected_family": selected,
        "family_evaluations": evaluations,
        "reason": reason,
        "weighted_score_used": False,
        "v0_2_decision_changed": False,
        "performance_claim_made": False,
    }


def validate_decision(value: dict, *, joined, summaries, config, schema) -> None:
    require(
        set(value) == set(schema["contracts"]["diagnostic_decision"]["required_keys"]),
        "diagnostic decision fields changed",
    )
    expected = build_decision(evaluate_families(joined, summaries, config=config))
    require(
        json_bytes(value) == json_bytes(expected),
        "diagnostic decision differs from the fixed evidence and rule",
    )


def build_manifest(output: Path, *, source_commit: str, schema: dict) -> dict:
    require(re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None, "invalid execution commit")
    paths = set(INPUT_HASHES) | {DECISION_NAME}
    require(
        {p.name for p in output.iterdir()} == paths
        or {p.name for p in output.iterdir()} == paths | {MANIFEST_NAME},
        "unexpected bundle inventory",
    )
    contracts = {v["path"]: (k, v) for k, v in schema["contracts"].items()}
    from few_shot_anomaly_poc.v0_3_diagnostic_contract import sha256_file

    files = []
    for name in sorted(paths):
        path = output / name
        require(path.is_file() and not path.is_symlink(), "non-regular bundle entry")
        kind, contract = contracts[name]
        count = len(read_table(path, kind, schema=schema)) if contract["format"] == "csv" else 1
        files.append(
            {
                "artifact_type": kind,
                "record_count": count,
                "relative_path": name,
                "sha256": sha256_file(path),
            }
        )
    return {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "source_commit": source_commit,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "schema_sha256": EXPECTED_SCHEMA_SHA256,
        "files": files,
    }


def write_closure(output: Path, decision: dict, *, source_commit: str, schema: dict) -> dict:
    """Exclusive writes; a partial storage failure remains visible and cannot be retried."""
    verify_inventory(output, completed=False)
    require(
        set(decision) == set(schema["contracts"]["diagnostic_decision"]["required_keys"]),
        "invalid decision fields",
    )
    require(re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None, "invalid execution commit")
    write_once(output / DECISION_NAME, json_bytes(decision))
    for name, digest in INPUT_HASHES.items():
        verify_file(output / name, digest)
    manifest = build_manifest(output, source_commit=source_commit, schema=schema)
    write_once(output / MANIFEST_NAME, json_bytes(manifest))
    return manifest


def verify_closure(root: Path) -> tuple[dict, dict]:
    config, schema, joined, summaries = load_closure_inputs(root, completed=True)
    output = root / ARTIFACT_ROOT
    for name in CLOSURE_NAMES:
        require(not (output / name).is_symlink(), "symlinked closure artifact")
    decision = json.loads((output / DECISION_NAME).read_text("utf-8"))
    validate_decision(decision, joined=joined, summaries=summaries, config=config, schema=schema)
    require(
        (output / DECISION_NAME).read_bytes() == json_bytes(decision),
        "decision serialization changed",
    )
    manifest = json.loads((output / MANIFEST_NAME).read_text("utf-8"))
    require(
        set(manifest) == set(schema["contracts"]["bundle_manifest"]["required_keys"]),
        "manifest fields changed",
    )
    expected = build_manifest(output, source_commit=manifest["source_commit"], schema=schema)
    require(
        (output / MANIFEST_NAME).read_bytes() == json_bytes(expected),
        "bundle manifest content, ordering, count, hash, or serialization changed",
    )
    return decision, manifest


def preflight_closure(root: Path) -> tuple[str, dict, dict, dict]:
    source = check_pushed_source(root)
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROBE_EVIDENCE_COMMIT, source], cwd=root, check=True
    )
    config, schema, joined, summaries = load_closure_inputs(root)
    decision = build_decision(evaluate_families(joined, summaries, config=config))
    validate_decision(decision, joined=joined, summaries=summaries, config=config, schema=schema)
    return source, config, schema, decision
