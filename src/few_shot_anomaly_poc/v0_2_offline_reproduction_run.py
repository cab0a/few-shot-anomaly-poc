"""Run the fixed v0.2.6 first-ten offline score reproduction."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.jsonio import write_json_atomic
from few_shot_anomaly_poc.v0_2_boundary_preparation import (
    RUN_ID,
    validate_boundary_execution_identity,
)
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    ARTIFACT_CONTRACT_VERSION,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_SCHEMA_SHA256,
    METHODS,
    load_v0_2_artifact_schema,
    load_v0_2_config,
    validate_tabular_records,
)
from few_shot_anomaly_poc.v0_2_scoring_artifacts import read_method_scoring_artifacts

MILESTONE = "v0.2.6"
RUN_KIND = "final_test"
ASSET_COUNT = 10
ABSOLUTE_SCORE_TOLERANCE = 1e-6
SCORING_EXECUTION_COMMIT = "ba23a2fe12a715161b420bc7d73d42f4de3bfc8c"
OPAQUE_SCORING_MANIFEST_SHA256 = "32ea52ed1b9872f39ae27f5d58a353ea84b8b143642e3a7f0fabe940184705e8"
STATE_HASHES = {
    "ecc_residual": "f796dcef8fb7b6197f656c2a57800766c0398ac4842fd65f85372781063b800b",
    "patch_hog_ocsvm": "ba7e1d47e8ff6fd7873edfce84c027c6ada37f6aab85d1b26cf92426463056f9",
    "dinov2_vits14_224_nn": ("11ac0a0a4b0c082e2450fcf708e1c96a804ec738967aff392b727959ec425f8d"),
}
SCORING_ARTIFACT_HASHES = {
    "ecc_residual/scores.csv": ("43545c1ec1c75039a7fd73389861fa7de4a6125fd264901b1e367ee1170b22c4"),
    "ecc_residual/classifications.csv": (
        "4e6b24d3553507a8d403900a209840c4d0e4d16ab3882001435e8875d93fb8c8"
    ),
    "ecc_residual/latency-observations.csv": (
        "3d2be6bab3bcbe35910b102c87dfed9e70786b0072b49d81d63dfc493fa2e50b"
    ),
    "patch_hog_ocsvm/scores.csv": (
        "fd403f19584c3170b15b4f9c3bef320ac8fbea23a0d715b26188bf850aef46e2"
    ),
    "patch_hog_ocsvm/classifications.csv": (
        "b17a977a570b1d2becfb9359941fe09d2936efc474253be1c11d133e369b95fa"
    ),
    "patch_hog_ocsvm/latency-observations.csv": (
        "5ff7424018220a7a8dd79e3dd646225de205a43373ff74c13b569100af45d1cb"
    ),
    "dinov2_vits14_224_nn/scores.csv": (
        "38885820447538763850bbc8820657ff75a8980e6e39ff09416970f32be6b282"
    ),
    "dinov2_vits14_224_nn/classifications.csv": (
        "71f4ae76d453583d030646ffbdf746683603d6d4eacad5784e3792d39cd468ba"
    ),
    "dinov2_vits14_224_nn/latency-observations.csv": (
        "1477493dac4a645283dc349d811d92bfcb8739c240a880ab9067e09d823ff63f"
    ),
}
REPRODUCTION_COLUMNS = (
    "contract_version",
    "run_id",
    "run_kind",
    "method",
    "asset_id",
    "expected_score_status",
    "reproduced_score_status",
    "expected_failure_code",
    "reproduced_failure_code",
    "expected_score",
    "reproduced_score",
    "absolute_difference",
    "within_tolerance",
)
EXPECTED_ASSET_IDS = tuple(f"asset-{index:06d}" for index in range(ASSET_COUNT))


class V0_2OfflineReproductionError(Exception):
    """Reject changed identities, leakage, overwrite, or incomplete reproduction."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V0_2OfflineReproductionError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V0_2OfflineReproductionError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise V0_2OfflineReproductionError(f"{label} must contain one JSON object")
    return value


def _score_value(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "asset_id": record["asset_id"],
        "score_status": record["score_status"],
        "score_failure_code": record["score_failure_code"],
        "anomaly_score": float(record["anomaly_score"]),
    }


def build_reproduction_records(
    *,
    method: str,
    expected: Sequence[Mapping[str, Any]],
    reproduced: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Build the fixed first-ten comparison without relaxing a failed result."""
    _require(method in METHODS, "method is outside the fixed inventory")
    _require(len(expected) == len(reproduced) == ASSET_COUNT, "first-ten records are incomplete")
    records: list[dict[str, Any]] = []
    for index, (fixed, observed) in enumerate(zip(expected, reproduced, strict=True)):
        asset_id = EXPECTED_ASSET_IDS[index]
        _require(
            fixed.get("asset_id") == observed.get("asset_id") == asset_id,
            "first-ten asset identity or order changed",
        )
        expected_score = float(fixed["anomaly_score"])
        reproduced_score = float(observed["anomaly_score"])
        difference = abs(expected_score - reproduced_score)
        identity_match = fixed.get("score_status") == observed.get("score_status") and fixed.get(
            "score_failure_code"
        ) == observed.get("score_failure_code")
        records.append(
            {
                "contract_version": ARTIFACT_CONTRACT_VERSION,
                "run_id": RUN_ID,
                "run_kind": RUN_KIND,
                "method": method,
                "asset_id": asset_id,
                "expected_score_status": fixed["score_status"],
                "reproduced_score_status": observed["score_status"],
                "expected_failure_code": fixed["score_failure_code"],
                "reproduced_failure_code": observed["score_failure_code"],
                "expected_score": expected_score,
                "reproduced_score": reproduced_score,
                "absolute_difference": difference,
                "within_tolerance": identity_match and difference <= ABSOLUTE_SCORE_TOLERANCE,
            }
        )
    return validate_tabular_records("reproduction", records, schema=schema)


def _write_reproduction_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REPRODUCTION_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        key: (
                            ""
                            if record[key] is None
                            else "true"
                            if record[key] is True
                            else "false"
                            if record[key] is False
                            else record[key]
                        )
                        for key in REPRODUCTION_COLUMNS
                    }
                )
    except Exception:
        path.unlink(missing_ok=True)
        raise


def read_reproduction_csv(path: Path, *, schema: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Read and contract-validate one method's reproduction evidence."""
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            _require(
                reader.fieldnames == list(REPRODUCTION_COLUMNS), "reproduction columns changed"
            )
            raw = list(reader)
    except (csv.Error, OSError, UnicodeError) as error:
        raise V0_2OfflineReproductionError("cannot read reproduction CSV") from error
    _require(
        len(raw) == ASSET_COUNT and all(None not in row for row in raw),
        "reproduction rows are incomplete",
    )
    try:
        records = [
            {
                **row,
                "expected_failure_code": row["expected_failure_code"] or None,
                "reproduced_failure_code": row["reproduced_failure_code"] or None,
                "expected_score": float(row["expected_score"]),
                "reproduced_score": float(row["reproduced_score"]),
                "absolute_difference": float(row["absolute_difference"]),
                "within_tolerance": (
                    True
                    if row["within_tolerance"] == "true"
                    else False
                    if row["within_tolerance"] == "false"
                    else None
                ),
            }
            for row in raw
        ]
    except ValueError as error:
        raise V0_2OfflineReproductionError("reproduction numeric value is invalid") from error
    _require(
        all(row["within_tolerance"] is not None for row in records),
        "reproduction boolean is invalid",
    )
    return validate_tabular_records("reproduction", records, schema=schema)


def _validate_public_scoring(
    public_root: Path, *, schema: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    for relative_path, expected_sha256 in SCORING_ARTIFACT_HASHES.items():
        _require(
            sha256_file(public_root / relative_path) == expected_sha256,
            f"committed scoring artifact changed: {relative_path}",
        )
    first_ten: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        bundle = read_method_scoring_artifacts(public_root / method, schema=schema)
        first_ten[method] = [_score_value(row) for row in bundle.score_records[:ASSET_COUNT]]
        _require(
            [row["asset_id"] for row in first_ten[method]] == list(EXPECTED_ASSET_IDS),
            f"{method} expected first-ten score order changed",
        )
        _require(
            not (public_root / method / "offline-reproduction.csv").exists(),
            "offline reproduction output already exists",
        )
    return first_ten


def _validate_boundary_state(state: Mapping[str, Any]) -> None:
    scoring = state.get("label_free_scoring")
    _require(
        state.get("schema_version") == "v0.2-boundary-state-v1"
        and state.get("run_id") == RUN_ID
        and state.get("boundary", {}).get("anomaly_score_computed") is True
        and state.get("boundary", {}).get("final_test_label_revealed") is False
        and isinstance(scoring, dict)
        and scoring.get("milestone") == "v0.2.5"
        and scoring.get("execution_commit") == SCORING_EXECUTION_COMMIT
        and scoring.get("anomaly_labels_used") is False
        and scoring.get("semantic_paths_accessed") is False
        and scoring.get("sealed_mapping_accessed") is False
        and scoring.get("official_split_accessed") is False
        and "offline_reproduction" not in state
        and "pre_reveal_checkpoint" not in state,
        "boundary state is not the fixed post-v0.2.5 pre-reveal state",
    )


def _worker_python(project_root: Path, method: str) -> Path:
    if method == "dinov2_vits14_224_nn":
        return project_root / "environments/v0.2-preflight/.venv/bin/python"
    return project_root / ".venv/bin/python"


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "MKL_NUM_THREADS": "4",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "42",
            "PYTHONNOUSERSITE": "1",
            "XFORMERS_DISABLED": "1",
        }
    )
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    return environment


def _run_worker(
    *,
    project_root: Path,
    execution_commit: str,
    method: str,
    expected_path: Path,
    input_path: Path,
    state_path: Path,
    report_path: Path,
) -> None:
    command = [
        str(_worker_python(project_root, method)),
        "-I",
        "-B",
        str(project_root / "scripts/run_v0_2_6_reproduction_worker.py"),
        "--project-root",
        str(project_root),
        "--execution-commit",
        execution_commit,
        "--method",
        method,
        "--expected",
        str(expected_path),
        "--input",
        str(input_path),
        "--fitted-state",
        str(state_path),
        "--report",
        str(report_path),
    ]
    serialized = " ".join(command).lower()
    _require(
        all(term not in serialized for term in ("sealed", "official_split", "source_path", "hmac")),
        "worker command exposes a protected input",
    )
    completed = subprocess.run(
        command,
        cwd=project_root,
        check=False,
        env=_worker_environment(),
        text=True,
    )
    _require(completed.returncode == 0, f"{method} reproduction worker failed")


def _create_dino_store(project_root: Path, input_path: Path) -> Path:
    import numpy as np

    from few_shot_anomaly_poc.v0_2_label_free_scoring import _adapt_dinov2, _decode

    scorer_root = project_root / f"data/external/v0.2/evaluation/{RUN_ID}/scorer"
    store_path = input_path.with_suffix(".npy")
    images = []
    records = []
    for index, asset_id in enumerate(EXPECTED_ASSET_IDS):
        asset_path = scorer_root / f"assets/{asset_id}.jpg"
        image = _adapt_dinov2(_decode(asset_path, color=True))
        _require(
            image.shape == (512, 512, 3) and image.dtype == np.uint8,
            "DINOv2 adapter output changed",
        )
        images.append(image)
        records.append(
            {
                "asset_id": asset_id,
                "index": index,
                "rgb_sha256": hashlib.sha256(image.tobytes(order="C")).hexdigest(),
            }
        )
    store = np.stack(images, axis=0)
    np.save(store_path, store, allow_pickle=False)
    write_json_atomic(
        input_path,
        {
            "schema_version": "v0.2.6-dinov2-first-ten-store-v1",
            "run_id": RUN_ID,
            "shape": list(store.shape),
            "dtype": str(store.dtype),
            "store_sha256": sha256_file(store_path),
            "records": records,
            "labels_accessed": False,
            "semantic_paths_accessed": False,
            "sealed_mapping_accessed": False,
            "official_split_accessed": False,
        },
    )
    return store_path


def _write_worker_input(
    *, project_root: Path, method: str, assets: Sequence[Any], path: Path
) -> None:
    if method == "dinov2_vits14_224_nn":
        _create_dino_store(project_root, path)
        return
    write_json_atomic(
        path,
        {
            "schema_version": "v0.2.6-classical-first-ten-input-v1",
            "run_id": RUN_ID,
            "opaque_scoring_manifest_sha256": OPAQUE_SCORING_MANIFEST_SHA256,
            "records": [
                {
                    "asset_id": asset.asset_id,
                    "byte_count": asset.byte_count,
                    "relative_path": asset.relative_path,
                    "sha256": asset.sha256,
                }
                for asset in assets[:ASSET_COUNT]
            ],
            "labels_accessed": False,
            "semantic_paths_accessed": False,
            "sealed_mapping_accessed": False,
            "official_split_accessed": False,
        },
    )


def _worker_report(path: Path, *, method: str) -> list[dict[str, Any]]:
    report = _read_json(path, label=f"{method} worker report")
    _require(
        set(report)
        == {
            "schema_version",
            "run_id",
            "method",
            "execution_commit",
            "records",
            "network_attempted",
            "labels_accessed",
            "semantic_paths_accessed",
            "sealed_mapping_accessed",
            "official_split_accessed",
        }
        and report.get("schema_version") == "v0.2.6-offline-reproduction-worker-v1"
        and report.get("run_id") == RUN_ID
        and report.get("method") == method
        and report.get("network_attempted") is False
        and report.get("labels_accessed") is False
        and report.get("semantic_paths_accessed") is False
        and report.get("sealed_mapping_accessed") is False
        and report.get("official_split_accessed") is False
        and isinstance(report.get("records"), list),
        f"{method} worker report is invalid",
    )
    return report["records"]


def run_v0_2_offline_reproduction(
    *,
    project_root: Path,
    execution_commit: str,
    external_root: Path,
    public_root: Path,
    work_root: Path,
) -> dict[str, str]:
    """Run each method once in a fresh offline process and publish all comparisons."""
    project_root = project_root.resolve()
    external_root = external_root.resolve()
    public_root = public_root.resolve()
    work_root = work_root.resolve()
    _require(
        external_root == (project_root / f"data/external/v0.2/evaluation/{RUN_ID}").resolve()
        and public_root == (project_root / f"artifacts/v0.2/evaluation/{RUN_ID}").resolve()
        and work_root == (project_root / f"work/v0.2/evaluation/{RUN_ID}").resolve(),
        "evaluation roots differ from the fixed contract",
    )
    validate_boundary_execution_identity(
        project_root=project_root, execution_commit=execution_commit
    )
    config = load_v0_2_config(project_root / "configs/v0.2.yaml")
    schema = load_v0_2_artifact_schema(project_root / "schemas/v0.2/evaluation-artifacts.json")
    _require(
        config["reproduction"]
        == {
            "phase": "after_committed_label_free_scoring_before_reveal",
            "fresh_process": True,
            "offline": True,
            "asset_selection": "first_10_opaque_asset_ids",
            "asset_count": ASSET_COUNT,
            "absolute_score_tolerance": ABSOLUTE_SCORE_TOLERANCE,
            "status_and_failure_code_must_match": True,
            "network_allowed": False,
            "failed_check_rerun_allowed": False,
        },
        "reproduction configuration changed",
    )
    _validate_boundary_state(
        _read_json(external_root / "boundary-state.json", label="boundary state")
    )
    from few_shot_anomaly_poc.v0_2_label_free_scoring import _validate_opaque_assets

    assets = _validate_opaque_assets(external_root / "scorer")
    expected = _validate_public_scoring(public_root, schema=schema)
    _require(
        not (work_root / "offline-reproduction-state.json").exists(),
        "local reproduction state exists",
    )

    stage_root = Path(tempfile.mkdtemp(dir=work_root, prefix=".v0.2.6-reproduction-"))
    statuses: dict[str, str] = {}
    try:
        staged_paths: dict[str, Path] = {}
        for method in METHODS:
            expected_path = stage_root / f"{method}-expected.json"
            input_path = stage_root / f"{method}-input.json"
            report_path = stage_root / f"{method}-report.json"
            write_json_atomic(
                expected_path,
                {
                    "schema_version": "v0.2.6-first-ten-expected-scores-v1",
                    "run_id": RUN_ID,
                    "method": method,
                    "records": expected[method],
                },
            )
            _write_worker_input(
                project_root=project_root, method=method, assets=assets, path=input_path
            )
            suffix = ".pt" if method == "dinov2_vits14_224_nn" else ".pkl"
            _run_worker(
                project_root=project_root,
                execution_commit=execution_commit,
                method=method,
                expected_path=expected_path,
                input_path=input_path,
                state_path=work_root / f"fitted-state/{method}{suffix}",
                report_path=report_path,
            )
            records = build_reproduction_records(
                method=method,
                expected=expected[method],
                reproduced=_worker_report(report_path, method=method),
                schema=schema,
            )
            output_path = stage_root / method / "offline-reproduction.csv"
            _write_reproduction_csv(output_path, records)
            read_reproduction_csv(output_path, schema=schema)
            staged_paths[method] = output_path
            statuses[method] = "pass" if all(row["within_tolerance"] for row in records) else "fail"

        moved: list[Path] = []
        identities = []
        try:
            for method in METHODS:
                destination = public_root / method / "offline-reproduction.csv"
                os.replace(staged_paths[method], destination)
                moved.append(destination)
                identities.append(
                    {
                        "relative_path": destination.relative_to(public_root).as_posix(),
                        "sha256": sha256_file(destination),
                    }
                )
            local_state = {
                "schema_version": "v0.2.6-offline-reproduction-state-v1",
                "milestone": MILESTONE,
                "run_id": RUN_ID,
                "execution_commit": execution_commit,
                "source_scoring_commit": SCORING_EXECUTION_COMMIT,
                "config_sha256": EXPECTED_CONFIG_SHA256,
                "schema_sha256": EXPECTED_SCHEMA_SHA256,
                "opaque_scoring_manifest_sha256": OPAQUE_SCORING_MANIFEST_SHA256,
                "state_sha256": STATE_HASHES,
                "artifact_identities": identities,
                "reproduction_status": statuses,
                "asset_count_per_method": ASSET_COUNT,
                "absolute_score_tolerance": ABSOLUTE_SCORE_TOLERANCE,
                "fresh_process_per_method": True,
                "offline": True,
                "network_attempted": False,
                "labels_accessed": False,
                "semantic_paths_accessed": False,
                "sealed_mapping_accessed": False,
                "official_split_accessed": False,
            }
            write_json_atomic(work_root / "offline-reproduction-state.json", local_state)
            state = _read_json(external_root / "boundary-state.json", label="boundary state")
            state["offline_reproduction"] = {
                "milestone": MILESTONE,
                "execution_commit": execution_commit,
                "source_scoring_commit": SCORING_EXECUTION_COMMIT,
                "artifact_identities": identities,
                "reproduction_status": statuses,
                "asset_count_per_method": ASSET_COUNT,
                "absolute_score_tolerance": ABSOLUTE_SCORE_TOLERANCE,
                "fresh_process_per_method": True,
                "offline": True,
                "network_attempted": False,
                "labels_accessed": False,
                "semantic_paths_accessed": False,
                "sealed_mapping_accessed": False,
                "official_split_accessed": False,
            }
            write_json_atomic(external_root / "boundary-state.json", state, overwrite=True)
        except Exception:
            (work_root / "offline-reproduction-state.json").unlink(missing_ok=True)
            for path in moved:
                path.unlink(missing_ok=True)
            raise
        return statuses
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def run_reproduction_worker(
    *,
    project_root: Path,
    execution_commit: str,
    method: str,
    expected_path: Path,
    input_path: Path,
    fitted_state_path: Path,
    report_path: Path,
) -> None:
    """Regenerate ten scores inside one isolated, network-blocked process."""
    project_root = project_root.resolve()
    _require(method in METHODS, "worker method is outside the fixed inventory")
    _require(not report_path.exists(), "worker report already exists")
    validate_boundary_execution_identity(
        project_root=project_root, execution_commit=execution_commit
    )
    expected = _read_json(expected_path, label="expected scores")
    _require(
        expected.get("schema_version") == "v0.2.6-first-ten-expected-scores-v1"
        and expected.get("run_id") == RUN_ID
        and expected.get("method") == method
        and isinstance(expected.get("records"), list)
        and len(expected["records"]) == ASSET_COUNT,
        "expected score input changed",
    )
    if method == "dinov2_vits14_224_nn":
        records, network_attempted = _run_dino_worker_scores(
            project_root=project_root,
            input_path=input_path,
            fitted_state_path=fitted_state_path,
        )
    else:
        records, network_attempted = _run_classical_worker_scores(
            project_root=project_root,
            method=method,
            input_path=input_path,
            fitted_state_path=fitted_state_path,
        )
    _require(
        [row["asset_id"] for row in records] == list(EXPECTED_ASSET_IDS),
        "worker score order changed",
    )
    write_json_atomic(
        report_path,
        {
            "schema_version": "v0.2.6-offline-reproduction-worker-v1",
            "run_id": RUN_ID,
            "method": method,
            "execution_commit": execution_commit,
            "records": records,
            "network_attempted": network_attempted,
            "labels_accessed": False,
            "semantic_paths_accessed": False,
            "sealed_mapping_accessed": False,
            "official_split_accessed": False,
        },
    )


def _run_classical_worker_scores(
    *, project_root: Path, method: str, input_path: Path, fitted_state_path: Path
) -> tuple[list[dict[str, Any]], bool]:
    from few_shot_anomaly_poc.config import load_config
    from few_shot_anomaly_poc.model_compatibility import NetworkGuard
    from few_shot_anomaly_poc.v0_2_label_free_scoring import (
        _classical_evidence,
        _decode,
        _load_classical_state,
    )

    manifest = _read_json(input_path, label="classical worker input")
    records = manifest.get("records")
    _require(
        manifest.get("schema_version") == "v0.2.6-classical-first-ten-input-v1"
        and manifest.get("run_id") == RUN_ID
        and manifest.get("opaque_scoring_manifest_sha256") == OPAQUE_SCORING_MANIFEST_SHA256
        and isinstance(records, list)
        and len(records) == ASSET_COUNT
        and all(
            manifest.get(field) is False
            for field in (
                "labels_accessed",
                "semantic_paths_accessed",
                "sealed_mapping_accessed",
                "official_split_accessed",
            )
        ),
        "classical worker input changed",
    )
    _require(
        sha256_file(fitted_state_path) == STATE_HASHES[method], "fitted-state identity changed"
    )
    state = _load_classical_state(fitted_state_path, method=method)
    config = load_config(project_root / "configs/v0.1.yaml")
    scorer_root = project_root / f"data/external/v0.2/evaluation/{RUN_ID}/scorer"
    output = []
    with NetworkGuard() as guard:
        for index, record in enumerate(records):
            _require(
                isinstance(record, dict)
                and record.get("asset_id") == EXPECTED_ASSET_IDS[index]
                and record.get("relative_path") == f"assets/{EXPECTED_ASSET_IDS[index]}.jpg",
                "classical worker asset identity changed",
            )
            path = scorer_root / record["relative_path"]
            _require(
                path.is_file()
                and path.stat().st_size == record.get("byte_count")
                and sha256_file(path) == record.get("sha256"),
                "classical worker asset bytes changed",
            )
            evidence = _classical_evidence(
                _decode(path, color=False),
                asset_id=record["asset_id"],
                method=method,
                state=state,
                config=config,
            )
            output.append(_score_value(vars(evidence)))
    return output, bool(guard.attempts)


def _run_dino_worker_scores(
    *, project_root: Path, input_path: Path, fitted_state_path: Path
) -> tuple[list[dict[str, Any]], bool]:
    import numpy as np

    from few_shot_anomaly_poc.dinov2_timing import _load_fixed_runtime
    from few_shot_anomaly_poc.model_assets import SOURCE_ROOT
    from few_shot_anomaly_poc.model_compatibility import EXPECTED_SOURCE_SHA256
    from few_shot_anomaly_poc.v0_2_dinov2_scoring_run import _load_memory_bank, _score

    manifest = _read_json(input_path, label="DINOv2 worker input")
    records = manifest.get("records")
    store_path = input_path.with_suffix(".npy")
    _require(
        manifest.get("schema_version") == "v0.2.6-dinov2-first-ten-store-v1"
        and manifest.get("run_id") == RUN_ID
        and manifest.get("shape") == [ASSET_COUNT, 512, 512, 3]
        and manifest.get("dtype") == "uint8"
        and sha256_file(store_path) == manifest.get("store_sha256")
        and isinstance(records, list)
        and len(records) == ASSET_COUNT
        and all(
            manifest.get(field) is False
            for field in (
                "labels_accessed",
                "semantic_paths_accessed",
                "sealed_mapping_accessed",
                "official_split_accessed",
            )
        ),
        "DINOv2 worker input changed",
    )
    store = np.load(store_path, mmap_mode="r", allow_pickle=False)
    _require(
        store.shape == (ASSET_COUNT, 512, 512, 3) and store.dtype == np.uint8,
        "DINOv2 store changed",
    )
    artifact_dir = project_root / "data/external/v0.2/model-assets"
    source_root = artifact_dir / f"dinov2-source-sha256-{EXPECTED_SOURCE_SHA256}" / SOURCE_ROOT
    torch = model = guard = previous_sys_path = None
    output = []
    try:
        torch, model, _runtime, guard, previous_sys_path = _load_fixed_runtime(
            acquisition_path=project_root / "artifacts/v0.2/model-assets/acquisition.json",
            import_smoke_path=project_root / "artifacts/v0.2/environment/import-smoke.json",
            strict_load_path=project_root / "artifacts/v0.2/model-compatibility/strict-load.json",
            artifact_dir=artifact_dir,
            source_root=source_root,
            environment_root=project_root / "environments/v0.2-preflight/.venv",
        )
        memory_bank = _load_memory_bank(
            fitted_state_path,
            expected_sha256=STATE_HASHES["dinov2_vits14_224_nn"],
            torch=torch,
        )
        for index, record in enumerate(records):
            image = np.array(store[index], dtype=np.uint8, order="C", copy=True)
            _require(
                record.get("asset_id") == EXPECTED_ASSET_IDS[index]
                and record.get("index") == index
                and hashlib.sha256(image.tobytes(order="C")).hexdigest()
                == record.get("rgb_sha256"),
                "DINOv2 worker asset identity changed",
            )
            output.append(
                _score_value(
                    vars(
                        _score(
                            image,
                            asset_id=record["asset_id"],
                            model=model,
                            memory_bank=memory_bank,
                            torch=torch,
                        )
                    )
                )
            )
        attempted = bool(guard.attempts)
    finally:
        if previous_sys_path is not None:
            sys.path[:] = previous_sys_path
        if guard is not None:
            guard.__exit__(None, None, None)
    return output, attempted
