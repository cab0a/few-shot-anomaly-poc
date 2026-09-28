"""Fixed normal-only perturbations and fail-closed orchestration of 900 scores."""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

from few_shot_anomaly_poc.v0_3_diagnostic_contract import (
    CONDITIONS,
    EXPECTED_CONFIG_SHA256,
    METHODS,
    sha256_file,
)
from few_shot_anomaly_poc.v0_3_first_observation import ARTIFACT_ROOT, INVENTORY_HASHES, PARENT_ROOT
from few_shot_anomaly_poc.v0_3_inventory import (
    NORMAL_PARTITION_FIELDS,
    load_normal_manifest,
    select_normal_partition,
    serialize_csv,
)
from few_shot_anomaly_poc.v0_3_observation_join import (
    JOIN_NAME,
    OBSERVATION_HASHES,
    check_pushed_source,
    load_join_inputs,
)
from few_shot_anomaly_poc.v0_3_probe_artifacts import (
    BASE_COUNT,
    SCORE_NAME,
    STORE_SHAPE,
    SUMMARY_NAME,
    ProbeError,
    append_audit,
    bytes_sha256,
    json_bytes,
    require,
    score_record,
    serialize_table,
    summarize_scores,
    validate_partition,
    validate_scores,
    verify_probe_outputs,
    write_once,
)
from few_shot_anomaly_poc.v0_3_probe_runtime import (
    ENVIRONMENT_ROOT,
    check_isolated_runtime,
    verify_file,
    verify_model_assets,
    verify_scorer_sources,
    worker_environment,
)

JOIN_COMMIT = "c07eee51c8d5993cceaeeb92b071005c636e4661"
JOIN_SHA256 = "e78117d7affe0c3800911cd0cebe62038b872bec2264cf2fc2c5014fa51fc5b8"
SESSION_ROOT = Path("work/v0.3.6/controlled-probe")
PUBLIC_INPUT_HASHES = {**INVENTORY_HASHES, **OBSERVATION_HASHES, JOIN_NAME: JOIN_SHA256}
ROOT_VERSIONS = {
    "numpy": "2.5.1",
    "opencv-python-headless": "4.13.0.92",
    "scikit-image": "0.26.0",
    "scikit-learn": "1.9.0",
}


@dataclass(frozen=True)
class ProbePreflight:
    source_commit: str
    config: dict
    schema: dict
    partition: tuple[dict, ...]
    fixed_files: Mapping[Path, str]
    source_root: Path
    state_root: Path
    output_root: Path
    session_root: Path
    repository_root: Path | None = None

    def verify_unchanged(self) -> None:
        if self.repository_root is not None:
            root = self.repository_root
            observed = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            require(observed == self.source_commit and not status, "execution source changed")
        for path, digest in self.fixed_files.items():
            verify_file(path, digest)


def check_unstarted(output: Path, session: Path) -> None:
    require(not session.exists() and not session.is_symlink(), "probe session already exists")
    require(
        not any(p.is_symlink() for p in (*output.parents, output, *session.parents)),
        "symlinked probe path",
    )
    require(output.is_dir(), "diagnostic metadata directory missing")
    require(
        {p.name for p in output.iterdir()} == set(PUBLIC_INPUT_HASHES),
        "probe output or unexpected diagnostic entry already exists",
    )


def preflight_probe(*, root: Path, external_root: Path, state_root: Path) -> ProbePreflight:
    """Check metadata, dependencies, state hashes and model assets without image-byte access."""
    source = check_pushed_source(root)
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", JOIN_COMMIT, source], cwd=root, check=True
    )
    output, session = root / ARTIFACT_ROOT, root / SESSION_ROOT
    check_unstarted(output, session)
    config, schema, _observations, _links = load_join_inputs(root)
    verify_file(output / JOIN_NAME, JOIN_SHA256)
    require(platform.python_version() == config["dependencies"]["python"], "root Python changed")
    require(
        platform.system() == "Linux" and platform.machine() == "x86_64", "Linux x86-64 required"
    )
    require(
        all(version(name) == expected for name, expected in ROOT_VERSIONS.items()),
        "root runtime distribution changed",
    )
    fixed = {output / name: digest for name, digest in PUBLIC_INPUT_HASHES.items()}
    for relative in (
        "configs/v0.3.yaml",
        "schemas/v0.3/diagnostic-artifacts.json",
        "configs/v0.2.yaml",
        "schemas/v0.2/evaluation-artifacts.json",
        "uv.lock",
        "environments/v0.2-preflight/uv.lock",
        config["preregistration"]["document"],
    ):
        fixed[root / relative] = sha256_file(root / relative)
    manifest_path = root / PARENT_ROOT / "artifact-manifest.json"
    fixed[manifest_path] = config["parent_evidence"]["artifact_manifest_sha256"]
    manifest = json.loads(manifest_path.read_text("utf-8"))
    for entry in manifest["files"]:
        fixed[root / PARENT_ROOT / entry["relative_path"]] = entry["sha256"]
    from few_shot_anomaly_poc.v0_2_normal_calibration import CLASSICAL_CONFIG_SHA256

    fixed[root / "configs/v0.1.yaml"] = CLASSICAL_CONFIG_SHA256
    parent = config["parent_evidence"]
    normal_manifests = {}
    for part, key in (
        ("reference", "normal_reference_manifest_sha256"),
        ("calibration", "normal_calibration_manifest_sha256"),
    ):
        path = external_root / "normal-manifests" / f"{part}.jsonl"
        fixed[path] = parent[key]
        normal_manifests[part] = load_normal_manifest(
            path, expected_sha256=parent[key], partition=part
        )
    reference_paths = {row["relative_path"] for row in normal_manifests["reference"]}
    require(
        not reference_paths.intersection(
            row["relative_path"] for row in normal_manifests["calibration"]
        ),
        "reference/calibration overlap",
    )
    selected = select_normal_partition(
        normal_manifests["calibration"],
        selection_prefix=config["normal_probe"]["selection_prefix"],
        selected_count=BASE_COUNT,
    )
    require(
        serialize_csv(NORMAL_PARTITION_FIELDS, selected)
        == (output / "normal-diagnostic-partition.csv").read_bytes(),
        "reconstructed normal partition differs from fixed metadata",
    )
    validate_partition(selected, schema=schema)
    source_root = external_root / "source"
    require(
        source_root.is_dir() and not source_root.is_symlink(), "normal source directory missing"
    )
    for method, suffix in zip(METHODS, (".pkl", ".pkl", ".pt"), strict=True):
        fixed[state_root / f"{method}{suffix}"] = config["methods"][method]["fitted_state_sha256"]
    fixed.update(verify_model_assets(root))
    fixed.update(verify_scorer_sources(root))
    for path, digest in fixed.items():
        verify_file(path, digest)
    check_isolated_runtime(root)
    result = ProbePreflight(
        source,
        config,
        schema,
        tuple(selected),
        fixed,
        source_root,
        state_root,
        output,
        session,
        root,
    )
    result.verify_unchanged()
    return result


def immutable_bgr(pixels: np.ndarray) -> np.ndarray:
    require(
        isinstance(pixels, np.ndarray)
        and pixels.dtype == np.uint8
        and pixels.ndim == 3
        and pixels.shape[2] == 3
        and pixels.size > 0,
        "invalid decoded BGR input",
    )
    return np.frombuffer(pixels.tobytes(order="C"), dtype=np.uint8).reshape(pixels.shape)


def transform_image(base: np.ndarray, condition: str) -> np.ndarray:
    require(condition in CONDITIONS, "unknown fixed condition")
    require(
        base.dtype == np.uint8 and base.ndim == 3 and base.shape[2] == 3 and base.size > 0,
        "transform needs BGR uint8 input",
    )
    if condition == "control":
        result = base
    elif condition in ("brightness_085", "brightness_115"):
        factor = 0.85 if condition == "brightness_085" else 1.15
        result = np.clip(np.rint(base.astype(np.float64) * factor), 0, 255).astype(np.uint8)
    elif condition == "gaussian_blur":
        result = cv2.GaussianBlur(base, (5, 5), 1.0, borderType=cv2.BORDER_REFLECT_101)
    else:
        result = cv2.warpAffine(
            base,
            np.array([[1, 0, 4], [0, 1, 4]], dtype=np.float32),
            (base.shape[1], base.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    require(result.shape == base.shape, "transform changed decoded extent")
    return immutable_bgr(result)


def normal_image_path(source_root: Path, row: Mapping) -> Path:
    require(row["relative_path"].startswith("pcb2/Data/Images/Normal/"), "unauthorized source path")
    path = source_root / row["relative_path"]
    require(path.resolve().is_relative_to(source_root.resolve()), "normal path escaped source root")
    require(
        path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
        "invalid normal image path",
    )
    return path


def decode_normal(source_root: Path, row: Mapping) -> np.ndarray:
    path = normal_image_path(source_root, row)
    content = path.read_bytes()
    require(
        len(content) == row["byte_count"] and bytes_sha256(content) == row["sha256"],
        "normal image bytes changed",
    )
    decoded = cv2.imdecode(
        np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    return immutable_bgr(decoded)


@contextmanager
def classical_scorers(preflight: ProbePreflight):
    from few_shot_anomaly_poc.config import load_config
    from few_shot_anomaly_poc.model_compatibility import NetworkGuard
    from few_shot_anomaly_poc.v0_2_label_free_scoring import (
        _classical_evidence,
        _load_classical_state,
    )

    require(preflight.repository_root is not None, "real classical scorer requires a repository")
    config = load_config(preflight.repository_root / "configs/v0.1.yaml")
    with NetworkGuard() as guard:
        states = {}
        for method in METHODS[:2]:
            path = preflight.state_root / f"{method}.pkl"
            verify_file(path, preflight.config["methods"][method]["fitted_state_sha256"])
            states[method] = _load_classical_state(path, method=method)

        def score(method: str, bgr: np.ndarray):
            return _classical_evidence(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                asset_id="development-normal",
                method=method,
                state=states[method],
                config=config,
            )

        yield score
        require(not guard.attempts, "classical scorer attempted network access")


def run_dinov2_worker(preflight: ProbePreflight, manifest_path: Path) -> list[dict]:
    require(preflight.repository_root is not None, "real worker requires a repository")
    root = preflight.repository_root
    command = [
        str(root / ENVIRONMENT_ROOT / "bin/python"),
        "-I",
        "-B",
        str(root / "scripts/run_v0_3_6_dinov2_probe_worker.py"),
        "--input-manifest",
        str(manifest_path),
        "--input-manifest-sha256",
        sha256_file(manifest_path),
        "--state-root",
        str(preflight.state_root),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=worker_environment(),
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired as error:
        write_once(
            preflight.session_root / "worker-timeout.json", json_bytes({"worker_timeout": True})
        )
        raise ProbeError("DINOv2 worker timed out; partial worker audit preserved") from error
    write_once(preflight.session_root / "worker-stdout.txt", completed.stdout.encode())
    write_once(preflight.session_root / "worker-stderr.txt", completed.stderr.encode())
    require(completed.returncode == 0, "DINOv2 worker failed; inspect preserved worker audit")
    worker_root = preflight.session_root / "dinov2-worker"
    report = json.loads((worker_root / "complete.json").read_text("utf-8"))
    require(
        report
        == {
            "source_commit": preflight.source_commit,
            "score_count": 300,
            "network_access_attempted": False,
            "scores_sha256": sha256_file(worker_root / "scores.jsonl"),
        },
        "DINOv2 completion report changed",
    )
    rows = [
        json.loads(line) for line in (worker_root / "scores.jsonl").read_text("utf-8").splitlines()
    ]
    validate_scores(
        rows,
        preflight.partition,
        config=preflight.config,
        schema=preflight.schema,
        methods=(METHODS[2],),
    )
    return rows


def run_probe(
    preflight: ProbePreflight,
    *,
    loader=decode_normal,
    scorer_factory=classical_scorers,
    dino_worker=run_dinov2_worker,
    progress: Callable[[str], None] = print,
) -> dict:
    """Run once; preserve every attempted input and returned score before final publication."""
    check_unstarted(preflight.output_root, preflight.session_root)
    preflight.verify_unchanged()
    validate_partition(preflight.partition, schema=preflight.schema)
    preflight.session_root.mkdir(parents=True, exist_ok=False)
    session = preflight.session_root
    rows: list[dict] = []
    phase = "start"
    store = None
    try:
        write_once(
            session / "start.json",
            json_bytes(
                {
                    "source_commit": preflight.source_commit,
                    "config_sha256": EXPECTED_CONFIG_SHA256,
                    "first_probe_complete": False,
                    "normal_partition_sha256": INVENTORY_HASHES["normal-diagnostic-partition.csv"],
                    "image_count": BASE_COUNT,
                    "expected_score_count": 900,
                    "latency_measured": False,
                }
            ),
        )
        phase = "verify_normal_images"
        for identity in preflight.partition:
            append_audit(
                session / "input-verification-attempts.jsonl",
                {"selection_rank": identity["selection_rank"], "sha256": identity["sha256"]},
            )
            path = normal_image_path(preflight.source_root, identity)
            require(path.stat().st_size == identity["byte_count"], "normal image bytes changed")
            verify_file(path, identity["sha256"])
        store_path = session / "dinov2-input.npy"
        store = np.lib.format.open_memmap(store_path, mode="w+", dtype=np.uint8, shape=STORE_SHAPE)
        transport_rows = []
        phase = "classical_scoring_and_transport"
        cv2.setNumThreads(4)
        with scorer_factory(preflight) as scorer:
            for identity in preflight.partition:
                rank = identity["selection_rank"]
                append_audit(
                    session / "image-attempts.jsonl",
                    {"selection_rank": rank, "sha256": identity["sha256"]},
                )
                base = immutable_bgr(loader(preflight.source_root, identity))
                for condition in CONDITIONS:
                    transformed = transform_image(base, condition)
                    for method in METHODS[:2]:
                        key = {"method": method, "selection_rank": rank, "condition": condition}
                        append_audit(session / "score-attempts.jsonl", key)
                        evidence = scorer(method, transformed)
                        row = score_record(
                            **key,
                            base_sha256=identity["sha256"],
                            status=evidence.score_status,
                            failure_code=evidence.score_failure_code,
                            score=evidence.anomaly_score,
                            config=preflight.config,
                            schema=preflight.schema,
                        )
                        append_audit(session / "scores.jsonl", row)
                        rows.append(row)
                        require(
                            row["score_status"] == "ok",
                            "classical score failed; incomplete probe preserved",
                        )
                    from few_shot_anomaly_poc.v0_2_label_free_scoring import _adapt_dinov2

                    rgb = _adapt_dinov2(transformed)
                    index = len(transport_rows)
                    store[index] = rgb
                    transport_rows.append(
                        {
                            "index": index,
                            "selection_rank": rank,
                            "condition": condition,
                            "base_image_sha256": identity["sha256"],
                            "rgb_sha256": bytes_sha256(rgb.tobytes()),
                        }
                    )
                progress(f"Normal probe: {rank}/60 bases; {len(rows)}/600 classical scores")
        store.flush()
        del store
        store = None
        manifest_path = session / "dinov2-input.json"
        write_once(
            manifest_path,
            json_bytes(
                {
                    "schema_version": "v0.3-dinov2-probe-store-v1",
                    "source_commit": preflight.source_commit,
                    "shape": list(STORE_SHAPE),
                    "dtype": "uint8",
                    "store_sha256": sha256_file(store_path),
                    "records": transport_rows,
                }
            ),
        )
        preflight.verify_unchanged()
        phase = "dinov2_scoring"
        progress("DINOv2 probe: scoring the 300 fixed RGB inputs in the isolated CPU worker")
        rows.extend(dino_worker(preflight, manifest_path))
        rows.sort(
            key=lambda row: (
                METHODS.index(row["method"]),
                row["selection_rank"],
                CONDITIONS.index(row["condition"]),
            )
        )
        phase = "validate_and_publish"
        summaries = summarize_scores(
            rows, preflight.partition, config=preflight.config, schema=preflight.schema
        )
        preflight.verify_unchanged()
        require(
            {p.name for p in preflight.output_root.iterdir()} == set(PUBLIC_INPUT_HASHES),
            "public output changed during probe",
        )
        scores_content = serialize_table("diagnostic_score", rows, schema=preflight.schema)
        summaries_content = serialize_table("condition_summary", summaries, schema=preflight.schema)
        write_once(preflight.output_root / SCORE_NAME, scores_content)
        write_once(preflight.output_root / SUMMARY_NAME, summaries_content)
        verify_probe_outputs(
            preflight.output_root,
            preflight.partition,
            config=preflight.config,
            schema=preflight.schema,
        )
        completion = {
            "source_commit": preflight.source_commit,
            "first_probe_complete": True,
            "base_image_count": BASE_COUNT,
            "score_count": 900,
            "summary_count": 15,
            "controlled_scores_sha256": bytes_sha256(scores_content),
            "condition_summaries_sha256": bytes_sha256(summaries_content),
            "v0_2_final_test_rescored": False,
            "latency_measured": False,
            "model_refitted": False,
            "threshold_recalibrated": False,
            "diagnostic_decision_written": False,
        }
        write_once(session / "complete.json", json_bytes(completion))
        return completion
    except BaseException as error:
        write_once(
            session / "stopped.json",
            json_bytes(
                {
                    "source_commit": preflight.source_commit,
                    "phase": phase,
                    "first_probe_complete": False,
                    "collected_parent_score_count": len(rows),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            ),
        )
        raise
    finally:
        if store is not None:
            store.flush()
