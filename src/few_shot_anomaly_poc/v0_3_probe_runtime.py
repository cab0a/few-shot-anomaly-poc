"""Restore and verify the already pinned DINOv2 runtime without loading weights."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from few_shot_anomaly_poc.model_assets import (
    CHECKPOINT_HOSTS,
    SOURCE_HOSTS,
    SOURCE_ROOT,
    _download_to_partial,
    extract_source_archive,
)
from few_shot_anomaly_poc.model_compatibility import EXPECTED_ENVIRONMENT, EXPECTED_SOURCE_SHA256
from few_shot_anomaly_poc.v0_3_diagnostic_contract import sha256_file

MODEL_RECORD_HASHES = {
    "artifacts/v0.2/model-assets/acquisition.json": (
        "ba976ed08369fd80423d241129b8a86b05fcef650a39befa4ee67c8314233dac"
    ),
    "artifacts/v0.2/environment/import-smoke.json": (
        "b0f38afb103f7084a0e5e09e8fd00e4cf2e0e5825d7a3fe8d5e3b48afd7b1f74"
    ),
    "artifacts/v0.2/model-compatibility/strict-load.json": (
        "4491f2fb472df813642d296d92d396e62476a2fd257d6b9da431c3a90b6aa604"
    ),
}
SCORER_SOURCE_COMMIT = "c07eee51c8d5993cceaeeb92b071005c636e4661"
SCORER_MODULES = (
    "config",
    "preprocessing",
    "ecc_residual",
    "ecc_template",
    "registration",
    "errors",
    "hog_features",
    "hog_scalers",
    "hog_models",
    "hog_scoring",
    "dinov2_scoring",
    "dinov2_errors",
    "dinov2_timing",
    "dinov2_timing_preflight",
    "model_compatibility",
    "model_assets",
    "hashing",
    "jsonio",
    "v0_2_label_free_scoring",
    "v0_2_dinov2_scoring_run",
    "v0_2_boundary_preparation",
    "v0_2_evaluation_contract",
    "v0_2_normal_calibration",
    "v0_2_scoring_artifacts",
)
ASSET_DIR = Path("data/external/v0.2/model-assets")
ENVIRONMENT_ROOT = Path("environments/v0.2-preflight/.venv")
SOURCE_DIR = ASSET_DIR / f"dinov2-source-sha256-{EXPECTED_SOURCE_SHA256}" / SOURCE_ROOT


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_file(path: Path, digest: str) -> None:
    require(
        path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
        f"required regular file is missing: {path}",
    )
    require(sha256_file(path) == digest, f"fixed file SHA-256 changed: {path}")


def model_records(root: Path) -> dict:
    for relative, digest in MODEL_RECORD_HASHES.items():
        verify_file(root / relative, digest)
    return json.loads((root / "artifacts/v0.2/model-assets/acquisition.json").read_text("utf-8"))


def verify_scorer_sources(root: Path) -> dict[Path, str]:
    """Require the inherited adapters, algorithms and loaders to match the joined evidence."""
    fixed = {}
    for module in SCORER_MODULES:
        relative = f"src/few_shot_anomaly_poc/{module}.py"
        original = subprocess.run(
            ["git", "show", f"{SCORER_SOURCE_COMMIT}:{relative}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        digest = hashlib.sha256(original).hexdigest()
        verify_file(root / relative, digest)
        fixed[root / relative] = digest
    return fixed


def verify_model_assets(root: Path) -> dict[Path, str]:
    """Hash model bytes and the whole extracted source tree; never deserialize."""
    record = model_records(root)
    fixed = {root / path: digest for path, digest in MODEL_RECORD_HASHES.items()}
    for kind in ("source", "checkpoint"):
        identity = record[kind]["artifact"]
        path = root / ASSET_DIR / identity["filename"]
        verify_file(path, identity["observed_sha256"])
        require(path.stat().st_size == identity["byte_count"], "model asset size changed")
        fixed[path] = identity["observed_sha256"]
    tree = root / SOURCE_DIR
    require(
        tree.is_dir() and not any(p.is_symlink() for p in (tree, *tree.parents)),
        "fixed DINOv2 extraction is missing",
    )
    entries = []
    for path in sorted(tree.rglob("*")):
        require(not path.is_symlink(), "DINOv2 source tree contains a symlink")
        if path.is_file():
            digest = sha256_file(path)
            fixed[path] = digest
            entries.append((path.relative_to(tree.parent).as_posix(), path.stat().st_size, digest))
    manifest = "".join(f"{digest}  {size}  {name}\n" for name, size, digest in sorted(entries))
    strict = json.loads((root / "artifacts/v0.2/model-compatibility/strict-load.json").read_text())
    expected = strict["source"]["extraction"]
    require(len(entries) == expected["file_count"], "DINOv2 source file count changed")
    require(
        hashlib.sha256(manifest.encode()).hexdigest() == expected["tree_manifest_sha256"],
        "DINOv2 extracted source bytes changed",
    )
    return fixed


def restore_model_assets(root: Path) -> None:
    """Retrieve only missing, previously approved assets at their exact recorded identities."""
    record = model_records(root)
    for kind, hosts in (("source", SOURCE_HOSTS), ("checkpoint", CHECKPOINT_HOSTS)):
        identity = record[kind]["artifact"]
        target = root / ASSET_DIR / identity["filename"]
        if not target.exists():
            partial, transport = _download_to_partial(
                record[kind]["transport"]["requested_url"],
                artifact_dir=root / ASSET_DIR,
                prefix=f"restore-{kind}",
                allowed_hosts=hosts,
                expected_bytes=identity["byte_count"],
            )
            require(
                transport["observed_sha256"] == identity["observed_sha256"],
                f"downloaded {kind} differs from the fixed asset; partial file preserved",
            )
            os.link(partial, target)
            partial.unlink()
        verify_file(target, identity["observed_sha256"])
    if not (root / SOURCE_DIR).exists():
        extract_source_archive(
            root / ASSET_DIR / record["source"]["artifact"]["filename"],
            expected_sha256=EXPECTED_SOURCE_SHA256,
            destination=root / SOURCE_DIR.parent,
            project_root=root,
        )
    verify_model_assets(root)


def worker_environment() -> dict[str, str]:
    environment = {**os.environ, **EXPECTED_ENVIRONMENT, "PYTHONHASHSEED": "42"}
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    return environment


def check_isolated_runtime(root: Path) -> dict:
    python = root / ENVIRONMENT_ROOT / "bin/python"
    require(python.is_file(), "DINOv2 environment missing; run prepare_v0_3_6_probe_runtime.py")
    completed = subprocess.run(
        [
            str(python),
            "-I",
            "-B",
            str(root / "scripts/run_v0_3_6_dinov2_probe_worker.py"),
            "--check-runtime",
        ],
        cwd=root,
        env=worker_environment(),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = json.loads(completed.stdout)
    require(result.get("runtime_verified") is True, "DINOv2 runtime verification failed")
    require(result.get("model_loaded") is False, "preflight loaded a model")
    return result
