"""Fail-closed orchestration of the first fixed, metadata-blinded observation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import cv2

from few_shot_anomaly_poc.v0_3_blinded_review import (
    BlindReviewItem,
    ReviewAssetIdentity,
    Reviewer,
    V0_3BlindedReviewError,
    build_review_completion_checkpoint,
    collect_blind_observations,
    decode_verified_opaque_asset,
    read_review_assets_csv,
    serialize_review_completion_checkpoint,
    write_blind_observations_csv,
)
from few_shot_anomaly_poc.v0_3_diagnostic_contract import (
    load_v0_3_config,
    load_v0_3_schema,
    sha256_file,
    validate_checkpoint_record,
)

ARTIFACT_ROOT = Path("artifacts/v0.3/diagnostics/pcb2-development")
PARENT_ROOT = Path("artifacts/v0.2/evaluation/visa-pcb2-v0-2-final")
INVENTORY_HASHES = {
    "review-assets.csv": "399ebef79498917ff0cf5b2bf83467541a5417dce74bdd8e0e462fec70a7fd58",
    "review-case-linkage.csv": "861cf3428808de3e300278d54be84de95fa42204393050358c9c2b592a638d7e",
    "normal-diagnostic-partition.csv": (
        "39415c626652c4cbfb856e68d8b559d0f119392f869e0569d47165df5a48e110"
    ),
    "pre-access-checkpoint.json": (
        "a8cade456af699df734007e3b575babd1f624f390dbf19728f912e8a43960df0"
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V0_3BlindedReviewError(message)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_once(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _regular_file(path: Path) -> bool:
    return path.is_file() and not any(part.is_symlink() for part in (path, *path.parents))


def _verify_hashes(identities: Mapping[Path, str]) -> None:
    for path, expected in identities.items():
        _require(_regular_file(path), f"required regular file is missing: {path}")
        _require(sha256_file(path) == expected, f"fixed file identity changed: {path}")


def _check_unstarted(output_root: Path, session_root: Path) -> None:
    _require(
        not any(path.is_symlink() for path in (*session_root.parents, *output_root.parents)),
        "session or output ancestor is symlinked",
    )
    _require(not session_root.exists() and not session_root.is_symlink(), "session already exists")
    _require(
        output_root.is_dir() and not output_root.is_symlink(), "diagnostic inventory is missing"
    )
    _require(
        {path.name for path in output_root.iterdir()} == set(INVENTORY_HASHES),
        "review/probe output or unexpected diagnostic entry already exists",
    )


@dataclass(frozen=True)
class ReviewPreflight:
    """Operator-only evidence; never pass this object to a reviewer."""

    source_commit: str
    assets: tuple[ReviewAssetIdentity, ...]
    schema: Mapping[str, Any]
    fixed_files: Mapping[Path, str]
    repository_root: Path | None = None

    def verify_unchanged(self) -> None:
        if self.repository_root is not None:
            _require(
                not _git(self.repository_root, "status", "--porcelain")
                and _git(self.repository_root, "rev-parse", "HEAD") == self.source_commit,
                "execution source changed during observation",
            )
        _verify_hashes(self.fixed_files)


def preflight_review(
    *,
    repository_root: Path,
    external_root: Path,
    fitted_state_root: Path,
    session_root: Path,
) -> ReviewPreflight:
    """Check pushed code and fixed metadata/state without opening image files."""
    _require(not _git(repository_root, "status", "--porcelain"), "worktree must be clean")
    source = _git(repository_root, "rev-parse", "HEAD")
    _require(source == _git(repository_root, "rev-parse", "origin/main"), "HEAD != origin/main")
    remote = _git(repository_root, "ls-remote", "origin", "refs/heads/main").split()
    _require(bool(remote) and remote[0] == source, "implementation must be pushed to origin/main")
    _git(repository_root, "merge-base", "--is-ancestor", "c2b5992", source)
    output_root = repository_root / ARTIFACT_ROOT
    _check_unstarted(output_root, session_root)
    config = load_v0_3_config(repository_root / "configs/v0.3.yaml")
    schema = load_v0_3_schema(repository_root / "schemas/v0.3/diagnostic-artifacts.json")
    dependencies = config["dependencies"]
    _require(platform.python_version() == dependencies["python"], "Python version changed")
    for distribution, key in (
        ("numpy", "numpy"),
        ("opencv-python-headless", "opencv_python_headless"),
    ):
        _require(version(distribution) == dependencies[key], f"{distribution} version changed")

    parent = config["parent_evidence"]
    fixed = {output_root / name: digest for name, digest in INVENTORY_HASHES.items()}
    fixed.update(
        {
            repository_root / config["preregistration"]["document"]: config["preregistration"][
                "document_sha256"
            ],
            repository_root / "configs/v0.3.yaml": sha256_file(
                repository_root / "configs/v0.3.yaml"
            ),
            repository_root / "schemas/v0.3/diagnostic-artifacts.json": sha256_file(
                repository_root / "schemas/v0.3/diagnostic-artifacts.json"
            ),
            repository_root / "configs/v0.2.yaml": parent["v0_2_config_sha256"],
            repository_root / "schemas/v0.2/evaluation-artifacts.json": parent[
                "v0_2_schema_sha256"
            ],
            repository_root / "uv.lock": dependencies["root_lock_sha256"],
            repository_root / "environments/v0.2-preflight/uv.lock": dependencies[
                "dinov2_lock_sha256"
            ],
            repository_root / PARENT_ROOT / "artifact-manifest.json": parent[
                "artifact_manifest_sha256"
            ],
        }
    )
    _verify_hashes(fixed)
    checkpoint = json.loads((output_root / "pre-access-checkpoint.json").read_text("utf-8"))
    validate_checkpoint_record(checkpoint)
    manifest = json.loads(
        (repository_root / PARENT_ROOT / "artifact-manifest.json").read_text("utf-8")
    )
    for entry in manifest["files"]:
        fixed[repository_root / PARENT_ROOT / entry["relative_path"]] = entry["sha256"]
    for relative, key in (
        ("scorer/scoring-manifest.json", "opaque_scoring_manifest_sha256"),
        ("normal-manifests/manifest-set.json", "normal_manifest_set_sha256"),
        ("normal-manifests/reference.jsonl", "normal_reference_manifest_sha256"),
        ("normal-manifests/calibration.jsonl", "normal_calibration_manifest_sha256"),
    ):
        fixed[external_root / relative] = parent[key]
    for method, suffix in (
        ("ecc_residual", ".pkl"),
        ("patch_hog_ocsvm", ".pkl"),
        ("dinov2_vits14_224_nn", ".pt"),
    ):
        fixed[fitted_state_root / f"{method}{suffix}"] = config["methods"][method][
            "fitted_state_sha256"
        ]
    _verify_hashes(fixed)
    assets_root = external_root / "scorer/assets"
    _require(
        assets_root.is_dir()
        and not any(path.is_symlink() for path in (assets_root, *assets_root.parents)),
        "opaque asset root is missing or symlinked",
    )
    assets = read_review_assets_csv(
        output_root / "review-assets.csv",
        expected_sha256=INVENTORY_HASHES["review-assets.csv"],
        schema=schema,
    )
    return ReviewPreflight(source, assets, schema, fixed, repository_root)


class FileExchangeReviewer:
    """Present one lossless raster and blank form, with no operator metadata."""

    def __init__(self, root: Path, *, timeout_seconds: float = 3600) -> None:
        _require(timeout_seconds > 0, "review timeout must be positive")
        self.root = root
        self.timeout_seconds = timeout_seconds

    def __call__(self, item: BlindReviewItem) -> Mapping[str, Any]:
        directory = self.root / item.asset_id
        directory.mkdir()
        success, encoded = cv2.imencode(".png", item.pixels)
        _require(success, "cannot serialize review pixels")
        pixels_path = directory / "pixels.png"
        _write_once(pixels_path, encoded.tobytes())
        pixels_path.chmod(0o444)
        _write_once(
            directory / "item.json",
            _json_bytes(
                {
                    "asset_id": item.asset_id,
                    "pixels": "pixels.png",
                    "form": asdict(item.form),
                }
            ),
        )
        # The ready marker is written last, after both files have reached disk.
        _write_once(directory / "item.ready", b"")
        deadline = time.monotonic() + self.timeout_seconds
        while not (directory / "response.ready").exists():
            _require(time.monotonic() < deadline, "review response timed out")
            time.sleep(0.2)
        path = directory / "response.json"
        _require(_regular_file(path), "review response is missing or symlinked")
        _require(_regular_file(directory / "response.ready"), "response marker is invalid")
        _require(path.stat().st_size <= 16384, "review response is too large")
        return json.loads(path.read_text(encoding="utf-8"))


def run_first_observation(
    *,
    preflight: ReviewPreflight,
    assets_root: Path,
    output_root: Path,
    session_root: Path,
    reviewer_kind: str,
    reviewer_factory: Callable[[Path], Reviewer] = FileExchangeReviewer,
) -> dict[str, Any]:
    """Attempt each image once; preserve a stopped attempt without a completion claim."""
    _require(reviewer_kind in {"human", "isolated_agent"}, "reviewer kind is invalid")
    _check_unstarted(output_root, session_root)
    preflight.verify_unchanged()
    session_root.mkdir(parents=True)
    operator = session_root / "operator"
    reviewer_root = session_root / "reviewer"
    operator.mkdir()
    reviewer_root.mkdir()
    _write_once(
        operator / "start.json",
        _json_bytes(
            {
                "source_commit": preflight.source_commit,
                "reviewer_kind": reviewer_kind,
                "review_assets_sha256": INVENTORY_HASHES["review-assets.csv"],
                "first_pass_complete": False,
            }
        ),
    )
    attempted: list[str] = []
    responses: list[str] = []

    def load(identity: ReviewAssetIdentity):
        # A durable attempt record precedes byte access, including a failed decode.
        _write_once(
            operator / f"{identity.asset_id}.attempt.json",
            _json_bytes(
                {
                    "asset_id": identity.asset_id,
                }
            ),
        )
        attempted.append(identity.asset_id)
        return decode_verified_opaque_asset(identity, assets_root=assets_root)

    try:
        reviewer = reviewer_factory(reviewer_root)

        def observe(item: BlindReviewItem):
            response = reviewer(item)
            _write_once(operator / f"{item.asset_id}.response.json", _json_bytes(response))
            responses.append(item.asset_id)
            return response

        records = collect_blind_observations(
            preflight.assets, image_loader=load, reviewer=observe, schema=preflight.schema
        )
        preflight.verify_unchanged()
        _require(
            {path.name for path in output_root.iterdir()} == set(INVENTORY_HASHES),
            "diagnostic output changed during observation",
        )
        digest = write_blind_observations_csv(
            output_root / "blind-observations.csv", records, schema=preflight.schema
        )
        with (output_root / "blind-observations.csv").open("rb") as stream:
            os.fsync(stream.fileno())
        checkpoint = build_review_completion_checkpoint(
            review_assets_sha256=INVENTORY_HASHES["review-assets.csv"],
            blind_observations_sha256=digest,
        )
        content = serialize_review_completion_checkpoint(checkpoint)
        _write_once(
            operator / "collection.json",
            _json_bytes(
                {
                    "source_commit": preflight.source_commit,
                    "reviewer_kind": reviewer_kind,
                    "attempted_asset_ids": attempted,
                    "response_asset_ids": responses,
                    "blind_observations_sha256": digest,
                    "completion_checkpoint_sha256": hashlib.sha256(content).hexdigest(),
                    "observation_collection_complete": True,
                    "method_metadata_joined": False,
                    "anomaly_scorer_executed": False,
                }
            ),
        )
        # Publish the completion marker last, after all other durable writes succeed.
        _write_once(output_root / "review-completion-checkpoint.json", content)
        return checkpoint
    except BaseException as error:
        # Preserve even an interrupt. Never retry, delete, or rewrite a previous attempt.
        _write_once(
            operator / "stopped.json",
            _json_bytes(
                {
                    "source_commit": preflight.source_commit,
                    "attempted_asset_ids": attempted,
                    "response_asset_ids": responses,
                    "exception_type": type(error).__name__,
                    "first_pass_complete": False,
                }
            ),
        )
        raise
