"""Isolated DINOv2 probe over verified RGB arrays; no source images or final-test scores."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from few_shot_anomaly_poc.v0_3_diagnostic_contract import (
    CONDITIONS,
    METHODS,
    load_v0_3_config,
    load_v0_3_schema,
    sha256_file,
)
from few_shot_anomaly_poc.v0_3_probe_artifacts import (
    BASE_COUNT,
    STORE_SHAPE,
    append_audit,
    bytes_sha256,
    json_bytes,
    read_table,
    require,
    score_record,
    validate_partition,
    write_once,
)
from few_shot_anomaly_poc.v0_3_probe_runtime import (
    ASSET_DIR,
    ENVIRONMENT_ROOT,
    SOURCE_DIR,
    model_records,
    verify_file,
    verify_model_assets,
    verify_scorer_sources,
)

PARTITION_PATH = Path("artifacts/v0.3/diagnostics/pcb2-development/normal-diagnostic-partition.csv")
PARTITION_SHA256 = "39415c626652c4cbfb856e68d8b559d0f119392f869e0569d47165df5a48e110"


def check_runtime(root: Path) -> dict:
    from few_shot_anomaly_poc.model_compatibility import (
        EXPECTED_TORCH_VERSION,
        NetworkGuard,
        _load_import_smoke_record,
        _validate_environment,
    )

    model_records(root)
    smoke = _load_import_smoke_record(root / "artifacts/v0.2/environment/import-smoke.json")
    _validate_environment(environment_root=root / ENVIRONMENT_ROOT, import_smoke_record=smoke)
    with NetworkGuard() as guard:
        import torch

        require(
            torch.__version__ == EXPECTED_TORCH_VERSION
            and torch.version.cuda is None
            and torch.version.hip is None,
            "fixed CPU-only PyTorch required",
        )
    require(not guard.attempts, "runtime check attempted network access")
    return {"runtime_verified": True, "torch_version": torch.__version__, "model_loaded": False}


@contextmanager
def fixed_dino_scorer(root: Path, state_root: Path, config: dict):
    verify_scorer_sources(root)
    from few_shot_anomaly_poc.dinov2_timing import _load_fixed_runtime
    from few_shot_anomaly_poc.v0_2_dinov2_scoring_run import _load_memory_bank, _score

    verify_model_assets(root)
    state_path = state_root / f"{METHODS[2]}.pt"
    state_hash = config["methods"][METHODS[2]]["fitted_state_sha256"]
    verify_file(state_path, state_hash)
    guard = previous = None
    try:
        torch, model, _runtime, guard, previous = _load_fixed_runtime(
            acquisition_path=root / "artifacts/v0.2/model-assets/acquisition.json",
            import_smoke_path=root / "artifacts/v0.2/environment/import-smoke.json",
            strict_load_path=root / "artifacts/v0.2/model-compatibility/strict-load.json",
            artifact_dir=root / ASSET_DIR,
            source_root=root / SOURCE_DIR,
            environment_root=root / ENVIRONMENT_ROOT,
        )
        bank = _load_memory_bank(state_path, expected_sha256=state_hash, torch=torch)

        def score(rgb: np.ndarray):
            return _score(
                rgb, asset_id="development-normal", model=model, memory_bank=bank, torch=torch
            )

        yield score
        require(not guard.attempts, "DINOv2 attempted network access")
    finally:
        if previous is not None:
            sys.path[:] = previous
        if guard is not None:
            guard.__exit__(None, None, None)


def score_transport(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    partition: list[dict],
    config: dict,
    schema: dict,
    scorer_factory: Callable,
    source_commit: str,
) -> dict:
    """Validate all 300 identities before the first model load and retain partial attempts."""
    verify_file(manifest_path, expected_manifest_sha256)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    require(
        set(manifest)
        == {"schema_version", "source_commit", "shape", "dtype", "store_sha256", "records"},
        "transport manifest fields changed",
    )
    require(
        manifest["schema_version"] == "v0.3-dinov2-probe-store-v1"
        and manifest["source_commit"] == source_commit
        and manifest["shape"] == list(STORE_SHAPE)
        and manifest["dtype"] == "uint8",
        "transport identity changed",
    )
    validate_partition(partition, schema=schema)
    expected = [(rank, condition) for rank in range(1, BASE_COUNT + 1) for condition in CONDITIONS]
    require(len(manifest["records"]) == len(expected), "transport record count changed")
    for index, (record, key) in enumerate(zip(manifest["records"], expected, strict=True)):
        require(
            set(record)
            == {"index", "selection_rank", "condition", "base_image_sha256", "rgb_sha256"},
            "transport record fields changed",
        )
        require(
            record["index"] == index
            and (record["selection_rank"], record["condition"]) == key
            and record["base_image_sha256"] == partition[key[0] - 1]["sha256"],
            "transport ordering or base identity changed",
        )
    store_path = manifest_path.with_suffix(".npy")
    verify_file(store_path, manifest["store_sha256"])
    store = np.load(store_path, mmap_mode="r", allow_pickle=False)
    require(
        isinstance(store, np.memmap)
        and store.dtype == np.uint8
        and store.shape == STORE_SHAPE
        and store.flags.c_contiguous,
        "transport array contract changed",
    )
    output = manifest_path.parent / "dinov2-worker"
    require(not output.exists(), "DINOv2 worker session already exists")
    output.mkdir()
    count = 0
    try:
        with scorer_factory() as scorer:
            for record in manifest["records"]:
                append_audit(
                    output / "attempts.jsonl",
                    {"selection_rank": record["selection_rank"], "condition": record["condition"]},
                )
                rgb = np.array(store[record["index"]], dtype=np.uint8, order="C", copy=True)
                require(
                    bytes_sha256(rgb.tobytes()) == record["rgb_sha256"],
                    "RGB transport pixels changed",
                )
                evidence = scorer(rgb)
                row = score_record(
                    method=METHODS[2],
                    selection_rank=record["selection_rank"],
                    condition=record["condition"],
                    base_sha256=record["base_image_sha256"],
                    status=evidence.score_status,
                    failure_code=evidence.score_failure_code,
                    score=evidence.anomaly_score,
                    config=config,
                    schema=schema,
                )
                append_audit(output / "scores.jsonl", row)
                count += 1
                require(
                    row["score_status"] == "ok", "DINOv2 score failed; incomplete probe preserved"
                )
        report = {
            "source_commit": source_commit,
            "score_count": count,
            "network_access_attempted": False,
            "scores_sha256": sha256_file(output / "scores.jsonl"),
        }
        write_once(output / "complete.json", json_bytes(report))
        return report
    except BaseException as error:
        write_once(
            output / "stopped.json",
            json_bytes(
                {
                    "score_count": count,
                    "first_probe_complete": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            ),
        )
        raise


def run_worker(root: Path, manifest_path: Path, *, expected_sha256: str, state_root: Path) -> None:
    require(
        manifest_path.absolute() == root / "work/v0.3.6/controlled-probe/dinov2-input.json",
        "worker input outside the fixed probe session",
    )
    config = load_v0_3_config(root / "configs/v0.3.yaml")
    schema = load_v0_3_schema(root / "schemas/v0.3/diagnostic-artifacts.json")
    verify_file(root / PARTITION_PATH, PARTITION_SHA256)
    partition = read_table(root / PARTITION_PATH, "normal_partition", schema=schema)
    source = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    score_transport(
        manifest_path,
        expected_manifest_sha256=expected_sha256,
        partition=partition,
        config=config,
        schema=schema,
        source_commit=source,
        scorer_factory=lambda: fixed_dino_scorer(root, state_root, config),
    )
