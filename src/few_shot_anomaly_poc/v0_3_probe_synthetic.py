"""Synthetic image and fake-score fixtures for the controlled-probe transport."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from few_shot_anomaly_poc.v0_3_controlled_probe import PUBLIC_INPUT_HASHES, ProbePreflight
from few_shot_anomaly_poc.v0_3_diagnostic_contract import (
    ARTIFACT_CONTRACT_VERSION,
    DIAGNOSTIC_ID,
    load_v0_3_config,
    load_v0_3_schema,
    sha256_file,
)
from few_shot_anomaly_poc.v0_3_dinov2_probe import score_transport
from few_shot_anomaly_poc.v0_3_probe_artifacts import bytes_sha256, require


def synthetic_preflight(work: Path, repository: Path) -> ProbePreflight:
    config = load_v0_3_config(repository / "configs/v0.3.yaml")
    schema = load_v0_3_schema(repository / "schemas/v0.3/diagnostic-artifacts.json")
    output = work / "output"
    output.mkdir(parents=True)
    fixed = {}
    for name in PUBLIC_INPUT_HASHES:
        path = output / name
        path.write_bytes(f"Synthetic metadata fixture: {name}\n".encode())
        fixed[path] = sha256_file(path)
    source = work / "synthetic-source"
    partition = []
    for rank in range(1, 61):
        path_text = f"pcb2/Data/Images/Normal/{rank:04d}.JPG"
        path = source / path_text
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = (
            ((np.arange(9 * 13 * 3, dtype=np.uint16) + rank * 7) % 256)
            .astype(np.uint8)
            .reshape(9, 13, 3)
        )
        ok, encoded = cv2.imencode(".png", pixels)
        require(ok, "synthetic image encoding failed")
        content = encoded.tobytes()
        path.write_bytes(content)
        partition.append(
            {
                "contract_version": ARTIFACT_CONTRACT_VERSION,
                "diagnostic_id": DIAGNOSTIC_ID,
                "selection_rank": rank,
                "selection_sha256": bytes_sha256(
                    (config["normal_probe"]["selection_prefix"] + path_text).encode()
                ),
                "relative_path": path_text,
                "byte_count": len(content),
                "sha256": bytes_sha256(content),
            }
        )
    return ProbePreflight(
        "0" * 40,
        config,
        schema,
        tuple(partition),
        fixed,
        source,
        work / "no-real-fitted-state",
        output,
        work / "session",
    )


def fake_evidence(pixels: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        score_status="ok", score_failure_code=None, anomaly_score=0.1 + float(pixels.mean()) / 2550
    )


@contextmanager
def fake_classical_scorers(_preflight):
    yield lambda _method, pixels: fake_evidence(pixels)


@contextmanager
def fake_rgb_scorer():
    yield fake_evidence


def synthetic_dino_worker(preflight: ProbePreflight, manifest: Path) -> list[dict]:
    score_transport(
        manifest,
        expected_manifest_sha256=sha256_file(manifest),
        partition=list(preflight.partition),
        config=preflight.config,
        schema=preflight.schema,
        scorer_factory=fake_rgb_scorer,
        source_commit=preflight.source_commit,
    )
    path = manifest.parent / "dinov2-worker/scores.jsonl"
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]
