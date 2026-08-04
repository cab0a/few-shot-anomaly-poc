from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.jsonio import write_json_atomic
from few_shot_anomaly_poc.v0_2_boundary_preparation import NORMAL_MANIFEST_SCHEMA
from few_shot_anomaly_poc.v0_2_calibration_artifacts import build_fit_record
from few_shot_anomaly_poc.v0_2_evaluation_contract import (
    load_v0_2_artifact_schema,
    load_v0_2_config,
)
from few_shot_anomaly_poc.v0_2_normal_calibration import (
    DINOv2InputAdapterError,
    NormalInputRecord,
    V0_2CalibrationInputs,
    V0_2NormalCalibrationError,
    _isolated_worker_environment,
    _selection_sha256,
    _validate_staged_state_identities,
    adapt_dinov2_source_image,
    load_verified_normal_manifest,
    run_dinov2_worker,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "visa-pcb2-v0-2-final"
REFERENCE_SHA256 = "e587f1808262480261ae8a7b940faff0d9ef5f83cf215028b31490ba48369b99"


def _contract() -> tuple[dict, dict]:
    return (
        load_v0_2_config(ROOT / "configs/v0.2.yaml"),
        load_v0_2_artifact_schema(ROOT / "schemas/v0.2/evaluation-artifacts.json"),
    )


def _inputs() -> V0_2CalibrationInputs:
    config, schema = _contract()
    return V0_2CalibrationInputs(
        boundary_state={},
        calibration_records=(),
        classical_config=None,  # type: ignore[arg-type]
        config=config,
        reference_records=(),
        schema=schema,
    )


def test_dinov2_adapter_matches_fixed_opencv_operations(tmp_path: Path) -> None:
    bgr = np.zeros((7, 9, 3), dtype=np.uint8)
    bgr[:, :, 0] = 17
    bgr[:, :, 1] = np.arange(9, dtype=np.uint8)
    bgr[:, :, 2] = 231
    encoded_ok, encoded = cv2.imencode(".png", bgr)
    assert encoded_ok
    path = tmp_path / "normal.png"
    path.write_bytes(encoded.tobytes())

    observed = adapt_dinov2_source_image(path)
    expected = np.ascontiguousarray(
        cv2.resize(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            (512, 512),
            interpolation=cv2.INTER_AREA,
        ),
        dtype=np.uint8,
    )

    assert observed.shape == (512, 512, 3)
    assert observed.dtype == np.uint8
    assert observed.flags.c_contiguous
    assert np.array_equal(observed, expected)


def test_dinov2_adapter_returns_a_stable_failure_code(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jpg"

    with pytest.raises(DINOv2InputAdapterError) as raised:
        adapt_dinov2_source_image(missing)

    assert raised.value.code == "DINO_IMAGE_READ_FAILED"


def _write_normal_fixture(
    tmp_path: Path,
    *,
    count: int = 2,
) -> tuple[Path, Path]:
    source_root = tmp_path / "source"
    manifest_path = tmp_path / "calibration.jsonl"
    lines = []
    for offset in range(count):
        rank = 21 + offset
        relative_path = f"pcb2/Data/Images/Normal/{offset:04d}.JPG"
        source_path = source_root / relative_path
        source_path.parent.mkdir(parents=True, exist_ok=True)
        content = f"normal-{offset}".encode()
        source_path.write_bytes(content)
        lines.append(
            json.dumps(
                {
                    "byte_count": len(content),
                    "partition": "calibration",
                    "relative_path": relative_path,
                    "schema_version": NORMAL_MANIFEST_SCHEMA,
                    "selection_rank": rank,
                    "selection_sha256": _selection_sha256(relative_path),
                    "sha256": hashlib.sha256(content).hexdigest(),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return source_root, manifest_path


def test_normal_manifest_verifies_rank_selection_and_source_bytes(tmp_path: Path) -> None:
    source_root, manifest_path = _write_normal_fixture(tmp_path)

    records = load_verified_normal_manifest(
        manifest_path,
        partition="calibration",
        expected_count=2,
        first_rank=21,
        source_root=source_root,
    )

    assert records == tuple(
        NormalInputRecord(
            byte_count=len(f"normal-{offset}".encode()),
            partition="calibration",
            relative_path=f"pcb2/Data/Images/Normal/{offset:04d}.JPG",
            selection_rank=21 + offset,
            selection_sha256=_selection_sha256(f"pcb2/Data/Images/Normal/{offset:04d}.JPG"),
            sha256=hashlib.sha256(f"normal-{offset}".encode()).hexdigest(),
        )
        for offset in range(2)
    )


def test_normal_manifest_rejects_changed_source_bytes(tmp_path: Path) -> None:
    source_root, manifest_path = _write_normal_fixture(tmp_path)
    (source_root / "pcb2/Data/Images/Normal/0001.JPG").write_bytes(b"changed")

    with pytest.raises(V0_2NormalCalibrationError, match="byte identity"):
        load_verified_normal_manifest(
            manifest_path,
            partition="calibration",
            expected_count=2,
            first_rank=21,
            source_root=source_root,
        )


def test_staged_state_hashes_must_match_successful_fit_records(tmp_path: Path) -> None:
    inputs = _inputs()
    public_root = tmp_path / "public"
    state_root = tmp_path / "state"
    state_root.mkdir()
    state_path = state_root / "ecc_residual.pkl"
    state_path.write_bytes(b"trusted local state fixture")
    for method in ("ecc_residual", "patch_hog_ocsvm", "dinov2_vits14_224_nn"):
        successful = method == "ecc_residual"
        fit = build_fit_record(
            run_id=RUN_ID,
            method=method,
            status="fit_ok" if successful else "fit_failed",
            successful_reference_count=20 if successful else 0,
            failed_reference_count=0 if successful else 20,
            reference_manifest_sha256=REFERENCE_SHA256,
            fitted_state_sha256=sha256_file(state_path) if successful else None,
            failure_code=None if successful else "synthetic_fit_failure",
            config=inputs.config,
            schema=inputs.schema,
        )
        write_json_atomic(public_root / method / "fit.json", fit)

    _validate_staged_state_identities(
        public_stage_root=public_root,
        state_stage_root=state_root,
        inputs=inputs,
    )
    state_path.write_bytes(b"tampered")
    with pytest.raises(V0_2NormalCalibrationError, match="fitted-state identity"):
        _validate_staged_state_identities(
            public_stage_root=public_root,
            state_stage_root=state_root,
            inputs=inputs,
        )


def test_dinov2_worker_command_excludes_evaluation_boundary_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    def fake_run(command, **_kwargs):
        observed.extend(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "few_shot_anomaly_poc.v0_2_normal_calibration.subprocess.run",
        fake_run,
    )
    run_dinov2_worker(
        project_root=ROOT,
        execution_commit="a" * 40,
        input_store_path=tmp_path / "normal.npy",
        input_manifest_path=tmp_path / "normal.json",
        public_stage_root=tmp_path / "public",
        state_stage_root=tmp_path / "state",
        progress=None,
    )

    serialized = " ".join(observed).lower()
    assert "sealed" not in serialized
    assert "scorer" not in serialized
    assert "final-test" not in serialized
    assert "ordering-key" not in serialized


def test_isolated_worker_environment_preserves_cpu_and_memory_bounds() -> None:
    environment = _isolated_worker_environment()

    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["OMP_NUM_THREADS"] == "4"
    assert environment["MKL_NUM_THREADS"] == "4"
    assert environment["OPENBLAS_NUM_THREADS"] == "4"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["XFORMERS_DISABLED"] == "1"
    assert "PYTHONPATH" not in environment
