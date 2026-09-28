from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import few_shot_anomaly_poc.v0_3_controlled_probe as module
from few_shot_anomaly_poc.v0_3_diagnostic_contract import (
    CONDITIONS,
    METHODS,
    V0_3DiagnosticContractError,
    load_v0_3_config,
    load_v0_3_schema,
    sha256_file,
)
from few_shot_anomaly_poc.v0_3_dinov2_probe import score_transport
from few_shot_anomaly_poc.v0_3_probe_artifacts import (
    SCORE_NAME,
    SUMMARY_NAME,
    ProbeError,
    json_bytes,
    read_table,
    score_record,
    serialize_table,
    summarize_scores,
    validate_score,
    validate_scores,
    verify_probe_outputs,
)
from few_shot_anomaly_poc.v0_3_probe_synthetic import (
    fake_classical_scorers,
    fake_evidence,
    synthetic_dino_worker,
    synthetic_preflight,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_v0_3_config(ROOT / "configs/v0.3.yaml")
SCHEMA = load_v0_3_schema(ROOT / "schemas/v0.3/diagnostic-artifacts.json")


@pytest.fixture
def preflight(tmp_path):
    return synthetic_preflight(tmp_path, ROOT)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_transforms_preserve_extent_dtype_original_and_immutable_results(condition):
    base = np.arange(13 * 17 * 3, dtype=np.uint16).astype(np.uint8).reshape(13, 17, 3)
    original = base.copy()
    transformed = module.transform_image(base, condition)
    assert transformed.shape == base.shape
    assert transformed.dtype == np.uint8
    assert transformed.flags.c_contiguous and not transformed.flags.writeable
    assert np.array_equal(base, original)
    with pytest.raises(ValueError):
        transformed.flags.writeable = True
    if condition == "control":
        assert transformed.tobytes() == original.tobytes()


def test_brightness_uses_original_uint8_half_even_rounding_and_clipping():
    values = [0, 1, 10, 20, 30, 100, 200, 254, 255]
    base = np.repeat(np.array(values, dtype=np.uint8)[None, :, None], 3, axis=2)
    assert module.transform_image(base, "brightness_085")[0, :, 0].tolist() == [
        0,
        1,
        8,
        17,
        26,
        85,
        170,
        216,
        217,
    ]
    assert module.transform_image(base, "brightness_115")[0, :, 0].tolist() == [
        0,
        1,
        12,
        23,
        34,
        115,
        230,
        255,
        255,
    ]


def test_translation_is_positive_four_pixels_with_reflected_original_borders():
    base = np.arange(11 * 13 * 3, dtype=np.uint16).astype(np.uint8).reshape(11, 13, 3)
    expected = np.pad(base, ((4, 0), (4, 0), (0, 0)), mode="reflect")[:11, :13]
    assert np.array_equal(module.transform_image(base, "translate_4_4"), expected)


def test_gaussian_matches_independent_sigma_one_five_tap_convolution():
    base = np.zeros((9, 13, 3), dtype=np.uint8)
    base[0, 0] = [255, 127, 31]
    base[4, 6] = [80, 180, 240]
    weights = np.array([math.exp(-x * x / 2) for x in range(-2, 3)])
    weights /= weights.sum()
    padded = np.pad(base.astype(np.float64), ((2, 2), (2, 2), (0, 0)), mode="reflect")
    expected = sum(
        weights[y] * weights[x] * padded[y : y + 9, x : x + 13] for y in range(5) for x in range(5)
    )
    observed = module.transform_image(base, "gaussian_blur")
    assert np.max(np.abs(observed.astype(float) - np.rint(expected))) <= 1
    assert observed[4, 6].tolist() != base[4, 6].tolist()


def make_scores(partition):
    rows = []
    for method in METHODS:
        threshold = CONFIG["methods"][method]["threshold"]
        for item in partition:
            rank = item["selection_rank"]
            control = threshold + (0 if rank <= 30 else 0.02)
            changes = {
                "control": 0,
                "brightness_085": -0.05,
                "brightness_115": 0.1,
                "gaussian_blur": rank / 1000,
                "translate_4_4": 0,
            }
            for condition in CONDITIONS:
                rows.append(
                    score_record(
                        method=method,
                        selection_rank=rank,
                        condition=condition,
                        base_sha256=item["sha256"],
                        status="ok",
                        failure_code=None,
                        score=control + changes[condition],
                        config=CONFIG,
                        schema=SCHEMA,
                    )
                )
    return rows


def test_paired_summaries_use_strict_threshold_and_nearest_rank_p95(preflight):
    rows = make_scores(preflight.partition)
    summaries = summarize_scores(rows, preflight.partition, config=CONFIG, schema=SCHEMA)
    assert len(rows) == 900 and len(summaries) == 15
    for method in METHODS:
        selected = {row["condition"]: row for row in summaries if row["method"] == method}
        assert selected["control"]["above_threshold_count"] == 30
        assert selected["control"]["above_threshold_rate"] == 0.5
        assert selected["control"]["median_signed_delta"] is None
        assert selected["control"]["normal_to_anomalous_crossing_count"] == 0
        assert selected["brightness_085"]["above_threshold_count"] == 0
        assert selected["brightness_085"]["anomalous_to_normal_crossing_count"] == 30
        assert selected["brightness_115"]["above_threshold_count"] == 60
        assert selected["brightness_115"]["normal_to_anomalous_crossing_count"] == 30
        assert selected["gaussian_blur"]["median_signed_delta"] == pytest.approx(0.0305)
        assert selected["gaussian_blur"]["median_absolute_delta"] == pytest.approx(0.0305)
        assert selected["gaussian_blur"]["nearest_rank_p95_absolute_delta"] == pytest.approx(0.057)
        assert selected["translate_4_4"]["median_absolute_delta"] == 0


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "reorder", "hash", "threshold", "flag", "nan", "range"]
)
def test_bad_score_evidence_rejected(preflight, change):
    rows = make_scores(preflight.partition)
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    elif change == "reorder":
        rows[:2] = reversed(rows[:2])
    else:
        field, value = {
            "hash": ("base_image_sha256", "0" * 64),
            "threshold": ("threshold", 0.3),
            "flag": ("above_threshold", True),
            "nan": ("anomaly_score", float("nan")),
            "range": ("anomaly_score", 2.1),
        }[change]
        rows[0][field] = value
    with pytest.raises((ProbeError, V0_3DiagnosticContractError)):
        validate_scores(rows, preflight.partition, config=CONFIG, schema=SCHEMA)


def test_failed_score_is_retained_but_cannot_generate_complete_paired_summaries(preflight):
    rows = make_scores(preflight.partition)
    rows[0].update(
        score_status="failed",
        score_failure_code="ECC_SCORE_EXECUTION_FAILED",
        anomaly_score=1.0,
        above_threshold=True,
    )
    validate_score(rows[0], config=CONFIG, schema=SCHEMA)
    with pytest.raises(ProbeError, match="failed scores"):
        summarize_scores(rows, preflight.partition, config=CONFIG, schema=SCHEMA)


def test_csv_round_trip_and_tampered_summary_rejection(preflight):
    rows = make_scores(preflight.partition)
    summaries = summarize_scores(rows, preflight.partition, config=CONFIG, schema=SCHEMA)
    output = preflight.output_root
    (output / SCORE_NAME).write_bytes(serialize_table("diagnostic_score", rows, schema=SCHEMA))
    (output / SUMMARY_NAME).write_bytes(
        serialize_table("condition_summary", summaries, schema=SCHEMA)
    )
    assert read_table(output / SCORE_NAME, "diagnostic_score", schema=SCHEMA) == rows
    verify_probe_outputs(output, preflight.partition, config=CONFIG, schema=SCHEMA)
    summaries[1]["median_signed_delta"] = 0.123
    (output / SUMMARY_NAME).write_bytes(
        serialize_table("condition_summary", summaries, schema=SCHEMA)
    )
    with pytest.raises(ProbeError, match="paired summary differs"):
        verify_probe_outputs(output, preflight.partition, config=CONFIG, schema=SCHEMA)


def test_full_synthetic_pipeline_decodes_exactly_60_and_scores_each_slot_once(preflight):
    decoded, called = [], []

    def loader(root, identity):
        decoded.append(identity["selection_rank"])
        return module.decode_normal(root, identity)

    @contextmanager
    def scorers(_preflight):
        def score(method, pixels):
            assert not pixels.flags.writeable
            called.append(method)
            return fake_evidence(pixels)

        yield score

    complete = module.run_probe(
        preflight,
        loader=loader,
        scorer_factory=scorers,
        dino_worker=synthetic_dino_worker,
        progress=lambda _text: None,
    )
    assert decoded == list(range(1, 61))
    assert called.count(METHODS[0]) == called.count(METHODS[1]) == 300
    assert complete["first_probe_complete"] is True
    assert complete["score_count"] == 900 and complete["summary_count"] == 15
    assert not (preflight.session_root / "stopped.json").exists()
    assert sha256_file(preflight.output_root / SCORE_NAME) == complete["controlled_scores_sha256"]
    synthetic_report = json.loads(
        (ROOT / "artifacts/v0.3/synthetic/controlled-probe-verification.json").read_text()
    )
    assert (
        complete["controlled_scores_sha256"]
        == synthetic_report["synthetic_controlled_scores_sha256"]
    )
    assert (
        complete["condition_summaries_sha256"]
        == synthetic_report["synthetic_condition_summaries_sha256"]
    )
    assert all(sha256_file(path) == digest for path, digest in preflight.fixed_files.items())
    with pytest.raises(ProbeError, match="session already exists"):
        module.run_probe(preflight)


@pytest.mark.parametrize(
    "failure", ["changed_image", "score_failure", "exception", "interrupt", "changed_parent"]
)
def test_stopped_attempt_preserved_without_public_results(preflight, failure):
    if failure == "changed_image":
        (preflight.source_root / preflight.partition[0]["relative_path"]).write_bytes(b"changed")

    @contextmanager
    def scorers(_preflight):
        def score(method, pixels):
            if failure == "score_failure":
                return SimpleNamespace(
                    score_status="failed",
                    score_failure_code="SYNTHETIC_FAILURE",
                    anomaly_score=CONFIG["methods"][method]["failure_score"],
                )
            if failure == "exception":
                raise RuntimeError("synthetic failure")
            if failure == "interrupt":
                raise KeyboardInterrupt
            if failure == "changed_parent":
                next(iter(preflight.fixed_files)).write_bytes(b"changed evidence")
                raise RuntimeError("stop after mutation")
            return fake_evidence(pixels)

        yield score

    with pytest.raises((ProbeError, RuntimeError, KeyboardInterrupt)):
        module.run_probe(
            preflight,
            scorer_factory=scorers,
            dino_worker=lambda *_args: pytest.fail("worker called after failure"),
            progress=lambda _text: None,
        )
    assert (preflight.session_root / "input-verification-attempts.jsonl").is_file()
    assert (preflight.session_root / "stopped.json").is_file()
    assert not (preflight.output_root / SCORE_NAME).exists()
    assert not (preflight.output_root / SUMMARY_NAME).exists()
    if failure == "score_failure":
        row = json.loads((preflight.session_root / "scores.jsonl").read_text())
        assert row["score_status"] == "failed"


def test_existing_output_rejects_before_any_image_or_model_loading(preflight):
    (preflight.output_root / SCORE_NAME).write_bytes(b"preserve")
    with pytest.raises(ProbeError, match="already exists"):
        module.run_probe(preflight, loader=lambda *_args: pytest.fail("image accessed"))
    assert (preflight.output_root / SCORE_NAME).read_bytes() == b"preserve"
    assert not preflight.session_root.exists()


def test_changed_state_rejects_before_session_or_decode(preflight):
    first = next(iter(preflight.fixed_files))
    first.write_bytes(b"changed state")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        module.run_probe(preflight, loader=lambda *_args: pytest.fail("image accessed"))
    assert not preflight.session_root.exists()


def test_preflight_missing_external_metadata_opens_no_image(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "check_pushed_source", lambda _root: "a" * 40)
    # The real completed outputs stay intact; isolate this missing-input check.
    monkeypatch.setattr(module, "check_unstarted", lambda *_args: None)
    monkeypatch.setattr(module.subprocess, "run", lambda *_a, **_k: SimpleNamespace(stdout=""))
    monkeypatch.setattr(module, "decode_normal", lambda *_a: pytest.fail("image decoded"))
    monkeypatch.setattr(module, "classical_scorers", lambda *_a: pytest.fail("state loaded"))
    with pytest.raises(Exception, match="missing normal reference manifest"):
        module.preflight_probe(root=ROOT, external_root=tmp_path / "absent", state_root=tmp_path)


def test_transported_pixel_mutation_stops_worker_before_scoring(preflight, monkeypatch):
    # Generate the store through the real pipeline, but stop before invoking its worker.
    captured = []

    def stop_worker(_preflight, manifest):
        captured.append(manifest)
        raise RuntimeError("transport staged")

    with pytest.raises(RuntimeError, match="transport staged"):
        module.run_probe(
            preflight,
            scorer_factory=fake_classical_scorers,
            dino_worker=stop_worker,
            progress=lambda _text: None,
        )
    manifest_path = captured[0]
    manifest = json.loads(manifest_path.read_text())
    manifest["records"][0]["rgb_sha256"] = "f" * 64
    manifest_path.write_bytes(json_bytes(manifest))
    called = []

    @contextmanager
    def scorer():
        yield lambda image: called.append(image) or fake_evidence(image)

    with pytest.raises(ProbeError, match="RGB transport pixels changed"):
        score_transport(
            manifest_path,
            expected_manifest_sha256=sha256_file(manifest_path),
            partition=list(preflight.partition),
            config=CONFIG,
            schema=SCHEMA,
            scorer_factory=scorer,
            source_commit=preflight.source_commit,
        )
    assert not called
    assert (manifest_path.parent / "dinov2-worker/stopped.json").is_file()
    assert not (manifest_path.parent / "dinov2-worker/complete.json").exists()


def test_classical_adapter_uses_gray_after_original_extent_transform(preflight, monkeypatch):
    import few_shot_anomaly_poc.v0_2_label_free_scoring as parent

    (preflight.source_root / "configs").mkdir()
    (preflight.source_root / "configs/v0.1.yaml").write_bytes(
        (ROOT / "configs/v0.1.yaml").read_bytes()
    )
    monkeypatch.setattr(module, "verify_file", lambda *_args: None)
    monkeypatch.setattr(parent, "_load_classical_state", lambda *_a, **_k: {})
    captured = []
    monkeypatch.setattr(
        parent,
        "_classical_evidence",
        lambda gray, **_k: captured.append(gray) or fake_evidence(gray),
    )
    base = np.zeros((9, 13, 3), dtype=np.uint8)
    base[:, :] = [10, 80, 200]
    transformed = module.transform_image(base, "brightness_085")
    with module.classical_scorers(
        replace(preflight, repository_root=preflight.source_root)
    ) as scorer:
        scorer(METHODS[0], transformed)
    assert np.array_equal(captured[0], cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY))


def test_worker_partial_failure_keeps_returned_failure_record(preflight):
    captured = []

    def stop_worker(_preflight, manifest):
        captured.append(manifest)
        raise RuntimeError("transport staged")

    with pytest.raises(RuntimeError):
        module.run_probe(
            preflight,
            scorer_factory=fake_classical_scorers,
            dino_worker=stop_worker,
            progress=lambda _text: None,
        )

    @contextmanager
    def scorer():
        yield lambda _image: SimpleNamespace(
            score_status="failed", score_failure_code="DINO_TEST_FAILED", anomaly_score=2.0
        )

    manifest = captured[0]
    with pytest.raises(ProbeError, match="score failed"):
        score_transport(
            manifest,
            expected_manifest_sha256=sha256_file(manifest),
            partition=list(preflight.partition),
            config=CONFIG,
            schema=SCHEMA,
            scorer_factory=scorer,
            source_commit=preflight.source_commit,
        )
    worker = manifest.parent / "dinov2-worker"
    assert json.loads((worker / "scores.jsonl").read_text())["anomaly_score"] == 2.0
    assert json.loads((worker / "stopped.json").read_text())["score_count"] == 1


def test_changed_last_image_stops_before_loading_any_scorer(preflight):
    (preflight.source_root / preflight.partition[-1]["relative_path"]).write_bytes(b"changed")
    with pytest.raises(ProbeError, match="normal image bytes changed"):
        module.run_probe(
            preflight,
            loader=lambda *_args: pytest.fail("image decoded before all hashes verified"),
            scorer_factory=lambda *_args: pytest.fail("scorer loaded before all hashes verified"),
        )
    assert not (preflight.session_root / "score-attempts.jsonl").exists()


def test_inherited_scoring_source_cannot_change(tmp_path, monkeypatch):
    import few_shot_anomaly_poc.v0_3_probe_runtime as runtime

    path = tmp_path / "src/few_shot_anomaly_poc/config.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original source")
    monkeypatch.setattr(runtime, "SCORER_MODULES", ("config",))
    monkeypatch.setattr(
        runtime.subprocess, "run", lambda *_a, **_k: SimpleNamespace(stdout=b"original source")
    )
    assert runtime.verify_scorer_sources(tmp_path) == {path: sha256_file(path)}
    path.write_bytes(b"changed algorithm")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        runtime.verify_scorer_sources(tmp_path)


def test_committed_synthetic_report_identifies_current_pipeline_and_closed_boundaries():
    report = json.loads(
        (ROOT / "artifacts/v0.3/synthetic/controlled-probe-verification.json").read_text()
    )
    assert report["generated_image_count"] == 60
    assert report["synthetic_score_count"] == 900
    assert report["summary_count"] == 15
    for flag in (
        "visa_images_accessed",
        "anomaly_method_executed",
        "model_or_fitted_state_loaded",
        "latency_measured",
        "real_probe_executed",
    ):
        assert report[flag] is False
    for name, digest in report["implementation_sha256"].items():
        assert sha256_file(ROOT / "src/few_shot_anomaly_poc" / name) == digest
