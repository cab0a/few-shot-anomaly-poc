from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from few_shot_anomaly_poc.config import ProjectConfig, load_config
from few_shot_anomaly_poc.errors import EvaluationArtifactError
from few_shot_anomaly_poc.evaluation_artifacts import (
    EvaluationArtifactBundle,
    write_evaluation_artifact_bundle,
)
from few_shot_anomaly_poc.synthetic_evaluation import (
    SYNTHETIC_RUN_ID,
    build_synthetic_evaluation_bundle,
    run_synthetic_evaluation,
)

SOURCE_COMMIT = "a" * 40
COMMITTED_SOURCE_COMMIT = "7193a89e0cff8d543c0f7274e834d902026752d5"
CONFIG_PATH = Path("configs/v0.1.yaml")
COMMITTED_BUNDLE = Path("artifacts/v0.1/evaluation/synthetic-e2e")


@pytest.fixture(scope="module")
def project_config() -> ProjectConfig:
    return load_config(CONFIG_PATH)


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _run(tmp_path: Path, config: ProjectConfig) -> Path:
    return run_synthetic_evaluation(
        output_root=tmp_path,
        source_commit=SOURCE_COMMIT,
        config_path=CONFIG_PATH,
        config=config,
    )


def test_synthetic_evaluation_connects_all_primitives_and_artifacts(
    tmp_path: Path,
    project_config: ProjectConfig,
) -> None:
    output = _run(tmp_path, project_config)

    assert output == tmp_path / SYNTHETIC_RUN_ID
    manifest = json.loads((output / "artifact-manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_kind"] == "synthetic"
    assert manifest["dataset"] == "synthetic-records"
    assert manifest["category"] == "not-applicable"
    assert manifest["source_commit"] == SOURCE_COMMIT
    assert len(manifest["files"]) == 16
    assert tuple(entry["relative_path"] for entry in manifest["files"]) == tuple(
        sorted(entry["relative_path"] for entry in manifest["files"])
    )

    for method in ("ecc_residual", "patch_hog_one_class_svm"):
        method_dir = output / method
        metrics = json.loads((method_dir / "metrics.json").read_text(encoding="utf-8"))
        decision = json.loads((method_dir / "decision.json").read_text(encoding="utf-8"))
        with (method_dir / "scores.csv").open(encoding="utf-8", newline="") as stream:
            scores = tuple(csv.DictReader(stream))
        with (method_dir / "classifications.csv").open(
            encoding="utf-8",
            newline="",
        ) as stream:
            classifications = tuple(csv.DictReader(stream))
        with (method_dir / "revealed-labels.csv").open(
            encoding="utf-8",
            newline="",
        ) as stream:
            labels = tuple(csv.DictReader(stream))
        with (method_dir / "failure-cases.csv").open(
            encoding="utf-8",
            newline="",
        ) as stream:
            failures = tuple(csv.DictReader(stream))

        assert len(scores) == 60
        assert tuple(row["partition"] for row in scores[:20]) == ("calibration",) * 20
        assert tuple(row["partition"] for row in scores[20:]) == ("final_test",) * 40
        assert len(classifications) == 40
        assert len(labels) == 40
        assert len(failures) == 3
        assert "true_class" not in scores[0]
        assert "true_class" not in classifications[0]
        assert set(labels[0]) == {
            "contract_version",
            "run_id",
            "run_kind",
            "method",
            "relative_path",
            "true_class",
        }
        assert metrics["normal_false_positive_rate"] == 0.05
        assert metrics["anomaly_recall"] == 0.9
        assert metrics["false_positive_count"] == 1
        assert metrics["false_negative_count"] == 2
        assert decision["decision"] == "ADOPT"
        assert decision["all_hard_gates_passed"] is True
        assert len(decision["gate_outcomes"]) == 6
        assert all(gate["passed"] for gate in decision["gate_outcomes"])


def test_synthetic_bundle_is_byte_reproducible(
    tmp_path: Path,
    project_config: ProjectConfig,
) -> None:
    first = _run(tmp_path / "first", project_config)
    second = _run(tmp_path / "second", project_config)

    assert _files(first) == _files(second)


def test_committed_synthetic_bundle_matches_its_source_commit(
    tmp_path: Path,
    project_config: ProjectConfig,
) -> None:
    manifest = json.loads((COMMITTED_BUNDLE / "artifact-manifest.json").read_text(encoding="utf-8"))
    regenerated = run_synthetic_evaluation(
        output_root=tmp_path,
        source_commit=COMMITTED_SOURCE_COMMIT,
        config_path=CONFIG_PATH,
        config=project_config,
    )

    assert manifest["source_commit"] == COMMITTED_SOURCE_COMMIT
    assert manifest["run_kind"] == "synthetic"
    assert _files(COMMITTED_BUNDLE) == _files(regenerated)


def test_manifest_hashes_every_non_manifest_file(
    tmp_path: Path,
    project_config: ProjectConfig,
) -> None:
    output = _run(tmp_path, project_config)
    manifest = json.loads((output / "artifact-manifest.json").read_text(encoding="utf-8"))
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "artifact-manifest.json"
    }

    assert {entry["relative_path"] for entry in manifest["files"]} == actual_files
    for entry in manifest["files"]:
        content = (output / entry["relative_path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]
        assert not Path(entry["relative_path"]).is_absolute()


def test_synthetic_bundle_contains_no_timestamp_or_machine_path(
    tmp_path: Path,
    project_config: ProjectConfig,
) -> None:
    output = _run(tmp_path, project_config)
    serialized = b"\n".join(_files(output).values()).decode("utf-8")

    assert "run_kind,method" in serialized
    assert "synthetic" in serialized
    assert "not VisA performance evidence" not in serialized
    assert "/home/" not in serialized
    assert "\\\\wsl.localhost" not in serialized
    assert "2026-" not in serialized


def test_artifact_writer_refuses_overwrite(
    tmp_path: Path,
    project_config: ProjectConfig,
) -> None:
    _run(tmp_path, project_config)

    with pytest.raises(EvaluationArtifactError, match="refusing to overwrite"):
        _run(tmp_path, project_config)


def test_artifact_writer_rejects_tampered_intermediate_result_without_output(
    tmp_path: Path,
    project_config: ProjectConfig,
) -> None:
    bundle = build_synthetic_evaluation_bundle(
        source_commit=SOURCE_COMMIT,
        config=project_config,
    )
    first = bundle.methods[0]
    assert first.metrics.false_positive_count == 1
    tampered_method = replace(
        first,
        metrics=replace(first.metrics, false_positive_count=0),
    )
    tampered = EvaluationArtifactBundle(
        run_id=bundle.run_id,
        run_kind=bundle.run_kind,
        dataset=bundle.dataset,
        category=bundle.category,
        source_commit=bundle.source_commit,
        partition_manifest_sha256=bundle.partition_manifest_sha256,
        methods=(tampered_method, *bundle.methods[1:]),
    )

    with pytest.raises(EvaluationArtifactError, match="metric artifact"):
        write_evaluation_artifact_bundle(
            tampered,
            output_root=tmp_path,
            config_path=CONFIG_PATH,
            config=project_config,
        )

    assert not (tmp_path / SYNTHETIC_RUN_ID).exists()


@pytest.mark.parametrize(
    "source_commit",
    [
        "",
        "a" * 39,
        "g" * 40,
    ],
)
def test_synthetic_bundle_rejects_invalid_source_commit(
    tmp_path: Path,
    source_commit: str,
    project_config: ProjectConfig,
) -> None:
    bundle = build_synthetic_evaluation_bundle(
        source_commit=source_commit,
        config=project_config,
    )

    with pytest.raises(EvaluationArtifactError, match="bundle metadata"):
        write_evaluation_artifact_bundle(
            bundle,
            output_root=tmp_path,
            config_path=CONFIG_PATH,
            config=project_config,
        )
