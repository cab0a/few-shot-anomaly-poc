from __future__ import annotations

import json
from pathlib import Path

import pytest

from few_shot_anomaly_poc.config import load_config
from few_shot_anomaly_poc.errors import ConfigurationError


def test_repository_config_pins_official_split() -> None:
    config = load_config(Path("configs/v0.1.yaml"))

    assert config.dataset_name == "VisA"
    assert config.category == "pcb1"
    assert config.selection.reference_count == 20
    assert config.selection.seed == 42
    assert config.split.revision == "2a692ab575001cbde74d402d897a7286086c6199"
    assert config.split.sha256 == "a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995"


def test_config_rejects_path_outside_project(tmp_path: Path) -> None:
    raw = json.loads(Path("configs/v0.1.yaml").read_text(encoding="utf-8"))
    raw["paths"]["archive"] = "../outside.tar"
    config_path = tmp_path / "configs/v0.1.yaml"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="escapes"):
        load_config(config_path)


def test_config_rejects_unpinned_split_checksum(tmp_path: Path) -> None:
    raw = json.loads(Path("configs/v0.1.yaml").read_text(encoding="utf-8"))
    raw["dataset"]["split"]["sha256"] = "unknown"
    config_path = tmp_path / "configs/v0.1.yaml"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="SHA-256"):
        load_config(config_path)
