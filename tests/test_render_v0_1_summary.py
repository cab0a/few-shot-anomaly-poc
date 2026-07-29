from __future__ import annotations

from pathlib import Path

from scripts.render_v0_1_summary import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_OUTPUT,
    render_summary,
)


def test_committed_v0_1_summary_matches_numeric_artifacts(tmp_path: Path) -> None:
    regenerated = tmp_path / "v0.1-gate-summary.svg"

    render_summary(artifact_root=DEFAULT_ARTIFACT_ROOT, output=regenerated)

    assert regenerated.read_bytes() == DEFAULT_OUTPUT.read_bytes()


def test_v0_1_summary_contains_no_timestamp_or_machine_path(tmp_path: Path) -> None:
    regenerated = tmp_path / "v0.1-gate-summary.svg"

    render_summary(artifact_root=DEFAULT_ARTIFACT_ROOT, output=regenerated)
    content = regenerated.read_text(encoding="utf-8")

    assert "2026-" not in content
    assert "/home/" not in content
    assert "\\\\wsl.localhost" not in content
    assert "REJECT" in content
    assert "Normal FPR" in content
    assert "Anomaly recall" in content
    assert "CPU p95" in content
