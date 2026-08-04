from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_OR_MODEL_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".joblib",
    ".npy",
    ".npz",
    ".onnx",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".tar",
    ".tif",
    ".tiff",
    ".webp",
    ".zip",
}
PROHIBITED_PROJECT_EXPRESSIONS = (
    "open" + " source",
    "gpl" + " project",
    "mit" + " licensed",
    "free for " + "commercial use",
)
MARKDOWN_LINK = re.compile(r"\]\(([^)]+)\)")


def _tracked_paths() -> tuple[Path, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return tuple(
        ROOT / raw.decode("utf-8")
        for raw in completed.stdout.split(b"\0")
        if raw
    )


def test_tracked_repository_contains_no_raw_image_archive_or_model_asset() -> None:
    forbidden = tuple(
        path.relative_to(ROOT).as_posix()
        for path in _tracked_paths()
        if path.suffix.lower() in RAW_OR_MODEL_SUFFIXES
    )

    assert forbidden == ()


def test_public_license_language_and_rights_boundaries_are_explicit() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    notice = (ROOT / "NOTICE.md").read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    public_markdown = "\n".join(
        path.read_text(encoding="utf-8")
        for path in _tracked_paths()
        if path.suffix == ".md"
    ).lower()

    assert "source-available" in readme
    assert "noncommercially licensed public portfolio project" in readme
    assert "See [`LICENSE`](LICENSE) for the controlling terms." in readme
    assert all(expression not in public_markdown for expression in PROHIBITED_PROJECT_EXPRESSIONS)
    assert "Copyright © 2026 Takaya Nakanishi" in notice
    assert "The PolyForm license does not apply to VisA" in notice
    assert "Governed by their respective licenses" in notice
    assert license_text.startswith("# PolyForm Noncommercial License 1.0.0\n")
    assert "<https://polyformproject.org/licenses/noncommercial/1.0.0>" in license_text


def test_v0_2_scoring_stays_isolated_and_deferred_methods_remain_out_of_scope() -> None:
    source_names = {
        path.name.lower()
        for path in (ROOT / "src/few_shot_anomaly_poc").glob("*.py")
    }
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()

    assert {name for name in source_names if "dinov2" in name} == {
        "dinov2_errors.py",
        "dinov2_reproduction.py",
        "dinov2_scoring.py",
        "dinov2_scoring_smoke.py",
        "dinov2_timing.py",
        "dinov2_timing_preflight.py",
        "v0_2_dinov2_calibration.py",
        "v0_2_dinov2_scoring_run.py",
    }
    assert not any(
        token in name for name in source_names for token in ("anomalydino", "patchcore")
    )
    assert all(
        dependency not in project
        for dependency in ("torch", "torchvision", "transformers", "timm")
    )
    assert not any(
        "mask" in path.name.lower()
        for path in _tracked_paths()
        if path.is_file() and path.parts[: len(ROOT.parts)] == ROOT.parts
    )


def test_all_tracked_local_markdown_links_resolve_inside_the_repository() -> None:
    failures: list[str] = []
    for path in _tracked_paths():
        if path.suffix != ".md":
            continue
        for match in MARKDOWN_LINK.finditer(path.read_text(encoding="utf-8")):
            raw_target = match.group(1)
            target = raw_target.split("#", maxsplit=1)[0]
            if not target or re.match(r"^[a-z]+://", target) or target.startswith("mailto:"):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists() or (
                ROOT not in resolved.parents and resolved != ROOT
            ):
                failures.append(f"{path.relative_to(ROOT)} -> {raw_target}")

    assert failures == []
