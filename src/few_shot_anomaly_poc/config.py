"""Load the versioned data-preparation configuration."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from few_shot_anomaly_poc.errors import ConfigurationError

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ArchiveConfig:
    identifier: str
    url: str
    expected_sha256: str | None


@dataclass(frozen=True)
class SplitConfig:
    repository: str
    revision: str
    path: str
    url: str
    sha256: str


@dataclass(frozen=True)
class SelectionConfig:
    reference_count: int
    seed: int
    procedure_version: str
    namespace: str


@dataclass(frozen=True)
class PreprocessingConfig:
    decode_mode: str
    output_height: int
    output_width: int
    resize_interpolation: str
    output_dtype: str
    scale_divisor: float


@dataclass(frozen=True)
class ProjectPaths:
    archive: Path
    archive_provenance: Path
    extracted: Path
    extraction_provenance: Path
    split_csv: Path
    split_provenance: Path
    manifest_dir: Path


@dataclass(frozen=True)
class ProjectConfig:
    schema_version: str
    dataset_name: str
    category: str
    dataset_license: str
    archive: ArchiveConfig
    split: SplitConfig
    selection: SelectionConfig
    preprocessing: PreprocessingConfig
    paths: ProjectPaths
    project_root: Path


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must be a mapping")
    return value


def _string(mapping: dict[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{label}.{key} must be a non-empty string")
    return value


def _https_url(value: str, label: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ConfigurationError(f"{label} must be an HTTPS URL")
    return value


def _sha256(value: str | None, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ConfigurationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _project_path(project_root: Path, value: str, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise ConfigurationError(f"{label} must be relative to the repository root")
    resolved = (project_root / candidate).resolve()
    if not resolved.is_relative_to(project_root):
        raise ConfigurationError(f"{label} escapes the repository root")
    return resolved


def load_config(config_path: Path) -> ProjectConfig:
    """Load a JSON-compatible YAML file without adding a YAML dependency."""
    resolved_config = config_path.resolve()
    try:
        raw = json.loads(resolved_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"cannot read configuration: {error}") from error

    root = _mapping(raw, "configuration")
    dataset = _mapping(root.get("dataset"), "dataset")
    archive_raw = _mapping(dataset.get("archive"), "dataset.archive")
    split_raw = _mapping(dataset.get("split"), "dataset.split")
    selection_raw = _mapping(root.get("selection"), "selection")
    preprocessing_raw = _mapping(root.get("preprocessing"), "preprocessing")
    paths_raw = _mapping(root.get("paths"), "paths")
    project_root = resolved_config.parent.parent.resolve()

    archive_url = _https_url(_string(archive_raw, "url", "dataset.archive"), "dataset.archive.url")
    split_url = _https_url(_string(split_raw, "url", "dataset.split"), "dataset.split.url")
    repository_url = _https_url(
        _string(split_raw, "repository", "dataset.split"), "dataset.split.repository"
    )

    expected_archive = archive_raw.get("expected_sha256")
    if expected_archive is not None and not isinstance(expected_archive, str):
        raise ConfigurationError("dataset.archive.expected_sha256 must be null or a string")

    reference_count = selection_raw.get("reference_count")
    seed = selection_raw.get("seed")
    if not isinstance(reference_count, int) or isinstance(reference_count, bool):
        raise ConfigurationError("selection.reference_count must be an integer")
    if reference_count <= 0:
        raise ConfigurationError("selection.reference_count must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigurationError("selection.seed must be an integer")

    decode_mode = _string(preprocessing_raw, "decode_mode", "preprocessing")
    output_height = preprocessing_raw.get("output_height")
    output_width = preprocessing_raw.get("output_width")
    resize_interpolation = _string(
        preprocessing_raw,
        "resize_interpolation",
        "preprocessing",
    )
    output_dtype = _string(preprocessing_raw, "output_dtype", "preprocessing")
    scale_divisor = preprocessing_raw.get("scale_divisor")
    if decode_mode != "grayscale_uint8_ignore_orientation":
        raise ConfigurationError("preprocessing.decode_mode must remain preregistered")
    if (
        not isinstance(output_height, int)
        or isinstance(output_height, bool)
        or output_height != 512
    ):
        raise ConfigurationError("preprocessing.output_height must remain 512")
    if not isinstance(output_width, int) or isinstance(output_width, bool) or output_width != 512:
        raise ConfigurationError("preprocessing.output_width must remain 512")
    if resize_interpolation != "area":
        raise ConfigurationError("preprocessing.resize_interpolation must remain area")
    if output_dtype != "float32":
        raise ConfigurationError("preprocessing.output_dtype must remain float32")
    if (
        not isinstance(scale_divisor, (int, float))
        or isinstance(scale_divisor, bool)
        or float(scale_divisor) != 255.0
    ):
        raise ConfigurationError("preprocessing.scale_divisor must remain 255.0")

    path_values = {
        key: _project_path(
            project_root,
            _string(paths_raw, key, "paths"),
            f"paths.{key}",
        )
        for key in (
            "archive",
            "archive_provenance",
            "extracted",
            "extraction_provenance",
            "split_csv",
            "split_provenance",
            "manifest_dir",
        )
    }

    return ProjectConfig(
        schema_version=_string(root, "schema_version", "configuration"),
        dataset_name=_string(dataset, "name", "dataset"),
        category=_string(dataset, "category", "dataset"),
        dataset_license=_string(dataset, "license", "dataset"),
        archive=ArchiveConfig(
            identifier=_string(archive_raw, "identifier", "dataset.archive"),
            url=archive_url,
            expected_sha256=_sha256(
                expected_archive,
                "dataset.archive.expected_sha256",
                optional=True,
            ),
        ),
        split=SplitConfig(
            repository=repository_url,
            revision=_string(split_raw, "revision", "dataset.split"),
            path=_string(split_raw, "path", "dataset.split"),
            url=split_url,
            sha256=_sha256(
                _string(split_raw, "sha256", "dataset.split"),
                "dataset.split.sha256",
            ),
        ),
        selection=SelectionConfig(
            reference_count=reference_count,
            seed=seed,
            procedure_version=_string(
                selection_raw,
                "procedure_version",
                "selection",
            ),
            namespace=_string(selection_raw, "namespace", "selection"),
        ),
        preprocessing=PreprocessingConfig(
            decode_mode=decode_mode,
            output_height=output_height,
            output_width=output_width,
            resize_interpolation=resize_interpolation,
            output_dtype=output_dtype,
            scale_divisor=float(scale_divisor),
        ),
        paths=ProjectPaths(**path_values),
        project_root=project_root,
    )
