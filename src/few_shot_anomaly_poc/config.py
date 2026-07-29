"""Load the versioned v0.1 project configuration."""

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
class ECCRegistrationConfig:
    motion_model: str
    initial_warp: str
    termination: str
    max_iterations: int
    epsilon: float
    gaussian_filter_size: int
    warp_interpolation: str
    mask_interpolation: str
    warp_border: str
    max_abs_rotation_degrees: float
    max_abs_horizontal_translation_pixels: float
    max_abs_vertical_translation_pixels: float
    min_valid_fraction: float


@dataclass(frozen=True)
class ECCTemplateConfig:
    anchor_selection: str
    minimum_successful_references: int
    support_mask: str
    support_erosion_kernel_size: int
    support_erosion_iterations: int
    support_erosion_border: str
    minimum_support_fraction: float
    aggregation: str


@dataclass(frozen=True)
class ECCResidualScoringConfig:
    validity_erosion_kernel_size: int
    validity_erosion_iterations: int
    validity_erosion_border: str
    minimum_effective_support_fraction: float
    residual: str
    gaussian_kernel_size: int
    gaussian_sigma: float
    gaussian_border: str
    top_fraction: float
    top_count_rounding: str
    failure_score: float


@dataclass(frozen=True)
class PatchHOGConfig:
    patch_height: int
    patch_width: int
    vertical_stride: int
    horizontal_stride: int
    vertical_positions: int
    horizontal_positions: int
    patch_count: int
    ordering: str
    orientations: int
    pixels_per_cell_height: int
    pixels_per_cell_width: int
    cells_per_block_height: int
    cells_per_block_width: int
    block_norm: str
    transform_sqrt: bool
    visualize: bool
    feature_vector: bool
    channel_axis: str
    descriptor_length: int
    output_dtype: str


@dataclass(frozen=True)
class PatchHOGScalerConfig:
    reference_order: str
    fitting_scope: str
    copy: bool
    with_mean: bool
    with_std: bool


@dataclass(frozen=True)
class PatchHOGOneClassSVMConfig:
    fitting_scope: str
    kernel: str
    gamma: str
    nu: float
    tolerance: float
    shrinking: bool
    cache_size_mb: float
    max_iterations: int
    verbose: bool


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
    ecc_registration: ECCRegistrationConfig
    ecc_template: ECCTemplateConfig
    ecc_residual_scoring: ECCResidualScoringConfig
    patch_hog: PatchHOGConfig
    patch_hog_scaler: PatchHOGScalerConfig
    patch_hog_one_class_svm: PatchHOGOneClassSVMConfig
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


def _fixed_string(
    mapping: dict[str, Any],
    key: str,
    label: str,
    expected: str,
) -> str:
    value = _string(mapping, key, label)
    if value != expected:
        raise ConfigurationError(f"{label}.{key} must remain {expected}")
    return value


def _fixed_integer(
    mapping: dict[str, Any],
    key: str,
    label: str,
    expected: int,
) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        raise ConfigurationError(f"{label}.{key} must remain {expected}")
    return value


def _fixed_float(
    mapping: dict[str, Any],
    key: str,
    label: str,
    expected: float,
) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) != expected:
        raise ConfigurationError(f"{label}.{key} must remain {expected}")
    return float(value)


def _fixed_boolean(
    mapping: dict[str, Any],
    key: str,
    label: str,
    expected: bool,
) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool) or value is not expected:
        raise ConfigurationError(f"{label}.{key} must remain {expected}")
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
    ecc_registration_raw = _mapping(root.get("ecc_registration"), "ecc_registration")
    ecc_template_raw = _mapping(root.get("ecc_template"), "ecc_template")
    ecc_residual_scoring_raw = _mapping(
        root.get("ecc_residual_scoring"),
        "ecc_residual_scoring",
    )
    patch_hog_raw = _mapping(root.get("patch_hog"), "patch_hog")
    patch_hog_scaler_raw = _mapping(
        root.get("patch_hog_scaler"),
        "patch_hog_scaler",
    )
    patch_hog_one_class_svm_raw = _mapping(
        root.get("patch_hog_one_class_svm"),
        "patch_hog_one_class_svm",
    )
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

    decode_mode = _fixed_string(
        preprocessing_raw,
        "decode_mode",
        "preprocessing",
        "grayscale_uint8_ignore_orientation",
    )
    output_height = _fixed_integer(
        preprocessing_raw,
        "output_height",
        "preprocessing",
        512,
    )
    output_width = _fixed_integer(
        preprocessing_raw,
        "output_width",
        "preprocessing",
        512,
    )
    resize_interpolation = _fixed_string(
        preprocessing_raw,
        "resize_interpolation",
        "preprocessing",
        "area",
    )
    output_dtype = _fixed_string(
        preprocessing_raw,
        "output_dtype",
        "preprocessing",
        "float32",
    )
    scale_divisor = _fixed_float(
        preprocessing_raw,
        "scale_divisor",
        "preprocessing",
        255.0,
    )

    ecc_registration = ECCRegistrationConfig(
        motion_model=_fixed_string(
            ecc_registration_raw,
            "motion_model",
            "ecc_registration",
            "euclidean",
        ),
        initial_warp=_fixed_string(
            ecc_registration_raw,
            "initial_warp",
            "ecc_registration",
            "identity_2x3",
        ),
        termination=_fixed_string(
            ecc_registration_raw,
            "termination",
            "ecc_registration",
            "count_plus_epsilon",
        ),
        max_iterations=_fixed_integer(
            ecc_registration_raw,
            "max_iterations",
            "ecc_registration",
            100,
        ),
        epsilon=_fixed_float(
            ecc_registration_raw,
            "epsilon",
            "ecc_registration",
            1e-6,
        ),
        gaussian_filter_size=_fixed_integer(
            ecc_registration_raw,
            "gaussian_filter_size",
            "ecc_registration",
            5,
        ),
        warp_interpolation=_fixed_string(
            ecc_registration_raw,
            "warp_interpolation",
            "ecc_registration",
            "linear",
        ),
        mask_interpolation=_fixed_string(
            ecc_registration_raw,
            "mask_interpolation",
            "ecc_registration",
            "nearest",
        ),
        warp_border=_fixed_string(
            ecc_registration_raw,
            "warp_border",
            "ecc_registration",
            "constant_zero",
        ),
        max_abs_rotation_degrees=_fixed_float(
            ecc_registration_raw,
            "max_abs_rotation_degrees",
            "ecc_registration",
            10.0,
        ),
        max_abs_horizontal_translation_pixels=_fixed_float(
            ecc_registration_raw,
            "max_abs_horizontal_translation_pixels",
            "ecc_registration",
            64.0,
        ),
        max_abs_vertical_translation_pixels=_fixed_float(
            ecc_registration_raw,
            "max_abs_vertical_translation_pixels",
            "ecc_registration",
            64.0,
        ),
        min_valid_fraction=_fixed_float(
            ecc_registration_raw,
            "min_valid_fraction",
            "ecc_registration",
            0.80,
        ),
    )
    ecc_template = ECCTemplateConfig(
        anchor_selection=_fixed_string(
            ecc_template_raw,
            "anchor_selection",
            "ecc_template",
            "first_unicode_path",
        ),
        minimum_successful_references=_fixed_integer(
            ecc_template_raw,
            "minimum_successful_references",
            "ecc_template",
            16,
        ),
        support_mask=_fixed_string(
            ecc_template_raw,
            "support_mask",
            "ecc_template",
            "intersection",
        ),
        support_erosion_kernel_size=_fixed_integer(
            ecc_template_raw,
            "support_erosion_kernel_size",
            "ecc_template",
            5,
        ),
        support_erosion_iterations=_fixed_integer(
            ecc_template_raw,
            "support_erosion_iterations",
            "ecc_template",
            1,
        ),
        support_erosion_border=_fixed_string(
            ecc_template_raw,
            "support_erosion_border",
            "ecc_template",
            "constant_zero",
        ),
        minimum_support_fraction=_fixed_float(
            ecc_template_raw,
            "minimum_support_fraction",
            "ecc_template",
            0.75,
        ),
        aggregation=_fixed_string(
            ecc_template_raw,
            "aggregation",
            "ecc_template",
            "pixelwise_median_valid_values",
        ),
    )
    ecc_residual_scoring = ECCResidualScoringConfig(
        validity_erosion_kernel_size=_fixed_integer(
            ecc_residual_scoring_raw,
            "validity_erosion_kernel_size",
            "ecc_residual_scoring",
            5,
        ),
        validity_erosion_iterations=_fixed_integer(
            ecc_residual_scoring_raw,
            "validity_erosion_iterations",
            "ecc_residual_scoring",
            1,
        ),
        validity_erosion_border=_fixed_string(
            ecc_residual_scoring_raw,
            "validity_erosion_border",
            "ecc_residual_scoring",
            "constant_zero",
        ),
        minimum_effective_support_fraction=_fixed_float(
            ecc_residual_scoring_raw,
            "minimum_effective_support_fraction",
            "ecc_residual_scoring",
            0.95,
        ),
        residual=_fixed_string(
            ecc_residual_scoring_raw,
            "residual",
            "ecc_residual_scoring",
            "absolute_grayscale",
        ),
        gaussian_kernel_size=_fixed_integer(
            ecc_residual_scoring_raw,
            "gaussian_kernel_size",
            "ecc_residual_scoring",
            5,
        ),
        gaussian_sigma=_fixed_float(
            ecc_residual_scoring_raw,
            "gaussian_sigma",
            "ecc_residual_scoring",
            0.0,
        ),
        gaussian_border=_fixed_string(
            ecc_residual_scoring_raw,
            "gaussian_border",
            "ecc_residual_scoring",
            "constant_zero",
        ),
        top_fraction=_fixed_float(
            ecc_residual_scoring_raw,
            "top_fraction",
            "ecc_residual_scoring",
            0.01,
        ),
        top_count_rounding=_fixed_string(
            ecc_residual_scoring_raw,
            "top_count_rounding",
            "ecc_residual_scoring",
            "ceil",
        ),
        failure_score=_fixed_float(
            ecc_residual_scoring_raw,
            "failure_score",
            "ecc_residual_scoring",
            1.0,
        ),
    )
    patch_hog = PatchHOGConfig(
        patch_height=_fixed_integer(
            patch_hog_raw,
            "patch_height",
            "patch_hog",
            64,
        ),
        patch_width=_fixed_integer(
            patch_hog_raw,
            "patch_width",
            "patch_hog",
            64,
        ),
        vertical_stride=_fixed_integer(
            patch_hog_raw,
            "vertical_stride",
            "patch_hog",
            32,
        ),
        horizontal_stride=_fixed_integer(
            patch_hog_raw,
            "horizontal_stride",
            "patch_hog",
            32,
        ),
        vertical_positions=_fixed_integer(
            patch_hog_raw,
            "vertical_positions",
            "patch_hog",
            15,
        ),
        horizontal_positions=_fixed_integer(
            patch_hog_raw,
            "horizontal_positions",
            "patch_hog",
            15,
        ),
        patch_count=_fixed_integer(
            patch_hog_raw,
            "patch_count",
            "patch_hog",
            225,
        ),
        ordering=_fixed_string(
            patch_hog_raw,
            "ordering",
            "patch_hog",
            "row_major",
        ),
        orientations=_fixed_integer(
            patch_hog_raw,
            "orientations",
            "patch_hog",
            9,
        ),
        pixels_per_cell_height=_fixed_integer(
            patch_hog_raw,
            "pixels_per_cell_height",
            "patch_hog",
            16,
        ),
        pixels_per_cell_width=_fixed_integer(
            patch_hog_raw,
            "pixels_per_cell_width",
            "patch_hog",
            16,
        ),
        cells_per_block_height=_fixed_integer(
            patch_hog_raw,
            "cells_per_block_height",
            "patch_hog",
            2,
        ),
        cells_per_block_width=_fixed_integer(
            patch_hog_raw,
            "cells_per_block_width",
            "patch_hog",
            2,
        ),
        block_norm=_fixed_string(
            patch_hog_raw,
            "block_norm",
            "patch_hog",
            "L2-Hys",
        ),
        transform_sqrt=_fixed_boolean(
            patch_hog_raw,
            "transform_sqrt",
            "patch_hog",
            True,
        ),
        visualize=_fixed_boolean(
            patch_hog_raw,
            "visualize",
            "patch_hog",
            False,
        ),
        feature_vector=_fixed_boolean(
            patch_hog_raw,
            "feature_vector",
            "patch_hog",
            True,
        ),
        channel_axis=_fixed_string(
            patch_hog_raw,
            "channel_axis",
            "patch_hog",
            "none",
        ),
        descriptor_length=_fixed_integer(
            patch_hog_raw,
            "descriptor_length",
            "patch_hog",
            324,
        ),
        output_dtype=_fixed_string(
            patch_hog_raw,
            "output_dtype",
            "patch_hog",
            "float32",
        ),
    )
    patch_hog_scaler = PatchHOGScalerConfig(
        reference_order=_fixed_string(
            patch_hog_scaler_raw,
            "reference_order",
            "patch_hog_scaler",
            "unicode_path",
        ),
        fitting_scope=_fixed_string(
            patch_hog_scaler_raw,
            "fitting_scope",
            "patch_hog_scaler",
            "per_patch_position",
        ),
        copy=_fixed_boolean(
            patch_hog_scaler_raw,
            "copy",
            "patch_hog_scaler",
            True,
        ),
        with_mean=_fixed_boolean(
            patch_hog_scaler_raw,
            "with_mean",
            "patch_hog_scaler",
            True,
        ),
        with_std=_fixed_boolean(
            patch_hog_scaler_raw,
            "with_std",
            "patch_hog_scaler",
            True,
        ),
    )
    patch_hog_one_class_svm = PatchHOGOneClassSVMConfig(
        fitting_scope=_fixed_string(
            patch_hog_one_class_svm_raw,
            "fitting_scope",
            "patch_hog_one_class_svm",
            "per_patch_position",
        ),
        kernel=_fixed_string(
            patch_hog_one_class_svm_raw,
            "kernel",
            "patch_hog_one_class_svm",
            "rbf",
        ),
        gamma=_fixed_string(
            patch_hog_one_class_svm_raw,
            "gamma",
            "patch_hog_one_class_svm",
            "scale",
        ),
        nu=_fixed_float(
            patch_hog_one_class_svm_raw,
            "nu",
            "patch_hog_one_class_svm",
            0.05,
        ),
        tolerance=_fixed_float(
            patch_hog_one_class_svm_raw,
            "tolerance",
            "patch_hog_one_class_svm",
            0.001,
        ),
        shrinking=_fixed_boolean(
            patch_hog_one_class_svm_raw,
            "shrinking",
            "patch_hog_one_class_svm",
            True,
        ),
        cache_size_mb=_fixed_float(
            patch_hog_one_class_svm_raw,
            "cache_size_mb",
            "patch_hog_one_class_svm",
            200.0,
        ),
        max_iterations=_fixed_integer(
            patch_hog_one_class_svm_raw,
            "max_iterations",
            "patch_hog_one_class_svm",
            -1,
        ),
        verbose=_fixed_boolean(
            patch_hog_one_class_svm_raw,
            "verbose",
            "patch_hog_one_class_svm",
            False,
        ),
    )

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
        ecc_registration=ecc_registration,
        ecc_template=ecc_template,
        ecc_residual_scoring=ecc_residual_scoring,
        patch_hog=patch_hog,
        patch_hog_scaler=patch_hog_scaler,
        patch_hog_one_class_svm=patch_hog_one_class_svm,
        paths=ProjectPaths(**path_values),
        project_root=project_root,
    )
