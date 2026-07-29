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
    assert config.preprocessing.decode_mode == "grayscale_uint8_ignore_orientation"
    assert config.preprocessing.output_height == 512
    assert config.preprocessing.output_width == 512
    assert config.preprocessing.resize_interpolation == "area"
    assert config.preprocessing.output_dtype == "float32"
    assert config.preprocessing.scale_divisor == 255.0
    assert config.ecc_registration.motion_model == "euclidean"
    assert config.ecc_registration.initial_warp == "identity_2x3"
    assert config.ecc_registration.max_iterations == 100
    assert config.ecc_registration.epsilon == 1e-6
    assert config.ecc_registration.gaussian_filter_size == 5
    assert config.ecc_registration.max_abs_rotation_degrees == 10.0
    assert config.ecc_registration.max_abs_horizontal_translation_pixels == 64.0
    assert config.ecc_registration.max_abs_vertical_translation_pixels == 64.0
    assert config.ecc_registration.min_valid_fraction == 0.80
    assert config.ecc_template.anchor_selection == "first_unicode_path"
    assert config.ecc_template.minimum_successful_references == 16
    assert config.ecc_template.support_mask == "intersection"
    assert config.ecc_template.support_erosion_kernel_size == 5
    assert config.ecc_template.support_erosion_iterations == 1
    assert config.ecc_template.support_erosion_border == "constant_zero"
    assert config.ecc_template.minimum_support_fraction == 0.75
    assert config.ecc_template.aggregation == "pixelwise_median_valid_values"
    assert config.ecc_residual_scoring.validity_erosion_kernel_size == 5
    assert config.ecc_residual_scoring.validity_erosion_iterations == 1
    assert config.ecc_residual_scoring.validity_erosion_border == "constant_zero"
    assert config.ecc_residual_scoring.minimum_effective_support_fraction == 0.95
    assert config.ecc_residual_scoring.residual == "absolute_grayscale"
    assert config.ecc_residual_scoring.gaussian_kernel_size == 5
    assert config.ecc_residual_scoring.gaussian_sigma == 0.0
    assert config.ecc_residual_scoring.gaussian_border == "constant_zero"
    assert config.ecc_residual_scoring.top_fraction == 0.01
    assert config.ecc_residual_scoring.top_count_rounding == "ceil"
    assert config.ecc_residual_scoring.failure_score == 1.0
    assert config.patch_hog.patch_height == 64
    assert config.patch_hog.patch_width == 64
    assert config.patch_hog.vertical_stride == 32
    assert config.patch_hog.horizontal_stride == 32
    assert config.patch_hog.vertical_positions == 15
    assert config.patch_hog.horizontal_positions == 15
    assert config.patch_hog.patch_count == 225
    assert config.patch_hog.ordering == "row_major"
    assert config.patch_hog.orientations == 9
    assert config.patch_hog.pixels_per_cell_height == 16
    assert config.patch_hog.pixels_per_cell_width == 16
    assert config.patch_hog.cells_per_block_height == 2
    assert config.patch_hog.cells_per_block_width == 2
    assert config.patch_hog.block_norm == "L2-Hys"
    assert config.patch_hog.transform_sqrt is True
    assert config.patch_hog.visualize is False
    assert config.patch_hog.feature_vector is True
    assert config.patch_hog.channel_axis == "none"
    assert config.patch_hog.descriptor_length == 324
    assert config.patch_hog.output_dtype == "float32"
    assert config.patch_hog_scaler.reference_order == "unicode_path"
    assert config.patch_hog_scaler.fitting_scope == "per_patch_position"
    assert config.patch_hog_scaler.copy is True
    assert config.patch_hog_scaler.with_mean is True
    assert config.patch_hog_scaler.with_std is True
    assert config.patch_hog_one_class_svm.fitting_scope == "per_patch_position"
    assert config.patch_hog_one_class_svm.kernel == "rbf"
    assert config.patch_hog_one_class_svm.gamma == "scale"
    assert config.patch_hog_one_class_svm.nu == 0.05
    assert config.patch_hog_one_class_svm.tolerance == 0.001
    assert config.patch_hog_one_class_svm.shrinking is True
    assert config.patch_hog_one_class_svm.cache_size_mb == 200.0
    assert config.patch_hog_one_class_svm.max_iterations == -1
    assert config.patch_hog_one_class_svm.verbose is False
    assert config.patch_hog_scoring.patch_score == "negative_decision_function"
    assert config.patch_hog_scoring.maximum_absolute_patch_score_exclusive == 1e12
    assert config.patch_hog_scoring.aggregation == "mean_top_fraction"
    assert config.patch_hog_scoring.top_fraction == 0.05
    assert config.patch_hog_scoring.top_count_rounding == "ceil"
    assert config.patch_hog_scoring.failure_score == 1e12
    assert config.threshold_calibration.source_partition == "calibration"
    assert config.threshold_calibration.normal_only is True
    assert config.threshold_calibration.method_scope == "per_method"
    assert config.threshold_calibration.score_order == "ascending"
    assert config.threshold_calibration.quantile == 0.95
    assert config.threshold_calibration.rank_rounding == "ceil"
    assert config.threshold_calibration.threshold_index == "rank_minus_one"
    assert config.threshold_calibration.prediction_rule == "failed_or_strictly_greater"
    assert config.latency_measurement.boundary == "decoded_grayscale_uint8_to_image_score"
    assert config.latency_measurement.timer == "perf_counter_ns"
    assert config.latency_measurement.warmup_passes == 1
    assert config.latency_measurement.timed_passes == 3
    assert config.latency_measurement.path_order == "unicode_code_point_ascending"
    assert config.latency_measurement.summary_sample == "all_images_all_timed_passes"
    assert config.latency_measurement.median_rule == "mean_of_middle_pair_when_even"
    assert config.latency_measurement.p95_quantile == 0.95
    assert config.latency_measurement.p95_rule == "nearest_rank"
    assert config.latency_measurement.include_failed_score_timings is True
    assert config.latency_measurement.duration_unit == "nanoseconds"
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


@pytest.mark.parametrize("changed_passes", [2, 3.0, True])
def test_config_rejects_changed_latency_measurement(
    tmp_path: Path,
    changed_passes: object,
) -> None:
    raw = json.loads(Path("configs/v0.1.yaml").read_text(encoding="utf-8"))
    raw["latency_measurement"]["warmup_passes"] = changed_passes
    config_path = tmp_path / "configs/v0.1.yaml"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="warmup_passes"):
        load_config(config_path)


@pytest.mark.parametrize("changed_width", [256, 512.0, True])
def test_config_rejects_changed_preprocessing(
    tmp_path: Path,
    changed_width: object,
) -> None:
    raw = json.loads(Path("configs/v0.1.yaml").read_text(encoding="utf-8"))
    raw["preprocessing"]["output_width"] = changed_width
    config_path = tmp_path / "configs/v0.1.yaml"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="output_width"):
        load_config(config_path)


@pytest.mark.parametrize("changed_rotation", [11.0, "10.0", True])
def test_config_rejects_changed_ecc_registration(
    tmp_path: Path,
    changed_rotation: object,
) -> None:
    raw = json.loads(Path("configs/v0.1.yaml").read_text(encoding="utf-8"))
    raw["ecc_registration"]["max_abs_rotation_degrees"] = changed_rotation
    config_path = tmp_path / "configs/v0.1.yaml"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="max_abs_rotation_degrees"):
        load_config(config_path)


@pytest.mark.parametrize("changed_minimum", [15, 16.0, True])
def test_config_rejects_changed_ecc_template(
    tmp_path: Path,
    changed_minimum: object,
) -> None:
    raw = json.loads(Path("configs/v0.1.yaml").read_text(encoding="utf-8"))
    raw["ecc_template"]["minimum_successful_references"] = changed_minimum
    config_path = tmp_path / "configs/v0.1.yaml"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="minimum_successful_references"):
        load_config(config_path)


@pytest.mark.parametrize("changed_fraction", [0.94, "0.95", True])
def test_config_rejects_changed_ecc_residual_scoring(
    tmp_path: Path,
    changed_fraction: object,
) -> None:
    raw = json.loads(Path("configs/v0.1.yaml").read_text(encoding="utf-8"))
    raw["ecc_residual_scoring"]["minimum_effective_support_fraction"] = changed_fraction
    config_path = tmp_path / "configs/v0.1.yaml"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="minimum_effective_support_fraction"):
        load_config(config_path)


@pytest.mark.parametrize("changed_orientations", [8, 9.0, True])
def test_config_rejects_changed_patch_hog(
    tmp_path: Path,
    changed_orientations: object,
) -> None:
    raw = json.loads(Path("configs/v0.1.yaml").read_text(encoding="utf-8"))
    raw["patch_hog"]["orientations"] = changed_orientations
    config_path = tmp_path / "configs/v0.1.yaml"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="orientations"):
        load_config(config_path)


@pytest.mark.parametrize("changed_transform", [False, 1, "true"])
def test_config_rejects_changed_patch_hog_boolean(
    tmp_path: Path,
    changed_transform: object,
) -> None:
    raw = json.loads(Path("configs/v0.1.yaml").read_text(encoding="utf-8"))
    raw["patch_hog"]["transform_sqrt"] = changed_transform
    config_path = tmp_path / "configs/v0.1.yaml"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="transform_sqrt"):
        load_config(config_path)


@pytest.mark.parametrize("changed_scope", ["global", "", 1])
def test_config_rejects_changed_patch_hog_scaler(
    tmp_path: Path,
    changed_scope: object,
) -> None:
    raw = json.loads(Path("configs/v0.1.yaml").read_text(encoding="utf-8"))
    raw["patch_hog_scaler"]["fitting_scope"] = changed_scope
    config_path = tmp_path / "configs/v0.1.yaml"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="fitting_scope"):
        load_config(config_path)


@pytest.mark.parametrize("changed_nu", [0.1, "0.05", True])
def test_config_rejects_changed_patch_hog_one_class_svm(
    tmp_path: Path,
    changed_nu: object,
) -> None:
    raw = json.loads(Path("configs/v0.1.yaml").read_text(encoding="utf-8"))
    raw["patch_hog_one_class_svm"]["nu"] = changed_nu
    config_path = tmp_path / "configs/v0.1.yaml"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="nu"):
        load_config(config_path)


@pytest.mark.parametrize("changed_top_fraction", [0.1, "0.05", True])
def test_config_rejects_changed_patch_hog_scoring(
    tmp_path: Path,
    changed_top_fraction: object,
) -> None:
    raw = json.loads(Path("configs/v0.1.yaml").read_text(encoding="utf-8"))
    raw["patch_hog_scoring"]["top_fraction"] = changed_top_fraction
    config_path = tmp_path / "configs/v0.1.yaml"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="top_fraction"):
        load_config(config_path)


@pytest.mark.parametrize("changed_quantile", [0.9, "0.95", True])
def test_config_rejects_changed_threshold_calibration(
    tmp_path: Path,
    changed_quantile: object,
) -> None:
    raw = json.loads(Path("configs/v0.1.yaml").read_text(encoding="utf-8"))
    raw["threshold_calibration"]["quantile"] = changed_quantile
    config_path = tmp_path / "configs/v0.1.yaml"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="quantile"):
        load_config(config_path)
