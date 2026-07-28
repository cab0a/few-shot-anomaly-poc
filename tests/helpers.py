from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from few_shot_anomaly_poc.config import ProjectConfig, load_config


def write_split(path: Path, rows: list[dict[str, str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["object", "split", "label", "image", "mask"],
            lineterminator="\r\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normal_train(index: int) -> dict[str, str]:
    return {
        "object": "pcb1",
        "split": "train",
        "label": "normal",
        "image": f"pcb1/Data/Images/Normal/{index:04d}.JPG",
        "mask": "",
    }


def final_test_row(index: int, label: str) -> dict[str, str]:
    folder = "Normal" if label == "normal" else "Anomaly"
    return {
        "object": "pcb1",
        "split": "test",
        "label": label,
        "image": f"pcb1/Data/Images/{folder}/{index:04d}.JPG",
        "mask": "" if label == "normal" else f"pcb1/Data/Masks/Anomaly/{index:04d}.png",
    }


def create_config(
    project_root: Path,
    *,
    rows: list[dict[str, str]],
    reference_count: int = 2,
    write_archive_provenance: bool = True,
) -> ProjectConfig:
    split_path = project_root / "data/external/splits/visa-1cls.csv"
    split_sha256 = write_split(split_path, rows)
    raw: dict[str, Any] = {
        "schema_version": "v0.1",
        "dataset": {
            "name": "VisA",
            "category": "pcb1",
            "license": "CC BY 4.0",
            "archive": {
                "identifier": "VisA_20220922.tar",
                "url": "https://example.invalid/VisA_20220922.tar",
                "expected_sha256": None,
            },
            "split": {
                "repository": "https://github.com/amazon-science/spot-diff",
                "revision": "a" * 40,
                "path": "split_csv/1cls.csv",
                "url": "https://example.invalid/1cls.csv",
                "sha256": split_sha256,
            },
        },
        "selection": {
            "reference_count": reference_count,
            "seed": 42,
            "procedure_version": "sha256-path-ranking-v1",
            "namespace": "few-shot-anomaly-poc:v0.1",
        },
        "preprocessing": {
            "decode_mode": "grayscale_uint8_ignore_orientation",
            "output_height": 512,
            "output_width": 512,
            "resize_interpolation": "area",
            "output_dtype": "float32",
            "scale_divisor": 255.0,
        },
        "ecc_registration": {
            "motion_model": "euclidean",
            "initial_warp": "identity_2x3",
            "termination": "count_plus_epsilon",
            "max_iterations": 100,
            "epsilon": 0.000001,
            "gaussian_filter_size": 5,
            "warp_interpolation": "linear",
            "mask_interpolation": "nearest",
            "warp_border": "constant_zero",
            "max_abs_rotation_degrees": 10.0,
            "max_abs_horizontal_translation_pixels": 64.0,
            "max_abs_vertical_translation_pixels": 64.0,
            "min_valid_fraction": 0.80,
        },
        "paths": {
            "archive": "data/external/archives/VisA_20220922.tar",
            "archive_provenance": "data/provenance/visa-archive.json",
            "extracted": "data/external/visa",
            "extraction_provenance": "data/provenance/visa-extraction.json",
            "split_csv": "data/external/splits/visa-1cls.csv",
            "split_provenance": "data/provenance/visa-split.json",
            "manifest_dir": "data/manifests/v0.1",
        },
    }
    config_path = project_root / "configs/v0.1.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    if write_archive_provenance:
        provenance_path = project_root / "data/provenance/visa-archive.json"
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(
            json.dumps({"sha256": "b" * 64}),
            encoding="utf-8",
        )
    return load_config(config_path)
