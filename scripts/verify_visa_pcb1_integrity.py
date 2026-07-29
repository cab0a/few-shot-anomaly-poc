"""Verify the fixed local VisA pcb1 asset and write one aggregate record."""

from __future__ import annotations

import argparse
from pathlib import Path

from few_shot_anomaly_poc.config import load_config
from few_shot_anomaly_poc.dataset_integrity import (
    DatasetIntegrityError,
    verify_visa_pcb1_integrity,
)
from few_shot_anomaly_poc.jsonio import write_json_atomic

DEFAULT_CONFIG = Path("configs/v0.1.yaml")
DEFAULT_DATASET_RECORD = Path("artifacts/v0.1/data/dataset-record.json")
DEFAULT_OUTPUT = Path("artifacts/v0.1/data/pcb1-local-integrity.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify local VisA pcb1 bytes and paths without decoding images, "
            "revealing per-path labels, or computing scores."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split-csv", type=Path)
    parser.add_argument("--dataset-record", type=Path, default=DEFAULT_DATASET_RECORD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_config(args.config)
    try:
        result = verify_visa_pcb1_integrity(
            archive_path=args.archive or config.paths.archive,
            dataset_root=args.dataset_root or config.paths.extracted,
            split_csv=args.split_csv or config.paths.split_csv,
            dataset_record_path=args.dataset_record,
            category=config.category,
        )
        write_json_atomic(args.output, result)
    except (DatasetIntegrityError, FileExistsError, OSError) as error:
        print(f"error: {error}")
        return 1
    print(
        "local integrity verification passed: "
        f"category={result['dataset']['category']}, "
        f"files={result['archive_extraction_comparison']['extracted_file_count']}, "
        f"output={args.output.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
