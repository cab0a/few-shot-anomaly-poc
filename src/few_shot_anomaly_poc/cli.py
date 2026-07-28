"""Command-line entry point for the metadata-only data preparation stage."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

from few_shot_anomaly_poc.config import load_config
from few_shot_anomaly_poc.download import download_archive, fetch_official_split
from few_shot_anomaly_poc.errors import DataPreparationError
from few_shot_anomaly_poc.manifests import (
    ManifestSummary,
    build_manifests,
    validate_manifests,
)
from few_shot_anomaly_poc.safe_tar import extract_archive_safely

DEFAULT_CONFIG = Path("configs/v0.1.yaml")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="versioned JSON-compatible YAML configuration",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="few-shot-data",
        description="Prepare VisA metadata without decoding dataset images.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fetch_parser = commands.add_parser("fetch-split", help="fetch the pinned official split")
    _add_config_argument(fetch_parser)

    download_parser = commands.add_parser(
        "download-archive",
        help="download the official archive and record its observed checksum",
    )
    _add_config_argument(download_parser)
    download_parser.add_argument(
        "--expected-sha256",
        help="trusted archive SHA-256, when independently available",
    )

    extract_parser = commands.add_parser(
        "extract-archive",
        help="verify provenance and safely extract regular files and directories",
    )
    _add_config_argument(extract_parser)

    build_parser = commands.add_parser(
        "build-manifests",
        help="create reference, calibration, and final-test metadata manifests",
    )
    _add_config_argument(build_parser)

    validate_parser = commands.add_parser(
        "validate-manifests",
        help="validate manifest text, hashes, counts, and disjointness",
    )
    _add_config_argument(validate_parser)
    return parser


def _summary_text(summary: ManifestSummary) -> str:
    return (
        f"reference={summary.reference_count}, calibration={summary.calibration_count}, "
        f"final-test={summary.final_test_count}, "
        f"archive_sha256_recorded={summary.archive_sha256_recorded}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one bounded data-preparation command."""
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "fetch-split":
            provenance = fetch_official_split(config)
            print(f"verified split: sha256={provenance['sha256']}")
        elif args.command == "download-archive":
            expected = args.expected_sha256
            if expected is not None and not SHA256_PATTERN.fullmatch(expected):
                raise DataPreparationError(
                    "--expected-sha256 must be a lowercase 64-character digest"
                )
            provenance = download_archive(
                config,
                expected_sha256_override=expected,
            )
            print(
                "downloaded archive: "
                f"sha256={provenance['sha256']}, "
                f"status={provenance['checksum_status']}"
            )
        elif args.command == "extract-archive":
            summary = extract_archive_safely(
                archive_path=config.paths.archive,
                archive_provenance_path=config.paths.archive_provenance,
                destination=config.paths.extracted,
                extraction_provenance_path=config.paths.extraction_provenance,
                project_root=config.project_root,
                member_prefix=config.category,
            )
            print(
                f"safe extraction complete: members={summary.member_count}, "
                f"files={summary.file_count}"
            )
        elif args.command == "build-manifests":
            print(f"manifest build complete: {_summary_text(build_manifests(config))}")
        elif args.command == "validate-manifests":
            print(f"manifest validation passed: {_summary_text(validate_manifests(config))}")
        else:  # pragma: no cover - argparse enforces the command set
            raise AssertionError(f"unexpected command: {args.command}")
    except (DataPreparationError, FileExistsError, OSError) as error:
        print(f"error: {error}")
        return 1
    return 0
