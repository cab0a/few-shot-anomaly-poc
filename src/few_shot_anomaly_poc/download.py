"""Download pinned external assets and record their provenance."""

from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from few_shot_anomaly_poc.config import ProjectConfig
from few_shot_anomaly_poc.errors import ChecksumMismatchError
from few_shot_anomaly_poc.jsonio import write_json_atomic

DOWNLOAD_CHUNK_SIZE = 1024 * 1024
USER_AGENT = "few-shot-anomaly-poc/0.1 data-provenance"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def download_file(
    *,
    url: str,
    destination: Path,
    provenance_path: Path,
    expected_sha256: str | None,
    provenance_fields: dict[str, Any],
) -> dict[str, Any]:
    """Download once, optionally verify a trusted checksum, and write provenance."""
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    if provenance_path.exists():
        raise FileExistsError(f"refusing to overwrite {provenance_path}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    destination_committed = False
    digest = hashlib.sha256()
    byte_count = 0

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request) as response:
                while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                    output.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
                response_url = response.geturl()
                etag = response.headers.get("ETag")
                last_modified = response.headers.get("Last-Modified")

        observed_sha256 = digest.hexdigest()
        if expected_sha256 is not None and observed_sha256 != expected_sha256:
            raise ChecksumMismatchError(
                f"checksum mismatch for {destination.name}: "
                f"expected {expected_sha256}, observed {observed_sha256}"
            )

        os.replace(temporary_path, destination)
        temporary_path = None
        destination_committed = True
        provenance = {
            **provenance_fields,
            "requested_url": url,
            "effective_url": response_url,
            "downloaded_at": _utc_now(),
            "byte_count": byte_count,
            "sha256": observed_sha256,
            "expected_sha256": expected_sha256,
            "checksum_status": "verified" if expected_sha256 else "observed_only",
            "http_etag": etag,
            "http_last_modified": last_modified,
        }
        write_json_atomic(provenance_path, provenance)
        return provenance
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if destination_committed and not provenance_path.exists():
            destination.unlink(missing_ok=True)
        raise


def fetch_official_split(config: ProjectConfig) -> dict[str, Any]:
    """Fetch the pinned official one-class split and require its known checksum."""
    return download_file(
        url=config.split.url,
        destination=config.paths.split_csv,
        provenance_path=config.paths.split_provenance,
        expected_sha256=config.split.sha256,
        provenance_fields={
            "schema_version": 1,
            "asset_type": "official_split",
            "dataset": config.dataset_name,
            "dataset_license": config.dataset_license,
            "repository": config.split.repository,
            "revision": config.split.revision,
            "repository_path": config.split.path,
        },
    )


def download_archive(
    config: ProjectConfig,
    *,
    expected_sha256_override: str | None = None,
) -> dict[str, Any]:
    """Download the official archive without treating an observed hash as trusted."""
    expected_sha256 = expected_sha256_override or config.archive.expected_sha256
    return download_file(
        url=config.archive.url,
        destination=config.paths.archive,
        provenance_path=config.paths.archive_provenance,
        expected_sha256=expected_sha256,
        provenance_fields={
            "schema_version": 1,
            "asset_type": "dataset_archive",
            "dataset": config.dataset_name,
            "dataset_license": config.dataset_license,
            "archive_identifier": config.archive.identifier,
            "license_boundary": (
                "VisA remains under CC BY 4.0; the repository's PolyForm license "
                "does not apply to this archive."
            ),
        },
    )
