from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from few_shot_anomaly_poc.download import download_file
from few_shot_anomaly_poc.errors import ChecksumMismatchError


def test_download_records_verified_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"known asset")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "downloads/asset.bin"
    provenance = tmp_path / "provenance/asset.json"

    record = download_file(
        url=source.as_uri(),
        destination=destination,
        provenance_path=provenance,
        expected_sha256=expected,
        provenance_fields={"asset_type": "test"},
    )

    assert destination.read_bytes() == b"known asset"
    assert record["checksum_status"] == "verified"
    assert record["sha256"] == expected
    assert provenance.is_file()


def test_download_marks_checksum_as_observed_when_not_pinned(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"first observation")

    record = download_file(
        url=source.as_uri(),
        destination=tmp_path / "asset.bin",
        provenance_path=tmp_path / "asset.json",
        expected_sha256=None,
        provenance_fields={"asset_type": "test"},
    )

    assert record["checksum_status"] == "observed_only"
    assert record["expected_sha256"] is None


def test_checksum_mismatch_leaves_no_download_or_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"unexpected")
    destination = tmp_path / "asset.bin"
    provenance = tmp_path / "asset.json"

    with pytest.raises(ChecksumMismatchError):
        download_file(
            url=source.as_uri(),
            destination=destination,
            provenance_path=provenance,
            expected_sha256="0" * 64,
            provenance_fields={"asset_type": "test"},
        )

    assert not destination.exists()
    assert not provenance.exists()
    assert not list(tmp_path.glob("*.part"))


def test_download_refuses_to_overwrite_existing_file(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    destination = tmp_path / "asset.bin"
    destination.write_bytes(b"owned")

    with pytest.raises(FileExistsError, match="overwrite"):
        download_file(
            url=source.as_uri(),
            destination=destination,
            provenance_path=tmp_path / "asset.json",
            expected_sha256=None,
            provenance_fields={},
        )

    assert destination.read_bytes() == b"owned"
