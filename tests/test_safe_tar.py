from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from few_shot_anomaly_poc.errors import ChecksumMismatchError, UnsafeArchiveError
from few_shot_anomaly_poc.safe_tar import extract_archive_safely


def _tar_with_member(
    archive_path: Path,
    *,
    name: str,
    content: bytes = b"content",
    member_type: bytes = tarfile.REGTYPE,
) -> None:
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo(name)
        member.type = member_type
        member.size = len(content) if member_type == tarfile.REGTYPE else 0
        if member_type in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
            member.linkname = "target"
        archive.addfile(member, io.BytesIO(content) if member.size else None)


def _tar_with_regular_files(archive_path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(archive_path, "w") as archive:
        for name, content in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def _provenance(archive_path: Path, provenance_path: Path, digest: str | None = None) -> None:
    provenance_path.write_text(
        json.dumps(
            {
                "sha256": digest or hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def test_safe_extraction_accepts_regular_files_without_decoding(tmp_path: Path) -> None:
    archive = tmp_path / "data.tar"
    _tar_with_member(archive, name="VisA/pcb1/sample.txt", content=b"opaque bytes")
    provenance = tmp_path / "archive.json"
    _provenance(archive, provenance)
    destination = tmp_path / "external/visa"
    extraction_record = tmp_path / "extraction.json"

    summary = extract_archive_safely(
        archive_path=archive,
        archive_provenance_path=provenance,
        destination=destination,
        extraction_provenance_path=extraction_record,
        project_root=tmp_path,
    )

    assert (destination / "VisA/pcb1/sample.txt").read_bytes() == b"opaque bytes"
    assert summary.file_count == 1
    assert json.loads(extraction_record.read_text(encoding="utf-8"))["content_decoding"] is False


def test_safe_extraction_selects_only_requested_category(tmp_path: Path) -> None:
    archive = tmp_path / "data.tar"
    _tar_with_regular_files(
        archive,
        {
            "pcb1/Data/Images/Normal/0001.JPG": b"selected",
            "candle/Data/Images/Normal/0001.JPG": b"not selected",
        },
    )
    provenance = tmp_path / "archive.json"
    _provenance(archive, provenance)
    destination = tmp_path / "external/visa"
    extraction_record = tmp_path / "extraction.json"

    summary = extract_archive_safely(
        archive_path=archive,
        archive_provenance_path=provenance,
        destination=destination,
        extraction_provenance_path=extraction_record,
        project_root=tmp_path,
        member_prefix="pcb1",
    )

    assert (destination / "pcb1/Data/Images/Normal/0001.JPG").read_bytes() == b"selected"
    assert not (destination / "candle").exists()
    assert summary.file_count == 1
    record = json.loads(extraction_record.read_text(encoding="utf-8"))
    assert record["member_prefix"] == "pcb1"
    assert record["all_archive_members_validated"] is True


def test_safe_extraction_requires_requested_category(tmp_path: Path) -> None:
    archive = tmp_path / "data.tar"
    _tar_with_member(archive, name="candle/sample.txt")
    provenance = tmp_path / "archive.json"
    _provenance(archive, provenance)

    with pytest.raises(UnsafeArchiveError, match="no members"):
        extract_archive_safely(
            archive_path=archive,
            archive_provenance_path=provenance,
            destination=tmp_path / "external/visa",
            extraction_provenance_path=tmp_path / "extraction.json",
            project_root=tmp_path,
            member_prefix="pcb1",
        )


@pytest.mark.parametrize(
    ("name", "member_type"),
    [
        ("../../escape.txt", tarfile.REGTYPE),
        ("/absolute.txt", tarfile.REGTYPE),
        ("VisA/link", tarfile.SYMTYPE),
        ("VisA/hard-link", tarfile.LNKTYPE),
    ],
)
def test_safe_extraction_rejects_unsafe_members(
    tmp_path: Path,
    name: str,
    member_type: bytes,
) -> None:
    archive = tmp_path / "data.tar"
    _tar_with_member(archive, name=name, member_type=member_type)
    provenance = tmp_path / "archive.json"
    _provenance(archive, provenance)
    destination = tmp_path / "external/visa"

    with pytest.raises(UnsafeArchiveError):
        extract_archive_safely(
            archive_path=archive,
            archive_provenance_path=provenance,
            destination=destination,
            extraction_provenance_path=tmp_path / "extraction.json",
            project_root=tmp_path,
        )

    assert not destination.exists()
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extraction_requires_matching_provenance(tmp_path: Path) -> None:
    archive = tmp_path / "data.tar"
    _tar_with_member(archive, name="VisA/file.txt")
    provenance = tmp_path / "archive.json"
    _provenance(archive, provenance, digest="0" * 64)

    with pytest.raises(ChecksumMismatchError, match="provenance"):
        extract_archive_safely(
            archive_path=archive,
            archive_provenance_path=provenance,
            destination=tmp_path / "external/visa",
            extraction_provenance_path=tmp_path / "extraction.json",
            project_root=tmp_path,
        )
