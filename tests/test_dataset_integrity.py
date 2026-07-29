from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from few_shot_anomaly_poc.dataset_integrity import (
    DatasetIntegrityError,
    verify_visa_pcb1_integrity,
)


def _write_split(path: Path) -> str:
    rows = [
        {
            "object": "pcb1",
            "split": "train",
            "label": "normal",
            "image": "pcb1/Data/Images/Normal/0001.JPG",
            "mask": "",
        },
        {
            "object": "pcb1",
            "split": "test",
            "label": "normal",
            "image": "pcb1/Data/Images/Normal/0002.JPG",
            "mask": "",
        },
        {
            "object": "pcb1",
            "split": "test",
            "label": "anomaly",
            "image": "pcb1/Data/Images/Anomaly/0003.JPG",
            "mask": "pcb1/Data/Masks/Anomaly/0003.png",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["object", "split", "label", "image", "mask"],
            lineterminator="\r\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tar(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, mode="w") as archive:
        for relative_path, content in sorted(files.items()):
            member = tarfile.TarInfo(relative_path)
            member.size = len(content)
            member.mtime = 0
            archive.addfile(member, io.BytesIO(content))


def _fixture(tmp_path: Path) -> dict[str, Path]:
    selected_files = {
        "pcb1/Data/Images/Normal/0001.JPG": b"normal-reference",
        "pcb1/Data/Images/Normal/0002.JPG": b"normal-test",
        "pcb1/Data/Images/Anomaly/0003.JPG": b"anomaly-test",
        "pcb1/Data/Masks/Anomaly/0003.png": b"mask",
        "pcb1/image_anno.csv": b"annotation",
    }
    archive_files = {
        **selected_files,
        "pcb2/Data/Images/Normal/0001.JPG": b"other-category",
    }
    archive_path = tmp_path / "VisA.tar"
    _write_tar(archive_path, archive_files)

    dataset_root = tmp_path / "extracted"
    for relative_path, content in selected_files.items():
        target = dataset_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    split_csv = tmp_path / "split.csv"
    split_sha256 = _write_split(split_csv)
    record = {
        "schema_version": 1,
        "dataset": {
            "name": "Visual Anomaly (VisA)",
            "category": "pcb1",
            "license": "CC BY 4.0",
        },
        "archive": {
            "identifier": "VisA.tar",
            "checksum_status": "observed_only",
            "content_length_bytes": archive_path.stat().st_size,
            "observed_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "tar_entry_count": len(archive_files),
        },
        "official_split": {
            "revision": "a" * 40,
            "path": "split.csv",
            "sha256": split_sha256,
        },
        "pcb1_structure": {
            "normal_images": 2,
            "anomaly_images": 1,
            "anomaly_masks": 1,
            "official_train_normal": 1,
            "official_test_normal": 1,
            "official_test_anomaly": 1,
            "image_annotation_sha256": hashlib.sha256(b"annotation").hexdigest(),
        },
    }
    record_path = tmp_path / "dataset-record.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    return {
        "archive": archive_path,
        "dataset_root": dataset_root,
        "split": split_csv,
        "record": record_path,
    }


def _verify(paths: dict[str, Path]) -> dict:
    return verify_visa_pcb1_integrity(
        archive_path=paths["archive"],
        dataset_root=paths["dataset_root"],
        split_csv=paths["split"],
        dataset_record_path=paths["record"],
    )


def test_integrity_record_is_aggregate_and_deterministic(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    first = _verify(paths)
    second = _verify(paths)

    assert first == second
    assert first["status"] == "PASS"
    assert first["archive"]["matches_fixed_prior_observation"] is True
    assert first["archive_extraction_comparison"]["byte_for_byte_files_compared"] == 5
    assert first["pcb1"]["missing_split_images"] == 0
    assert first["pcb1"]["extra_masks_outside_split"] == 0
    assert first["evaluation_boundary"] == {
        "image_content_decoded": False,
        "image_content_displayed": False,
        "anomaly_score_computed": False,
        "per_path_final_test_label_exported": False,
        "final_test_label_join_performed": False,
        "metric_computed": False,
        "threshold_changed": False,
    }
    serialized = json.dumps(first)
    assert "0001.JPG" not in serialized
    assert str(tmp_path) not in serialized


def test_integrity_rejects_extracted_content_change(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    target = paths["dataset_root"] / "pcb1/Data/Images/Normal/0001.JPG"
    target.write_bytes(b"changed-content")

    with pytest.raises(DatasetIntegrityError, match=r"size mismatch|content mismatch"):
        _verify(paths)


def test_integrity_rejects_missing_and_extra_paths(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    missing = paths["dataset_root"] / "pcb1/Data/Images/Normal/0001.JPG"
    missing.unlink()
    extra = paths["dataset_root"] / "pcb1/Data/Images/Normal/extra.JPG"
    shutil.copyfile(
        paths["dataset_root"] / "pcb1/Data/Images/Normal/0002.JPG",
        extra,
    )

    with pytest.raises(DatasetIntegrityError, match="path mismatch"):
        _verify(paths)


def test_integrity_rejects_archive_identity_change(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    with paths["archive"].open("ab") as stream:
        stream.write(b"changed")

    with pytest.raises(DatasetIntegrityError, match=r"byte count|SHA-256"):
        _verify(paths)


def test_integrity_rejects_split_identity_change(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    with paths["split"].open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(DatasetIntegrityError, match="split SHA-256"):
        _verify(paths)
