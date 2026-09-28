from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("external_data", ROOT / "scripts/external_data.py")
assert SPEC is not None and SPEC.loader is not None
transfer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transfer)


@pytest.fixture
def example(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "source"
    roots = ["data/external/fixed-inputs", "work/fixed-inputs/fitted-state"]
    required = {}
    for index, relative in enumerate((f"{roots[0]}/scorer/asset.jpg", f"{roots[1]}/state.pkl")):
        path = root / relative
        path.parent.mkdir(parents=True)
        content = f"opaque synthetic bytes {index}".encode()
        path.write_bytes(content)
        required[relative] = hashlib.sha256(content).hexdigest()
    (root / ".git").mkdir()
    (root / ".git/config").write_text("must not enter backup")
    spec = {
        "roots": roots,
        "required": required,
        "archive_name": "inputs.zip",
        "receipt_name": "inputs.json",
        "spec_sha256": "a" * 64,
    }
    return root, spec


def test_round_trip_is_exact_scoped_and_repeatable(example, tmp_path: Path) -> None:
    source, spec = example
    store = tmp_path / "store"
    work = tmp_path / "pack"
    work.mkdir()
    first = transfer.publish(source, spec, store, work)
    second_work = tmp_path / "pack-again"
    second_work.mkdir()
    assert transfer.publish(source, spec, store, second_work) == first
    with zipfile.ZipFile(store / "inputs.zip") as bundle:
        assert set(bundle.namelist()) == {transfer.MANIFEST, *spec["required"]}
        assert all(".git" not in name for name in bundle.namelist())
    download = tmp_path / "download"
    download.mkdir()
    archive, receipt = transfer.fetch(spec, store, download)
    restored = tmp_path / "restored"
    assert transfer.restore_archive(restored, spec, archive, receipt, tmp_path / "stage") == 2
    assert transfer.restore_archive(restored, spec, archive, receipt, tmp_path / "stage-again") == 0
    for relative in spec["required"]:
        assert (restored / relative).read_bytes() == (source / relative).read_bytes()


@pytest.mark.parametrize("failure", ["missing", "changed"])
def test_incomplete_source_never_publishes_receipt(example, tmp_path: Path, failure: str) -> None:
    root, spec = example
    path = root / next(iter(spec["required"]))
    if failure == "missing":
        path.unlink()
    else:
        path.write_bytes(b"wrong source")
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(transfer.TransferError):
        transfer.publish(root, spec, tmp_path / "store", work)
    assert not (tmp_path / "store/inputs.json").exists()


def test_changed_sync_copy_fails_before_restore(example, tmp_path: Path) -> None:
    root, spec = example
    work = tmp_path / "work"
    work.mkdir()
    store = tmp_path / "store"
    transfer.publish(root, spec, store, work)
    path = store / "inputs.zip"
    damaged = bytearray(path.read_bytes())
    damaged[-1] ^= 1
    path.write_bytes(damaged)
    download = tmp_path / "download"
    download.mkdir()
    with pytest.raises(transfer.TransferError, match="hash differs"):
        transfer.fetch(spec, store, download)


def test_existing_input_conflict_prevents_all_installation(example, tmp_path: Path) -> None:
    source, spec = example
    archive = tmp_path / "inputs.zip"
    receipt = transfer.make_archive(source, spec, archive)
    destination = tmp_path / "destination"
    conflicting = destination / list(spec["required"])[-1]
    conflicting.parent.mkdir(parents=True)
    conflicting.write_bytes(b"preserve existing local work")
    with pytest.raises(transfer.TransferError, match="existing input differs"):
        transfer.restore_archive(destination, spec, archive, receipt, tmp_path / "stage")
    assert conflicting.read_bytes() == b"preserve existing local work"
    assert not (destination / next(iter(spec["required"]))).exists()


def test_parent_file_conflict_prevents_all_installation(example, tmp_path: Path) -> None:
    source, spec = example
    archive = tmp_path / "inputs.zip"
    receipt = transfer.make_archive(source, spec, archive)
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "work").write_bytes(b"keep existing file")
    with pytest.raises(transfer.TransferError, match="ancestor is not a directory"):
        transfer.restore_archive(destination, spec, archive, receipt, tmp_path / "stage")
    assert not (destination / "data").exists()
    assert (destination / "work").read_bytes() == b"keep existing file"


@pytest.mark.parametrize(
    "path",
    [
        "../escape",
        "/absolute",
        "data/external/fixed-inputs/../../escape",
        "scripts/executable.py",
        "data\\external\\escape",
    ],
)
def test_unsafe_archive_paths_are_rejected_before_writing(
    example, tmp_path: Path, path: str
) -> None:
    _, spec = example
    content = b"bad path"
    manifest = {
        "schema_version": "external-input-bundle-v1",
        "roots": spec["roots"],
        "files": [
            {
                "path": path,
                "byte_count": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        ],
    }
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(transfer.MANIFEST, json.dumps(manifest))
        bundle.writestr(path, content)
    receipt = {
        "archive_sha256": transfer.digest(archive),
        "file_count": 1,
        "unpacked_byte_count": len(content),
    }
    with pytest.raises(transfer.TransferError, match="unsafe or unauthorized"):
        transfer.restore_archive(
            tmp_path / "destination", spec, archive, receipt, tmp_path / "stage"
        )
    assert not (tmp_path / "destination").exists()
    assert not (tmp_path / "escape").exists()


def test_symlink_in_source_is_rejected(example, tmp_path: Path) -> None:
    source, spec = example
    private = tmp_path / "private.txt"
    private.write_text("not part of dataset")
    (source / spec["roots"][0] / "link").symlink_to(private)
    with pytest.raises(transfer.TransferError, match="symlink forbidden"):
        transfer.make_archive(source, spec, tmp_path / "inputs.zip")


def test_old_backup_is_never_overwritten(example, tmp_path: Path) -> None:
    source, spec = example
    store = tmp_path / "store"
    store.mkdir()
    (store / "inputs.zip").write_bytes(b"existing distinct archive")
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(transfer.TransferError, match="not overwritten"):
        transfer.publish(source, spec, store, work)
    assert (store / "inputs.zip").read_bytes() == b"existing distinct archive"
    assert not (store / "inputs.json").exists()


def test_repo_spec_pins_all_35_required_inputs_without_reading_images() -> None:
    spec = transfer.load_spec(ROOT)
    assert len(spec["required"]) == 35
    assert sum("/scorer/assets/" in path for path in spec["required"]) == 28
    assert sum("/fitted-state/" in path for path in spec["required"]) == 3
