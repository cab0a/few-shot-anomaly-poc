"""Back up and restore fixed external inputs through a Google Drive sync folder.

Uses Python's standard library only; no uv, model runtime, or image decoder is needed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

MANIFEST = "_transfer-manifest.json"
MAX_BYTES = 20 * 1024**3


class TransferError(Exception):
    """Refuse incomplete, altered, unsafe, or conflicting input data."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TransferError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def regular(path: Path) -> bool:
    return path.is_file() and not any(part.is_symlink() for part in (path, *path.parents))


def safe_relative(value: str, roots: list[str]) -> str:
    require(isinstance(value, str), "archive path is not text")
    path = PurePosixPath(value)
    require(
        bool(re.fullmatch(r"[A-Za-z0-9._/-]+", value))
        and path.as_posix() == value
        and not path.is_absolute()
        and all(part not in {".", ".."} for part in path.parts)
        and any(value.startswith(root + "/") for root in roots),
        f"unsafe or unauthorized archive path: {value}",
    )
    return value


def load_spec(root: Path) -> dict:
    spec_path = root / "configs/external-data-store.json"
    spec = json.loads(spec_path.read_text("utf-8"))
    require(spec["schema_version"] == "external-data-store-v1", "unknown store schema")
    config_path = root / "configs/v0.3.yaml"
    inventory = root / "artifacts/v0.3/diagnostics/pcb2-development/review-assets.csv"
    require(digest(config_path) == spec["diagnostic_config_sha256"], "diagnostic config changed")
    require(digest(inventory) == spec["review_inventory_sha256"], "review inventory changed")
    config = json.loads(config_path.read_text("utf-8"))
    external, fitted = spec["roots"]
    required = {}
    for relative, key in (
        ("scorer/scoring-manifest.json", "opaque_scoring_manifest_sha256"),
        ("normal-manifests/manifest-set.json", "normal_manifest_set_sha256"),
        ("normal-manifests/reference.jsonl", "normal_reference_manifest_sha256"),
        ("normal-manifests/calibration.jsonl", "normal_calibration_manifest_sha256"),
    ):
        required[f"{external}/{relative}"] = config["parent_evidence"][key]
    for method, suffix in (
        ("ecc_residual", ".pkl"),
        ("patch_hog_ocsvm", ".pkl"),
        ("dinov2_vits14_224_nn", ".pt"),
    ):
        required[f"{fitted}/{method}{suffix}"] = config["methods"][method]["fitted_state_sha256"]
    with inventory.open(encoding="utf-8", newline="") as stream:
        for record in csv.DictReader(stream):
            required[f"{external}/scorer/assets/{record['asset_id']}.jpg"] = record["sha256"]
    return {**spec, "spec_sha256": digest(spec_path), "required": required}


def verify_required(root: Path, spec: dict) -> None:
    for relative, expected in spec["required"].items():
        path = root / relative
        require(
            regular(path), f"required input missing: {path}; run backup on the original data PC"
        )
        require(digest(path) == expected, f"fixed input hash changed: {relative}")


def choose_store(spec: dict, override: str | None = None) -> Path:
    if override or os.environ.get("FEW_SHOT_DATA_STORE"):
        return Path(override or os.environ["FEW_SHOT_DATA_STORE"]).expanduser().absolute()
    drive_roots = (
        [Path(f"{letter}:/") for letter in "DEFGHIJKLMNOPQRSTUVWXYZ"]
        if os.name == "nt"
        else [Path("/mnt") / letter for letter in "defghijklmnopqrstuvwxyz"]
    )
    found = [
        drive / name / spec["drive_relative_path"]
        for drive in drive_roots
        for name in ("マイドライブ", "My Drive")
        if (drive / name).is_dir()
    ]
    require(
        len(found) == 1,
        "Google Drive mount not uniquely found; pass --store PATH. "
        "For WSL mount the Windows Drive letter first (see docs/external-data-storage.md).",
    )
    return found[0]


def inventory_files(root: Path, spec: dict) -> list[dict]:
    verify_required(root, spec)
    records = []
    for relative_root in spec["roots"]:
        source = root / relative_root
        require(source.is_dir() and not source.is_symlink(), f"missing input directory: {source}")
        for path in sorted(source.rglob("*")):
            require(not path.is_symlink(), f"symlink forbidden: {path}")
            if path.is_dir():
                continue
            require(regular(path), f"non-regular input: {path}")
            relative = safe_relative(path.relative_to(root).as_posix(), spec["roots"])
            records.append(
                {"path": relative, "byte_count": path.stat().st_size, "sha256": digest(path)}
            )
    records.sort(key=lambda row: row["path"])
    require(
        len(records) == len({row["path"].casefold() for row in records}),
        "case-insensitive path collision",
    )
    require(sum(row["byte_count"] for row in records) <= MAX_BYTES, "bundle exceeds 20 GiB limit")
    return records


def make_archive(root: Path, spec: dict, archive: Path) -> dict:
    records = inventory_files(root, spec)
    manifest = {
        "schema_version": "external-input-bundle-v1",
        "roots": spec["roots"],
        "files": records,
    }
    with zipfile.ZipFile(archive, "x", allowZip64=True) as bundle:
        for record in records:
            info = zipfile.ZipInfo(record["path"], date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            observed = hashlib.sha256()
            size = 0
            with (
                (root / record["path"]).open("rb") as source,
                bundle.open(info, "w", force_zip64=True) as target,
            ):
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(block)
                    observed.update(block)
                    size += len(block)
            require(
                size == record["byte_count"] and observed.hexdigest() == record["sha256"],
                "input changed while packaging; nothing published",
            )
        bundle.writestr(
            zipfile.ZipInfo(MANIFEST, date_time=(1980, 1, 1, 0, 0, 0)), json_bytes(manifest)
        )
    return {
        "schema_version": "external-input-receipt-v1",
        "spec_sha256": spec["spec_sha256"],
        "archive_name": spec["archive_name"],
        "archive_sha256": digest(archive),
        "archive_byte_count": archive.stat().st_size,
        "file_count": len(records),
        "unpacked_byte_count": sum(row["byte_count"] for row in records),
    }


def copy_once(source: Path, target: Path) -> None:
    require(not target.is_symlink(), f"symlink destination: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not any(p.is_symlink() for p in target.parents), "symlink destination ancestor")
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())


def publish(root: Path, spec: dict, store: Path, work: Path) -> dict:
    archive = work / spec["archive_name"]
    receipt = make_archive(root, spec, archive)
    store.mkdir(parents=True, exist_ok=True)
    target = store / spec["archive_name"]
    if target.exists():
        require(
            regular(target) and digest(target) == receipt["archive_sha256"],
            "a different or incomplete backup exists; it was not overwritten",
        )
    else:
        copy_once(archive, target)
    require(
        digest(target) == receipt["archive_sha256"], "Drive copy hash differs; receipt not written"
    )
    receipt_path = store / spec["receipt_name"]
    expected = json_bytes(receipt)
    if receipt_path.exists():
        require(
            regular(receipt_path) and receipt_path.read_bytes() == expected,
            "backup receipt differs",
        )
    else:
        with receipt_path.open("xb") as stream:
            stream.write(expected)
    return receipt


def fetch(spec: dict, store: Path, work: Path) -> tuple[Path, dict]:
    receipt_path = store / spec["receipt_name"]
    require(
        regular(receipt_path), "backup not yet available; run backup on the original data PC first"
    )
    require(receipt_path.stat().st_size <= 16384, "receipt is too large")
    receipt = json.loads(receipt_path.read_text("utf-8"))
    require(
        receipt["schema_version"] == "external-input-receipt-v1"
        and receipt["spec_sha256"] == spec["spec_sha256"]
        and receipt["archive_name"] == spec["archive_name"],
        "receipt identity differs",
    )
    source = store / spec["archive_name"]
    require(regular(source), "archive not yet synced; wait for Drive synchronization")
    require(
        source.stat().st_size == receipt["archive_byte_count"],
        "archive size differs or sync incomplete",
    )
    archive = work / spec["archive_name"]
    copy_once(source, archive)
    require(digest(archive) == receipt["archive_sha256"], "archive hash differs or sync incomplete")
    return archive, receipt


def restore_archive(root: Path, spec: dict, archive: Path, receipt: dict, stage: Path) -> int:
    require(digest(archive) == receipt["archive_sha256"], "archive hash differs")
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        names = [entry.filename for entry in entries]
        require(
            len(names) == len(set(name.casefold() for name in names)), "duplicate archive entry"
        )
        require(
            MANIFEST in names and bundle.getinfo(MANIFEST).file_size <= 16 * 1024**2,
            "manifest missing or too large",
        )
        manifest = json.loads(bundle.read(MANIFEST))
        require(
            manifest["schema_version"] == "external-input-bundle-v1"
            and manifest["roots"] == spec["roots"],
            "bundle scope differs",
        )
        records = manifest["files"]
        require(len(records) == receipt["file_count"] <= 20000, "bundle count differs")
        require(len(records) == len({row["path"] for row in records}), "duplicate manifest path")
        require(
            set(names) == {MANIFEST, *(row["path"] for row in records)}, "archive inventory differs"
        )
        require(
            sum(row["byte_count"] for row in records)
            == receipt["unpacked_byte_count"]
            <= MAX_BYTES,
            "bundle size differs",
        )
        for record in records:
            relative = safe_relative(record["path"], spec["roots"])
            info = bundle.getinfo(relative)
            require(
                not info.is_dir()
                and not stat.S_ISLNK(info.external_attr >> 16)
                and not (info.flag_bits & 1),
                "unsupported archive member",
            )
            require(info.file_size == record["byte_count"], "member size differs")
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            require(digest(destination) == record["sha256"], "member hash differs")
    verify_required(stage, spec)
    # Detect every existing-file conflict before installing any restored file.
    for record in records:
        destination = root / record["path"]
        require(
            not any(p.is_symlink() for p in (destination, *destination.parents)),
            "restore destination is symlinked",
        )
        require(
            all(not parent.exists() or parent.is_dir() for parent in destination.parents),
            "restore destination ancestor is not a directory",
        )
        if destination.exists():
            require(
                regular(destination) and digest(destination) == record["sha256"],
                f"existing input differs and was preserved: {record['path']}",
            )
    restored = 0
    for record in records:
        destination = root / record["path"]
        if not destination.exists():
            copy_once(stage / record["path"], destination)
            require(digest(destination) == record["sha256"], "restored file hash differs")
            restored += 1
    verify_required(root, spec)
    return restored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "status", "backup", "restore"))
    parser.add_argument("--store", help="Google Drive sync folder; auto-detected when omitted")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        spec = load_spec(root)
        store = choose_store(spec, args.store)
        require(
            not store.resolve().is_relative_to(root), "Drive store must be outside the checkout"
        )
        require(not any(p.is_symlink() for p in (store, *store.parents)), "store is symlinked")
        print(f"Store: {store}", flush=True)
        if args.action == "init":
            store.mkdir(parents=True, exist_ok=True)
            instructions = store / "README.txt"
            if not instructions.exists() and not instructions.is_symlink():
                instructions.write_text(
                    "Few-Shot Anomaly PoC: fixed external data store\n"
                    "Repository: https://github.com/cab0a/few-shot-anomaly-poc\n"
                    "Instructions: docs/external-data-storage.md\n"
                    "Original data PC: python3 scripts/external_data.py backup\n"
                    "Other PCs: python3 scripts/external_data.py restore\n"
                    "Keep this store private. Do not give it to the blinded reviewer.\n"
                    "Directory creation does not mean data has been backed up.\n",
                    encoding="utf-8",
                )
            print("Store initialized. No dataset uploaded by init.")
            return 0
        if args.action == "status":
            missing = [name for name in spec["required"] if not (root / name).is_file()]
            print(f"Required local files missing: {len(missing)}")
            print(f"Backup receipt present locally: {(store / spec['receipt_name']).is_file()}")
            print("Presence is not integrity verification or confirmation of cloud upload.")
            return 0
        work_root = root / "work/data-transfer"
        work_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="transfer-", dir=work_root) as temporary:
            work = Path(temporary)
            if args.action == "backup":
                receipt = publish(root, spec, store, work)
                print(f"Backup verified in sync folder: {receipt['file_count']} files")
                print("Wait for Google Drive to finish syncing before using another PC.")
            else:
                archive, receipt = fetch(spec, store, work)
                restored = restore_archive(root, spec, archive, receipt, work / "staged")
                print(
                    f"Restored {restored} files; fixed inputs verified. "
                    "No images decoded or scored."
                )
        return 0
    except (TransferError, OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(f"STOP: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
