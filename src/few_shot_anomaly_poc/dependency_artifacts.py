"""Acquire and inspect locked wheel artifacts without installing or importing them."""

from __future__ import annotations

import base64
import csv
import hashlib
import os
import re
import shutil
import stat
import tempfile
import tomllib
import zipfile
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.jsonio import write_json_atomic

USER_AGENT = "few-shot-anomaly-poc/0.2 dependency-artifact-inspection"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
MEMBER_CHUNK_SIZE = 1024 * 1024
MAX_WHEEL_MEMBERS = 100_000
MAX_MEMBER_BYTES = 2_000_000_000
MAX_TOTAL_UNCOMPRESSED_BYTES = 10_000_000_000
ALLOWED_ARTIFACT_HOSTS = frozenset(
    {
        "download-r2.pytorch.org",
        "files.pythonhosted.org",
    }
)
ALLOWED_REGISTRIES = frozenset(
    {
        "https://download.pytorch.org/whl/cpu",
        "https://pypi.org/simple",
    }
)
FIXED_ARTIFACT_SIZES = {
    "torch-2.13.0+cpu-cp313-cp313-manylinux_2_28_x86_64.whl": 191_815_667,
}
INSTALLATION_DECISIONS = frozenset({"DO_NOT_INSTALL", "INSTALL", "REVIEW_REQUIRED"})
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CANONICAL_NAME_PATTERN = re.compile(r"[-_.]+")
LICENSE_BASENAME_PATTERN = re.compile(
    r"^(?:authors?|copying|copyright|credits?|licen[cs]e|notice|"
    r"third[-_. ]party|licenses?[-_. ]bundled)(?:$|[-_. ])",
    re.IGNORECASE,
)


class DependencyArtifactError(Exception):
    """Reject an unsafe or inconsistent locked dependency artifact."""


@dataclass(frozen=True)
class LockedWheel:
    """One exact wheel selected from a uv lock for the fixed v0.2 target."""

    name: str
    version: str
    role: str
    direct: bool
    registry: str
    url: str
    filename: str
    sha256: str
    expected_size: int | None


def _canonical_name(value: str) -> str:
    return CANONICAL_NAME_PATTERN.sub("-", value).lower()


def _require_mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DependencyArtifactError(f"{field} must be a mapping")
    return value


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise DependencyArtifactError(f"{field} must be a non-empty string")
    return value


def _dependency_names(package: dict[str, Any]) -> set[str]:
    dependencies = package.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise DependencyArtifactError("package dependencies must be a list")
    names: set[str] = set()
    for item in dependencies:
        mapping = _require_mapping(item, field="dependency")
        names.add(_canonical_name(_require_string(mapping.get("name"), field="dependency.name")))
    return names


def _dependency_closure(
    direct_names: set[str],
    packages: dict[str, dict[str, Any]],
) -> set[str]:
    pending = list(direct_names)
    resolved: set[str] = set()
    while pending:
        name = pending.pop()
        if name in resolved:
            continue
        if name not in packages:
            raise DependencyArtifactError(f"locked dependency is missing: {name}")
        resolved.add(name)
        pending.extend(_dependency_names(packages[name]) - resolved)
    return resolved


def _wheel_filename(url: str) -> str:
    parsed = urlparse(url)
    filename = unquote(PurePosixPath(parsed.path).name)
    if not filename or "/" in filename or "\\" in filename or "\x00" in filename:
        raise DependencyArtifactError(f"invalid wheel filename in URL: {url}")
    return filename


def _validate_artifact_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_ARTIFACT_HOSTS:
        raise DependencyArtifactError(f"artifact URL is outside the allowlist: {url}")
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise DependencyArtifactError(f"artifact URL contains unsupported components: {url}")


def _is_fixed_target_wheel(filename: str) -> bool:
    lowered = filename.lower()
    if not lowered.endswith(".whl") or "musllinux" in lowered or "cp313t" in lowered:
        return False
    if lowered.endswith("-py3-none-any.whl"):
        return True
    if "x86_64" not in lowered or "manylinux" not in lowered:
        return False
    return "-cp313-cp313-" in lowered or "-py3-none-" in lowered


def _select_wheel(package: dict[str, Any]) -> tuple[str, str, int | None]:
    raw_wheels = package.get("wheels")
    if not isinstance(raw_wheels, list) or not raw_wheels:
        raise DependencyArtifactError(
            f"{package.get('name', '<unknown>')} has no locked wheel candidates"
        )

    candidates: list[tuple[str, str, int | None]] = []
    for raw_wheel in raw_wheels:
        wheel = _require_mapping(raw_wheel, field="package.wheels[]")
        url = _require_string(wheel.get("url"), field="wheel.url")
        filename = _wheel_filename(url)
        if not _is_fixed_target_wheel(filename):
            continue
        hash_value = _require_string(wheel.get("hash"), field="wheel.hash")
        if not hash_value.startswith("sha256:"):
            raise DependencyArtifactError(f"wheel has no SHA-256: {filename}")
        digest = hash_value.removeprefix("sha256:")
        if not SHA256_PATTERN.fullmatch(digest):
            raise DependencyArtifactError(f"wheel has an invalid SHA-256: {filename}")
        raw_size = wheel.get("size")
        if raw_size is not None and (
            isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size <= 0
        ):
            raise DependencyArtifactError(f"wheel has an invalid size: {filename}")
        candidates.append((url, digest, raw_size))

    if len(candidates) != 1:
        raise DependencyArtifactError(
            f"{package.get('name', '<unknown>')} has {len(candidates)} fixed-target wheels"
        )
    return candidates[0]


def load_locked_wheels(lock_path: Path) -> tuple[LockedWheel, ...]:
    """Resolve the exact CPython 3.13 glibc Linux x86-64 wheels from a uv lock."""
    try:
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DependencyArtifactError(f"cannot read dependency lock: {error}") from error

    if lock.get("requires-python") != "==3.13.14":
        raise DependencyArtifactError("dependency lock does not require exactly CPython 3.13.14")
    raw_packages = lock.get("package")
    if not isinstance(raw_packages, list):
        raise DependencyArtifactError("dependency lock has no package list")

    virtual_packages = [
        package
        for package in raw_packages
        if isinstance(package, dict)
        and isinstance(package.get("source"), dict)
        and "virtual" in package["source"]
    ]
    if len(virtual_packages) != 1:
        raise DependencyArtifactError("dependency lock must contain one virtual project")
    root = virtual_packages[0]

    registry_packages: dict[str, dict[str, Any]] = {}
    for raw_package in raw_packages:
        package = _require_mapping(raw_package, field="package")
        source = _require_mapping(package.get("source"), field="package.source")
        if "registry" not in source:
            continue
        name = _canonical_name(_require_string(package.get("name"), field="package.name"))
        if name in registry_packages:
            raise DependencyArtifactError(f"duplicate locked distribution: {name}")
        registry_packages[name] = package

    runtime_direct = _dependency_names(root)
    raw_dev = _require_mapping(root.get("dev-dependencies"), field="project.dev-dependencies")
    dev_items = raw_dev.get("dev")
    if not isinstance(dev_items, list):
        raise DependencyArtifactError("project.dev-dependencies.dev must be a list")
    development_direct = {
        _canonical_name(
            _require_string(
                _require_mapping(item, field="project.dev-dependencies.dev[]").get("name"),
                field="development dependency name",
            )
        )
        for item in dev_items
    }
    runtime = _dependency_closure(runtime_direct, registry_packages)
    development = _dependency_closure(development_direct, registry_packages) - runtime
    if runtime | development != set(registry_packages):
        unresolved = sorted(set(registry_packages) - runtime - development)
        raise DependencyArtifactError(f"orphan locked distributions: {', '.join(unresolved)}")

    selected: list[LockedWheel] = []
    for name, package in sorted(registry_packages.items()):
        source = _require_mapping(package.get("source"), field=f"{name}.source")
        registry = _require_string(source.get("registry"), field=f"{name}.source.registry")
        if registry not in ALLOWED_REGISTRIES:
            raise DependencyArtifactError(f"unapproved registry for {name}: {registry}")
        url, digest, locked_size = _select_wheel(package)
        _validate_artifact_url(url)
        filename = _wheel_filename(url)
        expected_size = locked_size or FIXED_ARTIFACT_SIZES.get(filename)
        selected.append(
            LockedWheel(
                name=name,
                version=_require_string(package.get("version"), field=f"{name}.version"),
                role="runtime" if name in runtime else "development",
                direct=name in runtime_direct or name in development_direct,
                registry=registry,
                url=url,
                filename=filename,
                sha256=digest,
                expected_size=expected_size,
            )
        )
    return tuple(selected)


def _validated_member_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    if not name or "\\" in name or "\x00" in name:
        raise DependencyArtifactError(f"unsafe wheel member name: {name!r}")
    trimmed = name[:-1] if name.endswith("/") else name
    raw_parts = tuple(trimmed.split("/"))
    pure_path = PurePosixPath(trimmed)
    if (
        not raw_parts
        or pure_path.is_absolute()
        or any(part in {"", ".", ".."} for part in raw_parts)
        or ":" in raw_parts[0]
    ):
        raise DependencyArtifactError(f"unsafe wheel member path: {name!r}")
    if info.flag_bits & 0x1:
        raise DependencyArtifactError(f"encrypted wheel member is forbidden: {name!r}")
    if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
        raise DependencyArtifactError(f"unsupported wheel compression: {name!r}")

    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    allowed_types = {0, stat.S_IFREG}
    if info.is_dir():
        allowed_types.add(stat.S_IFDIR)
    if file_type not in allowed_types:
        raise DependencyArtifactError(f"unsupported wheel member type: {name!r}")
    if info.file_size < 0 or info.file_size > MAX_MEMBER_BYTES:
        raise DependencyArtifactError(f"wheel member is too large: {name!r}")
    return name


def _sha256_stream(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    while chunk := stream.read(MEMBER_CHUNK_SIZE):
        digest.update(chunk)
        byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def _record_digest(hex_digest: str) -> str:
    return base64.urlsafe_b64encode(bytes.fromhex(hex_digest)).rstrip(b"=").decode("ascii")


def _parse_record(content: bytes, *, record_path: str) -> dict[str, tuple[str, str]]:
    try:
        text = content.decode("utf-8")
        rows = list(csv.reader(text.splitlines()))
    except (UnicodeDecodeError, csv.Error) as error:
        raise DependencyArtifactError(f"invalid wheel RECORD: {error}") from error
    records: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3 or not row[0]:
            raise DependencyArtifactError("wheel RECORD contains an invalid row")
        path, digest, size = row
        if path in records:
            raise DependencyArtifactError(f"wheel RECORD contains a duplicate path: {path}")
        if path == record_path:
            if digest or size:
                raise DependencyArtifactError("wheel RECORD must not hash itself")
        elif not digest.startswith("sha256=") or not size.isdecimal():
            raise DependencyArtifactError(f"wheel RECORD entry is not SHA-256 pinned: {path}")
        records[path] = (digest, size)
    return records


def _license_kind(path: str) -> str | None:
    lowered = path.lower()
    basename = PurePosixPath(path).name
    in_license_directory = ".dist-info/licenses/" in lowered
    if not in_license_directory and not LICENSE_BASENAME_PATTERN.match(basename):
        return None
    lowered_basename = basename.lower()
    if "notice" in lowered_basename or "third" in lowered_basename:
        return "notice"
    if "authors" in lowered_basename or "credits" in lowered_basename:
        return "attribution"
    if "bundled" in lowered_basename:
        return "bundled_licenses"
    return "license"


def _native_kind(path: str) -> str | None:
    basename = PurePosixPath(path).name.lower()
    if basename.endswith(".so") or ".so." in basename:
        return "elf_shared_object"
    if basename.endswith(".pyd"):
        return "python_extension"
    if basename.endswith(".dll"):
        return "windows_shared_library"
    if basename.endswith(".dylib"):
        return "macos_shared_library"
    return None


def _declared_license_matches(path: str, declared: str, dist_info_prefix: str) -> bool:
    normalized = declared.replace("\\", "/").lstrip("/")
    return path in {
        f"{dist_info_prefix}/{normalized}",
        f"{dist_info_prefix}/licenses/{normalized}",
    }


def inspect_wheel(path: Path, locked: LockedWheel) -> dict[str, Any]:
    """Validate and inventory one verified wheel without extracting or executing it."""
    observed_sha256 = sha256_file(path)
    if observed_sha256 != locked.sha256:
        raise DependencyArtifactError(
            f"checksum mismatch for {locked.filename}: "
            f"expected {locked.sha256}, observed {observed_sha256}"
        )
    byte_count = path.stat().st_size
    if locked.expected_size is not None and byte_count != locked.expected_size:
        raise DependencyArtifactError(
            f"size mismatch for {locked.filename}: "
            f"expected {locked.expected_size}, observed {byte_count}"
        )

    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > MAX_WHEEL_MEMBERS:
                raise DependencyArtifactError(
                    f"invalid wheel member count for {locked.filename}: {len(infos)}"
                )
            names: set[str] = set()
            casefolded_names: set[str] = set()
            total_uncompressed_bytes = 0
            metadata_paths: list[str] = []
            record_paths: list[str] = []
            for info in infos:
                name = _validated_member_name(info)
                if name in names or name.casefold() in casefolded_names:
                    raise DependencyArtifactError(f"duplicate wheel member target: {name!r}")
                names.add(name)
                casefolded_names.add(name.casefold())
                total_uncompressed_bytes += info.file_size
                if total_uncompressed_bytes > MAX_TOTAL_UNCOMPRESSED_BYTES:
                    raise DependencyArtifactError(
                        f"wheel expands beyond {MAX_TOTAL_UNCOMPRESSED_BYTES} bytes"
                    )
                if (
                    not info.is_dir()
                    and name.count("/") == 1
                    and name.endswith(".dist-info/METADATA")
                ):
                    metadata_paths.append(name)
                if (
                    not info.is_dir()
                    and name.count("/") == 1
                    and name.endswith(".dist-info/RECORD")
                ):
                    record_paths.append(name)

            if len(metadata_paths) != 1 or len(record_paths) != 1:
                raise DependencyArtifactError(
                    f"{locked.filename} must contain one METADATA and one RECORD"
                )
            metadata_path = metadata_paths[0]
            record_path = record_paths[0]
            dist_info_prefix = metadata_path.removesuffix("/METADATA")
            if record_path != f"{dist_info_prefix}/RECORD":
                raise DependencyArtifactError("METADATA and RECORD use different dist-info roots")

            metadata_bytes = archive.read(metadata_path)
            message = BytesParser(policy=policy.default).parsebytes(metadata_bytes)
            metadata_name = message.get("Name")
            metadata_version = message.get("Version")
            if _canonical_name(metadata_name or "") != locked.name:
                raise DependencyArtifactError(
                    f"METADATA name mismatch for {locked.filename}: {metadata_name!r}"
                )
            if metadata_version != locked.version:
                raise DependencyArtifactError(
                    f"METADATA version mismatch for {locked.filename}: {metadata_version!r}"
                )
            declared_license_files = sorted(set(message.get_all("License-File", [])))

            record_bytes = archive.read(record_path)
            records = _parse_record(record_bytes, record_path=record_path)
            file_infos = {info.filename: info for info in infos if not info.is_dir()}
            if set(records) != set(file_infos):
                missing = sorted(set(file_infos) - set(records))
                extra = sorted(set(records) - set(file_infos))
                raise DependencyArtifactError(
                    f"wheel RECORD path mismatch: missing={missing[:3]}, extra={extra[:3]}"
                )

            member_digests: dict[str, tuple[str, int]] = {}
            for name, info in sorted(file_infos.items()):
                with archive.open(info, "r") as stream:
                    digest, observed_size = _sha256_stream(stream)
                if observed_size != info.file_size:
                    raise DependencyArtifactError(f"wheel member size mismatch: {name}")
                member_digests[name] = (digest, observed_size)
                if name == record_path:
                    continue
                record_digest, record_size = records[name]
                if record_digest.removeprefix("sha256=") != _record_digest(digest):
                    raise DependencyArtifactError(f"wheel RECORD hash mismatch: {name}")
                if int(record_size) != observed_size:
                    raise DependencyArtifactError(f"wheel RECORD size mismatch: {name}")

            license_material: list[dict[str, Any]] = []
            native_files: list[dict[str, Any]] = []
            for name, (digest, observed_size) in sorted(member_digests.items()):
                kind = _license_kind(name)
                if kind is not None:
                    declared = sorted(
                        value
                        for value in declared_license_files
                        if _declared_license_matches(name, value, dist_info_prefix)
                    )
                    license_material.append(
                        {
                            "declared_by_metadata": bool(declared),
                            "declared_names": declared,
                            "kind": kind,
                            "path": name,
                            "sha256": digest,
                            "size": observed_size,
                        }
                    )
                native_kind = _native_kind(name)
                if native_kind is not None:
                    native_files.append(
                        {
                            "kind": native_kind,
                            "path": name,
                            "sha256": digest,
                            "size": observed_size,
                        }
                    )

            unmatched_declared = [
                declared
                for declared in declared_license_files
                if not any(
                    _declared_license_matches(item["path"], declared, dist_info_prefix)
                    for item in license_material
                )
            ]
            if unmatched_declared:
                raise DependencyArtifactError(
                    f"declared license files are missing from {locked.filename}: "
                    f"{unmatched_declared}"
                )
            if not license_material:
                raise DependencyArtifactError(
                    f"{locked.filename} contains no reviewable license material"
                )

            return {
                "archive": {
                    "casefold_collision_count": 0,
                    "directory_count": sum(info.is_dir() for info in infos),
                    "file_count": len(file_infos),
                    "member_count": len(infos),
                    "record_entry_count": len(records),
                    "record_hash_algorithm": "sha256",
                    "record_verification": "pass",
                    "safe_member_validation": "pass",
                    "total_uncompressed_bytes": total_uncompressed_bytes,
                },
                "artifact": {
                    "byte_count": byte_count,
                    "checksum_status": "upstream_verified",
                    "filename": locked.filename,
                    "observed_sha256": observed_sha256,
                    "published_sha256": locked.sha256,
                    "requested_url": locked.url,
                    "retained_outside_git": True,
                },
                "dependency": {
                    "direct": locked.direct,
                    "name": locked.name,
                    "registry": locked.registry,
                    "role": locked.role,
                    "version": locked.version,
                },
                "license": {
                    "classifiers": sorted(
                        classifier
                        for classifier in message.get_all("Classifier", [])
                        if classifier.startswith("License ::")
                    ),
                    "declared_files": declared_license_files,
                    "expression": message.get("License-Expression"),
                    "material": license_material,
                    "material_count": len(license_material),
                },
                "metadata": {
                    "metadata_version": message.get("Metadata-Version"),
                    "name": metadata_name,
                    "requires_python": message.get("Requires-Python"),
                    "version": metadata_version,
                },
                "native_files": native_files,
                "native_file_count": len(native_files),
            }
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        if isinstance(error, DependencyArtifactError):
            raise
        raise DependencyArtifactError(f"cannot inspect {locked.filename}: {error}") from error


def _validate_artifact_directory(artifact_dir: Path, project_root: Path) -> None:
    resolved_root = project_root.resolve()
    resolved_artifacts = artifact_dir.resolve()
    if resolved_artifacts == resolved_root:
        raise DependencyArtifactError("artifact directory cannot be the project root")
    try:
        relative = resolved_artifacts.relative_to(resolved_root)
    except ValueError:
        return
    if relative.parts[:2] != ("data", "external"):
        raise DependencyArtifactError(
            "an in-repository artifact directory must be under ignored data/external"
        )


def _download_locked_wheel(locked: LockedWheel, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    _validate_artifact_url(locked.url)
    temporary_path: Path | None = None
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
            request = Request(locked.url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=60) as response:
                effective_url = response.geturl()
                _validate_artifact_url(effective_url)
                while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                    output.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)

        observed_sha256 = digest.hexdigest()
        if observed_sha256 != locked.sha256:
            raise DependencyArtifactError(
                f"checksum mismatch for {locked.filename}: "
                f"expected {locked.sha256}, observed {observed_sha256}"
            )
        if locked.expected_size is not None and byte_count != locked.expected_size:
            raise DependencyArtifactError(
                f"size mismatch for {locked.filename}: "
                f"expected {locked.expected_size}, observed {byte_count}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_existing_artifacts(
    artifact_dir: Path,
    locked_wheels: tuple[LockedWheel, ...],
) -> None:
    if not artifact_dir.is_dir() or artifact_dir.is_symlink():
        raise DependencyArtifactError("reused artifact directory must be a real directory")
    expected = {wheel.filename for wheel in locked_wheels}
    observed = {path.name for path in artifact_dir.iterdir()}
    if observed != expected:
        raise DependencyArtifactError(
            "reused artifact set differs from the lock: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    if any(not path.is_file() or path.is_symlink() for path in artifact_dir.iterdir()):
        raise DependencyArtifactError("reused artifact directory contains a non-regular file")


def acquire_and_inspect_locked_wheels(
    *,
    lock_path: Path,
    environment_path: Path,
    artifact_dir: Path,
    output_path: Path,
    project_root: Path,
    reuse_existing: bool,
    installation_decision: str,
    decision_reason: str,
) -> dict[str, Any]:
    """Acquire or reuse exact wheels, inspect them, and write a deterministic report."""
    if installation_decision not in INSTALLATION_DECISIONS:
        raise DependencyArtifactError(
            f"invalid installation decision: {installation_decision}"
        )
    if not decision_reason.strip():
        raise DependencyArtifactError("installation decision requires a reason")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    _validate_artifact_directory(artifact_dir, project_root)
    locked_wheels = load_locked_wheels(lock_path)

    created_artifact_dir = False
    if reuse_existing:
        _validate_existing_artifacts(artifact_dir, locked_wheels)
    else:
        if artifact_dir.exists():
            raise FileExistsError(f"refusing to reuse {artifact_dir}")
        artifact_dir.mkdir(parents=True)
        created_artifact_dir = True
        try:
            for locked in locked_wheels:
                _download_locked_wheel(locked, artifact_dir / locked.filename)
        except Exception:
            if created_artifact_dir:
                shutil.rmtree(artifact_dir)
            raise

    inspections = [
        inspect_wheel(artifact_dir / locked.filename, locked) for locked in locked_wheels
    ]
    report = {
        "acquisition_boundary": {
            "artifact_execution": False,
            "artifact_installation": False,
            "artifact_storage": "outside_git",
            "dataset_access": False,
            "dinov2_checkpoint_acquired": False,
            "dinov2_source_acquired": False,
            "package_import": False,
            "wheel_extraction": False,
        },
        "decision": {
            "installation": installation_decision,
            "reason": decision_reason.strip(),
        },
        "environment": {
            "environment_definition_sha256": sha256_file(environment_path),
            "lock_sha256": sha256_file(lock_path),
            "python": "CPython 3.13.14",
            "target": "glibc Linux x86_64",
            "uv": "0.11.32",
        },
        "packages": inspections,
        "schema_version": "v0.2-dependency-artifact-inspection-v1",
        "summary": {
            "all_artifact_checksums_verified": True,
            "all_license_material_inventoried": True,
            "all_metadata_identities_verified": True,
            "all_wheel_records_verified": True,
            "all_wheels_safe_to_inspect": True,
            "development_distribution_count": sum(
                item["dependency"]["role"] == "development" for item in inspections
            ),
            "distribution_count": len(inspections),
            "license_material_count": sum(
                item["license"]["material_count"] for item in inspections
            ),
            "native_file_count": sum(item["native_file_count"] for item in inspections),
            "runtime_distribution_count": sum(
                item["dependency"]["role"] == "runtime" for item in inspections
            ),
        },
    }
    write_json_atomic(output_path, report)
    return report
