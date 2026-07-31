from __future__ import annotations

import base64
import csv
import hashlib
import io
import stat
import zipfile
from pathlib import Path

import pytest

from few_shot_anomaly_poc.dependency_artifacts import (
    DependencyArtifactError,
    LockedWheel,
    _validate_artifact_directory,
    inspect_wheel,
    load_locked_wheels,
)

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "environments/v0.2-preflight/uv.lock"


def _record_hash(content: bytes) -> str:
    digest = hashlib.sha256(content).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _write_wheel(
    path: Path,
    *,
    name: str = "example",
    version: str = "1.0",
    extra_members: list[tuple[str, bytes, int | None]] | None = None,
    corrupt_record_path: str | None = None,
) -> LockedWheel:
    prefix = f"{name}-{version}.dist-info"
    members: list[tuple[str, bytes, int | None]] = [
        (
            f"{prefix}/METADATA",
            (
                "Metadata-Version: 2.4\n"
                f"Name: {name}\n"
                f"Version: {version}\n"
                "License-Expression: MIT\n"
                "License-File: LICENSE.txt\n"
                "Requires-Python: >=3.13\n"
                "\n"
            ).encode(),
            None,
        ),
        (f"{prefix}/licenses/LICENSE.txt", b"The MIT License\n", None),
        (f"{name}/__init__.py", b"", None),
    ]
    members.extend(extra_members or [])
    record_path = f"{prefix}/RECORD"
    record_stream = io.StringIO(newline="")
    writer = csv.writer(record_stream, lineterminator="\n")
    for member_path, content, _ in members:
        digest = "broken" if member_path == corrupt_record_path else _record_hash(content)
        writer.writerow([member_path, f"sha256={digest}", str(len(content))])
    writer.writerow([record_path, "", ""])
    members.append((record_path, record_stream.getvalue().encode(), None))

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member_path, content, mode in members:
            info = zipfile.ZipInfo(member_path)
            info.compress_type = zipfile.ZIP_DEFLATED
            if mode is not None:
                info.create_system = 3
                info.external_attr = mode << 16
            archive.writestr(info, content)

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return LockedWheel(
        name=name,
        version=version,
        role="runtime",
        direct=True,
        registry="https://pypi.org/simple",
        url=f"https://files.pythonhosted.org/packages/example/{path.name}",
        filename=path.name,
        sha256=digest,
        expected_size=path.stat().st_size,
    )


def test_committed_lock_selects_exact_fixed_target_wheels() -> None:
    wheels = load_locked_wheels(LOCK_PATH)

    assert len(wheels) == 17
    assert sum(wheel.role == "runtime" for wheel in wheels) == 11
    assert sum(wheel.role == "development" for wheel in wheels) == 6
    assert {wheel.name for wheel in wheels if wheel.direct} == {
        "numpy",
        "pytest",
        "ruff",
        "torch",
    }
    torch = next(wheel for wheel in wheels if wheel.name == "torch")
    assert torch.filename == "torch-2.13.0+cpu-cp313-cp313-manylinux_2_28_x86_64.whl"
    assert torch.sha256 == "3fbf9c9d1f3c10c2d59d04aca426dee9ccc6ceb32d255c61e93acc3b4f75fae6"
    assert torch.expected_size == 191_815_667
    assert all("musllinux" not in wheel.filename for wheel in wheels)


def test_inspection_verifies_record_license_and_native_inventory(tmp_path: Path) -> None:
    wheel_path = tmp_path / "example-1.0-py3-none-any.whl"
    locked = _write_wheel(
        wheel_path,
        extra_members=[
            ("example/native.so", b"native bytes", None),
            ("example-1.0.dist-info/NOTICE", b"notice bytes", None),
            (
                "example/_vendor/vendored-2.0.dist-info/METADATA",
                b"Metadata-Version: 2.4\nName: vendored\nVersion: 2.0\n\n",
                None,
            ),
            ("example/_vendor/vendored-2.0.dist-info/RECORD", b"", None),
        ],
    )

    result = inspect_wheel(wheel_path, locked)

    assert result["archive"]["record_verification"] == "pass"
    assert result["artifact"]["checksum_status"] == "upstream_verified"
    assert result["metadata"]["name"] == "example"
    assert result["license"]["expression"] == "MIT"
    assert result["license"]["material_count"] == 2
    assert result["native_file_count"] == 1
    assert result["native_files"][0]["kind"] == "elf_shared_object"


def test_inspection_rejects_record_hash_mismatch(tmp_path: Path) -> None:
    wheel_path = tmp_path / "example-1.0-py3-none-any.whl"
    locked = _write_wheel(
        wheel_path,
        corrupt_record_path="example/__init__.py",
    )

    with pytest.raises(DependencyArtifactError, match="RECORD hash mismatch"):
        inspect_wheel(wheel_path, locked)


def test_inspection_rejects_symlink_member(tmp_path: Path) -> None:
    wheel_path = tmp_path / "example-1.0-py3-none-any.whl"
    locked = _write_wheel(
        wheel_path,
        extra_members=[
            (
                "example/link",
                b"target",
                stat.S_IFLNK | 0o777,
            )
        ],
    )

    with pytest.raises(DependencyArtifactError, match="unsupported wheel member type"):
        inspect_wheel(wheel_path, locked)


def test_inspection_rejects_path_traversal(tmp_path: Path) -> None:
    wheel_path = tmp_path / "example-1.0-py3-none-any.whl"
    locked = _write_wheel(
        wheel_path,
        extra_members=[("../escape", b"unsafe", None)],
    )

    with pytest.raises(DependencyArtifactError, match="unsafe wheel member path"):
        inspect_wheel(wheel_path, locked)


def test_artifacts_inside_repository_must_use_ignored_external_boundary(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()

    _validate_artifact_directory(project_root / "data/external/wheels", project_root)
    _validate_artifact_directory(tmp_path / "outside/wheels", project_root)
    with pytest.raises(DependencyArtifactError, match="data/external"):
        _validate_artifact_directory(project_root / "artifacts/wheels", project_root)
