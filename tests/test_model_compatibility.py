from __future__ import annotations

import hashlib
import io
import socket
import tarfile
from pathlib import Path

import pytest

import few_shot_anomaly_poc.model_compatibility as compatibility
from few_shot_anomaly_poc.model_assets import (
    SOURCE_ROOT,
    ModelAssetError,
    extract_source_archive,
)
from few_shot_anomaly_poc.model_compatibility import (
    ModelCompatibilityError,
    NetworkGuard,
    _state_key_manifest,
    _summarize_state_dict,
)


def _write_source_archive(path: Path) -> str:
    members = {
        "LICENSE": (
            b"Apache License\nVersion 2.0, January 2004\n"
            b"http://www.apache.org/licenses/\n"
        ),
        "MODEL_CARD.md": b"# Model Card\n",
        "README.md": (
            b"DINOv2 code and model weights are released under the "
            b"Apache License 2.0\n"
        ),
        "dinov2/__init__.py": b"__version__ = 'test'\n",
        "dinov2/hub/backbones.py": (
            b"_DINOV2_BASE_URL = 'https://example.invalid'\n"
            b"def dinov2_vits14():\n"
            b"    return _DINOV2_BASE_URL + '/model/model_pretrain.pth'\n"
        ),
        "hubconf.py": b"dependencies = ['torch']\n",
    }
    with tarfile.open(path, "w:gz") as archive:
        root = tarfile.TarInfo(SOURCE_ROOT)
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        for relative_name, content in members.items():
            member = tarfile.TarInfo(f"{SOURCE_ROOT}/{relative_name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeDevice:
    type = "cpu"

    def __str__(self) -> str:
        return self.type


class _FakeFiniteResult:
    def __init__(self, value: bool) -> None:
        self.value = value

    def all(self) -> _FakeFiniteResult:
        return self

    def item(self) -> bool:
        return self.value


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...], *, finite: bool = True) -> None:
        self.device = _FakeDevice()
        self.dtype = _FakeTorch.float32
        self.layout = _FakeTorch.strided
        self.shape = shape
        self.finite = finite

    def numel(self) -> int:
        result = 1
        for value in self.shape:
            result *= value
        return result

    def element_size(self) -> int:
        return 4


class _FakeDType:
    def __str__(self) -> str:
        return "torch.float32"


class _FakeTorch:
    Tensor = _FakeTensor
    float32 = _FakeDType()
    strided = object()

    @staticmethod
    def isfinite(tensor: _FakeTensor) -> _FakeFiniteResult:
        return _FakeFiniteResult(tensor.finite)


def test_source_extraction_requires_hash_and_writes_a_fixed_tree(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    archive = tmp_path / "source.tar.gz"
    digest = _write_source_archive(archive)
    destination = project / "data/external/source"

    result = extract_source_archive(
        archive,
        expected_sha256=digest,
        destination=destination,
        project_root=project,
    )

    assert result["safe_extraction"] == "pass"
    assert result["file_count"] == 6
    assert (destination / SOURCE_ROOT / "dinov2/__init__.py").is_file()
    assert len(result["tree_manifest_sha256"]) == 64


def test_source_extraction_rejects_checksum_mismatch(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    archive = tmp_path / "source.tar.gz"
    _write_source_archive(archive)
    destination = project / "data/external/source"

    with pytest.raises(ModelAssetError, match="checksum mismatch"):
        extract_source_archive(
            archive,
            expected_sha256="0" * 64,
            destination=destination,
            project_root=project,
        )

    assert not destination.exists()


def test_network_guard_rejects_socket_resolution_and_restores_it() -> None:
    original = socket.getaddrinfo

    with (
        NetworkGuard() as guard,
        pytest.raises(ModelCompatibilityError, match="network access"),
    ):
        socket.getaddrinfo("example.com", 443)

    assert guard.attempts == ["blocked_socket_operation"]
    assert socket.getaddrinfo is original


def test_state_dictionary_summary_requires_fixed_finite_float32_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = ["blocks.0.weight", "norm.weight"]
    monkeypatch.setattr(compatibility, "EXPECTED_STATE_KEY_COUNT", len(keys))
    monkeypatch.setattr(
        compatibility,
        "EXPECTED_STATE_KEY_MANIFEST_SHA256",
        _state_key_manifest(keys),
    )
    state = {
        "blocks.0.weight": _FakeTensor((2, 3)),
        "norm.weight": _FakeTensor((3,)),
    }

    result = _summarize_state_dict(_FakeTorch, state)

    assert result["tensor_count"] == 2
    assert result["total_tensor_elements"] == 9
    assert result["total_tensor_bytes"] == 36
    assert result["dtype_counts"] == {
        str(_FakeTorch.float32).removeprefix("torch."): 2
    }


def test_state_dictionary_summary_rejects_non_finite_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = ["norm.weight"]
    monkeypatch.setattr(compatibility, "EXPECTED_STATE_KEY_COUNT", 1)
    monkeypatch.setattr(
        compatibility,
        "EXPECTED_STATE_KEY_MANIFEST_SHA256",
        _state_key_manifest(keys),
    )

    with pytest.raises(ModelCompatibilityError, match="non-finite"):
        _summarize_state_dict(
            _FakeTorch,
            {"norm.weight": _FakeTensor((3,), finite=False)},
        )
