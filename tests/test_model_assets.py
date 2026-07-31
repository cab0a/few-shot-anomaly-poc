from __future__ import annotations

import io
import os
import pickle
import tarfile
import zipfile
from collections import OrderedDict
from pathlib import Path

import pytest

from few_shot_anomaly_poc.model_assets import (
    SOURCE_REVISION,
    SOURCE_ROOT,
    ModelAssetError,
    _hash_addressed_name,
    _validate_external_directory,
    _validate_https_url,
    _validated_commit,
    inspect_checkpoint_archive,
    inspect_source_archive,
)


def _source_members() -> dict[str, bytes]:
    return {
        "LICENSE": (
            b"Apache License\nVersion 2.0, January 2004\n"
            b"http://www.apache.org/licenses/\n"
        ),
        "MODEL_CARD.md": b"# Model Card\n",
        "README.md": (
            b"DINOv2 code and model weights are released under the "
            b"Apache License 2.0\n"
        ),
        "dinov2/hub/backbones.py": (
            b"_DINOV2_BASE_URL = 'https://example.invalid'\n"
            b"def dinov2_vits14():\n"
            b"    return _DINOV2_BASE_URL + '/model/model_pretrain.pth'\n"
        ),
        "hubconf.py": b"dependencies = ['torch']\n",
    }


def _write_source_archive(
    path: Path,
    *,
    members: dict[str, bytes] | None = None,
    extra_member: tarfile.TarInfo | None = None,
) -> None:
    with tarfile.open(path, "w:gz") as archive:
        root = tarfile.TarInfo(SOURCE_ROOT)
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        for relative_name, content in (members or _source_members()).items():
            member = tarfile.TarInfo(f"{SOURCE_ROOT}/{relative_name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        if extra_member is not None:
            archive.addfile(extra_member)


def _checkpoint_pickle() -> bytes:
    state = OrderedDict(
        (key, None)
        for key in (
            "blocks.0.attn.proj.weight",
            "blocks.0.attn.qkv.weight",
            "cls_token",
            "mask_token",
            "norm.weight",
            "patch_embed.proj.weight",
            "pos_embed",
        )
    )
    return pickle.dumps(state, protocol=2)


def _write_checkpoint(path: Path, *, data_pickle: bytes | None = None) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", data_pickle or _checkpoint_pickle())
        archive.writestr("archive/byteorder", b"little")
        archive.writestr("archive/version", b"3\n")
        archive.writestr("archive/data/0", b"opaque storage bytes")


def test_source_archive_inspection_verifies_fixed_structure_and_license(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.tar.gz"
    _write_source_archive(archive)

    result = inspect_source_archive(archive)

    assert result["safe_structure"] == "pass"
    assert result["top_level_directory"] == SOURCE_ROOT
    assert {item["path"] for item in result["required_files"]} == set(
        _source_members()
    )
    assert {item["path"] for item in result["license_material"]} == {"LICENSE"}


def test_source_archive_rejects_path_outside_fixed_root(tmp_path: Path) -> None:
    archive = tmp_path / "source.tar.gz"
    extra = tarfile.TarInfo("../escape")
    extra.size = 0
    _write_source_archive(archive, extra_member=extra)

    with pytest.raises(ModelAssetError, match="unsafe source archive member"):
        inspect_source_archive(archive)


def test_source_archive_rejects_link_member(tmp_path: Path) -> None:
    archive = tmp_path / "source.tar.gz"
    extra = tarfile.TarInfo(f"{SOURCE_ROOT}/link")
    extra.type = tarfile.SYMTYPE
    extra.linkname = "README.md"
    _write_source_archive(archive, extra_member=extra)

    with pytest.raises(ModelAssetError, match="unsupported member type"):
        inspect_source_archive(archive)


def test_checkpoint_inspection_parses_structure_without_deserialization(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    _write_checkpoint(checkpoint)

    result = inspect_checkpoint_archive(checkpoint)

    assert result["safe_structure"] == "pass"
    assert result["byteorder_member_present"] is True
    assert result["crc_verification"] == "pass"
    assert result["data_member_count"] == 1
    assert result["pickle"]["pickle_deserialized"] is False
    assert result["pickle"]["global_references"] == ["collections OrderedDict"]
    assert result["version_member_present"] is True


def test_checkpoint_inspection_rejects_unexpected_pickle_global(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    payload = pickle.dumps(os.system, protocol=2)
    _write_checkpoint(checkpoint, data_pickle=payload)

    with pytest.raises(ModelAssetError, match="globals differ"):
        inspect_checkpoint_archive(checkpoint)


def test_checkpoint_inspection_rejects_path_traversal(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    _write_checkpoint(checkpoint)
    with zipfile.ZipFile(checkpoint, "a") as archive:
        archive.writestr("../escape", b"unsafe")

    with pytest.raises(ModelAssetError, match="unsafe checkpoint member"):
        inspect_checkpoint_archive(checkpoint)


def test_external_asset_directory_is_restricted_inside_project(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()

    accepted = _validate_external_directory(
        project / "data/external/v0.2/model-assets",
        project,
    )

    assert accepted == (project / "data/external/v0.2/model-assets").resolve()
    with pytest.raises(ModelAssetError, match="data/external"):
        _validate_external_directory(project / "artifacts/model-assets", project)


def test_asset_urls_and_hash_addressed_names_are_fixed() -> None:
    _validate_https_url(
        f"https://github.com/facebookresearch/dinov2/archive/{SOURCE_REVISION}.tar.gz",
        allowed_hosts=frozenset({"github.com"}),
    )
    with pytest.raises(ModelAssetError, match="HTTPS boundary"):
        _validate_https_url(
            "https://example.com/checkpoint.pth",
            allowed_hosts=frozenset({"dl.fbaipublicfiles.com"}),
        )

    digest = "a" * 64
    assert _hash_addressed_name("checkpoint", digest, ".pth") == (
        f"checkpoint-sha256-{digest}.pth"
    )
    assert _validated_commit("a" * 40, field="test commit") == "a" * 40
    with pytest.raises(ModelAssetError, match="full lowercase Git commit"):
        _validated_commit("a" * 64, field="test commit")
