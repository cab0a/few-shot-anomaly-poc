"""Acquire and structurally inspect the fixed DINOv2 source and checkpoint."""

from __future__ import annotations

import hashlib
import os
import pickletools
import re
import stat
import tarfile
import tempfile
import zipfile
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import urlparse
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.jsonio import write_json_atomic

SOURCE_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
SOURCE_URL = (
    "https://github.com/facebookresearch/dinov2/archive/"
    f"{SOURCE_REVISION}.tar.gz"
)
CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/"
    "dinov2_vits14/dinov2_vits14_pretrain.pth"
)
SOURCE_ROOT = f"dinov2-{SOURCE_REVISION}"
EXPECTED_CHECKPOINT_BYTES = 88_283_115
OUTPUT_SCHEMA = "v0.2-model-asset-acquisition-v1"
USER_AGENT = "few-shot-anomaly-poc/0.2 model-asset-acquisition"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
MEMBER_CHUNK_SIZE = 1024 * 1024
MAX_SOURCE_MEMBERS = 20_000
MAX_SOURCE_MEMBER_BYTES = 100_000_000
MAX_SOURCE_TOTAL_BYTES = 1_000_000_000
MAX_CHECKPOINT_MEMBERS = 10_000
MAX_CHECKPOINT_MEMBER_BYTES = 1_000_000_000
MAX_CHECKPOINT_TOTAL_BYTES = 2_000_000_000
MAX_PICKLE_BYTES = 20_000_000
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SOURCE_HOSTS = frozenset({"github.com", "codeload.github.com"})
CHECKPOINT_HOSTS = frozenset({"dl.fbaipublicfiles.com"})
REQUIRED_SOURCE_FILES = frozenset(
    {
        "LICENSE",
        "MODEL_CARD.md",
        "README.md",
        "dinov2/hub/backbones.py",
        "hubconf.py",
    }
)
EXPECTED_CHECKPOINT_GLOBALS = frozenset(
    {
        "collections OrderedDict",
        "torch FloatStorage",
        "torch Tensor",
        "torch._tensor _rebuild_from_type_v2",
        "torch._utils _rebuild_tensor_v2",
    }
)
FORBIDDEN_PICKLE_OPCODES = frozenset(
    {
        "EXT1",
        "EXT2",
        "EXT4",
        "INST",
        "NEWOBJ_EX",
        "OBJ",
        "STACK_GLOBAL",
    }
)
EXPECTED_STATE_KEYS = frozenset(
    {
        "blocks.0.attn.proj.weight",
        "blocks.0.attn.qkv.weight",
        "cls_token",
        "mask_token",
        "norm.weight",
        "patch_embed.proj.weight",
        "pos_embed",
    }
)


class ModelAssetError(Exception):
    """Reject an unregistered, unsafe, or structurally unexpected model asset."""


class _RestrictedRedirectHandler(HTTPRedirectHandler):
    def __init__(self, allowed_hosts: frozenset[str]) -> None:
        super().__init__()
        self.allowed_hosts = allowed_hosts
        self.redirects: list[dict[str, Any]] = []

    def redirect_request(
        self,
        req: Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        _validate_https_url(newurl, allowed_hosts=self.allowed_hosts)
        self.redirects.append(
            {
                "from_url": req.full_url,
                "status": code,
                "to_url": newurl,
            }
        )
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and req.get_method() == "HEAD":
            redirected = Request(
                redirected.full_url,
                headers=dict(redirected.header_items()),
                origin_req_host=req.origin_req_host,
                unverifiable=True,
                method="HEAD",
            )
        return redirected


def _validate_https_url(url: str, *, allowed_hosts: frozenset[str]) -> None:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in allowed_hosts
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ModelAssetError(f"asset URL is outside the fixed HTTPS boundary: {url}")


def _validate_external_directory(artifact_dir: Path, project_root: Path) -> Path:
    project_root = project_root.resolve()
    resolved = artifact_dir.resolve()
    try:
        relative = resolved.relative_to(project_root)
    except ValueError:
        return resolved
    if relative.parts[:2] != ("data", "external"):
        raise ModelAssetError(
            "model assets inside the repository must be under data/external"
        )
    return resolved


def _validated_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ModelAssetError("acquisition date must use YYYY-MM-DD") from error
    if parsed.isoformat() != value:
        raise ModelAssetError("acquisition date must use canonical YYYY-MM-DD")
    return value


def _validated_commit(value: str, *, field: str) -> str:
    if not GIT_COMMIT_PATTERN.fullmatch(value):
        raise ModelAssetError(f"{field} must be a full lowercase Git commit")
    return value


def _response_metadata(response: Any, redirects: list[dict[str, Any]]) -> dict[str, Any]:
    raw_length = response.headers.get("Content-Length")
    content_length = int(raw_length) if raw_length and raw_length.isdecimal() else None
    return {
        "accept_ranges": response.headers.get("Accept-Ranges"),
        "content_length": content_length,
        "content_type": response.headers.get("Content-Type"),
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
        "redirect_chain": redirects,
        "resolved_url": response.geturl(),
        "status": response.getcode(),
    }


def _head_metadata(
    url: str,
    *,
    allowed_hosts: frozenset[str],
) -> dict[str, Any]:
    _validate_https_url(url, allowed_hosts=allowed_hosts)
    redirect_handler = _RestrictedRedirectHandler(allowed_hosts)
    opener = build_opener(redirect_handler)
    request = Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    with opener.open(request, timeout=60) as response:
        _validate_https_url(response.geturl(), allowed_hosts=allowed_hosts)
        metadata = _response_metadata(response, redirect_handler.redirects)
    if metadata["status"] != 200:
        raise ModelAssetError(f"unexpected HTTP status for {url}: {metadata['status']}")
    return metadata


def _download_to_partial(
    url: str,
    *,
    artifact_dir: Path,
    prefix: str,
    allowed_hosts: frozenset[str],
    expected_bytes: int | None,
) -> tuple[Path, dict[str, Any]]:
    _validate_https_url(url, allowed_hosts=allowed_hosts)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    digest = hashlib.sha256()
    byte_count = 0
    redirect_handler = _RestrictedRedirectHandler(allowed_hosts)
    opener = build_opener(redirect_handler)
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=artifact_dir,
            prefix=f".{prefix}.",
            suffix=".part",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with opener.open(request, timeout=180) as response:
                _validate_https_url(response.geturl(), allowed_hosts=allowed_hosts)
                response_metadata = _response_metadata(
                    response,
                    redirect_handler.redirects,
                )
                if response_metadata["status"] != 200:
                    raise ModelAssetError(
                        f"unexpected download status: {response_metadata['status']}"
                    )
                while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                    output.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
        if byte_count <= 0:
            raise ModelAssetError("downloaded asset is empty")
        if expected_bytes is not None and byte_count != expected_bytes:
            raise ModelAssetError(
                f"downloaded byte count changed: expected {expected_bytes}, "
                f"observed {byte_count}"
            )
        response_metadata.update(
            {
                "byte_count": byte_count,
                "observed_sha256": digest.hexdigest(),
            }
        )
        return temporary_path, response_metadata
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _validated_posix_parts(name: str, *, asset: str) -> tuple[str, ...]:
    if not name or "\\" in name or "\x00" in name:
        raise ModelAssetError(f"unsafe {asset} member name: {name!r}")
    pure_path = PurePosixPath(name)
    raw_parts = tuple(name.rstrip("/").split("/"))
    if (
        pure_path.is_absolute()
        or not raw_parts
        or any(part in {"", ".", ".."} for part in raw_parts)
        or ":" in raw_parts[0]
    ):
        raise ModelAssetError(f"unsafe {asset} member path: {name!r}")
    return raw_parts


def _sha256_stream(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    while chunk := stream.read(MEMBER_CHUNK_SIZE):
        digest.update(chunk)
        byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def inspect_source_archive(path: Path) -> dict[str, Any]:
    """Inspect the fixed GitHub source snapshot without extracting or executing it."""
    names: set[str] = set()
    casefolded_names: set[str] = set()
    required: dict[str, dict[str, Any]] = {}
    license_material: list[dict[str, Any]] = []
    file_count = 0
    directory_count = 0
    total_file_bytes = 0

    try:
        with tarfile.open(path, mode="r:gz") as archive:
            for index, member in enumerate(archive, start=1):
                if index > MAX_SOURCE_MEMBERS:
                    raise ModelAssetError("source archive has too many members")
                parts = _validated_posix_parts(member.name, asset="source archive")
                if parts[0] != SOURCE_ROOT:
                    raise ModelAssetError(
                        f"source archive has an unexpected top-level path: {member.name!r}"
                    )
                normalized_name = "/".join(parts)
                if (
                    normalized_name in names
                    or normalized_name.casefold() in casefolded_names
                ):
                    raise ModelAssetError(
                        f"source archive has a duplicate target: {normalized_name!r}"
                    )
                names.add(normalized_name)
                casefolded_names.add(normalized_name.casefold())

                if member.isdir():
                    directory_count += 1
                    continue
                if not member.isreg():
                    raise ModelAssetError(
                        f"source archive has an unsupported member type: {member.name!r}"
                    )
                if member.size < 0 or member.size > MAX_SOURCE_MEMBER_BYTES:
                    raise ModelAssetError(
                        f"source archive member is too large: {member.name!r}"
                    )
                file_count += 1
                total_file_bytes += member.size
                if total_file_bytes > MAX_SOURCE_TOTAL_BYTES:
                    raise ModelAssetError("source archive expands beyond the fixed limit")

                relative_name = "/".join(parts[1:])
                source = archive.extractfile(member)
                if source is None:
                    raise ModelAssetError(
                        f"cannot read source archive member: {member.name!r}"
                    )
                with source:
                    member_sha256, observed_size = _sha256_stream(source)
                if observed_size != member.size:
                    raise ModelAssetError(
                        f"source member size changed while reading: {member.name!r}"
                    )
                member_record = {
                    "byte_count": observed_size,
                    "path": relative_name,
                    "sha256": member_sha256,
                }
                if relative_name in REQUIRED_SOURCE_FILES:
                    required[relative_name] = member_record
                basename = PurePosixPath(relative_name).name.casefold()
                if basename.startswith(("license", "notice")):
                    license_material.append(member_record)
    except (tarfile.TarError, OSError) as error:
        raise ModelAssetError(f"cannot inspect source archive: {error}") from error

    missing = sorted(REQUIRED_SOURCE_FILES - set(required))
    if missing:
        raise ModelAssetError(f"source archive is missing required files: {missing}")

    with tarfile.open(path, mode="r:gz") as archive:
        root_license = archive.extractfile(f"{SOURCE_ROOT}/LICENSE")
        readme = archive.extractfile(f"{SOURCE_ROOT}/README.md")
        backbones = archive.extractfile(f"{SOURCE_ROOT}/dinov2/hub/backbones.py")
        if root_license is None or readme is None or backbones is None:
            raise ModelAssetError("source archive required content cannot be read")
        with root_license, readme, backbones:
            license_text = root_license.read().decode("utf-8")
            readme_text = readme.read().decode("utf-8")
            backbones_text = backbones.read().decode("utf-8")

    if (
        "Apache License" not in license_text
        or "Version 2.0, January 2004" not in license_text
        or "http://www.apache.org/licenses/" not in license_text
    ):
        raise ModelAssetError("root source license is not Apache License 2.0")
    if (
        "DINOv2 code and model weights are released under the Apache License 2.0"
        not in readme_text
    ):
        raise ModelAssetError("standard model-weight license statement is missing")
    if (
        "def dinov2_vits14(" not in backbones_text
        or "_DINOV2_BASE_URL" not in backbones_text
        or "_pretrain.pth" not in backbones_text
    ):
        raise ModelAssetError("fixed non-register ViT-S/14 entry point is missing")

    return {
        "archive_format": "tar.gz",
        "all_members_validated": True,
        "directory_count": directory_count,
        "file_count": file_count,
        "license_material": sorted(license_material, key=lambda item: item["path"]),
        "member_count": file_count + directory_count,
        "required_files": [required[name] for name in sorted(required)],
        "safe_structure": "pass",
        "top_level_directory": SOURCE_ROOT,
        "total_file_bytes": total_file_bytes,
    }


def _validated_zip_member(info: zipfile.ZipInfo) -> tuple[str, ...]:
    parts = _validated_posix_parts(info.filename, asset="checkpoint")
    if info.flag_bits & 0x1:
        raise ModelAssetError(f"encrypted checkpoint member is forbidden: {info.filename!r}")
    if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
        raise ModelAssetError(
            f"unsupported checkpoint compression: {info.filename!r}"
        )
    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    allowed_types = {0, stat.S_IFREG}
    if info.is_dir():
        allowed_types.add(stat.S_IFDIR)
    if file_type not in allowed_types:
        raise ModelAssetError(
            f"unsupported checkpoint member type: {info.filename!r}"
        )
    if info.file_size < 0 or info.file_size > MAX_CHECKPOINT_MEMBER_BYTES:
        raise ModelAssetError(f"checkpoint member is too large: {info.filename!r}")
    return parts


def _inspect_checkpoint_pickle(content: bytes) -> dict[str, Any]:
    if not content or len(content) > MAX_PICKLE_BYTES:
        raise ModelAssetError("checkpoint data.pkl size is outside the fixed limit")
    globals_seen: set[str] = set()
    state_keys: set[str] = set()
    opcode_count = 0
    protocol: int | None = None
    try:
        for opcode, argument, _ in pickletools.genops(content):
            opcode_count += 1
            if opcode.name == "PROTO":
                protocol = argument
            if opcode.name in FORBIDDEN_PICKLE_OPCODES:
                raise ModelAssetError(
                    f"checkpoint pickle uses unsupported opcode: {opcode.name}"
                )
            if opcode.name == "GLOBAL":
                if not isinstance(argument, str):
                    raise ModelAssetError("checkpoint GLOBAL opcode has no text target")
                globals_seen.add(argument)
            if (
                opcode.name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}
                and isinstance(argument, str)
                and ("." in argument or argument in EXPECTED_STATE_KEYS)
            ):
                state_keys.add(argument)
    except ValueError as error:
        raise ModelAssetError(f"checkpoint pickle is malformed: {error}") from error
    if (
        "collections OrderedDict" not in globals_seen
        or not globals_seen.issubset(EXPECTED_CHECKPOINT_GLOBALS)
    ):
        raise ModelAssetError(
            "checkpoint pickle globals differ from the fixed allowlist: "
            f"{sorted(globals_seen)}"
        )
    missing_keys = sorted(EXPECTED_STATE_KEYS - state_keys)
    if missing_keys:
        raise ModelAssetError(
            f"checkpoint pickle lacks expected state-key markers: {missing_keys}"
        )
    state_key_manifest = "\n".join(sorted(state_keys)).encode()
    return {
        "candidate_state_key_count": len(state_keys),
        "candidate_state_key_manifest_sha256": hashlib.sha256(
            state_key_manifest
        ).hexdigest(),
        "global_references": sorted(globals_seen),
        "opcode_count": opcode_count,
        "pickle_deserialized": False,
        "protocol": protocol,
        "required_state_key_markers": sorted(EXPECTED_STATE_KEYS),
    }


def inspect_checkpoint_archive(path: Path) -> dict[str, Any]:
    """Inspect the PyTorch ZIP container and pickle opcodes without deserialization."""
    names: set[str] = set()
    casefolded_names: set[str] = set()
    root_names: set[str] = set()
    file_count = 0
    directory_count = 0
    total_file_bytes = 0
    data_member_count = 0
    pickle_path: str | None = None
    byteorder_member_present = False
    version_member_present = False
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > MAX_CHECKPOINT_MEMBERS:
                raise ModelAssetError(
                    f"invalid checkpoint member count: {len(infos)}"
                )
            for info in infos:
                parts = _validated_zip_member(info)
                root_names.add(parts[0])
                normalized_name = "/".join(parts)
                if (
                    normalized_name in names
                    or normalized_name.casefold() in casefolded_names
                ):
                    raise ModelAssetError(
                        f"checkpoint has a duplicate target: {normalized_name!r}"
                    )
                names.add(normalized_name)
                casefolded_names.add(normalized_name.casefold())
                if info.is_dir():
                    directory_count += 1
                    continue
                file_count += 1
                total_file_bytes += info.file_size
                if total_file_bytes > MAX_CHECKPOINT_TOTAL_BYTES:
                    raise ModelAssetError("checkpoint expands beyond the fixed limit")
                relative_name = "/".join(parts[1:])
                if relative_name == "data.pkl":
                    if pickle_path is not None:
                        raise ModelAssetError("checkpoint has multiple data.pkl members")
                    pickle_path = normalized_name
                elif relative_name == "byteorder":
                    byteorder_member_present = True
                elif relative_name == "version":
                    version_member_present = True
                elif relative_name.startswith("data/") and len(parts) == 3:
                    data_member_count += 1

            if len(root_names) != 1:
                raise ModelAssetError(
                    f"checkpoint must have one top-level directory: {sorted(root_names)}"
                )
            if pickle_path is None or not version_member_present:
                raise ModelAssetError("checkpoint lacks required PyTorch ZIP members")
            if data_member_count <= 0:
                raise ModelAssetError("checkpoint has no tensor-storage members")
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise ModelAssetError(
                    f"checkpoint ZIP CRC failed: {corrupt_member!r}"
                )
            pickle_info = archive.getinfo(pickle_path)
            if pickle_info.file_size > MAX_PICKLE_BYTES:
                raise ModelAssetError("checkpoint data.pkl exceeds the fixed limit")
            pickle_record = _inspect_checkpoint_pickle(archive.read(pickle_path))
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ModelAssetError(f"cannot inspect checkpoint ZIP: {error}") from error

    return {
        "all_members_validated": True,
        "archive_format": "pytorch_zip",
        "byteorder_member_present": byteorder_member_present,
        "crc_verification": "pass",
        "data_member_count": data_member_count,
        "directory_count": directory_count,
        "file_count": file_count,
        "member_count": file_count + directory_count,
        "pickle": pickle_record,
        "safe_structure": "pass",
        "top_level_directory": next(iter(root_names)),
        "total_file_bytes": total_file_bytes,
        "version_member_present": version_member_present,
    }


def _hash_addressed_name(prefix: str, digest: str, suffix: str) -> str:
    if not SHA256_PATTERN.fullmatch(digest):
        raise ModelAssetError("cannot create an asset name from an invalid SHA-256")
    return f"{prefix}-sha256-{digest}{suffix}"


def acquire_model_assets(
    *,
    artifact_dir: Path,
    output_path: Path,
    project_root: Path,
    acquisition_date: str,
    acquisition_base_commit: str,
    preregistration_commit: str,
    preregistration_path: Path,
) -> dict[str, Any]:
    """Acquire both fixed assets and commit only a deterministic provenance record."""
    acquisition_date = _validated_date(acquisition_date)
    acquisition_base_commit = _validated_commit(
        acquisition_base_commit,
        field="acquisition base commit",
    )
    preregistration_commit = _validated_commit(
        preregistration_commit,
        field="preregistration commit",
    )
    artifact_dir = _validate_external_directory(artifact_dir, project_root)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if not preregistration_path.is_file():
        raise ModelAssetError("preregistration document is missing")

    source_head = _head_metadata(SOURCE_URL, allowed_hosts=SOURCE_HOSTS)
    checkpoint_head = _head_metadata(
        CHECKPOINT_URL,
        allowed_hosts=CHECKPOINT_HOSTS,
    )
    if checkpoint_head["content_length"] != EXPECTED_CHECKPOINT_BYTES:
        raise ModelAssetError(
            "checkpoint HEAD byte count changed: "
            f"expected {EXPECTED_CHECKPOINT_BYTES}, "
            f"observed {checkpoint_head['content_length']}"
        )

    source_partial: Path | None = None
    checkpoint_partial: Path | None = None
    committed_paths: list[Path] = []
    try:
        source_partial, source_download = _download_to_partial(
            SOURCE_URL,
            artifact_dir=artifact_dir,
            prefix="dinov2-source",
            allowed_hosts=SOURCE_HOSTS,
            expected_bytes=None,
        )
        source_inspection = inspect_source_archive(source_partial)

        checkpoint_partial, checkpoint_download = _download_to_partial(
            CHECKPOINT_URL,
            artifact_dir=artifact_dir,
            prefix="dinov2-vits14-checkpoint",
            allowed_hosts=CHECKPOINT_HOSTS,
            expected_bytes=EXPECTED_CHECKPOINT_BYTES,
        )
        checkpoint_inspection = inspect_checkpoint_archive(checkpoint_partial)

        source_name = _hash_addressed_name(
            f"dinov2-source-{SOURCE_REVISION}",
            source_download["observed_sha256"],
            ".tar.gz",
        )
        checkpoint_name = _hash_addressed_name(
            "dinov2_vits14_pretrain",
            checkpoint_download["observed_sha256"],
            ".pth",
        )
        source_destination = artifact_dir / source_name
        checkpoint_destination = artifact_dir / checkpoint_name
        if source_destination.exists() or checkpoint_destination.exists():
            raise FileExistsError("refusing to overwrite an accepted model asset")

        os.replace(source_partial, source_destination)
        source_partial = None
        committed_paths.append(source_destination)
        os.replace(checkpoint_partial, checkpoint_destination)
        checkpoint_partial = None
        committed_paths.append(checkpoint_destination)

        report = {
            "acquisition": {
                "acquisition_base_commit": acquisition_base_commit,
                "acquisition_date": acquisition_date,
                "external_cache": "data/external/v0.2/model-assets",
                "preregistration_commit": preregistration_commit,
                "preregistration_document": (
                    "docs/v0.2-preflight-preregistration.md"
                ),
                "preregistration_document_sha256": sha256_file(
                    preregistration_path
                ),
                "preregistration_id": "v0.2-dinov2-cpu-preflight-1",
            },
            "boundary": {
                "checkpoint_acquired": True,
                "checkpoint_deserialized": False,
                "checkpoint_pickle_executed": False,
                "checkpoint_tensor_values_inspected": False,
                "dataset_access": False,
                "model_constructed": False,
                "model_inference_performed": False,
                "source_acquired": True,
                "source_executed": False,
                "source_extracted": False,
                "tensor_operation_performed": False,
            },
            "checkpoint": {
                "artifact": {
                    "byte_count": checkpoint_download["byte_count"],
                    "checksum_status": "observed_only",
                    "filename": checkpoint_name,
                    "observed_sha256": checkpoint_download["observed_sha256"],
                    "published_sha256": None,
                    "storage": "outside_git",
                },
                "identity": {
                    "architecture": "ViT-S/14",
                    "model_identifier": "dinov2_vits14",
                    "pretraining_identity": "LVD142M",
                    "register_tokens": False,
                },
                "inspection": checkpoint_inspection,
                "license": {
                    "identifier": "Apache-2.0",
                    "source_license_path": "LICENSE",
                    "weight_license_statement_path": "README.md",
                },
                "transport": {
                    "download": checkpoint_download,
                    "head": checkpoint_head,
                    "requested_url": CHECKPOINT_URL,
                },
            },
            "decision": {
                "next_step": "PROCEED_TO_WEIGHTS_ONLY_STRICT_LOAD_VERIFICATION",
                "reason": (
                    "The fixed source and checkpoint were acquired outside Git, "
                    "their transport and observed identities were recorded, and "
                    "their containers passed non-executing structural checks."
                ),
            },
            "schema_version": OUTPUT_SCHEMA,
            "source": {
                "artifact": {
                    "byte_count": source_download["byte_count"],
                    "checksum_status": "observed_only",
                    "filename": source_name,
                    "observed_sha256": source_download["observed_sha256"],
                    "published_sha256": None,
                    "storage": "outside_git",
                },
                "identity": {
                    "project": "facebookresearch/dinov2",
                    "revision": SOURCE_REVISION,
                },
                "inspection": source_inspection,
                "license": {
                    "identifier": "Apache-2.0",
                    "root_license_path": "LICENSE",
                    "standard_model_weight_statement_path": "README.md",
                },
                "transport": {
                    "download": source_download,
                    "head": source_head,
                    "requested_url": SOURCE_URL,
                },
            },
            "summary": {
                "asset_count": 2,
                "checkpoint_archive_safe": True,
                "checkpoint_byte_count": checkpoint_download["byte_count"],
                "checkpoint_checksum_status": "observed_only",
                "source_archive_safe": True,
                "source_byte_count": source_download["byte_count"],
                "source_checksum_status": "observed_only",
            },
        }
        write_json_atomic(output_path, report)
        return report
    except Exception:
        if source_partial is not None:
            source_partial.unlink(missing_ok=True)
        if checkpoint_partial is not None:
            checkpoint_partial.unlink(missing_ok=True)
        if not output_path.exists():
            for path in committed_paths:
                path.unlink(missing_ok=True)
        raise
