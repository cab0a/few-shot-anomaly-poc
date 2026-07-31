"""Verify weights-only DINOv2 checkpoint and strict local-source compatibility."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import site
import socket
import sys
import sysconfig
import warnings
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from types import ModuleType
from typing import Any

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.jsonio import write_json_atomic
from few_shot_anomaly_poc.model_assets import (
    OUTPUT_SCHEMA as ACQUISITION_SCHEMA,
)
from few_shot_anomaly_poc.model_assets import (
    SOURCE_REVISION,
    SOURCE_ROOT,
    extract_source_archive,
)

OUTPUT_SCHEMA = "v0.2-weights-only-strict-load-v1"
IMPORT_SMOKE_SCHEMA = "v0.2-isolated-import-smoke-v1"
EXPECTED_PYTHON_VERSION = "3.13.14"
EXPECTED_TORCH_VERSION = "2.13.0+cpu"
EXPECTED_MODEL_ENTRY_POINT = "dinov2.hub.backbones.dinov2_vits14"
EXPECTED_SOURCE_SHA256 = "c27dcdaf50e9fb5bbdf2bb529da357716372e19c6afab17d5350f3f0094aed4b"
EXPECTED_CHECKPOINT_SHA256 = (
    "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
)
EXPECTED_CHECKPOINT_BYTES = 88_283_115
EXPECTED_STATE_KEY_COUNT = 175
EXPECTED_STATE_KEY_MANIFEST_SHA256 = (
    "21dec8566e545b724414a5881a72aa9590525cb81b8b38d416d21e7952eff0f1"
)
EXPECTED_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "MKL_NUM_THREADS": "4",
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "XFORMERS_DISABLED": "1",
}
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class ModelCompatibilityError(Exception):
    """Reject an environment, state dictionary, or model outside the fixed boundary."""


class NetworkGuard(AbstractContextManager["NetworkGuard"]):
    """Block socket-based network access while source and checkpoint are executed."""

    def __init__(self) -> None:
        self.attempts: list[str] = []
        self._original_create_connection: Any = None
        self._original_getaddrinfo: Any = None
        self._original_connect: Any = None
        self._original_connect_ex: Any = None

    def _blocked(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.attempts.append("blocked_socket_operation")
        raise ModelCompatibilityError("network access is forbidden during strict-load verification")

    def __enter__(self) -> NetworkGuard:
        self._original_create_connection = socket.create_connection
        self._original_getaddrinfo = socket.getaddrinfo
        self._original_connect = socket.socket.connect
        self._original_connect_ex = socket.socket.connect_ex
        socket.create_connection = self._blocked
        socket.getaddrinfo = self._blocked
        socket.socket.connect = self._blocked
        socket.socket.connect_ex = self._blocked
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        socket.create_connection = self._original_create_connection
        socket.getaddrinfo = self._original_getaddrinfo
        socket.socket.connect = self._original_connect
        socket.socket.connect_ex = self._original_connect_ex
        return None


def _require_mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelCompatibilityError(f"{field} must be a mapping")
    return value


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelCompatibilityError(f"{field} must be a non-empty string")
    return value


def _load_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelCompatibilityError(f"cannot read {field}: {error}") from error
    return _require_mapping(value, field=field)


def _validate_asset_filename(value: object, *, field: str) -> str:
    filename = _require_string(value, field=field)
    if Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ModelCompatibilityError(f"{field} must be a basename")
    return filename


def _validate_commit(value: str) -> str:
    if not GIT_COMMIT_PATTERN.fullmatch(value):
        raise ModelCompatibilityError("execution commit must be a full lowercase Git commit")
    return value


def _validate_date(value: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ModelCompatibilityError("verification date must use YYYY-MM-DD")
    return value


def _state_key_manifest(keys: list[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


def _load_acquisition_record(path: Path) -> dict[str, Any]:
    record = _load_json(path, field="model-asset acquisition record")
    if (
        record.get("schema_version") != ACQUISITION_SCHEMA
        or record.get("decision", {}).get("next_step")
        != "PROCEED_TO_WEIGHTS_ONLY_STRICT_LOAD_VERIFICATION"
    ):
        raise ModelCompatibilityError(
            "model-asset acquisition record does not authorize strict-load verification"
        )
    source = _require_mapping(record.get("source"), field="source")
    checkpoint = _require_mapping(record.get("checkpoint"), field="checkpoint")
    source_artifact = _require_mapping(source.get("artifact"), field="source.artifact")
    checkpoint_artifact = _require_mapping(
        checkpoint.get("artifact"),
        field="checkpoint.artifact",
    )
    checkpoint_inspection = _require_mapping(
        checkpoint.get("inspection"),
        field="checkpoint.inspection",
    )
    pickle_inspection = _require_mapping(
        checkpoint_inspection.get("pickle"),
        field="checkpoint.inspection.pickle",
    )
    if (
        source_artifact.get("observed_sha256") != EXPECTED_SOURCE_SHA256
        or source_artifact.get("checksum_status") != "observed_only"
        or checkpoint_artifact.get("observed_sha256") != EXPECTED_CHECKPOINT_SHA256
        or checkpoint_artifact.get("byte_count") != EXPECTED_CHECKPOINT_BYTES
        or checkpoint_artifact.get("checksum_status") != "observed_only"
        or pickle_inspection.get("candidate_state_key_count")
        != EXPECTED_STATE_KEY_COUNT
        or pickle_inspection.get("candidate_state_key_manifest_sha256")
        != EXPECTED_STATE_KEY_MANIFEST_SHA256
        or pickle_inspection.get("pickle_deserialized") is not False
    ):
        raise ModelCompatibilityError("model-asset acquisition identity has changed")
    return record


def _load_import_smoke_record(path: Path) -> dict[str, Any]:
    record = _load_json(path, field="isolated import-smoke record")
    if (
        record.get("schema_version") != IMPORT_SMOKE_SCHEMA
        or record.get("torch", {}).get("module_version") != EXPECTED_TORCH_VERSION
        or record.get("summary", {}).get("distribution_count") != 17
    ):
        raise ModelCompatibilityError("isolated import-smoke identity has changed")
    return record


def _installed_distributions() -> dict[str, str]:
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not isinstance(name, str) or not name:
            raise ModelCompatibilityError("installed distribution has no valid name")
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        if canonical in installed:
            raise ModelCompatibilityError(f"duplicate installed distribution: {canonical}")
        installed[canonical] = distribution.version
    return installed


def _validate_environment(
    *,
    environment_root: Path,
    import_smoke_record: dict[str, Any],
) -> dict[str, Any]:
    environment_root = environment_root.resolve()
    if Path(sys.prefix).resolve() != environment_root:
        raise ModelCompatibilityError("sys.prefix is not the requested v0.2 environment")
    if Path(sys.base_prefix).resolve() == environment_root:
        raise ModelCompatibilityError("interpreter is not running from a virtual environment")
    if (
        platform.python_version() != EXPECTED_PYTHON_VERSION
        or sys.platform != "linux"
        or platform.machine() != "x86_64"
        or sys.byteorder != "little"
    ):
        raise ModelCompatibilityError("interpreter identity is outside the fixed boundary")
    if not sys.flags.isolated or not sys.dont_write_bytecode or site.ENABLE_USER_SITE is not False:
        raise ModelCompatibilityError("Python isolation flags are incomplete")
    mismatched_environment = {
        name: {"expected": expected, "observed": os.environ.get(name)}
        for name, expected in EXPECTED_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if mismatched_environment:
        raise ModelCompatibilityError(
            f"required process environment differs: {mismatched_environment}"
        )

    expected_packages = {
        item["name"]: item["version"]
        for item in import_smoke_record.get("packages", [])
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and isinstance(item.get("version"), str)
    }
    installed_packages = _installed_distributions()
    if installed_packages != expected_packages:
        raise ModelCompatibilityError("installed distributions differ from import smoke")
    return {
        "environment_path": "environments/v0.2-preflight/.venv",
        "implementation": platform.python_implementation(),
        "isolated_mode": True,
        "platform": sysconfig.get_platform(),
        "python_version": platform.python_version(),
        "system_byteorder": sys.byteorder,
        "user_site_enabled": False,
    }


def _verify_asset(
    *,
    path: Path,
    expected_sha256: str,
    expected_bytes: int,
    field: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise ModelCompatibilityError(f"{field} is missing")
    byte_count = path.stat().st_size
    if byte_count != expected_bytes:
        raise ModelCompatibilityError(
            f"{field} byte count changed: expected {expected_bytes}, observed {byte_count}"
        )
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ModelCompatibilityError(
            f"{field} checksum changed: expected {expected_sha256}, "
            f"observed {observed_sha256}"
        )
    return {
        "byte_count": byte_count,
        "sha256": observed_sha256,
        "verification": "pass",
    }


def _module_origins(source_root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for name, module in sorted(sys.modules.items()):
        if name != "dinov2" and not name.startswith("dinov2."):
            continue
        raw_origin = getattr(module, "__file__", None)
        if not isinstance(raw_origin, str) or not raw_origin:
            raise ModelCompatibilityError(f"imported DINOv2 module has no origin: {name}")
        origin = Path(raw_origin).resolve()
        try:
            relative = origin.relative_to(source_root)
        except ValueError as error:
            raise ModelCompatibilityError(
                f"DINOv2 module loaded outside fixed source: {name}"
            ) from error
        records.append({"module": name, "origin": relative.as_posix()})
    if not records:
        raise ModelCompatibilityError("no DINOv2 module origin was recorded")
    return records


def _summarize_state_dict(torch: ModuleType, state: object) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise ModelCompatibilityError("checkpoint root object is not a mapping")
    keys = list(state)
    if (
        len(keys) != EXPECTED_STATE_KEY_COUNT
        or any(not isinstance(key, str) or not key for key in keys)
        or len(set(keys)) != len(keys)
    ):
        raise ModelCompatibilityError("checkpoint state keys differ from the fixed count")
    manifest_sha256 = _state_key_manifest(keys)
    if manifest_sha256 != EXPECTED_STATE_KEY_MANIFEST_SHA256:
        raise ModelCompatibilityError("checkpoint state-key manifest has changed")

    tensors: list[dict[str, Any]] = []
    dtype_counts: dict[str, int] = {}
    device_counts: dict[str, int] = {}
    total_elements = 0
    total_bytes = 0
    for key in sorted(keys):
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            raise ModelCompatibilityError(f"checkpoint value is not a tensor: {key}")
        device = str(tensor.device)
        dtype = str(tensor.dtype).removeprefix("torch.")
        if device != "cpu":
            raise ModelCompatibilityError(f"checkpoint tensor is not on CPU: {key}")
        if tensor.dtype is not torch.float32:
            raise ModelCompatibilityError(f"checkpoint tensor is not float32: {key}")
        if tensor.layout is not torch.strided:
            raise ModelCompatibilityError(f"checkpoint tensor is not strided: {key}")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ModelCompatibilityError(f"checkpoint tensor has a non-finite value: {key}")
        element_count = tensor.numel()
        byte_count = element_count * tensor.element_size()
        total_elements += element_count
        total_bytes += byte_count
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        device_counts[device] = device_counts.get(device, 0) + 1
        tensors.append(
            {
                "byte_count": byte_count,
                "device": device,
                "dtype": dtype,
                "finite": True,
                "key": key,
                "numel": element_count,
                "shape": list(tensor.shape),
            }
        )
    return {
        "device_counts": dict(sorted(device_counts.items())),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "finite_tensor_count": len(tensors),
        "key_count": len(keys),
        "root_type": f"{type(state).__module__}.{type(state).__qualname__}",
        "state_key_manifest_sha256": manifest_sha256,
        "tensor_count": len(tensors),
        "tensors": tensors,
        "total_tensor_bytes": total_bytes,
        "total_tensor_elements": total_elements,
    }


def _model_summary(torch: ModuleType, model: Any, state: Mapping[str, Any]) -> dict[str, Any]:
    model.eval()
    load_result = model.load_state_dict(state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ModelCompatibilityError("strict state-dictionary load reported incompatible keys")

    model_state = model.state_dict()
    checkpoint_keys = sorted(state)
    if sorted(model_state) != checkpoint_keys:
        raise ModelCompatibilityError("model state keys differ after strict load")
    exact_value_matches = 0
    for key in checkpoint_keys:
        model_tensor = model_state[key]
        checkpoint_tensor = state[key]
        if (
            model_tensor.device.type != "cpu"
            or model_tensor.dtype is not torch.float32
            or tuple(model_tensor.shape) != tuple(checkpoint_tensor.shape)
            or not bool(torch.equal(model_tensor, checkpoint_tensor))
        ):
            raise ModelCompatibilityError(f"loaded model tensor differs: {key}")
        exact_value_matches += 1

    if (
        model.embed_dim != 384
        or model.patch_size != 14
        or model.n_blocks != 12
        or model.num_heads != 6
        or model.num_register_tokens != 0
        or model.register_tokens is not None
    ):
        raise ModelCompatibilityError("constructed model is not fixed non-register ViT-S/14")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    buffer_count = sum(buffer.numel() for buffer in model.buffers())
    return {
        "buffer_element_count": buffer_count,
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "embedding_dimension": model.embed_dim,
        "entry_point": EXPECTED_MODEL_ENTRY_POINT,
        "eval_mode": model.training is False,
        "exact_value_match_count": exact_value_matches,
        "missing_keys": list(load_result.missing_keys),
        "num_heads": model.num_heads,
        "num_register_tokens": model.num_register_tokens,
        "parameter_count": parameter_count,
        "patch_size": model.patch_size,
        "register_tokens_is_none": model.register_tokens is None,
        "state_key_count": len(model_state),
        "strict_load": "pass",
        "trainable_parameter_count": trainable_parameter_count,
        "transformer_block_count": model.n_blocks,
        "unexpected_keys": list(load_result.unexpected_keys),
    }


def verify_model_compatibility(
    *,
    acquisition_path: Path,
    import_smoke_path: Path,
    artifact_dir: Path,
    extraction_dir: Path,
    environment_root: Path,
    project_root: Path,
    execution_commit: str,
    verification_date: str,
    output_path: Path,
) -> dict[str, Any]:
    """Run the fixed weights-only and strict-load verification without inference."""
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    execution_commit = _validate_commit(execution_commit)
    verification_date = _validate_date(verification_date)
    acquisition_record = _load_acquisition_record(acquisition_path)
    import_smoke_record = _load_import_smoke_record(import_smoke_path)
    environment = _validate_environment(
        environment_root=environment_root,
        import_smoke_record=import_smoke_record,
    )

    source_artifact = acquisition_record["source"]["artifact"]
    checkpoint_artifact = acquisition_record["checkpoint"]["artifact"]
    source_filename = _validate_asset_filename(
        source_artifact.get("filename"),
        field="source filename",
    )
    checkpoint_filename = _validate_asset_filename(
        checkpoint_artifact.get("filename"),
        field="checkpoint filename",
    )
    source_path = artifact_dir / source_filename
    checkpoint_path = artifact_dir / checkpoint_filename
    source_identity = _verify_asset(
        path=source_path,
        expected_sha256=EXPECTED_SOURCE_SHA256,
        expected_bytes=source_artifact["byte_count"],
        field="source archive",
    )
    checkpoint_identity = _verify_asset(
        path=checkpoint_path,
        expected_sha256=EXPECTED_CHECKPOINT_SHA256,
        expected_bytes=EXPECTED_CHECKPOINT_BYTES,
        field="checkpoint",
    )

    expected_extraction_name = f"dinov2-source-sha256-{EXPECTED_SOURCE_SHA256}"
    if extraction_dir.name != expected_extraction_name:
        raise ModelCompatibilityError(
            f"extraction directory must be hash-addressed as {expected_extraction_name}"
        )
    if extraction_dir.resolve().parent != artifact_dir.resolve():
        raise ModelCompatibilityError(
            "extraction directory must be a direct child of the model-asset cache"
        )
    extraction_created = False
    try:
        extraction = extract_source_archive(
            source_path,
            expected_sha256=EXPECTED_SOURCE_SHA256,
            destination=extraction_dir,
            project_root=project_root,
        )
        extraction_created = True
        source_root = (extraction_dir / SOURCE_ROOT).resolve()
        if not source_root.is_dir():
            raise ModelCompatibilityError("extracted DINOv2 import root is missing")

        previous_sys_path = list(sys.path)
        for name in tuple(sys.modules):
            if name == "dinov2" or name.startswith("dinov2."):
                raise ModelCompatibilityError("DINOv2 was imported before the network guard")

        with NetworkGuard() as network_guard:
            try:
                sys.path.insert(0, str(source_root))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    torch = importlib.import_module("torch")
                    backbones = importlib.import_module("dinov2.hub.backbones")
                if (
                    torch.__version__ != EXPECTED_TORCH_VERSION
                    or torch.version.cuda is not None
                    or torch.version.hip is not None
                ):
                    raise ModelCompatibilityError("PyTorch is not the fixed CPU build")
                torch.set_num_threads(4)
                torch.set_num_interop_threads(1)
                state = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    mmap=True,
                    weights_only=True,
                )
                state_summary = _summarize_state_dict(torch, state)
                torch.manual_seed(42)
                model = backbones.dinov2_vits14(pretrained=False)
                model_summary = _model_summary(torch, model, state)
                module_origins = _module_origins(source_root)
                if "xformers" in sys.modules or any(
                    name.startswith("xformers.") for name in sys.modules
                ):
                    raise ModelCompatibilityError("xformers was imported unexpectedly")
            finally:
                sys.path[:] = previous_sys_path

        if network_guard.attempts:
            raise ModelCompatibilityError("a network operation was attempted")

        report = {
            "boundary": {
                "accelerator_runtime_probe_performed": False,
                "checkpoint_deserialized": True,
                "checkpoint_pickle_executed_by_weights_only_loader": True,
                "checkpoint_tensor_values_inspected": True,
                "dataset_access": False,
                "feature_extraction_performed": False,
                "latency_measurement_performed": False,
                "model_constructed": True,
                "model_inference_performed": False,
                "network_access": False,
                "source_executed": True,
                "source_extracted": True,
                "synthetic_workload_generated": False,
                "tensor_operations_performed": True,
            },
            "checkpoint": {
                "identity": checkpoint_identity,
                "load": {
                    "map_location": "cpu",
                    "mmap": True,
                    "system_byteorder": sys.byteorder,
                    "weights_only": True,
                },
                "state_dictionary": state_summary,
            },
            "decision": {
                "next_step": "PROCEED_TO_FIXED_DINOV2_SCORING_PATH_IMPLEMENTATION",
                "reason": (
                    "The fixed checkpoint passed weights-only CPU loading, finite "
                    "tensor inventory, and exact strict loading into the fixed local "
                    "non-register ViT-S/14 architecture."
                ),
            },
            "environment": {
                **environment,
                "accelerator_distribution_count": 0,
                "cuda_build_version": None,
                "hip_build_version": None,
                "interop_threads": torch.get_num_interop_threads(),
                "intraop_threads": torch.get_num_threads(),
                "required_process_environment": dict(sorted(EXPECTED_ENVIRONMENT.items())),
                "torch_version": torch.__version__,
                "xformers_imported": False,
            },
            "execution": {
                "execution_commit": execution_commit,
                "verification_date": verification_date,
            },
            "inputs": {
                "acquisition_record": "artifacts/v0.2/model-assets/acquisition.json",
                "acquisition_record_sha256": sha256_file(acquisition_path),
                "import_smoke_record": "artifacts/v0.2/environment/import-smoke.json",
                "import_smoke_record_sha256": sha256_file(import_smoke_path),
                "preregistration_id": "v0.2-dinov2-cpu-preflight-1",
            },
            "model": model_summary,
            "schema_version": OUTPUT_SCHEMA,
            "source": {
                "extraction": extraction,
                "identity": {
                    **source_identity,
                    "revision": SOURCE_REVISION,
                },
                "module_import_count": len(module_origins),
                "module_origins": module_origins,
                "source_import_root": (
                    f"data/external/v0.2/model-assets/{extraction_dir.name}/{SOURCE_ROOT}"
                ),
            },
        }
        write_json_atomic(output_path, report)
        return report
    except Exception:
        if extraction_created and not output_path.exists():
            shutil.rmtree(extraction_dir, ignore_errors=True)
        raise
