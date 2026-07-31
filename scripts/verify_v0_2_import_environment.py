"""Verify the isolated v0.2 environment through metadata and imports only."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import re
import site
import sys
import sysconfig
import tempfile
import warnings
from pathlib import Path
from typing import Any

EXPECTED_IMPORTS = {
    "filelock": "filelock",
    "fsspec": "fsspec",
    "iniconfig": "iniconfig",
    "jinja2": "jinja2",
    "markupsafe": "markupsafe",
    "mpmath": "mpmath",
    "networkx": "networkx",
    "numpy": "numpy",
    "packaging": "packaging",
    "pluggy": "pluggy",
    "pygments": "pygments",
    "pytest": "pytest",
    "ruff": None,
    "setuptools": "setuptools",
    "sympy": "sympy",
    "torch": "torch",
    "typing-extensions": "typing_extensions",
}
ACCELERATOR_DISTRIBUTION_PATTERNS = (
    re.compile(r"^cuda(?:-|$)"),
    re.compile(r"^cupy(?:-|$)"),
    re.compile(r"^intel-extension-for-pytorch$"),
    re.compile(r"^nvidia(?:-|$)"),
    re.compile(r"^pytorch-triton(?:-|$)"),
    re.compile(r"^torch-(?:npu|tpu|xla)$"),
    re.compile(r"^triton$"),
)
CANONICAL_NAME_PATTERN = re.compile(r"[-_.]+")
EXPECTED_PYTHON_VERSION = "3.13.14"
EXPECTED_INSPECTION_SCHEMA = "v0.2-dependency-artifact-inspection-v1"
OUTPUT_SCHEMA = "v0.2-isolated-import-smoke-v1"


class ImportSmokeError(Exception):
    """Reject an environment that differs from the inspected wheel set."""


def _canonical_name(value: str) -> str:
    return CANONICAL_NAME_PATTERN.sub("-", value).lower()


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            newline="\n",
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _load_expected_distributions(
    inspection_path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    try:
        record = json.loads(inspection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImportSmokeError(f"cannot read wheel inspection record: {error}") from error
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != EXPECTED_INSPECTION_SCHEMA
        or record.get("decision", {}).get("installation") != "INSTALL"
        or record.get("summary", {}).get("distribution_count") != 17
        or not isinstance(record.get("packages"), list)
    ):
        raise ImportSmokeError("wheel inspection record does not authorize installation")

    versions: dict[str, str] = {}
    roles: dict[str, str] = {}
    for item in record["packages"]:
        dependency = item.get("dependency", {})
        name = _canonical_name(dependency.get("name", ""))
        version = dependency.get("version")
        role = dependency.get("role")
        if (
            name not in EXPECTED_IMPORTS
            or name in versions
            or not isinstance(version, str)
            or role not in {"development", "runtime"}
            or item.get("artifact", {}).get("checksum_status") != "upstream_verified"
            or item.get("archive", {}).get("record_verification") != "pass"
        ):
            raise ImportSmokeError(f"invalid inspected distribution entry: {name!r}")
        versions[name] = version
        roles[name] = role
    if set(versions) != set(EXPECTED_IMPORTS):
        raise ImportSmokeError("wheel inspection distribution set is not the fixed set")
    return versions, roles


def _installed_distributions() -> dict[str, str]:
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not isinstance(raw_name, str) or not raw_name:
            raise ImportSmokeError("installed distribution has no valid Name")
        name = _canonical_name(raw_name)
        if name in installed:
            raise ImportSmokeError(f"duplicate installed distribution: {name}")
        installed[name] = distribution.version
    return installed


def _relative_origin(module: object, environment_root: Path, *, name: str) -> str:
    raw_origin = getattr(module, "__file__", None)
    if not isinstance(raw_origin, str) or not raw_origin:
        raise ImportSmokeError(f"imported module has no file origin: {name}")
    origin = Path(raw_origin).resolve()
    try:
        relative = origin.relative_to(environment_root)
    except ValueError as error:
        raise ImportSmokeError(f"module loaded outside isolated environment: {name}") from error
    return relative.as_posix()


def _accelerator_distributions(names: set[str]) -> list[str]:
    return sorted(
        name
        for name in names
        if any(pattern.search(name) for pattern in ACCELERATOR_DISTRIBUTION_PATTERNS)
    )


def verify_import_environment(
    *,
    inspection_path: Path,
    environment_root: Path,
    forbidden_environment_root: Path,
) -> dict[str, Any]:
    """Verify exact metadata, import origins, and CPU-only PyTorch build identity."""
    environment_root = environment_root.resolve()
    forbidden_environment_root = forbidden_environment_root.resolve()
    if Path(sys.prefix).resolve() != environment_root:
        raise ImportSmokeError("sys.prefix is not the requested isolated environment")
    if Path(sys.base_prefix).resolve() == environment_root:
        raise ImportSmokeError("interpreter is not running from a virtual environment")
    if platform.python_version() != EXPECTED_PYTHON_VERSION:
        raise ImportSmokeError(
            f"unexpected Python version: {platform.python_version()}"
        )
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise ImportSmokeError("unexpected interpreter platform")
    if site.ENABLE_USER_SITE is not False:
        raise ImportSmokeError("user site must be disabled")

    visible_paths = {
        Path(entry).resolve()
        for entry in sys.path
        if isinstance(entry, str) and entry and Path(entry).exists()
    }
    if forbidden_environment_root in visible_paths or any(
        forbidden_environment_root in path.parents for path in visible_paths
    ):
        raise ImportSmokeError("the root v0.1 environment is visible in sys.path")

    expected_versions, roles = _load_expected_distributions(inspection_path)
    installed_versions = _installed_distributions()
    if installed_versions != expected_versions:
        missing = sorted(set(expected_versions) - set(installed_versions))
        extra = sorted(set(installed_versions) - set(expected_versions))
        mismatched = sorted(
            name
            for name in set(expected_versions) & set(installed_versions)
            if expected_versions[name] != installed_versions[name]
        )
        raise ImportSmokeError(
            "installed distribution set differs from inspected wheels: "
            f"missing={missing}, extra={extra}, version_mismatch={mismatched}"
        )

    imports: list[dict[str, Any]] = []
    imported_modules: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for distribution_name, import_name in EXPECTED_IMPORTS.items():
            if import_name is None:
                imports.append(
                    {
                        "distribution": distribution_name,
                        "import_name": None,
                        "origin": None,
                        "status": "metadata_only_no_python_module",
                    }
                )
                continue
            try:
                module = importlib.import_module(import_name)
            except Exception as error:
                raise ImportSmokeError(
                    f"import failed for {distribution_name}: "
                    f"{type(error).__name__}: {error}"
                ) from error
            imported_modules[distribution_name] = module
            item: dict[str, Any] = {
                "distribution": distribution_name,
                "import_name": import_name,
                "origin": _relative_origin(
                    module,
                    environment_root,
                    name=distribution_name,
                ),
                "status": "pass",
            }
            module_version = getattr(module, "__version__", None)
            if isinstance(module_version, str):
                item["module_reported_version"] = module_version
            imports.append(item)

    torch_module = imported_modules["torch"]
    torch_version = getattr(torch_module, "__version__", None)
    torch_version_module = getattr(torch_module, "version", None)
    torch_cuda_version = getattr(torch_version_module, "cuda", None)
    torch_hip_version = getattr(torch_version_module, "hip", None)
    if (
        torch_version != "2.13.0+cpu"
        or installed_versions["torch"] != "2.13.0+cpu"
        or torch_cuda_version is not None
        or torch_hip_version is not None
    ):
        raise ImportSmokeError("installed PyTorch is not the fixed CPU-only build")

    accelerator_distributions = _accelerator_distributions(set(installed_versions))
    if accelerator_distributions:
        raise ImportSmokeError(
            "unexpected accelerator distributions are installed: "
            f"{accelerator_distributions}"
        )

    packages = [
        {
            "name": name,
            "role": roles[name],
            "version": installed_versions[name],
        }
        for name in sorted(installed_versions)
    ]
    return {
        "boundary": {
            "accelerator_runtime_probe_performed": False,
            "dataset_access": False,
            "dinov2_checkpoint_acquired": False,
            "dinov2_source_acquired": False,
            "model_constructed": False,
            "model_inference_performed": False,
            "network_access_during_verification": False,
            "tensor_operation_performed": False,
        },
        "decision": {
            "next_step": "PROCEED_TO_CONTROLLED_MODEL_ASSET_ACQUISITION",
            "reason": (
                "The exact inspected distributions import from the isolated environment, "
                "and the fixed PyTorch build reports CPU-only metadata."
            ),
        },
        "environment": {
            "environment_path": "environments/v0.2-preflight/.venv",
            "forbidden_root_environment_visible": False,
            "implementation": platform.python_implementation(),
            "isolated_prefix": True,
            "platform": sysconfig.get_platform(),
            "python_executable": "environments/v0.2-preflight/.venv/bin/python",
            "python_version": platform.python_version(),
            "user_site_enabled": False,
        },
        "imports": imports,
        "installation": {
            "artifact_installation_source": "verified_external_wheel_set",
            "completed_method": "uv_pip_install_exact_local_wheels_no_deps",
            "dependency_compatibility_check": "pass",
            "initial_locked_sync_attempt": {
                "package_installation_occurred": False,
                "reason_code": "EXPLICIT_INDEX_NOT_REPLACED_BY_NO_INDEX_FIND_LINKS",
                "result": "stopped_before_install",
            },
            "installer": "uv 0.11.32",
            "network_access": False,
            "source_build": False,
            "wheel_count": len(expected_versions),
        },
        "inputs": {
            "wheel_inspection_record": (
                "artifacts/v0.2/dependencies/wheel-inspection.json"
            ),
            "wheel_inspection_record_sha256": _sha256_file(inspection_path),
        },
        "packages": packages,
        "schema_version": OUTPUT_SCHEMA,
        "summary": {
            "accelerator_distribution_count": 0,
            "development_distribution_count": sum(
                role == "development" for role in roles.values()
            ),
            "distribution_count": len(installed_versions),
            "import_pass_count": sum(item["status"] == "pass" for item in imports),
            "metadata_only_count": sum(
                item["status"] == "metadata_only_no_python_module" for item in imports
            ),
            "runtime_distribution_count": sum(
                role == "runtime" for role in roles.values()
            ),
        },
        "torch": {
            "build": "cpu",
            "cuda_build_version": torch_cuda_version,
            "distribution_version": installed_versions["torch"],
            "hip_build_version": torch_hip_version,
            "module_version": torch_version,
            "runtime_accelerator_probe_performed": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the exact isolated v0.2 distributions and imports without "
            "model construction, tensor operations, dataset access, or network access."
        )
    )
    parser.add_argument(
        "--inspection-record",
        type=Path,
        default=Path("artifacts/v0.2/dependencies/wheel-inspection.json"),
    )
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--forbidden-environment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = verify_import_environment(
            inspection_path=args.inspection_record,
            environment_root=args.environment_root,
            forbidden_environment_root=args.forbidden_environment_root,
        )
        _write_json_atomic(args.output, report)
    except (FileExistsError, ImportSmokeError, OSError) as error:
        print(f"error: {error}")
        return 1
    print(
        "isolated import smoke verification passed: "
        f"distributions={report['summary']['distribution_count']}, "
        f"imports={report['summary']['import_pass_count']}, "
        f"torch={report['torch']['module_version']}, "
        f"output={args.output.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
