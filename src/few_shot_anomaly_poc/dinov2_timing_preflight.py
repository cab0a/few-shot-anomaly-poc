"""Check ordered prerequisites before the preregistered DINOv2 timing run."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from few_shot_anomaly_poc.hashing import sha256_file
from few_shot_anomaly_poc.jsonio import write_json_atomic

OUTPUT_SCHEMA = "v0.2-cpu-timing-preconditions-v2"
PREREGISTRATION_ID = "v0.2-dinov2-cpu-preflight-2"
PREREGISTRATION_PATH = "docs/v0.2-memory-bounded-cpu-preflight.md"
PREREGISTRATION_COMMIT = "a177b5648c450b1e33ca3bbf5c16a051410ef756"
PREREGISTRATION_SHA256 = "8d2d055d6f311719e28f52fb7e8f2f87fb3202c04414b56440d0a420832658ba"
SUPERSEDED_PREREGISTRATION_ID = "v0.2-dinov2-cpu-preflight-1"
SUPERSEDED_PREREGISTRATION_PATH = "docs/v0.2-preflight-preregistration.md"
SUPERSEDED_PREREGISTRATION_COMMIT = "e9330be10742947e4227ced4c99acafe4d098566"
SUPERSEDED_PREREGISTRATION_SHA256 = (
    "19d4cf4079c6df7c9042be464859ccf98d41108656ba0259c8940ace740ebf42"
)
EXPECTED_CPU_MODEL = "Intel(R) Core(TM) i7-3630QM CPU @ 2.40GHz"
EXPECTED_PHYSICAL_CORES = 4
EXPECTED_LOGICAL_CORES = 8
SUPERSEDED_RAM_SNAPSHOT_BYTES = 4_045_017_088
EXPECTED_MACHINE = "x86_64"
EXPECTED_PLATFORM = "linux"
EXPECTED_AFFINITY = tuple(range(EXPECTED_LOGICAL_CORES))
EXPECTED_NICE = 0
EXPECTED_SCHEDULER = "SCHED_OTHER"
AC_POWER_BATTERY_STATUSES = frozenset({2, 3, 6, 7, 8, 9, 11})
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
POWERSHELL_PATH = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
CONCURRENT_BENCHMARK_MARKERS = (
    "run_v0_2_cpu_timing.py",
    "run_v0_2_cpu_timing_resolution.py",
)
REQUIRED_RECORDS = {
    "dependency_wheel_inspection": {
        "path": "artifacts/v0.2/dependencies/wheel-inspection.json",
        "sha256": "402e35c32a7c31e2fd2470877f8047685a372e933d48167046366553eea1d0ad",
        "schema_version": "v0.2-dependency-artifact-inspection-v1",
        "decision_path": ("decision", "installation"),
        "decision_value": "INSTALL",
    },
    "isolated_import_smoke": {
        "path": "artifacts/v0.2/environment/import-smoke.json",
        "sha256": "b0f38afb103f7084a0e5e09e8fd00e4cf2e0e5825d7a3fe8d5e3b48afd7b1f74",
        "schema_version": "v0.2-isolated-import-smoke-v1",
        "decision_path": ("decision", "next_step"),
        "decision_value": "PROCEED_TO_CONTROLLED_MODEL_ASSET_ACQUISITION",
    },
    "model_asset_acquisition": {
        "path": "artifacts/v0.2/model-assets/acquisition.json",
        "sha256": "ba976ed08369fd80423d241129b8a86b05fcef650a39befa4ee67c8314233dac",
        "schema_version": "v0.2-model-asset-acquisition-v1",
        "decision_path": ("decision", "next_step"),
        "decision_value": "PROCEED_TO_WEIGHTS_ONLY_STRICT_LOAD_VERIFICATION",
    },
    "model_strict_load": {
        "path": "artifacts/v0.2/model-compatibility/strict-load.json",
        "sha256": "4491f2fb472df813642d296d92d396e62476a2fd257d6b9da431c3a90b6aa604",
        "schema_version": "v0.2-weights-only-strict-load-v1",
        "decision_path": ("decision", "next_step"),
        "decision_value": "PROCEED_TO_FIXED_DINOV2_SCORING_PATH_IMPLEMENTATION",
    },
    "fixed_scoring_path": {
        "path": "artifacts/v0.2/scoring-path/synthetic-smoke.json",
        "sha256": "56b5f342c3b8875df6c9baec61fdd8339c0f40d1f69c83245bdbf580ac23f7b8",
        "schema_version": "v0.2-fixed-dinov2-scoring-smoke-v1",
        "decision_path": ("decision", "next_step"),
        "decision_value": "PROCEED_TO_PREREGISTERED_CPU_TIMING_WORKLOAD",
    },
    "first_cpu_preflight_attempt": {
        "path": "artifacts/v0.2/cpu-preflight/attempt-001-target-machine-stop.json",
        "sha256": "b334ae369437636cc7c4e368e48e73687f3344a12a8e329134ba3871ed35a283",
        "schema_version": "v0.2-cpu-timing-preconditions-v1",
        "decision_path": ("decision", "outcome"),
        "decision_value": "DO NOT PROCEED",
    },
}


class DINOv2TimingPreflightError(Exception):
    """Reject a prerequisite check that cannot establish ordered evidence."""


@dataclass(frozen=True)
class TargetMachineObservation:
    """Observed fields used by the preregistered target-machine condition."""

    ac_power: bool
    battery_charge_percent: int | None
    battery_status: int
    cpu_affinity: tuple[int, ...]
    cpu_model: str
    logical_core_count: int
    machine: str
    nice: int
    operating_system: str
    physical_core_count: int
    mem_available_bytes: int | None
    mem_total_bytes: int | None
    meminfo_status: str
    ram_bytes: int | None
    ram_bytes_status: str
    scheduler: str
    swap_free_bytes: int | None
    swap_total_bytes: int | None
    sys_platform: str
    wsl2: bool


def _load_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DINOv2TimingPreflightError(f"cannot read {field}: {error}") from error
    if not isinstance(value, dict):
        raise DINOv2TimingPreflightError(f"{field} must contain a JSON object")
    return value


def _nested_value(record: dict[str, Any], path: tuple[str, ...]) -> object:
    current: object = record
    for field in path:
        if not isinstance(current, dict) or field not in current:
            raise DINOv2TimingPreflightError(f"required record is missing {'.'.join(path)}")
        current = current[field]
    return current


def _resolve_project_path(path: Path, *, project_root: Path, field: str) -> Path:
    candidate = path if path.is_absolute() else project_root / path
    resolved = candidate.resolve()
    if not resolved.is_relative_to(project_root):
        raise DINOv2TimingPreflightError(f"{field} must remain within project_root")
    return resolved


def _git_output(project_root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise DINOv2TimingPreflightError("cannot verify Git identity") from error


def _validate_execution_identity(
    *,
    project_root: Path,
    execution_commit: str,
    verification_date: str,
) -> dict[str, Any]:
    if not COMMIT_PATTERN.fullmatch(execution_commit):
        raise DINOv2TimingPreflightError("execution_commit must be a full lowercase Git commit")
    if not DATE_PATTERN.fullmatch(verification_date):
        raise DINOv2TimingPreflightError("verification_date must use YYYY-MM-DD")
    if _git_output(project_root, "rev-parse", "HEAD") != execution_commit:
        raise DINOv2TimingPreflightError("execution_commit is not the checked-out Git HEAD")
    if _git_output(project_root, "status", "--porcelain"):
        raise DINOv2TimingPreflightError("worktree must be clean before the prerequisite check")
    preregistration_commit_type = _git_output(
        project_root,
        "cat-file",
        "-t",
        PREREGISTRATION_COMMIT,
    )
    if preregistration_commit_type != "commit":
        raise DINOv2TimingPreflightError("controlling preregistration commit is unavailable")
    superseded_commit_type = _git_output(
        project_root,
        "cat-file",
        "-t",
        SUPERSEDED_PREREGISTRATION_COMMIT,
    )
    if superseded_commit_type != "commit":
        raise DINOv2TimingPreflightError("superseded preregistration commit is unavailable")
    return {
        "execution_commit": execution_commit,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "superseded_preregistration_commit": SUPERSEDED_PREREGISTRATION_COMMIT,
        "verification_date": verification_date,
        "worktree_clean": True,
    }


def _validate_required_records(project_root: Path) -> dict[str, Any]:
    validated: dict[str, Any] = {}
    for name, specification in REQUIRED_RECORDS.items():
        path = _resolve_project_path(
            Path(specification["path"]),
            project_root=project_root,
            field=name,
        )
        observed_sha256 = sha256_file(path)
        if observed_sha256 != specification["sha256"]:
            raise DINOv2TimingPreflightError(f"required record hash changed: {name}")
        record = _load_json(path, field=name)
        if record.get("schema_version") != specification["schema_version"]:
            raise DINOv2TimingPreflightError(f"required record schema changed: {name}")
        decision_path = specification["decision_path"]
        if _nested_value(record, decision_path) != specification["decision_value"]:
            raise DINOv2TimingPreflightError(f"required record decision changed: {name}")
        validated[name] = {
            "path": specification["path"],
            "sha256": observed_sha256,
            "verification": "pass",
        }
    return validated


def _read_cpuinfo() -> str:
    try:
        return Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError as error:
        raise DINOv2TimingPreflightError("cannot read /proc/cpuinfo") from error


def _cpu_model(cpuinfo: str) -> str:
    for line in cpuinfo.splitlines():
        if line.startswith("model name") and ":" in line:
            return line.split(":", maxsplit=1)[1].strip()
    raise DINOv2TimingPreflightError("CPU model is unavailable")


def _physical_core_count(cpuinfo: str) -> int:
    pairs: set[tuple[str, str]] = set()
    physical_id: str | None = None
    core_id: str | None = None
    for line in [*cpuinfo.splitlines(), ""]:
        stripped = line.strip()
        if not stripped:
            if physical_id is not None and core_id is not None:
                pairs.add((physical_id, core_id))
            physical_id = None
            core_id = None
        elif ":" in line:
            key, value = (part.strip() for part in line.split(":", maxsplit=1))
            if key == "physical id":
                physical_id = value
            elif key == "core id":
                core_id = value
    if not pairs:
        raise DINOv2TimingPreflightError("physical core count is unavailable")
    return len(pairs)


def _ram_bytes() -> tuple[int | None, str]:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None, "unavailable"
    if (
        not isinstance(pages, int)
        or isinstance(pages, bool)
        or pages <= 0
        or not isinstance(page_size, int)
        or isinstance(page_size, bool)
        or page_size <= 0
    ):
        return None, "unavailable"
    return pages * page_size, "available"


def _meminfo_bytes() -> tuple[dict[str, int | None], str]:
    fields = {
        "MemAvailable": "mem_available_bytes",
        "MemTotal": "mem_total_bytes",
        "SwapFree": "swap_free_bytes",
        "SwapTotal": "swap_total_bytes",
    }
    values: dict[str, int | None] = {output_name: None for output_name in fields.values()}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return values, "unavailable"
    for line in lines:
        if ":" not in line:
            continue
        name, raw_value = line.split(":", maxsplit=1)
        if name not in fields:
            continue
        parts = raw_value.split()
        if len(parts) == 2 and parts[1] == "kB" and parts[0].isdigit():
            values[fields[name]] = int(parts[0]) * 1_024
    available_count = sum(value is not None for value in values.values())
    if available_count == len(values):
        status = "available"
    elif available_count:
        status = "partial"
    else:
        status = "unavailable"
    return values, status


def _windows_power_status() -> tuple[int, int | None]:
    if not POWERSHELL_PATH.is_file():
        raise DINOv2TimingPreflightError("Windows PowerShell is unavailable")
    command = (
        "$battery = Get-CimInstance Win32_Battery; "
        "if ($null -eq $battery) { throw 'Win32_Battery is unavailable' }; "
        "[pscustomobject]@{BatteryStatus=[int]$battery.BatteryStatus; "
        "EstimatedChargeRemaining=[int]$battery.EstimatedChargeRemaining} "
        "| ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            [
                str(POWERSHELL_PATH),
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        record = json.loads(completed.stdout)
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ) as error:
        raise DINOv2TimingPreflightError("cannot capture Windows battery status") from error
    if not isinstance(record, dict):
        raise DINOv2TimingPreflightError("Windows battery status is invalid")
    status = record.get("BatteryStatus")
    charge = record.get("EstimatedChargeRemaining")
    if (
        not isinstance(status, int)
        or isinstance(status, bool)
        or not isinstance(charge, int)
        or isinstance(charge, bool)
        or not 0 <= charge <= 100
    ):
        raise DINOv2TimingPreflightError("Windows battery fields are invalid")
    return status, charge


def _scheduler_name() -> str:
    try:
        scheduler = os.sched_getscheduler(0)
    except (AttributeError, OSError) as error:
        raise DINOv2TimingPreflightError("process scheduler is unavailable") from error
    names = {
        value: name
        for name in ("SCHED_OTHER", "SCHED_BATCH", "SCHED_IDLE", "SCHED_FIFO", "SCHED_RR")
        if isinstance(value := getattr(os, name, None), int)
    }
    return names.get(scheduler, f"UNKNOWN_{scheduler}")


def capture_target_machine() -> TargetMachineObservation:
    """Capture the exact target-machine fields before any timing work."""
    cpuinfo = _read_cpuinfo()
    logical_cores = os.cpu_count()
    if not isinstance(logical_cores, int) or logical_cores <= 0:
        raise DINOv2TimingPreflightError("logical core count is unavailable")
    try:
        affinity = tuple(sorted(os.sched_getaffinity(0)))
        nice = os.getpriority(os.PRIO_PROCESS, 0)
        os_release = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").strip()
    except (AttributeError, OSError) as error:
        raise DINOv2TimingPreflightError("process or WSL identity is unavailable") from error
    battery_status, charge = _windows_power_status()
    ram_bytes, ram_bytes_status = _ram_bytes()
    meminfo, meminfo_status = _meminfo_bytes()
    return TargetMachineObservation(
        ac_power=battery_status in AC_POWER_BATTERY_STATUSES,
        battery_charge_percent=charge,
        battery_status=battery_status,
        cpu_affinity=affinity,
        cpu_model=_cpu_model(cpuinfo),
        logical_core_count=logical_cores,
        machine=platform.machine(),
        mem_available_bytes=meminfo["mem_available_bytes"],
        mem_total_bytes=meminfo["mem_total_bytes"],
        meminfo_status=meminfo_status,
        nice=nice,
        operating_system=platform.platform(),
        physical_core_count=_physical_core_count(cpuinfo),
        ram_bytes=ram_bytes,
        ram_bytes_status=ram_bytes_status,
        scheduler=_scheduler_name(),
        swap_free_bytes=meminfo["swap_free_bytes"],
        swap_total_bytes=meminfo["swap_total_bytes"],
        sys_platform=os.sys.platform,
        wsl2="microsoft-standard-wsl2" in os_release.lower(),
    )


def evaluate_target_machine(
    observation: TargetMachineObservation,
) -> dict[str, Any]:
    """Evaluate fixed target identity while keeping memory values diagnostic."""
    checks = {
        "ac_power": observation.ac_power is True,
        "cpu_affinity": observation.cpu_affinity == EXPECTED_AFFINITY,
        "cpu_model": observation.cpu_model == EXPECTED_CPU_MODEL,
        "logical_core_count": (observation.logical_core_count == EXPECTED_LOGICAL_CORES),
        "machine": observation.machine == EXPECTED_MACHINE,
        "nice": observation.nice == EXPECTED_NICE,
        "physical_core_count": (observation.physical_core_count == EXPECTED_PHYSICAL_CORES),
        "scheduler": observation.scheduler == EXPECTED_SCHEDULER,
        "sys_platform": observation.sys_platform == EXPECTED_PLATFORM,
        "wsl2": observation.wsl2 is True,
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    return {
        "checks": checks,
        "diagnostics": {
            "meminfo_status": observation.meminfo_status,
            "ram_bytes_status": observation.ram_bytes_status,
            "superseded_ram_snapshot_bytes": SUPERSEDED_RAM_SNAPSHOT_BYTES,
            "total_ram_is_gating": False,
            "total_ram_meets_superseded_snapshot": (
                None
                if observation.ram_bytes is None
                else observation.ram_bytes >= SUPERSEDED_RAM_SNAPSHOT_BYTES
            ),
        },
        "failures": list(failures),
        "status": "pass" if not failures else "fail",
    }


def _concurrent_benchmark_count() -> int:
    matches = 0
    current_pid = os.getpid()
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            pid = int(path.parent.name)
            if pid == current_pid:
                continue
            command = path.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        if any(marker in command for marker in CONCURRENT_BENCHMARK_MARKERS):
            matches += 1
    return matches


def _ordered_conditions(target_status: str) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = [
        {"condition": 1, "name": "preregistration_identity", "status": "pass"},
        {"condition": 2, "name": "source_and_license", "status": "pass"},
        {"condition": 3, "name": "dependency_resolution", "status": "pass"},
        {"condition": 4, "name": "checkpoint_acquisition", "status": "pass"},
        {"condition": 5, "name": "third_party_separation", "status": "pass"},
        {"condition": 6, "name": "target_machine", "status": target_status},
    ]
    later_status = "not_evaluated" if target_status == "fail" else "pending"
    conditions.extend(
        {"condition": index, "name": name, "status": later_status}
        for index, name in (
            (7, "execution_integrity"),
            (8, "cpu_result"),
            (9, "reproducibility"),
            (10, "evaluation_boundary"),
        )
    )
    return conditions


def run_timing_preconditions(
    *,
    project_root: Path,
    execution_commit: str,
    verification_date: str,
    no_concurrent_project_benchmark_confirmed: bool,
    output_path: Path,
) -> dict[str, Any]:
    """Write ordered evidence and stop before timing when a prerequisite fails."""
    project_root = project_root.resolve()
    output_path = _resolve_project_path(
        output_path,
        project_root=project_root,
        field="output_path",
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    execution = _validate_execution_identity(
        project_root=project_root,
        execution_commit=execution_commit,
        verification_date=verification_date,
    )
    preregistration_path = project_root / PREREGISTRATION_PATH
    if sha256_file(preregistration_path) != PREREGISTRATION_SHA256:
        raise DINOv2TimingPreflightError("preregistration bytes have changed")
    superseded_preregistration_path = project_root / SUPERSEDED_PREREGISTRATION_PATH
    if (
        sha256_file(superseded_preregistration_path)
        != SUPERSEDED_PREREGISTRATION_SHA256
    ):
        raise DINOv2TimingPreflightError("superseded preregistration bytes have changed")
    records = _validate_required_records(project_root)
    if no_concurrent_project_benchmark_confirmed is not True:
        raise DINOv2TimingPreflightError(
            "explicit no-concurrent-project-benchmark confirmation is required"
        )
    concurrent_benchmark_count = _concurrent_benchmark_count()
    if concurrent_benchmark_count:
        raise DINOv2TimingPreflightError("another v0.2 project timing process is already running")
    observation = capture_target_machine()
    target_evaluation = evaluate_target_machine(observation)
    target_passed = target_evaluation["status"] == "pass"
    report = {
        "boundary": {
            "dataset_access": False,
            "labels_accessed": False,
            "model_constructed": False,
            "model_inference_performed": False,
            "network_access": False,
            "scoring_performed": False,
            "timing_invocation_count": 0,
        },
        "decision": {
            "first_failed_condition": None if target_passed else 6,
            "next_step": (
                "PROCEED_TO_FRESH_PROCESS_TIMING_RUN"
                if target_passed
                else "DO_NOT_START_TIMING_WORKLOAD"
            ),
            "outcome": "PENDING" if target_passed else "DO NOT PROCEED",
            "status": "pass" if target_passed else "stop",
        },
        "execution": execution,
        "inputs": {
            "preregistration_id": PREREGISTRATION_ID,
            "preregistration_path": PREREGISTRATION_PATH,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "required_records": records,
            "superseded_preregistration": {
                "commit": SUPERSEDED_PREREGISTRATION_COMMIT,
                "id": SUPERSEDED_PREREGISTRATION_ID,
                "path": SUPERSEDED_PREREGISTRATION_PATH,
                "sha256": SUPERSEDED_PREREGISTRATION_SHA256,
            },
        },
        "ordered_stop_conditions": _ordered_conditions(target_evaluation["status"]),
        "schema_version": OUTPUT_SCHEMA,
        "target_machine": {
            "concurrent_project_benchmark": {
                "matching_process_count": concurrent_benchmark_count,
                "operator_confirmation": True,
                "status": "pass",
            },
            "evaluation": target_evaluation,
            "observed": asdict(observation),
            "required": {
                "ac_power": True,
                "cpu_affinity": list(EXPECTED_AFFINITY),
                "cpu_model": EXPECTED_CPU_MODEL,
                "logical_core_count": EXPECTED_LOGICAL_CORES,
                "machine": EXPECTED_MACHINE,
                "nice": EXPECTED_NICE,
                "physical_core_count": EXPECTED_PHYSICAL_CORES,
                "scheduler": EXPECTED_SCHEDULER,
                "sys_platform": EXPECTED_PLATFORM,
                "wsl2": True,
            },
            "resource_policy": {
                "hard_memory_failures": [
                    "memory_error",
                    "framework_out_of_memory",
                    "operating_system_termination",
                    "nonzero_resolution_process_exit",
                    "missing_observation",
                    "incomplete_timing_run",
                ],
                "peak_rss_is_gating": False,
                "superseded_ram_snapshot_bytes": SUPERSEDED_RAM_SNAPSHOT_BYTES,
                "total_ram_is_gating": False,
            },
        },
    }
    write_json_atomic(output_path, report)
    return report
