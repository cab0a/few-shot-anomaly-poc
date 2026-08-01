from __future__ import annotations

from dataclasses import replace

import pytest

from few_shot_anomaly_poc.dinov2_timing_preflight import (
    EXPECTED_CPU_MODEL,
    EXPECTED_LOGICAL_CORES,
    EXPECTED_PHYSICAL_CORES,
    SUPERSEDED_RAM_SNAPSHOT_BYTES,
    TargetMachineObservation,
    _ordered_conditions,
    evaluate_target_machine,
)


@pytest.fixture
def passing_machine() -> TargetMachineObservation:
    return TargetMachineObservation(
        ac_power=True,
        battery_charge_percent=100,
        battery_status=2,
        cpu_affinity=tuple(range(EXPECTED_LOGICAL_CORES)),
        cpu_model=EXPECTED_CPU_MODEL,
        logical_core_count=EXPECTED_LOGICAL_CORES,
        machine="x86_64",
        mem_available_bytes=2_000_000_000,
        mem_total_bytes=SUPERSEDED_RAM_SNAPSHOT_BYTES - 3 * 4_096,
        meminfo_status="available",
        nice=0,
        operating_system="Linux-test-microsoft-standard-WSL2",
        physical_core_count=EXPECTED_PHYSICAL_CORES,
        ram_bytes=SUPERSEDED_RAM_SNAPSHOT_BYTES - 3 * 4_096,
        ram_bytes_status="available",
        scheduler="SCHED_OTHER",
        swap_free_bytes=1_000_000_000,
        swap_total_bytes=1_000_000_000,
        sys_platform="linux",
        wsl2=True,
    )


def test_target_machine_requires_every_fixed_field(
    passing_machine: TargetMachineObservation,
) -> None:
    result = evaluate_target_machine(passing_machine)

    assert result["status"] == "pass"
    assert result["failures"] == []
    assert all(result["checks"].values())


def test_three_page_ram_shortfall_is_recorded_but_not_gating(
    passing_machine: TargetMachineObservation,
) -> None:
    result = evaluate_target_machine(passing_machine)

    assert result["status"] == "pass"
    assert result["failures"] == []
    assert "ram_bytes" not in result["checks"]
    assert result["diagnostics"] == {
        "meminfo_status": "available",
        "ram_bytes_status": "available",
        "superseded_ram_snapshot_bytes": SUPERSEDED_RAM_SNAPSHOT_BYTES,
        "total_ram_is_gating": False,
        "total_ram_meets_superseded_snapshot": False,
    }


def test_unavailable_ram_diagnostics_do_not_claim_workload_failure(
    passing_machine: TargetMachineObservation,
) -> None:
    observed = replace(
        passing_machine,
        mem_available_bytes=None,
        mem_total_bytes=None,
        meminfo_status="unavailable",
        ram_bytes=None,
        ram_bytes_status="unavailable",
        swap_free_bytes=None,
        swap_total_bytes=None,
    )

    result = evaluate_target_machine(observed)

    assert result["status"] == "pass"
    assert result["failures"] == []
    assert result["diagnostics"]["total_ram_meets_superseded_snapshot"] is None


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("ac_power", False, "ac_power"),
        ("cpu_affinity", tuple(range(7)), "cpu_affinity"),
        ("cpu_model", "another CPU", "cpu_model"),
        ("logical_core_count", 4, "logical_core_count"),
        ("machine", "aarch64", "machine"),
        ("nice", 1, "nice"),
        ("physical_core_count", 2, "physical_core_count"),
        ("scheduler", "SCHED_BATCH", "scheduler"),
        ("sys_platform", "win32", "sys_platform"),
        ("wsl2", False, "wsl2"),
    ],
)
def test_each_changed_target_field_is_a_hard_failure(
    passing_machine: TargetMachineObservation,
    field: str,
    value: object,
    failure: str,
) -> None:
    result = evaluate_target_machine(replace(passing_machine, **{field: value}))

    assert result["status"] == "fail"
    assert result["failures"] == [failure]


def test_ordered_conditions_do_not_evaluate_after_target_failure() -> None:
    conditions = _ordered_conditions("fail")

    assert [item["status"] for item in conditions[:5]] == ["pass"] * 5
    assert conditions[5] == {
        "condition": 6,
        "name": "target_machine",
        "status": "fail",
    }
    assert [item["status"] for item in conditions[6:]] == ["not_evaluated"] * 4


def test_ordered_conditions_leave_later_work_pending_after_target_pass() -> None:
    conditions = _ordered_conditions("pass")

    assert [item["status"] for item in conditions[:6]] == ["pass"] * 6
    assert [item["status"] for item in conditions[6:]] == ["pending"] * 4
