"""Resource Governor — monitors system resources and provides structured constraints.

Reads CPU, RAM, disk metrics via psutil. Consumes AcceleratorProbe results.
Returns ResourceConstraints consumed by the Backend Selection Engine.
Does NOT execute inference or select backends — provides state only.
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

import psutil

DEFAULT_CPU_COUNT_FALLBACK = 2
RESERVED_CORES = 1

from control_panel.common.config import get_config
from control_panel.common.logging import get_logger
from control_panel.common.models import (
    ModelRequirements,
    ResourceRequirements,
    SpawnDecision,
)
from control_panel.resources.models import (
    AcceleratorStatus,
    ResourceConstraints,
    ResourceSnapshot,
)
from control_panel.resources.probe import AcceleratorProbe, get_probe

__all__ = ["ResourceGovernor"]

log = get_logger(__name__)


class ResourceGovernor:
    """Monitors system resources and provides structured constraints.

    The Orchestrator calls this before spawning workers or routing inference.
    The Governor does NOT choose backends — it provides facts and constraints.
    """

    def __init__(self, probe: Optional[AcceleratorProbe] = None) -> None:
        self._probe = probe or get_probe()
        self._accelerator_cache: Optional[dict[str, AcceleratorStatus]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot(self) -> ResourceSnapshot:
        """Return a point-in-time snapshot of all system resources."""
        cfg = get_config().resources

        # CPU (2-second interval for accuracy)
        cpu_percent = psutil.cpu_percent(interval=0.5)
        cpu_count = psutil.cpu_count(logical=True) or 1

        # Memory
        mem = psutil.virtual_memory()
        ram_total_mb = int(mem.total / 1_048_576)
        ram_available_mb = int(mem.available / 1_048_576)
        ram_percent_used = mem.percent

        # Disk (on C:\ for Windows)
        try:
            disk = shutil.disk_usage("C:\\")
            disk_free_mb = int(disk.free / 1_048_576)
        except Exception:
            disk_free_mb = 0

        # Accelerators (use cached probe results)
        accelerators = self._get_accelerators()

        # Power source (best-effort)
        power_source: Optional[str] = None
        battery_percent: Optional[float] = None
        try:
            battery = psutil.sensors_battery()
            if battery is not None:
                power_source = "ac" if battery.power_plugged else "battery"
                battery_percent = round(battery.percent, 1)
        except (AttributeError, Exception):
            pass  # sensors_battery() not available on all platforms

        return ResourceSnapshot(
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            ram_total_mb=ram_total_mb,
            ram_available_mb=ram_available_mb,
            ram_percent_used=ram_percent_used,
            disk_free_mb=disk_free_mb,
            accelerators=accelerators,
            power_source=power_source,
            battery_percent=battery_percent,
        )

    def can_spawn(self, requirements: Optional[ResourceRequirements] = None) -> SpawnDecision:
        """Determine whether a new worker can be spawned given current resources."""
        cfg = get_config().resources
        req = requirements or ResourceRequirements()

        snap = self.snapshot()

        # RAM check
        if snap.ram_available_mb < req.min_free_ram_mb:
            return SpawnDecision(
                allowed=False,
                reason=(
                    f"Insufficient RAM: {snap.ram_available_mb} MB available, "
                    f"need {req.min_free_ram_mb} MB free"
                ),
            )

        # CPU check
        if snap.cpu_percent > req.max_cpu_percent:
            return SpawnDecision(
                allowed=False,
                reason=(
                    f"CPU too busy: {snap.cpu_percent:.0f}% used, "
                    f"threshold is {req.max_cpu_percent:.0f}%"
                ),
            )

        # Disk check
        if snap.disk_free_mb < req.min_free_disk_mb:
            return SpawnDecision(
                allowed=False,
                reason=(
                    f"Insufficient disk: {snap.disk_free_mb} MB free, "
                    f"need {req.min_free_disk_mb} MB"
                ),
            )

        return SpawnDecision(
            allowed=True,
            reason=(
                f"Resources OK: RAM={snap.ram_available_mb} MB, "
                f"CPU={snap.cpu_percent:.0f}%, Disk={snap.disk_free_mb} MB"
            ),
        )

    def get_resource_constraints(
        self,
        model_req: Optional[ModelRequirements] = None,
    ) -> ResourceConstraints:
        """Return structured constraints for the Backend Selection Engine.

        The Selection Engine uses these to filter and score inference backends.
        """
        cfg = get_config().resources
        snap = self.snapshot()

        safety_margin = cfg.safety_margin_mb
        available_ram = snap.ram_available_mb
        max_model_ram = max(0, available_ram - safety_margin)

        # CPU pressure classification
        if snap.cpu_percent < 40:
            cpu_pressure = "low"
        elif snap.cpu_percent < 70:
            cpu_pressure = "moderate"
        else:
            cpu_pressure = "high"

        # Power policy
        if snap.power_source == "battery" and (snap.battery_percent or 100) < 20:
            power_policy = "critical"
        elif snap.power_source == "battery":
            power_policy = "low_power"
        else:
            power_policy = "normal"

        # Preferred device from accelerators
        accelerators = snap.accelerators
        preferred_device: Optional[str] = None
        if accelerators.get("npu") and accelerators["npu"].available:
            preferred_device = "npu"
        elif accelerators.get("gpu") and accelerators["gpu"].available:
            preferred_device = "gpu"
        else:
            preferred_device = "cpu"

        # Can a local model be loaded at all?
        can_load_local = max_model_ram > 0
        if model_req is not None:
            can_load_local = model_req.estimated_ram_mb <= max_model_ram

        # Compute dynamic worker limit from hardware
        cpu_count = os.cpu_count() or DEFAULT_CPU_COUNT_FALLBACK
        max_concurrent_workers = max(1, cpu_count - RESERVED_CORES)

        reason_parts = [
            f"RAM available={available_ram} MB",
            f"max_model={max_model_ram} MB",
            f"cpu={cpu_pressure}",
            f"power={power_policy}",
            f"preferred_device={preferred_device}",
        ]

        return ResourceConstraints(
            available_ram_mb=available_ram,
            safety_margin_mb=safety_margin,
            max_model_ram_mb=max_model_ram,
            cpu_pressure=cpu_pressure,
            power_policy=power_policy,
            max_concurrent_workers=max_concurrent_workers,
            preferred_device=preferred_device,
            can_load_local_model=can_load_local,
            reason=", ".join(reason_parts),
        )

    def refresh_accelerators(self) -> dict[str, AcceleratorStatus]:
        """Force re-probe all accelerators and update cache."""
        self._probe.invalidate()
        self._accelerator_cache = self._probe.probe_all()
        return self._accelerator_cache

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_accelerators(self) -> dict[str, AcceleratorStatus]:
        """Return cached accelerator status, probing if needed."""
        if self._accelerator_cache is None:
            self._accelerator_cache = self._probe.probe_all()
        return self._accelerator_cache
