"""Data models for the Resource Governor and Accelerator Probe."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AcceleratorStatus",
    "ResourceSnapshot",
    "ResourceConstraints",
]


class AcceleratorStatus(BaseModel):
    """Detailed status of one hardware accelerator from the AcceleratorProbe."""

    model_config = ConfigDict(frozen=False)

    accelerator_type: str          # "npu" | "gpu" | "cpu"
    available: bool                # Final determination: can we use this?
    device_name: Optional[str] = None
    provider: Optional[str] = None  # e.g. "QNNExecutionProvider"
    driver_detected: bool = False
    driver_version: Optional[str] = None
    provider_loadable: bool = False  # Can the EP be imported/loaded?
    model_compatible: bool = False   # Can the current model execute on this EP?
    probe_inference_ok: bool = False  # Did a lightweight test inference succeed?
    failure_reason: Optional[str] = None
    probed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ResourceSnapshot(BaseModel):
    """Point-in-time snapshot of system resource state."""

    model_config = ConfigDict(frozen=False)

    # CPU & memory (from psutil)
    cpu_percent: float
    cpu_count: int
    ram_total_mb: int
    ram_available_mb: int
    ram_percent_used: float
    disk_free_mb: int

    # Accelerators (from AcceleratorProbe)
    accelerators: dict[str, AcceleratorStatus] = Field(default_factory=dict)

    # Power (best-effort, None if unavailable)
    power_source: Optional[str] = None   # "ac" | "battery" | None
    battery_percent: Optional[float] = None

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ResourceConstraints(BaseModel):
    """Structured resource constraints passed to the Backend Selection Engine."""

    model_config = ConfigDict(frozen=False)

    available_ram_mb: int
    safety_margin_mb: int = 512
    max_model_ram_mb: int       # = available_ram_mb - safety_margin_mb
    cpu_pressure: str           # "low" | "moderate" | "high"
    power_policy: str           # "normal" | "low_power" | "critical"
    max_concurrent_workers: int
    preferred_device: Optional[str] = None  # From accelerator availability
    can_load_local_model: bool = False
    reason: str = ""

    def can_execute(
        self,
        model_req: "Any",  # ModelRequirements — avoid circular import
    ) -> "Any":  # ExecutionDecision
        """Evaluate whether a model can execute given current constraints."""
        from control_panel.common.models import ExecutionDecision, ModelRequirements

        if not isinstance(model_req, ModelRequirements):
            from control_panel.common.models import ModelRequirements as MR
            model_req = MR(**model_req) if isinstance(model_req, dict) else model_req

        needed = model_req.estimated_ram_mb + self.safety_margin_mb
        if needed > self.available_ram_mb:
            return ExecutionDecision(
                allowed=False,
                reason=(
                    f"Model needs ~{model_req.estimated_ram_mb} MB RAM + "
                    f"{self.safety_margin_mb} MB margin = {needed} MB, "
                    f"but only {self.available_ram_mb} MB available"
                ),
                suggested_alternative="api",
            )
        if self.power_policy == "critical":
            return ExecutionDecision(
                allowed=False,
                reason="Power policy is critical — local model execution deferred",
                suggested_alternative="rule_based",
            )
        return ExecutionDecision(
            allowed=True,
            reason=f"RAM sufficient ({self.available_ram_mb - needed} MB headroom)",
        )
