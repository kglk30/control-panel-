"""Experience data models — full V14/V15 experience schema."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from control_panel.common.models import (
    ExecutionTrace,
    InferenceRecord,
    WorkerOutcome,
    new_id,
    utc_now,
)
from control_panel.resources.models import ResourceSnapshot

__all__ = [
    "ExperienceOutcome",
    "StrategyDecision",
    "ExperienceInference",
    "ExperienceExecution",
    "ExperienceResourceContext",
    "ExperiencePlan",
    "ExperienceTask",
    "Experience",
    "CategoryStats",
    "BackendExperience",
]


class ExperienceOutcome(BaseModel):
    """Final outcome assessment of an experience."""

    model_config = ConfigDict(frozen=False)

    success: bool
    completeness: float = 1.0      # 0.0–1.0
    items_processed: int = 0
    items_failed: int = 0
    error_summary: Optional[str] = None


class StrategyDecision(BaseModel):
    """Orchestrator's final strategy decision after comparing experiences."""

    model_config = ConfigDict(frozen=False)

    action: str  # KEEP | UPDATE | MERGE | ARCHIVE | DELETE | CREATE_NEW
    reason: str
    target_template: Optional[str] = None


class ExperienceInference(BaseModel):
    """All inference calls made during a task execution."""

    model_config = ConfigDict(frozen=False)

    calls: list[InferenceRecord] = Field(default_factory=list)


class ExperienceExecution(BaseModel):
    """Execution results for all workers in a task."""

    model_config = ConfigDict(frozen=False)

    traces: list[ExecutionTrace] = Field(default_factory=list)
    total_runtime_ms: int = 0
    worker_outcomes: list[WorkerOutcome] = Field(default_factory=list)


class ExperienceResourceContext(BaseModel):
    """Resource state at start and end of a task."""

    model_config = ConfigDict(frozen=False)

    snapshot_at_start: Optional[ResourceSnapshot] = None
    snapshot_at_end: Optional[ResourceSnapshot] = None
    peak_memory_mb: float = 0.0
    accelerators_used: list[str] = Field(default_factory=list)


class ExperiencePlan(BaseModel):
    """What was planned for this task."""

    model_config = ConfigDict(frozen=False)

    template_id: Optional[str] = None
    template_version: Optional[int] = None
    execution_plan_json: dict[str, Any] = Field(default_factory=dict)
    worker_count: int = 0
    execution_mode: str = "sequential"
    strategy_version: int = 1  # Tracks strategy evolution


class ExperienceTask(BaseModel):
    """What was requested."""

    model_config = ConfigDict(frozen=False)

    raw_input: str
    task_spec_json: dict[str, Any] = Field(default_factory=dict)
    category: str = ""


class Experience(BaseModel):
    """Complete record of one task execution, stored in SQLite."""

    model_config = ConfigDict(frozen=False)

    id: str = Field(default_factory=new_id)
    task_id: str = Field(default_factory=new_id)
    created_at: datetime = Field(default_factory=utc_now)

    task: ExperienceTask = Field(default_factory=ExperienceTask.model_construct)
    plan: ExperiencePlan = Field(default_factory=ExperiencePlan)
    inference: ExperienceInference = Field(default_factory=ExperienceInference)
    execution: ExperienceExecution = Field(default_factory=ExperienceExecution)
    resource_context: ExperienceResourceContext = Field(
        default_factory=ExperienceResourceContext
    )
    outcome: ExperienceOutcome = Field(
        default_factory=lambda: ExperienceOutcome(success=False)
    )
    strategy_decision: Optional[StrategyDecision] = None

    # Fast-query columns (denormalized)
    backend_used: str = ""
    device_used: str = ""
    strategy_type: str = "sequential"
    worker_count: int = 0
    parallelizable_fraction: float = 0.0
    is_archived: bool = False

    def primary_backend(self) -> str:
        """Return the most-used backend across inference calls."""
        if not self.inference.calls:
            return self.backend_used or "rule_based"
        # Most common backend
        from collections import Counter
        counts = Counter(c.backend_used for c in self.inference.calls)
        return counts.most_common(1)[0][0]


class CategoryStats(BaseModel):
    """Aggregate statistics for a task category."""

    model_config = ConfigDict(frozen=False)

    category: str
    total_executions: int = 0
    successful_executions: int = 0
    success_rate: float = 0.0
    avg_runtime_ms: float = 0.0
    most_used_backend: str = ""


class BackendExperience(BaseModel):
    """Historical performance metrics for one backend."""

    model_config = ConfigDict(frozen=False)

    backend: str
    total_calls: int = 0
    successful_calls: int = 0
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
