"""Common shared data models used across the entire Control Panel system."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "utc_now",
    "new_id",
    "TaskSpec",
    "WorkerConstraints",
    "WorkerSpec",
    "ExecutionPlan",
    "InferenceRequest",
    "InferenceResult",
    "BackendSelection",
    "BackendInfo",
    "SpawnDecision",
    "ExecutionDecision",
    "ResourceRequirements",
    "ModelRequirements",
    "SideEffect",
    "ToolResult",
    "WorkerEvent",
    "TraceStep",
    "TraceSummary",
    "ExecutionTrace",
    "WorkerOutcome",
    "InferenceRecord",
    "ExecutionStrategy",
]


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def new_id() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


class TaskSpec(BaseModel):
    """Structured representation of a user task produced by the Task Analyzer."""

    model_config = ConfigDict(frozen=False)

    task_id: str = Field(default_factory=new_id)
    raw_input: str
    objective: str
    subtasks: list[str] = Field(default_factory=list)
    category: str  # e.g. "file_management.organization"
    novelty_score: float = 0.5  # 0.0-1.0
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class WorkerConstraints(BaseModel):
    """Runtime constraints applied to a single worker."""

    model_config = ConfigDict(frozen=False)

    max_runtime_seconds: int = 60
    max_retries: int = 3
    allowed_paths: list[str] = Field(default_factory=list)


class WorkerSpec(BaseModel):
    """Specification for creating a single ephemeral worker."""

    model_config = ConfigDict(frozen=False)

    worker_id: str = Field(default_factory=new_id)
    task_id: str
    role: str  # e.g. "file_scanner"
    tools_granted: list[str] = Field(default_factory=list)
    permission_mode: str = "read_only"  # "read_only" | "read_write"
    input_data: dict[str, Any] = Field(default_factory=dict)
    constraints: WorkerConstraints = Field(default_factory=WorkerConstraints)


class ExecutionPlan(BaseModel):
    """Full plan produced by the Planner: which workers, in what order, with what tools."""

    model_config = ConfigDict(frozen=False)

    plan_id: str = Field(default_factory=new_id)
    task_id: str
    template_id: Optional[str] = None
    template_version: Optional[int] = None
    workers: list[WorkerSpec] = Field(default_factory=list)
    execution_mode: str = "sequential"  # "sequential" | "parallel"
    rollback_strategy: str = "log_and_reverse"
    resource_snapshot_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class InferenceRequest(BaseModel):
    """Unified request submitted by the Orchestrator to the Inference Runtime."""

    model_config = ConfigDict(frozen=False)

    request_id: str = Field(default_factory=new_id)
    prompt: str
    context: Optional[dict[str, Any]] = None
    response_schema: Optional[dict[str, Any]] = None  # JSON Schema for structured output
    task_type: str  # "analyze"|"classify"|"plan"|"reason"|"generate"|"embed"
    temperature: float = 0.3
    max_tokens: int = 1024
    created_at: datetime = Field(default_factory=utc_now)


class InferenceResult(BaseModel):
    """Result returned by the Inference Runtime to the Orchestrator."""

    model_config = ConfigDict(frozen=False)

    request_id: str
    content: str
    structured: Optional[dict[str, Any]] = None
    backend_used: str  # "snapdragon"|"local_cpu"|"api"|"rule_based"
    device_used: str  # "npu"|"gpu"|"cpu"|"remote"|"none"
    runtime_type: str  # "genai"|"structured"|"api"|"rule_based"
    model_id: str
    tokens_used: int = 0
    latency_ms: int = 0
    source: str  # "local"|"remote"|"rule_based"


class BackendSelection(BaseModel):
    """Output of the Backend Selection Engine."""

    model_config = ConfigDict(frozen=False)

    backend: str  # "snapdragon"|"local_cpu"|"api"|"rule_based"
    device: str  # "npu"|"gpu"|"cpu"|"remote"|"none"
    runtime_type: str  # "genai"|"structured"|"api"|"rule_based"
    model_id: str
    reason: str
    alternatives: list[str] = Field(default_factory=list)


class BackendInfo(BaseModel):
    """Current availability status of a backend adapter."""

    model_config = ConfigDict(frozen=False)

    name: str
    available: bool
    device: Optional[str] = None
    reason: Optional[str] = None  # Why unavailable, if so


class SpawnDecision(BaseModel):
    """Decision from Resource Governor on whether a worker can be spawned."""

    model_config = ConfigDict(frozen=False)

    allowed: bool
    reason: str
    suggested_alternative: Optional[str] = None


class ExecutionDecision(BaseModel):
    """Decision on whether a model can execute given current resources."""

    model_config = ConfigDict(frozen=False)

    allowed: bool
    reason: str
    suggested_alternative: Optional[str] = None


class ResourceRequirements(BaseModel):
    """Minimum resource thresholds for spawning a worker."""

    model_config = ConfigDict(frozen=False)

    min_free_ram_mb: int = 512
    max_cpu_percent: float = 80.0
    min_free_disk_mb: int = 100
    estimated_ram_mb: Optional[int] = None


class ModelRequirements(BaseModel):
    """Resource requirements declared by a ModelProfile."""

    model_config = ConfigDict(frozen=False)

    disk_size_mb: int = 0
    estimated_ram_mb: int = 0
    minimum_free_ram_mb: int = 512
    accelerator_requirements: Optional[str] = None  # "qnn"|"directml"|None


class SideEffect(BaseModel):
    """A reversible side effect produced by a tool execution."""

    model_config = ConfigDict(frozen=False)

    type: str  # e.g. "file_move", "dir_create"
    params: dict[str, Any]
    reversible: bool
    undo_params: Optional[dict[str, Any]] = None


class ToolResult(BaseModel):
    """Result returned by any Tool after execution."""

    model_config = ConfigDict(frozen=False)

    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: int = 0
    side_effects: list[SideEffect] = Field(default_factory=list)


class WorkerEvent(BaseModel):
    """Progress event emitted during worker execution."""

    model_config = ConfigDict(frozen=False)

    event_type: str  # "started"|"step_complete"|"completed"|"failed"
    worker_id: str
    task_id: str
    message: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


class TraceStep(BaseModel):
    """One recorded step in a worker's execution trace."""

    model_config = ConfigDict(frozen=False)

    step_index: int
    tool_name: str
    tool_input: dict[str, Any]
    tool_output_summary: str
    started_at: datetime
    completed_at: datetime
    duration_ms: int
    status: str  # "success"|"error"|"skipped"|"retried"
    error: Optional[str] = None
    retry_attempt: int = 0


class TraceSummary(BaseModel):
    """Aggregate statistics for a completed execution trace."""

    model_config = ConfigDict(frozen=False)

    total_steps: int
    successful_steps: int
    failed_steps: int
    retried_steps: int
    total_runtime_ms: int
    peak_memory_mb: float = 0.0


class ExecutionTrace(BaseModel):
    """Full execution trace for a single worker."""

    model_config = ConfigDict(frozen=False)

    trace_id: str = Field(default_factory=new_id)
    worker_id: str
    task_id: str
    worker_role: str
    tools_granted: list[str] = Field(default_factory=list)
    permission_mode: str = "read_only"
    allowed_paths: list[str] = Field(default_factory=list)
    max_runtime_seconds: int = 60
    max_retries: int = 3
    steps: list[TraceStep] = Field(default_factory=list)
    summary: Optional[TraceSummary] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class WorkerOutcome(BaseModel):
    """High-level outcome of a single worker (stored in Experience)."""

    model_config = ConfigDict(frozen=False)

    worker_id: str
    role: str
    success: bool
    retry_count: int = 0
    runtime_ms: int = 0
    error: Optional[str] = None


class InferenceRecord(BaseModel):
    """Record of a single inference call within a task execution (stored in Experience)."""

    model_config = ConfigDict(frozen=False)

    request_task_type: str  # "analyze"|"classify"|"plan"|...
    backend_used: str
    device_used: str
    runtime_type: str
    model_id: str
    latency_ms: int
    tokens_used: int = 0
    success: bool = True
    error: Optional[str] = None

class ExecutionStrategy(BaseModel):
    """Output of the ExecutionStrategyPlanner detailing how to execute a task."""

    model_config = ConfigDict(frozen=False)

    strategy_type: str  # "direct"|"single_worker"|"sequential"|"parallel"|"hybrid"
    recommended_worker_count: int
    parallelizable_fraction: float
    theoretical_speedup: float
    rationale: str
