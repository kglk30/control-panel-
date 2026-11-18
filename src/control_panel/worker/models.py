"""Worker data models — Worker instance that executes within a task."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from control_panel.common.models import ExecutionTrace, WorkerSpec, new_id

__all__ = ["WorkerState", "WorkerResult", "Worker"]


class WorkerState(str, Enum):
    """Worker lifecycle states."""

    CONFIGURING = "configuring"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    DESTROYED = "destroyed"


class WorkerResult(BaseModel):
    """Output produced by a completed or failed worker."""

    model_config = ConfigDict(frozen=False)

    worker_id: str
    task_id: str
    role: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    trace: Optional[ExecutionTrace] = None
    retry_count: int = 0
    runtime_ms: int = 0


class Worker(BaseModel):
    """A single ephemeral worker created by WorkerFactory.

    Workers are in-process execution contexts. They are created from a
    WorkerSpec, execute their assigned tools, and are destroyed when done.
    They share the underlying inference runtime — they do NOT get separate model copies.
    """

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    worker_id: str = Field(default_factory=new_id)
    task_id: str
    spec: WorkerSpec
    state: WorkerState = WorkerState.CONFIGURING

    # Injected at creation time
    tools: dict[str, Any] = Field(default_factory=dict)  # name → Tool instance
    safety_context: Any = None  # SafetyContext

    # Input data passed forward from previous workers
    input_data: dict[str, Any] = Field(default_factory=dict)
