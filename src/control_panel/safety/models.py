"""Safety Manager data models — SafetyContext and related types."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["SafetyContext", "SafetyDecision"]


class SafetyContext(BaseModel):
    """Per-worker safety scope injected at worker creation time.

    Defines the exact filesystem permissions and sandbox for one worker.
    Workers cannot expand their own SafetyContext.
    """

    model_config = ConfigDict(frozen=False)

    worker_id: str
    task_id: str
    permission_mode: str  # "read_only" | "read_write"
    allowed_paths: list[str] = Field(default_factory=list)
    dry_run: bool = False


class SafetyDecision(BaseModel):
    """Result of a Safety Manager validation check."""

    model_config = ConfigDict(frozen=False)

    allowed: bool
    reason: str
    sanitized_path: Optional[str] = None  # Resolved absolute path if allowed
