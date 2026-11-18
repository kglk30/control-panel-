"""Template data models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from control_panel.common.models import utc_now

__all__ = ["Template", "TemplatePerformance"]


class TemplatePerformance(BaseModel):
    """Runtime performance statistics for a template."""

    model_config = ConfigDict(frozen=False)

    success_rate: float = 0.0
    avg_runtime_ms: float = 0.0
    usage_count: int = 0


class Template(BaseModel):
    """A reusable workflow strategy template."""

    model_config = ConfigDict(frozen=False)

    id: str
    version: int = 1
    name: str
    category: str
    description: Optional[str] = None
    source: str = "builtin"  # "builtin" | "learned"
    workflow: dict[str, Any] = Field(default_factory=dict)
    requirements: dict[str, Any] = Field(default_factory=dict)
    performance: TemplatePerformance = Field(default_factory=TemplatePerformance)
    usage_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
