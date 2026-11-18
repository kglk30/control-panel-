"""Tool base protocol and abstract base class for Control Panel tools.

Every tool must implement this interface. Tools are deterministic — they
never call the inference runtime. AI decides; tools execute safely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from control_panel.common.models import ToolResult
from control_panel.safety.models import SafetyContext

__all__ = ["Tool", "ToolError"]


class ToolError(Exception):
    """Raised when a tool encounters an unrecoverable error."""


class Tool(ABC):
    """Abstract base class for all Control Panel tools.

    Subclasses must implement `name`, `category`, `permissions`, `parameters`,
    and `execute()`.

    Tools are:
    - Deterministic (same input → same output, given same filesystem state)
    - Safety-gated (validated by SafetyManager before execution)
    - Traceable (return ToolResult with side_effects for undo logging)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier, e.g. 'fs.move_file'."""

    @property
    @abstractmethod
    def category(self) -> str:
        """Tool category, e.g. 'filesystem'."""

    @property
    @abstractmethod
    def permissions(self) -> set[str]:
        """Set of required permission strings, e.g. {'read', 'write'}."""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema describing the tool's input parameters."""

    @abstractmethod
    def execute(self, params: dict[str, Any], safety_ctx: SafetyContext) -> ToolResult:
        """Execute the tool with validated parameters.

        Args:
            params: Tool input parameters (validated against self.parameters).
            safety_ctx: The calling worker's safety context.

        Returns:
            ToolResult with success/failure, output, and any side effects.
        """

    def __repr__(self) -> str:
        return f"<Tool name={self.name!r} category={self.category!r}>"
