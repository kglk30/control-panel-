"""Tool Registry — catalog of all available tools, injectable into workers.

Workers receive a scoped subset of tools at creation time.
The registry is the single source of truth for tool lookup.
"""

from __future__ import annotations

from typing import Optional

from control_panel.common.logging import get_logger
from control_panel.tools.base import Tool

__all__ = ["ToolRegistry", "get_registry"]

log = get_logger(__name__)


class ToolRegistry:
    """Catalog of all registered Tool instances.

    Tools are registered at startup. Workers receive a view of only
    the tools explicitly granted to them by the Planner.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises ValueError if name already registered."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        log.debug("tool.registered", name=tool.name, category=tool.category)

    def register_many(self, tools: list[Tool]) -> None:
        """Register multiple tools at once."""
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Optional[Tool]:
        """Return a tool by name, or None if not found."""
        return self._tools.get(name)

    def get_required(self, name: str) -> Tool:
        """Return a tool by name. Raises KeyError if not found."""
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not found in registry")
        return tool

    def get_subset(self, names: list[str]) -> dict[str, Tool]:
        """Return a dict of tools for the given names.

        Raises KeyError if any name is not registered.
        Used by Worker Factory to build per-worker tool scopes.
        """
        result: dict[str, Tool] = {}
        for name in names:
            result[name] = self.get_required(name)
        return result

    def list_all(self) -> list[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    def list_by_category(self, category: str) -> list[Tool]:
        """Return all tools in a given category."""
        return [t for t in self._tools.values() if t.category == category]

    def __len__(self) -> int:
        return len(self._tools)


# ---------------------------------------------------------------------------
# Module-level singleton with default filesystem tools pre-registered
# ---------------------------------------------------------------------------

_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Return the global ToolRegistry singleton.

    Automatically registers all built-in filesystem tools on first call.
    """
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        _register_defaults(_registry)
    return _registry


def _register_defaults(registry: ToolRegistry) -> None:
    """Register all built-in tools into the registry."""
    from control_panel.tools.filesystem import (
        ListDirectoryTool,
        GetMetadataTool,
        CreateDirectoryTool,
        MoveFileTool,
        CopyFileTool,
    )
    registry.register_many([
        ListDirectoryTool(),
        GetMetadataTool(),
        CreateDirectoryTool(),
        MoveFileTool(),
        CopyFileTool(),
    ])
    log.info("tool_registry.initialized", count=len(registry))
