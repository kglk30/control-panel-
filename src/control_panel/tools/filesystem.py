"""Filesystem tools for Control Panel workers.

All operations:
  - Are validated by SafetyManager before execution
  - Record SideEffect entries for undo logging
  - Support dry_run simulation

V0.1 tools: list_directory, get_metadata, create_directory, move_file, copy_file
Forbidden in V0.1: delete_file, delete_directory (require confirmation UI)
"""

from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from control_panel.common.models import SideEffect, ToolResult
from control_panel.common.logging import get_logger
from control_panel.safety.manager import SafetyManager
from control_panel.safety.models import SafetyContext
from control_panel.tools.base import Tool

__all__ = [
    "ListDirectoryTool",
    "GetMetadataTool",
    "CreateDirectoryTool",
    "MoveFileTool",
    "CopyFileTool",
]

log = get_logger(__name__)

# Shared Safety Manager — tools import the same instance as the Orchestrator
_safety: SafetyManager | None = None


def _get_safety() -> SafetyManager:
    global _safety
    if _safety is None:
        _safety = SafetyManager()
    return _safety


def _now_ms() -> float:
    return time.perf_counter() * 1000


# ---------------------------------------------------------------------------
# fs.list_directory
# ---------------------------------------------------------------------------

class ListDirectoryTool(Tool):
    """List the contents of a directory."""

    @property
    def name(self) -> str:
        return "fs.list_directory"

    @property
    def category(self) -> str:
        return "filesystem"

    @property
    def permissions(self) -> set[str]:
        return {"read"}

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list"},
                "recursive": {"type": "boolean", "default": False},
            },
            "required": ["path"],
        }

    def execute(self, params: dict[str, Any], safety_ctx: SafetyContext) -> ToolResult:
        t0 = _now_ms()
        path = params.get("path", "")
        recursive = params.get("recursive", False)

        decision = _get_safety().validate_path(path, safety_ctx, is_write=False)
        if not decision.allowed:
            return ToolResult(success=False, error=decision.reason,
                              duration_ms=int(_now_ms() - t0))

        p = Path(decision.sanitized_path)  # type: ignore[arg-type]
        if not p.exists():
            return ToolResult(success=False, error=f"Path does not exist: {p}",
                              duration_ms=int(_now_ms() - t0))
        if not p.is_dir():
            return ToolResult(success=False, error=f"Path is not a directory: {p}",
                              duration_ms=int(_now_ms() - t0))

        try:
            if recursive:
                entries = [
                    {"path": str(e), "name": e.name, "is_dir": e.is_dir(),
                     "size": e.stat().st_size if e.is_file() else 0}
                    for e in sorted(p.rglob("*"))
                ]
            else:
                entries = [
                    {"path": str(e), "name": e.name, "is_dir": e.is_dir(),
                     "size": e.stat().st_size if e.is_file() else 0}
                    for e in sorted(p.iterdir())
                ]
        except PermissionError as e:
            return ToolResult(success=False, error=str(e),
                              duration_ms=int(_now_ms() - t0))

        return ToolResult(
            success=True,
            output={"directory": str(p), "entries": entries, "count": len(entries)},
            duration_ms=int(_now_ms() - t0),
        )


# ---------------------------------------------------------------------------
# fs.get_metadata
# ---------------------------------------------------------------------------

class GetMetadataTool(Tool):
    """Get metadata for a file or directory."""

    @property
    def name(self) -> str:
        return "fs.get_metadata"

    @property
    def category(self) -> str:
        return "filesystem"

    @property
    def permissions(self) -> set[str]:
        return {"read"}

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File or directory path"},
            },
            "required": ["path"],
        }

    def execute(self, params: dict[str, Any], safety_ctx: SafetyContext) -> ToolResult:
        t0 = _now_ms()
        path = params.get("path", "")

        decision = _get_safety().validate_path(path, safety_ctx, is_write=False)
        if not decision.allowed:
            return ToolResult(success=False, error=decision.reason,
                              duration_ms=int(_now_ms() - t0))

        p = Path(decision.sanitized_path)  # type: ignore[arg-type]
        if not p.exists():
            return ToolResult(success=False, error=f"Path does not exist: {p}",
                              duration_ms=int(_now_ms() - t0))

        try:
            stat = p.stat()
            metadata = {
                "path": str(p),
                "name": p.name,
                "suffix": p.suffix.lower(),
                "is_file": p.is_file(),
                "is_dir": p.is_dir(),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                "created_at": datetime.fromtimestamp(
                    stat.st_ctime, tz=timezone.utc
                ).isoformat(),
            }
        except Exception as e:
            return ToolResult(success=False, error=str(e),
                              duration_ms=int(_now_ms() - t0))

        return ToolResult(success=True, output=metadata, duration_ms=int(_now_ms() - t0))


# ---------------------------------------------------------------------------
# fs.create_directory
# ---------------------------------------------------------------------------

class CreateDirectoryTool(Tool):
    """Create a directory (and parents if needed)."""

    @property
    def name(self) -> str:
        return "fs.create_directory"

    @property
    def category(self) -> str:
        return "filesystem"

    @property
    def permissions(self) -> set[str]:
        return {"read", "write"}

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to create"},
                "exist_ok": {"type": "boolean", "default": True},
            },
            "required": ["path"],
        }

    def execute(self, params: dict[str, Any], safety_ctx: SafetyContext) -> ToolResult:
        t0 = _now_ms()
        path = params.get("path", "")
        exist_ok = params.get("exist_ok", True)

        tool_check = _get_safety().validate_tool(self.name, safety_ctx)
        if not tool_check.allowed:
            return ToolResult(success=False, error=tool_check.reason,
                              duration_ms=int(_now_ms() - t0))

        decision = _get_safety().validate_path(path, safety_ctx, is_write=True)
        if not decision.allowed:
            return ToolResult(success=False, error=decision.reason,
                              duration_ms=int(_now_ms() - t0))

        p = Path(decision.sanitized_path)  # type: ignore[arg-type]

        if _get_safety().is_dry_run(safety_ctx):
            return ToolResult(
                success=True,
                output={"dry_run": True, "would_create": str(p)},
                duration_ms=int(_now_ms() - t0),
                side_effects=[SideEffect(
                    type="dir_create", params={"path": str(p)},
                    reversible=True, undo_params={"path": str(p)},
                )],
            )

        try:
            already_existed = p.exists()
            p.mkdir(parents=True, exist_ok=exist_ok)
            return ToolResult(
                success=True,
                output={"path": str(p), "already_existed": already_existed},
                duration_ms=int(_now_ms() - t0),
                side_effects=[] if already_existed else [
                    SideEffect(
                        type="dir_create", params={"path": str(p)},
                        reversible=True, undo_params={"path": str(p)},
                    )
                ],
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), duration_ms=int(_now_ms() - t0))


# ---------------------------------------------------------------------------
# fs.move_file
# ---------------------------------------------------------------------------

class MoveFileTool(Tool):
    """Move a file to a new location. Records undo entry."""

    @property
    def name(self) -> str:
        return "fs.move_file"

    @property
    def category(self) -> str:
        return "filesystem"

    @property
    def permissions(self) -> set[str]:
        return {"read", "write"}

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source file path"},
                "destination": {"type": "string", "description": "Destination path"},
            },
            "required": ["source", "destination"],
        }

    def execute(self, params: dict[str, Any], safety_ctx: SafetyContext) -> ToolResult:
        t0 = _now_ms()
        source = params.get("source", "")
        destination = params.get("destination", "")

        tool_check = _get_safety().validate_tool(self.name, safety_ctx)
        if not tool_check.allowed:
            return ToolResult(success=False, error=tool_check.reason,
                              duration_ms=int(_now_ms() - t0))

        src_decision = _get_safety().validate_path(source, safety_ctx, is_write=False)
        if not src_decision.allowed:
            return ToolResult(success=False, error=f"Source: {src_decision.reason}",
                              duration_ms=int(_now_ms() - t0))

        dst_decision = _get_safety().validate_path(destination, safety_ctx, is_write=True)
        if not dst_decision.allowed:
            return ToolResult(success=False, error=f"Destination: {dst_decision.reason}",
                              duration_ms=int(_now_ms() - t0))

        src = Path(src_decision.sanitized_path)  # type: ignore[arg-type]
        dst = Path(dst_decision.sanitized_path)  # type: ignore[arg-type]

        if not src.exists():
            return ToolResult(success=False, error=f"Source does not exist: {src}",
                              duration_ms=int(_now_ms() - t0))

        side_effect = SideEffect(
            type="file_move",
            params={"source": str(src), "destination": str(dst)},
            reversible=True,
            undo_params={"source": str(dst), "destination": str(src)},
        )

        if _get_safety().is_dry_run(safety_ctx):
            return ToolResult(
                success=True,
                output={"dry_run": True, "would_move": str(src), "to": str(dst)},
                duration_ms=int(_now_ms() - t0),
                side_effects=[side_effect],
            )

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            log.info("tool.move_file", source=str(src), destination=str(dst))
            return ToolResult(
                success=True,
                output={"moved": str(src), "to": str(dst)},
                duration_ms=int(_now_ms() - t0),
                side_effects=[side_effect],
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), duration_ms=int(_now_ms() - t0))


# ---------------------------------------------------------------------------
# fs.copy_file
# ---------------------------------------------------------------------------

class CopyFileTool(Tool):
    """Copy a file to a new location."""

    @property
    def name(self) -> str:
        return "fs.copy_file"

    @property
    def category(self) -> str:
        return "filesystem"

    @property
    def permissions(self) -> set[str]:
        return {"read", "write"}

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source file path"},
                "destination": {"type": "string", "description": "Destination path"},
            },
            "required": ["source", "destination"],
        }

    def execute(self, params: dict[str, Any], safety_ctx: SafetyContext) -> ToolResult:
        t0 = _now_ms()
        source = params.get("source", "")
        destination = params.get("destination", "")

        tool_check = _get_safety().validate_tool(self.name, safety_ctx)
        if not tool_check.allowed:
            return ToolResult(success=False, error=tool_check.reason,
                              duration_ms=int(_now_ms() - t0))

        src_decision = _get_safety().validate_path(source, safety_ctx, is_write=False)
        if not src_decision.allowed:
            return ToolResult(success=False, error=f"Source: {src_decision.reason}",
                              duration_ms=int(_now_ms() - t0))

        dst_decision = _get_safety().validate_path(destination, safety_ctx, is_write=True)
        if not dst_decision.allowed:
            return ToolResult(success=False, error=f"Destination: {dst_decision.reason}",
                              duration_ms=int(_now_ms() - t0))

        src = Path(src_decision.sanitized_path)  # type: ignore[arg-type]
        dst = Path(dst_decision.sanitized_path)  # type: ignore[arg-type]

        if not src.exists():
            return ToolResult(success=False, error=f"Source does not exist: {src}",
                              duration_ms=int(_now_ms() - t0))

        side_effect = SideEffect(
            type="file_copy",
            params={"source": str(src), "destination": str(dst)},
            reversible=True,
            undo_params={"delete": str(dst)},
        )

        if _get_safety().is_dry_run(safety_ctx):
            return ToolResult(
                success=True,
                output={"dry_run": True, "would_copy": str(src), "to": str(dst)},
                duration_ms=int(_now_ms() - t0),
                side_effects=[side_effect],
            )

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            return ToolResult(
                success=True,
                output={"copied": str(src), "to": str(dst)},
                duration_ms=int(_now_ms() - t0),
                side_effects=[side_effect],
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), duration_ms=int(_now_ms() - t0))
