"""Safety Manager — final authority for all filesystem mutations.

Rules (V0.1):
  - Global denylist blocks access to system directories unconditionally.
  - Workers with 'read_only' permission cannot execute mutating tools.
  - dry_run=True simulates mutations without executing them.
  - No file deletion permitted in V0.1.
  - No shell execution permitted in V0.1.
  - Rate limiting enforced (max ops/sec configurable).

The Safety Manager is the ONLY component that validates filesystem operations.
No tool may bypass it.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Optional

from control_panel.common.config import get_config
from control_panel.common.logging import get_logger
from control_panel.safety.models import SafetyContext, SafetyDecision

__all__ = ["SafetyManager"]

log = get_logger(__name__)

# Tools that require write permission
_MUTATING_TOOLS = {"fs.move_file", "fs.copy_file", "fs.create_directory"}

# Tools that are forbidden in V0.1
_FORBIDDEN_V01 = {"fs.delete_file", "fs.delete_directory", "shell.execute"}


class SafetyManager:
    """Validates all tool operations before execution.

    Architectural invariant: Safety Manager is the final authority.
    The model/planner never receives unrestricted filesystem authority.
    """

    def __init__(
        self,
        denylist: Optional[list[str]] = None,
        max_ops_per_second: int = 10,
        dry_run: bool = False,
    ) -> None:
        cfg = get_config().safety
        self._denylist: list[Path] = [
            Path(p) for p in (denylist or cfg.denylist)
        ]
        self._max_ops_per_second = max_ops_per_second or cfg.max_ops_per_second
        self._global_dry_run = dry_run or cfg.dry_run
        # Sliding window for rate limiting (timestamps)
        self._op_timestamps: deque[float] = deque()

    # ------------------------------------------------------------------
    # Core validation — called by every tool before execution
    # ------------------------------------------------------------------

    def validate_path(
        self,
        path: str,
        context: SafetyContext,
        *,
        is_write: bool = False,
    ) -> SafetyDecision:
        """Validate a filesystem path against all safety rules.

        Args:
            path: The path the tool wants to access.
            context: The worker's safety context (permissions, allowed paths).
            is_write: True for mutating operations (move, copy, mkdir).

        Returns:
            SafetyDecision with allowed=True and resolved path, or allowed=False
            with a human-readable reason.
        """
        try:
            resolved = Path(path).resolve()
        except Exception as e:
            return SafetyDecision(
                allowed=False,
                reason=f"Path resolution failed: {e}",
            )

        # 1. Global denylist (non-overridable)
        for denied in self._denylist:
            try:
                resolved.relative_to(denied.resolve())
                return SafetyDecision(
                    allowed=False,
                    reason=f"Path is in global denylist: {denied}",
                )
            except ValueError:
                continue

        # 2. Write permission check
        if is_write and context.permission_mode == "read_only":
            return SafetyDecision(
                allowed=False,
                reason=(
                    f"Worker '{context.worker_id}' has read_only permission "
                    f"but attempted a write operation on {resolved}"
                ),
            )

        # 3. Allowed paths check (if non-empty, path must be within one)
        if context.allowed_paths:
            in_allowed = any(
                self._is_subpath(resolved, Path(ap))
                for ap in context.allowed_paths
            )
            if not in_allowed:
                return SafetyDecision(
                    allowed=False,
                    reason=(
                        f"Path {resolved} is not within worker's allowed paths: "
                        f"{context.allowed_paths}"
                    ),
                )

        return SafetyDecision(
            allowed=True,
            reason="Path validated",
            sanitized_path=str(resolved),
        )

    def validate_tool(self, tool_name: str, context: SafetyContext) -> SafetyDecision:
        """Validate that a tool is permitted for this worker.

        Args:
            tool_name: The tool being requested (e.g. "fs.move_file").
            context: The worker's safety context.
        """
        # V0.1 forbidden tools — no exceptions
        if tool_name in _FORBIDDEN_V01:
            return SafetyDecision(
                allowed=False,
                reason=f"Tool '{tool_name}' is forbidden in V0.1",
            )

        # Mutating tools require write permission
        if tool_name in _MUTATING_TOOLS and context.permission_mode == "read_only":
            return SafetyDecision(
                allowed=False,
                reason=(
                    f"Tool '{tool_name}' requires write permission, "
                    f"but worker '{context.worker_id}' is read_only"
                ),
            )

        # Rate limiting
        if not self._check_rate_limit():
            return SafetyDecision(
                allowed=False,
                reason=(
                    f"Rate limit exceeded: max {self._max_ops_per_second} ops/sec"
                ),
            )

        return SafetyDecision(allowed=True, reason="Tool permitted")

    def is_dry_run(self, context: SafetyContext) -> bool:
        """Return True if this operation should be simulated only."""
        return self._global_dry_run or context.dry_run

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_subpath(self, path: Path, parent: Path) -> bool:
        """Return True if path is equal to or under parent."""
        try:
            path.relative_to(parent.resolve())
            return True
        except ValueError:
            return False

    def _check_rate_limit(self) -> bool:
        """Sliding window rate limiter. Returns False if limit exceeded."""
        now = time.monotonic()
        # Remove timestamps older than 1 second
        while self._op_timestamps and now - self._op_timestamps[0] > 1.0:
            self._op_timestamps.popleft()
        if len(self._op_timestamps) >= self._max_ops_per_second:
            return False
        self._op_timestamps.append(now)
        return True
