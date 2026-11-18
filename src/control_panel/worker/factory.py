"""Worker Factory — creates Worker instances from WorkerSpec.

Injects the granted tools and scoped SafetyContext at creation time.
Workers receive exactly the tools listed in their WorkerSpec — no more.
"""

from __future__ import annotations

from control_panel.common.logging import get_logger
from control_panel.common.models import WorkerSpec
from control_panel.safety.manager import SafetyManager
from control_panel.safety.models import SafetyContext
from control_panel.tools.registry import ToolRegistry, get_registry
from control_panel.worker.models import Worker, WorkerState

__all__ = ["WorkerFactory"]

log = get_logger(__name__)


class WorkerFactory:
    """Creates Worker instances with properly scoped tools and safety context.

    Architectural invariant: Workers cannot expand their own permissions.
    The Planner decides what tools each worker gets; Factory enforces it.
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        safety_manager: Optional[SafetyManager] = None,
        dry_run: bool = False,
    ) -> None:
        from typing import Optional
        self._registry = tool_registry or get_registry()
        self._safety = safety_manager or SafetyManager()
        self._dry_run = dry_run

    def create(self, spec: WorkerSpec) -> Worker:
        """Create a Worker from a WorkerSpec.

        Args:
            spec: The WorkerSpec produced by the Planner.

        Returns:
            Worker in CONFIGURING → READY state.

        Raises:
            KeyError: If any tool in spec.tools_granted is not registered.
        """
        # Get the exact set of tools granted — no extras
        tools = self._registry.get_subset(spec.tools_granted)

        # Create scoped SafetyContext for this worker
        safety_ctx = SafetyContext(
            worker_id=spec.worker_id,
            task_id=spec.task_id,
            permission_mode=spec.permission_mode,
            allowed_paths=spec.constraints.allowed_paths,
            dry_run=self._dry_run,
        )

        worker = Worker(
            worker_id=spec.worker_id,
            task_id=spec.task_id,
            spec=spec,
            state=WorkerState.READY,
            tools=tools,
            safety_context=safety_ctx,
            input_data=spec.input_data,
        )

        log.info(
            "worker.created",
            worker_id=spec.worker_id,
            role=spec.role,
            tools=list(tools.keys()),
            permission_mode=spec.permission_mode,
        )
        return worker


# Allow Optional import in method signature
from typing import Optional  # noqa: E402
