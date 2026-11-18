"""Planner — converts TaskSpec + templates + resources into an ExecutionPlan.

Produces a list of WorkerSpecs in execution order, with tools and permissions.
Respects ResourceConstraints from the Governor.
"""

from __future__ import annotations

import json
from typing import Optional

from control_panel.common.logging import get_logger
from control_panel.common.models import (
    ExecutionPlan,
    ExecutionStrategy,
    TaskSpec,
    WorkerConstraints,
    WorkerSpec,
    new_id,
    utc_now,
)
from control_panel.resources.models import ResourceConstraints
from control_panel.templates.models import Template

__all__ = ["Planner"]

log = get_logger(__name__)


class Planner:
    """Produces ExecutionPlan from TaskSpec, template, and resource constraints.

    The Planner does NOT select inference backends.
    It produces WorkerSpecs that tell the Factory what tools each worker gets.
    """

    def plan(
        self,
        task_spec: TaskSpec,
        constraints: ResourceConstraints,
        template: Optional[Template] = None,
        target_path: Optional[str] = None,
        strategy: Optional[ExecutionStrategy] = None,
    ) -> ExecutionPlan:
        """Create an ExecutionPlan for the given task.

        Args:
            task_spec: Analyzed task specification.
            constraints: Current resource constraints.
            template: Best matching template (if any).
            target_path: Target filesystem path from user.
            strategy: Recommended execution strategy.

        Returns:
            ExecutionPlan with ordered WorkerSpecs.
        """
        log.info(
            "planner.start",
            category=task_spec.category,
            template=template.id if template else None,
        )

        if template and template.category.startswith("file_management"):
            workers = self._plan_file_management(task_spec, template, target_path)
        else:
            workers = self._plan_generic(task_spec, target_path)

        plan = ExecutionPlan(
            task_id=task_spec.task_id,
            template_id=template.id if template else None,
            template_version=template.version if template else None,
            workers=workers,
            execution_mode=strategy.strategy_type if strategy else "sequential",
            rollback_strategy="log_and_reverse",
            created_at=utc_now(),
        )

        log.info(
            "planner.complete",
            workers=len(workers),
            execution_mode=plan.execution_mode,
        )
        return plan

    # ------------------------------------------------------------------
    # File management planning
    # ------------------------------------------------------------------

    def _plan_file_management(
        self,
        task_spec: TaskSpec,
        template: Template,
        target_path: Optional[str],
    ) -> list[WorkerSpec]:
        """Create workers for a file organization task using the template."""
        workers_config = template.workflow.get("workers", [])
        workers: list[WorkerSpec] = []

        for wc in workers_config:
            role = wc.get("role", "worker")
            tools = wc.get("tools", [])
            permission_mode = wc.get("permission_mode", "read_only")
            max_runtime = wc.get("max_runtime_seconds", 60)

            input_data: dict = {}
            if target_path:
                input_data["target_path"] = target_path
                input_data["categories"] = template.workflow.get("categories", {})

            spec = WorkerSpec(
                worker_id=new_id(),
                task_id=task_spec.task_id,
                role=role,
                tools_granted=tools,
                permission_mode=permission_mode,
                input_data=input_data,
                constraints=WorkerConstraints(
                    max_runtime_seconds=max_runtime,
                    max_retries=2,
                    allowed_paths=[target_path] if target_path else [],
                ),
            )
            workers.append(spec)

        return workers

    # ------------------------------------------------------------------
    # Generic planning
    # ------------------------------------------------------------------

    def _plan_generic(
        self,
        task_spec: TaskSpec,
        target_path: Optional[str],
    ) -> list[WorkerSpec]:
        """Create a simple single-worker plan for unknown task types."""
        input_data: dict = {}
        if target_path:
            input_data["target_path"] = target_path

        return [
            WorkerSpec(
                worker_id=new_id(),
                task_id=task_spec.task_id,
                role="general_executor",
                tools_granted=["fs.list_directory", "fs.get_metadata"],
                permission_mode="read_only",
                input_data=input_data,
                constraints=WorkerConstraints(
                    max_runtime_seconds=60,
                    max_retries=2,
                    allowed_paths=[target_path] if target_path else [],
                ),
            )
        ]
