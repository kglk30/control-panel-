"""Experience Collector — aggregates WorkerResults into a complete Experience.

Called by the Orchestrator in the COLLECTING state after all workers complete.
Records: backend/device/model per inference call, resource snapshots,
worker outcomes, traces, and the overall outcome.
"""

from __future__ import annotations

from typing import Optional

from control_panel.common.models import (
    ExecutionTrace,
    InferenceRecord,
    InferenceResult,
    WorkerOutcome,
    utc_now,
    new_id,
)
from control_panel.common.logging import get_logger
from control_panel.experience.models import (
    Experience,
    ExperienceExecution,
    ExperienceInference,
    ExperiencePlan,
    ExperienceResourceContext,
    ExperienceTask,
    ExperienceOutcome,
)
from control_panel.resources.models import ResourceSnapshot

__all__ = ["ExperienceCollector"]

log = get_logger(__name__)


class ExperienceCollector:
    """Aggregates all task execution data into a single Experience record.

    The Orchestrator builds a collector, adds inference records and worker
    traces as they complete, then calls build() to produce the final Experience.
    """

    def __init__(
        self,
        task_id: str,
        raw_input: str,
        category: str,
        task_spec: dict,
        plan: dict,
        template_id: Optional[str] = None,
        template_version: Optional[int] = None,
    ) -> None:
        self._task_id = task_id
        self._raw_input = raw_input
        self._category = category
        self._task_spec = task_spec
        self._plan = plan
        self._template_id = template_id
        self._template_version = template_version

        self._inference_records: list[InferenceRecord] = []
        self._traces: list[ExecutionTrace] = []
        self._worker_outcomes: list[WorkerOutcome] = []
        self._snapshot_start: Optional[ResourceSnapshot] = None
        self._snapshot_end: Optional[ResourceSnapshot] = None
        self._peak_memory_mb: float = 0.0
        self._accelerators_used: list[str] = []
        self.execution_mode: str = "sequential"
        self.concurrency_limit: int = 1
        self.wall_clock_ms: int = 0
        self.parallelizable_fraction: float = 0.0

    # ------------------------------------------------------------------
    # Collection API (called during execution)
    # ------------------------------------------------------------------

    def set_start_snapshot(self, snapshot: ResourceSnapshot) -> None:
        """Record resource state at task start."""
        self._snapshot_start = snapshot

    def set_end_snapshot(self, snapshot: ResourceSnapshot) -> None:
        """Record resource state at task end."""
        self._snapshot_end = snapshot

    def record_inference(self, result: InferenceResult) -> None:
        """Add one InferenceResult to the experience record."""
        record = InferenceRecord(
            request_task_type="unknown",  # Will be set properly via record_inference_call
            backend_used=result.backend_used,
            device_used=result.device_used,
            runtime_type=result.runtime_type,
            model_id=result.model_id,
            latency_ms=result.latency_ms,
            tokens_used=result.tokens_used,
            success=True,
        )
        self._inference_records.append(record)

        # Track which accelerators were actually used
        if result.device_used not in ("none", "remote", "cpu") and \
                result.device_used not in self._accelerators_used:
            self._accelerators_used.append(result.device_used)

    def record_inference_call(self, record: InferenceRecord) -> None:
        """Add a complete InferenceRecord (includes task_type)."""
        self._inference_records.append(record)
        if record.device_used not in ("none", "remote", "cpu") and \
                record.device_used not in self._accelerators_used:
            self._accelerators_used.append(record.device_used)

    def add_trace(self, trace: ExecutionTrace) -> None:
        """Add a completed worker trace."""
        self._traces.append(trace)

    def add_worker_outcome(self, outcome: WorkerOutcome) -> None:
        """Add a worker outcome summary."""
        self._worker_outcomes.append(outcome)

    def set_peak_memory(self, peak_mb: float) -> None:
        """Record peak memory usage during execution."""
        self._peak_memory_mb = peak_mb

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        success: bool,
        items_processed: int = 0,
        items_failed: int = 0,
        error_summary: Optional[str] = None,
    ) -> Experience:
        """Build and return the complete Experience record."""
        # Determine primary backend used
        primary_backend = "rule_based"
        primary_device = "none"
        if self._inference_records:
            from collections import Counter
            backend_counts = Counter(r.backend_used for r in self._inference_records)
            primary_backend = backend_counts.most_common(1)[0][0]
            device_counts = Counter(r.device_used for r in self._inference_records)
            primary_device = device_counts.most_common(1)[0][0]

        experience = Experience(
            task_id=self._task_id,
            created_at=utc_now(),
            task=ExperienceTask(
                raw_input=self._raw_input,
                task_spec_json=self._task_spec,
                category=self._category,
            ),
            plan=ExperiencePlan(
                template_id=self._template_id,
                template_version=self._template_version,
                execution_plan_json=self._plan,
                worker_count=self.concurrency_limit,
                execution_mode=self.execution_mode,
            ),
            inference=ExperienceInference(calls=self._inference_records),
            execution=ExperienceExecution(
                traces=self._traces,
                total_runtime_ms=self.wall_clock_ms,
                worker_outcomes=self._worker_outcomes,
            ),
            resource_context=ExperienceResourceContext(
                snapshot_at_start=self._snapshot_start,
                snapshot_at_end=self._snapshot_end,
                peak_memory_mb=self._peak_memory_mb,
                accelerators_used=self._accelerators_used,
            ),
            outcome=ExperienceOutcome(
                success=success,
                completeness=1.0 if success else (
                    items_processed / (items_processed + items_failed)
                    if (items_processed + items_failed) > 0 else 0.0
                ),
                items_processed=items_processed,
                items_failed=items_failed,
                error_summary=error_summary,
            ),
            backend_used=primary_backend,
            device_used=primary_device,
            strategy_type=self.execution_mode,
            worker_count=self.concurrency_limit,
            parallelizable_fraction=self.parallelizable_fraction,
        )

        log.info(
            "experience.built",
            task_id=self._task_id,
            category=self._category,
            success=success,
            workers=len(self._worker_outcomes),
            inference_calls=len(self._inference_records),
        )
        return experience
