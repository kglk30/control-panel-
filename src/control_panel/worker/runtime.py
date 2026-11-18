"""Worker Runtime — executes workers sequentially with trace capture, retry, timeout.

V0.1: execute_sequential() only.
V0.2+: execute_parallel() raises NotImplementedError (extension point).

Each worker's execution is fully traced. Partial results are returned on failure.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Optional

from control_panel.common.events import EventType, get_event_bus
from control_panel.common.logging import get_logger
from control_panel.common.models import (
    ExecutionTrace,
    TraceSummary,
    TraceStep,
    WorkerEvent,
    WorkerOutcome,
    utc_now,
)
from control_panel.worker.models import Worker, WorkerResult, WorkerState

__all__ = ["WorkerRuntime"]

log = get_logger(__name__)

ProgressCallback = Callable[[WorkerEvent], None]


class WorkerRuntime:
    """Executes Worker instances sequentially with full trace capture.

    Passes each worker's output as input_data to the next worker.
    """

    def __init__(self) -> None:
        import threading
        self._bus = get_event_bus()
        self._progress_lock = threading.Lock()

    def execute_sequential(
        self,
        workers: list[Worker],
        on_progress: Optional[ProgressCallback] = None,
    ) -> list[WorkerResult]:
        """Execute workers one by one, passing output forward.

        Args:
            workers: Ordered list of workers from the Planner.
            on_progress: Optional callback for progress events.

        Returns:
            List of WorkerResult (one per worker). Partial results on failure.
        """
        results: list[WorkerResult] = []
        accumulated_output: dict = {}

        for worker in workers:
            # Inject accumulated output from previous workers
            worker.input_data.update(accumulated_output)

            result = self._execute_worker(worker, on_progress)
            results.append(result)

            if result.success and isinstance(result.output, dict):
                accumulated_output.update(result.output)

            # On failure, still return partial results — don't silently drop them
            if not result.success:
                log.warning(
                    "runtime.worker_failed",
                    worker_id=worker.worker_id,
                    role=worker.spec.role,
                    error=result.error,
                )
                # Continue with remaining workers (best-effort)

        return results

    def execute_parallel(
        self,
        workers: list[Worker],
        graph: "DependencyGraph",
        n_max: int,
        on_progress: Optional[ProgressCallback] = None,
    ) -> list[WorkerResult]:
        """Execute workers concurrently in topological dependency groups."""
        import concurrent.futures

        results: list[WorkerResult] = []
        accumulated_output: dict = {}
        failed_roles = set()

        # Map roles to worker instances
        worker_by_role = {w.spec.role: w for w in workers}

        # Get execution groups from dependency graph
        groups = graph.parallel_groups()

        for group in groups:
            group_workers = [worker_by_role[role] for role in group if role in worker_by_role]

            if not group_workers:
                continue

            runnable_workers = []
            group_results: list[WorkerResult] = []

            for worker in group_workers:
                deps = graph.get_dependencies(worker.spec.role)
                if any(dep in failed_roles for dep in deps):
                    failed_roles.add(worker.spec.role)
                    group_results.append(WorkerResult(
                        task_id=worker.spec.task_id,
                        worker_id=worker.spec.worker_id,
                        role=worker.spec.role,
                        success=False,
                        error="Prerequisite failed or skipped."
                    ))
                else:
                    runnable_workers.append(worker)

            if not runnable_workers:
                results.extend(group_results)
                continue

            # Inject accumulated output from previous groups
            for worker in runnable_workers:
                worker.input_data.update(accumulated_output)

            # Execute the group concurrently respecting n_max
            max_workers = min(len(runnable_workers), max(1, n_max))

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._execute_worker, worker, on_progress)
                    for worker in runnable_workers
                ]

                for future in futures:
                    group_results.append(future.result())

            results.extend(group_results)

            # Check for failures and merge output
            for r in group_results:
                if r.success and isinstance(r.output, dict):
                    accumulated_output.update(r.output)
                if not r.success:
                    failed_roles.add(r.role)
                    log.warning(
                        "runtime.parallel_worker_failed",
                        worker_id=r.worker_id,
                        role=r.role,
                        error=r.error
                    )

        return results

    # ------------------------------------------------------------------
    # Single worker execution
    # ------------------------------------------------------------------

    def _execute_worker(
        self,
        worker: Worker,
        on_progress: Optional[ProgressCallback],
    ) -> WorkerResult:
        """Execute one worker with retry and timeout, capturing a full trace."""
        spec = worker.spec
        max_retries = spec.constraints.max_retries
        max_runtime_s = spec.constraints.max_runtime_seconds

        # Build trace skeleton
        trace = ExecutionTrace(
            worker_id=worker.worker_id,
            task_id=worker.task_id,
            worker_role=spec.role,
            tools_granted=spec.tools_granted,
            permission_mode=spec.permission_mode,
            allowed_paths=spec.constraints.allowed_paths,
            max_runtime_seconds=max_runtime_s,
            max_retries=max_retries,
        )

        worker.state = WorkerState.EXECUTING
        trace.started_at = utc_now()
        t0 = time.perf_counter()

        self._emit(on_progress, WorkerEvent(
            event_type="started",
            worker_id=worker.worker_id,
            task_id=worker.task_id,
            message=f"Worker '{spec.role}' starting",
        ))
        self._bus.emit_simple(
            EventType.WORKER_STARTED,
            task_id=worker.task_id,
            worker_id=worker.worker_id,
            role=spec.role,
        )

        last_error: Optional[str] = None
        retry_count = 0
        step_index = 0
        success = False
        output: dict = {}

        for attempt in range(max_retries + 1):
            try:
                # Check timeout
                elapsed = time.perf_counter() - t0
                if elapsed > max_runtime_s:
                    last_error = (
                        f"Worker timed out after {elapsed:.1f}s "
                        f"(max {max_runtime_s}s)"
                    )
                    break

                # Execute the worker's logic
                step_result = self._run_worker_logic(
                    worker, trace, step_index, on_progress, t0, max_runtime_s
                )
                output = step_result
                success = True
                break

            except Exception as e:
                last_error = str(e)
                retry_count = attempt
                log.warning(
                    "runtime.worker_retry",
                    worker_id=worker.worker_id,
                    attempt=attempt,
                    error=last_error,
                )
                if attempt < max_retries:
                    time.sleep(0.5 * (attempt + 1))

        runtime_ms = int((time.perf_counter() - t0) * 1000)
        trace.completed_at = utc_now()

        # Build trace summary
        successful_steps = sum(1 for s in trace.steps if s.status == "success")
        failed_steps = sum(1 for s in trace.steps if s.status == "error")
        retried_steps = sum(1 for s in trace.steps if s.status == "retried")
        trace.summary = TraceSummary(
            total_steps=len(trace.steps),
            successful_steps=successful_steps,
            failed_steps=failed_steps,
            retried_steps=retried_steps,
            total_runtime_ms=runtime_ms,
        )

        # Destroy worker
        worker.state = WorkerState.COMPLETED if success else WorkerState.FAILED

        event_type = "completed" if success else "failed"
        self._emit(on_progress, WorkerEvent(
            event_type=event_type,
            worker_id=worker.worker_id,
            task_id=worker.task_id,
            message=f"Worker '{spec.role}' {event_type}",
        ))
        self._bus.emit_simple(
            EventType.WORKER_COMPLETED if success else EventType.WORKER_FAILED,
            task_id=worker.task_id,
            worker_id=worker.worker_id,
            runtime_ms=runtime_ms,
            success=success,
        )

        worker.state = WorkerState.DESTROYED

        return WorkerResult(
            worker_id=worker.worker_id,
            task_id=worker.task_id,
            role=spec.role,
            success=success,
            output=output if success else None,
            error=last_error if not success else None,
            trace=trace,
            retry_count=retry_count,
            runtime_ms=runtime_ms,
        )

    def _run_worker_logic(
        self,
        worker: Worker,
        trace: ExecutionTrace,
        step_offset: int,
        on_progress: Optional[ProgressCallback],
        t0: float,
        max_runtime_s: float,
    ) -> dict:
        """Run the worker's tool sequence, recording each step in the trace."""
        spec = worker.spec
        safety_ctx = worker.safety_context
        combined_output: dict = dict(worker.input_data)
        step_index = step_offset

        # The worker logic: determine what to do based on role
        if spec.role == "file_classifier":
            import os
            entries = combined_output.get("entries", [])
            categories = combined_output.get("categories", {})
            target_path = combined_output.get("target_path", "")

            moves = []
            for entry in entries:
                if entry.get("is_dir"):
                    continue

                name = entry.get("name", "")
                path = entry.get("path", "")
                if not name or not path:
                    continue

                ext = os.path.splitext(name)[1].lower()
                cat_found = "Other"
                for cat, exts in categories.items():
                    if ext in exts:
                        cat_found = cat
                        break

                # Format paths properly. Handle cases where target_path might be empty.
                # Avoid os.path.join because paths might be mixed slashes; string concat is safe if target_path is normalized.
                # Better to use pathlib to be safe.
                import pathlib
                base = pathlib.Path(target_path) if target_path else pathlib.Path("")
                dest_dir = base / cat_found
                destination = dest_dir / name

                moves.append({
                    "source": path,
                    "dest_dir": str(dest_dir).replace('\\', '/'),
                    "destination": str(destination).replace('\\', '/')
                })

            combined_output["moves"] = moves

        # For V0.1, worker behavior is determined by which tools are available.
        # The Planner injects tool names; worker executes them in sequence.

        tool_sequence = self._get_tool_sequence(worker)

        for tool_name, params in tool_sequence:
            # Timeout check
            if time.perf_counter() - t0 > max_runtime_s:
                raise TimeoutError(f"Worker exceeded {max_runtime_s}s runtime")

            tool = worker.tools.get(tool_name)
            if tool is None:
                log.warning("runtime.tool_not_found", tool=tool_name, worker=worker.worker_id)
                continue

            step_start = utc_now()
            ts0 = time.perf_counter_ns()

            self._bus.emit_simple(
                EventType.TOOL_CALLED,
                task_id=worker.task_id,
                worker_id=worker.worker_id,
                tool=tool_name,
            )

            try:
                result = tool.execute(params, safety_ctx)
            except Exception as e:
                result_output_summary = f"ERROR: {e}"
                status = "error"
                error_str: Optional[str] = str(e)
                result = None
            else:
                result_output_summary = str(result.output)[:200] if result.output else ""
                status = "success" if result.success else "error"
                error_str = result.error
                if result.success and isinstance(result.output, dict):
                    combined_output.update(result.output)

            duration_ms = int((time.perf_counter_ns() - ts0) / 1_000_000)

            step = TraceStep(
                step_index=step_index,
                tool_name=tool_name,
                tool_input=params,
                tool_output_summary=result_output_summary,
                started_at=step_start,
                completed_at=utc_now(),
                duration_ms=duration_ms,
                status=status,
                error=error_str,
            )
            trace.steps.append(step)
            step_index += 1

            self._emit(on_progress, WorkerEvent(
                event_type="step_complete",
                worker_id=worker.worker_id,
                task_id=worker.task_id,
                message=f"{tool_name}: {status}",
            ))
            self._bus.emit_simple(
                EventType.TOOL_COMPLETED,
                task_id=worker.task_id,
                worker_id=worker.worker_id,
                tool=tool_name,
                status=status,
                duration_ms=duration_ms,
            )

        return combined_output

    def _get_tool_sequence(self, worker: Worker) -> list[tuple[str, dict]]:
        """Determine the tool call sequence for a worker based on its role and input.

        For V0.1, the sequence is derived from the worker's granted tools
        and input_data (the target path is injected by the Orchestrator).
        """
        spec = worker.spec
        target_path = worker.input_data.get("target_path", "")
        entries = worker.input_data.get("entries", [])

        sequence: list[tuple[str, dict]] = []

        if spec.role == "file_scanner":
            if "fs.list_directory" in worker.tools:
                sequence.append(("fs.list_directory", {"path": target_path}))

        elif spec.role == "file_classifier":
            # Classify each entry from scanner output
            for entry in entries[:100]:  # Safety cap
                if entry.get("is_dir"):
                    continue
                if "fs.get_metadata" in worker.tools:
                    sequence.append(("fs.get_metadata", {"path": entry.get("path", "")}))

        elif spec.role == "file_organizer":
            # Get category → path mappings from input
            moves = worker.input_data.get("moves", [])
            for move in moves:
                src = move.get("source", "")
                dst = move.get("destination", "")
                dst_dir = move.get("dest_dir", "")
                if dst_dir and "fs.create_directory" in worker.tools:
                    sequence.append(("fs.create_directory", {"path": dst_dir}))
                if src and dst and "fs.move_file" in worker.tools:
                    sequence.append(("fs.move_file", {"source": src, "destination": dst}))

        # Generic fallback: run available read tools on target path
        if not sequence and target_path:
            for tool_name in spec.tools_granted:
                if "list" in tool_name or "metadata" in tool_name:
                    sequence.append((tool_name, {"path": target_path}))
                    break

        return sequence

    def _emit(
        self,
        callback: Optional[ProgressCallback],
        event: WorkerEvent,
    ) -> None:
        """Safely call the progress callback in a thread-safe manner."""
        if callback:
            with self._progress_lock:
                try:
                    callback(event)
                except Exception:
                    pass
