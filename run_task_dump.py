    async def run_task(
        self,
        raw_input: str,
        target_path: Optional[str] = None,
        on_progress: Optional[Callable] = None,
    ) -> OrchestratorResult:
        """Execute a complete task lifecycle.

        Args:
            raw_input: Natural-language task description from the user.
            target_path: Filesystem path to operate on.
            on_progress: Optional progress callback.

        Returns:
            OrchestratorResult with outcome, trace, and backend info.
        """
        if self._state != OrchestratorState.IDLE:
            raise RuntimeError(
                f"Orchestrator is busy (state={self._state.value}). "
                "Only one task at a time in V0.1."
            )

        task_id = new_id()
        t_global = time.perf_counter()

        task_spec: Optional[TaskSpec] = None
        plan: Optional[ExecutionPlan] = None
        worker_results: list[WorkerResult] = []
        backend_used = "rule_based"
        device_used = "none"
        items_processed = 0

        try:
            # ── ANALYZING ─────────────────────────────────────────────
            self._transition(OrchestratorState.ANALYZING, task_id=task_id)
            self._check_timeout(t_global)

            task_spec = self._analyzer.analyze(raw_input)
            task_spec.task_id = task_id

            # Capture start snapshot
            start_snapshot = self._governor.snapshot()
            collector = ExperienceCollector(
                task_id=task_id,
                raw_input=raw_input,
                category=task_spec.category,
                task_spec=task_spec.model_dump(mode="json"),
                plan={},
                template_id=None,
            )
            collector.set_start_snapshot(start_snapshot)

            # ── RETRIEVING ────────────────────────────────────────────
            self._transition(OrchestratorState.RETRIEVING, task_id=task_id)
            self._check_timeout(t_global)

            templates = await self._template_store.find_by_category(task_spec.category)
            best_template = templates[0] if templates else None

            history = await self._experience_store.find_by_category(
                task_spec.category, limit=10
            )

            # ── PLANNING ──────────────────────────────────────────────
            self._transition(OrchestratorState.PLANNING, task_id=task_id)
            self._check_timeout(t_global)

            constraints = self._governor.get_resource_constraints()
            plan = self._planner.plan(
                task_spec=task_spec,
                constraints=constraints,
                template=best_template,
                target_path=target_path,
            )
            collector._plan = plan.model_dump(mode="json")
            if best_template:
                collector._template_id = best_template.id
                collector._template_version = best_template.version

            # ── SPAWNING ──────────────────────────────────────────────
            self._transition(OrchestratorState.SPAWNING, task_id=task_id)
            self._check_timeout(t_global)

            # Resource gate
            spawn_decision = self._governor.can_spawn()
            if not spawn_decision.allowed:
                raise RuntimeError(f"Cannot spawn workers: {spawn_decision.reason}")

            # Safety check on target path
            if target_path:
                from control_panel.safety.models import SafetyContext
                dummy_ctx = SafetyContext(
                    worker_id="orchestrator",
                    task_id=task_id,
                    permission_mode="read_write",
                    allowed_paths=[],
                )
                path_check = self._safety.validate_path(
                    target_path, dummy_ctx, is_write=False
                )
                if not path_check.allowed:
                    raise RuntimeError(f"Target path rejected: {path_check.reason}")

            workers = [self._worker_factory.create(spec) for spec in plan.workers]

            # ── EXECUTING ─────────────────────────────────────────────
            self._transition(OrchestratorState.EXECUTING, task_id=task_id)
            self._check_timeout(t_global)

            worker_results = self._worker_runtime.execute_sequential(
                workers, on_progress=on_progress
            )

            # ── COLLECTING ────────────────────────────────────────────
            self._transition(OrchestratorState.COLLECTING, task_id=task_id)

            # Determine what was processed
            items_processed = self._count_items(worker_results)
            items_failed = sum(1 for r in worker_results if not r.success)
            overall_success = all(r.success for r in worker_results)

            # Collect traces and outcomes
            for r in worker_results:
                if r.trace:
                    collector.add_trace(r.trace)
                collector.add_worker_outcome(WorkerOutcome(
                    worker_id=r.worker_id,
                    role=r.role,
                    success=r.success,
                    retry_count=r.retry_count,
                    runtime_ms=r.runtime_ms,
                    error=r.error,
                ))

            end_snapshot = self._governor.snapshot()
            collector.set_end_snapshot(end_snapshot)

            # Determine primary backend used
            if worker_results:
                backend_used = start_snapshot.accelerators.get("npu") and \
                    start_snapshot.accelerators["npu"].available and "snapdragon" or "local_cpu"
                # Use actual inference backend from the inference runtime
                backends = self._inference.available_backends()
                for b in backends:
                    if b.available and b.name != "rule_based":
                        backend_used = b.name
                        device_used = b.device or "cpu"
                        break

            experience = collector.build(
                success=overall_success,
                items_processed=items_processed,
                items_failed=items_failed,
            )
            experience.backend_used = backend_used
            experience.device_used = device_used

            # ── COMPARING ─────────────────────────────────────────────
            self._transition(OrchestratorState.COMPARING, task_id=task_id)

            strategy = self._comparator.compare(experience, history)
            experience.strategy_decision = strategy

            # ── OPTIMIZING ────────────────────────────────────────────
            self._transition(OrchestratorState.OPTIMIZING, task_id=task_id)

            # Apply strategy decision
            if strategy.action == "UPDATE" and strategy.target_template and best_template:
                await self._template_store.update(
                    best_template.id,
                    {"workflow": best_template.workflow}
                )
            if best_template:
                await self._template_store.increment_usage(
                    best_template.id, success=overall_success
                )

            # ── STORING ───────────────────────────────────────────────
            self._transition(OrchestratorState.STORING, task_id=task_id)
            await self._experience_store.save(experience)

            self._bus.emit_simple(
                EventType.EXPERIENCE_STORED,
                task_id=task_id,
                experience_id=experience.id,
                success=overall_success,
            )

            return OrchestratorResult(
                success=overall_success,
                task_id=task_id,
                task_spec=task_spec,
                plan=plan,
                worker_results=worker_results,
                backend_used=backend_used,
                device_used=device_used,
                total_runtime_ms=int((time.perf_counter() - t_global) * 1000),
                items_processed=items_processed,
            )

        except Exception as e:
            log.exception("orchestrator.error", state=self._state.value, error=str(e))
            return OrchestratorResult(
                success=False,
                task_id=task_id,
                task_spec=task_spec,
                plan=plan,
                worker_results=worker_results,
                backend_used=backend_used,
                device_used=device_used,
                total_runtime_ms=int((time.perf_counter() - t_global) * 1000),
                items_processed=items_processed,
                error=str(e),
            )

        finally:
            # ── CLEANUP ───────────────────────────────────────────────
            self._transition(OrchestratorState.CLEANUP, task_id=task_id)
            # Workers are already destroyed by WorkerRuntime
            self._transition(OrchestratorState.IDLE, task_id=task_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _transition(self, new_state: OrchestratorState, task_id: Optional[str] = None) -> None:
        """Transition to a new state and emit a STATE_CHANGED event."""
        old_state = self._state
        self._state = new_state
        log.info("orchestrator.state", from_state=old_state.value, to_state=new_state.value)
        self._bus.emit_simple(
            EventType.STATE_CHANGED,
            task_id=task_id,
            from_state=old_state.value,
            to_state=new_state.value,
        )

    def _check_timeout(self, t_global: float) -> None:
        """Raise TimeoutError if the global task timeout has been exceeded."""
        elapsed = time.perf_counter() - t_global
        if elapsed > self._timeout_seconds:
            raise TimeoutError(
                f"Task exceeded global timeout of {self._timeout_seconds}s "
                f"(elapsed {elapsed:.1f}s)"
            )

    def _count_items(self, worker_results: list[WorkerResult]) -> int:
        """Count items processed across all workers."""
        count = 0
        for r in worker_results:
            if isinstance(r.output, dict):
                count += r.output.get("count", 0)
                count += len(r.output.get("entries", []))
                count += len(r.output.get("moves", []))
        return max(count, sum(1 for r in worker_results if r.success))

    @property
    def state(self) -> OrchestratorState:
        """Current orchestrator state."""
        return self._state
