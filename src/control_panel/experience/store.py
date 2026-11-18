"""Experience Store — SQLite-backed persistence for task execution experiences."""

from __future__ import annotations


import json
from datetime import datetime
from typing import Any, Optional


def _json_dumps(obj: Any) -> str:
    """JSON serializer that handles datetime objects."""
    class _Encoder(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            if isinstance(o, datetime):
                return o.isoformat()
            return super().default(o)
    return json.dumps(obj, cls=_Encoder)



from control_panel.common.logging import get_logger
from control_panel.experience.models import (
    BackendExperience,
    CategoryStats,
    Experience,
)
from control_panel.persistence.database import Database, get_database

__all__ = ["ExperienceStore"]

log = get_logger(__name__)


class ExperienceStore:
    """Persist and query Experience records in SQLite.

    Implements the ExperienceStore protocol from V15 §13.6.
    """

    def __init__(self, db: Optional[Database] = None) -> None:
        self._db = db

    async def _get_db(self) -> Database:
        if self._db is None:
            self._db = await get_database()
        return self._db

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def save(self, experience: Experience) -> None:
        """Persist an Experience to SQLite."""
        db = await self._get_db()
        await db.execute(
            """
            INSERT OR REPLACE INTO experiences (
                id, task_id, category, raw_input,
                task_spec_json, plan_json, inference_json,
                execution_json, resource_context_json,
                outcome_json, strategy_json,
                success, total_runtime_ms,
                backend_used, device_used,
                strategy_type, worker_count, parallelizable_fraction,
                is_archived, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                experience.id,
                experience.task_id,
                experience.task.category,
                experience.task.raw_input,
                _json_dumps(experience.task.task_spec_json),
                _json_dumps(experience.plan.model_dump()),
                _json_dumps(experience.inference.model_dump()),
                _json_dumps(experience.execution.model_dump()),
                _json_dumps(experience.resource_context.model_dump()),
                _json_dumps(experience.outcome.model_dump()),
                _json_dumps(experience.strategy_decision.model_dump())
                    if experience.strategy_decision else None,
                1 if experience.outcome.success else 0,
                experience.execution.total_runtime_ms,
                experience.backend_used,
                experience.device_used,
                experience.strategy_type,
                experience.worker_count,
                experience.parallelizable_fraction,
                1 if experience.is_archived else 0,
                experience.created_at.isoformat(),
            ),
        )
        log.info(
            "experience.saved",
            id=experience.id,
            category=experience.task.category,
            success=experience.outcome.success,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def find_by_category(
        self,
        category: str,
        limit: int = 10,
        include_archived: bool = False,
    ) -> list[Experience]:
        """Return the most recent experiences for a task category."""
        db = await self._get_db()
        archived_filter = "" if include_archived else "AND is_archived = 0"
        rows = await db.fetch_all(
            f"""
            SELECT * FROM experiences
            WHERE category = ? {archived_filter}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (category, limit),
        )
        return [self._row_to_experience(r) for r in rows]

    async def find_by_template(self, template_id: str) -> list[Experience]:
        """Return experiences that used a specific template."""
        db = await self._get_db()
        rows = await db.fetch_all(
            "SELECT * FROM experiences WHERE plan_json LIKE ? ORDER BY created_at DESC",
            (f'%"template_id": "{template_id}"%',),
        )
        return [self._row_to_experience(r) for r in rows]

    async def get_statistics(self, category: str) -> CategoryStats:
        """Return aggregate statistics for a category."""
        db = await self._get_db()
        row = await db.fetch_one(
            """
            SELECT
                COUNT(*) as total,
                SUM(success) as successes,
                AVG(total_runtime_ms) as avg_runtime,
                backend_used
            FROM experiences
            WHERE category = ? AND is_archived = 0
            GROUP BY category
            """,
            (category,),
        )
        if row is None:
            return CategoryStats(category=category)

        total = row["total"] or 0
        successes = row["successes"] or 0
        return CategoryStats(
            category=category,
            total_executions=total,
            successful_executions=successes,
            success_rate=successes / total if total > 0 else 0.0,
            avg_runtime_ms=float(row["avg_runtime"] or 0),
            most_used_backend=row["backend_used"] or "",
        )

    async def get_backend_experience(self, backend: str) -> BackendExperience:
        """Return latency/success stats for a specific backend."""
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT inference_json FROM experiences
            WHERE backend_used = ? AND is_archived = 0
            ORDER BY created_at DESC LIMIT 50
            """,
            (backend,),
        )

        calls_total = 0
        calls_success = 0
        latencies: list[float] = []

        for row in rows:
            try:
                inf_data = json.loads(row["inference_json"])
                for call in inf_data.get("calls", []):
                    if call.get("backend_used") == backend:
                        calls_total += 1
                        if call.get("success", True):
                            calls_success += 1
                        lat = call.get("latency_ms", 0)
                        if lat > 0:
                            latencies.append(lat)
            except Exception:
                pass

        return BackendExperience(
            backend=backend,
            total_calls=calls_total,
            successful_calls=calls_success,
            success_rate=calls_success / calls_total if calls_total > 0 else 0.0,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
            min_latency_ms=min(latencies) if latencies else 0.0,
            max_latency_ms=max(latencies) if latencies else 0.0,
        )

    async def list_recent(self, limit: int = 20) -> list[Experience]:
        """Return the most recent experiences across all categories."""
        db = await self._get_db()
        rows = await db.fetch_all(
            "SELECT * FROM experiences WHERE is_archived = 0 ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_experience(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row_to_experience(self, row: object) -> Experience:
        """Convert a SQLite row to an Experience object."""
        from control_panel.experience.models import (
            ExperienceExecution,
            ExperienceInference,
            ExperiencePlan,
            ExperienceResourceContext,
            ExperienceTask,
            ExperienceOutcome,
            StrategyDecision,
        )
        from datetime import datetime, timezone

        def parse_dt(s: str) -> datetime:
            try:
                return datetime.fromisoformat(s)
            except Exception:
                return datetime.now(timezone.utc)

        r = dict(row)  # type: ignore[call-overload]

        task = ExperienceTask(
            raw_input=r.get("raw_input", ""),
            task_spec_json=json.loads(r.get("task_spec_json") or "{}"),
            category=r.get("category", ""),
        )

        plan_data = json.loads(r.get("plan_json") or "{}")
        plan = ExperiencePlan(**plan_data) if plan_data else ExperiencePlan()

        inf_data = json.loads(r.get("inference_json") or "{}")
        inference = ExperienceInference(**inf_data) if inf_data else ExperienceInference()

        exec_data = json.loads(r.get("execution_json") or "{}")
        execution = ExperienceExecution(
            total_runtime_ms=exec_data.get("total_runtime_ms", r.get("total_runtime_ms", 0)),
            worker_outcomes=exec_data.get("worker_outcomes", []),
            traces=exec_data.get("traces", []),
        )

        rc_data = json.loads(r.get("resource_context_json") or "{}")
        resource_context = ExperienceResourceContext(
            accelerators_used=rc_data.get("accelerators_used", []),
            peak_memory_mb=rc_data.get("peak_memory_mb", 0.0),
        )

        out_data = json.loads(r.get("outcome_json") or "{}")
        outcome = ExperienceOutcome(
            success=bool(r.get("success", 0)),
            completeness=out_data.get("completeness", 1.0),
            items_processed=out_data.get("items_processed", 0),
            items_failed=out_data.get("items_failed", 0),
            error_summary=out_data.get("error_summary"),
        )

        strategy = None
        strat_json = r.get("strategy_json")
        if strat_json:
            try:
                strategy = StrategyDecision(**json.loads(strat_json))
            except Exception:
                pass

        return Experience(
            id=r.get("id", ""),
            task_id=r.get("task_id", ""),
            created_at=parse_dt(r.get("created_at", "")),
            task=task,
            plan=plan,
            inference=inference,
            execution=execution,
            resource_context=resource_context,
            outcome=outcome,
            strategy_decision=strategy,
            backend_used=r.get("backend_used", ""),
            device_used=r.get("device_used", ""),
            strategy_type=r.get("strategy_type", "sequential"),
            worker_count=r.get("worker_count", 0) or 0,
            parallelizable_fraction=r.get("parallelizable_fraction", 0.0) or 0.0,
            is_archived=bool(r.get("is_archived", 0)),
        )
