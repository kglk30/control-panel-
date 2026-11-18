"""Template Store — SQLite-backed versioned template storage.

Templates are reusable workflow strategies. The Orchestrator retrieves
matching templates during the RETRIEVING state.

V0.1 supports: CREATE NEW and UPDATE operations (not MERGE).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from control_panel.common.logging import get_logger
from control_panel.common.models import utc_now
from control_panel.persistence.database import Database, get_database
from control_panel.templates.models import Template, TemplatePerformance

__all__ = ["TemplateStore"]

log = get_logger(__name__)

# Path to the built-in seed template
_BUILTIN_DIR = Path(__file__).parent / "builtins"


class TemplateStore:
    """SQLite-backed store for workflow templates.

    Implements the TemplateStore protocol from V15 §13.6.
    """

    def __init__(self, db: Optional[Database] = None) -> None:
        self._db = db
        self._seeded = False

    async def _get_db(self) -> Database:
        if self._db is None:
            self._db = await get_database()
        return self._db

    async def initialize(self) -> None:
        """Seed built-in templates if not already present."""
        if self._seeded:
            return
        await self._seed_builtins()
        self._seeded = True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def find_by_category(self, category: str) -> list[Template]:
        """Return the latest version of all templates matching a category prefix."""
        db = await self._get_db()
        # Match category prefix (e.g. "file_management" matches "file_management.organization")
        rows = await db.fetch_all(
            """
            SELECT t.* FROM templates t
            INNER JOIN (
                SELECT id, MAX(version) as max_ver FROM templates
                WHERE category LIKE ?
                GROUP BY id
            ) latest ON t.id = latest.id AND t.version = latest.max_ver
            ORDER BY t.usage_count DESC
            """,
            (f"{category}%",),
        )
        return [self._row_to_template(r) for r in rows]

    async def get(self, template_id: str) -> Optional[Template]:
        """Return the latest version of a template by ID."""
        db = await self._get_db()
        row = await db.fetch_one(
            "SELECT * FROM templates WHERE id = ? ORDER BY version DESC LIMIT 1",
            (template_id,),
        )
        return self._row_to_template(row) if row else None

    async def list_all(self) -> list[Template]:
        """Return all templates (latest versions only)."""
        db = await self._get_db()
        rows = await db.fetch_all(
            """
            SELECT t.* FROM templates t
            INNER JOIN (
                SELECT id, MAX(version) as max_ver FROM templates GROUP BY id
            ) latest ON t.id = latest.id AND t.version = latest.max_ver
            ORDER BY t.category, t.name
            """
        )
        return [self._row_to_template(r) for r in rows]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def create(self, template: Template) -> None:
        """Insert a new template or new version."""
        db = await self._get_db()
        now = utc_now().isoformat()
        await db.execute(
            """
            INSERT INTO templates
                (id, version, name, category, description, source,
                 workflow_json, requirements_json, performance_json,
                 usage_count, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                template.id,
                template.version,
                template.name,
                template.category,
                template.description,
                template.source,
                json.dumps(template.workflow),
                json.dumps(template.requirements),
                json.dumps(template.performance.model_dump()),
                template.usage_count,
                now,
                now,
            ),
        )
        log.info("template.created", id=template.id, version=template.version)

    async def update(self, template_id: str, updates: dict) -> None:
        """Update fields on the latest version of a template."""
        existing = await self.get(template_id)
        if existing is None:
            log.warning("template.update_not_found", id=template_id)
            return

        for key, value in updates.items():
            if hasattr(existing, key):
                setattr(existing, key, value)

        existing.updated_at = utc_now()
        db = await self._get_db()
        await db.execute(
            """
            UPDATE templates SET
                name=?, description=?, workflow_json=?,
                requirements_json=?, performance_json=?,
                usage_count=?, updated_at=?
            WHERE id=? AND version=?
            """,
            (
                existing.name,
                existing.description,
                json.dumps(existing.workflow),
                json.dumps(existing.requirements),
                json.dumps(existing.performance.model_dump()),
                existing.usage_count,
                existing.updated_at.isoformat(),
                template_id,
                existing.version,
            ),
        )
        log.info("template.updated", id=template_id)

    async def increment_usage(self, template_id: str, success: bool) -> None:
        """Increment usage_count and update success_rate."""
        existing = await self.get(template_id)
        if existing is None:
            return

        existing.usage_count += 1
        perf = existing.performance
        old_count = perf.usage_count
        perf.usage_count = existing.usage_count
        # Running average success rate
        if old_count > 0:
            perf.success_rate = (
                (perf.success_rate * old_count) + (1.0 if success else 0.0)
            ) / existing.usage_count
        else:
            perf.success_rate = 1.0 if success else 0.0

        db = await self._get_db()
        await db.execute(
            "UPDATE templates SET usage_count=?, performance_json=?, updated_at=? WHERE id=? AND version=?",
            (
                existing.usage_count,
                json.dumps(perf.model_dump()),
                utc_now().isoformat(),
                template_id,
                existing.version,
            ),
        )

    # ------------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------------

    async def _seed_builtins(self) -> None:
        """Load built-in templates from JSON files if not already in DB."""
        builtin_files = list(_BUILTIN_DIR.glob("*.json"))
        for fpath in builtin_files:
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                template_id = data.get("id", fpath.stem)
                existing = await self.get(template_id)
                if existing is None:
                    tmpl = Template(
                        id=template_id,
                        version=data.get("version", 1),
                        name=data.get("name", fpath.stem),
                        category=data.get("category", "general"),
                        description=data.get("description"),
                        source="builtin",
                        workflow=data.get("workflow", {}),
                        requirements=data.get("requirements", {}),
                    )
                    await self.create(tmpl)
                    log.info("template.seeded", id=template_id, file=fpath.name)
            except Exception as e:
                log.warning("template.seed_failed", file=str(fpath), error=str(e))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row_to_template(self, row: object) -> Template:
        r = dict(row)  # type: ignore[call-overload]
        perf_data = json.loads(r.get("performance_json") or "{}")
        perf = TemplatePerformance(**perf_data) if perf_data else TemplatePerformance()
        from datetime import datetime, timezone
        def parse_dt(s: str) -> datetime:
            try:
                return datetime.fromisoformat(s)
            except Exception:
                return datetime.now(timezone.utc)
        return Template(
            id=r["id"],
            version=r["version"],
            name=r["name"],
            category=r["category"],
            description=r.get("description"),
            source=r.get("source", "builtin"),
            workflow=json.loads(r.get("workflow_json") or "{}"),
            requirements=json.loads(r.get("requirements_json") or "{}"),
            performance=perf,
            usage_count=r.get("usage_count", 0),
            created_at=parse_dt(r.get("created_at", "")),
            updated_at=parse_dt(r.get("updated_at", "")),
        )
