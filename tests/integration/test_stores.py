"""Integration tests — SQLite database, template store, experience store."""

import asyncio
import pytest
import pytest_asyncio
from pathlib import Path

from control_panel.persistence.database import Database
from control_panel.templates.store import TemplateStore
from control_panel.templates.models import Template
from control_panel.experience.store import ExperienceStore
from control_panel.experience.models import (
    Experience,
    ExperienceTask,
    ExperienceOutcome,
    ExperienceExecution,
)


@pytest.fixture
async def db(tmp_path):
    """Async fixture — fresh in-memory DB per test."""
    db_path = tmp_path / "test.db"
    database = Database(db_path)
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
async def template_store(db):
    store = TemplateStore(db)
    await store.initialize()
    return store


@pytest.fixture
async def experience_store(db):
    return ExperienceStore(db)


class TestTemplateStore:
    async def test_seed_builtins(self, template_store) -> None:
        templates = await template_store.list_all()
        assert len(templates) >= 1
        ids = [t.id for t in templates]
        assert "file_organizer_v1" in ids

    async def test_create_and_get(self, template_store) -> None:
        tmpl = Template(
            id="test_tmpl",
            name="Test Template",
            category="test.category",
            source="builtin",
            workflow={"workers": []},
        )
        await template_store.create(tmpl)
        fetched = await template_store.get("test_tmpl")
        assert fetched is not None
        assert fetched.name == "Test Template"
        assert fetched.category == "test.category"

    async def test_find_by_category(self, template_store) -> None:
        templates = await template_store.find_by_category("file_management")
        assert len(templates) >= 1

    async def test_increment_usage(self, template_store) -> None:
        await template_store.increment_usage("file_organizer_v1", success=True)
        tmpl = await template_store.get("file_organizer_v1")
        assert tmpl.usage_count >= 1

    async def test_update(self, template_store) -> None:
        await template_store.update("file_organizer_v1", {"name": "Updated Name"})
        tmpl = await template_store.get("file_organizer_v1")
        assert tmpl.name == "Updated Name"


class TestExperienceStore:
    async def test_save_and_retrieve(self, experience_store) -> None:
        exp = Experience(
            task_id="t1",
            task=ExperienceTask(
                raw_input="Organize files",
                category="file_management.organization",
            ),
            outcome=ExperienceOutcome(success=True, items_processed=10),
            backend_used="local_cpu",
            device_used="cpu",
        )
        await experience_store.save(exp)
        history = await experience_store.find_by_category("file_management.organization")
        assert len(history) == 1
        assert history[0].outcome.success
        assert history[0].backend_used == "local_cpu"

    async def test_statistics(self, experience_store) -> None:
        for i in range(3):
            exp = Experience(
                task_id=f"t{i}",
                task=ExperienceTask(raw_input=f"Task {i}", category="test.stats"),
                outcome=ExperienceOutcome(success=i < 2),
            )
            await experience_store.save(exp)

        stats = await experience_store.get_statistics("test.stats")
        assert stats.total_executions == 3
        assert stats.successful_executions == 2
        assert abs(stats.success_rate - 2/3) < 0.01

    async def test_list_recent(self, experience_store) -> None:
        for i in range(5):
            exp = Experience(
                task_id=f"recent_{i}",
                task=ExperienceTask(raw_input=f"Task {i}", category="recent.test"),
                outcome=ExperienceOutcome(success=True),
            )
            await experience_store.save(exp)

        results = await experience_store.list_recent(limit=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Pytest-asyncio configuration
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.asyncio
