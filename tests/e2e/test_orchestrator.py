"""E2E test — full orchestrator lifecycle with a real temp directory."""

import asyncio
import pytest
from pathlib import Path

from control_panel.persistence.database import Database, close_database, get_database
from control_panel.orchestrator.core import Orchestrator


@pytest.fixture
async def fresh_db(tmp_path):
    db_path = tmp_path / "e2e.db"
    db = Database(db_path)
    await db.initialize()
    # Override the module singleton for this test
    import control_panel.persistence.database as db_module
    db_module._db = db
    yield db
    await db.close()
    db_module._db = None


@pytest.mark.asyncio
async def test_full_file_organization_lifecycle(tmp_path, fresh_db) -> None:
    """Test the complete IDLE → ... → IDLE lifecycle."""
    # Create test files
    (tmp_path / "photo.jpg").write_bytes(b"jpg")
    (tmp_path / "report.pdf").write_bytes(b"pdf")
    (tmp_path / "song.mp3").write_bytes(b"mp3")
    (tmp_path / "notes.txt").write_text("text")
    (tmp_path / "code.py").write_text("print('x')")

    orch = Orchestrator(dry_run=True)
    await orch.initialize()

    from control_panel.orchestrator.core import OrchestratorState
    assert orch.state == OrchestratorState.IDLE

    result = await orch.run_task(
        raw_input="Organize my files by type",
        target_path=str(tmp_path),
    )

    # Must return to IDLE after task
    assert orch.state == OrchestratorState.IDLE

    # Task should succeed
    assert result.success, f"Task failed: {result.error}"

    # Must have used 3 workers (scanner + classifier + organizer)
    assert len(result.worker_results) == 3

    # All workers should have succeeded
    assert all(w.success for w in result.worker_results), \
        [f"{w.role}: {w.error}" for w in result.worker_results if not w.success]

    # Scanner should have found the 5 files
    scanner = next(w for w in result.worker_results if w.role == "file_scanner")
    assert scanner.success

    # Result must have task_spec and plan
    assert result.task_spec is not None
    assert result.task_spec.category == "file_management.organization"
    assert result.plan is not None
    assert result.plan.execution_mode == "sequential"

    # Backend must be real (not None)
    assert result.backend_used in ("local_cpu", "rule_based", "api", "snapdragon")


@pytest.mark.asyncio
async def test_orchestrator_only_one_task_at_a_time(tmp_path, fresh_db) -> None:
    """Test that a second concurrent task is rejected."""
    orch = Orchestrator(dry_run=True)
    await orch.initialize()

    # Manually set state to ANALYZING to simulate a running task
    from control_panel.orchestrator.core import OrchestratorState
    orch._state = OrchestratorState.ANALYZING

    with pytest.raises(RuntimeError, match="busy"):
        await orch.run_task("Another task", target_path=str(tmp_path))
