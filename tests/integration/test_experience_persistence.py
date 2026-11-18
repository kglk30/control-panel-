import pytest
import time
from control_panel.experience.collector import ExperienceCollector
from control_panel.common.models import WorkerOutcome, ExecutionTrace, TraceSummary
from control_panel.experience.store import ExperienceStore
from control_panel.persistence.database import Database

@pytest.fixture
async def experience_store(tmp_path):
    db_path = tmp_path / "test.db"
    database = Database(db_path)
    await database.initialize()
    store = ExperienceStore(database)
    yield store
    await database.close()

@pytest.mark.asyncio
async def test_parallel_execution_persistence(experience_store: ExperienceStore) -> None:
    # 1. Setup collector like Orchestrator does
    collector = ExperienceCollector(
        task_id="task_persist_1",
        raw_input="parallel test",
        category="parallel.test",
        task_spec={},
        plan={}
    )

    # 2. Add traces and outcomes
    trace = ExecutionTrace(
        trace_id="tr1",
        worker_id="w1",
        task_id="task_persist_1",
        worker_role="A",
        summary=TraceSummary(total_steps=1, successful_steps=1, failed_steps=0, retried_steps=0, total_runtime_ms=100)
    )
    collector.add_trace(trace)
    collector.add_worker_outcome(WorkerOutcome(worker_id="w1", role="A", success=True, runtime_ms=100))

    # 3. Simulate Orchestrator timing and assignment
    collector.execution_mode = "parallel"
    collector.concurrency_limit = 4
    collector.wall_clock_ms = 123
    collector.parallelizable_fraction = 0.75

    # 4. Build Experience
    exp = collector.build(success=True)

    # 5. Persist to DB
    await experience_store.save(exp)

    # 6. Retrieve from DB
    history = await experience_store.find_by_category("parallel.test")
    assert len(history) == 1

    persisted = history[0]

    # Verify exactly as requested:
    # 1. execution_mode = "parallel"
    assert persisted.strategy_type == "parallel"
    assert persisted.plan.execution_mode == "parallel"

    # 2. worker_count = the actual effective concurrency limit N (which is 4, not 1 worker)
    assert persisted.worker_count == 4
    assert persisted.plan.worker_count == 4

    # Check that parallelizable_fraction survives
    assert persisted.parallelizable_fraction == 0.75

    # 3. total_runtime_ms = measured wall-clock execution time
    assert persisted.execution.total_runtime_ms == 123

    # 4. outcome.success = true
    assert persisted.outcome.success is True

    # 5. the execution trace still exists
    assert len(persisted.execution.traces) == 1
    assert persisted.execution.traces[0].trace_id == "tr1"


@pytest.mark.asyncio
async def test_sequential_execution_persistence(experience_store: ExperienceStore) -> None:
    collector = ExperienceCollector(
        task_id="task_persist_2",
        raw_input="seq test",
        category="seq.test",
        task_spec={},
        plan={}
    )

    collector.add_worker_outcome(WorkerOutcome(worker_id="w2", role="B", success=True, runtime_ms=50))

    # Simulate sequential execution metrics
    collector.execution_mode = "sequential"
    collector.concurrency_limit = 1
    collector.wall_clock_ms = 50

    exp = collector.build(success=True)
    await experience_store.save(exp)

    history = await experience_store.find_by_category("seq.test")
    assert len(history) == 1

    persisted = history[0]

    # Verify sequential parameters
    assert persisted.strategy_type == "sequential"
    assert persisted.plan.execution_mode == "sequential"
    assert persisted.worker_count == 1
    assert persisted.plan.worker_count == 1
    assert persisted.execution.total_runtime_ms == 50
    assert persisted.outcome.success is True
