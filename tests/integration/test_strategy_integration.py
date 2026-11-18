"""Integration tests for Strategy -> Orchestrator -> Planner loop."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from control_panel.orchestrator.core import Orchestrator, OrchestratorState
from control_panel.resources.governor import ResourceGovernor
from control_panel.templates.store import TemplateStore
from control_panel.experience.store import ExperienceStore


@pytest.mark.asyncio
async def test_strategizing_phase_before_planning():
    """Test that STRATEGIZING occurs before PLANNING and passes the ExecutionStrategy to Planner."""
    orchestrator = Orchestrator()

    orchestrator._experience_store.find_by_category = AsyncMock(return_value=[])
    orchestrator._experience_store.save = AsyncMock()
    orchestrator._template_store.find_by_category = AsyncMock(return_value=[])

    # We will hook into _transition to record the states visited
    states_visited = []
    original_transition = orchestrator._transition

    def mock_transition(state, task_id=None):
        states_visited.append(state)
        original_transition(state, task_id)

    orchestrator._transition = mock_transition

    # Spy on Planner.plan
    original_plan = orchestrator._planner.plan
    planner_strategy = None
    def mock_plan(*args, **kwargs):
        nonlocal planner_strategy
        planner_strategy = kwargs.get("strategy")
        return original_plan(*args, **kwargs)

    orchestrator._planner.plan = mock_plan

    # Mocking analyzer and execution
    from control_panel.common.models import TaskSpec
    orchestrator._analyzer.analyze = MagicMock(return_value=TaskSpec(
        task_id="fake_task",
        raw_input="do something generic",
        objective="test",
        category="general.unknown",
        structured_requirements={}
    ))
    orchestrator._worker_runtime.execute_sequential = MagicMock(return_value=[])

    # Run a generic task
    await orchestrator.run_task("do something generic")

    # 1. STRATEGIZING occurs before PLANNING
    assert OrchestratorState.RETRIEVING in states_visited
    assert OrchestratorState.STRATEGIZING in states_visited
    assert OrchestratorState.PLANNING in states_visited

    idx_strat = states_visited.index(OrchestratorState.STRATEGIZING)
    idx_plan = states_visited.index(OrchestratorState.PLANNING)
    assert idx_strat < idx_plan

    # 2. ExecutionStrategy is produced and passed to Planner
    assert planner_strategy is not None
    assert planner_strategy.strategy_type == "sequential"
    assert planner_strategy.recommended_worker_count == 1


@pytest.mark.asyncio
async def test_file_organization_still_succeeds():
    """Test that existing file organization execution works and execution is sequential."""
    orchestrator = Orchestrator()

    orchestrator._experience_store.find_by_category = AsyncMock(return_value=[])
    orchestrator._experience_store.save = AsyncMock()

    # Need to return a valid template for file management
    from control_panel.templates.models import Template
    fake_template = Template(
        id="fake-id",
        name="Fake File Organizer",
        category="file_management.organize",
        workflow={
            "workers": [
                {"role": "file_classifier"},
                {"role": "file_organizer"}
            ]
        }
    )
    orchestrator._template_store.find_by_category = AsyncMock(return_value=[fake_template])

    # Spy on worker runtime
    orchestrator._worker_runtime.execute_sequential = MagicMock(return_value=[])

    # Fake an analyzer output to simulate file organization task
    from control_panel.common.models import TaskSpec
    orchestrator._analyzer.analyze = MagicMock(return_value=TaskSpec(
        task_id="fake_task",
        raw_input="organize files in selected folder",
        objective="organize",
        category="file_management",
        structured_requirements={}
    ))

    # Run task
    result = await orchestrator.run_task("organize files in selected folder", target_path="/fake/path")

    # The file organization MVP continues to work (it spawns 2 workers)
    # 1 file_classifier, 1 file_organizer
    orchestrator._worker_runtime.execute_sequential.assert_called_once()
    workers_passed = orchestrator._worker_runtime.execute_sequential.call_args[0][0]

    assert len(workers_passed) == 2
    assert workers_passed[0].spec.role == "file_classifier"
    assert workers_passed[1].spec.role == "file_organizer"

    # Execution remains sequential via execute_sequential
    # (Verified by the assert_called_once() above)

from unittest.mock import patch

@pytest.mark.asyncio
@patch("control_panel.orchestrator.core.ExecutionStrategyPlanner")
async def test_orchestrator_passes_dynamic_n_max(mock_planner_cls):
    """Test 5: Orchestrator passes the ResourceGovernor-derived value into ExecutionStrategyPlanner."""
    orchestrator = Orchestrator()

    mock_constraints = MagicMock()
    mock_constraints.max_concurrent_workers = 42
    orchestrator._governor.get_resource_constraints = MagicMock(return_value=mock_constraints)

    orchestrator._experience_store.find_by_category = AsyncMock(return_value=[])
    orchestrator._experience_store.save = AsyncMock()
    orchestrator._template_store.find_by_category = AsyncMock(return_value=[])
    orchestrator._worker_runtime.execute_sequential = MagicMock(return_value=[])

    from control_panel.common.models import TaskSpec
    orchestrator._analyzer.analyze = MagicMock(return_value=TaskSpec(
        task_id="fake_task",
        raw_input="do something",
        objective="test",
        category="general.unknown",
        structured_requirements={}
    ))

    mock_planner_instance = mock_planner_cls.return_value
    mock_planner_instance.plan_strategy.return_value = MagicMock(strategy_type="sequential")

    await orchestrator.run_task("do something")

    mock_planner_cls.assert_called_once_with(n_max=42)
