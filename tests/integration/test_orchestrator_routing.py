import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from control_panel.orchestrator.core import Orchestrator
from control_panel.common.models import TaskSpec, ExecutionPlan, ExecutionStrategy
from control_panel.resources.models import ResourceConstraints
from control_panel.orchestrator.depgraph import DependencyGraph

@pytest.fixture
def base_orchestrator():
    orchestrator = Orchestrator()
    orchestrator._experience_store.find_by_category = AsyncMock(return_value=[])
    orchestrator._experience_store.save = AsyncMock()
    orchestrator._template_store.find_by_category = AsyncMock(return_value=[])

    orchestrator._analyzer.analyze = MagicMock(return_value=TaskSpec(
        task_id="fake_task",
        raw_input="do something",
        objective="test",
        category="general.unknown",
        structured_requirements={}
    ))

    orchestrator._worker_runtime.execute_parallel = MagicMock(return_value=[])
    orchestrator._worker_runtime.execute_sequential = MagicMock(return_value=[])

    return orchestrator


@pytest.mark.asyncio
@patch("control_panel.orchestrator.core.ExecutionStrategyPlanner")
async def test_orchestrator_enforces_recommended_worker_count_below_limit(mock_strategy_planner_cls, base_orchestrator):
    """Verify when recommended_worker_count < max_concurrent_workers, it uses recommended."""

    # Setup Strategy Planner mock
    mock_strategy_planner = mock_strategy_planner_cls.return_value
    mock_strategy = ExecutionStrategy(
        strategy_type="parallel",
        recommended_worker_count=2, # RECOMMENDED = 2
        parallelizable_fraction=0.8,
        theoretical_speedup=1.5,
        rationale="test"
    )
    mock_strategy_planner.plan_strategy.return_value = mock_strategy

    # Setup ResourceGovernor mock
    base_orchestrator._governor.get_resource_constraints = MagicMock(return_value=ResourceConstraints(
        available_ram_mb=8000,
        max_model_ram_mb=4000,
        cpu_pressure="low",
        power_policy="balanced",
        max_concurrent_workers=4 # HARDWARE LIMIT = 4
    ))

    # Setup Planner mock
    base_orchestrator._planner.plan = MagicMock(return_value=ExecutionPlan(
        task_id="fake_task",
        execution_mode="parallel",
        workers=[]
    ))

    await base_orchestrator.run_task("do something")

    base_orchestrator._worker_runtime.execute_parallel.assert_called_once()
    base_orchestrator._worker_runtime.execute_sequential.assert_not_called()

    _, call_kwargs = base_orchestrator._worker_runtime.execute_parallel.call_args
    assert call_kwargs["n_max"] == 2 # 2 < 4, so it should use 2


@pytest.mark.asyncio
@patch("control_panel.orchestrator.core.ExecutionStrategyPlanner")
async def test_orchestrator_enforces_hardware_limit_when_recommended_exceeds(mock_strategy_planner_cls, base_orchestrator):
    """Verify when recommended_worker_count > max_concurrent_workers, it clamps to limit."""

    # Setup Strategy Planner mock
    mock_strategy_planner = mock_strategy_planner_cls.return_value
    mock_strategy = ExecutionStrategy(
        strategy_type="parallel",
        recommended_worker_count=10, # RECOMMENDED = 10
        parallelizable_fraction=0.8,
        theoretical_speedup=1.5,
        rationale="test"
    )
    mock_strategy_planner.plan_strategy.return_value = mock_strategy

    # Setup ResourceGovernor mock
    base_orchestrator._governor.get_resource_constraints = MagicMock(return_value=ResourceConstraints(
        available_ram_mb=8000,
        max_model_ram_mb=4000,
        cpu_pressure="low",
        power_policy="balanced",
        max_concurrent_workers=4 # HARDWARE LIMIT = 4
    ))

    # Setup Planner mock
    base_orchestrator._planner.plan = MagicMock(return_value=ExecutionPlan(
        task_id="fake_task",
        execution_mode="parallel",
        workers=[]
    ))

    await base_orchestrator.run_task("do something")

    base_orchestrator._worker_runtime.execute_parallel.assert_called_once()

    _, call_kwargs = base_orchestrator._worker_runtime.execute_parallel.call_args
    assert call_kwargs["n_max"] == 4 # 10 > 4, so it should clamp to 4


@pytest.mark.asyncio
async def test_orchestrator_routes_to_sequential(base_orchestrator):
    """Verify Orchestrator reads execution_mode='sequential' and calls execute_sequential."""

    base_orchestrator._planner.plan = MagicMock(return_value=ExecutionPlan(
        task_id="fake_task",
        execution_mode="sequential",
        workers=[]
    ))

    await base_orchestrator.run_task("do something")

    base_orchestrator._worker_runtime.execute_sequential.assert_called_once()
    base_orchestrator._worker_runtime.execute_parallel.assert_not_called()
