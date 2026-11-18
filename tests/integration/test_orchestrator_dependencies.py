import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from control_panel.orchestrator.core import Orchestrator
from control_panel.common.models import TaskSpec, ExecutionPlan
from control_panel.templates.models import Template
from control_panel.resources.models import ResourceConstraints
from control_panel.orchestrator.depgraph import DependencyGraph

@pytest.fixture
def base_orchestrator():
    orchestrator = Orchestrator()
    orchestrator._experience_store.find_by_category = AsyncMock(return_value=[])
    orchestrator._experience_store.save = AsyncMock()

    orchestrator._analyzer.analyze = MagicMock(return_value=TaskSpec(
        task_id="fake_task",
        raw_input="do something",
        objective="test",
        category="general.unknown",
        structured_requirements={}
    ))

    # Mock governor constraints
    orchestrator._governor.get_resource_constraints = MagicMock(return_value=ResourceConstraints(
        available_ram_mb=8000,
        max_model_ram_mb=4000,
        cpu_pressure="low",
        power_policy="balanced",
        max_concurrent_workers=4
    ))

    # Mock Planner
    orchestrator._planner.plan = MagicMock(return_value=ExecutionPlan(
        task_id="fake_task",
        execution_mode="parallel",
        workers=[]
    ))

    orchestrator._worker_runtime.execute_parallel = MagicMock(return_value=[])
    orchestrator._worker_runtime.execute_sequential = MagicMock(return_value=[])

    return orchestrator

@pytest.mark.asyncio
async def test_linear_dependency_fallback(base_orchestrator):
    """Test 1: Existing template with no depends_on remains linear."""
    template = Template(
        id="t1",
        version=1,
        name="Test",
        category="general.test",
        description="test",
        workflow={
            "workers": [
                {"role": "A"},
                {"role": "B"},
                {"role": "C"}
            ]
        },
        requirements={}
    )
    base_orchestrator._template_store.find_by_category = AsyncMock(return_value=[template])

    await base_orchestrator.run_task("test")

    call_args, call_kwargs = base_orchestrator._worker_runtime.execute_parallel.call_args
    graph: DependencyGraph = call_kwargs["graph"]
    groups = graph.parallel_groups()

    assert groups == [["A"], ["B"], ["C"]]

@pytest.mark.asyncio
async def test_root_dependency(base_orchestrator):
    """Test 2 & 3: Workers with depends_on=[] become roots and appear in same group."""
    template = Template(
        id="t2",
        version=1,
        name="Test",
        category="general.test",
        description="test",
        workflow={
            "workers": [
                {"role": "A", "depends_on": []},
                {"role": "B", "depends_on": []}
            ]
        },
        requirements={}
    )
    base_orchestrator._template_store.find_by_category = AsyncMock(return_value=[template])

    await base_orchestrator.run_task("test")

    call_args, call_kwargs = base_orchestrator._worker_runtime.execute_parallel.call_args
    graph: DependencyGraph = call_kwargs["graph"]
    groups = graph.parallel_groups()

    assert len(groups) == 1
    assert set(groups[0]) == {"A", "B"}

@pytest.mark.asyncio
async def test_multiple_dependencies(base_orchestrator):
    """Test 4: Worker depending on two roots appears in following group."""
    template = Template(
        id="t3",
        version=1,
        name="Test",
        category="general.test",
        description="test",
        workflow={
            "workers": [
                {"role": "A", "depends_on": []},
                {"role": "B", "depends_on": []},
                {"role": "C", "depends_on": ["A", "B"]}
            ]
        },
        requirements={}
    )
    base_orchestrator._template_store.find_by_category = AsyncMock(return_value=[template])

    await base_orchestrator.run_task("test")

    call_args, call_kwargs = base_orchestrator._worker_runtime.execute_parallel.call_args
    graph: DependencyGraph = call_kwargs["graph"]
    groups = graph.parallel_groups()

    assert len(groups) == 2
    assert set(groups[0]) == {"A", "B"}
    assert groups[1] == ["C"]

@pytest.mark.asyncio
async def test_invalid_dependency_rejected(base_orchestrator):
    """Test 5: Invalid dependency name is rejected clearly."""
    template = Template(
        id="t4",
        version=1,
        name="Test",
        category="general.test",
        description="test",
        workflow={
            "workers": [
                {"role": "A", "depends_on": []},
                {"role": "B", "depends_on": ["phantom_node"]}
            ]
        },
        requirements={}
    )
    base_orchestrator._template_store.find_by_category = AsyncMock(return_value=[template])

    result = await base_orchestrator.run_task("test")

    assert result.success is False
    assert "Invalid dependency: 'phantom_node'" in result.error
