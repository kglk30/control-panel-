"""Unit tests for ExecutionStrategyPlanner."""

import pytest
from control_panel.common.models import ExecutionStrategy
from control_panel.experience.models import Experience, ExperienceOutcome, ExperienceExecution
from control_panel.orchestrator.depgraph import DependencyGraph, DepNode, OperationCost
from control_panel.orchestrator.strategy_planner import ExecutionStrategyPlanner


def _build_graph(nodes: list[str], edges: list[tuple[str, str]], cost: int = 100) -> DependencyGraph:
    graph = DependencyGraph()
    for n in nodes:
        graph.add_node(DepNode(operation_id=n, estimated_cost=OperationCost(estimated_runtime_ms=cost)))
    for u, v in edges:
        graph.add_edge(u, v)
    return graph

def _make_exp(worker_count: int, runtime_ms: int, p: float) -> Experience:
    exp = Experience()
    exp.worker_count = worker_count
    exp.execution = ExperienceExecution(total_runtime_ms=runtime_ms)
    exp.parallelizable_fraction = p
    exp.outcome = ExperienceOutcome(success=True)
    return exp


def test_no_experience_history():
    """Test planner with no history uses default parameters."""
    planner = ExecutionStrategyPlanner(n_max=4)
    graph = _build_graph(["A", "B", "C", "D"], [])

    strategy = planner.plan_strategy(graph, [])

    assert isinstance(strategy, ExecutionStrategy)
    assert strategy.strategy_type == "parallel"
    assert strategy.parallelizable_fraction == 1.0
    assert strategy.recommended_worker_count > 1


def test_calibrated_history():
    """Test planner with perfectly scaling history."""
    planner = ExecutionStrategyPlanner(n_max=4)
    graph = _build_graph(["A", "B", "C", "D"], [])

    history = [
        _make_exp(1, 100, 1.0),
        _make_exp(2, 50, 1.0),
        _make_exp(4, 25, 1.0)
    ]

    strategy = planner.plan_strategy(graph, history)
    assert strategy.recommended_worker_count == 4
    assert strategy.theoretical_speedup == 4.0


def test_sequential_dependency_graph():
    """Test strictly sequential graph forces sequential strategy despite history."""
    planner = ExecutionStrategyPlanner(n_max=4)
    graph = _build_graph(["A", "B", "C", "D"], [("A", "B"), ("B", "C"), ("C", "D")])

    history = [_make_exp(1, 100, 1.0), _make_exp(4, 25, 1.0)]

    strategy = planner.plan_strategy(graph, history)
    assert strategy.strategy_type == "sequential"
    assert strategy.recommended_worker_count == 1
    assert strategy.theoretical_speedup == 1.0


def test_fully_parallel_graph():
    """Test fully parallel graph correctly limits to N_max."""
    planner = ExecutionStrategyPlanner(n_max=2)  # Limited to 2
    graph = _build_graph(["A", "B", "C", "D"], [])

    history = [_make_exp(1, 100, 1.0), _make_exp(4, 25, 1.0)]

    strategy = planner.plan_strategy(graph, history)
    assert strategy.recommended_worker_count == 2
    assert strategy.theoretical_speedup == 2.0


def test_contention_heavy_history():
    """Test history with high contention reduces recommended workers."""
    planner = ExecutionStrategyPlanner(n_max=8)
    graph = _build_graph(["A", "B", "C", "D", "E", "F", "G", "H"], [])

    history = [_make_exp(1, 100, 0.2)]

    strategy = planner.plan_strategy(graph, history)
    assert isinstance(strategy, ExecutionStrategy)
    # Even with 8 independent tasks, contention is so high that optimal workers < 8
    assert strategy.recommended_worker_count < 8


def test_coherency_heavy_history():
    """Test history with massive coherency penalty forces fewer workers."""
    planner = ExecutionStrategyPlanner(n_max=8)
    graph = _build_graph(["A", "B", "C", "D", "E", "F", "G", "H"], [])

    history = [
        _make_exp(1, 100, 1.0),
        _make_exp(2, 80, 1.0),
        _make_exp(4, 120, 1.0)
    ]

    strategy = planner.plan_strategy(graph, history)
    # The optimal speedup should peak at N=2 due to coherency
    assert strategy.recommended_worker_count == 2
