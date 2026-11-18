"""Unit tests for ExecutionStrategy mathematical selection layer."""

import pytest
from control_panel.orchestrator.depgraph import DependencyGraph, DepNode, OperationCost
from control_panel.orchestrator.strategy_math import select_optimal_strategy


def _build_graph(nodes: list[str], edges: list[tuple[str, str]], cost: int = 100) -> DependencyGraph:
    graph = DependencyGraph()
    for n in nodes:
        graph.add_node(DepNode(operation_id=n, estimated_cost=OperationCost(estimated_runtime_ms=cost)))
    for u, v in edges:
        graph.add_edge(u, v)
    return graph


def test_p_low_strictly_sequential():
    """Test strictly sequential chain where P is > 0 but optimal is still N=1."""
    graph = _build_graph(["A", "B", "C"], [("A", "B"), ("B", "C")])

    result = select_optimal_strategy(graph, alpha=0.1, beta=0.01, n_max=10)
    assert result["strategy_type"] == "sequential"
    assert result["recommended_worker_count"] == 1
    # Cost-weighted P = 100 / 300 = 1/3
    assert pytest.approx(result["parallelizable_fraction"]) == 1/3
    assert result["theoretical_speedup"] == 1.0


def test_p_one_fully_parallel():
    """Test P=1 (all independent nodes)."""
    graph = _build_graph(["A", "B", "C", "D"], [])

    result = select_optimal_strategy(graph, alpha=0.0, beta=0.0, n_max=10)
    assert result["strategy_type"] == "parallel"
    assert result["recommended_worker_count"] == 4
    assert result["parallelizable_fraction"] == 1.0
    assert result["theoretical_speedup"] == 4.0


def test_one_independent_operation():
    """Test graph with only 1 independent operation (no parallel potential)."""
    graph = _build_graph(["A"], [])
    result = select_optimal_strategy(graph, alpha=0.0, beta=0.0, n_max=10)
    assert result["recommended_worker_count"] == 1


def test_worker_count_limited_by_available_work():
    """Test N_max > width caps properly at max width."""
    # Width is 2 (B and C run parallel)
    graph = _build_graph(["A", "B", "C", "D"], [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    result = select_optimal_strategy(graph, alpha=0.0, beta=0.0, n_max=10)
    assert result["recommended_worker_count"] == 2
    assert result["strategy_type"] == "parallel"


def test_usl_diminishing_returns_optimal_in_middle():
    """Test USL where increasing workers eventually becomes worse (optimal N in middle)."""
    graph = _build_graph(["A", "B", "C", "D", "E", "F", "G", "H"], [])
    # 8 independent tasks. P=1.0.
    # High coherency penalty (beta = 0.2). S(N) = N / (1 + 0.2*N*(N-1))
    # S(1) = 1
    # S(2) = 2 / (1 + 0.2*2) = 2 / 1.4 = 1.42
    # S(3) = 3 / (1 + 0.2*6) = 3 / 2.2 = 1.36
    # Optimal is N=2! Not N=8.
    result = select_optimal_strategy(graph, alpha=0.0, beta=0.2, n_max=8)
    assert result["recommended_worker_count"] == 2
    assert round(result["theoretical_speedup"], 2) == 1.43


def test_contention_heavy_workload():
    """Test contention-heavy workload (alpha dominates)."""
    graph = _build_graph(["A", "B", "C", "D"], [])
    # High alpha = 0.8
    # S(2) = 2 / (1 + 0.8) = 1.11
    # S(4) = 4 / (1 + 0.8*3) = 4 / 3.4 = 1.17
    result = select_optimal_strategy(graph, alpha=0.8, beta=0.0, n_max=4)
    assert result["recommended_worker_count"] == 4
    assert round(result["theoretical_speedup"], 2) == 1.18


def test_coherency_heavy_workload():
    """Test coherency-heavy workload (beta dominates, dropping speedup fast)."""
    graph = _build_graph(["A", "B", "C", "D"], [])
    # Huge beta = 1.0
    # S(2) = 2 / (1 + 1.0*2) = 2 / 3 = 0.66
    # Since S(2) < 1.0, it should fallback to N=1.
    result = select_optimal_strategy(graph, alpha=0.0, beta=1.0, n_max=4)
    assert result["recommended_worker_count"] == 1
    assert result["theoretical_speedup"] == 1.0
    assert result["strategy_type"] == "sequential"
