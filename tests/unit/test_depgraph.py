"""Unit tests for DependencyGraph."""

import pytest
from pydantic import ValidationError
from control_panel.orchestrator.depgraph import DependencyGraph, DepNode, DepEdge, OperationCost


def test_empty_graph():
    """Test empty graph operations."""
    graph = DependencyGraph()
    assert graph.total_subtasks == 0
    assert graph.independent_operations() == []
    assert graph.parallelizable_fraction() == 0.0


def test_independent_nodes():
    """Test multiple independent nodes (no edges)."""
    graph = DependencyGraph()
    graph.add_node(DepNode(operation_id="A", estimated_cost=OperationCost(estimated_runtime_ms=100)))
    graph.add_node(DepNode(operation_id="B", estimated_cost=OperationCost(estimated_runtime_ms=200)))

    assert graph.total_subtasks == 2
    assert set(graph.independent_operations()) == {"A", "B"}
    assert graph.get_dependencies("A") == []

    # Cost-weighted P = sum(cost of independent nodes) / sum(cost of all nodes)
    # A=100, B=200, total=300
    # Independent = A, B -> cost = 300
    # P = 300 / 300 = 1.0
    assert pytest.approx(graph.parallelizable_fraction()) == 1.0


def test_simple_dependency_chain():
    """Test A -> B -> C chain."""
    graph = DependencyGraph()
    graph.add_node(DepNode(operation_id="A", estimated_cost=OperationCost(estimated_runtime_ms=100)))
    graph.add_node(DepNode(operation_id="B", estimated_cost=OperationCost(estimated_runtime_ms=200)))
    graph.add_node(DepNode(operation_id="C", estimated_cost=OperationCost(estimated_runtime_ms=50)))

    graph.add_edge("A", "B")
    graph.add_edge("B", "C")

    assert graph.total_subtasks == 3
    assert graph.independent_operations() == ["A"]
    assert graph.get_dependencies("C") == ["B"]
    assert graph.get_dependents("A") == ["B"]

    # Independent = A -> cost = 100
    # Total cost = 350
    # P = 100 / 350
    assert pytest.approx(graph.parallelizable_fraction()) == 100 / 350


def test_multiple_dependencies():
    """Test complex DAG."""
    graph = DependencyGraph()
    graph.add_node(DepNode(operation_id="scan", estimated_cost=OperationCost(estimated_runtime_ms=200)))
    graph.add_node(DepNode(operation_id="classify", estimated_cost=OperationCost(estimated_runtime_ms=1500)))
    graph.add_node(DepNode(operation_id="plan_dirs", estimated_cost=OperationCost(estimated_runtime_ms=50)))
    graph.add_node(DepNode(operation_id="move", estimated_cost=OperationCost(estimated_runtime_ms=800)))

    graph.add_edge("scan", "classify")
    graph.add_edge("scan", "plan_dirs")
    graph.add_edge("classify", "move")
    graph.add_edge("plan_dirs", "move")

    assert graph.total_subtasks == 4
    assert graph.independent_operations() == ["scan"]
    assert set(graph.get_dependencies("move")) == {"classify", "plan_dirs"}

    # Independent = scan -> cost = 200
    # Total cost = 2550
    # P = 200 / 2550
    assert pytest.approx(graph.parallelizable_fraction()) == 200 / 2550


def test_invalid_and_duplicate_dependencies():
    """Test validation handles invalid and duplicate edges/nodes."""
    graph = DependencyGraph()
    graph.add_node(DepNode(operation_id="A"))
    graph.add_node(DepNode(operation_id="B"))

    with pytest.raises(ValueError, match="already exists"):
        graph.add_node(DepNode(operation_id="A"))

    with pytest.raises(ValueError, match="not in graph"):
        graph.add_edge("A", "C")

    with pytest.raises(ValueError, match="not in graph"):
        graph.add_edge("C", "B")

    graph.add_edge("A", "B")
    with pytest.raises(ValueError, match="already exists"):
        graph.add_edge("A", "B")


def test_fallback_parallelizable_fraction():
    """Test calculation of P when runtime costs are zero."""
    graph = DependencyGraph()
    # Cost defaults to 0
    graph.add_node(DepNode(operation_id="A"))
    graph.add_node(DepNode(operation_id="B"))
    graph.add_node(DepNode(operation_id="C"))
    graph.add_node(DepNode(operation_id="D"))

    graph.add_edge("A", "B")
    graph.add_edge("B", "D")
    graph.add_edge("A", "C")
    graph.add_edge("C", "D")

    # Independent = A
    # Total nodes = 4
    # Fallback P = len([A]) / 4 = 1/4 = 0.25
    assert graph.parallelizable_fraction() == 0.25

def test_cycle_detection():
    """Test that calculating P raises an error if graph has a cycle."""
    graph = DependencyGraph()
    graph.add_node(DepNode(operation_id="A"))
    graph.add_node(DepNode(operation_id="B"))
    graph.add_edge("A", "B")
    graph.add_edge("B", "A")

    with pytest.raises(ValueError, match="cycle"):
        graph.parallelizable_fraction()

def test_cost_weighted_vs_node_count():
    """Prove that cost-weighted P differs from node-count P."""
    graph = DependencyGraph()

    # Independent nodes
    graph.add_node(DepNode(operation_id="A", estimated_cost=OperationCost(estimated_runtime_ms=10)))
    graph.add_node(DepNode(operation_id="B", estimated_cost=OperationCost(estimated_runtime_ms=90)))

    # Dependent nodes
    graph.add_node(DepNode(operation_id="C", estimated_cost=OperationCost(estimated_runtime_ms=50)))
    graph.add_node(DepNode(operation_id="D", estimated_cost=OperationCost(estimated_runtime_ms=50)))

    graph.add_edge("A", "C")
    graph.add_edge("A", "D")

    # Node count P would be: 2 independent / 4 total = 0.5
    # Cost weighted P is: (10 + 90) / 200 = 100 / 200 = 0.5
    # Wait, let's change the costs so it differs.
    graph2 = DependencyGraph()
    graph2.add_node(DepNode(operation_id="A", estimated_cost=OperationCost(estimated_runtime_ms=10)))
    graph2.add_node(DepNode(operation_id="B", estimated_cost=OperationCost(estimated_runtime_ms=10)))
    graph2.add_node(DepNode(operation_id="C", estimated_cost=OperationCost(estimated_runtime_ms=80)))
    graph2.add_node(DepNode(operation_id="D", estimated_cost=OperationCost(estimated_runtime_ms=100)))
    graph2.add_edge("A", "C")
    graph2.add_edge("B", "D")

    # Independent nodes = A, B
    # Number of nodes = 4
    # Node count P = 2 / 4 = 0.5

    # Total cost = 10 + 10 + 80 + 100 = 200
    # Independent cost = 10 + 10 = 20
    # Cost-weighted P = 20 / 200 = 0.1

    assert graph2.parallelizable_fraction() == 0.1
    assert graph2.parallelizable_fraction() != (len(graph2.independent_operations()) / graph2.total_subtasks)


def test_pg_empty_graph():
    graph = DependencyGraph()
    assert graph.parallel_groups() == []

def test_pg_single_node():
    graph = DependencyGraph()
    graph.add_node(DepNode(operation_id="A"))
    assert graph.parallel_groups() == [["A"]]

def test_pg_linear_chain():
    graph = DependencyGraph()
    graph.add_node(DepNode(operation_id="A"))
    graph.add_node(DepNode(operation_id="B"))
    graph.add_node(DepNode(operation_id="C"))
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    assert graph.parallel_groups() == [["A"], ["B"], ["C"]]

def test_pg_multiple_independent():
    graph = DependencyGraph()
    graph.add_node(DepNode(operation_id="C"))
    graph.add_node(DepNode(operation_id="A"))
    graph.add_node(DepNode(operation_id="B"))
    assert graph.parallel_groups() == [["A", "B", "C"]]

def test_pg_diamond_dependency():
    graph = DependencyGraph()
    for op in ["A", "B", "C", "D"]:
        graph.add_node(DepNode(operation_id=op))
    graph.add_edge("A", "B")
    graph.add_edge("A", "C")
    graph.add_edge("B", "D")
    graph.add_edge("C", "D")
    assert graph.parallel_groups() == [["A"], ["B", "C"], ["D"]]

def test_pg_multiple_dependency_levels():
    graph = DependencyGraph()
    for op in ["A", "B", "C"]:
        graph.add_node(DepNode(operation_id=op))
    graph.add_edge("A", "C")
    graph.add_edge("B", "C")
    assert graph.parallel_groups() == [["A", "B"], ["C"]]

def test_pg_cycle_detection():
    graph = DependencyGraph()
    graph.add_node(DepNode(operation_id="A"))
    graph.add_node(DepNode(operation_id="B"))
    graph.add_edge("A", "B")
    graph.add_edge("B", "A")
    import pytest
    with pytest.raises(ValueError, match="cycle"):
        graph.parallel_groups()

def test_pg_every_node_appears_exactly_once():
    graph = DependencyGraph()
    for op in ["A", "B", "C", "D", "E"]:
        graph.add_node(DepNode(operation_id=op))
    graph.add_edge("A", "B")
    graph.add_edge("A", "C")
    graph.add_edge("C", "D")
    graph.add_edge("B", "E")

    groups = graph.parallel_groups()
    flattened = [node for group in groups for node in group]
    assert sorted(flattened) == ["A", "B", "C", "D", "E"]
    assert len(flattened) == len(set(flattened))
