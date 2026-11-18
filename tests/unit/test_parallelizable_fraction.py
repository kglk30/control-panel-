import pytest
from control_panel.orchestrator.depgraph import DependencyGraph, DepNode, DepEdge, OperationCost

def test_parallelizable_fraction_math():
    # DAG: A -> C, B -> C
    graph = DependencyGraph()

    # A and B are independent
    graph.add_node(DepNode(
        operation_id="A",
        estimated_cost=OperationCost(estimated_runtime_ms=100)
    ))
    graph.add_node(DepNode(
        operation_id="B",
        estimated_cost=OperationCost(estimated_runtime_ms=100)
    ))

    # C depends on A and B
    graph.add_node(DepNode(
        operation_id="C",
        estimated_cost=OperationCost(estimated_runtime_ms=100)
    ))
    graph.add_edge("A", "C")
    graph.add_edge("B", "C")

    # Calculate fraction
    p = graph.parallelizable_fraction()

    # Total runtime = 300ms.
    # Independent nodes = A, B (200ms)
    # p = 200 / 300 = 0.666666...
    assert round(p, 4) == 0.6667
