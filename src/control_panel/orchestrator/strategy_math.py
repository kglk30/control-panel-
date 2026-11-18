"""Mathematical layer for ExecutionStrategy selection."""

from typing import Any

from control_panel.orchestrator.depgraph import DependencyGraph
from control_panel.orchestrator.amdahl import theoretical_speedup
from control_panel.orchestrator.usl import theoretical_speedup_usl

__all__ = ["select_optimal_strategy"]


def _max_concurrent_tasks(graph: DependencyGraph) -> int:
    """Determine the maximum width of the dependency graph (available independent work)."""
    if not graph.nodes:
        return 0

    adj: dict[str, list[str]] = {node_id: [] for node_id in graph.nodes}
    in_degree: dict[str, int] = {node_id: 0 for node_id in graph.nodes}
    for edge in graph.edges:
        adj[edge.from_op].append(edge.to_op)
        in_degree[edge.to_op] += 1

    # Standard BFS topological level traversal
    queue = [n for n, d in in_degree.items() if d == 0]
    max_width = 0

    while queue:
        max_width = max(max_width, len(queue))
        next_queue = []
        for u in queue:
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    next_queue.append(v)
        queue = next_queue

    return max_width


def select_optimal_strategy(
    graph: DependencyGraph,
    alpha: float,
    beta: float,
    n_max: int
) -> dict[str, Any]:
    """
    Select optimal execution strategy based on USL and Amdahl estimates.

    Args:
        graph: The DependencyGraph of the task.
        alpha: Contention coefficient.
        beta: Coherency coefficient.
        n_max: Maximum allowable workers.

    Returns:
        A dictionary containing fields for the ExecutionStrategy.
    """
    if n_max < 1:
        n_max = 1

    p = graph.parallelizable_fraction()
    available_work = _max_concurrent_tasks(graph)

    # 5. Never recommend more workers than available independent work
    # 6. N=1 must always be a valid baseline
    max_workers = max(1, min(n_max, available_work))

    best_n = 1
    best_speedup = 1.0
    best_reason = "Sequential execution is optimal."

    for n in range(1, max_workers + 1):
        # 1. Calculate Amdahl speedup (for reference/logging, though USL drives the decision)
        amdahl_s = theoretical_speedup(p, n)

        # 2. Calculate USL speedup
        usl_s = theoretical_speedup_usl(n, alpha, beta)

        # 3. Use USL as the contention/coherency-aware estimate
        # 4. Determine the worker count with the highest predicted speedup
        if usl_s > best_speedup:
            best_speedup = usl_s
            best_n = n
            best_reason = f"USL predicts {usl_s:.2f}x speedup with {n} workers (Amdahl was {amdahl_s:.2f}x)."

    strategy_type = "sequential" if best_n == 1 else "parallel"

    return {
        "strategy_type": strategy_type,
        "recommended_worker_count": best_n,
        "parallelizable_fraction": p,
        "theoretical_speedup": best_speedup,
        "rationale": best_reason
    }
