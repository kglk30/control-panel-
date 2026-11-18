"""Dependency Graph for representing task subtasks and execution order."""

from __future__ import annotations

import collections
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["OperationCost", "DepNode", "DepEdge", "DependencyGraph"]


class OperationCost(BaseModel):
    """Estimated cost of an operation."""

    model_config = ConfigDict(frozen=False)

    estimated_runtime_ms: int = 0
    requires_inference: bool = False
    estimated_memory_mb: int = 0


class DepNode(BaseModel):
    """A node representing a subtask operation."""

    model_config = ConfigDict(frozen=False)

    operation_id: str
    classification: str = "independent"  # "independent"|"dependent"|"conditional"
    execution_type: str = "deterministic"  # "deterministic"|"ai_required"|"hybrid"
    requires_worker: bool = False
    estimated_cost: OperationCost = Field(default_factory=OperationCost)


class DepEdge(BaseModel):
    """A directed dependency between two operations."""

    model_config = ConfigDict(frozen=False)

    from_op: str
    to_op: str
    edge_type: str = "data"
    description: str = ""


class DependencyGraph(BaseModel):
    """Lightweight directed acyclic graph for subtasks."""

    model_config = ConfigDict(frozen=False)

    nodes: dict[str, DepNode] = Field(default_factory=dict)
    edges: list[DepEdge] = Field(default_factory=list)

    def add_node(self, node: DepNode) -> None:
        """Add a node to the graph."""
        if node.operation_id in self.nodes:
            raise ValueError(f"Node '{node.operation_id}' already exists")
        self.nodes[node.operation_id] = node

    def add_edge(self, from_op: str, to_op: str, edge_type: str = "data", description: str = "") -> None:
        """Add a directed dependency edge."""
        if from_op not in self.nodes:
            raise ValueError(f"from_op '{from_op}' not in graph")
        if to_op not in self.nodes:
            raise ValueError(f"to_op '{to_op}' not in graph")

        # Check for duplicates
        for edge in self.edges:
            if edge.from_op == from_op and edge.to_op == to_op:
                raise ValueError(f"Edge from '{from_op}' to '{to_op}' already exists")

        self.edges.append(
            DepEdge(from_op=from_op, to_op=to_op, edge_type=edge_type, description=description)
        )

    @property
    def total_subtasks(self) -> int:
        """Total number of nodes/subtasks in the graph."""
        return len(self.nodes)

    def get_dependencies(self, node_id: str) -> list[str]:
        """Return the list of node IDs that the given node depends on (incoming edges)."""
        if node_id not in self.nodes:
            raise ValueError(f"Node '{node_id}' not in graph")
        return [edge.from_op for edge in self.edges if edge.to_op == node_id]

    def get_dependents(self, node_id: str) -> list[str]:
        """Return the list of node IDs that depend on the given node (outgoing edges)."""
        if node_id not in self.nodes:
            raise ValueError(f"Node '{node_id}' not in graph")
        return [edge.to_op for edge in self.edges if edge.from_op == node_id]

    def independent_operations(self) -> list[str]:
        """Return operations that have no dependencies (entry points) and can potentially execute independently."""
        has_incoming = {edge.to_op for edge in self.edges}
        return [node_id for node_id in self.nodes if node_id not in has_incoming]

    def _check_for_cycles(self) -> None:
        """Check if the graph contains a cycle."""
        in_degree: dict[str, int] = {node_id: 0 for node_id in self.nodes}
        adj: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}

        for edge in self.edges:
            adj[edge.from_op].append(edge.to_op)
            in_degree[edge.to_op] += 1

        import collections
        queue = collections.deque([n for n, d in in_degree.items() if d == 0])
        sorted_nodes = []
        while queue:
            u = queue.popleft()
            sorted_nodes.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("DependencyGraph contains a cycle")

    def parallel_groups(self) -> list[list[str]]:
        """
        Group nodes into parallel execution layers based on dependencies.
        Returns a list of node_id lists, where each inner list can execute concurrently.
        """
        self._check_for_cycles()

        in_degree: dict[str, int] = {node_id: 0 for node_id in self.nodes}
        adj: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}

        for edge in self.edges:
            adj[edge.from_op].append(edge.to_op)
            in_degree[edge.to_op] += 1

        groups: list[list[str]] = []

        # Nodes with no unresolved dependencies
        current_layer = [n for n, d in in_degree.items() if d == 0]
        # Preserve deterministic ordering
        current_layer.sort()

        while current_layer:
            groups.append(current_layer)
            next_layer = []
            for u in current_layer:
                for v in adj[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        next_layer.append(v)
            next_layer.sort()
            current_layer = next_layer

        return groups

    def parallelizable_fraction(self) -> float:
        """
        P in Amdahl's law: cost-weighted fraction of independent nodes.
        Returns a float between 0.0 and 1.0.
        """
        if not self.nodes:
            return 0.0

        self._check_for_cycles()

        total_time = sum(n.estimated_cost.estimated_runtime_ms for n in self.nodes.values())
        indep_nodes = self.independent_operations()

        if total_time == 0:
            return float(len(indep_nodes)) / float(len(self.nodes))

        indep_time = sum(self.nodes[n_id].estimated_cost.estimated_runtime_ms for n_id in indep_nodes)
        return float(indep_time) / float(total_time)
