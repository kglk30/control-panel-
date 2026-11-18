"""ExecutionStrategyPlanner for determining optimal task execution strategy."""

from typing import Sequence

from control_panel.common.models import ExecutionStrategy
from control_panel.experience.models import Experience
from control_panel.orchestrator.depgraph import DependencyGraph
from control_panel.orchestrator.calibration import calibrate_usl_parameters
from control_panel.orchestrator.strategy_math import select_optimal_strategy

__all__ = ["ExecutionStrategyPlanner"]


class ExecutionStrategyPlanner:
    """Planner that decides how to execute a task based on USL and experience."""

    def __init__(self, n_max: int = 10):
        """
        Initialize the planner.

        Args:
            n_max (int): The absolute maximum number of concurrent workers allowed
                         by the system/configuration.
        """
        self.n_max = n_max

    def plan_strategy(
        self,
        graph: DependencyGraph,
        history: Sequence[Experience]
    ) -> ExecutionStrategy:
        """
        Produce an ExecutionStrategy for the given graph and historical experiences.

        Args:
            graph (DependencyGraph): The parsed dependency graph for the task.
            history (Sequence[Experience]): List of relevant historical executions.

        Returns:
            ExecutionStrategy: The optimal execution strategy parameters.
        """
        # 1. Calibrate USL parameters (alpha, beta) using historical experience
        # If history is empty or invalid, this safely falls back to defaults.
        alpha, beta = calibrate_usl_parameters(history)

        # 2. Select optimal strategy using mathematical layer
        # Evaluates candidates from N=1 to N=min(n_max, available_independent_work)
        strategy_dict = select_optimal_strategy(graph, alpha, beta, self.n_max)

        # 3. Construct and return the final ExecutionStrategy Pydantic model
        return ExecutionStrategy(
            strategy_type=strategy_dict["strategy_type"],
            recommended_worker_count=strategy_dict["recommended_worker_count"],
            parallelizable_fraction=strategy_dict["parallelizable_fraction"],
            theoretical_speedup=strategy_dict["theoretical_speedup"],
            rationale=strategy_dict["rationale"]
        )
