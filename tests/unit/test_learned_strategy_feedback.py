import pytest
from control_panel.experience.models import Experience, ExperienceTask, ExperiencePlan, ExperienceOutcome, ExperienceExecution
from control_panel.orchestrator.depgraph import DependencyGraph, DepNode, OperationCost
from control_panel.orchestrator.strategy_planner import ExecutionStrategyPlanner
from control_panel.orchestrator.calibration import calibrate_usl_parameters
from control_panel.orchestrator.strategy_math import theoretical_speedup_usl

def make_exp(n: int, t: int, success: bool = True, p: float = 0.9) -> Experience:
    return Experience(
        task=ExperienceTask(raw_input="test", category="test"),
        plan=ExperiencePlan(execution_mode="parallel" if n > 1 else "sequential", worker_count=n),
        outcome=ExperienceOutcome(success=success),
        execution=ExperienceExecution(total_runtime_ms=t),
        worker_count=n,
        parallelizable_fraction=p,
        strategy_type="parallel" if n > 1 else "sequential"
    )

def test_learned_strategy_feedback():
    # 1. Seed History
    history = [
        make_exp(n=1, t=1000, success=True, p=0.9),
        make_exp(n=2, t=600, success=True, p=0.9),
        make_exp(n=4, t=500, success=True, p=0.9),
    ]

    alpha, beta = calibrate_usl_parameters(history)
    # alpha = 0.1, beta = 0.054167

    # 2. Construct new DAG with N=8 parallelizable branches
    graph = DependencyGraph()
    for i in range(8):
        graph.add_node(DepNode(
            operation_id=f"op_{i}",
            estimated_cost=OperationCost(estimated_runtime_ms=100)
        ))

    assert graph.parallelizable_fraction() == 1.0

    # 3. Strategy Planner with generous hardware limit (n_max = 10)
    planner = ExecutionStrategyPlanner(n_max=10)
    strategy = planner.plan_strategy(graph, history)

    # Check USL outputs for understanding
    s1 = theoretical_speedup_usl(1, alpha, beta)
    s2 = theoretical_speedup_usl(2, alpha, beta)
    s3 = theoretical_speedup_usl(3, alpha, beta)
    s4 = theoretical_speedup_usl(4, alpha, beta)
    s5 = theoretical_speedup_usl(5, alpha, beta)
    s8 = theoretical_speedup_usl(8, alpha, beta)
    print(f"USL Speedups: S(1)={s1:.2f}, S(2)={s2:.2f}, S(3)={s3:.2f}, S(4)={s4:.2f}, S(5)={s5:.2f}, S(8)={s8:.2f}")

    # 4. Assertions
    # It should recommend N=4 because S(4) > S(5) due to coherency penalty (beta)
    assert strategy.recommended_worker_count == 4
    assert strategy.strategy_type == "parallel"
    assert strategy.parallelizable_fraction == 1.0

    # Hardware bound respected (if n_max was 3, it should cap at 3)
    planner_capped = ExecutionStrategyPlanner(n_max=3)
    strategy_capped = planner_capped.plan_strategy(graph, history)
    assert strategy_capped.recommended_worker_count == 3
