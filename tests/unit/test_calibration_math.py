import pytest
from control_panel.experience.models import Experience, ExperienceOutcome, ExperienceExecution, ExperiencePlan, ExperienceTask
from control_panel.orchestrator.calibration import calibrate_usl_parameters

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

def test_usl_calibration_from_persisted_history():
    history = [
        make_exp(n=1, t=1000, success=True, p=0.9),
        make_exp(n=2, t=600, success=True, p=0.9),
        make_exp(n=4, t=500, success=True, p=0.9),
        # Failed execution that should be ignored
        make_exp(n=8, t=100, success=False, p=0.9)
    ]

    alpha, beta = calibrate_usl_parameters(history)

    # 1. Check alpha: avg_p = 0.9 -> alpha = 1.0 - 0.9 = 0.1
    assert round(alpha, 6) == 0.1

    # 2. Check beta:
    # For N=2, T=600, T(1)=1000 -> S(2) = 1.666
    # z2 = ((2 * 600 / 1000) - 1.0) / 1 = 0.2
    # beta2 = (0.2 - 0.1) / 2 = 0.05
    #
    # For N=4, T=500, T(1)=1000 -> S(4) = 2.0
    # z4 = ((4 * 500 / 1000) - 1.0) / 3 = 1.0 / 3 = 0.33333333
    # beta4 = (0.33333333 - 0.1) / 4 = 0.23333333 / 4 = 0.05833333
    #
    # avg_beta = (0.05 + 0.05833333) / 2 = 0.05416666666666667

    assert round(beta, 6) == 0.054167
