"""Unit tests for USL calibration."""

import pytest
from control_panel.experience.models import Experience, ExperienceOutcome, ExperienceExecution
from control_panel.orchestrator.calibration import calibrate_usl_parameters, DEFAULT_ALPHA, DEFAULT_BETA


def _make_exp(worker_count: int, runtime_ms: int, p: float, success: bool = True) -> Experience:
    exp = Experience()
    exp.worker_count = worker_count
    exp.execution = ExperienceExecution(total_runtime_ms=runtime_ms)
    exp.parallelizable_fraction = p
    exp.outcome = ExperienceOutcome(success=success)
    return exp


def test_no_history():
    """Test calibration with empty history."""
    alpha, beta = calibrate_usl_parameters([])
    assert alpha == DEFAULT_ALPHA
    assert beta == DEFAULT_BETA


def test_insufficient_history():
    """Test calibration with only N=1 data or failing data."""
    # Only failing data
    exp1 = _make_exp(1, 100, 0.9, False)
    alpha, beta = calibrate_usl_parameters([exp1])
    assert alpha == DEFAULT_ALPHA
    assert beta == DEFAULT_BETA

    # Only N=1 data -> can estimate alpha from P, but not beta
    exp2 = _make_exp(1, 100, 0.9, True)
    alpha, beta = calibrate_usl_parameters([exp2])
    assert pytest.approx(alpha) == 0.1  # 1.0 - 0.9
    assert beta == DEFAULT_BETA

    # Only N=2 data -> missing T(1), can estimate alpha but not beta
    exp3 = _make_exp(2, 60, 0.9, True)
    alpha, beta = calibrate_usl_parameters([exp3])
    assert pytest.approx(alpha) == 0.1
    assert beta == DEFAULT_BETA


def test_enough_successful_history():
    """Test calibration with sufficient N=1 and N>1 data."""
    # Base T(1) = 100. P = 0.9 -> alpha = 0.1
    # If beta = 0.05, S(2) = 2 / (1 + 0.1(1) + 0.05*2*1) = 2 / 1.2 = 1.666
    # T(2) = 100 / 1.666 = 60
    exp1 = _make_exp(1, 100, 0.9)
    exp2 = _make_exp(2, 60, 0.9)

    alpha, beta = calibrate_usl_parameters([exp1, exp2])
    assert pytest.approx(alpha) == 0.1
    # z = ( (2 * 60 / 100) - 1 ) / 1 = 1.2 - 1 = 0.2
    # beta = (0.2 - 0.1) / 2 = 0.05
    assert pytest.approx(beta) == 0.05


def test_contention_increasing_with_worker_count():
    """Test contention increasing via alpha logic (lower P averages)."""
    exp1 = _make_exp(1, 100, 0.5)
    exp2 = _make_exp(2, 75, 0.5)

    alpha, beta = calibrate_usl_parameters([exp1, exp2])
    assert pytest.approx(alpha) == 0.5  # Higher contention because P=0.5
    # S(2) = 100/75 = 1.333
    # z = (2 * 75 / 100 - 1) / 1 = 0.5
    # beta = (0.5 - 0.5) / 2 = 0.0
    assert pytest.approx(beta) == 0.0


def test_coherency_increasing_with_worker_count():
    """Test beta captures dramatic scaling penalties at higher N."""
    exp1 = _make_exp(1, 100, 1.0) # alpha = 0.0
    # T(2) = 60 => S(2) = 1.66 => z = (2*60/100 - 1) = 0.2 => beta = 0.1
    exp2 = _make_exp(2, 60, 1.0)
    # T(3) = 53 => S(3) = 1.88 => z = (3*53/100 - 1)/2 = 0.295 => beta = 0.295/3 = 0.098
    exp3 = _make_exp(3, 53, 1.0)

    alpha, beta = calibrate_usl_parameters([exp1, exp2, exp3])
    assert alpha == 0.0
    assert beta > 0.09 and beta < 0.11


def test_stable_low_contention_history():
    """Test a nearly perfectly scalable workload yields low alpha and beta."""
    exp1 = _make_exp(1, 100, 1.0)
    exp2 = _make_exp(2, 50, 1.0)
    exp3 = _make_exp(4, 25, 1.0)

    alpha, beta = calibrate_usl_parameters([exp1, exp2, exp3])
    assert alpha == 0.0
    assert pytest.approx(beta, abs=1e-5) == 0.0
