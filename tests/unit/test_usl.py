"""Unit tests for Universal Scalability Law computation."""

import pytest
from control_panel.orchestrator.usl import theoretical_speedup_usl


def test_usl_n_one():
    """Test that N=1 always returns a speedup of exactly 1.0."""
    assert theoretical_speedup_usl(1, 0.0, 0.0) == 1.0
    assert theoretical_speedup_usl(1, 0.5, 0.0) == 1.0
    assert theoretical_speedup_usl(1, 0.1, 0.1) == 1.0


def test_usl_alpha_beta_zero():
    """Test that alpha=0 and beta=0 results in linear scaling (S=N)."""
    assert theoretical_speedup_usl(1, 0.0, 0.0) == 1.0
    assert theoretical_speedup_usl(2, 0.0, 0.0) == 2.0
    assert theoretical_speedup_usl(10, 0.0, 0.0) == 10.0


def test_usl_contention_only():
    """Test contention without coherency overhead (effectively Amdahl's Law)."""
    # alpha = 0.5 (equivalent to P=0.5 in Amdahl's Law where alpha = 1 - P)
    # S(2) = 2 / (1 + 0.5*(1)) = 2 / 1.5 = 1.333...
    assert pytest.approx(theoretical_speedup_usl(2, 0.5, 0.0)) == 1.3333333333333333

    # S(3) = 3 / (1 + 0.5*(2)) = 3 / 2 = 1.5
    assert theoretical_speedup_usl(3, 0.5, 0.0) == 1.5


def test_usl_coherency_only():
    """Test coherency penalty without contention."""
    # beta = 0.1, N = 4
    # S(4) = 4 / (1 + 0 + 0.1 * 4 * 3) = 4 / (1 + 1.2) = 4 / 2.2 = 1.8181...
    assert pytest.approx(theoretical_speedup_usl(4, 0.0, 0.1)) == 4.0 / 2.2


def test_usl_both_parameters():
    """Test both contention and coherency parameters active."""
    # N = 5, alpha = 0.1, beta = 0.05
    # denominator = 1 + 0.1*(4) + 0.05*(5)*(4) = 1 + 0.4 + 1.0 = 2.4
    # S(5) = 5 / 2.4 = 2.08333...
    assert pytest.approx(theoretical_speedup_usl(5, 0.1, 0.05)) == 5.0 / 2.4


def test_usl_invalid_n():
    """Test validation of N < 1."""
    with pytest.raises(ValueError, match="worker_count must be >= 1"):
        theoretical_speedup_usl(0, 0.1, 0.1)

    with pytest.raises(ValueError, match="worker_count must be >= 1"):
        theoretical_speedup_usl(-1, 0.1, 0.1)


def test_usl_invalid_alpha_beta():
    """Test validation of negative alpha and beta."""
    with pytest.raises(ValueError, match="alpha must be >= 0"):
        theoretical_speedup_usl(2, -0.1, 0.0)

    with pytest.raises(ValueError, match="beta must be >= 0"):
        theoretical_speedup_usl(2, 0.1, -0.01)


def test_usl_invalid_types():
    """Test validation of invalid types."""
    with pytest.raises(TypeError, match="worker_count must be an integer"):
        theoretical_speedup_usl(2.5, 0.1, 0.1)  # type: ignore

    with pytest.raises(TypeError, match="alpha must be a float"):
        theoretical_speedup_usl(2, "0.1", 0.1)  # type: ignore

    with pytest.raises(TypeError, match="beta must be a float"):
        theoretical_speedup_usl(2, 0.1, "0.1")  # type: ignore
