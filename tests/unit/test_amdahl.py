"""Unit tests for Amdahl's Law computation."""

import pytest
from control_panel.orchestrator.amdahl import theoretical_speedup, max_theoretical_speedup


def test_amdahl_p_zero():
    """Test P=0 (fully sequential)."""
    assert theoretical_speedup(0.0, 1) == 1.0
    assert theoretical_speedup(0.0, 5) == 1.0
    assert theoretical_speedup(0.0, 100) == 1.0
    assert max_theoretical_speedup(0.0) == 1.0


def test_amdahl_p_one():
    """Test P=1 (fully parallelizable)."""
    assert theoretical_speedup(1.0, 1) == 1.0
    assert theoretical_speedup(1.0, 2) == 2.0
    assert theoretical_speedup(1.0, 5) == 5.0
    assert max_theoretical_speedup(1.0) == float('inf')


def test_amdahl_intermediate_p():
    """Test intermediate P values."""
    # P = 0.5, N = 2 => S = 1 / (0.5 + 0.25) = 1 / 0.75 = 1.333...
    assert pytest.approx(theoretical_speedup(0.5, 2)) == 1.3333333333333333

    # P = 0.95, N = 3 => S = 1 / (0.05 + 0.95/3) = 1 / (0.05 + 0.31666) = 1 / 0.36666 = 2.7272...
    assert pytest.approx(theoretical_speedup(0.95, 3)) == 2.7272727272727275

    # max for P=0.5 is 2.0
    assert max_theoretical_speedup(0.5) == 2.0
    # max for P=0.95 is 20.0
    assert pytest.approx(max_theoretical_speedup(0.95)) == 20.0


def test_amdahl_worker_counts():
    """Test same P with increasing worker counts."""
    p = 0.8
    s1 = theoretical_speedup(p, 1)
    s2 = theoretical_speedup(p, 2)
    s4 = theoretical_speedup(p, 4)
    s8 = theoretical_speedup(p, 8)

    assert s1 == 1.0
    assert s2 > s1
    assert s4 > s2
    assert s8 > s4
    assert s8 < max_theoretical_speedup(p)


def test_invalid_p():
    """Test invalid P values (<0 or >1)."""
    with pytest.raises(ValueError):
        theoretical_speedup(-0.1, 2)

    with pytest.raises(ValueError):
        theoretical_speedup(1.01, 2)

    with pytest.raises(ValueError):
        max_theoretical_speedup(-0.5)

    with pytest.raises(ValueError):
        max_theoretical_speedup(1.5)


def test_invalid_worker_count():
    """Test invalid worker counts (<1)."""
    with pytest.raises(ValueError):
        theoretical_speedup(0.5, 0)

    with pytest.raises(ValueError):
        theoretical_speedup(0.5, -1)


def test_invalid_types():
    """Test passing invalid types."""
    with pytest.raises(TypeError):
        theoretical_speedup("0.5", 2)

    with pytest.raises(TypeError):
        theoretical_speedup(0.5, "2")

    with pytest.raises(TypeError):
        max_theoretical_speedup("0.5")
