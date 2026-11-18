"""Tests for common data models."""

from control_panel.common.models import ExecutionStrategy
import pytest
from pydantic import ValidationError

def test_execution_strategy_valid():
    """Test creating a valid ExecutionStrategy."""
    strategy = ExecutionStrategy(
        strategy_type="hybrid",
        recommended_worker_count=3,
        parallelizable_fraction=0.75,
        theoretical_speedup=2.5,
        rationale="P=0.75 supports 3 workers"
    )
    assert strategy.strategy_type == "hybrid"
    assert strategy.recommended_worker_count == 3
    assert strategy.parallelizable_fraction == 0.75
    assert strategy.theoretical_speedup == 2.5
    assert strategy.rationale == "P=0.75 supports 3 workers"

def test_execution_strategy_missing_fields():
    """Test ExecutionStrategy requires all mandatory fields."""
    with pytest.raises(ValidationError):
        ExecutionStrategy(
            strategy_type="sequential"
        )

def test_execution_strategy_type_conversion():
    """Test Pydantic type conversion."""
    strategy = ExecutionStrategy(
        strategy_type="parallel",
        recommended_worker_count="4",
        parallelizable_fraction="0.9",
        theoretical_speedup=3,
        rationale="Tests conversion"
    )
    assert strategy.recommended_worker_count == 4
    assert strategy.parallelizable_fraction == 0.9
    assert strategy.theoretical_speedup == 3.0
