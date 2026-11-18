from unittest.mock import patch, MagicMock

import pytest

from control_panel.resources.governor import ResourceGovernor
from control_panel.resources.models import ResourceConstraints

def test_resource_constraints_exposes_max_concurrent_workers():
    """Test 1: ResourceConstraints exposes max_concurrent_workers."""
    rc = ResourceConstraints(
        available_ram_mb=1000,
        safety_margin_mb=512,
        max_model_ram_mb=488,
        cpu_pressure="low",
        power_policy="normal",
        max_concurrent_workers=5,
        preferred_device="cpu",
        can_load_local_model=True,
        reason="ok"
    )
    assert rc.max_concurrent_workers == 5

@patch("os.cpu_count")
def test_normal_cpu_detection(mock_cpu_count):
    """Test 2: Normal CPU detection (mock os.cpu_count() = 8 -> 7)."""
    mock_cpu_count.return_value = 8
    governor = ResourceGovernor()
    constraints = governor.get_resource_constraints()
    assert constraints.max_concurrent_workers == 7

@patch("os.cpu_count")
def test_fallback_cpu_detection(mock_cpu_count):
    """Test 3: Fallback (mock os.cpu_count() = None -> 1)."""
    mock_cpu_count.return_value = None
    # FALLBACK is 2, RESERVED is 1. max(1, 2 - 1) = 1.
    governor = ResourceGovernor()
    constraints = governor.get_resource_constraints()
    assert constraints.max_concurrent_workers == 1

@patch("os.cpu_count")
def test_low_cpu_count(mock_cpu_count):
    """Test 4: Low CPU count (mock os.cpu_count() = 1 -> 1)."""
    mock_cpu_count.return_value = 1
    # max(1, 1 - 1) = 1
    governor = ResourceGovernor()
    constraints = governor.get_resource_constraints()
    assert constraints.max_concurrent_workers == 1
