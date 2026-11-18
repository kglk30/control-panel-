"""Amdahl's Law computation for execution strategy planning."""

__all__ = ["theoretical_speedup", "max_theoretical_speedup"]


def theoretical_speedup(parallelizable_fraction: float, worker_count: int) -> float:
    """
    Calculate theoretical speedup S(N) using Amdahl's Law.

    Formula: S(N) = 1 / ((1 - P) + (P / N))

    Args:
        parallelizable_fraction (float): P, the fraction of execution time that can be parallelized (0.0 to 1.0).
        worker_count (int): N, the number of concurrent workers (> 0).

    Returns:
        float: Theoretical speedup multiplier.
    """
    if not isinstance(parallelizable_fraction, (int, float)):
        raise TypeError("parallelizable_fraction must be a float")
    if not isinstance(worker_count, int):
        raise TypeError("worker_count must be an integer")

    p = float(parallelizable_fraction)
    if p < 0.0 or p > 1.0:
        raise ValueError("parallelizable_fraction must be between 0.0 and 1.0")
    if worker_count < 1:
        raise ValueError("worker_count must be a positive integer (N >= 1)")

    if worker_count == 1 or p == 0.0:
        return 1.0

    if p == 1.0:
        return float(worker_count)

    return 1.0 / ((1.0 - p) + (p / worker_count))


def max_theoretical_speedup(parallelizable_fraction: float) -> float:
    """
    Calculate maximum theoretical speedup as N approaches infinity.

    Formula: S_max = 1 / (1 - P)

    Args:
        parallelizable_fraction (float): P, between 0.0 and 1.0.

    Returns:
        float: Maximum speedup multiplier, or float('inf') if P == 1.0.
    """
    if not isinstance(parallelizable_fraction, (int, float)):
        raise TypeError("parallelizable_fraction must be a float")

    p = float(parallelizable_fraction)
    if p < 0.0 or p > 1.0:
        raise ValueError("parallelizable_fraction must be between 0.0 and 1.0")

    if p == 1.0:
        return float('inf')

    return 1.0 / (1.0 - p)
