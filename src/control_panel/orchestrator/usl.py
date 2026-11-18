"""Universal Scalability Law (USL) computation for execution strategy planning."""

__all__ = ["theoretical_speedup_usl"]


def theoretical_speedup_usl(worker_count: int, alpha: float, beta: float) -> float:
    """
    Calculate theoretical speedup S(N) using the Universal Scalability Law.

    Formula: S(N) = N / (1 + alpha*(N-1) + beta*N*(N-1))

    Args:
        worker_count (int): N, the number of concurrent workers (>= 1).
        alpha (float): Contention parameter (>= 0). Represents serialization.
        beta (float): Coherency parameter (>= 0). Represents crosstalk/overhead.

    Returns:
        float: Theoretical speedup multiplier.
    """
    if not isinstance(worker_count, int) or isinstance(worker_count, bool):
        raise TypeError("worker_count must be an integer")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise TypeError("alpha must be a float")
    if not isinstance(beta, (int, float)) or isinstance(beta, bool):
        raise TypeError("beta must be a float")

    n = float(worker_count)
    a = float(alpha)
    b = float(beta)

    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")
    if a < 0.0:
        raise ValueError("alpha must be >= 0")
    if b < 0.0:
        raise ValueError("beta must be >= 0")

    if worker_count == 1:
        return 1.0

    denominator = 1.0 + a * (n - 1.0) + b * n * (n - 1.0)
    return n / denominator
