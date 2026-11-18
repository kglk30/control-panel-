"""Experience-based calibration for USL parameters."""

from typing import Sequence

from control_panel.experience.models import Experience

__all__ = ["calibrate_usl_parameters"]

DEFAULT_ALPHA = 0.1
DEFAULT_BETA = 0.01


def calibrate_usl_parameters(history: Sequence[Experience]) -> tuple[float, float]:
    """
    Estimate USL alpha (contention) and beta (coherency) from historical executions.

    Args:
        history: A sequence of Experience records.

    Returns:
        (alpha, beta) tuple.
    """
    valid_history = [
        exp for exp in history
        if exp.outcome.success and exp.execution.total_runtime_ms > 0 and exp.worker_count >= 1
    ]

    if not valid_history:
        return DEFAULT_ALPHA, DEFAULT_BETA

    # Calculate alpha based on average parallelizable_fraction from the history
    total_p = sum(exp.parallelizable_fraction for exp in valid_history)
    avg_p = total_p / len(valid_history)
    alpha = max(0.0, 1.0 - avg_p)

    # Calculate average runtime per worker count T(N)
    runtimes_by_n: dict[int, list[int]] = {}
    for exp in valid_history:
        n = exp.worker_count
        runtimes_by_n.setdefault(n, []).append(exp.execution.total_runtime_ms)

    avg_t = {n: sum(times) / len(times) for n, times in runtimes_by_n.items()}

    # Need T(1) and at least one N > 1 to estimate beta empirically
    if 1 not in avg_t or len(avg_t) < 2:
        return alpha, DEFAULT_BETA

    t_1 = avg_t[1]
    beta_estimates = []

    for n, t_n in avg_t.items():
        if n == 1:
            continue

        # From USL: S(N) = N / (1 + alpha*(N-1) + beta*N*(N-1))
        # and S(N) = T(1) / T(N)
        # Therefore: (N * T(N) / T(1)) - 1 = alpha*(N-1) + beta*N*(N-1)
        # z = ((N * T(N) / T(1)) - 1) / (N - 1) = alpha + beta * N
        # beta = (z - alpha) / N

        z = ((float(n) * float(t_n) / float(t_1)) - 1.0) / float(n - 1)
        beta = (z - alpha) / float(n)
        beta_estimates.append(beta)

    avg_beta = sum(beta_estimates) / len(beta_estimates)
    final_beta = max(0.0, avg_beta)

    return alpha, final_beta
