"""Comparator — compares new Experience against history to recommend strategy.

The Orchestrator makes the final decision. The Comparator only recommends.
"""

from __future__ import annotations

from typing import Optional

from control_panel.common.logging import get_logger
from control_panel.experience.models import Experience, StrategyDecision

__all__ = ["Comparator"]

log = get_logger(__name__)


class Comparator:
    """Compares a new Experience against historical experiences.

    Returns a recommended StrategyDecision.
    The Orchestrator makes the final call (KEEP / UPDATE / CREATE_NEW).
    """

    def compare(
        self,
        new_experience: Experience,
        history: list[Experience],
    ) -> StrategyDecision:
        """Compare new experience against history and recommend a strategy.

        V0.1: Simple deterministic comparison.
        """
        if not history:
            return StrategyDecision(
                action="CREATE_NEW",
                reason="No prior experience for this category",
                target_template=None,
            )

        # Find the most recent successful experience
        best_prior = next(
            (e for e in history if e.outcome.success), None
        )

        if best_prior is None:
            return StrategyDecision(
                action="CREATE_NEW",
                reason="All prior experiences failed — creating fresh template",
            )

        # Compare performance
        new_runtime = new_experience.execution.total_runtime_ms
        prior_runtime = best_prior.execution.total_runtime_ms

        if new_experience.outcome.success:
            if prior_runtime > 0 and new_runtime < prior_runtime * 0.8:
                # Significantly faster — update the template
                return StrategyDecision(
                    action="UPDATE",
                    reason=(
                        f"New execution {new_runtime}ms is 20%+ faster than prior "
                        f"{prior_runtime}ms — updating template"
                    ),
                    target_template=best_prior.plan.template_id,
                )
            else:
                return StrategyDecision(
                    action="KEEP",
                    reason=(
                        f"Performance similar to prior experience "
                        f"({new_runtime}ms vs {prior_runtime}ms) — keeping template"
                    ),
                    target_template=best_prior.plan.template_id,
                )
        else:
            return StrategyDecision(
                action="KEEP",
                reason="Current execution failed — keeping existing template unchanged",
                target_template=best_prior.plan.template_id,
            )
