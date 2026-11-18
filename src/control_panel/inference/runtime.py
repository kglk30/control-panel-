"""Unified Inference Runtime — the single interface the Orchestrator uses.

The Orchestrator calls infer(request) and receives InferenceResult.
It never selects backends, never knows which model ran, never knows the device.

Internal routing:
  1. BackendSelectionEngine selects (backend, model, device)
  2. Dispatch to StructuredModelRuntime or GenAIRuntime based on task_type
  3. Backend adapter executes (Snapdragon / LocalCPU / API / RuleBased)
  4. Return InferenceResult
"""

from __future__ import annotations

from typing import Optional

from control_panel.common.models import BackendInfo, BackendSelection, InferenceRequest, InferenceResult
from control_panel.common.logging import get_logger
from control_panel.inference.api_adapter import APIAdapter
from control_panel.inference.local_cpu import LocalCPUAdapter
from control_panel.inference.models import ModelRegistry, get_registry
from control_panel.inference.rule_based import RuleBasedAdapter
from control_panel.inference.selector import BackendSelectionEngine
from control_panel.inference.snapdragon import SnapdragonAdapter
from control_panel.resources.governor import ResourceGovernor
from control_panel.resources.models import ResourceConstraints

__all__ = ["InferenceRuntime"]

log = get_logger(__name__)


class InferenceRuntime:
    """Unified inference interface. Orchestrator only sees this.

    Internally manages all backend adapters and routing logic.
    """

    def __init__(
        self,
        governor: Optional[ResourceGovernor] = None,
        model_registry: Optional[ModelRegistry] = None,
    ) -> None:
        self._governor = governor or ResourceGovernor()
        self._registry = model_registry or get_registry()
        self._selector = BackendSelectionEngine(self._registry)

        # Backend adapters
        self._snapdragon = SnapdragonAdapter()
        self._local_cpu = LocalCPUAdapter()
        self._api = APIAdapter()
        self._rule_based = RuleBasedAdapter()

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Automatic backend selection and inference execution.

        The Orchestrator always calls this. Backend selection is internal.
        """
        # Get current resource constraints and accelerator status
        constraints = self._governor.get_resource_constraints()
        accelerators = self._governor._get_accelerators()

        # Select best backend
        selection = self._selector.select(
            request=request,
            constraints=constraints,
            accelerators=accelerators,
        )

        # ------------------------------------------------------------------
        # CAPABILITY-AWARE EXECUTION VALIDATION
        # ------------------------------------------------------------------
        # Ensure the selected artifact actually runs on the backend before claiming it.
        if selection.backend == "snapdragon":
            model = self._registry.get(selection.model_id)
            if model:
                is_capable, reason = self._snapdragon.verify_compatibility(model)
                if not is_capable:
                    log.warning(
                        "inference.capability_rejected",
                        backend=selection.backend,
                        model_id=selection.model_id,
                        reason=reason,
                        fallback="local_cpu",
                    )
                    # Re-select forcing CPU fallback
                    selection = self._selector.select(
                        request=request,
                        constraints=constraints,
                        accelerators=accelerators,
                        config_policy={"preferred_backend": "local_cpu"},
                    )
                    # Preserve the compatibility rejection reason for telemetry
                    selection.reason = f"Snapdragon NPU available, but selected model/context is incompatible with the current runtime. Detail: {reason} | Fallback: LocalCPU"
        # ------------------------------------------------------------------

        log.info(
            "inference.routing",
            request_id=request.request_id,
            task_type=request.task_type,
            backend=selection.backend,
            device=selection.device,
            model_id=selection.model_id,
        )

        return self._dispatch(request, selection)

    def infer_with_backend(
        self,
        request: InferenceRequest,
        backend: str,
    ) -> InferenceResult:
        """Force a specific backend. Used for CLI --backend override and testing."""
        # Get the model registry's best model for this backend
        constraints = self._governor.get_resource_constraints()
        accelerators = self._governor._get_accelerators()

        # Override preferred backend in selection
        selection = self._selector.select(
            request=request,
            constraints=constraints,
            accelerators=accelerators,
            config_policy={"preferred_backend": backend},
        )

        # If the forced backend wasn't selected (unavailable), still honor the intent
        if selection.backend != backend:
            # Force route to requested backend regardless
            model = self._registry.find_compatible(
                task_type=request.task_type,
                runtime_type=selection.runtime_type,
                backend=backend,
            )
            model_id = model[0].model_id if model else "rule_based_v1"
            selection = BackendSelection(
                backend=backend,
                device=selection.device,
                runtime_type=selection.runtime_type,
                model_id=model_id,
                reason=f"Forced by CLI --backend={backend}",
            )

        log.info(
            "inference.forced_backend",
            backend=backend,
            request_id=request.request_id,
        )
        return self._dispatch(request, selection)

    def available_backends(self) -> list[BackendInfo]:
        """Return current availability status of all backends."""
        return [
            self._snapdragon.check_availability(),
            self._local_cpu.check_availability(),
            self._api.check_availability(),
            self._rule_based.check_availability(),
        ]

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        request: InferenceRequest,
        selection: BackendSelection,
    ) -> InferenceResult:
        """Route to the selected backend adapter."""
        model = self._registry.get(selection.model_id)

        try:
            if selection.backend == "snapdragon":
                return self._snapdragon.execute(request, model, selection.runtime_type)

            elif selection.backend == "local_cpu":
                return self._local_cpu.execute(request, model, selection.runtime_type)

            elif selection.backend == "api":
                return self._api.execute(request, model, selection.runtime_type)

            else:
                # rule_based — always available
                return self._rule_based.execute(request, model, selection.runtime_type)

        except Exception as e:
            log.warning(
                "inference.backend_failed",
                backend=selection.backend,
                error=str(e),
                fallback="rule_based",
            )
            # Emergency fallback to rule-based
            return self._rule_based.execute(request)
