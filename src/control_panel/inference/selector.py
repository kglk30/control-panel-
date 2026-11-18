"""Backend Selection Engine — chooses the best inference backend for each request.

Implements the full candidate filtering + scoring algorithm from V15 §8.2.
Does NOT execute inference — only selects. The Inference Runtime executes.

Selection order:
  1. Determine runtime type from task_type
  2. Query ModelRegistry for compatible models
  3. Build candidate list (model × backend × device)
  4. Filter by availability, resources, config policy
  5. Score remaining candidates
  6. Return top candidate (fallback: RuleBasedAdapter)
"""

from __future__ import annotations

from typing import Optional

from control_panel.common.config import get_config
from control_panel.common.logging import get_logger
from control_panel.common.models import (
    BackendSelection,
    InferenceRequest,
)
from control_panel.inference.models import ModelProfile, ModelRegistry
from control_panel.resources.models import AcceleratorStatus, ResourceConstraints

__all__ = ["BackendSelectionEngine"]

log = get_logger(__name__)

# Task type → runtime type mapping
_GENAI_TASKS = {"analyze", "plan", "reason", "generate"}
_STRUCTURED_TASKS = {"classify", "embed", "extract"}

# Device scoring (higher = preferred)
_DEVICE_SCORES = {"npu": 100, "gpu": 80, "cpu": 60, "remote": 40, "none": 20}

# Fallback model IDs
_RULE_BASED_MODEL_ID = "rule_based_v1"


class _Candidate:
    """Internal candidate representation during selection."""

    __slots__ = (
        "model", "backend", "device", "runtime_type", "score", "reason_parts"
    )

    def __init__(
        self,
        model: ModelProfile,
        backend: str,
        device: str,
        runtime_type: str,
    ) -> None:
        self.model = model
        self.backend = backend
        self.device = device
        self.runtime_type = runtime_type
        self.score: float = 0.0
        self.reason_parts: list[str] = []


class BackendSelectionEngine:
    """Selects the best (backend, model, device) for an InferenceRequest.

    Consumes:
    - InferenceRequest (task_type, prompt)
    - ResourceConstraints (from ResourceGovernor)
    - AcceleratorStatus dict (from AcceleratorProbe)
    - ModelRegistry (compatible models)
    - ConfigPolicy (user preference)

    Returns a BackendSelection with the chosen backend and reasoning.
    """

    def __init__(self, model_registry: ModelRegistry) -> None:
        self._registry = model_registry

    def select(
        self,
        request: InferenceRequest,
        constraints: ResourceConstraints,
        accelerators: dict[str, AcceleratorStatus],
        config_policy: Optional[dict] = None,
        experience: Optional[dict] = None,
    ) -> BackendSelection:
        """Run the full selection algorithm and return the best candidate."""
        cfg = get_config().inference
        policy = config_policy or {}
        preferred_backend = (
            policy.get("preferred_backend")
            or (cfg.preferred_backend if cfg.preferred_backend else None)
        )

        # Step 1: Determine runtime type
        task_type = request.task_type
        if task_type in _GENAI_TASKS:
            runtime_type = "genai"
        elif task_type in _STRUCTURED_TASKS:
            runtime_type = "structured"
        else:
            runtime_type = "genai"  # Default for unknown task types

        # Step 2: Query ModelRegistry
        candidate_models = self._registry.find_compatible(
            task_type=task_type,
            runtime_type=runtime_type,
        )

        # Always include rule_based as a universal fallback
        rule_model = self._registry.get(_RULE_BASED_MODEL_ID)

        # Step 3: Build candidate list
        candidates: list[_Candidate] = []
        for model in candidate_models:
            for backend in model.supported_backends:
                for device in model.supported_devices:
                    candidates.append(_Candidate(model, backend, device, runtime_type))

        # Step 4: Filter by availability
        candidates = self._filter_availability(candidates, accelerators)

        # Step 5: Filter by resource constraints
        candidates = self._filter_resources(candidates, constraints)

        # Step 6: Score remaining candidates
        for cand in candidates:
            self._score(cand, constraints, accelerators, preferred_backend, experience)

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)

        if candidates:
            best = candidates[0]
            alternatives = [
                f"{c.backend}:{c.device}" for c in candidates[1:4]
            ]
            reason = (
                f"Selected {best.backend}/{best.device} "
                f"(score={best.score:.1f}): {', '.join(best.reason_parts[:2])}"
            )
            log.info(
                "selector.selected",
                backend=best.backend,
                device=best.device,
                model_id=best.model.model_id,
                score=best.score,
                task_type=task_type,
            )
            return BackendSelection(
                backend=best.backend,
                device=best.device,
                runtime_type=runtime_type,
                model_id=best.model.model_id,
                reason=reason,
                alternatives=alternatives,
            )

        # Fallback: RuleBasedAdapter always survives
        log.warning(
            "selector.fallback_to_rule_based",
            task_type=task_type,
            reason="No candidates survived filtering",
        )
        return BackendSelection(
            backend="rule_based",
            device="none",
            runtime_type="rule_based",
            model_id=_RULE_BASED_MODEL_ID,
            reason="No viable backends found — falling back to rule-based",
            alternatives=[],
        )

    # ------------------------------------------------------------------
    # Step 4: Filter by availability
    # ------------------------------------------------------------------

    def _filter_availability(
        self,
        candidates: list[_Candidate],
        accelerators: dict[str, AcceleratorStatus],
    ) -> list[_Candidate]:
        """Remove candidates whose backend/device is unavailable."""
        available: list[_Candidate] = []
        for cand in candidates:
            if cand.backend == "snapdragon":
                # Snapdragon requires NPU or GPU to be actually available
                npu = accelerators.get("npu")
                gpu = accelerators.get("gpu")
                npu_ok = npu is not None and npu.available
                gpu_ok = gpu is not None and gpu.available
                if cand.device == "npu" and not npu_ok:
                    continue
                if cand.device == "gpu" and not gpu_ok:
                    continue
                if cand.device not in ("npu", "gpu"):
                    continue

            elif cand.backend == "local_cpu":
                # CPU EP requires onnxruntime
                cpu = accelerators.get("cpu")
                if cpu is None or not cpu.available:
                    continue

            elif cand.backend == "api":
                from control_panel.common.config import get_api_key
                if not get_api_key():
                    continue

            # rule_based always passes
            available.append(cand)

        return available

    # ------------------------------------------------------------------
    # Step 5: Filter by resource constraints
    # ------------------------------------------------------------------

    def _filter_resources(
        self,
        candidates: list[_Candidate],
        constraints: ResourceConstraints,
    ) -> list[_Candidate]:
        """Remove candidates that exceed RAM constraints."""
        affordable: list[_Candidate] = []
        for cand in candidates:
            req = cand.model.requirements
            if req.estimated_ram_mb == 0:
                # No RAM requirement (e.g. rule_based)
                affordable.append(cand)
                continue
            needed = req.estimated_ram_mb + constraints.safety_margin_mb
            if needed <= constraints.available_ram_mb:
                affordable.append(cand)
            else:
                log.debug(
                    "selector.filtered_ram",
                    model_id=cand.model.model_id,
                    needed_mb=needed,
                    available_mb=constraints.available_ram_mb,
                )
        return affordable

    # ------------------------------------------------------------------
    # Step 7: Score candidates
    # ------------------------------------------------------------------

    def _score(
        self,
        cand: _Candidate,
        constraints: ResourceConstraints,
        accelerators: dict[str, AcceleratorStatus],
        preferred_backend: Optional[str],
        experience: Optional[dict],
    ) -> None:
        """Compute a weighted score for a candidate."""
        score = 0.0
        parts: list[str] = []

        # Accelerator match score
        device_score = _DEVICE_SCORES.get(cand.device, 0)
        score += device_score
        parts.append(f"device={cand.device}(+{device_score})")

        # Model available on disk (+30)
        if cand.model.is_available_on_disk():
            score += 30
            parts.append("model_on_disk(+30)")

        # Config preference (+20)
        if preferred_backend and cand.backend == preferred_backend:
            score += 20
            parts.append(f"preferred(+20)")

        # Historical latency (from experience)
        if experience:
            key = f"{cand.backend}:{cand.device}"
            hist = experience.get(key, {})
            if hist.get("success_rate", 0) > 0.8:
                score += 10
                parts.append("history_good(+10)")
            avg_latency = hist.get("avg_latency_ms", 0)
            if 0 < avg_latency < 500:
                score += 5
                parts.append("latency_ok(+5)")

        # Resource efficiency: headroom score
        ram = cand.model.requirements.estimated_ram_mb
        if ram > 0 and constraints.available_ram_mb > 0:
            efficiency = (constraints.available_ram_mb - ram) / constraints.available_ram_mb
            score += max(0, efficiency * 10)

        cand.score = score
        cand.reason_parts = parts
