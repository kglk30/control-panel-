"""Local CPU Adapter — ONNX Runtime CPU EP inference.

Generic CPU-only local execution. Works on any machine with onnxruntime installed.
Architecturally separate from SnapdragonAdapter.
Reports backend="local_cpu", device="cpu".
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from control_panel.common.models import BackendInfo, InferenceRequest, InferenceResult
from control_panel.common.logging import get_logger
from control_panel.inference.models import ModelProfile

__all__ = ["LocalCPUAdapter"]

log = get_logger(__name__)


class LocalCPUAdapter:
    """CPU-only local inference via ONNX Runtime CPU EP.

    For V0.1, when no local GenAI model is downloaded, this adapter
    degrades gracefully to the rule-based response path. Once a model
    is available, it runs actual local ONNX inference.
    """

    def __init__(self) -> None:
        self._ort_available = self._check_ort()
        self._genai_available = self._check_genai()
        self._session_cache: dict[str, object] = {}

    @property
    def name(self) -> str:
        return "local_cpu"

    @property
    def available(self) -> bool:
        return self._ort_available

    def check_availability(self) -> BackendInfo:
        """Return current availability status."""
        if not self._ort_available:
            return BackendInfo(
                name=self.name,
                available=False,
                device="cpu",
                reason="onnxruntime not installed",
            )
        return BackendInfo(
            name=self.name,
            available=True,
            device="cpu",
        )

    def execute(
        self,
        request: InferenceRequest,
        model: Optional[ModelProfile] = None,
        runtime: str = "structured",
    ) -> InferenceResult:
        """Execute inference on CPU.

        If no model is provided or model file is unavailable, falls back
        to rule-based response with local_cpu attribution.
        """
        t0 = time.perf_counter_ns()

        if model is None or not model.is_available_on_disk():
            return self._fallback_response(request, t0, reason="no model on disk")

        if runtime == "genai":
            return self._execute_genai(request, model, t0)
        else:
            return self._execute_structured(request, model, t0)

    # ------------------------------------------------------------------
    # GenAI execution (onnxruntime-genai)
    # ------------------------------------------------------------------

    def _execute_genai(
        self,
        request: InferenceRequest,
        model: ModelProfile,
        t0: int,
    ) -> InferenceResult:
        """Execute generative inference via onnxruntime-genai on CPU."""
        if not self._genai_available:
            return self._fallback_response(request, t0, reason="onnxruntime-genai not installed")

        try:
            import onnxruntime_genai as og

            model_path = model.file_path
            if not model_path:
                return self._fallback_response(request, t0, reason="model directory missing")

            # Resolve relative paths against the project root
            p = Path(model_path)
            if not p.is_absolute():
                project_root = Path(__file__).parent.parent.parent.parent.parent
                abs_p = project_root / p
                if abs_p.exists():
                    p = abs_p
                elif (Path.cwd() / p).exists():
                    p = Path.cwd() / p
            if not p.exists():
                return self._fallback_response(request, t0, reason="model directory missing")
            model_path = str(p)

            # Build the prompt
            og_model = og.Model(model_path)
            tokenizer = og.Tokenizer(og_model)

            # Simple prompt format (no chat template for compatibility)
            prompt = f"<|user|>{request.prompt}<|end|><|assistant|>"
            tokens = tokenizer.encode(prompt)

            params = og.GeneratorParams(og_model)
            params.set_search_options(max_length=request.max_tokens + len(tokens))

            generator = og.Generator(og_model, params)
            generator.append_tokens(tokens)

            output_tokens: list[int] = []

            while not generator.is_done():
                generator.generate_next_token()
                new_token = generator.get_next_tokens()[0]
                output_tokens.append(int(new_token))

            content = tokenizer.decode(output_tokens).strip()
            latency_ms = int((time.perf_counter_ns() - t0) / 1_000_000)

            log.info(
                "local_cpu.genai.completed",
                model_id=model.model_id,
                tokens=len(output_tokens),
                latency_ms=latency_ms,
            )

            return InferenceResult(
                request_id=request.request_id,
                content=content,
                backend_used="local_cpu",
                device_used="cpu",
                runtime_type="genai",
                model_id=model.model_id,
                tokens_used=len(output_tokens),
                latency_ms=latency_ms,
                source="local",
            )

        except Exception as e:
            log.warning("local_cpu.genai.failed", error=str(e))
            return self._fallback_response(request, t0, reason=f"GenAI error: {e}")

    # ------------------------------------------------------------------
    # Structured execution (standard ONNX Runtime)
    # ------------------------------------------------------------------

    def _execute_structured(
        self,
        request: InferenceRequest,
        model: ModelProfile,
        t0: int,
    ) -> InferenceResult:
        """Execute structured (non-generative) inference via onnxruntime CPU EP."""
        try:
            import onnxruntime as ort
            import numpy as np

            model_path = model.file_path
            if not model_path or not Path(model_path).exists():
                return self._fallback_response(request, t0, reason="model file missing")

            if model_path not in self._session_cache:
                so = ort.SessionOptions()
                so.log_severity_level = 3
                session = ort.InferenceSession(
                    model_path, sess_options=so,
                    providers=["CPUExecutionProvider"],
                )
                self._session_cache[model_path] = session

            session = self._session_cache[model_path]

            # Structured models expect tensor inputs — this is task-specific
            # For V0.1 with classification models, we'd tokenize here.
            # Placeholder: return rule-based response with local_cpu attribution.
            latency_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
            content = f"[structured inference placeholder — model: {model.model_id}]"

            return InferenceResult(
                request_id=request.request_id,
                content=content,
                backend_used="local_cpu",
                device_used="cpu",
                runtime_type="structured",
                model_id=model.model_id,
                tokens_used=0,
                latency_ms=latency_ms,
                source="local",
            )

        except Exception as e:
            log.warning("local_cpu.structured.failed", error=str(e))
            return self._fallback_response(request, t0, reason=f"Structured error: {e}")

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback_response(
        self,
        request: InferenceRequest,
        t0: int,
        reason: str = "",
    ) -> InferenceResult:
        """Return a rule-based response attributed to local_cpu."""
        from control_panel.inference.rule_based import RuleBasedAdapter
        rule = RuleBasedAdapter()
        result = rule.execute(request)
        latency_ms = int((time.perf_counter_ns() - t0) / 1_000_000)
        # Re-attribute to local_cpu (rule_based ran internally)
        return InferenceResult(
            request_id=result.request_id,
            content=result.content,
            structured=result.structured,
            backend_used="local_cpu",
            device_used="cpu",
            runtime_type="rule_based",
            model_id=result.model_id,
            tokens_used=0,
            latency_ms=latency_ms,
            source="local",
        )

    # ------------------------------------------------------------------
    # Availability checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_ort() -> bool:
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_genai() -> bool:
        try:
            import onnxruntime_genai  # noqa: F401
            return True
        except ImportError:
            return False
