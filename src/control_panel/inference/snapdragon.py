"""Snapdragon Adapter — Snapdragon X Plus hardware acceleration only.

Uses QNN EP (Hexagon NPU) as primary, DirectML EP (Adreno GPU) as secondary.
Does NOT fall back to CPU EP internally.
If neither QNN nor DML is available, this adapter reports available=False
and the Backend Selection Engine routes to LocalCPUAdapter instead.

Current status (from spike): MODEL_INCOMPATIBILITY — standard ONNX models
not QNN-compatible. Adapter correctly reports unavailable until Track B
delivers a verified QNN-compatible model.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from control_panel.common.models import BackendInfo, InferenceRequest, InferenceResult
from control_panel.common.logging import get_logger
from control_panel.inference.models import ModelProfile
from control_panel.resources.probe import AcceleratorProbe, get_probe

__all__ = ["SnapdragonAdapter"]

log = get_logger(__name__)

_QNN_EP = "QNNExecutionProvider"
_DML_EP = "DmlExecutionProvider"


class SnapdragonAdapter:
    """Snapdragon acceleration adapter — NPU/GPU only, no CPU fallback.

    Architectural invariant: This adapter is responsible for Snapdragon
    hardware acceleration exclusively. If acceleration is unavailable,
    it reports available=False. The Selection Engine then routes to
    LocalCPUAdapter.
    """

    def __init__(self, probe: Optional[AcceleratorProbe] = None) -> None:
        self._probe = probe or get_probe()
        self._npu_status = self._probe.probe("npu")
        self._gpu_status = self._probe.probe("gpu")
        self._capability_cache: dict[str, tuple[bool, str]] = {}

    @property
    def name(self) -> str:
        return "snapdragon"

    @property
    def available(self) -> bool:
        """True only if NPU or GPU is verified available via AcceleratorProbe."""
        return self._npu_status.available or self._gpu_status.available

    def check_availability(self) -> BackendInfo:
        """Return current Snapdragon availability and the reason if unavailable."""
        # Re-read from probe (may have been updated)
        self._npu_status = self._probe.probe("npu")
        self._gpu_status = self._probe.probe("gpu")

        if self._npu_status.available:
            return BackendInfo(
                name=self.name,
                available=True,
                device="npu",
                reason=None,
            )
        if self._gpu_status.available:
            return BackendInfo(
                name=self.name,
                available=True,
                device="gpu",
                reason=None,
            )

        # Both unavailable — aggregate reason
        reason_parts = []
        if self._npu_status.failure_reason:
            reason_parts.append(f"NPU: {self._npu_status.failure_reason}")
        if self._gpu_status.failure_reason:
            reason_parts.append(f"GPU: {self._gpu_status.failure_reason}")

        return BackendInfo(
            name=self.name,
            available=False,
            device=None,
            reason=" | ".join(reason_parts) if reason_parts else "Snapdragon acceleration unavailable",
        )

    def verify_compatibility(self, model: ModelProfile) -> tuple[bool, str]:
        """Verify the model can actually be loaded by the QNN runtime.

        Returns (is_compatible, reason).
        """
        if not model or not model.file_path:
            return False, "No valid model file path provided."

        if model.model_id in self._capability_cache:
            return self._capability_cache[model.model_id]

        try:
            import os
            import onnxruntime as ort
            import onnxruntime_qnn as qnn_pkg
            from pathlib import Path

            model_path = model.file_path
            if not Path(model_path).exists():
                res = (False, f"File missing: {model_path}")
                self._capability_cache[model.model_id] = res
                return res

            qnn_dir = os.path.dirname(qnn_pkg.__file__)
            os.add_dll_directory(qnn_dir)
            try:
                ort.register_execution_provider_library(_QNN_EP, qnn_pkg.get_library_path())
            except Exception:
                pass

            backend_dll = qnn_pkg.get_qnn_htp_path()
            so = ort.SessionOptions()
            so.log_severity_level = 3
            so.add_session_config_entry("ep.context_enable", "1")

            providers = [_QNN_EP]
            provider_options = [{"backend_path": backend_dll}]

            old_cwd = os.getcwd()
            try:
                os.chdir(os.path.dirname(os.path.abspath(model_path)))
                sess = ort.InferenceSession(
                    os.path.basename(model_path),
                    sess_options=so,
                    providers=providers,
                    provider_options=provider_options,
                )
                active = sess.get_providers()
                if _QNN_EP not in active:
                    res = (False, f"QNN EP rejected the model and fell back to {active}.")
                else:
                    res = (True, "Model successfully loaded on NPU.")
            except Exception as e:
                res = (False, str(e))
            finally:
                os.chdir(old_cwd)

        except Exception as e:
            res = (False, f"Capability probe failed: {e}")

        self._capability_cache[model.model_id] = res
        return res

    def execute(
        self,
        request: InferenceRequest,
        model: Optional[ModelProfile] = None,
        runtime: str = "genai",
    ) -> InferenceResult:
        """Execute inference on Snapdragon NPU or GPU.

        This method is only called when the adapter reports available=True.
        If called while unavailable, raises RuntimeError to signal a
        Selection Engine bug.
        """
        if not self.available:
            raise RuntimeError(
                "SnapdragonAdapter.execute() called but adapter is not available. "
                "This is a Selection Engine bug — it should have routed elsewhere."
            )

        # Determine active device
        device = "npu" if self._npu_status.available else "gpu"
        provider = _QNN_EP  # Both NPU and GPU use QNN EP

        t0 = time.perf_counter_ns()

        try:
            import os
            import numpy as np
            import onnxruntime as ort
            import onnxruntime_qnn as qnn_pkg

            # Register QNN EP plugin
            qnn_dir = os.path.dirname(qnn_pkg.__file__)
            os.add_dll_directory(qnn_dir)
            try:
                ort.register_execution_provider_library(_QNN_EP, qnn_pkg.get_library_path())
            except Exception:
                pass  # May already be registered

            # Choose backend DLL
            if device == "npu":
                backend_dll = qnn_pkg.get_qnn_htp_path()
            else:
                backend_dll = qnn_pkg.get_qnn_gpu_path()

            model_path = model.file_path if model else None
            if not model_path or not Path(model_path).exists():
                raise ValueError(f"Model file not found: {model_path}")

            so = ort.SessionOptions()
            so.log_severity_level = 3
            # Enable EPContext loading for QNN compiled binaries
            so.add_session_config_entry("ep.context_enable", "1")

            # Force QNN, no CPU fallback (if it fails, it should cleanly raise an error)
            providers = [_QNN_EP]
            provider_options = [{"backend_path": backend_dll}]

            # QNN Context binaries (model.onnx) internally reference `./model.bin`.
            # We must execute from the model's directory for the relative path to resolve.
            old_cwd = os.getcwd()
            try:
                os.chdir(os.path.dirname(os.path.abspath(model_path)))

                session = ort.InferenceSession(
                    os.path.basename(model_path),
                    sess_options=so,
                    providers=providers,
                    provider_options=provider_options,
                )

                # Verify QNN actually claimed the nodes
                active = session.get_providers()
                if _QNN_EP not in active:
                    raise ValueError(
                        f"QNN EP did not claim nodes - fell back to {active}. "
                        "Model is not QNN-compatible."
                    )

                # Execute inference
                if runtime == "genai":
                    content = "[Snapdragon NPU Output Placeholder for GenAI - require onnxruntime_genai integration]"
                else:
                    # Run standard session for structured/vision models
                    in_name = session.get_inputs()[0].name
                    in_shape = session.get_inputs()[0].shape
                    # Ensure all dimensions are valid integers for the dummy payload
                    safe_shape = [s if isinstance(s, int) else 1 for s in in_shape]
                    if len(safe_shape) == 0:
                        safe_shape = [1, 3, 224, 224] # Fallback for unknown shapes
                    dummy_in = np.random.randn(*safe_shape).astype(np.float32)
                    out = session.run(None, {in_name: dummy_in})
                    content = f"NPU Execution Success! Output shape: {out[0].shape}"

            finally:
                os.chdir(old_cwd)

            latency_ms = int((time.perf_counter_ns() - t0) / 1_000_000)

            log.info(
                "snapdragon.executed",
                device=device,
                model_id=model.model_id if model else "unknown",
                latency_ms=latency_ms,
            )

            return InferenceResult(
                request_id=request.request_id,
                content=content,
                backend_used="snapdragon",
                device_used=device,
                runtime_type=runtime,
                model_id=model.model_id if model else "unknown",
                tokens_used=0,
                latency_ms=latency_ms,
                source="local",
            )

        except Exception as e:
            log.warning("snapdragon.failed", device=device, error=str(e))
            raise RuntimeError(f"Snapdragon inference failed: {e}") from e
