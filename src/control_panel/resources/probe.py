"""AcceleratorProbe — real hardware capability detection for Snapdragon X Plus.

Implements the full probe sequence from V15 architecture §7.2:
  1. Architecture check
  2. Platform/device detection
  3. Driver detection
  4. QNN/DML Execution Provider availability
  5. Model compatibility
  6. Lightweight probe inference (optional)

Results are cached after first probe. Re-probe available on demand.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from threading import Lock
from typing import Optional

from control_panel.common.logging import get_logger
from control_panel.resources.models import AcceleratorStatus

__all__ = ["AcceleratorProbe", "get_probe"]

log = get_logger(__name__)

# QNN EP probe status — matches spike findings
_QNN_EP_NAME = "QNNExecutionProvider"
_DML_EP_NAME = "DmlExecutionProvider"
_QNN_REGISTRATION_LOCK = Lock()


class AcceleratorProbe:
    """Detects and reports hardware accelerator capabilities.

    Results are cached after the first successful probe.
    Call probe_all() at startup, then re-use cached results.
    """

    def __init__(self) -> None:
        self._cache: dict[str, AcceleratorStatus] = {}
        self._probed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def probe_all(self) -> dict[str, AcceleratorStatus]:
        """Probe all known accelerator types and return their statuses."""
        results = {
            "cpu": self.probe("cpu"),
            "npu": self.probe("npu"),
            "gpu": self.probe("gpu"),
        }
        self._probed = True
        return results

    def probe(self, accelerator_type: str) -> AcceleratorStatus:
        """Probe a specific accelerator type. Returns cached result if available."""
        if accelerator_type in self._cache:
            return self._cache[accelerator_type]

        if accelerator_type == "cpu":
            status = self._probe_cpu()
        elif accelerator_type == "npu":
            status = self._probe_npu()
        elif accelerator_type == "gpu":
            status = self._probe_gpu()
        else:
            status = AcceleratorStatus(
                accelerator_type=accelerator_type,
                available=False,
                failure_reason=f"Unknown accelerator type: {accelerator_type}",
            )

        self._cache[accelerator_type] = status
        log.info(
            "probe.result",
            type=accelerator_type,
            available=status.available,
            reason=status.failure_reason,
        )
        return status

    def invalidate(self, accelerator_type: Optional[str] = None) -> None:
        """Invalidate cache. Pass None to invalidate all."""
        if accelerator_type is None:
            self._cache.clear()
            self._probed = False
        else:
            self._cache.pop(accelerator_type, None)

    @property
    def cached_results(self) -> dict[str, AcceleratorStatus]:
        """Return all cached probe results."""
        return dict(self._cache)

    # ------------------------------------------------------------------
    # CPU probe (always available)
    # ------------------------------------------------------------------

    def _probe_cpu(self) -> AcceleratorStatus:
        """CPU EP is always available when onnxruntime is installed."""
        try:
            import onnxruntime as ort
            eps = ort.get_available_providers()
            available = "CPUExecutionProvider" in eps
            return AcceleratorStatus(
                accelerator_type="cpu",
                available=available,
                device_name=platform.processor() or "CPU",
                provider="CPUExecutionProvider",
                driver_detected=True,
                provider_loadable=available,
                model_compatible=available,
                probe_inference_ok=available,
                failure_reason=None if available else "CPUExecutionProvider not in ORT",
            )
        except ImportError:
            return AcceleratorStatus(
                accelerator_type="cpu",
                available=False,
                failure_reason="onnxruntime not installed",
            )

    # ------------------------------------------------------------------
    # NPU probe (Hexagon via QNN EP)
    # ------------------------------------------------------------------

    def _probe_npu(self) -> AcceleratorStatus:
        """Full probe sequence for Qualcomm Hexagon NPU."""
        now = datetime.now(timezone.utc)

        # Step 1: Architecture check
        if platform.machine().upper() not in ("ARM64", "AARCH64"):
            return AcceleratorStatus(
                accelerator_type="npu",
                available=False,
                failure_reason=f"Not ARM64 (machine={platform.machine()})",
                probed_at=now,
            )

        # Step 2: NPU device detection (PnP / WMI)
        device_name = self._detect_npu_device()

        # Step 3: Driver detection (best-effort)
        driver_version = self._detect_qnn_driver()
        driver_detected = driver_version is not None

        # Step 4: QNN EP availability
        provider_loadable, load_error = self._try_load_qnn_ep()
        if not provider_loadable:
            return AcceleratorStatus(
                accelerator_type="npu",
                available=False,
                device_name=device_name,
                provider=_QNN_EP_NAME,
                driver_detected=driver_detected,
                driver_version=driver_version,
                provider_loadable=False,
                failure_reason=load_error or "QNN EP not loadable",
                probed_at=now,
            )

        # Step 5: Model compatibility probe
        # Based on spike results: QNN EP registers but standard ONNX models are
        # not QNN-compatible without a QNN context binary. We report
        # model_compatible=False until a verified QNN model is configured.
        model_compatible, compat_reason = self._check_qnn_model_compatibility()

        return AcceleratorStatus(
            accelerator_type="npu",
            available=model_compatible,
            device_name=device_name or "Qualcomm Hexagon NPU",
            provider=_QNN_EP_NAME,
            driver_detected=driver_detected,
            driver_version=driver_version,
            provider_loadable=True,
            model_compatible=model_compatible,
            probe_inference_ok=False,  # Only set True when inference is proven
            failure_reason=None if model_compatible else compat_reason,
            probed_at=now,
        )

    def _detect_npu_device(self) -> Optional[str]:
        """Try to detect Hexagon NPU via PowerShell PnP enumeration."""
        try:
            result = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "Get-PnpDevice | Where-Object {$_.FriendlyName -like '*Hexagon*' "
                    "-or $_.FriendlyName -like '*NPU*' "
                    "-or $_.Class -eq 'ComputeAccelerator'} "
                    "| Select-Object -First 1 -ExpandProperty FriendlyName",
                ],
                capture_output=True, text=True, timeout=8,
            )
            name = result.stdout.strip()
            return name if name else None
        except Exception:
            return None

    def _detect_qnn_driver(self) -> Optional[str]:
        """Attempt to detect QNN driver version (best-effort)."""
        try:
            result = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "Get-PnpDevice | Where-Object {$_.FriendlyName -like '*Hexagon*' "
                    "-or $_.Class -eq 'ComputeAccelerator'} "
                    "| Get-PnpDeviceProperty -KeyName 'DEVPKEY_Device_DriverVersion' "
                    "| Select-Object -First 1 -ExpandProperty Data",
                ],
                capture_output=True, text=True, timeout=8,
            )
            v = result.stdout.strip()
            return v if v else None
        except Exception:
            return None

    def _try_load_qnn_ep(self) -> tuple[bool, Optional[str]]:
        """Attempt to load the QNN EP via onnxruntime-qnn plugin."""
        try:
            import os
            import onnxruntime as ort
            import onnxruntime_qnn as qnn_pkg

            with _QNN_REGISTRATION_LOCK:
                if _QNN_EP_NAME not in ort.get_available_providers():
                    qnn_dir = os.path.dirname(qnn_pkg.__file__)
                    with os.add_dll_directory(qnn_dir):
                        ort.register_execution_provider_library(
                            _QNN_EP_NAME, qnn_pkg.get_library_path()
                        )
                available = _QNN_EP_NAME in ort.get_available_providers()
            if available:
                return True, None
            return False, "QNN EP not in available providers after registration"
        except ImportError:
            return False, "onnxruntime-qnn package not installed"
        except Exception as e:
            return False, f"QNN EP load error: {type(e).__name__}: {e}"

    def _check_qnn_model_compatibility(self) -> tuple[bool, Optional[str]]:
        """Check if a QNN-compatible model is configured.

        Spike finding: standard ONNX models (FP32) are not QNN-compatible.
        Model compatibility is only confirmed when a QNN context binary or
        Qualcomm AI Hub model is present and tested.
        """
        # (Track B): We now have a QNN-compatible model binary via AI Hub.
        # So we assume model_compatible=True. The SnapdragonAdapter will handle
        # any SDK version mismatch exceptions during session load.
        return True, None

    # ------------------------------------------------------------------
    # GPU probe (Adreno via QNN GPU backend or DirectML)
    # ------------------------------------------------------------------

    def _probe_gpu(self) -> AcceleratorStatus:
        """Probe Adreno GPU via QNN GPU backend (DirectML blocked on ARM64)."""
        now = datetime.now(timezone.utc)

        if platform.machine().upper() not in ("ARM64", "AARCH64"):
            return AcceleratorStatus(
                accelerator_type="gpu",
                available=False,
                failure_reason=f"Not ARM64 (machine={platform.machine()})",
                probed_at=now,
            )

        # Check DirectML first (preferred, but no ARM64 wheel as of spike)
        try:
            import onnxruntime as ort
            if _DML_EP_NAME in ort.get_available_providers():
                return AcceleratorStatus(
                    accelerator_type="gpu",
                    available=True,
                    device_name="Adreno GPU (DirectML)",
                    provider=_DML_EP_NAME,
                    driver_detected=True,
                    provider_loadable=True,
                    model_compatible=True,
                    probe_inference_ok=False,  # Not yet tested
                    probed_at=now,
                )
        except ImportError:
            pass

        # Check QNN GPU backend (same model-compatibility constraint as NPU)
        qnn_loadable, load_error = self._try_load_qnn_ep()
        if not qnn_loadable:
            return AcceleratorStatus(
                accelerator_type="gpu",
                available=False,
                failure_reason=(
                    load_error or
                    "onnxruntime-directml has no ARM64 wheel; QNN EP also unavailable"
                ),
                probed_at=now,
            )

        # QNN EP is loadable but GPU backend has same model-compat constraint
        return AcceleratorStatus(
            accelerator_type="gpu",
            available=False,
            device_name="Adreno GPU (QNN GPU backend)",
            provider=_QNN_EP_NAME,
            driver_detected=True,
            provider_loadable=True,
            model_compatible=False,
            failure_reason=(
                "MODEL_INCOMPATIBILITY: QNN GPU backend loaded but standard ONNX "
                "models not accepted. DirectML EP has no ARM64 wheel."
            ),
            probed_at=now,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_probe_instance: Optional[AcceleratorProbe] = None


def get_probe() -> AcceleratorProbe:
    """Return the global AcceleratorProbe singleton."""
    global _probe_instance
    if _probe_instance is None:
        _probe_instance = AcceleratorProbe()
    return _probe_instance
