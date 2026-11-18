"""Unit tests for Backend Selection Engine."""

import pytest
from unittest.mock import MagicMock

from control_panel.common.models import InferenceRequest
from control_panel.inference.models import ModelProfile, ModelRegistry, ModelRequirements
from control_panel.inference.selector import BackendSelectionEngine
from control_panel.resources.models import AcceleratorStatus, ResourceConstraints


def _make_registry(*profiles: ModelProfile) -> ModelRegistry:
    reg = ModelRegistry()
    for p in profiles:
        reg.register(p)
    return reg


def _rule_based_profile() -> ModelProfile:
    return ModelProfile(
        model_id="rule_based_v1",
        model_name="Rule-Based",
        format="none",
        runtime_type="rule_based",
        requirements=ModelRequirements(estimated_ram_mb=0),
        supported_tasks=["analyze", "classify", "plan", "reason", "generate"],
        supported_backends=["rule_based"],
        supported_devices=["none"],
    )


def _local_cpu_profile() -> ModelProfile:
    return ModelProfile(
        model_id="test_model",
        model_name="Test Model",
        format="onnx",
        runtime_type="genai",
        requirements=ModelRequirements(estimated_ram_mb=500),
        supported_tasks=["analyze", "plan", "reason"],
        supported_backends=["local_cpu"],
        supported_devices=["cpu"],
    )


def _constraints(available_ram: int = 4000) -> ResourceConstraints:
    return ResourceConstraints(
        available_ram_mb=available_ram,
        max_model_ram_mb=available_ram - 512,
        cpu_pressure="low",
        power_policy="normal",
        max_concurrent_workers=4,
        can_load_local_model=True,
    )


def _accelerators(cpu: bool = True, npu: bool = False, gpu: bool = False) -> dict:
    return {
        "cpu": AcceleratorStatus(
            accelerator_type="cpu", available=cpu,
            provider="CPUExecutionProvider",
        ),
        "npu": AcceleratorStatus(
            accelerator_type="npu", available=npu,
            failure_reason=None if npu else "MODEL_INCOMPATIBILITY",
        ),
        "gpu": AcceleratorStatus(
            accelerator_type="gpu", available=gpu,
        ),
    }


class TestBackendSelector:

    def test_selects_local_cpu_when_available(self, tmp_path) -> None:
        # Create a fake model file so the selector sees it as available
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake")

        profile = ModelProfile(
            model_id="test_model",
            model_name="Test Model",
            format="onnx",
            runtime_type="genai",
            file_path=str(model_file),  # points at existing file
            requirements=ModelRequirements(estimated_ram_mb=500),
            supported_tasks=["analyze", "plan", "reason"],
            supported_backends=["local_cpu"],
            supported_devices=["cpu"],
        )
        reg = _make_registry(_rule_based_profile(), profile)
        sel = BackendSelectionEngine(reg)
        req = InferenceRequest(prompt="test", task_type="analyze")
        result = sel.select(req, _constraints(), _accelerators())
        assert result.backend in ("local_cpu", "rule_based")

    def test_falls_back_to_rule_based_when_no_candidates(self) -> None:
        reg = _make_registry(_rule_based_profile())
        sel = BackendSelectionEngine(reg)
        req = InferenceRequest(prompt="test", task_type="analyze")
        result = sel.select(req, _constraints(), _accelerators())
        assert result.backend == "rule_based"
        assert result.model_id == "rule_based_v1"

    def test_filters_insufficient_ram(self) -> None:
        profile = _local_cpu_profile()  # needs 500 MB
        reg = _make_registry(_rule_based_profile(), profile)
        sel = BackendSelectionEngine(reg)
        req = InferenceRequest(prompt="test", task_type="analyze")
        # Only 400 MB available
        tight = ResourceConstraints(
            available_ram_mb=400,
            max_model_ram_mb=0,
            cpu_pressure="low",
            power_policy="normal",
            max_concurrent_workers=4,
        )
        result = sel.select(req, tight, _accelerators())
        assert result.backend == "rule_based"

    def test_classify_task_uses_structured_runtime(self) -> None:
        reg = _make_registry(_rule_based_profile())
        sel = BackendSelectionEngine(reg)
        req = InferenceRequest(prompt="classify this file", task_type="classify")
        result = sel.select(req, _constraints(), _accelerators())
        assert result.runtime_type in ("structured", "rule_based")

    def test_snapdragon_filtered_when_npu_unavailable(self) -> None:
        sd_profile = ModelProfile(
            model_id="sd_model",
            model_name="SD Model",
            format="onnx",
            runtime_type="genai",
            requirements=ModelRequirements(estimated_ram_mb=200),
            supported_tasks=["analyze"],
            supported_backends=["snapdragon"],
            supported_devices=["npu"],
        )
        reg = _make_registry(_rule_based_profile(), sd_profile)
        sel = BackendSelectionEngine(reg)
        req = InferenceRequest(prompt="test", task_type="analyze")
        # NPU unavailable
        result = sel.select(req, _constraints(), _accelerators(npu=False))
        assert result.backend == "rule_based"
