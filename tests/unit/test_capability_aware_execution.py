
import pytest
from unittest.mock import MagicMock
from control_panel.common.models import InferenceRequest, BackendSelection
from control_panel.inference.runtime import InferenceRuntime

def test_capability_compatible_backend_selected_normally():
    rt = InferenceRuntime()
    # Mock selector to pick snapdragon
    rt._selector.select = MagicMock(return_value=BackendSelection(
        backend="snapdragon", device="npu", runtime_type="structured", model_id="mobilenetv2_qnn_htp", reason="selected normally"
    ))
    # Mock registry
    model_mock = MagicMock()
    model_mock.model_id = "mobilenetv2_qnn_htp"
    rt._registry.get = MagicMock(return_value=model_mock)
    # Mock snapdragon verification success
    rt._snapdragon.verify_compatibility = MagicMock(return_value=(True, "Compatible"))
    # Mock dispatch
    rt._dispatch = MagicMock(return_value="success")

    req = InferenceRequest(prompt="test", task_type="analyze", response_schema={})
    res = rt.infer(req)

    rt._snapdragon.verify_compatibility.assert_called_once_with(model_mock)
    # Dispatch should be called with snapdragon
    args = rt._dispatch.call_args[0]
    assert args[1].backend == "snapdragon"


def test_capability_incompatible_model_cpu_fallback():
    rt = InferenceRuntime()
    # Mock selector to pick snapdragon first, then local_cpu on fallback
    def mock_select(*args, **kwargs):
        if kwargs.get("config_policy", {}).get("preferred_backend") == "local_cpu":
            return BackendSelection(
                backend="local_cpu", device="cpu", runtime_type="structured", model_id="fallback_model", reason="fallback selected"
            )
        return BackendSelection(
            backend="snapdragon", device="npu", runtime_type="structured", model_id="mobilenetv2_qnn_htp", reason="initial selection"
        )
    rt._selector.select = MagicMock(side_effect=mock_select)

    model_mock = MagicMock()
    model_mock.model_id = "mobilenetv2_qnn_htp"
    rt._registry.get = MagicMock(return_value=model_mock)
    # Mock verification failure
    rt._snapdragon.verify_compatibility = MagicMock(return_value=(False, "Incompatible SDK version"))
    rt._dispatch = MagicMock(return_value="success")

    req = InferenceRequest(prompt="test", task_type="analyze", response_schema={})
    res = rt.infer(req)

    rt._snapdragon.verify_compatibility.assert_called_once_with(model_mock)
    # Selector should be called twice (initial + fallback)
    assert rt._selector.select.call_count == 2
    # Dispatch should be called with local_cpu
    args = rt._dispatch.call_args[0]
    assert args[1].backend == "local_cpu"
    assert "Incompatible SDK version" in args[1].reason
    assert "Fallback: LocalCPU" in args[1].reason

def test_capability_provider_exists_but_model_incompatible_accelerator_not_used():
    # This is effectively tested by the above test since the fallback backend is local_cpu.
    # Just a dedicated check that NPU is avoided.
    rt = InferenceRuntime()
    # Mock selector
    call_count = [0]
    def mock_select(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return BackendSelection(backend="snapdragon", device="npu", runtime_type="structured", model_id="mobilenet", reason="test")
        else:
            return BackendSelection(backend="local_cpu", device="cpu", runtime_type="structured", model_id="cpu_model", reason="test")
    rt._selector.select = MagicMock(side_effect=mock_select)

    rt._registry.get = MagicMock(return_value=MagicMock())
    rt._snapdragon.verify_compatibility = MagicMock(return_value=(False, "EPContext node incompatible"))

    # Track which backend _dispatch gets
    dispatched = []
    def mock_dispatch(req, sel):
        dispatched.append(sel.backend)
        return "success"
    rt._dispatch = MagicMock(side_effect=mock_dispatch)

    req = InferenceRequest(prompt="test", task_type="analyze", response_schema={})
    rt.infer(req)

    # Should not dispatch to snapdragon
    assert "snapdragon" not in dispatched
    assert "local_cpu" in dispatched
