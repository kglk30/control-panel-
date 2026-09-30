import sys
import os
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep
import platform

print(f"Python: {sys.version.split()[0]} {platform.machine()}")
print("ORT Version:", ort.__version__)
print("QNN EP Version:", qnn_ep.__version__)

try:
    from onnxruntime_qnn.build_and_package_info import qnn_version
    print("Embedded QNN SDK version:", qnn_version)
except Exception:
    pass

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])
node_def = helper.make_node('Identity', ['X'], ['Y'])
graph_def = helper.make_graph([node_def], 'test-model', [X], [Y])
opset = helper.make_opsetid('ai.onnx', 13)
model_def = helper.make_model(graph_def, producer_name='dummy', opset_imports=[opset])
model_def.ir_version = 8
dummy_path = 'dummy_init.onnx'
onnx.save(model_def, dummy_path)

ep_lib_path = qnn_ep.get_library_path()
lib_registration_name = "QNNExecutionProvider"
ort.register_execution_provider_library(lib_registration_name, ep_lib_path)
print("Registered QNN EP library from:", ep_lib_path)

ep_options = {
    "backend_path": qnn_ep.get_qnn_htp_path(),
    "soc_model": "60"
}

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 0
ort.set_default_logger_severity(0)

print("\n--- ATTEMPTING QNN INITIALIZATION ---")
sys.stdout.flush()

try:
    if hasattr(sess_options, "add_provider_for_devices"):
        print("Using new add_provider_for_devices API...")
        all_ep_devices = ort.get_ep_devices()
        selected_ep_devices = [
            d for d in all_ep_devices
            if d.ep_name == lib_registration_name and getattr(d.device, "type", None) == ort.OrtHardwareDeviceType.NPU
        ]
        if not selected_ep_devices:
            print("No NPU device found, trying classic fallback")
            session = ort.InferenceSession(dummy_path, sess_options=sess_options, providers=['QNNExecutionProvider'], provider_options=[ep_options])
        else:
            sess_options.add_provider_for_devices(selected_ep_devices, ep_options)
            session = ort.InferenceSession(dummy_path, sess_options=sess_options)
    else:
        print("Using classic providers list API...")
        session = ort.InferenceSession(dummy_path, sess_options=sess_options, providers=['QNNExecutionProvider'], provider_options=[ep_options])
        
    print("\nINITIALIZATION SUCCESS!")
except BaseException as e:
    print(f"\nINITIALIZATION FAILED with Python Exception: {e}")
sys.stdout.flush()
