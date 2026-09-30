"""
Test script: run the AI-Hub-compiled, QNN/HTP-targeted MobileNetV2 model
locally on the Snapdragon X Plus Hexagon NPU via ONNX Runtime's plugin QNN EP.
"""

import os
os.environ["QNN_LOG_LEVEL"] = "0" # Try to bypass QnnLog_create
os.environ["ADSP_LIBRARY_PATH"] = r"C:\qnn_test"
os.chdir(r"C:\qnn_test")

import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

MODEL_PATH = "model.onnx"

print("onnxruntime-qnn version:", qnn_ep.__version__)

# 1. Register the QNN plugin EP library with ONNX Runtime
ep_lib_path = qnn_ep.get_library_path()
lib_registration_name = "QNNExecutionProvider"
ort.register_execution_provider_library(lib_registration_name, ep_lib_path)
print("Registered QNN EP library from:", ep_lib_path)

# 2. Find the QNN-capable device(s) ORT discovered on this machine
all_ep_devices = ort.get_ep_devices()
print("All discovered EP devices:")
for d in all_ep_devices:
    print("  -", d.ep_name, "|", d.ep_vendor if hasattr(d, "ep_vendor") else "")

selected_ep_devices = [
    d for d in all_ep_devices
    if d.ep_name == lib_registration_name and d.device.type == ort.OrtHardwareDeviceType.NPU
]
if not selected_ep_devices:
    raise RuntimeError("No QNN NPU device found. Is the plugin EP registered correctly?")

print(f"Found {len(selected_ep_devices)} QNN NPU device(s).")
import sys; sys.stdout.flush()

# 3. Configure the session to use HTP (NPU) backend and create it
# NOTE: The model.bin MUST have been compiled with `--os windows` on AI Hub!
# If it was compiled for Linux/Android (the default), loading it here will
# cause a STATUS_STACK_OVERFLOW in QcSoCServiceUtils.dll on Windows ARM64.
ep_options = {
    "backend_path": qnn_ep.get_qnn_htp_path(),
    "soc_model": "69"  # 69 is the QNN soc_model id for Snapdragon X Elite/Plus
}
print("HTP backend path:", ep_options["backend_path"])
sys.stdout.flush()

session_options = ort.SessionOptions()
print("SessionOptions created.")
sys.stdout.flush()

session_options.add_provider_for_devices(selected_ep_devices, ep_options)
print("add_provider_for_devices completed.")
sys.stdout.flush()

session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
print("Disabled CPU EP fallback.")
sys.stdout.flush()

print("About to create InferenceSession with model:", MODEL_PATH)
sys.stdout.flush()

session = ort.InferenceSession(MODEL_PATH, sess_options=session_options)

print("InferenceSession created successfully.")
sys.stdout.flush()
print("Session created. Inputs:", [i.name for i in session.get_inputs()])
print("Outputs:", [o.name for o in session.get_outputs()])

# 4. Run a dummy inference (random input matching the model's expected shape)
input_name = session.get_inputs()[0].name
dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

run_options = ort.RunOptions()
run_options.add_run_config_entry("qnn.perf_mode", "burst")

result = session.run(None, {input_name: dummy_input}, run_options=run_options)

print("Inference succeeded.")
print("Output shape:", result[0].shape)
print("Top-5 class indices:", np.argsort(result[0][0])[-5:][::-1])
