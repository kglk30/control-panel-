# QNN Setup Guide — Hexagon NPU Inference on Snapdragon X Plus

This guide explains how to run inference on the **Qualcomm Hexagon NPU** using the QNN Execution Provider in ONNX Runtime.

---

## Why Standard ONNX Doesn't Work on the NPU

The QNN HTP (Hexagon Tensor Processor) does **not** accept standard FP32 or INT8 ONNX models directly. It requires a **QNN context binary** (`.bin`) — a model that has been:

1. Quantized to INT8 (using Qualcomm-specific quantization)
2. Compiled and optimized for the HTP architecture
3. Exported as a proprietary binary format

This is why during the inference spike, QNN EP registered successfully but placed all nodes on CPU EP — the model was standard FP32, not HTP-compatible.

---

## Method 1: Qualcomm AI Hub (Recommended, Free)

AI Hub is Qualcomm's cloud service for compiling models to run on Snapdragon hardware.

### Step 1 — Create a free account

Go to: https://aihub.qualcomm.com and sign up.

### Step 2 — Install SDK

```powershell
uv run pip install qai-hub qai-hub-models
qai-hub configure --api_token YOUR_TOKEN
```

### Step 3 — List available devices

```python
import qai_hub
devices = qai_hub.get_devices()
for d in devices:
    print(d.name)
# Look for: "Snapdragon X Plus", "Snapdragon X Elite"
```

### Step 4 — Compile a model

```powershell
# Start with a small model to verify the pipeline
uv run python scripts/compile_qnn_model.py --model MobileNetV2 --device "Snapdragon X Plus"

# For language models:
uv run python scripts/compile_qnn_model.py --model Phi3Mini --device "Snapdragon X Plus"
```

The script submits a compile job, waits for completion (5–20 min), and downloads the `.bin`.

### Step 5 — Use the compiled model

```powershell
# Add the .bin path to config/models.toml (script prints the exact snippet)
# Then verify:
uv run control-panel backends  # Should show snapdragon as ✓
uv run control-panel run "organize my files" --path "C:\path" --verbose
```

---

## Method 2: Download Pre-compiled Models

Qualcomm provides pre-compiled models via AI Hub:

```python
import qai_hub

# List pre-compiled jobs (if you've compiled before, they're cached)
jobs = qai_hub.get_jobs()
for j in jobs:
    print(j.name, j.status)
```

Or use the web UI at https://app.aihub.qualcomm.com/jobs/ to browse and download.

---

## Method 3: Qualcomm Neural Processing SDK (Local)

For offline compilation (no cloud required):

1. Download **Qualcomm Neural Processing SDK** (AI Stack) from: https://qpm.qualcomm.com/
2. Install on Windows — includes `qnn-net-run`, `qnn-onnx-converter`
3. Convert an ONNX model:

```bash
qnn-onnx-converter \
  --input_network model.onnx \
  --output_path model_qnn \
  --float_fallback

qnn-context-binary-generator \
  --model model_qnn.cpp \
  --backend libQnnHtp.so \
  --output_dir context_bin/
```

---

## Using the QNN Context Binary in Control Panel

Once you have a `.bin` file:

### 1. Add to `config/models.toml`

```toml
[[models]]
model_id = "my_qnn_model"
model_name = "My QNN Model (NPU)"
format = "qnn_context_binary"
runtime_type = "qnn"
quantization = "int8"
file_path = "models/qnn/my_model.bin"

[models.requirements]
disk_size_mb = 200
estimated_ram_mb = 400
minimum_free_ram_mb = 256

supported_tasks = ["analyze", "classify"]
supported_backends = ["snapdragon"]
supported_devices = ["npu"]
context_length = 512
structured_output_support = false
source = "qualcomm_ai_hub"
version = "1.0"
```

### 2. Verify the SnapdragonAdapter picks it up

```powershell
uv run control-panel backends
# Should show: snapdragon ✓  npu
```

### 3. Run with NPU

```powershell
uv run control-panel run "organize my files" --path "C:\path" --backend snapdragon --verbose
```

---

## QNN EP Loading Sequence (Technical Reference)

The `SnapdragonAdapter` in [`inference/snapdragon.py`](../src/control_panel/inference/snapdragon.py) uses this sequence:

```python
import os, onnxruntime_qnn as qnn
os.add_dll_directory(os.path.dirname(qnn.__file__))
import onnxruntime as ort

session = ort.InferenceSession(
    "model.bin",  # QNN context binary
    providers=["QNNExecutionProvider", "CPUExecutionProvider"],
    provider_options=[
        {"backend_path": qnn.get_qnn_htp_path()},  # HTP backend
        {}
    ]
)
```

The `.bin` format signals to QNN EP that this is a pre-compiled context binary, bypassing the normal graph compilation step.

---

## Known Limitations (V0.1)

| Issue | Status |
|-------|--------|
| Standard FP32 ONNX → 0 nodes on NPU | Expected — needs QNN context binary |
| DirectML EP | No `win_arm64` wheel available yet |
| QNN SDK local compilation | Requires Windows SDK install + `qnn-onnx-converter` |
| AI Hub free tier | 100 compile jobs/month, suitable for development |

---

## Current Status

| Path | Status |
|------|--------|
| QNN EP registration | ✅ Works (`onnxruntime-qnn 2.6.0` installed) |
| QNN context binary | ⏳ Need to compile via AI Hub |
| CPU fallback | ✅ Fully working with Oryon CPU |
| NPU verified | 🔬 Pending model compilation |
