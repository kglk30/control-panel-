#!/usr/bin/env python3
"""
Compile a model for Qualcomm QNN HTP (Hexagon NPU) using Qualcomm AI Hub.

Prerequisites:
    pip install qai-hub qai-hub-models
    qai-hub configure --api_token YOUR_TOKEN   # free account at aihub.qualcomm.com

Usage:
    uv run python scripts/compile_qnn_model.py --list-models
    uv run python scripts/compile_qnn_model.py --model Phi3Mini --device "Snapdragon X Plus"
    uv run python scripts/compile_qnn_model.py --model WhisperTinyEn --device "Snapdragon X Plus"

This script:
1. Checks for qai-hub and qai-hub-models installation
2. Submits a compile job to Qualcomm AI Hub
3. Downloads the resulting QNN context binary (.bin)
4. Places it in models/qnn/
5. Prints the SnapdragonAdapter configuration update needed

Supported models (from qai-hub-models):
  - Phi3Mini
  - WhisperTinyEn / WhisperBaseEn
  - MobileNetV2
  - BERTBase
  - ResNet50

Reference:
  https://aihub.qualcomm.com
  https://github.com/quic/ai-hub-models
  https://app.aihub.qualcomm.com/docs/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

QNN_MODELS = {
    "Phi3Mini": {
        "hub_model": "Phi3Mini",
        "description": "Phi-3 Mini language model — optimized for Hexagon NPU",
        "output_name": "phi3_mini_qnn",
    },
    "WhisperTinyEn": {
        "hub_model": "WhisperTinyEn",
        "description": "Whisper Tiny English — speech recognition on NPU",
        "output_name": "whisper_tiny_en_qnn",
    },
    "WhisperBaseEn": {
        "hub_model": "WhisperBaseEn",
        "description": "Whisper Base English — higher quality speech recognition",
        "output_name": "whisper_base_en_qnn",
    },
    "MobileNetV2": {
        "hub_model": "MobileNetV2",
        "description": "MobileNet V2 — image classification, good QNN test model",
        "output_name": "mobilenet_v2_qnn",
    },
}


def check_qai_hub() -> bool:
    try:
        import qai_hub
        return True
    except ImportError:
        return False


def check_qai_hub_models() -> bool:
    try:
        import qai_hub_models
        return True
    except ImportError:
        return False


def compile_model(model_key: str, device: str, output_dir: Path) -> None:
    if not check_qai_hub():
        print("\nERROR: qai-hub not installed.")
        print("Install it:")
        print("  uv run pip install qai-hub")
        print("  qai-hub configure --api_token YOUR_TOKEN")
        print("  (Get a free token at https://aihub.qualcomm.com)")
        sys.exit(1)

    if not check_qai_hub_models():
        print("\nERROR: qai-hub-models not installed.")
        print("Install it:")
        print("  uv run pip install qai-hub-models")
        sys.exit(1)

    import qai_hub
    import importlib

    info = QNN_MODELS[model_key]
    print(f"\nCompiling: {info['description']}")
    print(f"  Target device: {device}")
    print(f"  Output: {output_dir / info['output_name']}")

    # Dynamically import the model class from qai_hub_models
    try:
        model_module = importlib.import_module(
            f"qai_hub_models.models.{info['hub_model'].lower()}.model"
        )
        ModelClass = getattr(model_module, info['hub_model'])
        model_instance = ModelClass.from_pretrained()
    except Exception as e:
        print(f"\nERROR loading model: {e}")
        print(f"Try: uv run python -c \"from qai_hub_models.models import {info['hub_model']}\"")
        sys.exit(1)

    # Export to ONNX (needed for compile job)
    print("  Exporting to ONNX...")
    try:
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp())
        onnx_path = tmp_dir / "model.onnx"
        model_instance.export(str(onnx_path))
    except Exception as e:
        print(f"\nERROR during export: {e}")
        print("Attempting direct compile job submission...")

    # Submit compile job to Qualcomm AI Hub
    print(f"  Submitting compile job for {device}...")
    try:
        compile_job = qai_hub.submit_compile_job(
            model=model_instance,
            device=qai_hub.Device(device),
            name=f"{info['output_name']}_compile",
            options="--target_runtime qnn_context_binary",
        )

        print(f"  Job submitted: {compile_job.job_id}")
        print("  Waiting for completion (may take 5-15 minutes)...")
        compile_job.wait()

        if compile_job.get_status().success:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"{info['output_name']}.bin"
            compile_job.download_target_model(str(out_path))
            print(f"\n✓ QNN context binary downloaded: {out_path}")
            _print_snapdragon_config(info['output_name'], str(out_path))
        else:
            print(f"\nERROR: Compile job failed: {compile_job.get_status()}")
            sys.exit(1)

    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nManual alternative:")
        print("  1. Go to https://aihub.qualcomm.com")
        print("  2. Upload your ONNX model")
        print("  3. Select 'Snapdragon X Plus' as target device")
        print("  4. Compile with target_runtime = qnn_context_binary")
        print("  5. Download the .bin file")
        print("  6. Place it in models/qnn/")
        sys.exit(1)


def _print_snapdragon_config(name: str, bin_path: str) -> None:
    print("\n--- ADD TO config/models.toml ---")
    print(f"""
[[models]]
model_id = "{name}"
model_name = "{name} (QNN context binary)"
format = "qnn_context_binary"
runtime_type = "qnn"
quantization = "int8"
file_path = "{bin_path}"

[models.requirements]
disk_size_mb = 200
estimated_ram_mb = 500
minimum_free_ram_mb = 256

supported_tasks = ["analyze", "plan", "reason", "generate"]
supported_backends = ["snapdragon"]
supported_devices = ["npu"]
context_length = 4096
structured_output_support = false
source = "qualcomm_ai_hub"
version = "1.0"
""")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile a model for Qualcomm NPU via AI Hub"
    )
    parser.add_argument("--model", default="MobileNetV2",
                        choices=list(QNN_MODELS.keys()),
                        help="Model to compile (start with MobileNetV2 to test the pipeline)")
    parser.add_argument("--device", default="Snapdragon X Plus",
                        help="Target device name on AI Hub")
    parser.add_argument("--output-dir", default="models/qnn",
                        help="Output directory for compiled model")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models")
    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable models for QNN compilation:")
        for key, info in QNN_MODELS.items():
            print(f"  {key:20s} {info['description']}")
        print("\nTip: Start with MobileNetV2 to verify the compile pipeline works.")
        return

    output_dir = Path(__file__).parent.parent / args.output_dir
    compile_model(args.model, args.device, output_dir)


if __name__ == "__main__":
    main()
