#!/usr/bin/env python3
"""
Download Phi-3 Mini 4K Instruct INT4 ONNX model for local CPU inference.

Usage:
    uv run python scripts/download_model.py
    uv run python scripts/download_model.py --model phi2
    uv run python scripts/download_model.py --list
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MODELS = {
    "phi3-mini": {
        "repo_id": "microsoft/Phi-3-mini-4k-instruct-onnx",
        "allow_patterns": ["cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/**"],
        "local_dir": "models/phi3-mini-int4",
        "model_path": "models/phi3-mini-int4/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4",
        "size_gb": 2.3,
        "description": "Phi-3 Mini 4K Instruct — INT4 quantized, ~2.3 GB, recommended",
        "runtime_type": "genai",
        "model_id_for_toml": "phi3_mini_int4_cpu",
    },
    "phi3-mini-128k": {
        "repo_id": "microsoft/Phi-3-mini-128k-instruct-onnx",
        "allow_patterns": ["cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/**"],
        "local_dir": "models/phi3-mini-128k-int4",
        "model_path": "models/phi3-mini-128k-int4/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4",
        "size_gb": 2.3,
        "description": "Phi-3 Mini 128K context — INT4, same size but longer context",
        "runtime_type": "genai",
        "model_id_for_toml": "phi3_mini_128k_int4_cpu",
    },
    "phi2": {
        "repo_id": "microsoft/phi-2-onnx",
        "allow_patterns": ["**"],
        "local_dir": "models/phi2",
        "model_path": "models/phi2",
        "size_gb": 1.5,
        "description": "Phi-2 — older, smaller, ~1.5 GB",
        "runtime_type": "genai",
        "model_id_for_toml": "phi2_cpu",
    },
}


def check_free_space(path: Path, required_gb: float) -> bool:
    free = shutil.disk_usage(str(path)).free / 1e9
    print(f"  Disk free: {free:.1f} GB, required: {required_gb:.1f} GB")
    return free >= required_gb + 1.0  # 1 GB buffer


def download_model(model_key: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface-hub not installed. Run: uv add huggingface-hub")
        sys.exit(1)

    info = MODELS[model_key]
    print(f"\nDownloading: {info['description']}")
    print(f"  Source: {info['repo_id']}")
    print(f"  Destination: {info['local_dir']}")

    root = Path(__file__).parent.parent
    if not check_free_space(root, info["size_gb"]):
        print("ERROR: Not enough disk space!")
        sys.exit(1)

    local_dir = root / info["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)

    print("  Downloading... (this may take several minutes)")
    try:
        path = snapshot_download(
            repo_id=info["repo_id"],
            allow_patterns=info["allow_patterns"],
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        print(f"\n✓ Downloaded to: {path}")
    except Exception as e:
        print(f"\nERROR: Download failed: {e}")
        print("\nTry manually:")
        print(f"  huggingface-cli download {info['repo_id']} --local-dir {local_dir}")
        sys.exit(1)

    # Find the actual model directory
    model_dir = root / info["model_path"]
    if not model_dir.exists():
        # Try to find it
        candidates = list(local_dir.rglob("genai_config.json"))
        if candidates:
            model_dir = candidates[0].parent
        else:
            candidates = list(local_dir.rglob("*.onnx"))
            if candidates:
                model_dir = candidates[0].parent

    print(f"\nModel directory: {model_dir}")
    print("\n--- UPDATE config/models.toml ---")
    print(f"""
[[models]]
model_id = "{info['model_id_for_toml']}"
model_name = "{info['description']}"
format = "onnx"
runtime_type = "{info['runtime_type']}"
quantization = "int4"
file_path = "{model_dir}"

[models.requirements]
disk_size_mb = {int(info['size_gb'] * 1024)}
estimated_ram_mb = {int(info['size_gb'] * 1200)}
minimum_free_ram_mb = 512

supported_tasks = ["analyze", "plan", "reason", "generate"]
supported_backends = ["local_cpu"]
supported_devices = ["cpu"]
context_length = 4096
structured_output_support = false
source = "huggingface"
version = "3.8"
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ONNX models for Control Panel")
    parser.add_argument("--model", default="phi3-mini", choices=list(MODELS.keys()),
                        help="Model to download (default: phi3-mini)")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable models:")
        for key, info in MODELS.items():
            print(f"  {key:20s} {info['description']}")
        return

    download_model(args.model)


if __name__ == "__main__":
    main()
