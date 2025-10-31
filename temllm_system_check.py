#!/usr/bin/env python3
"""
System Environment Info Printer
--------------------------------
Prints:
- Python version
- CPU details
- RAM (total + available)
- Disk (ROM) usage
- GPU name (if CUDA/torch available)
- Torch version
- LLaMA / TinyLlama model version (if installed)
"""

import platform
import os
import sys
import psutil
import torch
import subprocess
import glob
from pathlib import Path


def get_gpu_info():
    """Detect GPU name via torch or nvidia-smi"""
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return gpu_name
        else:
            # Try nvidia-smi fallback
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split("\n")[0]
            return "No GPU detected"
    except Exception:
        return "No GPU or driver not found"

def get_llama_version():
    """
    Detect installed or cached LLaMA / TinyLlama model.
    - Checks local Hugging Face cache for 'llama' or 'tinyllama' folders
    - Falls back to checking if 'transformers' and model weights exist
    - Returns model name/version string or 'Not found'
    """


    try:
        import transformers
        cache_dir = transformers.file_utils.default_cache_path
        # HF_CACHE = ~/.cache/huggingface
        candidates = []

        # Check for any cached model directories containing llama / tinyllama
        search_patterns = [
            os.path.join(cache_dir, "**/*llama*"),
            os.path.join(cache_dir, "**/*TinyLlama*"),
        ]

        for pattern in search_patterns:
            for path in glob.glob(pattern, recursive=True):
                if os.path.isdir(path) and any(x in path.lower() for x in ["llama", "tinyllama"]):
                    candidates.append(path)

        if candidates:
            # Sort by modification time (latest first)
            candidates.sort(key=os.path.getmtime, reverse=True)
            latest = candidates[0]
            return f"Found cached model: {Path(latest).name}"

        # Try to import any known model name
        possible_models = [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "TinyLlama/TinyLlama-1.1B-Chat-dv1.0",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-3-8b-instruct",
        ]
        for m in possible_models:
            try:
                from transformers import AutoModelForCausalLM
                AutoModelForCausalLM.from_pretrained(m, local_files_only=True)
                return f"Detected model: {m}"
            except Exception:
                continue

        return "No LLaMA/TinyLlama model found locally."

    except ImportError:
        return "transformers not installed."

def bytes_to_gb(b):
    return round(b / (1024 ** 3), 2)

def print_sys_info():
    print("=" * 70)
    print("🔍 SYSTEM ENVIRONMENT INFORMATION")
    print("=" * 70)

    # --- Python ---
    print(f"🐍 Python Version  : {platform.python_version()} ({sys.executable})")

    # --- OS ---
    print(f"🖥️  OS             : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"💻 Architecture    : {' '.join(platform.architecture())}")
    print(f"👤 Machine         : {platform.node()}")

    # --- CPU ---
    print(f"🧠 CPU             : {platform.processor() or os.cpu_count()} cores")
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        print(f"⚙️  CPU Name        : {info.get('brand_raw', 'Unknown CPU')}")
    except ImportError:
        print("⚙️  CPU Name        : Install `py-cpuinfo` for detailed info (`pip install py-cpuinfo`)")

    # --- Memory (RAM) ---
    ram = psutil.virtual_memory()
    print(f"🧮 RAM Total       : {bytes_to_gb(ram.total)} GB")
    print(f"   RAM Available   : {bytes_to_gb(ram.available)} GB")

    # --- Disk (ROM) ---
    disk = psutil.disk_usage('/')
    print(f"💾 Disk Total      : {bytes_to_gb(disk.total)} GB")
    print(f"   Disk Free       : {bytes_to_gb(disk.free)} GB")

    # --- GPU ---
    print(f"🎮 GPU             : {get_gpu_info()}")

    # --- Torch ---
    print(f"🔥 Torch Version   : {torch.__version__ if torch else 'Not Installed'}")

    # --- LLaMA / TinyLlama ---
    print(f"🦙 LLaMA Model     : {get_llama_version()}")

    print("=" * 70)
    print("✅ Environment check complete.\n")

if __name__ == "__main__":
    print_sys_info()
