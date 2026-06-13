# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""Minimal HF Jobs sanity check: confirm GPU is visible and report the env."""
import subprocess
import torch

print("=== HF Jobs GPU sanity check ===")
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
print("--- nvidia-smi ---")
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"])
print("=== OK ===")
