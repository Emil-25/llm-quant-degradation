#!/bin/bash
# Environment setup for the quantization-degradation study.
# Works on Colab (primary) and local Linux/WSL. Run once per fresh session.
#
#   bash setup_env.sh
#
# After setup, authenticate to HuggingFace for gated models (Llama, Gemma):
#   export HF_TOKEN="your_token_here"      # or: huggingface-cli login

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Installing Python dependencies ==="
pip install --quiet --upgrade pip
pip install --quiet -r "${SCRIPT_DIR}/requirements.txt"

# HumanEval's code-execution harness ships separately from lm-eval.
echo "=== Installing human-eval (HumanEval execution backend) ==="
pip install --quiet human-eval || \
    echo "WARNING: human-eval install failed; HumanEval runs will not work until it is installed."

echo "=== Verifying key imports ==="
python - <<'PY'
import importlib
mods = ["torch", "transformers", "accelerate", "lm_eval", "datasets",
        "scipy", "pandas", "numpy", "matplotlib", "seaborn", "yaml"]
ok, missing = [], []
for m in mods:
    try:
        importlib.import_module(m)
        ok.append(m)
    except Exception as e:
        missing.append(f"{m} ({e.__class__.__name__})")
print("OK     :", ", ".join(ok))
if missing:
    print("MISSING:", ", ".join(missing))
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# At least one INT4 backend must import for quantization to work.
echo "=== Checking quantization backends ==="
python - <<'PY'
def can(m):
    try:
        __import__(m); return True
    except Exception:
        return False
gptq = can("gptqmodel") or can("auto_gptq")
awq  = can("awq")
print("GPTQ backend available:", gptq, "(gptqmodel preferred)")
print("AWQ  backend available:", awq)
if not gptq:
    print("WARNING: no GPTQ backend imported. INT4-GPTQ quantization will fail.")
PY

echo ""
echo "=== Setup complete ==="
echo "Next:"
echo "  export HF_TOKEN=...   # for gated models"
echo "  bash scripts/run_all_evals.sh phase1"
