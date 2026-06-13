#!/bin/bash
# Launch one HF Jobs eval per model (each runs all precisions x 6 capabilities,
# incl. GPTQ + HumanEval, and uploads raw results to the dataset repo).
#
# Usage:
#   bash hf_jobs/launch_all.sh open     # ungated models (no Meta/Google approval needed)
#   bash hf_jobs/launch_all.sh gated    # Llama + Gemma (need accepted licenses)
#   bash hf_jobs/launch_all.sh all      # everything
#
# Requires HF_TOKEN in .env. Jobs run detached; this prints their job IDs.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
set -a; source "$ROOT/.env" 2>/dev/null; set +a
HF="$ROOT/.venv311/Scripts/hf.exe"

FLAVOR="${FLAVOR:-l4x1}"
TIMEOUT="${TIMEOUT:-6h}"   # HF Jobs bills actual minutes, not the cap; generous ceiling = free insurance vs slow HellaSwag
REPO="${REPO:-Emil-7/llm-quant-degradation}"
PRECISIONS="${PRECISIONS:-fp16,int4_bnb}"

OPEN=(
  "Qwen/Qwen2.5-3B-Instruct"
  "microsoft/Phi-3.5-mini-instruct"
  "HuggingFaceTB/SmolLM2-1.7B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
)
GATED=(
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "google/gemma-2-2b-it"
)

case "${1:-open}" in
  open)  MODELS=("${OPEN[@]}") ;;
  gated) MODELS=("${GATED[@]}") ;;
  all)   MODELS=("${OPEN[@]}" "${GATED[@]}") ;;
  *) echo "usage: launch_all.sh [open|gated|all]"; exit 1 ;;
esac

echo "Flavor=$FLAVOR timeout=$TIMEOUT repo=$REPO precisions=$PRECISIONS"
for m in "${MODELS[@]}"; do
  echo "--- launching $m ---"
  "$HF" jobs uv run --flavor "$FLAVOR" --timeout "$TIMEOUT" \
    --secrets HF_TOKEN="$HF_TOKEN" -d \
    "$SCRIPT_DIR/run_eval_job.py" \
    --model "$m" --precisions "$PRECISIONS" --results_repo "$REPO"
done
echo "All $1 jobs submitted. Monitor with: hf jobs ps"
