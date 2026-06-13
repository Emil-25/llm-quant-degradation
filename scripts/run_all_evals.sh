#!/bin/bash
# Run all evaluation combinations for a given phase.
# Handles INT4 end-to-end: quantizes each model to a checkpoint (if missing),
# then evaluates FP16/INT8 from the HF repo and INT4 from the quantized dir.
#
# Usage:
#   bash run_all_evals.sh phase1       # Pipeline check: 1 model, 2 benchmarks
#   bash run_all_evals.sh phase2       # Signal check: 2 models, 3 benchmarks
#   bash run_all_evals.sh phase3       # Full paper: broadened small-model set, 6 benchmarks
#   bash run_all_evals.sh phase3_open  # Full set, ungated models only (no HF_TOKEN)
#
# Set HF_TOKEN before running (gated models: Llama, Gemma):
#   export HF_TOKEN="your_token_here"

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/results/raw"
QUANT_DIR="${ROOT_DIR}/results/quantized"
mkdir -p "$OUTPUT_DIR" "$QUANT_DIR"

PHASE=${1:-phase1}

echo "========================================="
echo "Running evaluations: $PHASE"
echo "Output: $OUTPUT_DIR"
echo "========================================="

# Map an HF repo to a local quantized-checkpoint dir name.
quantized_path() {
    local model=$1
    local method=$2   # gptq | awq
    local short
    short=$(echo "$model" | awk -F/ '{print tolower($NF)}')
    echo "${QUANT_DIR}/${short}-${method}"
}

# Quantize a model to INT4 if its checkpoint doesn't already exist.
ensure_quantized() {
    local model=$1
    local method=$2
    local path
    path=$(quantized_path "$model" "$method")
    if [ -d "$path" ] && [ -n "$(ls -A "$path" 2>/dev/null)" ]; then
        echo "  [quantize] reuse existing: $path"
    else
        echo "  [quantize] $model -> $path ($method)"
        python "$SCRIPT_DIR/quantize_model.py" \
            --model "$model" \
            --method "$method" \
            --output_dir "$path"
    fi
    echo "$path"
}

# Resolve the --model argument for run_eval: HF repo for fp16/int8,
# quantized checkpoint dir for int4_*.
resolve_model() {
    local model=$1
    local precision=$2
    case $precision in
        int4_gptq) quantized_path "$model" "gptq" ;;
        int4_awq)  quantized_path "$model" "awq"  ;;
        *)         echo "$model" ;;
    esac
}

run_eval() {
    local model=$1
    local precision=$2
    local benchmark=$3
    local resolved
    resolved=$(resolve_model "$model" "$precision")
    echo ""
    echo "--- $model | $precision | $benchmark ---"
    python "$SCRIPT_DIR/run_eval.py" \
        --model "$resolved" \
        --precision "$precision" \
        --benchmark "$benchmark" \
        --output_dir "$OUTPUT_DIR"
}

case $PHASE in
    phase1)
        echo "Phase 1: Pipeline check (1 model, FP16+INT4, 2 benchmarks)"
        MODEL="Qwen/Qwen2.5-3B-Instruct"
        ensure_quantized "$MODEL" "gptq" >/dev/null
        run_eval "$MODEL" "fp16" "gsm8k"
        run_eval "$MODEL" "fp16" "arc_challenge"
        run_eval "$MODEL" "int4_gptq" "gsm8k"
        run_eval "$MODEL" "int4_gptq" "arc_challenge"
        ;;

    phase2)
        echo "Phase 2: Signal check (2 models, FP16+INT4, 3 benchmarks)"
        for model in "Qwen/Qwen2.5-3B-Instruct" "meta-llama/Llama-3.2-3B-Instruct"; do
            ensure_quantized "$model" "gptq" >/dev/null
            for precision in "fp16" "int4_gptq"; do
                for bench in "gsm8k" "arc_challenge" "humaneval"; do
                    run_eval "$model" "$precision" "$bench"
                done
            done
        done
        ;;

    phase3)
        echo "Phase 3: Full paper (broadened small-model set, FP16+INT8+INT4, 6 benchmarks)"
        # Cross-architecture families + size-axis pairs, all <=4B.
        # Ungated (no token): Qwen, Phi-3.5, SmolLM2. Gated: Llama, Gemma.
        MODELS=(
            "Qwen/Qwen2.5-3B-Instruct"
            "meta-llama/Llama-3.2-3B-Instruct"
            "google/gemma-2-2b-it"
            "microsoft/Phi-3.5-mini-instruct"
            "HuggingFaceTB/SmolLM2-1.7B-Instruct"
            "Qwen/Qwen2.5-1.5B-Instruct"
            "meta-llama/Llama-3.2-1B-Instruct"
        )
        for model in "${MODELS[@]}"; do
            ensure_quantized "$model" "gptq" >/dev/null
            for precision in "fp16" "int8" "int4_gptq"; do
                for bench in "gsm8k" "arc_challenge" "hellaswag" "humaneval" "ifeval" "mgsm_direct"; do
                    run_eval "$model" "$precision" "$bench"
                done
            done
        done
        ;;

    phase3_open)
        echo "Phase 3 (ungated only): no HF_TOKEN needed, FP16+INT8+INT4, 6 benchmarks"
        MODELS=(
            "Qwen/Qwen2.5-3B-Instruct"
            "microsoft/Phi-3.5-mini-instruct"
            "HuggingFaceTB/SmolLM2-1.7B-Instruct"
            "Qwen/Qwen2.5-1.5B-Instruct"
        )
        for model in "${MODELS[@]}"; do
            ensure_quantized "$model" "gptq" >/dev/null
            for precision in "fp16" "int8" "int4_gptq"; do
                for bench in "gsm8k" "arc_challenge" "hellaswag" "humaneval" "ifeval" "mgsm_direct"; do
                    run_eval "$model" "$precision" "$bench"
                done
            done
        done
        ;;

    *)
        echo "Unknown phase: $PHASE"
        echo "Usage: bash run_all_evals.sh [phase1|phase2|phase3|phase3_open]"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "All $PHASE evaluations complete."
echo "Results saved to: $OUTPUT_DIR"
echo "Next: python scripts/aggregate_results.py --raw_dir results/raw/ --output_dir results/processed/"
echo "========================================="
