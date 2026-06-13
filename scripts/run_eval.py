#!/usr/bin/env python3
"""
Run a single evaluation: one model, one precision, one benchmark.
Saves results as JSON with metadata for later aggregation.

For quantized precisions (int4_gptq, int4_awq) you must pass a path to an
ALREADY-QUANTIZED checkpoint via --model (either a local dir produced by
scripts/quantize_model.py, or a pre-quantized HF repo). lm-eval auto-detects
the quantization config from the checkpoint. INT8 (bitsandbytes) is the only
precision quantized on the fly at load time.

Usage:
    # FP16 baseline
    python run_eval.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --precision fp16 \
        --benchmark gsm8k \
        --output_dir results/raw/

    # INT4-GPTQ (point --model at the quantized checkpoint)
    python run_eval.py \
        --model results/quantized/qwen2.5-3b-instruct-gptq \
        --precision int4_gptq \
        --benchmark arc_challenge \
        --output_dir results/raw/
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Per-benchmark eval settings.
#   apply_chat_template: use the model's chat template (required for IFEval on
#       instruct models; left off elsewhere to match Open-LLM-Leaderboard style).
#   unsafe_code: HumanEval executes generated code, which lm-eval gates behind
#       an explicit confirmation flag.
# Generative tasks use the chat template: these are instruct models, and without
# it some emit empty output (observed: SmolLM2 GSM8K -> blank generations, 0.3%).
# Loglikelihood multiple-choice (ARC, HellaSwag) don't generate, so they keep the
# standard no-template loglikelihood setup (Open-LLM-Leaderboard style).
BENCHMARK_CONFIG = {
    "arc_challenge": {"num_fewshot": 25, "metric": "acc_norm", "capability": "Factual Knowledge",
                      "apply_chat_template": False, "unsafe_code": False},
    "gsm8k":         {"num_fewshot": 8,  "metric": "exact_match,flexible-extract", "capability": "Math Reasoning",
                      "apply_chat_template": True, "unsafe_code": False},
    "humaneval":     {"num_fewshot": 0,  "metric": "pass@1", "capability": "Code Generation",
                      "apply_chat_template": True, "unsafe_code": True},
    "hellaswag":     {"num_fewshot": 10, "metric": "acc_norm", "capability": "Commonsense",
                      "apply_chat_template": False, "unsafe_code": False},
    "ifeval":        {"num_fewshot": 0,  "metric": "prompt_level_strict_acc", "capability": "Instruction Following",
                      "apply_chat_template": True, "unsafe_code": False},
    "mgsm_direct":   {"num_fewshot": 8,  "metric": "exact_match", "capability": "Multilingual",
                      "apply_chat_template": True, "unsafe_code": False},
}


def build_model_args(model_path: str, precision: str) -> str:
    """Build the model_args string for lm-eval-harness."""
    parts = [f"pretrained={model_path}", "trust_remote_code=True"]

    if precision == "fp16":
        parts.append("dtype=float16")
    elif precision == "int8":
        # On-the-fly 8-bit quantization via bitsandbytes at load time.
        parts.append("dtype=float16")
        parts.append("load_in_8bit=True")
    elif precision == "int4_bnb":
        # On-the-fly 4-bit (NF4) quantization via bitsandbytes at load time.
        # Used where GPTQ tooling is unavailable (e.g. Windows local runs).
        parts.append("dtype=float16")
        parts.append("load_in_4bit=True")
        parts.append("bnb_4bit_quant_type=nf4")
        parts.append("bnb_4bit_compute_dtype=float16")
    elif precision in ("int4_gptq", "int4_awq"):
        # model_path must be an already-quantized checkpoint; the quantization
        # config travels with the checkpoint and is auto-detected by HF/optimum.
        parts.append("dtype=float16")
    else:
        raise ValueError(f"Unknown precision: {precision}")

    # Gemma-2 requires eager attention to apply logit soft-capping correctly;
    # other attention impls can silently degrade or warn.
    if "gemma-2" in model_path.lower() or "gemma2" in model_path.lower():
        parts.append("attn_implementation=eager")

    return ",".join(parts)


def build_lm_eval_command(model_path: str, precision: str, benchmark: str,
                          output_dir: str, batch_size: str, limit=None) -> list[str]:
    """Build the lm_eval CLI command."""
    bench_cfg = BENCHMARK_CONFIG[benchmark]

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", build_model_args(model_path, precision),
        "--tasks", benchmark,
        "--num_fewshot", str(bench_cfg["num_fewshot"]),
        "--batch_size", batch_size,
        "--output_path", output_dir,
        "--log_samples",
    ]
    if bench_cfg["apply_chat_template"]:
        cmd.append("--apply_chat_template")
        # With a chat template AND few-shot, each shot must be its own turn,
        # or lm-eval errors / malforms the prompt.
        if bench_cfg["num_fewshot"] > 0:
            cmd.append("--fewshot_as_multiturn")
    if bench_cfg["unsafe_code"]:
        cmd.append("--confirm_run_unsafe_code")
    if limit is not None:
        # Smoke-test convenience: evaluate only the first N examples.
        # NOT for reported results — produces partial-set scores.
        cmd += ["--limit", str(limit)]
    return cmd


def make_result_filename(model_path: str, precision: str, benchmark: str) -> str:
    """Create a descriptive filename stem for the result metadata."""
    model_short = model_path.rstrip("/").split("/")[-1].lower().replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_short}__{precision}__{benchmark}__{timestamp}"


def main():
    parser = argparse.ArgumentParser(description="Run single evaluation")
    parser.add_argument("--model", required=True,
                        help="HF model path (FP16/INT8) or quantized checkpoint path (INT4)")
    parser.add_argument("--precision", required=True,
                        choices=["fp16", "int8", "int4_bnb", "int4_gptq", "int4_awq"])
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARK_CONFIG.keys()))
    parser.add_argument("--output_dir", default="results/raw/", help="Directory to save results")
    parser.add_argument("--batch_size", default="auto",
                        help="lm-eval batch size; use a small int (e.g. 1) on tight VRAM")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N examples (smoke test only; not for reported results)")
    parser.add_argument("--dry_run", action="store_true", help="Print command without running")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_lm_eval_command(args.model, args.precision, args.benchmark,
                                str(output_dir), args.batch_size, limit=args.limit)

    logger.info(f"Model: {args.model}")
    logger.info(f"Precision: {args.precision}")
    logger.info(f"Benchmark: {args.benchmark}")
    logger.info(f"Command: {' '.join(cmd)}")

    if args.precision in ("int4_gptq", "int4_awq") and "/" not in args.model.replace("\\", "/"):
        logger.warning("INT4 precision expects a path to a pre-quantized checkpoint. "
                       "If this is a bare HF repo without a quantization_config, the run will fail. "
                       "Quantize first with scripts/quantize_model.py.")

    if args.dry_run:
        logger.info("Dry run — command printed but not executed.")
        return

    # Save metadata alongside results so the aggregator can tie scores back to
    # (model, precision, benchmark) regardless of lm-eval's output nesting.
    metadata = {
        "model": args.model,
        "precision": args.precision,
        "benchmark": args.benchmark,
        "capability": BENCHMARK_CONFIG[args.benchmark]["capability"],
        "command": " ".join(cmd),
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = output_dir / (make_result_filename(args.model, args.precision, args.benchmark) + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {meta_path}")

    # Run evaluation. Import subprocess lazily so --dry_run works without it.
    import os
    import subprocess
    # Force UTF-8 I/O for the child: lm-eval prints a results table containing
    # non-ASCII glyphs (e.g. arrows) that crash on Windows' default cp1252 console.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # Force the classic HF download path: the newer Xet storage backend stalls
    # on some networks (observed: large safetensors blobs hang at ~0%).
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    # HumanEval's code_eval metric refuses to execute generated code unless this
    # is set (in addition to the --confirm_run_unsafe_code CLI flag).
    if args.benchmark == "humaneval":
        env["HF_ALLOW_CODE_EVAL"] = "1"
    try:
        subprocess.run(cmd, check=True, capture_output=False, env=env)
        logger.info("Evaluation completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed with return code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("lm_eval not found. Install with: pip install lm-eval")
        sys.exit(1)


if __name__ == "__main__":
    main()
