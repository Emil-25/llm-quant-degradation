#!/usr/bin/env python3
"""
Resumable, resilient driver for the local evaluation matrix.

Runs (models x precisions x benchmarks) by calling run_eval.py per combo:
  - SKIPS combos whose result already exists -> resumable across shutdowns.
  - CONTINUES past individual failures (e.g. OOM), logging them, so one bad
    run never wastes the rest.
  - Adaptive batch size: FP16 generative -> 1 (VRAM-safe); else larger.

Order is chosen so the most useful results land first:
  - models small -> large (cheap/safe first; OOM-risky Phi-3.5 last)
  - precision passes: [fp16, int4_bnb] first (the key degradation comparison),
    then [int8] as a secondary pass.

Local note: INT4 here is int4_bnb (bitsandbytes NF4), because GPTQ tooling does
not build on Windows. GPTQ stays the paper's primary INT4 method (run on Colab).

Usage:
    python scripts/run_matrix.py            # run the full open matrix
    python scripts/run_matrix.py --dry_run  # print the plan, run nothing
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_results import find_result_jsons, parse_one, normalize_model_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Ungated models only (no HF token needed), small -> large.
MODELS = [
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]

# NOTE: humaneval is omitted locally -- its code_eval metric raises
# "not supported on Windows" (Unix-only process timeouts). Code generation is
# measured on Colab/Linux instead (together with GPTQ + the gated models).
BENCHMARKS = ["arc_challenge", "hellaswag", "gsm8k", "ifeval", "mgsm_direct"]
MULTI_CHOICE = {"arc_challenge", "hellaswag"}  # loglikelihood; batchable even at FP16

# Key degradation comparison first, INT8 robustness pass second.
PRECISION_PASSES = [["fp16", "int4_bnb"], ["int8"]]

# Approx params (billions), for VRAM-aware batch sizing on the 8 GB card.
MODEL_SIZE_B = {
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": 1.7,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
    "Qwen/Qwen2.5-3B-Instruct": 3.1,
    "microsoft/Phi-3.5-mini-instruct": 3.8,
}


def batch_for(model: str, precision: str, benchmark: str) -> str:
    """VRAM-aware batch size for an 8 GB card.

    FP16 weights dominate, so larger models must use smaller batches. Quantized
    precisions are light enough to batch freely. Generative FP16 stays at 1.
    """
    size = MODEL_SIZE_B.get(model, 3.1)
    if precision != "fp16":
        return "8"
    if benchmark in MULTI_CHOICE:
        if size < 2.0:
            return "4"
        if size < 3.5:
            return "2"
        return "1"  # Phi-3.8B: even multiple-choice is tight at FP16
    return "1"  # generative FP16


def build_done_set(raw_dir: str) -> set:
    """(model, precision, benchmark) tuples already present in results/raw."""
    done = set()
    for p in find_result_jsons(Path(raw_dir)):
        for row in parse_one(p):
            done.add((row["model"], row["precision"], row["benchmark"]))
    return done


def main():
    parser = argparse.ArgumentParser(description="Run the local evaluation matrix (resumable)")
    parser.add_argument("--raw_dir", default="results/raw")
    parser.add_argument("--dry_run", action="store_true", help="Print the plan, run nothing")
    args = parser.parse_args()

    raw_dir = args.raw_dir
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    run_eval = Path(__file__).resolve().parent / "run_eval.py"
    fail_log = Path("results/run_matrix_failures.log")

    done = build_done_set(raw_dir)
    logger.info(f"Found {len(done)} completed run(s) already on disk; these will be skipped.")

    # Build the ordered plan.
    plan = []
    for passes in PRECISION_PASSES:
        for model in MODELS:
            for precision in passes:
                for bench in BENCHMARKS:
                    plan.append((model, precision, bench))

    total = len(plan)
    skipped = ran = failed = 0
    for i, (model, precision, bench) in enumerate(plan, 1):
        key = (normalize_model_name(model), precision, bench)
        tag = f"[{i}/{total}] {model.split('/')[-1]} | {precision} | {bench}"
        if key in done:
            logger.info(f"{tag} -> SKIP (already done)")
            skipped += 1
            continue

        bs = batch_for(model, precision, bench)
        logger.info(f"{tag} -> RUN (bs={bs})")
        if args.dry_run:
            continue

        cmd = [sys.executable, str(run_eval),
               "--model", model, "--precision", precision,
               "--benchmark", bench, "--batch_size", bs,
               "--output_dir", raw_dir]
        result = subprocess.run(cmd)
        if result.returncode == 0:
            ran += 1
            done.add(key)
        else:
            failed += 1
            with open(fail_log, "a", encoding="utf-8") as f:
                f.write(f"{model} | {precision} | {bench} | rc={result.returncode}\n")
            logger.error(f"{tag} -> FAILED (rc={result.returncode}); logged, continuing")

    logger.info(f"Matrix complete. ran={ran} failed={failed} skipped={skipped} total={total}")
    if failed:
        logger.info(f"See {fail_log} for failed combos (e.g. OOM); rerun the script to retry them.")


if __name__ == "__main__":
    main()
