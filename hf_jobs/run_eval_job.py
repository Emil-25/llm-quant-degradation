# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "lm-eval>=0.4.5",
#   "vllm==0.6.6.post1",
#   "transformers>=4.47,<4.50",
#   "accelerate",
#   "datasets",
#   "numpy",
#   "scipy",
#   "sentencepiece",
#   "protobuf",
#   "bitsandbytes",
#   "human-eval",
#   "langdetect",
#   "immutabledict",
#   "nltk",
#   "absl-py",
#   "huggingface_hub",
# ]
# ///
"""
Self-contained HF Jobs eval: run ONE model across precisions x capabilities on a
Linux GPU, uploading each result to a HF dataset repo AS IT COMPLETES.

Backend: HYBRID, picked per task type on a 24 GB L4:
  - Generative tasks (gsm8k/humaneval/ifeval/mgsm) -> vLLM (gpu_mem=0.90). The hf
    backend crawled (20-30 min each, token-by-token); vLLM is ~5-10x faster.
  - Multiple-choice tasks (arc/hellaswag) -> hf with batch_size=auto. vLLM OOMs on
    loglikelihood regardless of the memory pool (lowering it makes the scheduler
    batch more prefill tokens, so the full-vocab logprob tensor GROWS: 2.2 GiB at
    0.90 -> 4.3 GiB at 0.45). vLLM is built for generation, not scoring. hf is
    proven; it's slow on HellaSwag's ~40k requests, absorbed by a 6h timeout +
    running all models in parallel.
Each capability uses ONE backend across all precisions/models, so the
fp16-vs-int4 comparison within a capability stays internally consistent.

Key properties (learned the hard way):
  - INCREMENTAL upload: each (precision, benchmark) result is uploaded right
    after it finishes, so a job timeout never loses completed work.
  - RESUMABLE: at startup it lists results already in the dataset for this model
    and skips them, so relaunching continues where it left off.
  - LOUD failures: stderr is captured and the tail is printed on failure, so a
    broken eval never disappears silently again.

INT4 method: bitsandbytes NF4 in-flight quantization (`int4_bnb`); GPTQ dropped.
HumanEval uses raw completion (no chat template); other generative tasks use the
chat template; multiple-choice tasks use standard loglikelihood.

Usage (invoked by `hf jobs uv run`):
    python run_eval_job.py --model Qwen/Qwen2.5-3B-Instruct \
        --precisions fp16,int4_bnb \
        --results_repo <user>/llm-quant-degradation
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("job")

BENCHMARK_CONFIG = {
    "arc_challenge": {"num_fewshot": 25, "apply_chat_template": False, "unsafe_code": False, "kind": "mc"},
    "hellaswag":     {"num_fewshot": 10, "apply_chat_template": False, "unsafe_code": False, "kind": "mc"},
    "gsm8k":         {"num_fewshot": 8,  "apply_chat_template": True,  "unsafe_code": False, "kind": "gen"},
    "humaneval":     {"num_fewshot": 0,  "apply_chat_template": False, "unsafe_code": True,  "kind": "gen"},
    "ifeval":        {"num_fewshot": 0,  "apply_chat_template": True,  "unsafe_code": False, "kind": "gen"},
    "mgsm_direct":   {"num_fewshot": 8,  "apply_chat_template": True,  "unsafe_code": False, "kind": "gen"},
}
DEFAULT_BENCHMARKS = ["arc_challenge", "hellaswag", "gsm8k", "humaneval", "ifeval", "mgsm_direct"]

# vLLM needs an explicit context budget. 25-shot ARC / 10-shot HellaSwag are the
# longest prompts and sit comfortably under 4096 tokens for these small models.
MAX_MODEL_LEN = 4096


def vllm_model_args(model_path: str, precision: str, gpu_mem: float = 0.90) -> str:
    # gpu_mem caps vLLM's pre-reserved pool. Generation wants it high (0.90) for a
    # big KV cache. Multiple-choice loglikelihood needs it LOW (~0.45): scoring
    # materializes a large logprob tensor over the full vocab, and that spike must
    # fit OUTSIDE vLLM's reserved pool or it OOMs (it did at 0.90).
    parts = [
        f"pretrained={model_path}",
        "trust_remote_code=True",
        "dtype=float16",
        f"max_model_len={MAX_MODEL_LEN}",
        f"gpu_memory_utilization={gpu_mem}",
    ]
    if precision == "int4_bnb":
        # vLLM in-flight bitsandbytes quantization (NF4, 4-bit weights).
        parts += ["quantization=bitsandbytes", "load_format=bitsandbytes"]
    elif precision == "int8":
        # vLLM bitsandbytes is 4-bit only; routing int8 here would silently
        # mislabel 4-bit weights as int8. Real int8 must use the hf backend.
        raise ValueError("int8 is not supported on the vLLM path; use the hf backend")
    if "gemma-2" in model_path.lower():
        parts.append("enforce_eager=True")
    return ",".join(parts)


def hf_model_args(model_path: str, precision: str) -> str:
    parts = [f"pretrained={model_path}", "trust_remote_code=True", "dtype=float16"]
    if precision == "int8":
        parts.append("load_in_8bit=True")
    elif precision == "int4_bnb":
        parts += ["load_in_4bit=True", "bnb_4bit_quant_type=nf4", "bnb_4bit_compute_dtype=float16"]
    if "gemma-2" in model_path.lower():
        parts.append("attn_implementation=eager")
    return ",".join(parts)


def newest_result_json(out_dir: str):
    files = list(Path(out_dir).rglob("results_*.json"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def run_one(model_path: str, precision: str, benchmark: str, out_dir: str):
    cfg = BENCHMARK_CONFIG[benchmark]
    # Generation -> vLLM (fast). Multiple-choice loglikelihood -> hf backend.
    # vLLM was tried for MC and OOMs no matter the memory pool: lowering the pool
    # only makes its scheduler batch MORE prefill tokens, so the full-vocab logprob
    # tensor grows (2.2 GiB @0.90 -> 4.3 GiB @0.45). vLLM is built for generation,
    # not loglikelihood scoring. hf with batch_size=auto is proven and reliable
    # (slower on HellaSwag's ~40k requests, absorbed by a 6h timeout + parallel runs).
    if cfg["kind"] == "gen":
        backend, margs = "vllm", vllm_model_args(model_path, precision, 0.90)
    else:
        backend, margs = "hf", hf_model_args(model_path, precision)
    cmd = [
        sys.executable, "-m", "lm_eval", "--model", backend,
        "--model_args", margs,
        "--tasks", benchmark, "--num_fewshot", str(cfg["num_fewshot"]),
        "--batch_size", "auto", "--output_path", out_dir,
    ]
    if cfg["apply_chat_template"]:
        cmd.append("--apply_chat_template")
        if cfg["num_fewshot"] > 0:
            cmd.append("--fewshot_as_multiturn")
    if cfg["unsafe_code"]:
        cmd.append("--confirm_run_unsafe_code")

    env = os.environ.copy()
    if benchmark == "humaneval":
        env["HF_ALLOW_CODE_EVAL"] = "1"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    log.info(f"RUN {precision} | {benchmark} ({backend})")

    # Stream + retain stderr so failures are diagnosable from the job log.
    tail = deque(maxlen=40)
    proc = subprocess.Popen(cmd, env=env, stderr=subprocess.PIPE, text=True)
    for line in proc.stderr:
        tail.append(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        log.error(f"FAILED {precision} | {benchmark} (rc={proc.returncode}) -- stderr tail:")
        for line in tail:
            log.error(f"  | {line}")
        return None
    return newest_result_json(out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--precisions", default="fp16,int4_bnb")
    ap.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    ap.add_argument("--results_repo", required=True)
    args = ap.parse_args()

    precisions = args.precisions.split(",")
    benchmarks = args.benchmarks.split(",")
    out_dir = "/tmp/results_raw"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model_short = args.model.rstrip("/").split("/")[-1].lower()

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.results_repo, repo_type="dataset", private=True, exist_ok=True)

    # Resume: which (precision, benchmark) results already exist for this model?
    prefix = f"raw/{model_short}/"
    try:
        existing = api.list_repo_files(args.results_repo, repo_type="dataset")
        done = {f[len(prefix):-5] for f in existing if f.startswith(prefix) and f.endswith(".json")}
    except Exception:
        done = set()
    log.info(f"Resume: {len(done)} result(s) already uploaded for {model_short}: {sorted(done)}")

    summary = {}
    for precision in precisions:
        for bench in benchmarks:
            key = f"{precision}__{bench}"
            if key in done:
                log.info(f"SKIP (already uploaded): {key}")
                summary[key] = "skip"
                continue
            rj = run_one(args.model, precision, bench, out_dir)
            if rj is None:
                summary[key] = "FAILED"
                continue
            # Upload this single result immediately under a deterministic name.
            api.upload_file(
                path_or_fileobj=str(rj),
                path_in_repo=f"{prefix}{key}.json",
                repo_id=args.results_repo,
                repo_type="dataset",
            )
            done.add(key)
            summary[key] = "ok"
            log.info(f"UPLOADED {key}")

    log.info(f"SUMMARY: {json.dumps(summary, indent=2)}")
    log.info("JOB DONE")


if __name__ == "__main__":
    main()
