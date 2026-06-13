#!/usr/bin/env python3
"""
Run the int4 HellaSwag cells that were deferred from the cloud (to save budget)
on the LOCAL GPU, then upload each result to the dataset so analysis stays
single-sourced. Uses the SAME hf-backend config as the cloud MC path, so the
numbers are consistent with the cloud-run cells.

HellaSwag is multiple-choice (loglikelihood) -> hf backend, batch_size=auto,
10-shot, no chat template. int4 = bitsandbytes NF4 (fits the 8 GB 4060).

Runs models sequentially (one at a time) to avoid OOM on 8 GB.
"""
import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("local")

REPO = "Emil-7/llm-quant-degradation"
# (model_path, needs_eager_attn)
DEFERRED = [
    ("Qwen/Qwen2.5-3B-Instruct", False),
    ("meta-llama/Llama-3.2-3B-Instruct", False),
    ("google/gemma-2-2b-it", True),
]
BENCHMARK = "hellaswag"
NUM_FEWSHOT = 10


def model_args(model_path, eager):
    parts = [
        f"pretrained={model_path}", "trust_remote_code=True", "dtype=float16",
        "load_in_4bit=True", "bnb_4bit_quant_type=nf4", "bnb_4bit_compute_dtype=float16",
    ]
    if eager:
        parts.append("attn_implementation=eager")
    return ",".join(parts)


def newest_json(out_dir):
    files = list(Path(out_dir).rglob("results_*.json"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true", help="upload each result to the dataset")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    token = os.environ.get("HF_TOKEN")

    # Resume: skip cells already in the dataset (robust to restarts).
    try:
        existing = set(api.list_repo_files(REPO, repo_type="dataset", token=token))
    except Exception:
        existing = set()

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    for model_path, eager in DEFERRED:
        short = model_path.rstrip("/").split("/")[-1].lower()
        repo_path = f"raw/{short}/int4_bnb__{BENCHMARK}.json"
        if repo_path in existing:
            log.info(f"SKIP (already in dataset): {short} | {BENCHMARK}")
            continue
        out_dir = f"results/raw_local/{short}"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "lm_eval", "--model", "hf",
            "--model_args", model_args(model_path, eager),
            "--tasks", BENCHMARK, "--num_fewshot", str(NUM_FEWSHOT),
            "--batch_size", "auto", "--output_path", out_dir,
        ]
        log.info(f"RUN (local) {short} | int4_bnb | {BENCHMARK}")
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            log.error(f"FAILED {short} | {BENCHMARK} (rc={e.returncode})")
            continue
        rj = newest_json(out_dir)
        if rj is None:
            log.error(f"no result json for {short}")
            continue
        log.info(f"DONE {short} -> {rj}")
        if args.upload:
            # Result is already saved locally; retry upload on transient network
            # errors and NEVER crash the run (a later pass can upload from disk).
            for attempt in range(6):
                try:
                    api.upload_file(
                        path_or_fileobj=str(rj),
                        path_in_repo=repo_path,
                        repo_id=REPO, repo_type="dataset", token=token,
                    )
                    log.info(f"UPLOADED {repo_path}")
                    break
                except Exception as e:
                    wait = 20 * (attempt + 1)
                    log.warning(f"upload attempt {attempt+1}/6 failed ({e}); retry in {wait}s")
                    time.sleep(wait)
            else:
                log.error(f"UPLOAD FAILED after retries: {repo_path} (saved locally at {rj})")

    log.info("ALL LOCAL DEFERRED DONE")


if __name__ == "__main__":
    main()
