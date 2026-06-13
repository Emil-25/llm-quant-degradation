#!/usr/bin/env python3
"""
Aggregate lm-eval-harness raw JSON outputs into the results table the analysis
step consumes. Closes the gap between `run_eval.py` outputs and
`analyze_results.py` inputs — no manual copying.

Walks --raw_dir for lm-eval result JSONs, extracts the headline metric per task,
and writes:
  - results_long.csv : one row per run (model, precision, benchmark, score, stderr)
  - results_table.csv: wide format (model, precision, <one column per capability>)

Usage:
    python aggregate_results.py \
        --raw_dir results/raw/ \
        --output_dir results/processed/
"""

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Metric to read per task. May be a bare base name ("acc_norm") or a fully
# qualified "<base>,<filter>" key.
# NOTE gsm8k: we use flexible-extract, not strict-match. strict-match requires
# the model to emit the exact "#### <num>" format, which instruct models do not,
# so it drastically understates their true math ability (e.g. 14.6% strict vs
# 66.3% flexible for Qwen2.5-3B-Instruct). flexible-extract reflects capability.
TASK_BASE_METRIC = {
    "arc_challenge": "acc_norm",
    "gsm8k": "exact_match,flexible-extract",
    "humaneval": "pass@1",
    "hellaswag": "acc_norm",
    "ifeval": "prompt_level_strict_acc",
    "mgsm_direct": "exact_match",
}

# Task -> column name in the wide results table.
TASK_TO_COLUMN = {
    "arc_challenge": "arc_challenge",
    "gsm8k": "gsm8k",
    "humaneval": "humaneval",
    "hellaswag": "hellaswag",
    "ifeval": "ifeval",
    "mgsm_direct": "mgsm",
}

WIDE_COLUMNS = ["gsm8k", "arc_challenge", "hellaswag", "humaneval", "ifeval", "mgsm"]

# Suffixes added to quantized checkpoint dirs; stripped so quantized and FP16
# rows share the same canonical model key (degradation groups by model).
QUANT_SUFFIXES = ["gptq", "awq", "int4", "int8", "4bit", "8bit", "gemm", "quantized"]


def normalize_model_name(pretrained: str) -> str:
    """Canonicalize a model path/repo to a name shared across precisions."""
    base = pretrained.rstrip("/\\").replace("\\", "/").split("/")[-1].lower()
    tokens = re.split(r"[-_]", base)
    kept = [t for t in tokens if t and t not in QUANT_SUFFIXES]
    return "-".join(kept)


def extract_model_and_precision(config: dict):
    """Get (pretrained, precision) from lm-eval config.

    lm-eval stores config['model_args'] either as a dict (newer) or as a
    'pretrained=...,key=val' string (older). Handle both.
    """
    ma = config.get("model_args", "")

    if isinstance(ma, dict):
        pretrained = str(ma.get("pretrained", config.get("model", "unknown")))
        # vLLM (gen tasks) encodes bnb 4-bit as quantization/load_format=bitsandbytes;
        # the hf backend (MC tasks) encodes it as load_in_4bit. Detect both so a
        # model's int4 gen cells aren't mislabeled fp16.
        if ma.get("load_in_8bit"):
            prec = "int8"
        elif ma.get("load_in_4bit") or "bitsandbytes" in str(ma.get("quantization", "")).lower() \
                or "bitsandbytes" in str(ma.get("load_format", "")).lower():
            prec = "int4_bnb"
        elif "gptq" in pretrained.lower():
            prec = "int4_gptq"
        elif "awq" in pretrained.lower():
            prec = "int4_awq"
        else:
            prec = "fp16"
        return pretrained, prec

    # string form
    m = re.search(r"pretrained=([^,]+)", ma)
    pretrained = m.group(1) if m else config.get("model", "unknown")
    low_ma, low_p = ma.lower(), pretrained.lower()
    if "load_in_8bit=true" in low_ma:
        prec = "int8"
    elif "load_in_4bit=true" in low_ma or "quantization=bitsandbytes" in low_ma \
            or "load_format=bitsandbytes" in low_ma:
        prec = "int4_bnb"
    elif "gptq" in low_p:
        prec = "int4_gptq"
    elif "awq" in low_p:
        prec = "int4_awq"
    else:
        prec = "fp16"
    return pretrained, prec


def extract_metric(task_results: dict, metric_spec: str):
    """Return (score, stderr) for a metric spec.

    metric_spec is either a bare base name ('acc_norm') or a fully qualified
    '<base>,<filter>' key ('exact_match,flexible-extract').
    """
    if "," in metric_spec:
        base, filt = metric_spec.split(",", 1)
        score_key = metric_spec if metric_spec in task_results else None
        stderr_key = f"{base}_stderr,{filt}"
    else:
        base = metric_spec
        score_key = next((k for k in task_results
                          if k.startswith(base + ",") or k == base), None)
        stderr_key = score_key.replace(base, base + "_stderr", 1) if score_key else None

    if score_key is None:
        return None, None
    score = task_results[score_key]
    stderr = task_results.get(stderr_key) if stderr_key else None
    # Scores are 0-1 fractions in lm-eval; express as percentages for the paper.
    return (round(score * 100, 2) if isinstance(score, (int, float)) else None,
            round(stderr * 100, 2) if isinstance(stderr, (int, float)) else None)


def find_result_jsons(raw_dir: Path) -> list[Path]:
    """All lm-eval result JSONs (exclude our .meta.json and per-sample logs)."""
    out = []
    for p in raw_dir.rglob("*.json"):
        name = p.name.lower()
        if name.endswith(".meta.json") or name.startswith("samples_"):
            continue
        out.append(p)
    return out


# MGSM reports per-language sub-tasks (mgsm_direct_<lang>), not one aggregate.
# We macro-average the standard 11-language set (excludes the non-standard
# mgsm_direct_es_spanish_bench variant).
MGSM_LANGS = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
MGSM_METRIC = "exact_match,flexible-extract"


def parse_one(path: Path) -> list[dict]:
    """Parse a single lm-eval result JSON into long-format rows."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        logger.warning(f"Skipping unreadable {path}: {e}")
        return []

    results = data.get("results")
    if not results:
        return []

    config = data.get("config", {})
    pretrained, precision = extract_model_and_precision(config)
    model = normalize_model_name(pretrained)

    rows = []
    for task, task_results in results.items():
        if task not in TASK_BASE_METRIC:
            continue
        score, stderr = extract_metric(task_results, TASK_BASE_METRIC[task])
        if score is None:
            logger.warning(f"No '{TASK_BASE_METRIC[task]}' metric for task '{task}' in {path.name}")
            continue
        rows.append({
            "model": model,
            "precision": precision,
            "benchmark": task,
            "column": TASK_TO_COLUMN[task],
            "score": score,
            "stderr": stderr,
            "source_file": path.name,
        })

    # MGSM: macro-average the per-language sub-tasks into one 'mgsm' score.
    lang_scores = []
    for lang in MGSM_LANGS:
        key = f"mgsm_direct_{lang}"
        if key in results:
            s, _ = extract_metric(results[key], MGSM_METRIC)
            if s is not None:
                lang_scores.append(s)
    if lang_scores:
        rows.append({
            "model": model,
            "precision": precision,
            "benchmark": "mgsm_direct",
            "column": "mgsm",
            "score": round(sum(lang_scores) / len(lang_scores), 2),
            "stderr": None,  # macro-average; per-language stderrs not pooled
            "source_file": path.name,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Aggregate lm-eval JSONs into result tables")
    parser.add_argument("--raw_dir", default="results/raw/", help="Directory of lm-eval JSON outputs")
    parser.add_argument("--output_dir", default="results/processed/", help="Where to write the CSVs")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for path in find_result_jsons(raw_dir):
        all_rows.extend(parse_one(path))

    if not all_rows:
        logger.error(f"No parseable results found under {raw_dir}. Have you run any evals?")
        return

    long_df = pd.DataFrame(all_rows)
    # If a (model, precision, benchmark) was run more than once, keep the last.
    long_df = long_df.drop_duplicates(subset=["model", "precision", "benchmark"], keep="last")
    long_df = long_df.sort_values(["model", "precision", "benchmark"]).reset_index(drop=True)
    long_path = output_dir / "results_long.csv"
    long_df.to_csv(long_path, index=False)
    logger.info(f"Wrote {len(long_df)} runs to {long_path}")

    # Pivot to the wide table the analysis step expects.
    wide = long_df.pivot_table(index=["model", "precision"], columns="column",
                               values="score", aggfunc="last")
    for col in WIDE_COLUMNS:
        if col not in wide.columns:
            wide[col] = pd.NA
    wide = wide[WIDE_COLUMNS].reset_index()
    wide_path = output_dir / "results_table.csv"
    wide.to_csv(wide_path, index=False)
    logger.info(f"Wrote results table to {wide_path}")
    logger.info(f"\n{wide.to_string(index=False)}")

    # Report any missing cells so you know what still needs running.
    missing = []
    for _, r in wide.iterrows():
        for col in WIDE_COLUMNS:
            if pd.isna(r[col]):
                missing.append(f"{r['model']} / {r['precision']} / {col}")
    if missing:
        logger.warning(f"{len(missing)} missing cell(s):")
        for m in missing:
            logger.warning(f"  - {m}")


if __name__ == "__main__":
    main()
