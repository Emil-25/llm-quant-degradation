#!/usr/bin/env python3
"""
Quantize a model to INT4 (GPTQ or AWQ) and save the checkpoint to disk.

The saved checkpoint carries its own quantization_config, so it can be passed
directly to run_eval.py (--precision int4_gptq / int4_awq) and lm-eval will
auto-detect the quantization.

Settings (bits, group_size, calibration dataset, etc.) are read from
configs/quantization.yaml so they stay in sync with the documented design.

Usage:
    # GPTQ (primary INT4 method)
    python quantize_model.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --method gptq \
        --output_dir results/quantized/qwen2.5-3b-instruct-gptq

    # AWQ (robustness check)
    python quantize_model.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --method awq \
        --output_dir results/quantized/qwen2.5-3b-instruct-awq

Notes:
    - GPTQ here uses gptqmodel if installed (maintained successor to auto-gptq),
      falling back to auto-gptq. On recent torch/CUDA, prefer gptqmodel.
    - Many models already have community pre-quantized GPTQ/AWQ checkpoints on
      HuggingFace; if one exists for your exact model+settings you can skip this
      step and point run_eval.py straight at that repo.
    - T4 (15GB): quantizing a 3B model is feasible but slow; keep n_samples modest.
"""

import argparse
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "quantization.yaml"


def load_quant_config(config_path: Path, method: str) -> dict:
    """Read the GPTQ/AWQ settings block from quantization.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    key = "int4_gptq" if method == "gptq" else "int4_awq"
    block = cfg["precisions"][key]
    return block.get("gptq_args" if method == "gptq" else "awq_args", {})


def get_calibration_texts(dataset_name: str, n_samples: int) -> list[str]:
    """Load a small calibration corpus (default: C4) for quantization."""
    from datasets import load_dataset

    logger.info(f"Loading {n_samples} calibration samples from '{dataset_name}'...")
    if dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        texts = []
        for ex in ds:
            text = ex.get("text", "").strip()
            if len(text) > 50:
                texts.append(text)
            if len(texts) >= n_samples:
                break
        return texts
    # Fallback: treat dataset_name as an HF dataset id with a 'text' column.
    ds = load_dataset(dataset_name, split="train")
    return [ex["text"] for ex in ds.select(range(min(n_samples, len(ds))))]


def quantize_gptq(model_path: str, output_dir: str, qcfg: dict, n_samples: int):
    """Quantize with GPTQ (gptqmodel preferred, auto-gptq fallback)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    calibration = get_calibration_texts(qcfg.get("dataset", "c4"), n_samples)

    try:
        from gptqmodel import GPTQModel, QuantizeConfig

        logger.info("Using gptqmodel backend.")
        quantize_config = QuantizeConfig(
            bits=qcfg.get("bits", 4),
            group_size=qcfg.get("group_size", 128),
            desc_act=qcfg.get("desc_act", False),
            damp_percent=qcfg.get("damp_percent", 0.1),
        )
        model = GPTQModel.load(model_path, quantize_config, trust_remote_code=True)
        model.quantize(calibration)
        model.save(output_dir)
    except ImportError:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

        logger.info("gptqmodel not found; using auto-gptq backend.")
        quantize_config = BaseQuantizeConfig(
            bits=qcfg.get("bits", 4),
            group_size=qcfg.get("group_size", 128),
            desc_act=qcfg.get("desc_act", False),
            damp_percent=qcfg.get("damp_percent", 0.1),
        )
        model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config,
                                                    trust_remote_code=True)
        examples = [tokenizer(t, return_tensors="pt", truncation=True, max_length=2048)
                    for t in calibration]
        model.quantize(examples)
        model.save_quantized(output_dir)

    tokenizer.save_pretrained(output_dir)
    logger.info(f"GPTQ checkpoint saved to {output_dir}")


def quantize_awq(model_path: str, output_dir: str, qcfg: dict):
    """Quantize with AWQ via AutoAWQ."""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    quant_config = {
        "w_bit": qcfg.get("w_bit", 4),
        "q_group_size": qcfg.get("q_group_size", 128),
        "version": qcfg.get("version", "gemm"),
        "zero_point": True,
    }
    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"AWQ checkpoint saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Quantize a model to INT4 (GPTQ/AWQ)")
    parser.add_argument("--model", required=True, help="HF model path to quantize")
    parser.add_argument("--method", required=True, choices=["gptq", "awq"])
    parser.add_argument("--output_dir", required=True, help="Where to save the quantized checkpoint")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to quantization.yaml")
    parser.add_argument("--n_samples", type=int, default=128,
                        help="Number of calibration samples (keep modest on T4)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qcfg = load_quant_config(Path(args.config), args.method)
    logger.info(f"Quantizing {args.model} via {args.method.upper()} with config: {qcfg}")

    if args.method == "gptq":
        quantize_gptq(args.model, str(output_dir), qcfg, args.n_samples)
    else:
        quantize_awq(args.model, str(output_dir), qcfg)


if __name__ == "__main__":
    main()
