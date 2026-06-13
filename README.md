# Capability-Specific Degradation Patterns in Quantized Small Language Models

**When you quantize a small LLM to 4-bit, _which capability_ breaks first — and is it the same across model families?**

This repo contains the full study: **7 open instruction-tuned models** (1–4B params, 5 architecture families) evaluated at **FP16 vs 4-bit (NF4)** across **6 capabilities** — 84 evaluations in total — analyzed not by aggregate accuracy but by **per-capability degradation**.

📄 **Paper:** [`paper/main.pdf`](paper/main.pdf) · 📊 **Results dataset:** [`Emil-7/llm-quant-degradation`](https://huggingface.co/datasets/Emil-7/llm-quant-degradation) (Hugging Face)

---

## TL;DR — Key findings

1. **Degradation is capability-specific, not uniform.** Within a single model, relative degradation spans from ~2% (commonsense) to ~23% (multilingual math / code) — an order of magnitude. Aggregate scores hide this.
2. **Multilingual math and code are the most fragile; commonsense is rock-solid.** MGSM is the most-degraded capability for 5/6 measurable models; HellaSwag barely moves anywhere.
3. **The degradation _ordering_ is only weakly consistent across architectures** (mean pairwise Kendall's τ = **0.29**). The extremes are universal; the middle ranking is architecture-dependent — so you **can't port a degradation profile** from one model to another.
4. **Smaller models degrade more.** In the Qwen and Llama families, the 1–1.5B model loses more than its 3B sibling.
5. **A measurement pitfall, caught and fixed:** SmolLM2 scores ~0% on HumanEval under raw-completion prompting (it emits prose, not code). With correct chat-template prompting it scores 34% (FP16) → 15% (4-bit) — a **57% code collapse**, the largest in the study, that the standard protocol completely hid.

### Relative degradation (FP16 → 4-bit, %), higher = worse

| Model | GSM8K | ARC | HellaSwag | HumanEval | IFEval | MGSM |
|---|---|---|---|---|---|---|
| Qwen2.5-3B   | −2.6 | 3.4  | 2.6 | 9.2  | 1.9  | **19.0** |
| Qwen2.5-1.5B | 10.6 | 9.3  | 2.6 | **22.6** | 11.8 | **22.9** |
| Llama-3.2-3B | 4.0  | 3.1  | 1.8 | −6.1 | −0.5 | **19.1** |
| Llama-3.2-1B | 16.7 | 2.7  | 5.1 | 12.3 | 4.5  | — |
| Gemma-2-2B   | 4.1  | 2.4  | 1.8 | −2.6 | 1.0  | **12.2** |
| Phi-3.5-mini | 4.3  | −1.7 | 1.7 | **21.0** | 1.1  | **19.0** |
| SmolLM2-1.7B | 18.1 | 2.7  | 2.4 | **57.2**† | −1.2 | — |

†chat-template protocol (see paper §5.5). "—" = FP16 baseline too low for a stable ratio.

---

## Experimental setup

- **Models:** Qwen2.5-3B/1.5B-Instruct, Llama-3.2-3B/1B-Instruct, Gemma-2-2B-it, Phi-3.5-mini-instruct, SmolLM2-1.7B-Instruct.
- **Precisions:** FP16 (baseline) and 4-bit weight quantization (bitsandbytes NF4).
- **Capabilities → benchmarks:** ARC-Challenge (knowledge), HellaSwag (commonsense), GSM8K (math), MGSM (multilingual math), HumanEval (code), IFEval (instruction following), via the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).
- **Pipeline:** generative tasks on vLLM (throughput), multiple-choice loglikelihood on the HF backend; the same backend per capability across all precisions/models so differences reflect quantization, not the harness.

## Reproduce

Raw per-cell results (one JSON per model × precision × benchmark) live in the HF dataset. To regenerate the tables and figures from them:

```bash
# 1. download the raw results
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Emil-7/llm-quant-degradation", repo_type="dataset",
                  local_dir="results/raw_cloud", allow_patterns=["raw/**"])
PY

# 2. aggregate -> CSVs
python scripts/aggregate_results.py --raw_dir results/raw_cloud/raw --output_dir results/processed

# 3. analysis: degradation, rankings, Kendall's tau, size-axis, figures
python scripts/analyze_results.py --results_csv results/processed/results_table.csv --output_dir results/processed/analysis

# 4. LaTeX tables for the paper
python analysis/make_paper_tables.py --results_csv results/processed/results_table.csv \
    --analysis_dir results/processed/analysis --output_dir paper/tables
```

To re-run the evaluations themselves, see [`hf_jobs/`](hf_jobs/) (cloud GPU jobs on Hugging Face) and [`scripts/run_eval.py`](scripts/run_eval.py) (local). The chat-template HumanEval scorer used for SmolLM2 is [`scripts/he_chat_eval.py`](scripts/he_chat_eval.py).

## Build the paper

```bash
cd paper
# any TeX distribution (TeX Live / MiKTeX / tectonic):
pdflatex main && bibtex main && pdflatex main && pdflatex main
# (style files acl.sty + acl_natbib.bst are included)
```

## Citation

```bibtex
@misc{rahimov2026capability,
  title  = {Capability-Specific Degradation Patterns in Quantized Small Language Models},
  author = {Rahimov, Emil},
  year   = {2026},
  note   = {Preprint}
}
```

## Author

**Emil Rahimov** — emilrahimov25@gmail.com

*This research was conducted with AI assistance for code and writing; all experiments were run by the author and the author is responsible for the content.*
