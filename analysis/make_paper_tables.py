#!/usr/bin/env python3
"""
Generate LaTeX tables for the paper from the processed CSVs.

Reads:
  - results/processed/results_table.csv      (absolute scores, from aggregate_results.py)
  - analysis/degradation.csv                  (relative degradation, from analyze_results.py)
  - analysis/kendall_tau.csv                  (ranking consistency, optional)

Writes booktabs LaTeX tables into paper/tables/ that main.tex \\input{}s:
  - results_main.tex      : absolute scores per model x precision x capability
  - degradation.tex       : relative degradation (%)
  - kendall_tau.tex       : pairwise Kendall's tau (if available)

Usage:
    python analysis/make_paper_tables.py \
        --results_csv results/processed/results_table.csv \
        --analysis_dir analysis/ \
        --output_dir paper/tables/
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _safe_read_csv(path):
    """Read a CSV, returning None if it is empty/headerless (optional tables that
    aren't ready yet write a 0-column file, which would otherwise raise)."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None

CAP_ORDER = ["arc_challenge", "gsm8k", "humaneval", "hellaswag", "ifeval", "mgsm"]
CAP_LABELS = {
    "arc_challenge": "Factual",
    "gsm8k": "Math",
    "humaneval": "Code",
    "hellaswag": "Common.",
    "ifeval": "Instr.",
    "mgsm": "Multiling.",
}
PRECISION_LABELS = {"fp16": "FP16", "int8": "INT8", "int4_gptq": "INT4-GPTQ", "int4_awq": "INT4-AWQ"}
PRECISION_ORDER = ["fp16", "int8", "int4_gptq", "int4_awq"]


def fmt(v) -> str:
    return "--" if pd.isna(v) else f"{v:.1f}"


def model_label(name: str) -> str:
    # Underscores are special in LaTeX; present a clean, escaped name.
    return name.replace("_", r"\_")


def order_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_p"] = df["precision"].map(lambda p: PRECISION_ORDER.index(p) if p in PRECISION_ORDER else 99)
    return df.sort_values(["model", "_p"]).drop(columns="_p")


def table_scores(df: pd.DataFrame) -> str:
    cols = [c for c in CAP_ORDER if c in df.columns]
    header = " & ".join(["Model", "Precision"] + [CAP_LABELS[c] for c in cols])
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
        header + r" \\", r"\midrule",
    ]
    prev_model = None
    for _, r in order_rows(df).iterrows():
        ml = model_label(r["model"]) if r["model"] != prev_model else ""
        prev_model = r["model"]
        cells = [ml, PRECISION_LABELS.get(r["precision"], r["precision"])] + [fmt(r[c]) for c in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Absolute benchmark scores (\%) per model, precision, and capability. "
        r"acc\_norm for ARC/HellaSwag, exact match for GSM8K/MGSM, pass@1 for HumanEval, "
        r"prompt-level strict accuracy for IFEval.}",
        r"\label{tab:results-main}", r"\end{table*}",
    ]
    return "\n".join(lines)


def table_degradation(df: pd.DataFrame) -> str:
    cols = [c for c in CAP_ORDER if c in df.columns]
    header = " & ".join(["Model", "Precision"] + [CAP_LABELS[c] for c in cols])
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
        header + r" \\", r"\midrule",
    ]
    prev_model = None
    for _, r in order_rows(df).iterrows():
        ml = model_label(r["model"]) if r["model"] != prev_model else ""
        prev_model = r["model"]
        cells = [ml, PRECISION_LABELS.get(r["precision"], r["precision"])] + [fmt(r[c]) for c in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Relative degradation (\%) from the FP16 baseline, "
        r"$(\text{FP16} - \text{quant}) / \text{FP16} \times 100$. "
        r"Higher = more degraded. Cells with an FP16 baseline below the floor "
        r"threshold are omitted (`--`) as unreliable.}",
        r"\label{tab:degradation}", r"\end{table*}",
    ]
    return "\n".join(lines)


def table_kendall(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{llrr}", r"\toprule",
        r"Precision & Model pair & $\tau$ & $p$ \\", r"\midrule",
    ]
    for _, r in df.iterrows():
        pair = f"{model_label(r['model_a'])} vs {model_label(r['model_b'])}"
        lines.append(f"{PRECISION_LABELS.get(r['precision'], r['precision'])} & {pair} & "
                     f"{r['kendall_tau']:.2f} & {r['p_value']:.3f}" + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Pairwise Kendall's $\tau$ between capability-degradation rankings. "
        r"High $\tau$ indicates a consistent degradation ordering across architectures.}",
        r"\label{tab:kendall}", r"\end{table}",
    ]
    return "\n".join(lines)


def table_size_axis(df: pd.DataFrame) -> str:
    """Within-family small-vs-large degradation-ranking agreement (size axis)."""
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{lllrr}", r"\toprule",
        r"Precision & Family & Sizes & $\tau$ & $p$ \\", r"\midrule",
    ]
    for _, r in df.iterrows():
        sm = r["model_small"].split("-")[-2] if "-" in r["model_small"] else r["model_small"]
        lg = r["model_large"].split("-")[-2] if "-" in r["model_large"] else r["model_large"]
        sizes = f"{sm}/{lg}"
        lines.append(f"{PRECISION_LABELS.get(r['precision'], r['precision'])} & "
                     f"{model_label(r['family'])} & {sizes} & "
                     f"{r['kendall_tau']:.2f} & {r['p_value']:.3f}" + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Size axis: Kendall's $\tau$ between the small and large variant's "
        r"capability-degradation ranking within a family. High $\tau$ indicates the "
        r"degradation ordering is stable as the model shrinks.}",
        r"\label{tab:size-axis}", r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for the paper")
    parser.add_argument("--results_csv", default="results/processed/results_table.csv")
    parser.add_argument("--analysis_dir", default="analysis/")
    parser.add_argument("--output_dir", default="paper/tables/")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    analysis_dir = Path(args.analysis_dir)

    scores = pd.read_csv(args.results_csv)
    (out / "results_main.tex").write_text(table_scores(scores), encoding="utf-8")
    logger.info(f"Wrote {out / 'results_main.tex'}")

    deg_path = analysis_dir / "degradation.csv"
    if deg_path.exists():
        (out / "degradation.tex").write_text(table_degradation(pd.read_csv(deg_path)), encoding="utf-8")
        logger.info(f"Wrote {out / 'degradation.tex'}")
    else:
        logger.warning(f"{deg_path} not found; run analyze_results.py first. Skipping degradation table.")

    tau_path = analysis_dir / "kendall_tau.csv"
    if tau_path.exists():
        tau_df = _safe_read_csv(tau_path)
        if tau_df is not None and not tau_df.empty:
            (out / "kendall_tau.tex").write_text(table_kendall(tau_df), encoding="utf-8")
            logger.info(f"Wrote {out / 'kendall_tau.tex'}")
        else:
            logger.warning("kendall_tau.csv is empty (need >=2 models at a precision). Skipping.")
    else:
        logger.warning(f"{tau_path} not found; run analyze_results.py first. Skipping tau table.")

    size_path = analysis_dir / "size_axis_tau.csv"
    if size_path.exists():
        size_df = _safe_read_csv(size_path)
        if size_df is not None and not size_df.empty:
            (out / "size_axis_tau.tex").write_text(table_size_axis(size_df), encoding="utf-8")
            logger.info(f"Wrote {out / 'size_axis_tau.tex'}")
        else:
            logger.warning("size_axis_tau.csv is empty (need a family at two sizes). Skipping.")
    else:
        logger.warning(f"{size_path} not found; run analyze_results.py first. Skipping size-axis table.")


if __name__ == "__main__":
    main()
