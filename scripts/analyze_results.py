#!/usr/bin/env python3
"""
Analyze evaluation results: compute relative degradation, capability rankings,
and cross-model ranking consistency (Kendall's tau).

Usage:
    python analyze_results.py --results_dir results/processed/ --output_dir analysis/

Expects a CSV file (results_table.csv) with columns:
    model, precision, gsm8k, arc_challenge, hellaswag, humaneval, ifeval, mgsm

If you don't have this CSV yet, create it manually from your lm-eval JSON outputs
or use the --from_raw flag to attempt auto-aggregation from results/raw/.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CAPABILITY_LABELS = {
    "gsm8k": "Math",
    "arc_challenge": "Factual",
    "hellaswag": "Commonsense",
    "humaneval": "Code",
    "ifeval": "Instruction",
    "mgsm": "Multilingual",
}

CAPABILITIES = list(CAPABILITY_LABELS.keys())


def load_results(csv_path: str) -> pd.DataFrame:
    """Load the results table."""
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    return df


def compute_relative_degradation(df: pd.DataFrame, min_baseline: float = 10.0) -> pd.DataFrame:
    """
    Compute relative degradation for each model × precision × capability.
    Formula: (FP16_score - quant_score) / FP16_score × 100

    Validity guard: relative degradation is unstable when the FP16 baseline is
    near the floor (a tiny absolute drop becomes a huge relative number). Cells
    whose FP16 baseline is below `min_baseline` (in score points) are excluded
    (set to NaN) so they cannot drive the rankings, and are logged.
    """
    rows = []
    models = df["model"].unique()

    for model in models:
        model_df = df[df["model"] == model]
        fp16_row = model_df[model_df["precision"] == "fp16"]

        if fp16_row.empty:
            logger.warning(f"No FP16 baseline for {model}, skipping")
            continue

        fp16_scores = fp16_row.iloc[0]

        for _, row in model_df.iterrows():
            if row["precision"] == "fp16":
                continue

            result = {"model": model, "precision": row["precision"]}
            for cap in CAPABILITIES:
                if cap not in row or cap not in fp16_scores or pd.isna(fp16_scores[cap]) or pd.isna(row[cap]):
                    result[cap] = np.nan
                elif fp16_scores[cap] < min_baseline:
                    logger.warning(
                        f"Excluding {model}/{row['precision']}/{cap}: FP16 baseline "
                        f"{fp16_scores[cap]:.1f} < min_baseline {min_baseline} "
                        f"(relative degradation would be unreliable)."
                    )
                    result[cap] = np.nan
                else:
                    degradation = (fp16_scores[cap] - row[cap]) / fp16_scores[cap] * 100
                    result[cap] = round(degradation, 2)
            rows.append(result)

    return pd.DataFrame(rows)


def compute_absolute_drop(df: pd.DataFrame) -> pd.DataFrame:
    """
    Absolute score drop (FP16_score - quant_score) in score points, reported
    alongside relative degradation. Robust to small baselines, so it serves as a
    sanity check that the relative-degradation rankings are not floor artifacts.
    """
    rows = []
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        fp16_row = model_df[model_df["precision"] == "fp16"]
        if fp16_row.empty:
            continue
        fp16_scores = fp16_row.iloc[0]
        for _, row in model_df.iterrows():
            if row["precision"] == "fp16":
                continue
            result = {"model": model, "precision": row["precision"]}
            for cap in CAPABILITIES:
                if cap in row and cap in fp16_scores and not pd.isna(fp16_scores[cap]) and not pd.isna(row[cap]):
                    result[cap] = round(fp16_scores[cap] - row[cap], 2)
                else:
                    result[cap] = np.nan
            rows.append(result)
    return pd.DataFrame(rows)


def compute_rankings(degradation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank capabilities by degradation for each model × precision.
    Rank 1 = most degraded capability.
    """
    rows = []
    for _, row in degradation_df.iterrows():
        caps_available = [c for c in CAPABILITIES if c in row and not pd.isna(row[c])]
        scores = {c: row[c] for c in caps_available}
        sorted_caps = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        ranking = {c: rank + 1 for rank, c in enumerate(sorted_caps)}

        result = {"model": row["model"], "precision": row["precision"]}
        for c in CAPABILITIES:
            result[c] = ranking.get(c, np.nan)
        rows.append(result)

    return pd.DataFrame(rows)


def compute_kendall_tau(ranking_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Kendall's tau between all model rankings at same precision.
    High tau = consistent degradation ordering across models.
    """
    results = []
    precisions = ranking_df["precision"].unique()

    for prec in precisions:
        prec_df = ranking_df[ranking_df["precision"] == prec]
        models = prec_df["model"].unique()

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                r1 = prec_df[prec_df["model"] == models[i]].iloc[0]
                r2 = prec_df[prec_df["model"] == models[j]].iloc[0]

                caps = [c for c in CAPABILITIES if not pd.isna(r1[c]) and not pd.isna(r2[c])]
                if len(caps) < 3:
                    continue

                tau, p_value = kendalltau(
                    [r1[c] for c in caps],
                    [r2[c] for c in caps]
                )
                results.append({
                    "precision": prec,
                    "model_a": models[i],
                    "model_b": models[j],
                    "kendall_tau": round(tau, 3),
                    "p_value": round(p_value, 4),
                    "n_capabilities": len(caps),
                })

    return pd.DataFrame(results)


def parse_family_size(model: str):
    """Split a normalized model name into (family_key, size_in_billions).

    'qwen2.5-3b-instruct'  -> ('qwen2.5-instruct', 3.0)
    'llama-3.2-1b-instruct'-> ('llama-3.2-instruct', 1.0)
    Returns (model, None) if no size token is found.
    """
    import re
    m = re.search(r"(\d+\.?\d*)b\b", model)
    if not m:
        return model, None
    size = float(m.group(1))
    family = (model[:m.start()] + model[m.end():]).replace("--", "-").strip("-")
    return family, size


def compute_size_axis_tau(ranking_df: pd.DataFrame) -> pd.DataFrame:
    """
    Within each model family present at two sizes (e.g. Qwen2.5 1.5B vs 3B),
    compute Kendall's tau between the small and large model's degradation
    ranking, per precision. Tests whether the degradation ORDER shifts as a
    model shrinks (contribution: the size axis).
    """
    df = ranking_df.copy()
    fam_size = df["model"].map(parse_family_size)
    df["family"] = [fs[0] for fs in fam_size]
    df["size_b"] = [fs[1] for fs in fam_size]
    df = df[df["size_b"].notna()]

    results = []
    for prec in df["precision"].unique():
        prec_df = df[df["precision"] == prec]
        for family, fam_df in prec_df.groupby("family"):
            sizes = sorted(fam_df["size_b"].unique())
            if len(sizes) < 2:
                continue
            small = fam_df[fam_df["size_b"] == sizes[0]].iloc[0]
            large = fam_df[fam_df["size_b"] == sizes[-1]].iloc[0]
            caps = [c for c in CAPABILITIES if not pd.isna(small[c]) and not pd.isna(large[c])]
            if len(caps) < 3:
                continue
            tau, p_value = kendalltau([small[c] for c in caps], [large[c] for c in caps])
            results.append({
                "precision": prec,
                "family": family,
                "model_small": small["model"],
                "model_large": large["model"],
                "kendall_tau": round(tau, 3),
                "p_value": round(p_value, 4),
                "n_capabilities": len(caps),
            })

    return pd.DataFrame(results)


def plot_degradation_heatmap(degradation_df: pd.DataFrame, output_path: str):
    """Create the main heatmap figure for the paper."""
    plot_df = degradation_df.copy()
    plot_df["label"] = plot_df["model"].str.split("/").str[-1] + " (" + plot_df["precision"] + ")"
    plot_df = plot_df.set_index("label")

    cap_cols = [c for c in CAPABILITIES if c in plot_df.columns]
    plot_data = plot_df[cap_cols].rename(columns=CAPABILITY_LABELS)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        plot_data,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Relative Degradation (%)"},
    )
    ax.set_title("Capability-Specific Degradation Under Quantization", fontsize=14, pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("Capability", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Heatmap saved to {output_path}")
    plt.close()


def plot_ranking_comparison(ranking_df: pd.DataFrame, output_path: str):
    """Visualize ranking consistency across models."""
    fig, axes = plt.subplots(1, len(ranking_df["precision"].unique()), figsize=(14, 5), sharey=True)
    if not hasattr(axes, "__len__"):
        axes = [axes]

    for ax, prec in zip(axes, ranking_df["precision"].unique()):
        prec_df = ranking_df[ranking_df["precision"] == prec]
        prec_df = prec_df.copy()
        prec_df["label"] = prec_df["model"].str.split("/").str[-1]
        prec_df = prec_df.set_index("label")

        cap_cols = [c for c in CAPABILITIES if c in prec_df.columns]
        plot_data = prec_df[cap_cols].rename(columns=CAPABILITY_LABELS)

        sns.heatmap(
            plot_data,
            annot=True,
            fmt=".0f",
            cmap="Blues_r",
            linewidths=0.5,
            ax=ax,
            vmin=1,
            vmax=len(cap_cols),
            cbar=False,
        )
        ax.set_title(f"Rankings at {prec.upper()}", fontsize=12)
        ax.set_ylabel("")

    plt.suptitle("Degradation Rankings (1 = Most Degraded)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Ranking plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze quantization degradation results")
    parser.add_argument("--results_csv", required=True, help="Path to results CSV")
    parser.add_argument("--output_dir", default="analysis/", help="Output directory")
    parser.add_argument("--min_baseline", type=float, default=10.0,
                        help="Exclude capabilities whose FP16 baseline (in score points) "
                             "is below this, since relative degradation is unreliable there")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Load and analyze
    df = load_results(args.results_csv)

    logger.info("Computing relative degradation...")
    deg_df = compute_relative_degradation(df, min_baseline=args.min_baseline)
    deg_df.to_csv(output_dir / "degradation.csv", index=False)
    logger.info(f"Degradation table (relative %):\n{deg_df.to_string()}")

    logger.info("Computing absolute score drop (sanity check)...")
    abs_df = compute_absolute_drop(df)
    abs_df.to_csv(output_dir / "absolute_drop.csv", index=False)
    logger.info(f"Absolute drop table (score points):\n{abs_df.to_string()}")

    logger.info("Computing capability rankings...")
    rank_df = compute_rankings(deg_df)
    rank_df.to_csv(output_dir / "rankings.csv", index=False)
    logger.info(f"Rankings:\n{rank_df.to_string()}")

    logger.info("Computing Kendall's tau...")
    tau_df = compute_kendall_tau(rank_df)
    tau_df.to_csv(output_dir / "kendall_tau.csv", index=False)
    logger.info(f"Kendall's tau:\n{tau_df.to_string()}")

    # Interpretation
    if not tau_df.empty:
        avg_tau = tau_df["kendall_tau"].mean()
        logger.info(f"\nAverage Kendall's tau: {avg_tau:.3f}")
        if avg_tau > 0.6:
            logger.info("-> HIGH consistency: degradation ordering is similar across models")
        elif avg_tau > 0.3:
            logger.info("-> MODERATE consistency: some shared patterns, some model-specific")
        else:
            logger.info("-> LOW consistency: degradation ordering is architecture-dependent")

    logger.info("Computing size-axis tau (within-family, small vs large)...")
    size_tau_df = compute_size_axis_tau(rank_df)
    size_tau_df.to_csv(output_dir / "size_axis_tau.csv", index=False)
    if size_tau_df.empty:
        logger.info("No family present at two sizes yet; size-axis analysis skipped.")
    else:
        logger.info(f"Size-axis tau:\n{size_tau_df.to_string()}")

    # Generate figures
    logger.info("Generating figures...")
    plot_degradation_heatmap(deg_df, str(figures_dir / "degradation_heatmap.png"))
    plot_ranking_comparison(rank_df, str(figures_dir / "ranking_comparison.png"))

    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
