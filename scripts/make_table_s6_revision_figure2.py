#!/usr/bin/env python3
"""Regenerate Figure 2 from the audited repeat-specific-seed outputs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_cache = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(_cache / "nanozyme_mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache / "nanozyme_xdg_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "figures" / "publication" / "fig2_internal_ood_performance_table_s6_revision"
OUTPUT_S1 = ROOT / "outputs" / "figures" / "publication" / "fig3_catboost_vs_baseline_table_s6_revision"
SPLIT_ORDER = [
    "random",
    "source_group",
    "formula_group",
    "composition_cluster",
    "leakage_graph",
]
SPLIT_LABELS = {
    "random": "Random",
    "source_group": "Source",
    "formula_group": "Formula",
    "composition_cluster": "Composition cluster",
    "leakage_graph": "Component",
}
MODEL_LABELS = {
    "median": "Training median",
    "elasticnet": "ElasticNet",
    "extratrees": "ExtraTrees",
    "catboost": "CatBoost",
}
MODEL_COLORS = {
    "median": "#9c9c9c",
    "elasticnet": "#6f8fb3",
    "extratrees": "#7aa77a",
    "catboost": "#c2755b",
}


def audited_repeat_metrics() -> pd.DataFrame:
    primary = pd.read_csv(
        ROOT
        / "outputs"
        / "tables"
        / "editor_major_revision"
        / "m2_m3_design_only_per_repeat_metrics.csv"
    )
    catboost = primary[
        ["repeat", "seed", "split_type", "catboost_mae"]
    ].rename(columns={"catboost_mae": "mae"})
    catboost["model"] = "catboost"
    median = primary[["repeat", "seed", "split_type", "median_mae"]].rename(
        columns={"median_mae": "mae"}
    )
    median["model"] = "median"
    auxiliary = pd.read_csv(
        ROOT
        / "outputs"
        / "tables"
        / "reviewer3_controls"
        / "design_only_auxiliary_model_metrics.csv"
    )[["repeat", "seed", "split_type", "model", "mae"]]
    combined = pd.concat([catboost, median, auxiliary], ignore_index=True)
    counts = combined.groupby(["split_type", "model"])["repeat"].nunique()
    if not counts.eq(20).all() or len(counts) != 20:
        raise AssertionError(f"Incomplete repeat matrix:\n{counts}")
    expected_seed = 2026 + combined["repeat"].astype(int)
    if not combined["seed"].astype(int).eq(expected_seed).all():
        raise AssertionError("Figure input does not follow seed = 2026 + repeat")
    return combined


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["split_type", "model"], as_index=False)
        .agg(
            mean=("mae", "mean"),
            low=("mae", lambda values: values.quantile(0.025)),
            high=("mae", lambda values: values.quantile(0.975)),
        )
    )


def main() -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=1.05)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    summary = summarize(audited_repeat_metrics())
    summary["split_type"] = pd.Categorical(
        summary["split_type"], categories=SPLIT_ORDER, ordered=True
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.05), sharex=True)

    for model in ["median", "extratrees", "catboost"]:
        group = summary.loc[summary["model"].eq(model)].sort_values("split_type")
        x = np.arange(len(group))
        yerr = np.vstack([group["mean"] - group["low"], group["high"] - group["mean"]])
        axes[0].errorbar(
            x,
            group["mean"],
            yerr=yerr,
            marker="o",
            linewidth=1.4,
            capsize=3,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    axes[0].set_ylabel(r"MAE (log$_{10}$ units)")
    axes[0].set_title("A.", loc="left", fontweight="bold")
    axes[0].set_ylim(0.3, 1.25)
    axes[0].legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.36),
        handlelength=1.6,
        columnspacing=1.1,
    )

    elasticnet = summary.loc[summary["model"].eq("elasticnet")].sort_values(
        "split_type"
    )
    x = np.arange(len(elasticnet))
    yerr = np.vstack(
        [elasticnet["mean"] - elasticnet["low"], elasticnet["high"] - elasticnet["mean"]]
    )
    axes[1].errorbar(
        x,
        elasticnet["mean"],
        yerr=yerr,
        marker="o",
        linewidth=1.4,
        capsize=3,
        color=MODEL_COLORS["elasticnet"],
        label=MODEL_LABELS["elasticnet"],
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"MAE (log$_{10}$ units; log scale)")
    axes[1].set_title("B.", loc="left", fontweight="bold")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.set_xticks(np.arange(len(SPLIT_ORDER)))
        axis.set_xticklabels(
            [SPLIT_LABELS[split] for split in SPLIT_ORDER], rotation=22, ha="right"
        )
        axis.set_xlabel("")
    sns.despine(fig=fig)
    fig.subplots_adjust(bottom=0.28, wspace=0.28)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=600)
    fig.savefig(OUTPUT.with_suffix(".pdf"))
    fig.savefig(OUTPUT.with_suffix(".svg"))
    plt.close(fig)

    matrix = pd.read_csv(
        ROOT
        / "outputs"
        / "tables"
        / "reviewer3_controls"
        / "design_only_model_split_matrix.csv"
    )
    compact = matrix.loc[matrix["model"].isin(["catboost", "median"])].copy()
    compact["split_type"] = pd.Categorical(
        compact["split_type"], categories=SPLIT_ORDER, ordered=True
    )
    compact = compact.sort_values(["split_type", "model"])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.35), sharex=True)
    x = np.arange(len(SPLIT_ORDER))
    width = 0.36
    for axis, metric, ylabel in [
        (axes[0], "mae_mean", r"MAE (log$_{10}$ units)"),
        (axes[1], "r2_mean", r"$R^2$"),
    ]:
        for offset, model in [(-width / 2, "catboost"), (width / 2, "median")]:
            group = compact.loc[compact["model"].eq(model)].sort_values("split_type")
            axis.bar(
                x + offset,
                group[metric],
                width=width,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        axis.set_xticks(x)
        axis.set_xticklabels(
            [SPLIT_LABELS[split] for split in SPLIT_ORDER], rotation=25, ha="right"
        )
        axis.set_xlabel("")
        axis.set_ylabel(ylabel)
    axes[0].set_title("A.", loc="left", fontweight="bold")
    axes[1].set_title("B.", loc="left", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
    )
    sns.despine(fig=fig)
    fig.subplots_adjust(bottom=0.34, wspace=0.25)
    fig.savefig(OUTPUT_S1.with_suffix(".png"), dpi=600)
    fig.savefig(OUTPUT_S1.with_suffix(".pdf"))
    fig.savefig(OUTPUT_S1.with_suffix(".svg"))
    plt.close(fig)
    print(OUTPUT.with_suffix(".png"))
    print(OUTPUT_S1.with_suffix(".png"))


if __name__ == "__main__":
    main()
