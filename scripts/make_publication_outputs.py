#!/usr/bin/env python
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

_tmp_cache = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(_tmp_cache / "nanozyme_mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(_tmp_cache / "nanozyme_xdg_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import prepare_modeling_data, read_table
from src.diagnostics import PROTOCOL_CATEGORICAL_COLUMNS, PROTOCOL_NUMERIC_COLUMNS
from src.features import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, build_feature_frame
from src.reporting import ensure_output_dirs


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
MODEL_ORDER = ["median", "elasticnet", "extratrees", "catboost"]
MODEL_LABELS = {
    "median": "Median",
    "elasticnet": "Elastic net",
    "extratrees": "ExtraTrees",
    "catboost": "CatBoost",
}
MODEL_COLORS = {
    "median": "#9c9c9c",
    "elasticnet": "#6f8fb3",
    "extratrees": "#7aa77a",
    "catboost": "#c2755b",
}


def main() -> None:
    dirs = ensure_output_dirs()
    fig_dir = dirs["figures"] / "publication"
    table_dir = dirs["tables"] / "publication"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

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

    benchmark = pd.read_csv(dirs["metrics"] / "benchmark_metrics.csv")
    conformal = pd.read_csv(dirs["metrics"] / "conformal_metrics.csv")
    modeling, _ = prepare_modeling_data(read_table("data/nanozymes.csv"))

    write_publication_tables(dirs["tables"], table_dir)
    build_external_collapse_mechanism_table(
        modeling,
        dirs["tables"] / "external_validation" / "external_validation_strict_model_ready.csv",
        dirs["tables"] / "external_validation" / "external_validation_strict_predictions.csv",
        dirs["tables"] / "internal_validation" / "internal_row_level_catboost_predictions.csv",
        dirs["split_manifests"],
        table_dir,
    )
    plot_dataset_structure(modeling, dirs["tables"] / "audit", fig_dir)
    plot_internal_performance(benchmark, fig_dir)
    plot_compact_model_comparison(dirs["tables"] / "main_table_3_compact_performance.csv", fig_dir)
    plot_conformal_baseline(conformal, fig_dir)
    plot_few_shot_recalibration(dirs["tables"] / "main_table_4_conformal_recalibration.csv", fig_dir)
    copy_external_figures(dirs["figures"] / "external_validation", fig_dir)
    plot_delta_mae_reversal(
        dirs["metrics"] / "benchmark_metrics.csv",
        dirs["tables"] / "external_validation" / "external_validation_strict_predictions.csv",
        fig_dir,
    )
    plot_severity_error_diagnostics(
        dirs["tables"] / "supp_table_s20_ood_severity_diagnostics.csv",
        dirs["metrics"] / "benchmark_metrics.csv",
        dirs["tables"] / "internal_validation" / "internal_row_level_catboost_predictions.csv",
        dirs["tables"] / "external_validation" / "external_validation_strict_predictions.csv",
        fig_dir,
    )
    write_manifests(dirs, fig_dir, table_dir)

    print(f"Wrote publication figures to {fig_dir}")
    print(f"Wrote publication tables to {table_dir}")


def write_publication_tables(source_table_dir: Path, table_dir: Path) -> None:
    table_map = {
        "main_table_1_modeling_cohort.csv": "table1_modeling_cohort.csv",
        "main_table_2_split_definitions.csv": "table2_split_definitions.csv",
        "main_table_3_compact_performance.csv": "table3_internal_performance.csv",
        "main_table_4_conformal_recalibration.csv": "table4_conformal_recalibration.csv",
        "main_table_5_reporting_checklist.csv": "table5_reporting_checklist.csv",
        "external_validation/external_validation_publication_summary_table.csv": "table6_external_summary.csv",
        "external_validation/external_validation_publication_metrics_table.csv": "table7_external_metrics.csv",
        "external_validation/external_validation_publication_failure_cases_catboost.csv": "table8_external_failure_cases.csv",
        "supp_table_s22_ood_severity_performance_association.csv": "table9_internal_severity_association.csv",
        "external_validation/external_severity_error_association.csv": "table10_external_severity_association.csv",
        "internal_validation/internal_row_level_severity_error_association.csv": "table12_internal_row_level_severity_association.csv",
    }
    for src_rel, dst_name in table_map.items():
        src = source_table_dir / src_rel
        if src.exists():
            shutil.copyfile(src, table_dir / dst_name)

    performance = pd.read_csv(source_table_dir / "main_table_3_compact_performance.csv")
    performance["split_label"] = performance["split_type"].map(SPLIT_LABELS)
    performance["model_label"] = performance["model"].map(MODEL_LABELS)
    ordered = [
        "split_label",
        "model_label",
        "n_repeats",
        "mae",
        "mae_ci95_low",
        "mae_ci95_high",
        "r2",
        "r2_ci95_low",
        "r2_ci95_high",
        "spearman",
        "spearman_ci95_low",
        "spearman_ci95_high",
    ]
    performance[ordered].round(4).to_csv(
        table_dir / "table3_internal_performance_rounded.csv", index=False
    )

    conformal = pd.read_csv(source_table_dir / "main_table_4_conformal_recalibration.csv")
    conformal["split_label"] = conformal["split_type"].map(SPLIT_LABELS)
    conformal["method_label"] = conformal["method"].str.replace("_", " ", regex=False).str.title()
    conformal_out = conformal[
        [
            "split_label",
            "method_label",
            "shots",
            "n_repeats",
            "empirical_coverage",
            "coverage_gap",
            "mean_interval_width",
            "interval_score",
        ]
    ].round(4)
    conformal_out.to_csv(table_dir / "table4_conformal_recalibration_rounded.csv", index=False)


def _formula_for_display(formula: str) -> str:
    """Return a publication-facing formula label without changing stored formula keys."""
    import re

    display = {
        "O5V2": "V2O5",
        "Fe2O4Zn": "ZnFe2O4",
    }.get(str(formula), str(formula))
    return re.sub(r"([A-Z][a-z]?)(\d+)", r"\1$_{\2}$", display)


def _reaction_for_display(reaction: str) -> str:
    display = str(reaction)
    display = display.replace("TMB + H2O2", "TMB varied\n($H_2O_2$ fixed)")
    display = display.replace("H2O2 + TMB", "$H_2O_2$ varied\n(TMB fixed)")
    display = display.replace("H2O2", "$H_2O_2$")
    return display


def _label_horizontal_bars(ax: plt.Axes, values: pd.Series, pad_fraction: float = 0.015) -> None:
    xmax = float(values.max())
    ax.set_xlim(0, xmax * 1.12)
    for patch, value in zip(ax.patches, values):
        ax.text(
            patch.get_width() + xmax * pad_fraction,
            patch.get_y() + patch.get_height() / 2,
            f"{int(value)}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#333333",
        )


def plot_dataset_structure(modeling: pd.DataFrame, audit_dir: Path, fig_dir: Path) -> None:
    source = pd.read_csv(audit_dir / "source_frequency.csv").head(12)
    formula = pd.read_csv(audit_dir / "formula_frequency.csv").head(12)
    reaction = pd.read_csv(audit_dir / "reaction_distribution.csv").head(8)
    source_all = pd.read_csv(audit_dir / "source_frequency.csv")
    source_hist = source_all["n_rows"].value_counts().sort_index().reset_index()
    source_hist.columns = ["rows_per_source", "n_sources"]
    formula["formula_display"] = formula["standardized_formula"].map(_formula_for_display)
    reaction["reaction_display"] = reaction["reaction_type"].map(_reaction_for_display)
    modeling = modeling.copy()
    modeling["log10_km_M"] = modeling["target_log10_km"].astype(float)
    if "substrate_for_km" in modeling.columns:
        modeling["substrate_label"] = modeling["substrate_for_km"].fillna("Unspecified")
    else:
        modeling["substrate_label"] = "All retained rows"

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2))
    sns.barplot(data=source_hist, x="rows_per_source", y="n_sources", ax=axes[0, 0], color="#6f8fb3")
    axes[0, 0].set_title("A. Source-size distribution")
    axes[0, 0].set_xlabel("Rows per source")
    axes[0, 0].set_ylabel("Number of sources")

    sns.barplot(data=formula, x="n_rows", y="formula_display", ax=axes[0, 1], color="#7aa77a")
    axes[0, 1].set_title("B. Most frequent formulas")
    axes[0, 1].set_xlabel("Rows")
    axes[0, 1].set_ylabel("")
    _label_horizontal_bars(axes[0, 1], formula["n_rows"])

    sns.barplot(data=reaction, x="n_rows", y="reaction_display", ax=axes[1, 0], color="#c2755b")
    axes[1, 0].set_title("C. Reaction systems")
    axes[1, 0].set_xlabel("Rows")
    axes[1, 0].set_ylabel("")
    _label_horizontal_bars(axes[1, 0], reaction["n_rows"])

    if modeling["substrate_label"].nunique() > 1:
        sns.histplot(
            data=modeling,
            x="log10_km_M",
            hue="substrate_label",
            bins=28,
            element="step",
            stat="count",
            common_norm=False,
            ax=axes[1, 1],
        )
    else:
        sns.histplot(
            data=modeling,
            x="log10_km_M",
            bins=28,
            element="step",
            color="#6f8fb3",
            ax=axes[1, 1],
        )
    axes[1, 1].set_title("D. Target distribution")
    axes[1, 1].set_xlabel(r"$\log_{10}(K_m / \mathrm{M})$")
    axes[1, 1].set_ylabel("Rows")
    sns.despine(fig=fig)
    save_figure(fig, fig_dir / "fig1_dataset_structure")


def plot_internal_performance(benchmark: pd.DataFrame, fig_dir: Path) -> None:
    summary = summarize(benchmark, ["split_type", "model"], "mae")
    summary["split_label"] = pd.Categorical(
        summary["split_type"].map(SPLIT_LABELS),
        categories=[SPLIT_LABELS[s] for s in SPLIT_ORDER],
        ordered=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.05), sharex=True)
    ax = axes[0]
    for model in ["median", "extratrees", "catboost"]:
        group = summary.loc[summary["model"].eq(model)].sort_values("split_label")
        x = np.arange(len(group))
        yerr = np.vstack([group["mean"] - group["low"], group["high"] - group["mean"]])
        ax.errorbar(
            x,
            group["mean"],
            yerr=yerr,
            marker="o",
            linewidth=1.4,
            capsize=3,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    ax.set_xticks(np.arange(len(SPLIT_ORDER)))
    ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLIT_ORDER], rotation=22, ha="right")
    ax.set_ylabel(r"MAE (log$_{10}$ units)")
    ax.set_xlabel("")
    ax.set_title("A.", loc="left", fontweight="bold")
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.36),
        handlelength=1.6,
        columnspacing=1.1,
    )
    ax.set_ylim(0.3, 1.25)

    ax = axes[1]
    group = summary.loc[summary["model"].eq("elasticnet")].sort_values("split_label")
    x = np.arange(len(group))
    yerr = np.vstack([group["mean"] - group["low"], group["high"] - group["mean"]])
    ax.errorbar(
        x,
        group["mean"],
        yerr=yerr,
        marker="o",
        linewidth=1.4,
        capsize=3,
        color=MODEL_COLORS["elasticnet"],
        label=MODEL_LABELS["elasticnet"],
    )
    ax.set_xticks(np.arange(len(SPLIT_ORDER)))
    ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLIT_ORDER], rotation=22, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel(r"MAE (log$_{10}$ units; log scale)")
    ax.set_xlabel("")
    ax.set_title("B.", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    sns.despine(fig=fig)
    fig.subplots_adjust(bottom=0.28, wspace=0.28)
    save_figure(fig, fig_dir / "fig2_internal_ood_performance")


def plot_compact_model_comparison(table_path: Path, fig_dir: Path) -> None:
    compact = pd.read_csv(table_path).copy()
    compact = compact.loc[compact["model"].isin(["median", "catboost"])]
    compact["split_label"] = pd.Categorical(
        compact["split_type"].map(SPLIT_LABELS),
        categories=[SPLIT_LABELS[s] for s in SPLIT_ORDER],
        ordered=True,
    )
    compact["model_label"] = compact["model"].map(MODEL_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.35), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "mae", r"MAE (log$_{10}$ units)"),
        (axes[1], "r2", r"$R^2$"),
    ]:
        sns.barplot(
            data=compact,
            x="split_label",
            y=metric,
            hue="model_label",
            palette=[MODEL_COLORS["catboost"], MODEL_COLORS["median"]],
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].set_title("A.", loc="left", fontweight="bold")
    axes[1].set_title("B.", loc="left", fontweight="bold")
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        columnspacing=1.4,
        handlelength=1.7,
    )
    sns.despine(fig=fig)
    fig.subplots_adjust(bottom=0.34, wspace=0.25)
    save_figure(fig, fig_dir / "fig3_catboost_vs_baseline")


def plot_conformal_baseline(conformal: pd.DataFrame, fig_dir: Path) -> None:
    base = conformal.loc[conformal["shots"].eq(0)].copy()
    base["method_label"] = base["method"].map(
        {
            "absolute_residual": "Absolute residual",
            "residual_normalized": "Residual-normalized",
        }
    )
    cov = summarize(base, ["split_type", "method_label"], "empirical_coverage")
    width = summarize(base, ["split_type", "method_label"], "mean_interval_width")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.95), sharex=True)
    for ax, data, metric, ylabel in [
        (axes[0], cov, "empirical_coverage", "Empirical coverage"),
        (axes[1], width, "mean_interval_width", r"Mean interval width ($\log_{10}$ units)"),
    ]:
        data = data.copy()
        data["split_label"] = pd.Categorical(
            data["split_type"].map(SPLIT_LABELS),
            categories=[SPLIT_LABELS[s] for s in SPLIT_ORDER],
            ordered=True,
        )
        sns.pointplot(
            data=data,
            x="split_label",
            y="mean",
            hue="method_label",
            errorbar=None,
            dodge=0.25,
            markers=["o", "s"],
            linestyles=["-", "--"],
            ax=ax,
        )
        if metric == "empirical_coverage":
            ax.axhline(0.90, color="0.25", lw=1.0, ls="--")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].set_title("A.", loc="left", fontweight="bold")
    axes[1].set_title("B.", loc="left", fontweight="bold")
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        columnspacing=1.4,
        handlelength=1.8,
    )
    sns.despine(fig=fig)
    fig.subplots_adjust(bottom=0.32, wspace=0.28)
    save_figure(fig, fig_dir / "fig4_conformal_baseline")


def plot_few_shot_recalibration(table_path: Path, fig_dir: Path) -> None:
    def sem(values: pd.Series) -> float:
        valid = pd.Series(values).dropna().astype(float)
        return float(valid.std(ddof=1) / np.sqrt(len(valid))) if len(valid) > 1 else 0.0

    metrics_path = table_path.parent.parent / "metrics" / "conformal_metrics.csv"
    if metrics_path.exists():
        raw = pd.read_csv(metrics_path)
        raw = raw.loc[~raw["underpowered"].astype(bool)].copy()
        repeat_level = (
            raw.groupby(["split_type", "method", "shots", "repeat"], as_index=False)
            .agg(
                coverage_gap=("coverage_gap", "mean"),
                mean_interval_width=("mean_interval_width", "mean"),
            )
        )
        conf = (
            repeat_level.groupby(["split_type", "method", "shots"], as_index=False)
            .agg(
                coverage_gap=("coverage_gap", "mean"),
                coverage_gap_sem=("coverage_gap", sem),
                mean_interval_width=("mean_interval_width", "mean"),
                mean_interval_width_sem=("mean_interval_width", sem),
                n_valid_outer=("repeat", "nunique"),
            )
        )
    else:
        conf = pd.read_csv(table_path).copy()
        conf["coverage_gap_sem"] = 0.0
        conf["mean_interval_width_sem"] = 0.0
        conf["n_valid_outer"] = conf.get("n_repeats", np.nan)

    palette = dict(zip(SPLIT_ORDER, sns.color_palette("deep", n_colors=len(SPLIT_ORDER))))
    x_positions = {0: 0.0, 5: 1.0, 10: 2.0, 20: 3.0}
    finite_sample_targets = {10: 10 / 11, 20: 19 / 21}
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.9), sharex=True)
    handles, labels = [], []

    panels = [
        (axes[0, 0], "absolute_residual", "coverage_gap", "coverage_gap_sem", "A. Absolute residual: coverage"),
        (axes[0, 1], "residual_normalized", "coverage_gap", "coverage_gap_sem", "B. Residual-normalized: coverage"),
        (
            axes[1, 0],
            "absolute_residual",
            "mean_interval_width",
            "mean_interval_width_sem",
            "C. Absolute residual: width",
        ),
        (
            axes[1, 1],
            "residual_normalized",
            "mean_interval_width",
            "mean_interval_width_sem",
            "D. Residual-normalized: width",
        ),
    ]
    for ax, method, metric, metric_sem, panel_title in panels:
        data = conf.loc[conf["method"].eq(method)].copy()
        for split_type in SPLIT_ORDER:
            split_data = data.loc[data["split_type"].eq(split_type)].copy()
            if split_data.empty:
                continue
            color = palette[split_type]
            ref = split_data.loc[split_data["shots"].eq(0)]
            if not ref.empty:
                ax.errorbar(
                    [x_positions[0]],
                    ref[metric],
                    yerr=ref[metric_sem],
                    fmt="o",
                    color=color,
                    ms=4.0,
                    capsize=2.0,
                    elinewidth=0.8,
                    alpha=0.95,
                )
            recal = split_data.loc[split_data["shots"].isin([10, 20])].copy()
            if not recal.empty:
                recal = recal.sort_values("shots")
                x_vals = recal["shots"].map(x_positions).to_numpy(dtype=float)
                y_vals = recal[metric].to_numpy(dtype=float)
                y_err = recal[metric_sem].to_numpy(dtype=float)
                (line,) = ax.plot(
                    x_vals,
                    y_vals,
                    "-o",
                    color=color,
                    lw=1.15,
                    ms=4.0,
                    alpha=0.95,
                )
                ax.errorbar(
                    x_vals,
                    y_vals,
                    yerr=y_err,
                    fmt="none",
                    ecolor=color,
                    capsize=2.0,
                    elinewidth=0.75,
                    alpha=0.8,
                )
                if ax is axes[0, 0]:
                    handles.append(line)
                    labels.append(SPLIT_LABELS[split_type])

        if metric == "coverage_gap":
            ax.axhline(0, color="0.25", lw=1.0, ls="--")
            for shots, target_coverage in finite_sample_targets.items():
                target_gap = target_coverage - 0.90
                center = x_positions[shots]
                ax.hlines(
                    target_gap,
                    center - 0.18,
                    center + 0.18,
                    color="0.2",
                    lw=1.15,
                    linestyles=":",
                    zorder=1,
                )
            ax.set_ylim(-0.08, 0.03)
            ax.text(
                x_positions[5],
                -0.075,
                "rank 6 > n = 5\nno finite 90% radius",
                ha="center",
                va="bottom",
                fontsize=6.3,
                color="0.35",
            )
        else:
            ax.set_ylim(bottom=2.0)

        ax.set_title(panel_title, loc="left", fontweight="bold")
        ax.set_xticks([x_positions[x] for x in [0, 5, 10, 20]])
        ax.set_xticklabels(["0\nlegacy", "5\nNA", "10", "20"])
        ax.axvspan(0.86, 1.14, color="0.92", zorder=-10)
        ax.set_xlim(-0.25, 3.25)

    axes[0, 0].set_ylabel("Coverage gap\n(empirical coverage − 0.900)")
    axes[1, 0].set_ylabel(r"Mean interval width (log$_{10}$ units)")
    for ax in axes[1, :]:
        ax.set_xlabel("Labeled target-domain calibration records", fontsize=8.5)

    if handles and labels:
        handles.append(Line2D([0], [0], color="0.2", lw=1.15, ls=":"))
        labels.append("Finite-sample target")
        fig.legend(
            handles,
            labels,
            frameon=False,
            ncol=1,
            loc="center left",
            bbox_to_anchor=(0.835, 0.56),
            labelspacing=0.65,
            handlelength=1.4,
            fontsize=7.5,
        )

    sns.despine(fig=fig)
    save_figure(
        fig,
        fig_dir / "fig5_few_shot_recalibration",
        tight_layout_rect=(0.0, 0.0, 0.82, 1.0),
    )


def copy_external_figures(external_dir: Path, publication_dir: Path) -> None:
    mapping = {
        "fig_external_predicted_vs_observed_catboost": "fig6_external_predicted_vs_observed_catboost",
        "fig_external_error_by_substrate": "fig7_external_error_by_substrate",
        "fig_external_scenario_metrics": "figS1_external_scenario_metrics",
    }
    for src_stem, dst_stem in mapping.items():
        for suffix in [".png", ".pdf", ".svg"]:
            src = external_dir / f"{src_stem}{suffix}"
            if src.exists():
                shutil.copyfile(src, publication_dir / f"{dst_stem}{suffix}")


def plot_delta_mae_reversal(
    benchmark_metrics_path: Path,
    external_predictions_path: Path,
    fig_dir: Path,
) -> None:
    benchmark = pd.read_csv(benchmark_metrics_path)
    rows = []
    for split in SPLIT_ORDER:
        sub = benchmark.loc[benchmark["split_type"].eq(split)]
        paired = sub.pivot_table(index="repeat", columns="model", values="mae", aggfunc="first")
        if {"catboost", "median"}.issubset(paired.columns):
            diff = (paired["catboost"] - paired["median"]).dropna().to_numpy(dtype=float)
            rows.append(delta_summary_row("Internal split benchmark", SPLIT_LABELS[split], diff, len(diff)))

    predictions = pd.read_csv(external_predictions_path)
    ext_scenarios = [
        ("all_model_ready", "All curated"),
        ("strict_external_unflagged_only", "Unflagged"),
        ("central_95pct_km", "Central 95%"),
        ("no_hard_km_extremes", "No hard extremes"),
    ]
    for scenario, label in ext_scenarios:
        sub = apply_external_delta_scenario(predictions, scenario)
        paired = sub.pivot_table(index="row_id", columns="model", values="abs_error_log10_km", aggfunc="first")
        if {"catboost", "median"}.issubset(paired.columns):
            diff = (paired["catboost"] - paired["median"]).dropna().to_numpy(dtype=float)
            rows.append(delta_summary_row("External validation", f"{label}\n(n={len(diff)})", diff, len(diff)))

    delta = pd.DataFrame(rows)
    delta.to_csv(fig_dir.parent.parent / "tables" / "publication" / "table11_delta_mae_reversal.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    for ax, domain in zip(axes, ["Internal split benchmark", "External validation"]):
        data = delta.loc[delta["domain"].eq(domain)]
        colors = ["#c2755b" if v > 0 else "#4f8a5b" for v in data["delta_mae"]]
        x = np.arange(len(data))
        yerr = np.vstack(
            [
                data["delta_mae"] - data["delta_mae_ci_low"],
                data["delta_mae_ci_high"] - data["delta_mae"],
            ]
        )
        ax.bar(x, data["delta_mae"], color=colors)
        ax.errorbar(x, data["delta_mae"], yerr=yerr, fmt="none", ecolor="0.25", elinewidth=0.8, capsize=2.5)
        ax.axhline(0, color="0.25", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(data["scenario"], rotation=28, ha="right")
        ax.set_title("A. Internal" if domain.startswith("Internal") else "B. External")
        ax.set_ylabel(r"$\Delta$MAE CatBoost - Median (log$_{10}$ units)" if ax is axes[0] else "")
        ax.set_ylim(-0.55, 0.16)
        for idx, value in enumerate(data["delta_mae"]):
            ci_high = float(data["delta_mae_ci_high"].iloc[idx])
            y = value + 0.028 if value < 0 else ci_high + 0.015
            ax.text(
                idx,
                y,
                f"{value:+.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    axes[0].text(
        0.02,
        0.96,
        "Negative: CatBoost better",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="0.25",
    )
    axes[1].text(
        0.02,
        0.68,
        "Positive: median baseline lower error",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="0.25",
    )
    sns.despine(fig=fig)
    save_figure(fig, fig_dir / "fig8_delta_mae_internal_external")


def delta_summary_row(domain: str, scenario: str, diff: np.ndarray, n: int) -> dict:
    low, high = bootstrap_mean_ci(diff)
    p_value = np.nan
    if len(diff) > 1 and np.nanstd(diff) > 0:
        try:
            p_value = float(wilcoxon(diff).pvalue)
        except ValueError:
            p_value = np.nan
    return {
        "domain": domain,
        "scenario": scenario,
        "delta_mae": float(np.nanmean(diff)),
        "delta_mae_ci_low": low,
        "delta_mae_ci_high": high,
        "n": int(n),
        "wilcoxon_p": p_value,
        "interpretation": "positive values indicate lower median-baseline absolute error than CatBoost",
    }


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 2026) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def apply_external_delta_scenario(predictions: pd.DataFrame, scenario: str) -> pd.DataFrame:
    base = predictions.copy()
    row_meta = base.drop_duplicates("row_id").set_index("row_id")
    if scenario in {"all_model_ready", "strict_external"}:
        keep_ids = row_meta.index
    elif scenario == "strict_external_unflagged_only":
        keep_ids = row_meta.index[~row_meta["retain_with_flag"].astype(bool)]
    elif scenario == "central_95pct_km":
        low, high = row_meta["km_millimolar"].quantile([0.025, 0.975])
        keep_ids = row_meta.index[row_meta["km_millimolar"].between(low, high, inclusive="both")]
    elif scenario == "no_hard_km_extremes":
        keep_ids = row_meta.index[row_meta["km_millimolar"].between(0.001, 100, inclusive="both")]
    else:
        keep_ids = row_meta.index
    return base.loc[base["row_id"].isin(keep_ids)]


def plot_severity_error_diagnostics(
    severity_path: Path,
    benchmark_path: Path,
    internal_row_predictions_path: Path,
    external_predictions_path: Path,
    fig_dir: Path,
) -> None:
    severity = pd.read_csv(severity_path)
    benchmark = pd.read_csv(benchmark_path)
    cat = benchmark.loc[benchmark["model"].eq("catboost")].merge(
        severity[["repeat", "split_type", "ood_severity_index"]],
        on=["repeat", "split_type"],
        how="left",
    )
    split_means = (
        cat.groupby("split_type", as_index=False)
        .agg(ood_severity_index=("ood_severity_index", "mean"), mae=("mae", "mean"))
        .sort_values("ood_severity_index")
    )
    split_association = spearmanr(
        cat["ood_severity_index"],
        cat["mae"],
        nan_policy="omit",
    )
    split_p_exponent = int(np.floor(np.log10(split_association.pvalue)))
    split_p_mantissa = split_association.pvalue / (10**split_p_exponent)
    internal_rows = pd.read_csv(internal_row_predictions_path)
    internal_unique = (
        internal_rows.groupby("row_id", as_index=False)
        .agg(
            ood_severity_index=("ood_severity_index", "mean"),
            abs_error_log10_km=("abs_error_log10_km", "mean"),
        )
    )
    ext = pd.read_csv(external_predictions_path)
    ext_cat = ext.loc[ext["model"].eq("catboost")].copy()

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.7))
    ax = axes[0]
    ax.scatter(cat["ood_severity_index"], cat["mae"], s=12, alpha=0.18, color="#c2755b")
    ax.scatter(
        split_means["ood_severity_index"],
        split_means["mae"],
        s=42,
        color="#c2755b",
        edgecolor="0.25",
        linewidth=0.5,
        zorder=3,
    )
    ax.text(
        0.03,
        0.96,
        "Descriptive only\n"
        rf"Spearman $\rho$ = {split_association.statistic:.3f}" "\n"
        rf"nominal $p$ = {split_p_mantissa:.1f} $\times$ 10$^{{{split_p_exponent}}}$" "\n"
        f"n = {len(cat):,} split–repeat summaries",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
    )
    ax.text(
        0.97,
        0.04,
        f"Large points: protocol means (n = {len(split_means)})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.25",
    )
    ax.set_xlabel("OOD severity index")
    ax.set_ylabel("CatBoost MAE (log$_{10}$ units)")
    ax.set_title("A. Internal split-level")

    ax = axes[1]
    row_sample = internal_unique
    if len(row_sample) > 4500:
        row_sample = row_sample.sample(4500, random_state=2026)
    ax.scatter(
        row_sample["ood_severity_index"],
        row_sample["abs_error_log10_km"],
        s=9,
        alpha=0.10,
        color="#7aa77a",
        edgecolor="none",
    )
    xs, ys = lowess_like(internal_unique["ood_severity_index"], internal_unique["abs_error_log10_km"])
    ax.plot(xs, ys, color="0.2", lw=1.0)
    r = spearmanr(
        internal_unique["ood_severity_index"],
        internal_unique["abs_error_log10_km"],
        nan_policy="omit",
    ).statistic
    ax.text(
        0.03,
        0.96,
        rf"Spearman $\rho$ = {r:.2f}" "\n"
        f"n = {len(internal_unique):,} unique rows",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
    )
    ax.set_xlabel("Internal row-level severity")
    ax.set_ylabel("Absolute error (log$_{10}$ units)")
    ax.set_title("B. Internal row-level")
    ax.text(
        0.03,
        0.08,
        "Repeated split predictions averaged per row",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color="0.25",
    )

    ax = axes[2]
    flagged = ext_cat["retain_with_flag"].astype(bool)
    ax.scatter(
        ext_cat.loc[~flagged, "ood_severity_index"],
        ext_cat.loc[~flagged, "abs_error_log10_km"],
        s=20,
        alpha=0.62,
        color="#6f8fb3",
        label="unflagged",
    )
    ax.scatter(
        ext_cat.loc[flagged, "ood_severity_index"],
        ext_cat.loc[flagged, "abs_error_log10_km"],
        s=32,
        alpha=0.92,
        facecolor="none",
        edgecolor="#c2755b",
        marker="s",
        label="flagged",
    )
    xs, ys = lowess_like(ext_cat["ood_severity_index"], ext_cat["abs_error_log10_km"])
    ax.plot(xs, ys, color="0.2", lw=1.2)
    r = spearmanr(ext_cat["ood_severity_index"], ext_cat["abs_error_log10_km"], nan_policy="omit").statistic
    ax.text(
        0.03,
        0.96,
        rf"Spearman $\rho$ = {r:.2f}" "\n"
        f"n = {len(ext_cat):,} rows",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
    )
    ax.set_xlabel("External row-level severity")
    ax.set_ylabel("Absolute error (log$_{10}$ units)")
    ax.set_title("C. External row-level")
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    for ax in axes[1:]:
        ax.set_xlim(0.0, 0.75)
        ax.set_ylim(0.0, 5.0)
    sns.despine(fig=fig)
    save_figure(fig, fig_dir / "fig9_severity_error_diagnostics")


def build_external_collapse_mechanism_table(
    train_df: pd.DataFrame,
    external_model_ready_path: Path,
    external_predictions_path: Path,
    internal_row_predictions_path: Path,
    split_manifest_dir: Path,
    table_dir: Path,
) -> None:
    external = pd.read_csv(external_model_ready_path).copy()
    predictions = pd.read_csv(external_predictions_path)
    cat = predictions.loc[predictions["model"].eq("catboost")].copy()
    internal_predictions = pd.read_csv(internal_row_predictions_path)
    train = train_df.copy()
    train["substrate_for_km_norm"] = train.apply(_substrate_from_training_reaction, axis=1)
    external["substrate_for_km_norm"] = external["substrate_for_km_norm"].astype("string").fillna("unknown")

    train_protocols = set(_protocol_signatures(train))
    external_protocols = _protocol_signatures(external)
    train_formula_substrate = set(
        _formula_substrate_pairs(train["formula_group_id"], train["substrate_for_km_norm"])
    )
    external_formula_substrate = _formula_substrate_pairs(
        external["formula_group_id"], external["substrate_for_km_norm"]
    )
    target_low, target_high = train["target_log10_km"].quantile([0.01, 0.99])
    composition_threshold = _internal_composition_distance_threshold(train)
    composition_distance = pd.Series(np.nan, index=external.index)
    if "composition_nearest_train_distance" in cat.columns:
        composition_by_record = (
            cat.drop_duplicates("curation_record_id")
            .set_index("curation_record_id")["composition_nearest_train_distance"]
        )
        composition_distance = external["record_id"].map(composition_by_record)

    external_masks = {
        "Unseen source group": ~external["source_group_id"].astype("string").isin(
            set(train["source_group_id"].astype("string"))
        ),
        "Unseen standardized formula": ~external["formula_group_id"].astype("string").isin(
            set(train["formula_group_id"].astype("string"))
        ),
        "Unseen protocol signature": ~pd.Series(external_protocols, index=external.index).isin(
            train_protocols
        ),
        "Unseen formula x substrate pair": ~pd.Series(
            external_formula_substrate, index=external.index
        ).isin(train_formula_substrate),
        "Target outside train P1-P99": ~external["target_log10_km"].between(
            target_low, target_high, inclusive="both"
        ),
        "High composition distance": pd.to_numeric(composition_distance, errors="coerce").gt(
            composition_threshold
        ),
    }
    internal_comparison = _internal_mechanism_comparison(
        train,
        internal_predictions,
        split_manifest_dir,
        composition_threshold,
    )

    rows = []
    for mechanism, mask in external_masks.items():
        mask = pd.Series(mask, index=external.index).fillna(False).astype(bool)
        row = {
            "diagnostic": mechanism,
            "metric_type": "fraction_rows",
            "external_value": float(mask.mean()) if len(mask) else np.nan,
            "external_n": int(len(mask)),
            "interpretation": _collapse_interpretation(mechanism),
        }
        row.update(internal_comparison.get(mechanism, {}))
        rows.append(row)

    rows.append(_schema_coverage_comparison_row(train, external, len(external)))

    if len(cat):
        rows.append(
            _comparison_row(
                "CatBoost predicted/observed SD ratio",
                "sd_ratio",
                _prediction_sd_ratio(cat),
                int(cat["row_id"].nunique()),
                internal_comparison,
                "Low values indicate prediction compression toward the training-set center.",
            )
        )

    pd.DataFrame(rows).to_csv(table_dir / "table13_external_collapse_mechanisms.csv", index=False)
    _write_external_schema_coverage_diagnostics(train, external, table_dir)


def _comparison_row(
    diagnostic: str,
    metric_type: str,
    external_value: float,
    external_n: int,
    internal_comparison: dict[str, dict[str, float]],
    interpretation: str,
) -> dict:
    row = {
        "diagnostic": diagnostic,
        "metric_type": metric_type,
        "external_value": external_value,
        "external_n": int(external_n),
        "interpretation": interpretation,
    }
    row.update(internal_comparison.get(diagnostic, {}))
    return row


def _internal_mechanism_comparison(
    train: pd.DataFrame,
    internal_predictions: pd.DataFrame,
    split_manifest_dir: Path,
    composition_threshold: float,
) -> dict[str, dict[str, float]]:
    split_labels = {
        "random": "internal_random",
        "formula_group": "internal_formula_heldout",
        "composition_cluster": "internal_composition_cluster",
        "leakage_graph": "internal_leakage_graph",
    }
    mechanisms = [
        "Unseen source group",
        "Unseen standardized formula",
        "Unseen protocol signature",
        "Unseen formula x substrate pair",
        "Target outside train P1-P99",
        "High composition distance",
        "CatBoost predicted/observed SD ratio",
    ]
    comparison = {name: {} for name in mechanisms}
    row_lookup = train.reset_index().set_index("row_id")["index"].to_dict()
    train_pairs = _formula_substrate_pairs(train["formula_group_id"], train["substrate_for_km_norm"])
    train = train.assign(_protocol_signature=_protocol_signatures(train), _formula_substrate_pair=train_pairs)

    for split_type, col_name in split_labels.items():
        repeat_values = {name: [] for name in mechanisms}
        for manifest_path in sorted(split_manifest_dir.glob(f"{split_type}_repeat_*.csv")):
            manifest = pd.read_csv(manifest_path)
            train_ids = manifest.loc[manifest["split"].eq("train"), "row_id"].astype(int)
            test_ids = manifest.loc[manifest["split"].eq("test"), "row_id"].astype(int)
            train_idx = [row_lookup[row_id] for row_id in train_ids if row_id in row_lookup]
            test_idx = [row_lookup[row_id] for row_id in test_ids if row_id in row_lookup]
            fold_train = train.loc[train_idx]
            fold_test = train.loc[test_idx]
            if fold_test.empty:
                continue
            fold_pred = internal_predictions.loc[
                internal_predictions["split_type"].eq(split_type)
                & internal_predictions["repeat"].eq(_repeat_from_manifest(manifest_path))
            ]
            fold_low, fold_high = fold_train["target_log10_km"].quantile([0.01, 0.99])
            fold_masks = {
                "Unseen source group": ~fold_test["source_group_id"].astype("string").isin(
                    set(fold_train["source_group_id"].astype("string"))
                ),
                "Unseen standardized formula": ~fold_test["formula_group_id"].astype("string").isin(
                    set(fold_train["formula_group_id"].astype("string"))
                ),
                "Unseen protocol signature": ~fold_test["_protocol_signature"].isin(
                    set(fold_train["_protocol_signature"])
                ),
                "Unseen formula x substrate pair": ~fold_test["_formula_substrate_pair"].isin(
                    set(fold_train["_formula_substrate_pair"])
                ),
                "Target outside train P1-P99": ~fold_test["target_log10_km"].between(
                    fold_low, fold_high, inclusive="both"
                ),
            }
            for mechanism, mask in fold_masks.items():
                repeat_values[mechanism].append(float(pd.Series(mask).fillna(False).mean()))
            if "composition_nearest_train_distance" in fold_pred.columns and len(fold_pred):
                repeat_values["High composition distance"].append(
                    float(
                        pd.to_numeric(
                            fold_pred["composition_nearest_train_distance"], errors="coerce"
                        )
                        .gt(composition_threshold)
                        .mean()
                    )
                )
            if len(fold_pred):
                repeat_values["CatBoost predicted/observed SD ratio"].append(
                    _prediction_sd_ratio(fold_pred)
                )
        for mechanism, values in repeat_values.items():
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            comparison[mechanism][col_name] = float(finite.mean()) if len(finite) else np.nan
    return comparison


def _repeat_from_manifest(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def _prediction_sd_ratio(predictions: pd.DataFrame) -> float:
    unique = (
        predictions.groupby("row_id", as_index=False)
        .agg(target_log10_km=("target_log10_km", "mean"), pred_log10_km=("pred_log10_km", "mean"))
    )
    observed_sd = float(unique["target_log10_km"].std(ddof=1))
    predicted_sd = float(unique["pred_log10_km"].std(ddof=1))
    if not np.isfinite(observed_sd) or observed_sd <= 0:
        return np.nan
    return predicted_sd / observed_sd


def _schema_coverage_comparison_row(train: pd.DataFrame, external: pd.DataFrame, n_external: int) -> dict:
    columns = _raw_descriptor_columns()
    train_available = [col for col in columns if col in train.columns]
    external_available = [col for col in train_available if col in external.columns]
    external_value = len(external_available) / max(len(train_available), 1)
    return {
        "diagnostic": "External feature schema coverage",
        "metric_type": "fraction_internal_descriptor_columns_present",
        "external_value": float(external_value),
        "external_n": int(n_external),
        "interpretation": (
            "Schema-level diagnostic, not row-level missingness: the external validation "
            "set was curated for Km/Vmax and TMB/H2O2 substrate fields, so most internal "
            "raw descriptor columns are unavailable by design."
        ),
        "internal_random": 1.0,
        "internal_formula_heldout": 1.0,
        "internal_composition_cluster": 1.0,
        "internal_leakage_graph": 1.0,
    }


def _write_external_schema_coverage_diagnostics(
    train: pd.DataFrame, external: pd.DataFrame, table_dir: Path
) -> None:
    rows = []
    for col in _raw_descriptor_columns():
        in_internal = col in train.columns
        in_external = col in external.columns
        external_non_missing = np.nan
        if in_external:
            values = external[col]
            external_non_missing = float(
                (~(values.isna() | values.astype("string").str.strip().fillna("").eq(""))).mean()
            )
        rows.append(
            {
                "descriptor_field": col,
                "in_internal_schema": bool(in_internal),
                "in_external_schema": bool(in_external),
                "external_non_missing_fraction_if_present": external_non_missing,
                "interpretation": (
                    "Schema coverage should be interpreted as an external benchmark design "
                    "constraint, not evidence that the original papers failed to report this field."
                ),
            }
        )
    pd.DataFrame(rows).to_csv(table_dir / "table14_external_schema_coverage_diagnostics.csv", index=False)


def _raw_descriptor_columns() -> list[str]:
    descriptor_columns = [
        c
        for c in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
        if c not in {"km_unit", "vmax_unit", "c_const_unit", "ccat_unit"}
    ]
    return descriptor_columns


def _collapse_interpretation(mechanism: str) -> str:
    messages = {
        "Unseen source group": "Source novelty relative to the internal training corpus.",
        "Unseen standardized formula": "Chemical identity novelty; compare against formula-held-out internal splits.",
        "Unseen protocol signature": "Exact protocol-context novelty based on available protocol descriptors.",
        "Unseen formula x substrate pair": "Joint material/substrate novelty for substrate-specific Km prediction.",
        "Target outside train P1-P99": "Target-range extrapolation; low external value argues against target extrapolation as the main collapse mechanism.",
        "High composition distance": "Composition-neighbourhood shift using a nearest-train-distance threshold calibrated from internal leave-one-out distances.",
    }
    return messages.get(mechanism, "")


def _protocol_signatures(df: pd.DataFrame) -> list[str]:
    available_categorical = [c for c in PROTOCOL_CATEGORICAL_COLUMNS if c in df.columns]
    available_numeric = [c for c in PROTOCOL_NUMERIC_COLUMNS if c in df.columns]
    signatures = []
    for _, row in df.iterrows():
        parts = []
        for col in available_categorical:
            parts.append(f"{col}={_clean_token(row[col])}")
        for col in available_numeric:
            value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.isna(value):
                parts.append(f"{col}=missing")
            else:
                parts.append(f"{col}={float(value):.3g}")
        signatures.append("|".join(parts) if parts else "__NO_PROTOCOL__")
    return signatures


def _clean_token(value: object) -> str:
    if pd.isna(value):
        return "missing"
    return str(value).strip().lower().replace(" ", "_") or "missing"


def _substrate_from_training_reaction(row: pd.Series) -> str:
    reaction = str(row.get("reaction_type", "")).strip().upper().replace(" ", "")
    if reaction.startswith("TMB+H2O2"):
        return "TMB"
    if reaction.startswith("H2O2+TMB"):
        return "H2O2"
    return "OTHER"


def _formula_substrate_pairs(formula: pd.Series, substrate: pd.Series) -> list[str]:
    return (
        formula.astype("string").fillna("__NO_FORMULA__")
        + "||"
        + substrate.astype("string").fillna("unknown")
    ).astype(str).tolist()


def _internal_composition_distance_threshold(train_df: pd.DataFrame) -> float:
    X, meta = build_feature_frame(train_df)
    comp_cols = list(meta["composition_columns"])
    if not comp_cols or len(train_df) < 3:
        return np.inf
    comp = X[comp_cols].apply(pd.to_numeric, errors="coerce")
    med = comp.median(numeric_only=True)
    scale = comp.std(ddof=0).replace(0, np.nan)
    scaled = ((comp.fillna(med) - med) / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    arr = scaled.to_numpy(dtype=float)
    distances = []
    for idx in range(len(arr)):
        diff = arr[idx][None, :] - np.delete(arr, idx, axis=0)
        nearest = np.sqrt(np.mean(diff * diff, axis=1)).min()
        distances.append(float(nearest))
    return float(np.quantile(distances, 0.95))


def lowess_like(x: pd.Series, y: pd.Series, frac: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    data = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
    if len(data) < 5:
        return data["x"].to_numpy(), data["y"].to_numpy()
    xs = np.linspace(data["x"].min(), data["x"].max(), 100)
    span = max(frac * (data["x"].max() - data["x"].min()), 1e-9)
    smooth = []
    for value in xs:
        distance = np.abs(data["x"].to_numpy() - value)
        weights = np.exp(-0.5 * (distance / span) ** 2)
        if np.all(weights == 0):
            smooth.append(np.nan)
        else:
            smooth.append(float(np.average(data["y"], weights=weights)))
    return xs, np.asarray(smooth)


def write_manifests(dirs: dict[str, Path], fig_dir: Path, table_dir: Path) -> None:
    figure_rows = []
    descriptions = {
        "fig1_dataset_structure": "Internal modeling cohort structure and target distribution.",
        "fig2_internal_ood_performance": "Model MAE under random and OOD split strategies.",
        "fig3_catboost_vs_baseline": "Compact CatBoost versus median baseline comparison.",
        "fig4_conformal_baseline": (
            "Baseline conformal coverage and interval width. Dashed line marks "
            "90% nominal coverage; error bars show empirical 2.5-97.5% intervals "
            "across 20 repeated splits."
        ),
        "fig5_few_shot_recalibration": "Few-shot target-domain recalibration sensitivity.",
        "fig6_external_predicted_vs_observed_catboost": "Strict external CatBoost predicted versus observed Km.",
        "fig7_external_error_by_substrate": "External CatBoost error distribution by substrate and audit flag.",
        "fig8_delta_mae_internal_external": "Internal-to-external reversal of CatBoost advantage relative to the median baseline.",
        "fig9_severity_error_diagnostics": "OOD severity versus prediction error in internal and external evaluations.",
        "figS1_external_scenario_metrics": "Supplementary external validation scenario sensitivity analysis.",
    }
    if (fig_dir / "fig10_restricted_feature_validation.png").exists():
        descriptions["fig10_restricted_feature_validation"] = (
            "Fair shared-schema validation using formula-derived composition descriptors "
            "and substrate identity only."
        )
    for stem, description in descriptions.items():
        figure_rows.append(
            {
                "figure_id": stem,
                "description": description,
                "png": str((fig_dir / f"{stem}.png").resolve()),
                "pdf": str((fig_dir / f"{stem}.pdf").resolve()),
                "svg": str((fig_dir / f"{stem}.svg").resolve()),
            }
        )
    pd.DataFrame(figure_rows).to_csv(table_dir / "publication_figure_manifest.csv", index=False)

    table_rows = []
    for path in sorted(table_dir.glob("table*.csv")):
        table_rows.append({"table_file": path.name, "path": str(path.resolve())})
    pd.DataFrame(table_rows).to_csv(table_dir / "publication_table_manifest.csv", index=False)


def summarize(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        vals = group[metric].dropna().astype(float)
        row = dict(zip(group_cols, keys))
        row["mean"] = vals.mean()
        row["low"] = vals.quantile(0.025) if len(vals) > 1 else vals.mean()
        row["high"] = vals.quantile(0.975) if len(vals) > 1 else vals.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def save_figure(
    fig: plt.Figure,
    stem: Path,
    tight_layout_rect: tuple[float, float, float, float] | None = None,
) -> None:
    fig.tight_layout(rect=tight_layout_rect)
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
