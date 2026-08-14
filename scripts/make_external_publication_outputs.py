#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting import ensure_output_dirs


MODEL_ORDER = ["median", "elasticnet", "extratrees", "catboost"]
MODEL_LABELS = {
    "median": "Median",
    "elasticnet": "Elastic net",
    "extratrees": "ExtraTrees",
    "catboost": "CatBoost",
}
SUBSTRATE_COLORS = {"TMB": "#2a7ab0", "H2O2": "#c85a3e"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create publication-ready external validation figures and summary tables."
    )
    parser.add_argument(
        "--predictions",
        default="outputs/tables/external_validation/external_validation_strict_predictions.csv",
    )
    parser.add_argument(
        "--model-ready",
        default="outputs/tables/external_validation/external_validation_model_ready_with_audit.csv",
    )
    parser.add_argument(
        "--strict-metrics",
        default="outputs/tables/external_validation/external_validation_strict_metrics.csv",
    )
    parser.add_argument(
        "--scenario-metrics",
        default="outputs/tables/external_validation/external_validation_scenario_metrics.csv",
    )
    parser.add_argument(
        "--overlap-diagnostics",
        default="outputs/tables/external_validation/external_overlap_diagnostics.csv",
    )
    args = parser.parse_args()

    dirs = ensure_output_dirs()
    figure_dir = dirs["figures"] / "external_validation"
    table_dir = dirs["tables"] / "external_validation"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    model_ready = pd.read_csv(args.model_ready)
    strict_metrics = pd.read_csv(args.strict_metrics)
    scenario_metrics = pd.read_csv(args.scenario_metrics)
    overlap = pd.read_csv(args.overlap_diagnostics)

    sns.set_theme(context="paper", style="ticks", font_scale=1.1)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )

    summary = publication_summary_table(model_ready, strict_metrics, scenario_metrics, overlap)
    summary_path = table_dir / "external_validation_publication_summary_table.csv"
    summary.to_csv(summary_path, index=False)

    concise_metrics = publication_metrics_table(strict_metrics, scenario_metrics)
    metrics_path = table_dir / "external_validation_publication_metrics_table.csv"
    concise_metrics.to_csv(metrics_path, index=False)

    failure_cases = publication_failure_table(predictions)
    failure_path = table_dir / "external_validation_publication_failure_cases_catboost.csv"
    failure_cases.to_csv(failure_path, index=False)

    plot_predicted_vs_observed(predictions, figure_dir)
    plot_error_by_substrate(predictions, figure_dir)
    plot_scenario_metrics(strict_metrics, scenario_metrics, figure_dir)

    print(f"Wrote {summary_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {failure_path}")
    print(f"Wrote figures to {figure_dir}")


def publication_summary_table(
    model_ready: pd.DataFrame,
    strict_metrics: pd.DataFrame,
    scenario_metrics: pd.DataFrame,
    overlap: pd.DataFrame,
) -> pd.DataFrame:
    diag = dict(zip(overlap["diagnostic"], overlap["value"]))
    cat_strict = strict_metrics.query(
        "scenario == 'strict_external' and model == 'catboost' and substrate == 'all'"
    ).iloc[0]
    med_strict = strict_metrics.query(
        "scenario == 'strict_external' and model == 'median' and substrate == 'all'"
    ).iloc[0]
    cat_unflagged = strict_metrics.query(
        "scenario == 'strict_external_unflagged_only' and model == 'catboost' and substrate == 'all'"
    ).iloc[0]
    central95 = scenario_metrics.query(
        "scenario == 'central_95pct_km' and model == 'catboost' and substrate == 'all'"
    ).iloc[0]
    all_rows = model_ready
    strict_rows = model_ready.loc[model_ready["strict_external_recommended"].astype(bool)]
    return pd.DataFrame(
        [
            {
                "item": "External validation rows",
                "value": int(len(strict_rows)),
                "note": "Strict external benchmark after manual audit",
            },
            {
                "item": "Unique curated records",
                "value": int(strict_rows["curation_record_id"].nunique()),
                "note": "Record-level literature entries",
            },
            {
                "item": "Unique standardized formulas",
                "value": int(strict_rows["standardized_formula"].nunique()),
                "note": "Formula parser grouping",
            },
            {
                "item": "TMB rows",
                "value": int(strict_rows["substrate_for_km_norm"].eq("TMB").sum()),
                "note": "Substrate-specific Km entries",
            },
            {
                "item": "H2O2 rows",
                "value": int(strict_rows["substrate_for_km_norm"].eq("H2O2").sum()),
                "note": "Substrate-specific Km entries",
            },
            {
                "item": "Priority-audited rows",
                "value": int(strict_rows["priority20_audited"].sum()),
                "note": "Highest-priority CatBoost failure cases manually audited",
            },
            {
                "item": "Retained with flag",
                "value": int(strict_rows["retain_with_flag"].sum()),
                "note": "Rows kept but flagged for manuscript interpretation",
            },
            {
                "item": "DOI overlap with training",
                "value": int(float(diag.get("external_rows_with_training_doi_overlap", 0))),
                "note": "Must be zero for DOI-level external claim",
            },
            {
                "item": "Formula overlap fraction",
                "value": float(diag.get("external_formula_overlap_fraction", np.nan)),
                "note": "Fraction of external rows sharing standardized formula with training",
            },
            {
                "item": "CatBoost strict MAE",
                "value": float(cat_strict["mae"]),
                "note": "MAE in log10(Km/M)",
            },
            {
                "item": "Median strict MAE",
                "value": float(med_strict["mae"]),
                "note": "Baseline MAE in log10(Km/M)",
            },
            {
                "item": "CatBoost unflagged MAE",
                "value": float(cat_unflagged["mae"]),
                "note": "Priority20 retain_with_flag rows removed",
            },
            {
                "item": "CatBoost central-95% MAE",
                "value": float(central95["mae"]),
                "note": "Central 95% of external Km distribution",
            },
            {
                "item": "All model-ready rows",
                "value": int(len(all_rows)),
                "note": "Before strict recommendation filter",
            },
        ]
    )


def publication_metrics_table(
    strict_metrics: pd.DataFrame, scenario_metrics: pd.DataFrame
) -> pd.DataFrame:
    selected_strict = strict_metrics.loc[
        strict_metrics["scenario"].isin(["strict_external", "strict_external_unflagged_only"])
    ].copy()
    selected_scenario = scenario_metrics.loc[
        scenario_metrics["scenario"].isin(["central_95pct_km", "no_hard_km_extremes"])
    ].copy()
    common_cols = ["scenario", "model", "substrate", "n_rows", "n_unique_records", "mae", "rmse", "r2", "spearman"]
    out = pd.concat(
        [selected_strict[common_cols], selected_scenario[common_cols]], ignore_index=True
    )
    out["model_label"] = out["model"].map(MODEL_LABELS).fillna(out["model"])
    return out[
        ["scenario", "model", "model_label", "substrate", "n_rows", "n_unique_records", "mae", "rmse", "r2", "spearman"]
    ].sort_values(["scenario", "model", "substrate"])


def publication_failure_table(predictions: pd.DataFrame) -> pd.DataFrame:
    cat = predictions.loc[predictions["model"].eq("catboost")].copy()
    cols = [
        "row_id",
        "curation_record_id",
        "formula",
        "standardized_formula",
        "substrate_for_km_norm",
        "km_millimolar",
        "pred_km_millimolar",
        "abs_error_log10_km",
        "fold_error",
        "manual_audit_decision",
        "manual_flags",
        "doi",
        "title",
        "curation_note",
    ]
    return cat.sort_values("abs_error_log10_km", ascending=False)[cols].head(25)


def plot_predicted_vs_observed(predictions: pd.DataFrame, figure_dir: Path) -> None:
    cat = predictions.loc[predictions["model"].eq("catboost")].copy()
    cat["observed_log10_km_mM"] = cat["target_log10_km"] + 3.0
    cat["predicted_log10_km_mM"] = cat["pred_log10_km"] + 3.0
    sd_ratio = cat["predicted_log10_km_mM"].std(ddof=1) / cat["observed_log10_km_mM"].std(ddof=1)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for substrate, group in cat.groupby("substrate_for_km_norm"):
        flagged = group["retain_with_flag"].astype(bool)
        ax.scatter(
            group.loc[~flagged, "observed_log10_km_mM"],
            group.loc[~flagged, "predicted_log10_km_mM"],
            s=24,
            alpha=0.78,
            color=SUBSTRATE_COLORS.get(substrate, "0.4"),
            edgecolor="none",
            label=f"{substrate} unflagged",
        )
        if flagged.any():
            ax.scatter(
                group.loc[flagged, "observed_log10_km_mM"],
                group.loc[flagged, "predicted_log10_km_mM"],
                s=42,
                alpha=0.95,
                facecolor="none",
                edgecolor=SUBSTRATE_COLORS.get(substrate, "0.4"),
                linewidth=1.1,
                marker="s",
                label=f"{substrate} flagged",
            )
    lo = min(cat["observed_log10_km_mM"].min(), cat["predicted_log10_km_mM"].min()) - 0.25
    hi = max(cat["observed_log10_km_mM"].max(), cat["predicted_log10_km_mM"].max()) + 0.25
    ax.plot([lo, hi], [lo, hi], color="0.25", linewidth=1.0, linestyle="--", label="ideal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Observed $\log_{10}(K_m\,/\,\mathrm{mM})$")
    ax.set_ylabel(r"Predicted $\log_{10}(K_m\,/\,\mathrm{mM})$")
    ax.text(
        0.03,
        0.97,
        f"n = {len(cat)}\nMAE = {cat['abs_error_log10_km'].mean():.3f}\npred/obs SD = {sd_ratio:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.9},
    )
    ax.text(
        0.03,
        0.74,
        "Open squares: retained records\nflagged for unit/reporting risk",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="0.25",
    )
    ax.legend(frameon=False, loc="lower right", ncol=1, fontsize=7)
    sns.despine(ax=ax)
    save_figure(fig, figure_dir / "fig_external_predicted_vs_observed_catboost")


def plot_error_by_substrate(predictions: pd.DataFrame, figure_dir: Path) -> None:
    cat = predictions.loc[predictions["model"].eq("catboost")].copy()
    cat["Audit group"] = np.where(cat["retain_with_flag"].astype(bool), "flagged", "unflagged")
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    sns.stripplot(
        data=cat,
        x="substrate_for_km_norm",
        y="abs_error_log10_km",
        hue="Audit group",
        order=["TMB", "H2O2"],
        hue_order=["unflagged", "flagged"],
        dodge=True,
        palette={"unflagged": "#406d8c", "flagged": "#a7472e"},
        size=2.2,
        alpha=0.45,
        linewidth=0,
        ax=ax,
        legend=False,
    )
    offsets = {"unflagged": -0.2, "flagged": 0.2}
    colors = {"unflagged": "#406d8c", "flagged": "#a7472e"}
    for x_base, substrate in enumerate(["TMB", "H2O2"]):
        for audit_group in ["unflagged", "flagged"]:
            vals = cat.loc[
                cat["substrate_for_km_norm"].eq(substrate) & cat["Audit group"].eq(audit_group),
                "abs_error_log10_km",
            ].dropna()
            if vals.empty:
                continue
            x = x_base + offsets[audit_group]
            median = vals.median()
            ax.hlines(median, x - 0.11, x + 0.11, color=colors[audit_group], lw=2.0)
            ax.text(
                x,
                vals.max() + 0.08,
                f"n={len(vals)}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=colors[audit_group],
            )
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#406d8c", markersize=5, label="unflagged"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#a7472e", markersize=5, label="flagged"),
    ]
    ax.legend(handles=legend_handles, frameon=False, title="", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_xlabel("Substrate")
    ax.set_ylabel(r"Absolute error (log$_{10}$ units)")
    ax.set_ylim(0, cat["abs_error_log10_km"].max() * 1.18)
    sns.despine(ax=ax)
    save_figure(fig, figure_dir / "fig_external_error_by_substrate")


def plot_scenario_metrics(
    strict_metrics: pd.DataFrame, scenario_metrics: pd.DataFrame, figure_dir: Path
) -> None:
    scenarios = [
        ("all_model_ready", "All curated"),
        ("strict_external_unflagged_only", "Unflagged"),
        ("central_95pct_km", "Central 95%"),
        ("no_hard_km_extremes", "No hard extremes"),
    ]
    combined = pd.concat([strict_metrics, scenario_metrics], ignore_index=True)
    combined = combined.loc[
        combined["substrate"].eq("all") & combined["scenario"].isin([s[0] for s in scenarios])
    ].copy()
    combined["scenario_label"] = combined["scenario"].map(dict(scenarios))
    combined["model_label"] = combined["model"].map(MODEL_LABELS)
    combined["scenario_label"] = pd.Categorical(
        combined["scenario_label"], categories=[s[1] for s in scenarios], ordered=True
    )
    combined["model_label"] = pd.Categorical(
        combined["model_label"], categories=[MODEL_LABELS[m] for m in MODEL_ORDER], ordered=True
    )
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    sns.barplot(
        data=combined,
        x="scenario_label",
        y="mae",
        hue="model_label",
        palette=["#9a9a9a", "#6f8fb3", "#7aa77a", "#c2755b"],
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(r"MAE (log$_{10}$ units)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, title="", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_ylim(0, max(1.05, combined["mae"].max() * 1.15))
    sns.despine(ax=ax)
    save_figure(fig, figure_dir / "fig_external_scenario_metrics")


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
