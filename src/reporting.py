from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def ensure_output_dirs(base: str | Path = "outputs") -> dict[str, Path]:
    base = Path(base)
    dirs = {
        "metrics": base / "metrics",
        "figures": base / "figures",
        "tables": base / "tables",
        "split_manifests": base / "split_manifests",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def make_main_tables(audit_dir: Path, metrics: pd.DataFrame, conformal: pd.DataFrame | None, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = audit_dir / "audit_summary.json"
    if summary_path.exists():
        summary = pd.read_json(summary_path, typ="series")["summary"]
        pd.DataFrame([summary]).to_csv(out_dir / "main_table_1_modeling_cohort.csv", index=False)

    table2 = pd.DataFrame(
        [
            {
                "split_type": "random",
                "scientific_question": "In-distribution benchmark baseline",
                "leakage_controlled": "no source/formula grouping",
                "grouping_variable": "none",
                "train_test_separation_rule": "row-level random split",
                "deployment_scenario": "similar literature/source/chemistry distribution",
                "main_limitation": "can inflate performance through repeated sources or formulas",
            },
            {
                "split_type": "source_group",
                "scientific_question": "generalization to unseen literature sources",
                "leakage_controlled": "source metadata only used for grouping",
                "grouping_variable": "source_group_id",
                "train_test_separation_rule": "entire source groups held out",
                "deployment_scenario": "new paper/lab/source",
                "main_limitation": "source and chemistry effects may be entangled",
            },
            {
                "split_type": "formula_group",
                "scientific_question": "generalization to unseen formula identities",
                "leakage_controlled": "formula identity held out",
                "grouping_variable": "formula_group_id",
                "train_test_separation_rule": "entire standardized formulas held out",
                "deployment_scenario": "new material formula",
                "main_limitation": "composition-similar formulas may remain nearby",
            },
            {
                "split_type": "composition_cluster",
                "scientific_question": "empirical chemistry-region stress test",
                "leakage_controlled": "clusters fit using training composition descriptors only",
                "grouping_variable": "nearest train-derived composition cluster",
                "train_test_separation_rule": "held-out rows assigned to predefined composition cluster",
                "deployment_scenario": "chemistry-shifted candidate set",
                "main_limitation": "not a guarantee of fully novel composition space",
            },
            {
                "split_type": "leakage_graph",
                "scientific_question": "generalization across disconnected audited metadata components",
                "leakage_controlled": "connected components of the source-formula-protocol graph held out",
                "grouping_variable": "leakage_graph_component_id",
                "train_test_separation_rule": "no graph-defined metadata component is split across train and test",
                "deployment_scenario": "new literature/material dependency component with audited protocol context",
                "main_limitation": "component topology is data-dependent; in the final dataset protocol nodes did not merge additional source-formula components",
            },
        ]
    )
    table2.to_csv(out_dir / "main_table_2_split_definitions.csv", index=False)

    if len(metrics):
        compact = metrics[metrics["model"].isin(["median", "catboost"])].copy()
        compact = _metric_summary(
            compact,
            ["split_type", "model"],
            ["mae", "r2", "spearman"],
        )
        compact.to_csv(out_dir / "main_table_3_compact_performance.csv", index=False)
    if conformal is not None and len(conformal):
        conformal_metrics = [
            "empirical_coverage",
            "coverage_gap",
            "mean_interval_width",
            "interval_score",
        ]
        conformal_group_cols = ["split_type", "method", "shots"]
        if "feature_schema" in conformal.columns:
            conformal_group_cols.insert(0, "feature_schema")
        repeat_level = (
            conformal.groupby([*conformal_group_cols, "repeat"], dropna=False, as_index=False)[
                conformal_metrics
            ]
            .mean(numeric_only=True)
        )
        conformal_summary = _metric_summary(
            repeat_level,
            conformal_group_cols,
            conformal_metrics,
        )
        conformal_summary.to_csv(out_dir / "main_table_4_conformal_recalibration.csv", index=False)
    checklist = pd.DataFrame(
        [
            {
                "required_field": "formula",
                "why_it_matters_for_ml": "defines composition descriptors and formula-group leakage",
                "observed_data_problem": "inconsistent strings and decorated formula names",
                "suggested_reporting_format": "standard chemical formula plus separate coating/support fields",
            },
            {
                "required_field": "source DOI or stable source ID",
                "why_it_matters_for_ml": "enables source-group OOD evaluation",
                "observed_data_problem": "missing or mixed article metadata",
                "suggested_reporting_format": "DOI where available, otherwise stable source identifier",
            },
            {
                "required_field": "Km value and unit",
                "why_it_matters_for_ml": "primary supervised target",
                "observed_data_problem": "unit heterogeneity and invalid/missing targets",
                "suggested_reporting_format": "numeric value plus explicit concentration unit",
            },
            {
                "required_field": "pH, temperature, substrate concentration",
                "why_it_matters_for_ml": "protocol variation can drive apparent kinetics",
                "observed_data_problem": "sparse or inconsistently named fields",
                "suggested_reporting_format": "numeric values with units and controlled vocabulary",
            },
        ]
    )
    checklist.to_csv(out_dir / "main_table_5_reporting_checklist.csv", index=False)


def make_figures(metrics_path: Path, conformal_path: Path | None, out_dir: Path) -> None:
    mpl_config = Path(os.environ.get("TMPDIR", "/tmp")) / "nanozyme_mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_config))
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    audit_dir = out_dir.parent / "tables" / "audit"
    source_path = audit_dir / "source_frequency.csv"
    formula_path = audit_dir / "formula_frequency.csv"
    reaction_path = audit_dir / "reaction_distribution.csv"
    if source_path.exists() and formula_path.exists():
        source = pd.read_csv(source_path).head(20)
        formula = pd.read_csv(formula_path).head(20)
        fig, axes = plt.subplots(1, 3 if reaction_path.exists() else 2, figsize=(11, 3.6))
        sns.barplot(data=source, x="n_rows", y="source_group_id", ax=axes[0], color="#5975a4")
        axes[0].set_title("Source Frequency")
        axes[0].set_xlabel("Rows")
        axes[0].set_ylabel("")
        sns.barplot(data=formula, x="n_rows", y="standardized_formula", ax=axes[1], color="#5f9e6e")
        axes[1].set_title("Formula Frequency")
        axes[1].set_xlabel("Rows")
        axes[1].set_ylabel("")
        if reaction_path.exists():
            reaction = pd.read_csv(reaction_path).head(10)
            y_col = "reaction_type" if "reaction_type" in reaction.columns else reaction.columns[0]
            sns.barplot(data=reaction, x="n_rows", y=y_col, ax=axes[2], color="#b55d60")
            axes[2].set_title("Reaction Systems")
            axes[2].set_xlabel("Rows")
            axes[2].set_ylabel("")
        plt.tight_layout()
        plt.savefig(out_dir / "main_figure_1_dataset_structure.png", dpi=300)
        plt.close()

    if not metrics_path.exists():
        return
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return
    plt.figure(figsize=(7, 4))
    sns.pointplot(data=metrics, x="split_type", y="mae", hue="model", errorbar=("ci", 95))
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("MAE on log10(Km)")
    plt.tight_layout()
    plt.savefig(out_dir / "main_figure_2_performance_degradation.png", dpi=300)
    plt.close()

    random_ref = metrics[metrics["split_type"] == "random"].groupby("model")[["mae", "r2"]].mean()
    gaps = []
    for _, row in metrics.iterrows():
        if row["split_type"] == "random" or row["model"] not in random_ref.index:
            continue
        gaps.append(
            {
                "model": row["model"],
                "split_type": row["split_type"],
                "delta_mae": row["mae"] - random_ref.loc[row["model"], "mae"],
                "delta_r2": random_ref.loc[row["model"], "r2"] - row["r2"],
            }
        )
    gap_df = pd.DataFrame(gaps)
    if len(gap_df):
        gap_df.to_csv(out_dir.parent / "tables" / "supp_table_s3_optimism_gap.csv", index=False)
        plt.figure(figsize=(7, 4))
        sns.pointplot(data=gap_df, x="split_type", y="delta_mae", hue="model", errorbar=("ci", 95))
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("Delta MAE vs random split")
        plt.tight_layout()
        plt.savefig(out_dir / "main_figure_3_optimism_gap.png", dpi=300)
        plt.close()

    if conformal_path and conformal_path.exists():
        conf = pd.read_csv(conformal_path)
        if len(conf):
            plt.figure(figsize=(7, 4))
            sns.pointplot(
                data=conf[conf["shots"] == 0],
                x="split_type",
                y="empirical_coverage",
                hue="method",
                errorbar=("ci", 95),
            )
            plt.axhline(0.9, color="black", linestyle="--", linewidth=0.8)
            plt.xticks(rotation=20, ha="right")
            plt.ylabel("Empirical coverage")
            plt.tight_layout()
            plt.savefig(out_dir / "main_figure_4_conformal_calibration.png", dpi=300)
            plt.close()

            powered = conf[~conf.get("underpowered", False).astype(bool)]
            if len(powered):
                plt.figure(figsize=(7, 4))
                sns.lineplot(
                    data=powered,
                    x="shots",
                    y="coverage_gap",
                    hue="split_type",
                    style="method",
                    estimator="mean",
                    errorbar=("ci", 95),
                )
                plt.axhline(0, color="black", linewidth=0.8)
                plt.ylabel("Coverage gap")
                plt.tight_layout()
                plt.savefig(out_dir / "main_figure_5_few_shot_recalibration.png", dpi=300)
                plt.close()


def _metric_summary(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_repeats"] = group["repeat"].nunique() if "repeat" in group.columns else len(group)
        for metric in metric_cols:
            vals = group[metric].dropna().astype(float)
            row[metric] = vals.mean() if len(vals) else pd.NA
            row[f"{metric}_ci95_low"] = vals.quantile(0.025) if len(vals) > 1 else pd.NA
            row[f"{metric}_ci95_high"] = vals.quantile(0.975) if len(vals) > 1 else pd.NA
        rows.append(row)
    return pd.DataFrame(rows)
