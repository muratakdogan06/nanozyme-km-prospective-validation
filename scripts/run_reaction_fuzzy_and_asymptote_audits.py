#!/usr/bin/env python
"""Target-semantics and fuzzy cross-source redundancy audits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import prepare_modeling_data, read_table
from src.diagnostics import source_formula_protocol_component_ids
from src.evaluate import regression_metrics
from src.features import build_design_only_feature_frame, build_feature_frame
from src.models import FeatureSpec, fit_predict, model_suite, require_catboost
from src.splits import make_split


INTERNAL_SPLIT_ORDER = [
    "random",
    "composition_cluster",
    "formula_group",
    "source_group",
    "leakage_graph",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/nanozymes.csv")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--analysis",
        choices=["all", "schema", "substrates"],
        default="all",
    )
    args = parser.parse_args()
    require_catboost()
    out = Path("outputs/tables/reviewer2_sensitivity")
    out.mkdir(parents=True, exist_ok=True)

    df, _ = prepare_modeling_data(read_table(args.data))
    df["leakage_graph_component_id"] = source_formula_protocol_component_ids(df)
    if args.analysis == "schema":
        design_only_schema_sensitivity(df, args.repeats, args.seed).to_csv(
            out / "design_only_vs_post_characterization.csv", index=False
        )
        print(f"Design-only schema audit complete: {out}")
        return
    if args.analysis == "substrates":
        design_only_substrate_stratified_performance(
            df, args.repeats, args.seed
        ).to_csv(
            out / "design_only_substrate_stratified_performance.csv", index=False
        )
        print(f"Design-only substrate audit complete: {out}")
        return

    reaction_semantics_audit(df).to_csv(out / "reaction_target_semantics_audit.csv", index=False)
    reaction_source_association(df).to_csv(
        out / "reaction_source_association.csv", index=False
    )
    substrate_stratified_performance(df).to_csv(
        out / "substrate_stratified_internal_performance.csv", index=False
    )
    matched_substrate_random_leakage(df).to_csv(
        out / "matched_substrate_random_leakage.csv", index=False
    )
    design_only_schema_sensitivity(df, args.repeats, args.seed).to_csv(
        out / "design_only_vs_post_characterization.csv", index=False
    )
    fuzzy_membership, clusters = fuzzy_cross_source_membership(df)
    clusters.to_csv(out / "fuzzy_cross_source_clusters.csv", index=False)
    fuzzy_sensitivity(df, fuzzy_membership, args.repeats, args.seed).to_csv(
        out / "fuzzy_cross_source_duplicate_sensitivity.csv", index=False
    )
    elasticnet_failure_diagnostic(df, fuzzy_membership, args.repeats, args.seed).to_csv(
        out / "elasticnet_component_failure_diagnostic.csv", index=False
    )
    print(f"Reaction/fuzzy audits complete: {out}")


def design_only_schema_sensitivity(
    df: pd.DataFrame, repeats: int, seed: int
) -> pd.DataFrame:
    schemas = {
        "design_only_primary": build_design_only_feature_frame,
        "post_characterization_sensitivity": build_feature_frame,
    }
    rows = []
    for schema_name, builder in schemas.items():
        cohort = df.copy()
        if schema_name == "design_only_primary":
            cohort["leakage_graph_component_id"] = source_formula_protocol_component_ids(
                cohort.drop(columns=["vmax_value", "vmax_unit"], errors="ignore")
            )
        X, meta = builder(cohort)
        y = cohort["target_log10_km"].astype(float)
        spec = FeatureSpec(meta["numeric_columns"], meta["categorical_columns"])
        for repeat in range(repeats):
            rep_seed = seed + repeat
            models = model_suite(spec, rep_seed)
            for split_type in [
                "random",
                "source_group",
                "formula_group",
                "composition_cluster",
                "leakage_graph",
            ]:
                train_idx, test_idx, _ = make_split(
                    cohort,
                    split_type,
                    rep_seed,
                    test_size=0.2,
                    composition=X[meta["composition_columns"]],
                    min_test_n=20,
                )
                for model_name in ["median", "catboost"]:
                    pred = fit_predict(
                        models[model_name],
                        X.loc[train_idx],
                        y.loc[train_idx],
                        X.loc[test_idx],
                    )
                    rows.append(
                        {
                            "feature_schema": schema_name,
                            "excluded_columns": ";".join(meta["excluded_columns"]),
                            "repeat": repeat,
                            "split_type": split_type,
                            "model": model_name,
                            "n_train": len(train_idx),
                            "n_test": len(test_idx),
                            **regression_metrics(y.loc[test_idx], pred),
                        }
                    )
            print(
                f"{schema_name}: completed repeat {repeat + 1}/{repeats}",
                flush=True,
            )
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(
            ["feature_schema", "excluded_columns", "split_type", "model"],
            as_index=False,
        )
        .agg(
            n_repeats=("repeat", "nunique"),
            mean_n_train=("n_train", "mean"),
            mean_n_test=("n_test", "mean"),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            spearman_mean=("spearman", "mean"),
            spearman_sd=("spearman", "std"),
        )
    )
    summary["analysis_role"] = np.where(
        summary["feature_schema"].eq("design_only_primary"),
        "primary prospective screening benchmark",
        "post-characterization sensitivity analysis",
    )
    return summary


def reaction_semantics_audit(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for column in ["c_min", "c_max", "c_const", "km_value"]:
        work[column] = pd.to_numeric(work.get(column), errors="coerce")
    rows = []
    for reaction, group in work.groupby("reaction_type", dropna=False):
        rows.append(
            {
                "reaction_type_raw": reaction,
                "varied_substrate": group["varied_substrate"].mode().iat[0],
                "canonical_reaction_system": group["canonical_reaction_system"].mode().iat[0],
                "n_rows": len(group),
                "n_sources": group["source_group_id"].nunique(),
                "c_min_median": group["c_min"].median(),
                "c_max_median": group["c_max"].median(),
                "c_const_median": group["c_const"].median(),
                "km_value_median_raw_units": group["km_value"].median(),
                "interpretation": (
                    "reaction order is retained because the first token denotes "
                    "the substrate varied for the reported Km"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("n_rows", ascending=False)


def design_only_substrate_stratified_performance(
    df: pd.DataFrame, repeats: int, seed: int
) -> pd.DataFrame:
    cohort = df.copy()
    cohort["leakage_graph_component_id"] = source_formula_protocol_component_ids(
        cohort.drop(columns=["vmax_value", "vmax_unit"], errors="ignore")
    )
    X, meta = build_design_only_feature_frame(cohort)
    y = cohort["target_log10_km"].astype(float)
    spec = FeatureSpec(meta["numeric_columns"], meta["categorical_columns"])
    model = model_suite(spec, seed)["catboost"]
    rows = []
    for repeat in range(repeats):
        for split_type in INTERNAL_SPLIT_ORDER:
            train_idx, test_idx, _ = make_split(
                cohort,
                split_type,
                seed + repeat,
                test_size=0.2,
                composition=X[meta["composition_columns"]],
                min_test_n=20,
            )
            pred = fit_predict(
                model, X.loc[train_idx], y.loc[train_idx], X.loc[test_idx]
            )
            fold = cohort.loc[test_idx, ["varied_substrate"]].copy()
            fold["target"] = y.loc[test_idx]
            fold["prediction"] = pred
            for substrate, group in fold.groupby("varied_substrate"):
                if len(group) < 2:
                    continue
                rows.append(
                    {
                        "feature_schema": "design_only_primary",
                        "split_type": split_type,
                        "varied_substrate": substrate,
                        "repeat": repeat,
                        "n_test": len(group),
                        **regression_metrics(group["target"], group["prediction"]),
                    }
                )
    per_repeat = pd.DataFrame(rows)
    summary = (
        per_repeat.groupby(
            ["feature_schema", "split_type", "varied_substrate"], as_index=False
        )
        .agg(
            n_repeats=("repeat", "nunique"),
            mean_n_test=("n_test", "mean"),
            min_n_test=("n_test", "min"),
            max_n_test=("n_test", "max"),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            spearman_mean=("spearman", "mean"),
            spearman_sd=("spearman", "std"),
        )
    )
    report_r2 = summary["varied_substrate"].isin(["TMB", "H2O2"]) & summary[
        "mean_n_test"
    ].ge(20)
    summary["r2_reporting_status"] = np.where(
        report_r2,
        "reported",
        "withheld: mean test stratum <20; use MAE and Spearman",
    )
    summary.loc[~report_r2, ["r2_mean", "r2_sd"]] = np.nan
    return summary


def reaction_source_association(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in ["reaction_type", "varied_substrate", "canonical_reaction_system"]:
        table = pd.crosstab(df[variable], df["source_group_id"])
        chi2, _, _, _ = chi2_contingency(table, correction=False)
        n = table.to_numpy().sum()
        phi2 = chi2 / n
        r, k = table.shape
        phi2_corrected = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
        r_corrected = r - ((r - 1) ** 2) / max(n - 1, 1)
        k_corrected = k - ((k - 1) ** 2) / max(n - 1, 1)
        denom = min(k_corrected - 1, r_corrected - 1)
        cramers_v = np.sqrt(phi2_corrected / denom) if denom > 0 else np.nan
        rows.append(
            {
                "categorical_variable": variable,
                "n_rows": n,
                "n_levels": r,
                "n_source_groups": k,
                "bias_corrected_cramers_v_with_source": cramers_v,
                "interpretation": (
                    "descriptive association only; high-cardinality source identity "
                    "can inflate uncorrected association measures"
                ),
            }
        )
    return pd.DataFrame(rows)


def substrate_stratified_performance(df: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_csv(
        "outputs/tables/internal_validation/internal_row_level_catboost_predictions.csv"
    )
    lookup = df.set_index("row_id")["varied_substrate"]
    pred["varied_substrate"] = pred["row_id"].map(lookup)
    rows = []
    for (split_type, substrate, repeat), group in pred.groupby(
        ["split_type", "varied_substrate", "repeat"]
    ):
        if len(group) < 2:
            continue
        rows.append(
            {
                "split_type": split_type,
                "varied_substrate": substrate,
                "repeat": repeat,
                "n_test": len(group),
                **regression_metrics(group["target_log10_km"], group["pred_log10_km"]),
            }
        )
    per_repeat = pd.DataFrame(rows)
    summary = (
        per_repeat.groupby(["split_type", "varied_substrate"], as_index=False)
        .agg(
            n_repeats=("repeat", "nunique"),
            mean_n_test=("n_test", "mean"),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            spearman_mean=("spearman", "mean"),
        )
    )
    summary["target_definition"] = "log10(Km/M), stratified by varied substrate"
    report_r2 = summary["varied_substrate"].isin(["TMB", "H2O2"]) & summary[
        "mean_n_test"
    ].ge(20)
    summary["r2_reporting_status"] = np.where(
        report_r2,
        "reported",
        "withheld: mean test stratum <20; use MAE and Spearman",
    )
    summary.loc[~report_r2, ["r2_mean", "r2_sd"]] = np.nan
    return summary


def matched_substrate_random_leakage(df: pd.DataFrame) -> pd.DataFrame:
    work = df.set_index("row_id")
    rows = []
    for repeat in range(20):
        manifest = pd.read_csv(f"outputs/split_manifests/random_repeat_{repeat}.csv")
        train_ids = manifest.loc[manifest["split"].eq("train"), "row_id"].astype(int)
        test_ids = manifest.loc[manifest["split"].eq("test"), "row_id"].astype(int)
        train = work.loc[train_ids]
        for definition in ["source_formula_pair", "protocol_matched_pair"]:
            eligible = 0
            supported = 0
            for row_id in test_ids:
                row = work.loc[row_id]
                if row["varied_substrate"] not in {"TMB", "H2O2"}:
                    continue
                counterpart = "H2O2" if row["varied_substrate"] == "TMB" else "TMB"
                candidates = train[
                    train["source_group_id"].eq(row["source_group_id"])
                    & train["standardized_formula"].eq(row["standardized_formula"])
                    & train["canonical_reaction_system"].eq(
                        row["canonical_reaction_system"]
                    )
                    & train["varied_substrate"].eq(counterpart)
                ]
                if definition == "protocol_matched_pair":
                    for column in ["ph", "temperature", "length", "width", "depth"]:
                        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
                        if pd.notna(value):
                            candidate_values = pd.to_numeric(
                                candidates[column], errors="coerce"
                            )
                            candidates = candidates[np.isclose(candidate_values, value)]
                    surface = str(row.get("surface", "")).strip().lower()
                    if surface and surface != "nan":
                        candidates = candidates[
                            candidates["surface"].astype(str).str.strip().str.lower().eq(surface)
                        ]
                eligible += 1
                supported += int(len(candidates) > 0)
            rows.append(
                {
                    "repeat": repeat,
                    "pair_definition": definition,
                    "eligible_tmb_h2o2_test_rows": eligible,
                    "test_rows_with_counter_substrate_training_pair": supported,
                    "fraction_with_training_pair": supported / eligible,
                }
            )
    raw = pd.DataFrame(rows)
    return (
        raw.groupby("pair_definition", as_index=False)
        .agg(
            n_repeats=("repeat", "nunique"),
            mean_eligible_test_rows=("eligible_tmb_h2o2_test_rows", "mean"),
            mean_fraction_with_training_pair=("fraction_with_training_pair", "mean"),
            sd_fraction_with_training_pair=("fraction_with_training_pair", "std"),
            min_fraction_with_training_pair=("fraction_with_training_pair", "min"),
            max_fraction_with_training_pair=("fraction_with_training_pair", "max"),
        )
        .assign(
            interpretation=(
                "fraction of random-split TMB/H2O2 test rows with an opposite-varied-"
                "substrate record from the same source/formula/reaction system in training"
            )
        )
    )


def fuzzy_cross_source_membership(
    df: pd.DataFrame,
    log10_km_tolerance: float = 0.1,
    ph_tolerance: float = 0.5,
) -> tuple[pd.Series, pd.DataFrame]:
    graph = nx.Graph()
    graph.add_nodes_from(df.index)
    group_cols = [
        "standardized_formula", "canonical_reaction_system", "varied_substrate"
    ]
    for _, group in df.groupby(group_cols, dropna=False):
        records = group[
            ["source_group_id", "target_log10_km", "ph"]
        ].copy()
        records["ph"] = pd.to_numeric(records["ph"], errors="coerce")
        indices = records.index.to_numpy()
        for i, left_idx in enumerate(indices):
            left = records.loc[left_idx]
            for right_idx in indices[i + 1 :]:
                right = records.loc[right_idx]
                if left["source_group_id"] == right["source_group_id"]:
                    continue
                if abs(float(left["target_log10_km"]) - float(right["target_log10_km"])) >= log10_km_tolerance:
                    continue
                if pd.isna(left["ph"]) or pd.isna(right["ph"]):
                    continue
                if abs(float(left["ph"]) - float(right["ph"])) >= ph_tolerance:
                    continue
                graph.add_edge(left_idx, right_idx)

    components = [sorted(c) for c in nx.connected_components(graph) if len(c) > 1]
    membership = pd.Series(False, index=df.index)
    rows = []
    for cluster_id, component in enumerate(components):
        membership.loc[component[1:]] = True
        subset = df.loc[component]
        rows.append(
            {
                "fuzzy_cluster_id": cluster_id,
                "cluster_size": len(component),
                "n_sources": subset["source_group_id"].nunique(),
                "standardized_formula": subset["standardized_formula"].iloc[0],
                "canonical_reaction_system": subset["canonical_reaction_system"].iloc[0],
                "varied_substrate": subset["varied_substrate"].iloc[0],
                "target_log10_km_range": subset["target_log10_km"].max()
                - subset["target_log10_km"].min(),
                "ph_range": pd.to_numeric(subset["ph"], errors="coerce").max()
                - pd.to_numeric(subset["ph"], errors="coerce").min(),
                "removal_rule": "keep first row_id; remove remaining component members",
            }
        )
    return membership, pd.DataFrame(rows)


def fuzzy_sensitivity(
    df: pd.DataFrame, membership: pd.Series, repeats: int, seed: int
) -> pd.DataFrame:
    cohort = df.loc[~membership].copy().reset_index(drop=True)
    cohort["leakage_graph_component_id"] = source_formula_protocol_component_ids(cohort)
    X, meta = build_feature_frame(cohort)
    y = cohort["target_log10_km"].astype(float)
    spec = FeatureSpec(meta["numeric_columns"], meta["categorical_columns"])
    rows = []
    for repeat in range(repeats):
        rep_seed = seed + repeat
        for split_type in ["random", "leakage_graph"]:
            train_idx, test_idx, _ = make_split(
                cohort,
                split_type,
                rep_seed,
                test_size=0.2,
                composition=X[meta["composition_columns"]],
                min_test_n=20,
            )
            models = model_suite(spec, rep_seed)
            for model_name in ["median", "elasticnet", "extratrees", "catboost"]:
                pred = fit_predict(
                    models[model_name],
                    X.loc[train_idx],
                    y.loc[train_idx],
                    X.loc[test_idx],
                )
                rows.append(
                    {
                        "cohort": "fuzzy_cross_source_keep_one",
                        "n_cohort": len(cohort),
                        "n_rows_excluded": int(membership.sum()),
                        "repeat": repeat,
                        "split_type": split_type,
                        "model": model_name,
                        "pred_min": float(np.min(pred)),
                        "pred_max": float(np.max(pred)),
                        "max_abs_error": float(
                            np.max(np.abs(y.loc[test_idx].to_numpy() - pred))
                        ),
                        "target_min": float(y.loc[test_idx].min()),
                        "target_max": float(y.loc[test_idx].max()),
                        **regression_metrics(y.loc[test_idx], pred),
                    }
                )
        print(f"fuzzy sensitivity: completed repeat {repeat + 1}/{repeats}", flush=True)
    raw = pd.DataFrame(rows)
    return (
        raw.groupby(
            ["cohort", "n_cohort", "n_rows_excluded", "split_type", "model"],
            as_index=False,
        )
        .agg(
            n_repeats=("repeat", "nunique"),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            spearman_mean=("spearman", "mean"),
            pred_min_across_repeats=("pred_min", "min"),
            pred_max_across_repeats=("pred_max", "max"),
            max_abs_error_across_repeats=("max_abs_error", "max"),
            target_min_across_repeats=("target_min", "min"),
            target_max_across_repeats=("target_max", "max"),
        )
    )


def elasticnet_failure_diagnostic(
    df: pd.DataFrame, membership: pd.Series, repeats: int, seed: int
) -> pd.DataFrame:
    cohort = df.loc[~membership].copy().reset_index(drop=True)
    cohort["leakage_graph_component_id"] = source_formula_protocol_component_ids(cohort)
    X, meta = build_feature_frame(cohort)
    y = cohort["target_log10_km"].astype(float)
    spec = FeatureSpec(meta["numeric_columns"], meta["categorical_columns"])
    rows = []
    for repeat in range(repeats):
        rep_seed = seed + repeat
        train_idx, test_idx, _ = make_split(
            cohort,
            "leakage_graph",
            rep_seed,
            test_size=0.2,
            composition=X[meta["composition_columns"]],
            min_test_n=20,
        )
        model = clone(model_suite(spec, rep_seed)["elasticnet"])
        model.fit(X.loc[train_idx], y.loc[train_idx])
        pred = model.predict(X.loc[test_idx])
        abs_error = np.abs(y.loc[test_idx].to_numpy() - pred)
        worst_pos = int(np.argmax(abs_error))
        worst_idx = int(test_idx[worst_pos])
        train_numeric = X.loc[train_idx, meta["numeric_columns"]].apply(
            pd.to_numeric, errors="coerce"
        )
        worst_numeric = X.loc[worst_idx, meta["numeric_columns"]].apply(
            pd.to_numeric, errors="coerce"
        )
        scale = train_numeric.std(ddof=0).replace(0, np.nan)
        z = ((worst_numeric - train_numeric.mean()) / scale).abs().sort_values(
            ascending=False
        )
        rows.append(
            {
                "repeat": repeat,
                "seed": rep_seed,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "prediction_min": float(np.min(pred)),
                "prediction_max": float(np.max(pred)),
                "target_min": float(y.loc[test_idx].min()),
                "target_max": float(y.loc[test_idx].max()),
                "worst_row_id": int(cohort.loc[worst_idx, "row_id"]),
                "worst_formula": cohort.loc[worst_idx, "standardized_formula"],
                "worst_varied_substrate": cohort.loc[worst_idx, "varied_substrate"],
                "worst_target": float(y.loc[worst_idx]),
                "worst_prediction": float(pred[worst_pos]),
                "worst_abs_error": float(abs_error[worst_pos]),
                "largest_numeric_train_z_feature": z.index[0] if len(z) else pd.NA,
                "largest_numeric_train_z": float(z.iloc[0]) if len(z) else np.nan,
                "diagnosis": (
                    "ElasticNet includes StandardScaler; failure is an unbounded linear "
                    "extrapolation under component shift, not absent numeric scaling"
                ),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
