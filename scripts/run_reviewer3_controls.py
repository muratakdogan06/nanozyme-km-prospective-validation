#!/usr/bin/env python
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import prepare_modeling_data, read_table
from src.diagnostics import source_formula_protocol_component_ids
from src.evaluate import regression_metrics
from src.features import build_design_only_feature_frame
from src.models import FeatureSpec, fit_predict, model_suite, sklearn_preprocessor
from src.splits import make_split
from scripts.run_restricted_feature_validation import (
    _prepare_internal_shared_substrate,
    _restricted_features,
)


SPLIT_TYPES = [
    "random",
    "source_group",
    "formula_group",
    "composition_cluster",
    "leakage_graph",
]
SEED = 2026
REPEATS = 20


def main() -> None:
    out = Path("outputs/tables/reviewer3_controls")
    out.mkdir(parents=True, exist_ok=True)

    full, _ = prepare_modeling_data(read_table("data/nanozymes.csv"))
    full["leakage_graph_component_id"] = source_formula_protocol_component_ids(full)

    dispersion_rows, dispersion_summary = matched_context_dispersion(full)
    dispersion_rows.to_csv(out / "matched_context_intersource_dispersion.csv", index=False)
    dispersion_summary.to_csv(out / "matched_context_intersource_dispersion_summary.csv", index=False)

    knn_raw, knn_summary = restricted_nearest_neighbor_sensitivity(full)
    knn_raw.to_csv(out / "restricted_knn_metrics.csv", index=False)
    knn_summary.to_csv(out / "restricted_knn_summary.csv", index=False)

    scramble_raw, scramble_summary = design_only_y_scrambling(full)
    scramble_raw.to_csv(out / "design_only_y_scrambling_metrics.csv", index=False)
    scramble_summary.to_csv(out / "design_only_y_scrambling_summary.csv", index=False)

    association = source_association_intervals(full)
    association.to_csv(out / "reaction_source_association_bootstrap.csv", index=False)

    model_matrix = pd.read_csv(
        "outputs/tables/reviewer2_sensitivity/design_only_vs_post_characterization.csv"
    )
    model_matrix = model_matrix.loc[
        model_matrix["feature_schema"].eq("design_only_primary"),
        [
            "split_type",
            "model",
            "mae_mean",
            "mae_sd",
            "r2_mean",
            "r2_sd",
            "spearman_mean",
        ],
    ]
    model_matrix.to_csv(out / "design_only_model_split_matrix.csv", index=False)

    ablation = pd.read_csv("outputs/tables/supp_table_s17_feature_group_ablation_summary.csv")
    schema = pd.read_csv(
        "outputs/tables/reviewer2_sensitivity/design_only_vs_post_characterization.csv"
    )
    baseline = schema.loc[
        schema["feature_schema"].eq("post_characterization_sensitivity")
        & schema["model"].eq("median"),
        ["split_type", "mae_mean"],
    ].rename(columns={"mae_mean": "median_baseline_mae_mean"})
    ablation = ablation.merge(baseline, on="split_type", how="left", validate="many_to_one")
    ablation["baseline_normalized_skill"] = (
        1 - ablation["mae_mean"] / ablation["median_baseline_mae_mean"]
    )
    ablation.to_csv(out / "supp_table_s17_feature_group_ablation_with_skill.csv", index=False)


def matched_context_dispersion(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["ph"] = pd.to_numeric(work["ph"], errors="coerce")
    work["temperature"] = pd.to_numeric(work["temperature"], errors="coerce")
    work = work.loc[
        work["ph"].notna()
        & work["temperature"].notna()
        & work["varied_substrate"].isin(["TMB", "H2O2"])
    ].copy()
    keys = ["standardized_formula", "varied_substrate", "ph", "temperature"]
    source_level = (
        work.groupby(keys + ["source_group_id"], as_index=False)
        .agg(
            source_median_log10_km=("target_log10_km", "median"),
            n_source_records=("row_id", "size"),
        )
    )
    rows: list[dict] = []
    for key, group in source_level.groupby(keys, sort=True):
        if len(group) < 2:
            continue
        pair_diffs = [
            abs(float(a) - float(b))
            for a, b in itertools.combinations(group["source_median_log10_km"], 2)
        ]
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_sources": int(len(group)),
                "n_records_before_source_collapse": int(group["n_source_records"].sum()),
                "n_source_pairs": int(len(pair_diffs)),
                "median_pairwise_abs_difference_log10_km": float(np.median(pair_diffs)),
                "mean_pairwise_abs_difference_log10_km": float(np.mean(pair_diffs)),
                "min_pairwise_abs_difference_log10_km": float(np.min(pair_diffs)),
                "max_pairwise_abs_difference_log10_km": float(np.max(pair_diffs)),
            }
        )
    result = pd.DataFrame(rows)
    values = result["median_pairwise_abs_difference_log10_km"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    boot = np.empty(10000, dtype=float)
    for i in range(len(boot)):
        boot[i] = np.median(rng.choice(values, size=len(values), replace=True))
    pairwise_median = float(np.median(values))
    normal_sigma = pairwise_median / (0.6744897501960817 * np.sqrt(2))
    normal_single_measurement_mae = normal_sigma * np.sqrt(2 / np.pi)
    summary = pd.DataFrame(
        [
            {
                "analysis": "exact_formula_substrate_ph_temperature_matched_intersource_dispersion",
                "n_matched_strata": int(len(result)),
                "n_source_level_observations": int(result["n_sources"].sum()),
                "n_underlying_records": int(result["n_records_before_source_collapse"].sum()),
                "median_stratum_pairwise_abs_difference_log10_km": pairwise_median,
                "bootstrap_95_ci_low": float(np.quantile(boot, 0.025)),
                "bootstrap_95_ci_high": float(np.quantile(boot, 0.975)),
                "mean_stratum_pairwise_abs_difference_log10_km": float(np.mean(values)),
                "normal_iid_implied_single_measurement_sigma": float(normal_sigma),
                "normal_iid_implied_single_measurement_mae": float(
                    normal_single_measurement_mae
                ),
                "bootstrap_unit": "matched formula-substrate-pH-temperature stratum",
                "n_boot": 10000,
                "seed": SEED,
                "interpretation": (
                    "Exploratory matched-context intersource dispersion; not an irreducible "
                    "noise floor because unrecorded protocol and material differences may remain."
                ),
            }
        ]
    )
    return result, summary


def restricted_nearest_neighbor_sensitivity(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = _prepare_internal_shared_substrate(df)
    work["leakage_graph_component_id"] = source_formula_protocol_component_ids(work)
    X, comp_cols = _restricted_features(work)
    y = work["target_log10_km"].astype(float)
    spec = FeatureSpec(comp_cols, ["substrate_for_km_norm"])
    rows: list[dict] = []
    for repeat in range(REPEATS):
        seed = SEED + repeat
        for split_type in SPLIT_TYPES:
            train_idx, test_idx, _ = make_split(
                work,
                split_type,
                seed,
                test_size=0.2,
                composition=X[comp_cols],
                min_test_n=20,
            )
            models = {"median": model_suite(spec, seed)["median"]}
            for n_neighbors in [1, 3, 5]:
                models[f"knn_k{n_neighbors}"] = Pipeline(
                    [
                        ("preprocess", sklearn_preprocessor(spec, scale_numeric=True)),
                        (
                            "model",
                            KNeighborsRegressor(
                                n_neighbors=n_neighbors,
                                weights="uniform",
                                metric="euclidean",
                            ),
                        ),
                    ]
                )
            for model_name, model in models.items():
                pred = fit_predict(model, X.loc[train_idx], y.loc[train_idx], X.loc[test_idx])
                rows.append(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "split_type": split_type,
                        "model": model_name,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        **regression_metrics(y.loc[test_idx], pred),
                    }
                )
    raw = pd.DataFrame(rows)
    agg = raw.groupby(["split_type", "model"], as_index=False).agg(
        n_repeats=("repeat", "nunique"),
        mean_n_train=("n_train", "mean"),
        mean_n_test=("n_test", "mean"),
        mae_mean=("mae", "mean"),
        mae_sd=("mae", "std"),
        r2_mean=("r2", "mean"),
        r2_sd=("r2", "std"),
    )
    baseline = agg.loc[agg["model"].eq("median"), ["split_type", "mae_mean"]].rename(
        columns={"mae_mean": "median_baseline_mae"}
    )
    summary = agg.loc[~agg["model"].eq("median")].merge(
        baseline, on="split_type", how="left", validate="many_to_one"
    )
    summary["skill"] = 1 - summary["mae_mean"] / summary["median_baseline_mae"]
    summary["distance_metric"] = "Euclidean after train-only standardization and one-hot encoding"
    return raw, summary


def design_only_y_scrambling(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X, meta = build_design_only_feature_frame(df)
    y = df["target_log10_km"].astype(float)
    spec = FeatureSpec(meta["numeric_columns"], meta["categorical_columns"])
    rows: list[dict] = []
    for repeat in range(REPEATS):
        seed = SEED + repeat
        for split_number, split_type in enumerate(SPLIT_TYPES):
            train_idx, test_idx, _ = make_split(
                df,
                split_type,
                seed,
                test_size=0.2,
                composition=X[meta["composition_columns"]],
                min_test_n=20,
            )
            permutation_seed = 100000 + 100 * repeat + split_number
            rng = np.random.default_rng(permutation_seed)
            y_scrambled = pd.Series(
                rng.permutation(y.loc[train_idx].to_numpy()), index=train_idx
            )
            model = model_suite(spec, seed)["catboost"]
            pred = fit_predict(model, X.loc[train_idx], y_scrambled, X.loc[test_idx])
            rows.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "permutation_seed": permutation_seed,
                    "split_type": split_type,
                    "model": "catboost_y_scrambled",
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    **regression_metrics(y.loc[test_idx], pred),
                }
            )
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["split_type", "model"], as_index=False).agg(
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
    return raw, summary


def source_association_intervals(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for offset, variable in enumerate(
        ["reaction_type", "varied_substrate", "canonical_reaction_system"]
    ):
        table = pd.crosstab(df[variable], df["source_group_id"])
        observed = bias_corrected_cramers_v(table.to_numpy(float))
        columns = table.to_numpy(float).T
        rng = np.random.default_rng(SEED + offset)
        boot = np.empty(5000, dtype=float)
        for i in range(len(boot)):
            sampled = columns[rng.integers(0, len(columns), size=len(columns))].T
            boot[i] = bias_corrected_cramers_v(sampled)
        rows.append(
            {
                "categorical_variable": variable,
                "n_rows": int(table.to_numpy().sum()),
                "n_levels": int(table.shape[0]),
                "n_source_groups": int(table.shape[1]),
                "bias_corrected_cramers_v_with_source": observed,
                "source_cluster_bootstrap_95_ci_low": float(np.quantile(boot, 0.025)),
                "source_cluster_bootstrap_95_ci_high": float(np.quantile(boot, 0.975)),
                "n_boot": 5000,
                "seed": SEED + offset,
                "bootstrap_unit": "source group",
            }
        )
    return pd.DataFrame(rows)


def bias_corrected_cramers_v(table: np.ndarray) -> float:
    n = float(table.sum())
    if n <= 1:
        return float("nan")
    row_total = table.sum(axis=1, keepdims=True)
    col_total = table.sum(axis=0, keepdims=True)
    expected = row_total @ col_total / n
    valid = expected > 0
    chi2 = float(np.sum(((table - expected) ** 2)[valid] / expected[valid]))
    phi2 = chi2 / n
    r, k = table.shape
    phi2_corrected = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corrected = r - ((r - 1) ** 2) / (n - 1)
    k_corrected = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corrected - 1, r_corrected - 1)
    return float(np.sqrt(phi2_corrected / denom)) if denom > 0 else float("nan")


if __name__ == "__main__":
    main()
