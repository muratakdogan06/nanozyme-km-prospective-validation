#!/usr/bin/env python
"""Reviewer-facing sensitivity analyses for the connectivity benchmark."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import near_duplicate_clusters, prepare_modeling_data, read_table
from src.diagnostics import source_formula_protocol_component_ids
from src.evaluate import regression_metrics
from src.features import build_design_only_feature_frame, build_feature_frame
from src.models import FeatureSpec, fit_predict, model_suite, require_catboost
from src.splits import make_split


SPLITS = ["random", "source_group", "formula_group", "composition_cluster", "leakage_graph"]
BASE_DUPLICATE_KEYS = [
    "source_group_id", "standardized_formula", "activity", "reaction_type", "ph",
    "temperature", "length", "width", "depth", "surface", "c_min", "c_max", "c_const",
]
LEARNING_FRACTIONS = [0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00]
CONTINUOUS_BOOTSTRAP_N = 10_000
# Quantile regression requires a linear-program fit for every resample; 500
# component resamples provide the prespecified heavy-tail sensitivity interval
# without conflating it with the 10,000-resample primary OLS analysis.
MEDIAN_REGRESSION_BOOTSTRAP_N = 500
SEED_SENSITIVITY_COUNT = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/nanozymes.csv")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--analysis",
        choices=[
            "all",
            "diagnostics",
            "bins",
            "design_continuous",
            "design_mechanism",
            "learning",
            "duplicates",
        ],
        default="all",
    )
    args = parser.parse_args()
    if args.repeats < 20 and args.analysis in {"all", "learning", "duplicates"}:
        print(
            "WARNING: fewer than 20 repeats requested; output uncertainty columns "
            "will be labeled as observed ranges, not 95% confidence intervals.",
            file=sys.stderr,
        )
    require_catboost()

    out = Path("outputs/tables/reviewer2_sensitivity")
    out.mkdir(parents=True, exist_ok=True)
    df, _ = prepare_modeling_data(read_table(args.data))
    df["leakage_graph_component_id"] = source_formula_protocol_component_ids(df)

    if args.analysis == "bins":
        component_bootstrap_connection_bins().to_csv(
            out / "connection_bins_component_bootstrap.csv", index=False
        )
        print(f"Component-bootstrap connection bins complete: {out}")
        return
    if args.analysis == "design_mechanism":
        _write_design_only_continuous_outputs(df, args.repeats, args.seed, out)
        design_only_learning_curves(df, args.repeats, args.seed).to_csv(
            out / "design_only_learning_curves.csv", index=False
        )
        print(f"Design-only mechanism analyses complete: {out}")
        return
    if args.analysis == "design_continuous":
        _write_design_only_continuous_outputs(df, args.repeats, args.seed, out)
        print(f"Design-only continuous analysis complete: {out}")
        return

    if args.analysis in {"all", "diagnostics"}:
        continuous_connectivity_analyses(df).to_csv(
            out / "continuous_connectivity_association.csv", index=False
        )
        split_dispersion().to_csv(out / "split_r2_dispersion.csv", index=False)
        largest_component_profile(df).to_csv(out / "largest_component_profile.csv", index=False)
        largest_component_articulation_points(df).to_csv(
            out / "largest_component_articulation_points.csv", index=False
        )
        component_bootstrap_connection_bins().to_csv(
            out / "connection_bins_component_bootstrap.csv", index=False
        )
    if args.analysis in {"all", "learning"}:
        learning_curves(df, args.repeats, args.seed).to_csv(
            out / "learning_curves.csv", index=False
        )
    if args.analysis in {"all", "duplicates"}:
        near_duplicate_sensitivity(df, args.repeats, args.seed).to_csv(
            out / "near_duplicate_exclusion_sensitivity.csv", index=False
        )
    print(f"Reviewer 2 sensitivity analyses complete: {out}")


def component_bootstrap_connection_bins(
    n_boot: int = 5000, seed: int = 2026
) -> pd.DataFrame:
    pred = pd.read_csv(
        "outputs/tables/restricted_feature_validation/"
        "restricted_feature_internal_predictions.csv"
    )
    selected = pred.loc[
        pred["split_type"].isin(["random", "eligible_pool_random", "leakage_graph"])
    ].copy()
    keys = [
        "split_type",
        "repeat",
        "internal_row_id",
        "leakage_graph_component_id",
        "train_same_component_n",
    ]
    paired = selected.pivot_table(
        index=keys,
        columns="model",
        values="abs_error_log10_km",
        aggfunc="first",
    ).reset_index()
    paired["connection_bin"] = pd.cut(
        paired["train_same_component_n"],
        bins=[-0.1, 0, 5, 25, 100, np.inf],
        labels=["0", "1-5", "6-25", "26-100", ">100"],
    ).astype(str)
    rng = np.random.default_rng(seed)
    rows = []
    for bin_label, group in paired.groupby("connection_bin", sort=False):
        if group.empty:
            continue
        median_error = group["median"].to_numpy(float)
        model_error = group["catboost"].to_numpy(float)
        clusters = group["leakage_graph_component_id"].astype(str).to_numpy()
        unique_clusters = np.unique(clusters)
        draws = []
        for _ in range(n_boot):
            sampled = rng.choice(
                unique_clusters, size=len(unique_clusters), replace=True
            )
            idx = np.concatenate(
                [np.flatnonzero(clusters == cluster) for cluster in sampled]
            )
            draws.append(
                1.0 - model_error[idx].mean() / median_error[idx].mean()
            )
        skill = 1.0 - model_error.mean() / median_error.mean()
        estimable = len(unique_clusters) >= 2
        rows.append(
            {
                "connection_bin": bin_label,
                "n_test_predictions": len(group),
                "n_unique_rows": group["internal_row_id"].nunique(),
                "n_components": len(unique_clusters),
                "skill_score_catboost_vs_median": skill,
                "skill_score_ci95_low": np.quantile(draws, 0.025) if estimable else np.nan,
                "skill_score_ci95_high": np.quantile(draws, 0.975) if estimable else np.nan,
                "bootstrap_unit": "leakage_graph_component_id",
                "interval_status": (
                    "component-bootstrap interval"
                    if estimable
                    else "not estimable: bin contains one graph component"
                ),
                "analysis_role": "descriptive component-cluster bootstrap",
            }
        )
    return pd.DataFrame(rows)


def design_only_continuous_connectivity(
    df: pd.DataFrame, repeats: int, seed: int
) -> pd.DataFrame:
    design = _design_only_connectivity_prediction_design(df, repeats, seed)
    return _design_only_continuous_results(design, seed)


def _write_design_only_continuous_outputs(
    df: pd.DataFrame, repeats: int, seed: int, out: Path
) -> None:
    design = _design_only_connectivity_prediction_design(df, repeats, seed)
    _design_only_continuous_results(design, seed).to_csv(
        out / "design_only_continuous_connectivity.csv", index=False
    )
    _continuous_seed_sensitivity(design, seed).to_csv(
        out / "design_only_continuous_connectivity_seed_sensitivity.csv",
        index=False,
    )


def _design_only_connectivity_prediction_design(
    df: pd.DataFrame, repeats: int, seed: int
) -> pd.DataFrame:
    cohort = df.drop(columns=["vmax_value", "vmax_unit"], errors="ignore").copy()
    cohort["leakage_graph_component_id"] = source_formula_protocol_component_ids(
        cohort
    )
    X, meta = build_design_only_feature_frame(cohort)
    y = cohort["target_log10_km"].astype(float)
    spec = FeatureSpec(meta["numeric_columns"], meta["categorical_columns"])
    records = []
    for repeat in range(repeats):
        rep_seed = seed + repeat
        models = model_suite(spec, rep_seed)
        train_idx, test_idx, _ = make_split(
            cohort,
            "random",
            rep_seed,
            test_size=0.2,
            composition=X[meta["composition_columns"]],
            min_test_n=20,
        )
        train_counts = cohort.loc[
            train_idx, "leakage_graph_component_id"
        ].value_counts()
        support = (
            cohort.loc[test_idx, "leakage_graph_component_id"]
            .map(train_counts)
            .fillna(0)
            .astype(int)
        )
        nearest_distance = _composition_nearest_distance(
            X, meta["composition_columns"], train_idx, test_idx
        )
        median_pred = fit_predict(
            models["median"], X.loc[train_idx], y.loc[train_idx], X.loc[test_idx]
        )
        model_pred = fit_predict(
            models["catboost"], X.loc[train_idx], y.loc[train_idx], X.loc[test_idx]
        )
        records.append(
            pd.DataFrame(
                {
                    "internal_row_id": cohort.loc[test_idx, "row_id"].astype(int),
                    "leakage_graph_component_id": cohort.loc[
                        test_idx, "leakage_graph_component_id"
                    ].astype(int),
                    "train_same_component_n": support.to_numpy(),
                    "nearest_train_feature_distance": nearest_distance,
                    "split_type": "random",
                    "model_advantage": np.abs(
                        y.loc[test_idx].to_numpy() - median_pred
                    )
                    - np.abs(y.loc[test_idx].to_numpy() - model_pred),
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def _design_only_continuous_results(
    design: pd.DataFrame, bootstrap_seed: int
) -> pd.DataFrame:
    result = pd.DataFrame(
        _connectivity_specification_rows(
            design,
            schema="design_only_primary",
            spec_name="within_random",
            fixed_effects=False,
            n_boot=CONTINUOUS_BOOTSTRAP_N,
            median_n_boot=MEDIAN_REGRESSION_BOOTSTRAP_N,
            bootstrap_seed=bootstrap_seed,
        )
    )
    primary = (
        result["adjustment"].eq("unadjusted")
        & result["estimator"].eq("OLS")
        & result["bootstrap_cluster"].eq("leakage_graph_component_id")
    )
    result.loc[primary, "analysis_role"] = (
        "primary continuous connectivity analysis"
    )
    result.loc[~primary, "analysis_role"] = (
        "prespecified robustness or bootstrap sensitivity"
    )
    return result


def _continuous_seed_sensitivity(
    design: pd.DataFrame, base_seed: int
) -> pd.DataFrame:
    working = design.copy()
    working["log1p_same_component_n"] = np.log1p(
        working["train_same_component_n"].astype(float)
    )
    predictors = ["log1p_same_component_n"]
    columns = [
        "internal_row_id",
        "leakage_graph_component_id",
        "model_advantage",
        *predictors,
    ]
    working = working[columns].replace([np.inf, -np.inf], np.nan).dropna()
    coefficient_index = predictors.index("log1p_same_component_n") + 1
    beta = _ols(
        working[predictors].to_numpy(float),
        working["model_advantage"].to_numpy(float),
    )[coefficient_index]
    rows = []
    for bootstrap_seed in range(base_seed, base_seed + SEED_SENSITIVITY_COUNT):
        draws = _cluster_bootstrap_coefficients(
            working,
            predictors,
            cluster_col="leakage_graph_component_id",
            n_boot=CONTINUOUS_BOOTSTRAP_N,
            seed=bootstrap_seed,
        )
        rows.append(
            {
                "feature_schema": "design_only_primary",
                "specification": "within_random_unadjusted_OLS",
                "bootstrap_cluster": "leakage_graph_component_id",
                "bootstrap_seed": bootstrap_seed,
                "n_boot": CONTINUOUS_BOOTSTRAP_N,
                "coefficient": beta,
                "bootstrap_ci95_low": np.quantile(
                    draws[:, coefficient_index], 0.025
                ),
                "bootstrap_ci95_high": np.quantile(
                    draws[:, coefficient_index], 0.975
                ),
                "n_predictions": len(working),
                "n_unique_rows": working["internal_row_id"].nunique(),
                "n_components": working[
                    "leakage_graph_component_id"
                ].nunique(),
                "analysis_role": (
                    "bootstrap Monte Carlo seed sensitivity for primary analysis"
                ),
            }
        )
    return pd.DataFrame(rows)


def design_only_learning_curves(
    df: pd.DataFrame, repeats: int, seed: int
) -> pd.DataFrame:
    cohort = df.drop(columns=["vmax_value", "vmax_unit"], errors="ignore").copy()
    cohort["leakage_graph_component_id"] = source_formula_protocol_component_ids(
        cohort
    )
    X, meta = build_design_only_feature_frame(cohort)
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
            permutation = np.random.default_rng(rep_seed).permutation(train_idx)
            for fraction in LEARNING_FRACTIONS:
                n_train = max(20, int(round(len(train_idx) * fraction)))
                sub_train = permutation[:n_train]
                support = _component_support(cohort, sub_train, test_idx)
                models = model_suite(spec, rep_seed)
                for model_name in ["median", "extratrees", "catboost"]:
                    pred = fit_predict(
                        models[model_name],
                        X.loc[sub_train],
                        y.loc[sub_train],
                        X.loc[test_idx],
                    )
                    rows.append(
                        {
                            "feature_schema": "design_only_primary",
                            "repeat": repeat,
                            "seed": rep_seed,
                            "split_type": split_type,
                            "train_fraction": fraction,
                            "n_train": len(sub_train),
                            "n_test": len(test_idx),
                            "model": model_name,
                            **support,
                            **regression_metrics(y.loc[test_idx], pred),
                        }
                    )
        print(
            f"design-only learning curves: completed repeat {repeat + 1}/{repeats}",
            flush=True,
        )
    return _summarize_sensitivity(
        pd.DataFrame(rows),
        ["feature_schema", "split_type", "train_fraction", "model"],
        repeats,
    )


def _composition_nearest_distance(
    X: pd.DataFrame,
    composition_columns: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    chunk_size: int = 256,
) -> np.ndarray:
    train = X.loc[train_idx, composition_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    medians = train.median(numeric_only=True)
    scale = train.std(ddof=0).replace(0, 1.0).fillna(1.0)
    train_values = (
        (train.fillna(medians).fillna(0.0) - medians) / scale
    ).to_numpy(float)
    test_values = (
        (
            X.loc[test_idx, composition_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(medians)
            .fillna(0.0)
            - medians
        )
        / scale
    ).to_numpy(float)
    nearest = np.empty(len(test_idx), dtype=float)
    for start in range(0, len(test_idx), chunk_size):
        stop = min(start + chunk_size, len(test_idx))
        diff = test_values[start:stop, None, :] - train_values[None, :, :]
        nearest[start:stop] = np.sqrt(
            np.min(np.sum(diff * diff, axis=2), axis=1)
        )
    return nearest


def continuous_connectivity_analyses(df: pd.DataFrame) -> pd.DataFrame:
    restricted = _restricted_prediction_design()
    full = _full_schema_prediction_design(df)
    results = []
    for schema, design in [("restricted_shared_schema", restricted), ("full_schema", full)]:
        available_splits = set(design["split_type"])
        specs = [
            ("within_random", ["random"], False),
            (
                "pooled_with_split_fixed_effects",
                [s for s in ["random", "eligible_pool_random", "leakage_graph"] if s in available_splits],
                True,
            ),
            (
                "pooled_naive_reference",
                [s for s in ["random", "eligible_pool_random", "leakage_graph"] if s in available_splits],
                False,
            ),
        ]
        for spec_name, split_subset, fixed_effects in specs:
            subset = design[design["split_type"].isin(split_subset)].copy()
            results.extend(
                _connectivity_specification_rows(
                    subset, schema, spec_name, fixed_effects=fixed_effects
                )
            )
    return pd.DataFrame(results)


def _restricted_prediction_design() -> pd.DataFrame:
    pred = pd.read_csv(
        "outputs/tables/restricted_feature_validation/restricted_feature_internal_predictions.csv"
    )
    keys = [
        "internal_row_id", "repeat", "split_type", "leakage_graph_component_id",
        "train_same_component_n", "nearest_train_feature_distance",
    ]
    wide = pred.pivot_table(
        index=keys, columns="model", values="abs_error_log10_km", aggfunc="first"
    ).reset_index()
    wide = wide.dropna(subset=["median", "catboost"]).copy()
    wide["model_advantage"] = wide["median"] - wide["catboost"]
    return wide


def _full_schema_prediction_design(df: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_csv(
        "outputs/tables/internal_validation/internal_row_level_catboost_predictions.csv"
    )
    pred = pred[pred["split_type"].isin(["random", "leakage_graph"])].copy()
    row_lookup = df.set_index("row_id")
    records = []
    for (repeat, split_type), group in pred.groupby(["repeat", "split_type"]):
        manifest = pd.read_csv(
            f"outputs/split_manifests/{split_type}_repeat_{int(repeat)}.csv"
        )
        train_ids = manifest.loc[manifest["split"].eq("train"), "row_id"].astype(int)
        train = row_lookup.loc[train_ids]
        train_component_counts = train["leakage_graph_component_id"].value_counts()
        train_median = float(train["target_log10_km"].median())
        temp = group.copy()
        temp["internal_row_id"] = temp["row_id"].astype(int)
        temp["leakage_graph_component_id"] = temp["internal_row_id"].map(
            row_lookup["leakage_graph_component_id"]
        )
        temp["train_same_component_n"] = (
            temp["leakage_graph_component_id"].map(train_component_counts).fillna(0).astype(int)
        )
        temp["nearest_train_feature_distance"] = temp["composition_nearest_train_distance"]
        temp["median"] = (temp["target_log10_km"] - train_median).abs()
        temp["catboost"] = temp["abs_error_log10_km"]
        temp["model_advantage"] = temp["median"] - temp["catboost"]
        records.append(temp)
    return pd.concat(records, ignore_index=True)


def _connectivity_specification_rows(
    data: pd.DataFrame,
    schema: str,
    spec_name: str,
    fixed_effects: bool,
    n_boot: int = 2000,
    median_n_boot: int = 500,
    bootstrap_seed: int = 2026,
) -> list[dict]:
    data = data.copy()
    data["log1p_same_component_n"] = np.log1p(data["train_same_component_n"].astype(float))
    rows = []
    for adjustment, base_predictors in [
        ("unadjusted", ["log1p_same_component_n"]),
        (
            "distance_adjusted_sensitivity",
            ["log1p_same_component_n", "nearest_train_feature_distance"],
        ),
    ]:
        working = data.copy()
        predictors = list(base_predictors)
        if fixed_effects and working["split_type"].nunique() > 1:
            dummies = pd.get_dummies(
                working["split_type"], prefix="split", drop_first=True, dtype=float
            )
            working = pd.concat([working, dummies], axis=1)
            predictors.extend(dummies.columns.tolist())
        columns = [
            "internal_row_id", "leakage_graph_component_id", "model_advantage", *predictors
        ]
        design = working[columns].replace([np.inf, -np.inf], np.nan).dropna()
        beta = _ols(design[predictors].to_numpy(float), design["model_advantage"].to_numpy(float))
        for cluster_col in ["internal_row_id", "leakage_graph_component_id"]:
            boot = _cluster_bootstrap_coefficients(
                design,
                predictors,
                cluster_col=cluster_col,
                n_boot=n_boot,
                seed=bootstrap_seed,
            )
            rows.append(
                {
                    "feature_schema": schema,
                    "specification": spec_name,
                    "included_splits": ";".join(sorted(working["split_type"].unique())),
                    "adjustment": adjustment,
                    "estimator": "OLS",
                    "bootstrap_cluster": cluster_col,
                    "outcome": "median_abs_error_minus_catboost_abs_error",
                    "predictor": "log1p_same_component_n",
                    "coefficient": beta[predictors.index("log1p_same_component_n") + 1],
                    "bootstrap_ci95_low": np.quantile(
                        boot[:, predictors.index("log1p_same_component_n") + 1], 0.025
                    ),
                    "bootstrap_ci95_high": np.quantile(
                        boot[:, predictors.index("log1p_same_component_n") + 1], 0.975
                    ),
                    "bootstrap_seed": bootstrap_seed,
                    "n_boot": n_boot,
                    "n_predictions": len(design),
                    "n_unique_rows": design["internal_row_id"].nunique(),
                    "n_components": design["leakage_graph_component_id"].nunique(),
                    "analysis_role": "descriptive sensitivity analysis; not a confirmatory test",
                    "causal_caution": (
                        "distance adjustment may over-adjust if feature distance mediates "
                        "the connectivity-performance relationship"
                    ),
                }
            )
        quantile = _median_regression(design, predictors)
        if quantile is not None:
            coefficient, low, high = quantile
            quantile_cluster = "leakage_graph_component_id"
            if spec_name == "within_random":
                quantile_draws = _cluster_bootstrap_median_regression(
                    design,
                    predictors,
                    cluster_col="leakage_graph_component_id",
                    n_boot=median_n_boot,
                    seed=bootstrap_seed,
                )
                low = float(np.quantile(quantile_draws, 0.025))
                high = float(np.quantile(quantile_draws, 0.975))
            rows.append(
                {
                    "feature_schema": schema,
                    "specification": spec_name,
                    "included_splits": ";".join(sorted(working["split_type"].unique())),
                    "adjustment": adjustment,
                    "estimator": "median_regression",
                    "bootstrap_cluster": quantile_cluster,
                    "outcome": "median_abs_error_minus_catboost_abs_error",
                    "predictor": "log1p_same_component_n",
                    "coefficient": coefficient,
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "bootstrap_seed": (
                        bootstrap_seed if spec_name == "within_random" else np.nan
                    ),
                    "n_boot": (
                        median_n_boot if spec_name == "within_random" else 0
                    ),
                    "n_predictions": len(design),
                    "n_unique_rows": design["internal_row_id"].nunique(),
                    "n_components": design["leakage_graph_component_id"].nunique(),
                    "analysis_role": "heavy-tail robustness sensitivity analysis",
                    "causal_caution": "same as OLS specification",
                }
            )
    return rows


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]


def _cluster_bootstrap_coefficients(
    data: pd.DataFrame,
    predictors: list[str],
    cluster_col: str,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = data[predictors].to_numpy(float)
    y = data["model_advantage"].to_numpy(float)
    groups = [indices for indices in data.groupby(cluster_col, sort=False).indices.values()]
    draws = np.empty((n_boot, len(predictors) + 1), dtype=float)
    for draw in range(n_boot):
        sampled_groups = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[item] for item in sampled_groups])
        draws[draw] = _ols(X[idx], y[idx])
    return draws


def _median_regression(
    design: pd.DataFrame, predictors: list[str]
) -> tuple[float, float, float] | None:
    from sklearn.linear_model import QuantileRegressor

    model = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs")
    model.fit(
        design[predictors].to_numpy(float),
        design["model_advantage"].to_numpy(float),
    )
    coefficient = float(model.coef_[predictors.index("log1p_same_component_n")])
    return coefficient, np.nan, np.nan


def _cluster_bootstrap_median_regression(
    data: pd.DataFrame,
    predictors: list[str],
    cluster_col: str,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    from sklearn.linear_model import QuantileRegressor

    rng = np.random.default_rng(seed)
    X = data[predictors].to_numpy(float)
    y = data["model_advantage"].to_numpy(float)
    groups = [indices for indices in data.groupby(cluster_col, sort=False).indices.values()]
    coefficient_index = predictors.index("log1p_same_component_n")
    draws = np.empty(n_boot, dtype=float)
    for draw in range(n_boot):
        sampled_groups = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[item] for item in sampled_groups])
        model = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs")
        model.fit(X[idx], y[idx])
        draws[draw] = model.coef_[coefficient_index]
    return draws


def split_dispersion() -> pd.DataFrame:
    metrics = pd.read_csv("outputs/metrics/benchmark_metrics.csv")
    counts = metrics.groupby(["split_type", "model"])["repeat"].nunique()
    cat_counts = counts.xs("catboost", level="model")
    if len(cat_counts) != len(SPLITS) or not cat_counts.eq(20).all():
        raise AssertionError(
            "benchmark_metrics.csv is not the 20-repeat Table S2 source for all five splits"
        )
    rows = []
    for split_type, values in metrics[metrics["model"].eq("catboost")].groupby("split_type")["r2"]:
        values = values.astype(float)
        rows.append(
            {
                "split_type": split_type,
                "n_repeats": len(values),
                "r2_mean": values.mean(),
                "r2_sd": values.std(ddof=1),
                "r2_empirical_ci95_low": values.quantile(0.025),
                "r2_empirical_ci95_high": values.quantile(0.975),
                "source_assertion": "same 20-repeat benchmark_metrics.csv used for Table S2",
            }
        )
    return pd.DataFrame(rows)


def largest_component_profile(df: pd.DataFrame) -> pd.DataFrame:
    largest = df["leakage_graph_component_id"].value_counts().index[0]
    subset = df[df["leakage_graph_component_id"].eq(largest)]
    rows = [{
        "profile_type": "summary", "rank": 0, "label": "largest_component",
        "n_rows": len(subset), "fraction_component": 1.0,
        "n_unique_sources": subset["source_group_id"].nunique(),
        "n_unique_formulas": subset["standardized_formula"].nunique(),
    }]
    for profile_type, column in [
        ("formula", "standardized_formula"), ("source", "source_group_id"),
        ("varied_substrate", "varied_substrate"),
        ("canonical_reaction_system", "canonical_reaction_system"),
    ]:
        for rank, (label, n_rows) in enumerate(subset[column].value_counts().head(10).items(), 1):
            rows.append({
                "profile_type": profile_type, "rank": rank, "label": label,
                "n_rows": n_rows, "fraction_component": n_rows / len(subset),
                "n_unique_sources": pd.NA, "n_unique_formulas": pd.NA,
            })
    return pd.DataFrame(rows)


def largest_component_articulation_points(df: pd.DataFrame) -> pd.DataFrame:
    graph = nx.Graph()
    for _, row in df.iterrows():
        graph.add_edge(
            f"source:{row['source_group_id']}",
            f"formula:{row['standardized_formula']}",
        )
    largest_nodes = max(nx.connected_components(graph), key=len)
    largest_graph = graph.subgraph(largest_nodes).copy()
    rows = []
    for node in nx.articulation_points(largest_graph):
        reduced = largest_graph.copy()
        reduced.remove_node(node)
        component_sizes = sorted((len(c) for c in nx.connected_components(reduced)), reverse=True)
        rows.append({
            "node_type": node.split(":", 1)[0],
            "node_label": node.split(":", 1)[1],
            "degree": largest_graph.degree(node),
            "n_components_after_removal": len(component_sizes),
            "largest_remaining_node_component": component_sizes[0],
            "second_largest_remaining_node_component": component_sizes[1] if len(component_sizes) > 1 else 0,
        })
    return pd.DataFrame(rows).sort_values(
        ["n_components_after_removal", "degree"], ascending=False
    )


def learning_curves(df: pd.DataFrame, repeats: int, seed: int) -> pd.DataFrame:
    X, meta = build_feature_frame(df)
    y = df["target_log10_km"].astype(float)
    spec = FeatureSpec(meta["numeric_columns"], meta["categorical_columns"])
    rows = []
    for rep in range(repeats):
        rep_seed = seed + rep
        for split_type in SPLITS:
            train_idx, test_idx, _ = make_split(
                df, split_type, rep_seed, test_size=0.2,
                composition=X[meta["composition_columns"]], min_test_n=20,
            )
            permutation = np.random.default_rng(rep_seed).permutation(train_idx)
            for fraction in LEARNING_FRACTIONS:
                n_train = max(20, int(round(len(train_idx) * fraction)))
                sub_train = permutation[:n_train]
                support = _component_support(df, sub_train, test_idx)
                models = model_suite(spec, rep_seed)
                for model_name in ["median", "extratrees", "catboost"]:
                    pred = fit_predict(
                        models[model_name], X.loc[sub_train], y.loc[sub_train], X.loc[test_idx]
                    )
                    rows.append({
                        "repeat": rep, "seed": rep_seed, "split_type": split_type,
                        "train_fraction": fraction, "n_train": len(sub_train),
                        "n_test": len(test_idx), "model": model_name, **support,
                        **regression_metrics(y.loc[test_idx], pred),
                    })
        print(f"learning curves: completed repeat {rep + 1}/{repeats}", flush=True)
    return _summarize_sensitivity(
        pd.DataFrame(rows), ["split_type", "train_fraction", "model"], repeats
    )


def _component_support(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray
) -> dict[str, float]:
    counts = df.loc[train_idx, "leakage_graph_component_id"].value_counts()
    test_support = (
        df.loc[test_idx, "leakage_graph_component_id"].map(counts).fillna(0).astype(int)
    )
    return {
        "mean_train_same_component_n": float(test_support.mean()),
        "frac_test_rows_with_component_support": float(test_support.gt(0).mean()),
    }


def near_duplicate_sensitivity(
    df: pd.DataFrame, repeats: int, seed: int
) -> pd.DataFrame:
    variants = [("full", df)]
    for scope in ["within_source", "cross_source"]:
        keys = BASE_DUPLICATE_KEYS if scope == "within_source" else [
            key for key in BASE_DUPLICATE_KEYS if key != "source_group_id"
        ]
        for mode in ["keep_one", "drop_all"]:
            membership, n_clusters = _near_duplicate_membership(df, keys, mode)
            cohort = df.loc[~membership].copy().reset_index(drop=True)
            cohort["leakage_graph_component_id"] = source_formula_protocol_component_ids(cohort)
            cohort.attrs["n_duplicate_clusters"] = n_clusters
            variants.append((f"{scope}_{mode}", cohort))

    frames = []
    for label, cohort in variants:
        # Duplicate-removal sensitivity must use the prospective primary schema.
        # Post-characterization Km/Vmax fields are deliberately excluded here.
        X, meta = build_design_only_feature_frame(cohort)
        y = cohort["target_log10_km"].astype(float)
        spec = FeatureSpec(meta["numeric_columns"], meta["categorical_columns"])
        rows = []
        for rep in range(repeats):
            rep_seed = seed + rep
            for split_type in SPLITS:
                train_idx, test_idx, _ = make_split(
                    cohort, split_type, rep_seed, test_size=0.2,
                    composition=X[meta["composition_columns"]], min_test_n=20,
                )
                models = model_suite(spec, rep_seed)
                for model_name in ["median", "extratrees", "catboost"]:
                    pred = fit_predict(
                        models[model_name], X.loc[train_idx], y.loc[train_idx], X.loc[test_idx]
                    )
                    rows.append({
                        "cohort": label, "n_cohort": len(cohort),
                        "n_rows_excluded": len(df) - len(cohort),
                        "n_duplicate_clusters": cohort.attrs.get("n_duplicate_clusters", 0),
                        "repeat": rep, "n_train": len(train_idx), "n_test": len(test_idx),
                        "split_type": split_type, "model": model_name,
                        **regression_metrics(y.loc[test_idx], pred),
                    })
            print(
                f"duplicate sensitivity ({label}): completed repeat {rep + 1}/{repeats}",
                flush=True,
            )
        frames.append(pd.DataFrame(rows))
    return _summarize_sensitivity(
        pd.concat(frames, ignore_index=True),
        ["cohort", "n_cohort", "n_rows_excluded", "n_duplicate_clusters", "split_type", "model"],
        repeats,
    )


def _near_duplicate_membership(
    df: pd.DataFrame, keys: list[str], mode: str
) -> tuple[pd.Series, int]:
    available = [column for column in keys if column in df.columns]
    normalized = df[available].copy()
    for column in available:
        if pd.api.types.is_numeric_dtype(normalized[column]):
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").round(3)
        else:
            normalized[column] = (
                normalized[column].astype("string").fillna("__MISSING__").str.lower().str.strip()
            )
    sizes = normalized.groupby(available, dropna=False)[available[0]].transform("size")
    n_clusters = int(
        normalized.loc[sizes.gt(1)].drop_duplicates(subset=available).shape[0]
    )
    if mode == "keep_one":
        membership = normalized.duplicated(subset=available, keep="first")
    elif mode == "drop_all":
        membership = sizes.gt(1)
    else:
        raise ValueError(f"Unknown duplicate-removal mode: {mode}")
    if "source_group_id" in available:
        audit_clusters = len(near_duplicate_clusters(df))
        if audit_clusters != n_clusters:
            raise AssertionError(
                f"within-source detector found {n_clusters} clusters, audit reports {audit_clusters}"
            )
    return membership, n_clusters


def _summarize_sensitivity(
    data: pd.DataFrame, group_cols: list[str], requested_repeats: int
) -> pd.DataFrame:
    rows = []
    interval_label = "empirical_ci95" if requested_repeats >= 20 else "observed_range"
    for keys, group in data.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row["n_repeats"] = group["repeat"].nunique()
        row["mean_n_train"] = group["n_train"].mean()
        row["mean_n_test"] = group["n_test"].mean()
        for optional in [
            "mean_train_same_component_n", "frac_test_rows_with_component_support"
        ]:
            if optional in group:
                row[optional] = group[optional].mean()
        for metric in ["mae", "r2", "spearman"]:
            values = group[metric].dropna().astype(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sd"] = values.std(ddof=1)
            if requested_repeats >= 20:
                row[f"{metric}_empirical_ci95_low"] = values.quantile(0.025)
                row[f"{metric}_empirical_ci95_high"] = values.quantile(0.975)
            else:
                row[f"{metric}_observed_range_low"] = values.min()
                row[f"{metric}_observed_range_high"] = values.max()
        row["uncertainty_label"] = interval_label
        row["analysis_role"] = "descriptive sensitivity analysis; not a confirmatory test"
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
