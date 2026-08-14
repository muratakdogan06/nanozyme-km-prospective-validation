#!/usr/bin/env python
"""Analyses requested in the editor's mandatory major-revision report.

The script writes additive outputs under ``outputs/tables/editor_major_revision``
and ``outputs/figures/editor_major_revision``.  It does not overwrite the frozen
benchmark, split manifests, or the reviewer-control outputs on which it relies.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable

_CACHE_ROOT = Path(tempfile.gettempdir()) / "nanozyme_editor_revision_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import prepare_modeling_data, read_table
from src.conformal import (
    absolute_residual_intervals,
    interval_metrics,
    residual_normalized_intervals,
)
from src.diagnostics import source_formula_protocol_component_ids
from src.evaluate import regression_metrics
from src.features import build_design_only_feature_frame, build_feature_frame
from src.models import (
    FeatureSpec,
    fit_predict,
    model_suite,
    require_catboost,
    sklearn_preprocessor,
)
from src.splits import make_split
from scripts.run_restricted_feature_validation import (
    _prepare_external_shared_substrate,
    _prepare_internal_shared_substrate,
    _restricted_features,
)


SEED = 2026
SPLIT_TYPES = [
    "random",
    "source_group",
    "formula_group",
    "composition_cluster",
    "leakage_graph",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/nanozymes.csv")
    parser.add_argument(
        "--external-data",
        default=(
            "outputs/tables/external_validation/"
            "external_validation_strict_model_ready.csv"
        ),
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument(
        "--sections",
        default="m1,m2m3,m4,m5,m6,conformal",
        help=(
            "Comma-separated blocks: m1,m2m3,m4,m5,m6,conformal. "
            "M6 consumes the row predictions written by m2m3."
        ),
    )
    args = parser.parse_args()

    sections = {item.strip().lower() for item in args.sections.split(",") if item.strip()}
    unknown = sections - {"m1", "m2m3", "m4", "m5", "m6", "conformal"}
    if unknown:
        parser.error(f"Unknown section(s): {', '.join(sorted(unknown))}")

    table_dir = Path("outputs/tables/editor_major_revision")
    figure_dir = Path("outputs/figures/editor_major_revision")
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    internal_full, _ = prepare_modeling_data(read_table(args.train_data))
    internal = _prepare_internal_shared_substrate(internal_full)
    internal["leakage_graph_component_id"] = source_formula_protocol_component_ids(
        internal
    )
    external = _prepare_external_shared_substrate(pd.read_csv(args.external_data))

    if sections & {"m2m3", "conformal"}:
        require_catboost()

    if "m1" in sections:
        print("M1: paired component-cluster bootstrap of the integrity effect", flush=True)
        m1 = component_bootstrap_integrity_effect(
            internal,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        m1.to_csv(
            table_dir / "m1_component_bootstrap_integrity_effect.csv", index=False
        )
        print("M1: complete", flush=True)

    design_prediction_path = table_dir / "m2_m3_design_only_row_predictions.csv"
    if "m2m3" in sections:
        print("M2-M3: rerunning the five-protocol design-only benchmark", flush=True)
        design_raw, design_summary, design_predictions = design_only_normalized_benchmark(
            internal_full,
            repeats=args.repeats,
            seed=args.seed,
        )
        design_raw.to_csv(
            table_dir / "m2_m3_design_only_per_repeat_metrics.csv", index=False
        )
        design_summary.to_csv(
            table_dir / "m2_m3_design_only_main_summary.csv", index=False
        )
        design_predictions.to_csv(design_prediction_path, index=False)
        print("M2-M3: complete", flush=True)

    if "m4" in sections:
        print("M4: building substrate-stratified training-only baselines", flush=True)
        m4_rows, m4_internal, m4_external = substrate_stratified_baseline_analysis(
            internal,
            external,
            repeats=args.repeats,
            seed=args.seed,
            n_boot=args.n_boot,
        )
        m4_rows.to_csv(table_dir / "m4_substrate_baseline_row_predictions.csv", index=False)
        m4_internal.to_csv(
            table_dir / "m4_substrate_baseline_internal_summary.csv", index=False
        )
        m4_external.to_csv(
            table_dir / "m4_substrate_baseline_external_summary.csv", index=False
        )
        plot_substrate_baseline_skill(
            m4_internal,
            m4_external,
            figure_dir / "fig_m4_substrate_stratified_baseline_skill.png",
        )
        print("M4: complete", flush=True)

    if "m5" in sections:
        print("M5: computing publication-clustered external residual decomposition", flush=True)
        m5_summary, formula_summary = external_bias_variance_analysis(
            internal,
            external,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        m5_summary.to_csv(
            table_dir / "m5_external_bias_variance_decomposition.csv", index=False
        )
        formula_summary.to_csv(
            table_dir / "m5_same_formula_distribution_summary.csv", index=False
        )
        offset_sensitivity = external_substrate_offset_sensitivity(
            internal,
            external,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        offset_sensitivity.to_csv(
            table_dir / "m5_external_substrate_offset_sensitivity.csv", index=False
        )
        plot_same_formula_distributions(
            internal,
            external,
            figure_dir / "fig_m5_same_formula_internal_external_distribution.png",
        )
        print("M5: complete", flush=True)

    if "m6" in sections:
        if not design_prediction_path.exists():
            raise FileNotFoundError(
                f"{design_prediction_path} is required; run --sections m2m3,m6"
            )
        print("M6: estimating connectivity associations within distance deciles", flush=True)
        m6_deciles, m6_interaction = distance_decile_connectivity_analysis(
            pd.read_csv(design_prediction_path),
            n_boot=args.n_boot,
            seed=args.seed,
        )
        m6_deciles.to_csv(
            table_dir / "m6_distance_decile_connectivity_association.csv", index=False
        )
        m6_interaction.to_csv(
            table_dir / "m6_distance_decile_interaction_sensitivity.csv", index=False
        )
        plot_distance_decile_association(
            m6_deciles,
            figure_dir / "fig_m6_distance_decile_connectivity.png",
        )
        print("M6: complete", flush=True)

    if "conformal" in sections:
        print("Conformal: comparing row-level and component-aware calibration", flush=True)
        conformal_raw, conformal_summary, conformal_partitions = (
            component_aware_conformal_control(
                internal_full,
                repeats=args.repeats,
                seed=args.seed,
                alpha=0.1,
            )
        )
        conformal_raw.to_csv(
            table_dir / "component_aware_conformal_per_repeat.csv", index=False
        )
        conformal_summary.to_csv(
            table_dir / "component_aware_conformal_summary.csv", index=False
        )
        conformal_partitions.to_csv(
            table_dir / "component_aware_conformal_partition_diagnostics.csv",
            index=False,
        )
        plot_component_aware_conformal(
            conformal_summary,
            figure_dir / "fig_component_aware_conformal_control.png",
        )
        print("Conformal: complete", flush=True)

    print(f"Editor revision outputs written to {table_dir}", flush=True)


def component_bootstrap_integrity_effect(
    internal: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Paired cluster bootstrap of eligible-pool minus component-holdout skill.

    The fitted predictions and outer repeats are held fixed.  Source-formula
    components outside the oversized component are the sampling units, and one
    shared component multiplicity vector is used for both split protocols in
    every draw.  This directly answers the editor's dependency-unit objection
    without treating the 20 repeats or individual rows as independent samples.
    """
    predictions = pd.read_csv(
        "outputs/tables/restricted_feature_validation/"
        "restricted_feature_internal_predictions.csv"
    )
    selected = predictions.loc[
        predictions["split_type"].isin(["eligible_pool_random", "leakage_graph"])
        & predictions["model"].isin(["median", "catboost"])
    ].copy()
    keys = [
        "split_type",
        "repeat",
        "internal_row_id",
        "leakage_graph_component_id",
    ]
    paired = selected.pivot_table(
        index=keys,
        columns="model",
        values="abs_error_log10_km",
        aggfunc="first",
    ).reset_index()
    paired = paired.dropna(subset=["median", "catboost"])

    component_sizes = internal["leakage_graph_component_id"].astype(str).value_counts()
    largest_component = str(component_sizes.index[0])
    eligible_components = sorted(
        component for component in component_sizes.index if str(component) != largest_component
    )
    eligible_components = [str(component) for component in eligible_components]
    paired["leakage_graph_component_id"] = paired[
        "leakage_graph_component_id"
    ].astype(str)
    paired = paired.loc[
        paired["leakage_graph_component_id"].isin(eligible_components)
    ].copy()

    point = _mean_repeat_skill_by_split(paired)
    point_effect = float(
        point["eligible_pool_random"] - point["leakage_graph"]
    )
    matrices = {
        split_type: _component_error_matrices(
            paired.loc[paired["split_type"].eq(split_type)], eligible_components
        )
        for split_type in ["eligible_pool_random", "leakage_graph"]
    }
    rng = np.random.default_rng(seed)
    component_weights = rng.multinomial(
        len(eligible_components),
        np.full(len(eligible_components), 1.0 / len(eligible_components)),
        size=n_boot,
    ).astype(float)
    draws = {}
    for split_type, (cat_sum, median_sum, counts) in matrices.items():
        cat_mae = _weighted_repeat_mae(component_weights, cat_sum, counts)
        median_mae = _weighted_repeat_mae(component_weights, median_sum, counts)
        skill = 1.0 - cat_mae / median_mae
        draws[split_type] = np.nanmean(skill, axis=1)
    effect_draws = draws["eligible_pool_random"] - draws["leakage_graph"]
    finite = effect_draws[np.isfinite(effect_draws)]
    total_drop = float(
        pd.read_csv(
            "outputs/tables/publication/"
            "table24_restricted_matched_pool_skill_decomposition.csv"
        ).loc[0, "total_drop_random_to_leakage_graph"]
    )
    row = {
        "analysis": "eligible_pool_random_skill_minus_component_holdout_skill",
        "feature_schema": "restricted_shared_schema_TMB_H2O2",
        "point_estimate": point_effect,
        "component_bootstrap_ci95_low": float(np.quantile(finite, 0.025)),
        "component_bootstrap_ci95_high": float(np.quantile(finite, 0.975)),
        "ci_excludes_zero": bool(np.quantile(finite, 0.025) > 0 or np.quantile(finite, 0.975) < 0),
        "n_boot_requested": int(n_boot),
        "n_boot_successful": int(len(finite)),
        "n_boot_failed": int(n_boot - len(finite)),
        "bootstrap_seed": int(seed),
        "bootstrap_unit": "non-giant source-formula leakage_graph_component_id",
        "n_eligible_components": int(len(eligible_components)),
        "oversized_component_id_excluded_from_sampling": largest_component,
        "oversized_component_n": int(component_sizes.iloc[0]),
        "eligible_pool_random_skill": float(point["eligible_pool_random"]),
        "component_holdout_skill": float(point["leakage_graph"]),
        "random_minus_component_total_skill_drop": total_drop,
        "integrity_fraction_of_total_drop_point_estimate": float(point_effect / total_drop),
        "pool_composition_fraction_of_total_drop_point_estimate": float(
            (total_drop - point_effect) / total_drop
        ),
        "inference_scope": (
            "prediction-conditional paired component-cluster bootstrap; fitted models and "
            "the 20 outer split repeats are held fixed"
        ),
        "reporting_rule": (
            "If the interval includes zero, report the 78/22 decomposition as a descriptive "
            "point estimate rather than an inferentially resolved partition."
        ),
    }
    return pd.DataFrame([row])


def _mean_repeat_skill_by_split(paired: pd.DataFrame) -> dict[str, float]:
    repeat = (
        paired.groupby(["split_type", "repeat"], as_index=False)
        .agg(catboost_mae=("catboost", "mean"), median_mae=("median", "mean"))
    )
    repeat["skill"] = 1.0 - repeat["catboost_mae"] / repeat["median_mae"]
    return repeat.groupby("split_type")["skill"].mean().to_dict()


def _component_error_matrices(
    data: pd.DataFrame,
    component_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    repeats = sorted(data["repeat"].astype(int).unique())
    shape = (len(repeats), len(component_ids))
    cat_sum = np.zeros(shape, dtype=float)
    median_sum = np.zeros(shape, dtype=float)
    counts = np.zeros(shape, dtype=float)
    repeat_lookup = {value: index for index, value in enumerate(repeats)}
    component_lookup = {value: index for index, value in enumerate(component_ids)}
    aggregated = (
        data.groupby(["repeat", "leakage_graph_component_id"], as_index=False)
        .agg(cat_sum=("catboost", "sum"), median_sum=("median", "sum"), n=("catboost", "size"))
    )
    for row in aggregated.itertuples(index=False):
        i = repeat_lookup[int(row.repeat)]
        j = component_lookup[str(row.leakage_graph_component_id)]
        cat_sum[i, j] = float(row.cat_sum)
        median_sum[i, j] = float(row.median_sum)
        counts[i, j] = float(row.n)
    return cat_sum, median_sum, counts


def _weighted_repeat_mae(
    component_weights: np.ndarray,
    error_sums: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    numerator = component_weights @ error_sums.T
    denominator = component_weights @ counts.T
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )


def design_only_normalized_benchmark(
    df: pd.DataFrame,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort = df.copy()
    cohort["leakage_graph_component_id"] = source_formula_protocol_component_ids(
        cohort.drop(columns=["vmax_value", "vmax_unit"], errors="ignore")
    )
    X, meta = build_design_only_feature_frame(cohort)
    y = cohort["target_log10_km"].astype(float)
    spec = FeatureSpec(meta["numeric_columns"], meta["categorical_columns"])
    metric_rows = []
    prediction_rows = []
    for repeat in range(repeats):
        rep_seed = seed + repeat
        models = model_suite(spec, rep_seed)
        for split_type in SPLIT_TYPES:
            train_idx, test_idx, _ = make_split(
                cohort,
                split_type,
                rep_seed,
                test_size=0.2,
                composition=X[meta["composition_columns"]],
                min_test_n=20,
            )
            median_pred = fit_predict(
                models["median"], X.loc[train_idx], y.loc[train_idx], X.loc[test_idx]
            )
            catboost_pred = fit_predict(
                models["catboost"], X.loc[train_idx], y.loc[train_idx], X.loc[test_idx]
            )
            median_metrics = regression_metrics(y.loc[test_idx], median_pred)
            catboost_metrics = regression_metrics(y.loc[test_idx], catboost_pred)
            target_sd = float(y.loc[test_idx].std(ddof=1)) if len(test_idx) > 1 else np.nan
            train_components = set(
                cohort.loc[train_idx, "leakage_graph_component_id"].astype(str)
            )
            test_component_values = cohort.loc[
                test_idx, "leakage_graph_component_id"
            ].astype(str)
            component_overlap = len(train_components & set(test_component_values))
            test_component_supported = test_component_values.isin(train_components)
            row = {
                "feature_schema": "design_only_primary",
                "repeat": repeat,
                "seed": rep_seed,
                "split_type": split_type,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "component_overlap_count": int(component_overlap),
                "fraction_test_rows_with_component_support": float(
                    test_component_supported.mean()
                ),
                "component_integrity": bool(component_overlap == 0),
                "test_target_sd": target_sd,
                "catboost_mae": float(catboost_metrics["mae"]),
                "median_mae": float(median_metrics["mae"]),
                "catboost_skill_vs_median": _skill(
                    float(catboost_metrics["mae"]), float(median_metrics["mae"])
                ),
                "catboost_mae_over_test_target_sd": (
                    float(catboost_metrics["mae"] / target_sd)
                    if np.isfinite(target_sd) and target_sd > 1e-12
                    else np.nan
                ),
                "median_mae_over_test_target_sd": (
                    float(median_metrics["mae"] / target_sd)
                    if np.isfinite(target_sd) and target_sd > 1e-12
                    else np.nan
                ),
                "catboost_rmse": float(catboost_metrics["rmse"]),
                "catboost_r2": float(catboost_metrics["r2"]),
                "catboost_spearman": float(catboost_metrics["spearman"]),
            }
            metric_rows.append(row)

            if split_type == "random":
                train_counts = cohort.loc[
                    train_idx, "leakage_graph_component_id"
                ].value_counts()
                support = (
                    cohort.loc[test_idx, "leakage_graph_component_id"]
                    .map(train_counts)
                    .fillna(0)
                    .astype(int)
                )
                distance = _design_feature_nearest_distance(
                    X, spec, train_idx, test_idx
                )
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "internal_row_id": cohort.loc[test_idx, "row_id"].astype(int),
                            "repeat": repeat,
                            "seed": rep_seed,
                            "split_type": split_type,
                            "leakage_graph_component_id": cohort.loc[
                                test_idx, "leakage_graph_component_id"
                            ].astype(str),
                            "train_same_component_n": support.to_numpy(),
                            "nearest_train_feature_distance": distance,
                            "target_log10_km": y.loc[test_idx].to_numpy(float),
                            "median_prediction": median_pred,
                            "catboost_prediction": catboost_pred,
                            "median_abs_error": np.abs(
                                y.loc[test_idx].to_numpy(float) - median_pred
                            ),
                            "catboost_abs_error": np.abs(
                                y.loc[test_idx].to_numpy(float) - catboost_pred
                            ),
                        }
                    )
                )
        print(f"M2-M3: completed repeat {repeat + 1}/{repeats}", flush=True)

    raw = pd.DataFrame(metric_rows)
    summary = _summarize_design_only_metrics(raw)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    return raw, summary, predictions


def _summarize_design_only_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "test_target_sd",
        "catboost_mae",
        "median_mae",
        "catboost_skill_vs_median",
        "catboost_mae_over_test_target_sd",
        "median_mae_over_test_target_sd",
        "catboost_r2",
        "catboost_spearman",
    ]
    for split_type, group in raw.groupby("split_type", sort=False):
        row = {
            "feature_schema": "design_only_primary",
            "split_type": split_type,
            "n_repeats": int(group["repeat"].nunique()),
            "mean_n_train": float(group["n_train"].mean()),
            "mean_n_test": float(group["n_test"].mean()),
            "mean_component_overlap_count": float(group["component_overlap_count"].mean()),
            "mean_fraction_test_rows_with_component_support": float(
                group["fraction_test_rows_with_component_support"].mean()
            ),
            "fraction_repeats_with_component_integrity": float(group["component_integrity"].mean()),
            "uncertainty_unit": "outer split repeat",
            "interval_type": "empirical 2.5th-97.5th percentile across 20 repeats",
            "ordering_caution": (
                "The ordering was not strictly monotone in the primary design-only schema."
            ),
        }
        for metric in metrics:
            values = group[metric].dropna().astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))
            row[f"{metric}_ci95_low"] = float(values.quantile(0.025))
            row[f"{metric}_ci95_high"] = float(values.quantile(0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def _design_feature_nearest_distance(
    X: pd.DataFrame,
    feature_spec: FeatureSpec,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    chunk_size: int = 256,
) -> np.ndarray:
    """Euclidean nearest-neighbour distance after train-only design preprocessing."""
    preprocessor = sklearn_preprocessor(feature_spec, scale_numeric=True)
    train_values = np.asarray(preprocessor.fit_transform(X.loc[train_idx]), dtype=float)
    test_values = np.asarray(preprocessor.transform(X.loc[test_idx]), dtype=float)
    nearest = np.empty(len(test_idx), dtype=float)
    for start in range(0, len(test_idx), chunk_size):
        stop = min(start + chunk_size, len(test_idx))
        diff = test_values[start:stop, None, :] - train_values[None, :, :]
        nearest[start:stop] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
    return nearest


def training_only_substrate_median_predictions(
    y_train: pd.Series,
    substrate_train: pd.Series,
    substrate_test: pd.Series,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    """Predict with training-only substrate medians and a global-median fallback."""
    y_train = pd.Series(y_train, copy=False).astype(float)
    substrate_train = pd.Series(substrate_train, copy=False).astype(str)
    substrate_test = pd.Series(substrate_test, copy=False).astype(str)
    global_median = float(y_train.median())
    medians = (
        pd.DataFrame({"target": y_train.to_numpy(), "substrate": substrate_train.to_numpy()})
        .groupby("substrate", dropna=False)["target"]
        .median()
        .to_dict()
    )
    mapped = substrate_test.map(medians)
    fallback = mapped.isna().to_numpy()
    predictions = mapped.fillna(global_median).to_numpy(float)
    return predictions, fallback, global_median, {str(k): float(v) for k, v in medians.items()}


def substrate_stratified_baseline_analysis(
    internal: pd.DataFrame,
    external: pd.DataFrame,
    repeats: int,
    seed: int,
    n_boot: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_path = Path(
        "outputs/tables/restricted_feature_validation/"
        "restricted_feature_internal_predictions.csv"
    )
    external_pred_path = Path(
        "outputs/tables/restricted_feature_validation/"
        "restricted_feature_external_predictions.csv"
    )
    predictions = pd.read_csv(pred_path)
    external_predictions = pd.read_csv(external_pred_path)
    X, composition_columns = _restricted_features(internal)
    y = internal["target_log10_km"].astype(float)

    frames: list[pd.DataFrame] = []
    for repeat in range(repeats):
        rep_seed = seed + repeat
        for split_type in SPLIT_TYPES:
            train_idx, test_idx, _ = make_split(
                internal,
                split_type,
                rep_seed,
                test_size=0.2,
                composition=X[composition_columns],
                min_test_n=20,
            )
            fold = _paired_model_errors(
                predictions.loc[
                    predictions["repeat"].eq(repeat)
                    & predictions["split_type"].eq(split_type)
                ],
                id_column="internal_row_id",
            )
            expected = set(map(int, test_idx))
            observed = set(fold["internal_row_id"].astype(int))
            if observed != expected:
                raise AssertionError(
                    f"Stored predictions do not match reconstructed {split_type} repeat {repeat}"
                )
            substrate_pred, fallback, global_median, medians = (
                training_only_substrate_median_predictions(
                    y.loc[train_idx],
                    internal.loc[train_idx, "substrate_for_km_norm"],
                    internal.loc[test_idx, "substrate_for_km_norm"],
                )
            )
            baseline = pd.DataFrame(
                {
                    "internal_row_id": test_idx.astype(int),
                    "substrate_median_prediction": substrate_pred,
                    "substrate_baseline_fallback": fallback,
                }
            )
            fold = fold.merge(baseline, on="internal_row_id", validate="one_to_one")
            fold["domain"] = "internal_restricted_shared_schema"
            fold["repeat"] = repeat
            fold["seed"] = rep_seed
            fold["split_type"] = split_type
            fold["cluster_id"] = fold["leakage_graph_component_id"].astype(str)
            fold["global_training_median"] = global_median
            fold["training_substrate_medians"] = _serialize_mapping(medians)
            fold["substrate_median_abs_error"] = (
                fold["target_log10_km"] - fold["substrate_median_prediction"]
            ).abs()
            frames.append(fold)

    internal_rows = pd.concat(frames, ignore_index=True)
    internal_summary = _summarize_m4_internal(internal_rows)

    ext_fold = _paired_model_errors(external_predictions, id_column="row_id")
    ext_meta = external[
        [
            "row_id",
            "external_source_cluster_id",
            "substrate_for_km_norm",
            "standardized_formula",
        ]
    ].copy()
    ext_fold = ext_fold.merge(ext_meta, on="row_id", validate="one_to_one", suffixes=("", "_meta"))
    substrate_pred, fallback, global_median, medians = training_only_substrate_median_predictions(
        internal["target_log10_km"],
        internal["substrate_for_km_norm"],
        external["substrate_for_km_norm"],
    )
    ext_fold["substrate_median_prediction"] = substrate_pred
    ext_fold["substrate_baseline_fallback"] = fallback
    ext_fold["substrate_median_abs_error"] = (
        ext_fold["target_log10_km"] - ext_fold["substrate_median_prediction"]
    ).abs()
    ext_fold["domain"] = "strict_external_restricted_shared_schema"
    ext_fold["repeat"] = pd.NA
    ext_fold["seed"] = seed
    ext_fold["split_type"] = "external"
    ext_fold["cluster_id"] = ext_fold["external_source_cluster_id"].astype(str)
    ext_fold["global_training_median"] = global_median
    ext_fold["training_substrate_medians"] = _serialize_mapping(medians)
    external_summary = _summarize_m4_external(ext_fold, n_boot=n_boot, seed=seed)

    common_columns = sorted(set(internal_rows.columns) | set(ext_fold.columns))
    combined = pd.concat(
        [internal_rows.reindex(columns=common_columns), ext_fold.reindex(columns=common_columns)],
        ignore_index=True,
    )
    return combined, internal_summary, external_summary


def _paired_model_errors(predictions: pd.DataFrame, id_column: str) -> pd.DataFrame:
    metadata = [
        col
        for col in predictions.columns
        if col not in {"model", "pred_log10_km", "abs_error_log10_km"}
    ]
    base = predictions[metadata].drop_duplicates(subset=[id_column])
    wide_pred = predictions.pivot_table(
        index=id_column, columns="model", values="pred_log10_km", aggfunc="first"
    ).reset_index()
    wide_error = predictions.pivot_table(
        index=id_column, columns="model", values="abs_error_log10_km", aggfunc="first"
    ).reset_index()
    required = {"median", "catboost"}
    if not required.issubset(wide_pred.columns) or not required.issubset(wide_error.columns):
        raise AssertionError("Both median and CatBoost predictions are required")
    wide_pred = wide_pred.rename(
        columns={"median": "global_median_prediction", "catboost": "catboost_prediction"}
    )
    wide_error = wide_error.rename(
        columns={"median": "global_median_abs_error", "catboost": "catboost_abs_error"}
    )
    return (
        base.merge(wide_pred, on=id_column, validate="one_to_one")
        .merge(wide_error, on=id_column, validate="one_to_one")
    )


def _summarize_m4_internal(rows: pd.DataFrame) -> pd.DataFrame:
    repeat_rows = []
    for (split_type, repeat), group in rows.groupby(["split_type", "repeat"], sort=False):
        metrics = _m4_metrics(group)
        repeat_rows.append(
            {
                "split_type": split_type,
                "repeat": int(repeat),
                "n_test": int(len(group)),
                "n_components": int(group["cluster_id"].nunique()),
                "n_fallback": int(group["substrate_baseline_fallback"].sum()),
                **metrics,
            }
        )
    per_repeat = pd.DataFrame(repeat_rows)
    output = []
    for split_type, group in per_repeat.groupby("split_type", sort=False):
        row = {
            "domain": "internal_restricted_shared_schema",
            "split_type": split_type,
            "n_repeats": int(group["repeat"].nunique()),
            "mean_n_test": float(group["n_test"].mean()),
            "mean_n_components": float(group["n_components"].mean()),
            "total_fallback_predictions": int(group["n_fallback"].sum()),
            "uncertainty_unit": "outer split repeat",
            "interval_type": "empirical 2.5th-97.5th percentile across 20 repeats",
            "fallback_rule": "global training median for a substrate absent from training",
        }
        for metric in _m4_metric_names():
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))
            row[f"{metric}_ci95_low"] = float(values.quantile(0.025))
            row[f"{metric}_ci95_high"] = float(values.quantile(0.975))
        output.append(row)
    return pd.DataFrame(output)


def _summarize_m4_external(rows: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    point = _m4_metrics(rows)
    draws = _cluster_bootstrap_rows(rows, "cluster_id", _m4_metrics, n_boot, seed)
    row = {
        "domain": "strict_external_restricted_shared_schema",
        "split_type": "external",
        "n_test": int(len(rows)),
        "n_publication_clusters": int(rows["cluster_id"].nunique()),
        "n_fallback": int(rows["substrate_baseline_fallback"].sum()),
        "bootstrap_unit": "external_source_cluster_id (publication)",
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        "fallback_rule": "global internal-training median for a substrate absent from training",
    }
    for metric in _m4_metric_names():
        values = draws[metric].astype(float)
        row[metric] = point[metric]
        row[f"{metric}_ci95_low"] = float(values.quantile(0.025))
        row[f"{metric}_ci95_high"] = float(values.quantile(0.975))
    return pd.DataFrame([row])


def _m4_metric_names() -> list[str]:
    return [
        "catboost_mae",
        "global_median_mae",
        "substrate_median_mae",
        "catboost_skill_vs_global_median",
        "catboost_skill_vs_substrate_median",
    ]


def _m4_metrics(group: pd.DataFrame) -> dict[str, float]:
    catboost_mae = float(group["catboost_abs_error"].mean())
    global_mae = float(group["global_median_abs_error"].mean())
    substrate_mae = float(group["substrate_median_abs_error"].mean())
    return {
        "catboost_mae": catboost_mae,
        "global_median_mae": global_mae,
        "substrate_median_mae": substrate_mae,
        "catboost_skill_vs_global_median": _skill(catboost_mae, global_mae),
        "catboost_skill_vs_substrate_median": _skill(catboost_mae, substrate_mae),
    }


def distance_decile_connectivity_analysis(
    predictions: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = predictions.copy()
    data["model_advantage"] = (
        data["median_abs_error"].astype(float) - data["catboost_abs_error"].astype(float)
    )
    data["log1p_same_component_n"] = np.log1p(
        data["train_same_component_n"].astype(float)
    )
    decile_frames = []
    for repeat, group in data.groupby("repeat", sort=True):
        ranked = group["nearest_train_feature_distance"].rank(
            method="first", pct=True
        )
        assigned = group.copy()
        assigned["distance_decile"] = np.minimum(
            np.ceil(ranked.to_numpy(float) * 10).astype(int), 10
        )
        decile_frames.append(assigned)
    data = pd.concat(decile_frames, ignore_index=True)

    rows = []
    for decile, group in data.groupby("distance_decile", sort=True):
        coefficient = _ols_coefficients(
            group[["log1p_same_component_n"]].to_numpy(float),
            group["model_advantage"].to_numpy(float),
        )[1]
        n_components = int(group["leakage_graph_component_id"].nunique())
        if n_components >= 2:
            draws = _cluster_bootstrap_ols(
                group,
                predictors=["log1p_same_component_n"],
                cluster_column="leakage_graph_component_id",
                outcome="model_advantage",
                n_boot=n_boot,
                seed=seed + int(decile),
            )[:, 1]
            low = float(np.quantile(draws, 0.025))
            high = float(np.quantile(draws, 0.975))
            interval_status = "component-cluster bootstrap interval"
        else:
            low = np.nan
            high = np.nan
            interval_status = "descriptive only: fewer than two components"
        rows.append(
            {
                "feature_schema": "design_only_primary",
                "outer_split": "random",
                "distance_decile": int(decile),
                "distance_decile_definition": "repeat-specific train-only composition-distance decile",
                "outcome": "median_abs_error_minus_catboost_abs_error",
                "predictor": "log1p_same_component_n",
                "support_coefficient": float(coefficient),
                "component_bootstrap_ci95_low": low,
                "component_bootstrap_ci95_high": high,
                "n_boot": int(n_boot),
                "bootstrap_seed": int(seed + int(decile)),
                "bootstrap_unit": "leakage_graph_component_id",
                "n_predictions": int(len(group)),
                "n_unique_rows": int(group["internal_row_id"].nunique()),
                "n_components": n_components,
                "n_outer_repeats": int(group["repeat"].nunique()),
                "distance_min": float(group["nearest_train_feature_distance"].min()),
                "distance_median": float(group["nearest_train_feature_distance"].median()),
                "distance_max": float(group["nearest_train_feature_distance"].max()),
                "same_component_support_median": float(group["train_same_component_n"].median()),
                "interval_status": interval_status,
                "dependence_caution": (
                    "Predictions repeat across outer splits; component clustering, not row count, "
                    "defines the resampling unit. Associations are descriptive and noncausal."
                ),
            }
        )

    interaction_data = data.copy()
    interaction_data["distance_decile_centered"] = (
        interaction_data["distance_decile"].astype(float) - 5.5
    )
    interaction_data["support_by_distance_decile"] = (
        interaction_data["log1p_same_component_n"]
        * interaction_data["distance_decile_centered"]
    )
    predictors = [
        "log1p_same_component_n",
        "distance_decile_centered",
        "support_by_distance_decile",
    ]
    beta = _ols_coefficients(
        interaction_data[predictors].to_numpy(float),
        interaction_data["model_advantage"].to_numpy(float),
    )
    draws = _cluster_bootstrap_ols(
        interaction_data,
        predictors=predictors,
        cluster_column="leakage_graph_component_id",
        outcome="model_advantage",
        n_boot=n_boot,
        seed=seed,
    )
    interaction_rows = []
    for index, predictor in enumerate(predictors, start=1):
        interaction_rows.append(
            {
                "feature_schema": "design_only_primary",
                "outer_split": "random",
                "analysis": "pooled_distance_decile_interaction_sensitivity",
                "predictor": predictor,
                "coefficient": float(beta[index]),
                "component_bootstrap_ci95_low": float(np.quantile(draws[:, index], 0.025)),
                "component_bootstrap_ci95_high": float(np.quantile(draws[:, index], 0.975)),
                "n_boot": int(n_boot),
                "bootstrap_seed": int(seed),
                "bootstrap_unit": "leakage_graph_component_id",
                "n_predictions": int(len(interaction_data)),
                "n_unique_rows": int(interaction_data["internal_row_id"].nunique()),
                "n_components": int(
                    interaction_data["leakage_graph_component_id"].nunique()
                ),
                "analysis_role": "summary sensitivity analysis; not a multiple-testing claim",
                "causal_caution": "residual association only; no causal connectivity interpretation",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(interaction_rows)


def _ols_coefficients(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]


def _cluster_bootstrap_ols(
    data: pd.DataFrame,
    predictors: list[str],
    cluster_column: str,
    outcome: str,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    working = data[[cluster_column, outcome, *predictors]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    X = working[predictors].to_numpy(float)
    y = working[outcome].to_numpy(float)
    groups = [
        positions
        for positions in working.groupby(cluster_column, sort=False).indices.values()
    ]
    if len(groups) < 2:
        raise ValueError("At least two clusters are required for cluster bootstrap")
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, len(predictors) + 1), dtype=float)
    for draw in range(n_boot):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[item] for item in sampled])
        draws[draw] = _ols_coefficients(X[indices], y[indices])
    return draws


def component_aware_calibration_split(
    train_idx: np.ndarray,
    component_ids: pd.Series,
    seed: int,
    calibration_fraction: float = 0.25,
    candidates: int = 5000,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str | bool]]:
    """Keep components intact while approximating a target calibration row count."""
    train_idx = np.asarray(train_idx)
    groups = component_ids.loc[train_idx].astype(str)
    sizes = groups.value_counts()
    target_n = int(round(len(train_idx) * calibration_fraction))
    eligible = sizes.loc[sizes.le(target_n)].index.to_numpy()
    if len(eligible) < 2:
        raise ValueError("Too few eligible training components for group-aware calibration")
    rng = np.random.default_rng(seed)
    best_components: list[str] | None = None
    best_gap = np.inf
    best_n = 0
    for _ in range(candidates):
        order = rng.permutation(eligible)
        selected = []
        count = 0
        for component in order:
            proposed = count + int(sizes.loc[component])
            if abs(proposed - target_n) <= abs(count - target_n):
                selected.append(str(component))
                count = proposed
            if count == target_n:
                break
        gap = abs(count - target_n)
        if selected and gap < best_gap:
            best_components = selected
            best_gap = gap
            best_n = count
        if gap == 0:
            break
    if not best_components:
        raise RuntimeError("Could not construct a component-aware calibration set")
    calibration_mask = groups.isin(best_components).to_numpy()
    calibration_idx = np.sort(train_idx[calibration_mask])
    proper_idx = np.sort(train_idx[~calibration_mask])
    proper_components = set(component_ids.loc[proper_idx].astype(str))
    calibration_components = set(component_ids.loc[calibration_idx].astype(str))
    overlap = proper_components & calibration_components
    diagnostics = {
        "target_calibration_n": target_n,
        "actual_calibration_n": int(len(calibration_idx)),
        "actual_calibration_fraction": float(len(calibration_idx) / len(train_idx)),
        "n_proper_training": int(len(proper_idx)),
        "n_calibration_components": int(len(calibration_components)),
        "n_proper_training_components": int(len(proper_components)),
        "proper_calibration_component_overlap": int(len(overlap)),
        "component_integrity_pass": bool(not overlap),
        "largest_training_component_n": int(sizes.max()),
        "largest_calibration_component_n": int(
            sizes.loc[list(calibration_components)].max()
        ),
        "largest_calibration_component_fraction": float(
            sizes.loc[list(calibration_components)].max() / len(calibration_idx)
        ),
        "oversized_components_forced_to_proper_training": int(sizes.gt(target_n).sum()),
        "selection_rule": (
            "randomized whole-component subset search minimizing absolute deviation from "
            "25% calibration rows; components larger than the target remain in proper training"
        ),
    }
    return proper_idx, calibration_idx, diagnostics


def component_aware_conformal_control(
    df: pd.DataFrame,
    repeats: int,
    seed: int,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort = df.copy()
    cohort["leakage_graph_component_id"] = source_formula_protocol_component_ids(cohort)
    X, meta = build_feature_frame(cohort)
    y = cohort["target_log10_km"].astype(float)
    spec = FeatureSpec(meta["numeric_columns"], meta["categorical_columns"])
    metric_rows = []
    partition_rows = []
    methods = {
        "absolute_residual": absolute_residual_intervals,
        "residual_normalized": residual_normalized_intervals,
    }
    for repeat in range(repeats):
        rep_seed = seed + repeat
        base_model = model_suite(spec, rep_seed)["catboost"]
        train_idx, test_idx, _ = make_split(
            cohort,
            "leakage_graph",
            rep_seed,
            test_size=0.2,
            composition=X[meta["composition_columns"]],
            min_test_n=20,
        )
        row_proper, row_calibration = train_test_split(
            train_idx,
            test_size=0.25,
            random_state=rep_seed,
            shuffle=True,
        )
        component_proper, component_calibration, component_diag = (
            component_aware_calibration_split(
                train_idx,
                cohort["leakage_graph_component_id"],
                seed=rep_seed,
                calibration_fraction=0.25,
            )
        )
        partitions = {
            "row_level": (np.sort(row_proper), np.sort(row_calibration)),
            "component_aware": (component_proper, component_calibration),
        }
        for strategy, (proper_idx, calibration_idx) in partitions.items():
            proper_components = set(
                cohort.loc[proper_idx, "leakage_graph_component_id"].astype(str)
            )
            calibration_components = set(
                cohort.loc[calibration_idx, "leakage_graph_component_id"].astype(str)
            )
            test_components = set(
                cohort.loc[test_idx, "leakage_graph_component_id"].astype(str)
            )
            partition_row = {
                "feature_schema": "post_characterization_full_992",
                "repeat": repeat,
                "seed": rep_seed,
                "outer_split": "leakage_graph",
                "calibration_strategy": strategy,
                "n_outer_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_proper_training": int(len(proper_idx)),
                "n_calibration": int(len(calibration_idx)),
                "actual_calibration_fraction": float(len(calibration_idx) / len(train_idx)),
                "n_proper_training_components": int(len(proper_components)),
                "n_calibration_components": int(len(calibration_components)),
                "proper_calibration_component_overlap": int(
                    len(proper_components & calibration_components)
                ),
                "calibration_test_component_overlap": int(
                    len(calibration_components & test_components)
                ),
                "proper_test_component_overlap": int(len(proper_components & test_components)),
                "component_integrity_pass": bool(
                    strategy != "component_aware"
                    or not (proper_components & calibration_components)
                ),
            }
            if strategy == "component_aware":
                partition_row.update(component_diag)
            partition_rows.append(partition_row)
            for method_name, method in methods.items():
                result = method(
                    base_model,
                    X.loc[proper_idx],
                    y.loc[proper_idx],
                    X.loc[calibration_idx],
                    y.loc[calibration_idx],
                    X.loc[test_idx],
                    alpha=alpha,
                )
                metrics = interval_metrics(
                    y.loc[test_idx],
                    result.y_pred,
                    result.lower,
                    result.upper,
                    alpha=alpha,
                )
                metric_rows.append(
                    {
                        "feature_schema": "post_characterization_full_992",
                        "repeat": repeat,
                        "seed": rep_seed,
                        "outer_split": "leakage_graph",
                        "calibration_strategy": strategy,
                        "method": method_name,
                        "n_proper_training": int(len(proper_idx)),
                        "n_calibration": int(len(calibration_idx)),
                        "n_test": int(len(test_idx)),
                        "qhat": float(result.qhat),
                        **metrics,
                    }
                )
        print(f"Conformal: completed repeat {repeat + 1}/{repeats}", flush=True)
    raw = pd.DataFrame(metric_rows)
    return raw, _summarize_component_conformal(raw), pd.DataFrame(partition_rows)


def _summarize_component_conformal(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "mae",
        "empirical_coverage",
        "coverage_gap",
        "mean_interval_width",
        "median_interval_width",
        "interval_score",
    ]
    for keys, group in raw.groupby(["calibration_strategy", "method"], sort=False):
        strategy, method = keys
        row = {
            "feature_schema": "post_characterization_full_992",
            "outer_split": "leakage_graph",
            "calibration_strategy": strategy,
            "method": method,
            "n_repeats": int(group["repeat"].nunique()),
            "mean_n_proper_training": float(group["n_proper_training"].mean()),
            "mean_n_calibration": float(group["n_calibration"].mean()),
            "mean_n_test": float(group["n_test"].mean()),
            "uncertainty_unit": "outer split repeat",
            "interval_type": "empirical 2.5th-97.5th percentile across 20 repeats",
        }
        for metric in metrics:
            values = group[metric].dropna().astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))
            row[f"{metric}_ci95_low"] = float(values.quantile(0.025))
            row[f"{metric}_ci95_high"] = float(values.quantile(0.975))
        rows.append(row)
    summary = pd.DataFrame(rows)
    paired_rows = []
    for method, group in raw.groupby("method", sort=False):
        wide = group.pivot_table(
            index="repeat",
            columns="calibration_strategy",
            values=["empirical_coverage", "mean_interval_width", "interval_score"],
            aggfunc="first",
        )
        row = {
            "feature_schema": "post_characterization_full_992",
            "outer_split": "leakage_graph",
            "calibration_strategy": "component_aware_minus_row_level",
            "method": method,
            "n_repeats": int(len(wide)),
            "mean_n_proper_training": np.nan,
            "mean_n_calibration": np.nan,
            "mean_n_test": np.nan,
            "uncertainty_unit": "paired outer split repeat",
            "interval_type": "empirical 2.5th-97.5th percentile of paired differences",
        }
        for metric in ["empirical_coverage", "mean_interval_width", "interval_score"]:
            difference = (
                wide[(metric, "component_aware")] - wide[(metric, "row_level")]
            ).dropna()
            row[f"{metric}_mean"] = float(difference.mean())
            row[f"{metric}_sd"] = float(difference.std(ddof=1))
            row[f"{metric}_ci95_low"] = float(difference.quantile(0.025))
            row[f"{metric}_ci95_high"] = float(difference.quantile(0.975))
        paired_rows.append(row)
    return pd.concat([summary, pd.DataFrame(paired_rows)], ignore_index=True, sort=False)


def external_bias_variance_analysis(
    internal: pd.DataFrame,
    external: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    external_predictions = pd.read_csv(
        "outputs/tables/restricted_feature_validation/"
        "restricted_feature_external_predictions.csv"
    )
    catboost = external_predictions.loc[
        external_predictions["model"].eq("catboost")
    ].copy()
    meta = external[
        ["row_id", "external_source_cluster_id", "substrate_for_km_norm", "standardized_formula"]
    ].copy()
    catboost = catboost.merge(
        meta,
        on="row_id",
        validate="one_to_one",
        suffixes=("", "_meta"),
    )
    catboost["cluster_id"] = catboost["external_source_cluster_id"].astype(str)
    catboost["residual"] = (
        catboost["pred_log10_km"].astype(float) - catboost["target_log10_km"].astype(float)
    )

    summary_rows = []
    strata: list[tuple[str, pd.DataFrame]] = [("pooled", catboost)]
    strata.extend(
        (str(substrate), group.copy())
        for substrate, group in catboost.groupby("substrate_for_km_norm", sort=True)
    )
    for stratum, group in strata:
        point = mse_bias_variance_metrics(group["residual"].to_numpy(float))
        draws = _cluster_bootstrap_rows(
            group,
            "cluster_id",
            lambda sampled: mse_bias_variance_metrics(sampled["residual"].to_numpy(float)),
            n_boot,
            seed,
        )
        row = {
            "domain": "strict_external_restricted_shared_schema",
            "stratum": stratum,
            "residual_definition": "prediction - observation",
            "n_rows": int(len(group)),
            "n_publication_clusters": int(group["cluster_id"].nunique()),
            "bootstrap_unit": "external_source_cluster_id (publication)",
            "n_boot": int(n_boot),
            "bootstrap_seed": int(seed),
        }
        for metric, value in point.items():
            row[metric] = value
            row[f"{metric}_ci95_low"] = float(draws[metric].quantile(0.025))
            row[f"{metric}_ci95_high"] = float(draws[metric].quantile(0.975))
        summary_rows.append(row)

    shared = set(internal["standardized_formula"].astype(str)) & set(
        external["standardized_formula"].astype(str)
    )
    internal_shared = internal.loc[internal["standardized_formula"].astype(str).isin(shared)]
    external_shared = external.loc[external["standardized_formula"].astype(str).isin(shared)]
    formula_summary = pd.DataFrame(
        [
            _distribution_summary_row(internal_shared, "internal", "source_group_id"),
            _distribution_summary_row(
                external_shared, "external", "external_source_cluster_id"
            ),
        ]
    )
    formula_summary["n_shared_formulas"] = len(shared)
    formula_summary["subset_definition"] = (
        "rows whose standardized_formula occurs in both internal and strict external cohorts"
    )
    return pd.DataFrame(summary_rows), formula_summary


def external_substrate_offset_sensitivity(
    internal: pd.DataFrame,
    external: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Post hoc sensitivity to the observed external H2O2 mean residual.

    The full-sample correction is deliberately labelled oracle because it uses
    external outcomes.  The leave-one-publication-out (LOPO) version estimates
    the H2O2 offset from all other external publications for each held-out
    publication.  Neither analysis is presented as a deployable prospective
    model; the estimand is whether a simple substrate-specific level shift can
    reveal positive skill beyond internal training-only substrate medians.
    """
    external_predictions = pd.read_csv(
        "outputs/tables/restricted_feature_validation/"
        "restricted_feature_external_predictions.csv"
    )
    catboost = external_predictions.loc[
        external_predictions["model"].eq("catboost")
    ].copy()
    meta = external[
        ["row_id", "external_source_cluster_id", "substrate_for_km_norm"]
    ].copy()
    catboost = catboost.merge(
        meta,
        on="row_id",
        validate="one_to_one",
        suffixes=("", "_meta"),
    )
    catboost["cluster_id"] = catboost["external_source_cluster_id"].astype(str)
    catboost["substrate"] = catboost["substrate_for_km_norm"].astype(str)
    catboost["observed"] = catboost["target_log10_km"].astype(float)
    catboost["prediction"] = catboost["pred_log10_km"].astype(float)
    catboost["raw_residual"] = catboost["prediction"] - catboost["observed"]

    training_substrate_medians = (
        internal.groupby("substrate_for_km_norm")["target_log10_km"].median()
    )
    global_training_median = float(internal["target_log10_km"].median())
    catboost["baseline_prediction"] = (
        catboost["substrate"].map(training_substrate_medians).fillna(global_training_median)
    )
    catboost["baseline_abs_error"] = (
        catboost["baseline_prediction"] - catboost["observed"]
    ).abs()

    h2_mask = catboost["substrate"].eq("H2O2")
    h2_full_offset = float(catboost.loc[h2_mask, "raw_residual"].mean())

    catboost["uncorrected_residual"] = catboost["raw_residual"]
    catboost["oracle_h2o2_residual"] = catboost["raw_residual"]
    catboost.loc[h2_mask, "oracle_h2o2_residual"] -= h2_full_offset

    lopo_offsets: dict[str, float] = {}
    for cluster_id in catboost["cluster_id"].unique():
        reference = catboost.loc[h2_mask & catboost["cluster_id"].ne(cluster_id)]
        lopo_offsets[str(cluster_id)] = float(reference["raw_residual"].mean())
    catboost["lopo_h2o2_residual"] = catboost["raw_residual"]
    catboost.loc[h2_mask, "lopo_h2o2_residual"] -= catboost.loc[
        h2_mask, "cluster_id"
    ].map(lopo_offsets)

    strategies = [
        (
            "uncorrected",
            "uncorrected_residual",
            0.0,
            "No external-label correction",
        ),
        (
            "oracle_h2o2_mean_offset",
            "oracle_h2o2_residual",
            h2_full_offset,
            "Full external H2O2 outcomes; post hoc oracle sensitivity",
        ),
        (
            "lopo_h2o2_mean_offset",
            "lopo_h2o2_residual",
            np.nan,
            "Other external publications' H2O2 outcomes; diagnostic LOPO sensitivity",
        ),
    ]

    def metrics(sampled: pd.DataFrame, residual_column: str) -> dict[str, float]:
        model_mae = float(sampled[residual_column].abs().mean())
        baseline_mae = float(sampled["baseline_abs_error"].mean())
        return {
            "catboost_mae": model_mae,
            "substrate_median_mae": baseline_mae,
            "skill_vs_substrate_median": _skill(model_mae, baseline_mae),
        }

    rows = []
    for index, (name, residual_column, correction, label_use) in enumerate(strategies):
        point = metrics(catboost, residual_column)
        draws = _cluster_bootstrap_rows(
            catboost,
            "cluster_id",
            lambda sampled, column=residual_column: metrics(sampled, column),
            n_boot,
            seed + 1000 + index,
        )
        row = {
            "domain": "strict_external_restricted_shared_schema",
            "strategy": name,
            "h2o2_offset_subtracted": correction,
            "external_label_use": label_use,
            "n_rows": int(len(catboost)),
            "n_publication_clusters": int(catboost["cluster_id"].nunique()),
            "bootstrap_unit": "external_source_cluster_id (publication)",
            "bootstrap_interpretation": (
                "Publication-cluster uncertainty conditional on the fitted correction; "
                "not an independent prospective performance interval"
            ),
            "n_boot": int(n_boot),
            "bootstrap_seed": int(seed + 1000 + index),
            **point,
        }
        for metric_name in point:
            row[f"{metric_name}_ci95_low"] = float(draws[metric_name].quantile(0.025))
            row[f"{metric_name}_ci95_high"] = float(draws[metric_name].quantile(0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def mse_bias_variance_metrics(residuals: np.ndarray) -> dict[str, float]:
    """Return the exact population-MSE decomposition for signed residuals."""
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) == 0:
        return {name: np.nan for name in _m5_metric_names()}
    signed_bias = float(np.mean(residuals))
    mse = float(np.mean(residuals**2))
    bias_squared = signed_bias**2
    residual_variance = float(np.var(residuals, ddof=0))
    return {
        "signed_bias": signed_bias,
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(mse)),
        "bias_corrected_mae": float(np.mean(np.abs(residuals - signed_bias))),
        "mse": mse,
        "bias_squared": bias_squared,
        "residual_variance": residual_variance,
        "bias_squared_fraction_of_mse": float(bias_squared / mse) if mse > 0 else np.nan,
        "decomposition_roundoff": float(mse - bias_squared - residual_variance),
    }


def _m5_metric_names() -> list[str]:
    return [
        "signed_bias",
        "mae",
        "rmse",
        "bias_corrected_mae",
        "mse",
        "bias_squared",
        "residual_variance",
        "bias_squared_fraction_of_mse",
        "decomposition_roundoff",
    ]


def plot_substrate_baseline_skill(
    internal_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    output_path: Path,
    save_pdf: bool = True,
) -> None:
    labels = {
        "random": "Random",
        "source_group": "Source",
        "formula_group": "Formula",
        "composition_cluster": "Composition cluster",
        "leakage_graph": "Component",
        "external": "External",
    }
    internal = internal_summary.copy()
    internal["label"] = internal["split_type"].map(labels)
    internal["estimate"] = internal[
        "catboost_skill_vs_substrate_median_mean"
    ]
    internal["low"] = internal[
        "catboost_skill_vs_substrate_median_ci95_low"
    ]
    internal["high"] = internal[
        "catboost_skill_vs_substrate_median_ci95_high"
    ]
    external = external_summary.copy()
    external["label"] = "External"
    external["estimate"] = external["catboost_skill_vs_substrate_median"]
    external["low"] = external[
        "catboost_skill_vs_substrate_median_ci95_low"
    ]
    external["high"] = external[
        "catboost_skill_vs_substrate_median_ci95_high"
    ]
    order = ["Random", "Composition cluster", "Formula", "Source", "Component", "External"]
    plot = pd.concat(
        [internal[["label", "estimate", "low", "high"]], external[["label", "estimate", "low", "high"]]],
        ignore_index=True,
    ).set_index("label").loc[order].reset_index()

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.45)
    fig, axis = plt.subplots(figsize=(9.5, 5.4))
    y = np.arange(len(plot))
    colors = ["#3b6ea8"] * 5 + ["#c65f46"]
    for position, row in plot.iterrows():
        color = colors[position]
        axis.errorbar(
            row["estimate"],
            position,
            xerr=[[row["estimate"] - row["low"]], [row["high"] - row["estimate"]]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=4,
            markersize=8,
        )
    axis.axvline(0, color="black", linestyle="--", linewidth=1.2)
    axis.set_yticks(y, plot["label"])
    axis.invert_yaxis()
    axis.set_xlabel("CatBoost skill vs training-only substrate median")
    axis.set_ylabel("")
    # Retain the original top spacing while omitting the redundant in-figure title.
    axis.set_title(" ")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_options: dict[str, object] = {"dpi": 300, "bbox_inches": "tight"}
    if output_path.suffix.lower() in {".tif", ".tiff"}:
        save_options["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(output_path, **save_options)
    if save_pdf:
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_distance_decile_association(summary: pd.DataFrame, output_path: Path) -> None:
    data = summary.sort_values("distance_decile").copy()
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.45)
    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    x = data["distance_decile"].to_numpy(int)
    estimate = data["support_coefficient"].to_numpy(float)
    low = data["component_bootstrap_ci95_low"].to_numpy(float)
    high = data["component_bootstrap_ci95_high"].to_numpy(float)
    axis.errorbar(
        x,
        estimate,
        yerr=np.vstack([estimate - low, high - estimate]),
        fmt="o-",
        color="#3b6ea8",
        ecolor="#708eab",
        linewidth=2,
        capsize=3,
    )
    axis.axhline(0, color="black", linestyle="--", linewidth=1.2)
    axis.set_xticks(x)
    axis.set_xlabel("Repeat-specific nearest-neighbour distance decile")
    axis.set_ylabel("Support coefficient")
    axis.set_title("Connectivity association within descriptor-distance strata")
    axis.text(
        0.01,
        0.02,
        "Component-bootstrap intervals; descriptive, noncausal estimates",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_component_aware_conformal(
    summary: pd.DataFrame,
    output_path: Path,
    save_pdf: bool = True,
) -> None:
    data = summary.loc[
        summary["calibration_strategy"].isin(["row_level", "component_aware"])
    ].copy()
    method_labels = {
        "absolute_residual": "Absolute residual",
        "residual_normalized": "Residual normalized",
    }
    strategy_labels = {
        "row_level": "Row-level calibration",
        "component_aware": "Component-aware calibration",
    }
    data["method_label"] = data["method"].map(method_labels)
    data["strategy_label"] = data["calibration_strategy"].map(strategy_labels)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.35)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    palette = {"Row-level calibration": "#8b8b8b", "Component-aware calibration": "#3b6ea8"}
    for method_index, method in enumerate(method_labels.values()):
        subset = data.loc[data["method_label"].eq(method)]
        for strategy_index, strategy in enumerate(strategy_labels.values()):
            row = subset.loc[subset["strategy_label"].eq(strategy)].iloc[0]
            x = method_index + (strategy_index - 0.5) * 0.18
            axes[0].errorbar(
                x,
                row["empirical_coverage_mean"],
                yerr=[[
                    row["empirical_coverage_mean"] - row["empirical_coverage_ci95_low"]
                ], [
                    row["empirical_coverage_ci95_high"] - row["empirical_coverage_mean"]
                ]],
                fmt="o",
                color=palette[strategy],
                capsize=4,
                markersize=8,
                label=strategy if method_index == 0 else None,
            )
            axes[1].scatter(
                x,
                row["mean_interval_width_mean"],
                color=palette[strategy],
                s=70,
                label=strategy if method_index == 0 else None,
            )
    axes[0].axhline(0.90, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_title("A.", loc="left", fontweight="bold")
    axes[1].set_ylabel(r"Mean width (log$_{10}$ units)")
    axes[1].set_title("B.", loc="left", fontweight="bold")
    for axis in axes:
        axis.set_xticks([0, 1], list(method_labels.values()), rotation=10)
        axis.set_xlabel("")
    axes[0].legend(frameon=False, fontsize=10, loc="lower right")
    # Preserve the original top margin without displaying an overall title.
    fig.suptitle(" ", y=1.03)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_options: dict[str, object] = {"dpi": 300, "bbox_inches": "tight"}
    if output_path.suffix.lower() in {".tif", ".tiff"}:
        save_options["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(output_path, **save_options)
    if save_pdf:
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_same_formula_distributions(
    internal: pd.DataFrame,
    external: pd.DataFrame,
    output_path: Path,
    save_pdf: bool = True,
) -> None:
    shared = set(internal["standardized_formula"].astype(str)) & set(
        external["standardized_formula"].astype(str)
    )
    internal_plot = internal.loc[
        internal["standardized_formula"].astype(str).isin(shared),
        ["target_log10_km", "substrate_for_km_norm", "standardized_formula", "source_group_id"],
    ].copy()
    internal_plot["corpus"] = "Internal"
    external_plot = external.loc[
        external["standardized_formula"].astype(str).isin(shared),
        [
            "target_log10_km",
            "substrate_for_km_norm",
            "standardized_formula",
            "external_source_cluster_id",
        ],
    ].copy()
    external_plot["corpus"] = "External"
    plot_data = pd.concat(
        [
            internal_plot[["target_log10_km", "substrate_for_km_norm", "corpus"]],
            external_plot[["target_log10_km", "substrate_for_km_norm", "corpus"]],
        ],
        ignore_index=True,
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    palette = {"Internal": "#3b6ea8", "External": "#c65f46"}
    for axis, substrate in zip(axes, ["TMB", "H2O2"]):
        subset = plot_data.loc[plot_data["substrate_for_km_norm"].eq(substrate)]
        sns.histplot(
            data=subset,
            x="target_log10_km",
            hue="corpus",
            stat="density",
            common_norm=False,
            element="step",
            fill=False,
            bins=18,
            linewidth=2,
            palette=palette,
            ax=axis,
        )
        counts = subset.groupby("corpus").size().to_dict()
        axis.set_title(
            f"{substrate}: internal n={counts.get('Internal', 0)}, "
            f"external n={counts.get('External', 0)}"
        )
        axis.set_xlabel(r"Observed log$_{10}$($K_m^{app}$/M)")
    axes[0].set_ylabel("Density")
    axes[1].set_ylabel("")
    # Retain the original top spacing while omitting the redundant overall title.
    fig.suptitle(" ", y=1.03)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_options: dict[str, object] = {"dpi": 300, "bbox_inches": "tight"}
    if output_path.suffix.lower() in {".tif", ".tiff"}:
        save_options["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(output_path, **save_options)
    if save_pdf:
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _distribution_summary_row(
    frame: pd.DataFrame, corpus: str, cluster_column: str
) -> dict[str, float | int | str]:
    target = frame["target_log10_km"].astype(float)
    return {
        "corpus": corpus,
        "n_rows": int(len(frame)),
        "n_clusters": int(frame[cluster_column].nunique()),
        "target_mean": float(target.mean()),
        "target_sd": float(target.std(ddof=1)),
        "target_median": float(target.median()),
        "target_iqr": float(target.quantile(0.75) - target.quantile(0.25)),
    }


def _cluster_bootstrap_rows(
    frame: pd.DataFrame,
    cluster_column: str,
    statistic: Callable[[pd.DataFrame], dict[str, float]],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    groups = [group.index.to_numpy() for _, group in frame.groupby(cluster_column, sort=False)]
    if len(groups) < 2:
        raise ValueError(f"At least two {cluster_column} clusters are required")
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[item] for item in sampled])
        rows.append(statistic(frame.loc[indices]))
    return pd.DataFrame(rows)


def _skill(model_mae: float, baseline_mae: float) -> float:
    return float(1.0 - model_mae / baseline_mae) if baseline_mae > 0 else np.nan


def _serialize_mapping(values: dict[str, float]) -> str:
    return ";".join(f"{key}={value:.12g}" for key, value in sorted(values.items()))


if __name__ == "__main__":
    main()
