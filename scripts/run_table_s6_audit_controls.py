#!/usr/bin/env python3
"""Run the optimization objective-alignment and convergence audits.

The CatBoost sensitivity compares the prespecified 500-iteration RMSE model
with MAE-loss fits at 500, 1000, and 2000 iterations under the restricted
composition-plus-substrate schema. All fits use the same 915 internal records
and 322 external records. External uncertainty uses the same frozen publication
clusters for individual intervals and paired differences from the RMSE model.

The ElasticNet audit refits the prespecified pipeline in every design-only
and shared-schema internal split/repeat and once on the restricted external task
while recording ``ConvergenceWarning``, ``n_iter_``, ``dual_gap_``, and the
fit-specific scaled stopping threshold.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_editor_major_revision_analyses import (
    training_only_substrate_median_predictions,
)
from scripts.run_restricted_feature_validation import (
    _eligible_pool_random_split,
    _prepare_external_shared_substrate,
    _prepare_internal_shared_substrate,
    _restricted_features,
)
from src.data import prepare_modeling_data, read_table
from src.diagnostics import source_formula_protocol_component_ids
from src.evaluate import regression_metrics
from src.features import build_design_only_feature_frame
from src.models import CatBoostFrameRegressor, FeatureSpec, model_suite, require_catboost
from src.splits import make_split


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "tables" / "table_s6_audit"
SEED = 2026
REPEATS = 20
N_BOOT = 10_000
SPLIT_TYPES = [
    "random",
    "composition_cluster",
    "formula_group",
    "source_group",
    "leakage_graph",
]
SHARED_SPLIT_TYPES = [
    "random",
    "eligible_pool_random",
    "source_group",
    "formula_group",
    "leakage_graph",
    "composition_cluster",
]
MAE_ITERATIONS = [500, 1000, 2000]


def publication_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    model_error_column: str,
    baseline_error_column: str,
    cluster_column: str,
    seed: int = SEED,
    n_boot: int = N_BOOT,
    reference_error_column: str | None = None,
) -> dict[str, float]:
    grouped = [
        group.index.to_numpy()
        for _, group in frame.groupby(cluster_column, sort=False)
    ]
    if len(grouped) != 127:
        raise AssertionError(f"Expected 127 publication clusters, found {len(grouped)}")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    paired_draws = (
        np.empty(n_boot, dtype=float)
        if reference_error_column is not None
        else None
    )
    for draw in range(n_boot):
        sampled_groups = rng.integers(0, len(grouped), size=len(grouped))
        sampled_index = np.concatenate([grouped[index] for index in sampled_groups])
        sampled = frame.loc[sampled_index]
        model_mae = float(sampled[model_error_column].mean())
        baseline_mae = float(sampled[baseline_error_column].mean())
        draws[draw] = 1.0 - model_mae / baseline_mae
        if paired_draws is not None:
            reference_mae = float(sampled[reference_error_column].mean())
            reference_skill = 1.0 - reference_mae / baseline_mae
            paired_draws[draw] = draws[draw] - reference_skill
    low, high = np.quantile(draws, [0.025, 0.975])
    result = {
        "skill_ci95_low": float(low),
        "skill_ci95_high": float(high),
        "bootstrap_skill_se": float(draws.std(ddof=1)),
    }
    if paired_draws is not None:
        paired_low, paired_high = np.quantile(paired_draws, [0.025, 0.975])
        result.update(
            {
                "paired_skill_difference_ci95_low": float(paired_low),
                "paired_skill_difference_ci95_high": float(paired_high),
                "paired_skill_difference_bootstrap_se": float(
                    paired_draws.std(ddof=1)
                ),
            }
        )
    return result


def catboost_objective_alignment(
    internal: pd.DataFrame,
    external: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_internal, composition_columns = _restricted_features(internal)
    X_external, _ = _restricted_features(external)
    X_external = X_external.reindex(columns=X_internal.columns)
    y_internal = internal["target_log10_km"].astype(float)
    y_external = external["target_log10_km"].astype(float)
    categorical_columns = ["substrate_for_km_norm"]

    baseline_prediction, fallback, global_median, substrate_medians = (
        training_only_substrate_median_predictions(
            y_internal,
            internal["substrate_for_km_norm"],
            external["substrate_for_km_norm"],
        )
    )
    if fallback.any():
        raise AssertionError("Unexpected substrate-median fallback in TMB/H2O2 external set")

    rows = external[
        [
            "row_id",
            "primary_publication_cluster_id",
            "substrate_for_km_norm",
            "target_log10_km",
        ]
    ].copy()
    rows["substrate_median_prediction"] = baseline_prediction
    rows["substrate_median_abs_error"] = np.abs(
        y_external.to_numpy() - baseline_prediction
    )

    configurations = [("RMSE", 500)] + [
        ("MAE", iterations) for iterations in MAE_ITERATIONS
    ]
    fitted: list[dict[str, object]] = []
    for objective, iterations in configurations:
        model = CatBoostFrameRegressor(
            random_seed=SEED,
            iterations=iterations,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            loss_function=objective,
        )
        model.fit(X_internal, y_internal)
        prediction = np.asarray(model.predict(X_external), dtype=float)
        model_key = f"{objective.lower()}_{iterations}"
        prediction_column = f"catboost_{model_key}_prediction"
        error_column = f"catboost_{model_key}_abs_error"
        rows[prediction_column] = prediction
        rows[error_column] = np.abs(y_external.to_numpy() - prediction)
        metrics = regression_metrics(y_external, prediction)
        fitted.append(
            {
                "objective": objective,
                "iterations": iterations,
                "model_key": model_key,
                "error_column": error_column,
                "metrics": metrics,
            }
        )

    baseline_mae = float(rows["substrate_median_abs_error"].mean())
    rmse_error_column = "catboost_rmse_500_abs_error"
    rmse_skill = 1.0 - float(
        next(item for item in fitted if item["model_key"] == "rmse_500")[
            "metrics"
        ]["mae"]
    ) / baseline_mae
    summaries: list[dict[str, object]] = []
    for item in fitted:
        objective = str(item["objective"])
        iterations = int(item["iterations"])
        metrics = item["metrics"]
        error_column = str(item["error_column"])
        baseline_mae = float(rows["substrate_median_abs_error"].mean())
        skill = 1.0 - float(metrics["mae"]) / baseline_mae
        bootstrap = publication_cluster_bootstrap(
            rows,
            model_error_column=error_column,
            baseline_error_column="substrate_median_abs_error",
            cluster_column="primary_publication_cluster_id",
            reference_error_column=(
                rmse_error_column if objective == "MAE" else None
            ),
        )
        summaries.append(
            {
                "analysis": "external_shared_schema_objective_alignment",
                "objective": objective,
                "n_train": len(internal),
                "n_external": len(external),
                "n_publication_clusters": external[
                    "primary_publication_cluster_id"
                ].nunique(),
                "iterations": iterations,
                "depth": 6,
                "learning_rate": 0.05,
                "l2_leaf_reg": 3.0,
                "random_seed": SEED,
                "model_mae": metrics["mae"],
                "model_rmse": metrics["rmse"],
                "model_r2": metrics["r2"],
                "substrate_median_mae": baseline_mae,
                "skill_vs_substrate_median": skill,
                **bootstrap,
                "paired_skill_difference_vs_rmse": (
                    skill - rmse_skill if objective == "MAE" else 0.0
                ),
                "paired_difference_reference": (
                    "prespecified RMSE, 500 iterations"
                    if objective == "MAE"
                    else "reference"
                ),
                "n_boot": N_BOOT,
                "bootstrap_seed": SEED,
                "global_training_median": global_median,
                "training_substrate_medians": json.dumps(
                    substrate_medians, sort_keys=True
                ),
                "unseen_substrate_fallback_count": int(fallback.sum()),
                "categorical_handling": (
                    "native categorical token; explicit __MISSING__ token"
                ),
                "numerical_handling": "training-partition median imputation",
            }
        )
    return pd.DataFrame(summaries), rows


def elasticnet_fit_diagnostic(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, object]:
    fitted = clone(model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fitted.fit(X_train, y_train)
    prediction = np.asarray(fitted.predict(X_test), dtype=float)
    estimator = fitted.named_steps["model"]
    y_values = np.asarray(y_train, dtype=float)
    y_centered = y_values - float(np.mean(y_values))
    normalized_scaled_threshold = float(
        estimator.tol * np.dot(y_centered, y_centered) / len(y_centered)
    )
    dual_gap = float(np.asarray(estimator.dual_gap_).max())
    convergence_messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    metrics = regression_metrics(y_test, prediction)
    return {
        "convergence_warning_count": len(convergence_messages),
        "convergence_warning_messages": " || ".join(convergence_messages),
        "n_iter": int(np.asarray(estimator.n_iter_).max()),
        "dual_gap": dual_gap,
        "normalized_scaled_tolerance_threshold": normalized_scaled_threshold,
        "dual_gap_to_scaled_threshold_ratio": (
            dual_gap / normalized_scaled_threshold
            if normalized_scaled_threshold > 0
            else np.nan
        ),
        "max_iter_reached": int(np.asarray(estimator.n_iter_).max())
        >= int(estimator.max_iter),
        "max_abs_coefficient": float(np.max(np.abs(estimator.coef_))),
        "max_abs_prediction": float(np.max(np.abs(prediction))),
        **{f"test_{key}": value for key, value in metrics.items()},
    }


def elasticnet_convergence_audit(
    full_internal: pd.DataFrame,
    shared_internal: pd.DataFrame,
    external: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = full_internal.copy()
    cohort["leakage_graph_component_id"] = source_formula_protocol_component_ids(
        cohort.drop(columns=["vmax_value", "vmax_unit"], errors="ignore")
    )
    X, meta = build_design_only_feature_frame(cohort)
    y = cohort["target_log10_km"].astype(float)
    spec = FeatureSpec(meta["numeric_columns"], meta["categorical_columns"])
    rows: list[dict[str, object]] = []
    for repeat in range(REPEATS):
        repeat_seed = SEED + repeat
        model = model_suite(spec, repeat_seed)["elasticnet"]
        for split_type in SPLIT_TYPES:
            train_index, test_index, _ = make_split(
                cohort,
                split_type,
                repeat_seed,
                test_size=0.2,
                composition=X[meta["composition_columns"]],
                min_test_n=20,
            )
            rows.append(
                {
                    "domain": "internal_design_only",
                    "repeat": repeat,
                    "seed": repeat_seed,
                    "split_type": split_type,
                    "n_train": len(train_index),
                    "n_test": len(test_index),
                    **elasticnet_fit_diagnostic(
                        model,
                        X.loc[train_index],
                        y.loc[train_index],
                        X.loc[test_index],
                        y.loc[test_index],
                    ),
                }
            )

    shared_cohort = shared_internal.copy()
    shared_cohort["leakage_graph_component_id"] = (
        source_formula_protocol_component_ids(shared_cohort)
    )
    X_shared, shared_comp_cols = _restricted_features(shared_cohort)
    y_shared = shared_cohort["target_log10_km"].astype(float)
    shared_spec = FeatureSpec(shared_comp_cols, ["substrate_for_km_norm"])
    for repeat in range(REPEATS):
        repeat_seed = SEED + repeat
        model = model_suite(shared_spec, repeat_seed)["elasticnet"]
        for split_type in SHARED_SPLIT_TYPES:
            if split_type == "eligible_pool_random":
                train_index, test_index, _ = _eligible_pool_random_split(
                    shared_cohort, repeat_seed, 0.2
                )
            else:
                train_index, test_index, _ = make_split(
                    shared_cohort,
                    split_type,
                    repeat_seed,
                    test_size=0.2,
                    composition=X_shared[shared_comp_cols],
                    min_test_n=20,
                )
            rows.append(
                {
                    "domain": "internal_shared_schema",
                    "repeat": repeat,
                    "seed": repeat_seed,
                    "split_type": split_type,
                    "n_train": len(train_index),
                    "n_test": len(test_index),
                    **elasticnet_fit_diagnostic(
                        model,
                        X_shared.loc[train_index],
                        y_shared.loc[train_index],
                        X_shared.loc[test_index],
                        y_shared.loc[test_index],
                    ),
                }
            )

    X_internal, comp_cols = _restricted_features(shared_internal)
    X_external, _ = _restricted_features(external)
    X_external = X_external.reindex(columns=X_internal.columns)
    external_model = model_suite(
        FeatureSpec(comp_cols, ["substrate_for_km_norm"]), SEED
    )["elasticnet"]
    rows.append(
        {
            "domain": "external_shared_schema",
            "repeat": pd.NA,
            "seed": SEED,
            "split_type": "prospective_external",
            "n_train": len(shared_internal),
            "n_test": len(external),
            **elasticnet_fit_diagnostic(
                external_model,
                X_internal,
                shared_internal["target_log10_km"].astype(float),
                X_external,
                external["target_log10_km"].astype(float),
            ),
        }
    )
    fits = pd.DataFrame(rows)
    summary = (
        fits.groupby(["domain", "split_type"], dropna=False, as_index=False)
        .agg(
            n_fits=("split_type", "size"),
            fits_with_convergence_warning=(
                "convergence_warning_count",
                lambda values: int((values > 0).sum()),
            ),
            max_n_iter=("n_iter", "max"),
            max_dual_gap=("dual_gap", "max"),
            max_dual_gap_to_scaled_threshold_ratio=(
                "dual_gap_to_scaled_threshold_ratio",
                "max",
            ),
            fits_reaching_max_iter=(
                "max_iter_reached",
                lambda values: int(values.sum()),
            ),
            max_abs_coefficient=("max_abs_coefficient", "max"),
            max_abs_prediction=("max_abs_prediction", "max"),
            mean_test_mae=("test_mae", "mean"),
        )
    )
    return fits, summary


def main() -> None:
    require_catboost()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_internal, _ = prepare_modeling_data(read_table(ROOT / "data" / "nanozymes.csv"))
    shared_internal = _prepare_internal_shared_substrate(full_internal)
    external = _prepare_external_shared_substrate(
        pd.read_csv(
            ROOT
            / "outputs"
            / "tables"
            / "external_validation"
            / "external_validation_strict_model_ready.csv"
        )
    )
    frozen_clusters = pd.read_csv(
        ROOT
        / "outputs"
        / "tables"
        / "editor_major_revision"
        / "m4_substrate_baseline_row_predictions.csv",
        usecols=["domain", "row_id", "cluster_id"],
    )
    frozen_clusters = (
        frozen_clusters.loc[
            frozen_clusters["domain"].eq(
                "strict_external_restricted_shared_schema"
            ),
            ["row_id", "cluster_id"],
        ]
        .drop_duplicates("row_id")
        .rename(columns={"cluster_id": "primary_publication_cluster_id"})
    )
    frozen_clusters["row_id"] = frozen_clusters["row_id"].astype(int)
    external = external.merge(frozen_clusters, on="row_id", validate="one_to_one")
    if external["primary_publication_cluster_id"].nunique() != 127:
        raise AssertionError("Expected 127 frozen primary publication clusters")
    if (len(full_internal), len(shared_internal), len(external)) != (992, 915, 322):
        raise AssertionError(
            "Unexpected analysis populations: "
            f"{len(full_internal)}, {len(shared_internal)}, {len(external)}"
        )

    objective_summary, objective_predictions = catboost_objective_alignment(
        shared_internal, external
    )
    convergence_fits, convergence_summary = elasticnet_convergence_audit(
        full_internal, shared_internal, external
    )
    objective_summary.to_csv(
        OUTPUT_DIR / "catboost_mae_objective_external_sensitivity.csv", index=False
    )
    objective_predictions.to_csv(
        OUTPUT_DIR / "catboost_objective_external_predictions.csv", index=False
    )
    convergence_fits.to_csv(
        OUTPUT_DIR / "elasticnet_convergence_fit_log.csv", index=False
    )
    convergence_summary.to_csv(
        OUTPUT_DIR / "elasticnet_convergence_summary.csv", index=False
    )
    manifest = {
        "seed_rule": "random_state = 2026 + repeat for repeated stochastic fits",
        "external_seed": SEED,
        "catboost_prespecified_configuration": {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
        },
        "mae_iteration_sweep": MAE_ITERATIONS,
        "bootstrap_unit": "frozen primary publication cluster",
        "n_boot": N_BOOT,
        "elasticnet_design_only_internal_fits": REPEATS * len(SPLIT_TYPES),
        "elasticnet_shared_schema_internal_fits": REPEATS
        * len(SHARED_SPLIT_TYPES),
        "elasticnet_external_fits": 1,
    }
    (OUTPUT_DIR / "table_s6_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(objective_summary.to_string(index=False))
    print(convergence_summary.to_string(index=False))


if __name__ == "__main__":
    main()
