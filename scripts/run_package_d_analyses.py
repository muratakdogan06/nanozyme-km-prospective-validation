#!/usr/bin/env python3
"""Run the presubmission Package D robustness analyses.

This script is additive: it reads the frozen Package C inputs and existing
shared-schema predictions, then writes new outputs under
``outputs/tables/package_d``. It does not perform an inter-curator analysis.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import prepare_modeling_data, read_table
from scripts.run_editor_major_revision_analyses import (
    training_only_substrate_median_predictions,
)
from scripts.run_restricted_feature_validation import (
    _prepare_external_shared_substrate,
    _prepare_internal_shared_substrate,
    _restricted_features,
)


SEED = 2026
N_SPLITS = 5
N_BOOT = 10_000
GRID = {
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.10],
    "iterations": [500, 1000, 2000],
    "l2_leaf_reg": [1, 3, 10],
}
FIXED_ERROR_COLUMNS = {
    "CatBoost (fixed)": "catboost_abs_error",
    "ExtraTrees (fixed)": "extratrees_y",
    "ElasticNet (fixed)": "elasticnet_y",
}


def fit_predict_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    config: dict[str, float | int],
    seed: int,
) -> np.ndarray:
    X_train = X_train.copy()
    X_test = X_test.reindex(columns=X_train.columns).copy()
    categorical = [
        column
        for column in X_train.columns
        if X_train[column].dtype == object
        or str(X_train[column].dtype).startswith("string")
    ]
    numeric = [column for column in X_train.columns if column not in categorical]
    for column in categorical:
        X_train[column] = X_train[column].astype("string").fillna("__MISSING__").astype(str)
        X_test[column] = X_test[column].astype("string").fillna("__MISSING__").astype(str)
    for column in numeric:
        X_train[column] = pd.to_numeric(X_train[column], errors="coerce")
        X_test[column] = pd.to_numeric(X_test[column], errors="coerce")
    medians = X_train[numeric].median(numeric_only=True)
    X_train[numeric] = X_train[numeric].fillna(medians).fillna(0.0)
    X_test[numeric] = X_test[numeric].fillna(medians).fillna(0.0)
    model = CatBoostRegressor(
        iterations=int(config["iterations"]),
        depth=int(config["depth"]),
        learning_rate=float(config["learning_rate"]),
        l2_leaf_reg=float(config["l2_leaf_reg"]),
        loss_function="RMSE",
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )
    model.fit(X_train, y_train.astype(float), cat_features=categorical)
    return np.asarray(model.predict(X_test), dtype=float)


def skill(model_mae: float, baseline_mae: float) -> float:
    return float(1.0 - model_mae / baseline_mae)


def grouped_folds(groups: pd.Series, n_splits: int = N_SPLITS):
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(groups), dtype=float)
    yield from splitter.split(dummy, groups=groups.astype(str).to_numpy())


def tune_catboost(
    internal: pd.DataFrame,
    X_internal: pd.DataFrame,
    y_internal: pd.Series,
    output_dir: Path,
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    folds = list(grouped_folds(internal["source_group_id"]))
    configs = [
        dict(zip(GRID.keys(), values))
        for values in itertools.product(*(GRID[key] for key in GRID))
    ]
    grid_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for config_index, config in enumerate(configs, start=1):
        maes: list[float] = []
        baseline_maes: list[float] = []
        skills: list[float] = []
        for fold_index, (train_index, test_index) in enumerate(folds, start=1):
            prediction = fit_predict_catboost(
                X_internal.iloc[train_index],
                y_internal.iloc[train_index],
                X_internal.iloc[test_index],
                config=config,
                seed=SEED + fold_index - 1,
            )
            baseline_prediction, _, _, _ = training_only_substrate_median_predictions(
                y_internal.iloc[train_index],
                internal.iloc[train_index]["substrate_for_km_norm"],
                internal.iloc[test_index]["substrate_for_km_norm"],
            )
            model_mae = float(np.mean(np.abs(y_internal.iloc[test_index].to_numpy() - prediction)))
            baseline_mae = float(
                np.mean(np.abs(y_internal.iloc[test_index].to_numpy() - baseline_prediction))
            )
            fold_skill = skill(model_mae, baseline_mae)
            maes.append(model_mae)
            baseline_maes.append(baseline_mae)
            skills.append(fold_skill)
            fold_rows.append(
                {
                    "config_index": config_index,
                    "fold": fold_index,
                    "n_train": len(train_index),
                    "n_test": len(test_index),
                    "n_train_sources": internal.iloc[train_index]["source_group_id"].nunique(),
                    "n_test_sources": internal.iloc[test_index]["source_group_id"].nunique(),
                    **config,
                    "model_mae": model_mae,
                    "substrate_median_mae": baseline_mae,
                    "skill_vs_substrate_median": fold_skill,
                }
            )
        grid_rows.append(
            {
                "config_index": config_index,
                **config,
                "mean_fold_mae": float(np.mean(maes)),
                "sd_fold_mae": float(np.std(maes, ddof=1)),
                "mean_substrate_median_mae": float(np.mean(baseline_maes)),
                "mean_fold_skill_vs_substrate_median": float(np.mean(skills)),
                "sd_fold_skill_vs_substrate_median": float(np.std(skills, ddof=1)),
            }
        )
        print(
            f"tuning {config_index:02d}/{len(configs)} "
            f"MAE={np.mean(maes):.6f} config={config}",
            flush=True,
        )
    grid = pd.DataFrame(grid_rows).sort_values(
        ["mean_fold_mae", "depth", "learning_rate", "iterations", "l2_leaf_reg"],
        kind="mergesort",
    )
    grid.insert(0, "rank", np.arange(1, len(grid) + 1))
    folds_frame = pd.DataFrame(fold_rows)
    selected = {
        key: int(grid.iloc[0][key]) if key in {"depth", "iterations", "l2_leaf_reg"}
        else float(grid.iloc[0][key])
        for key in GRID
    }
    grid.to_csv(output_dir / "package_d_catboost_tuning_grid.csv", index=False)
    folds_frame.to_csv(output_dir / "package_d_catboost_tuning_fold_metrics.csv", index=False)
    return selected, grid, folds_frame


def cluster_bootstrap_skill(
    frame: pd.DataFrame,
    *,
    model_error_column: str,
    cluster_column: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    cluster_rows = [
        group.index.to_numpy()
        for _, group in frame.groupby(cluster_column, sort=False)
    ]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for draw in range(n_boot):
        sampled_clusters = rng.integers(0, len(cluster_rows), size=len(cluster_rows))
        indices = np.concatenate([cluster_rows[index] for index in sampled_clusters])
        sampled = frame.loc[indices]
        draws[draw] = skill(
            float(sampled[model_error_column].mean()),
            float(sampled["substrate_median_abs_error"].mean()),
        )
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high), float(draws.std(ddof=1))


def prospective_tuned_evaluation(
    internal: pd.DataFrame,
    external: pd.DataFrame,
    X_internal: pd.DataFrame,
    X_external: pd.DataFrame,
    selected: dict[str, float | int],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_internal = internal["target_log10_km"].astype(float)
    y_external = external["target_log10_km"].astype(float)
    prediction = fit_predict_catboost(
        X_internal,
        y_internal,
        X_external,
        config=selected,
        seed=SEED,
    )
    baseline_prediction, _, global_median, medians = training_only_substrate_median_predictions(
        y_internal,
        internal["substrate_for_km_norm"],
        external["substrate_for_km_norm"],
    )
    rows = external[
        [
            "row_id",
            "external_source_cluster_id",
            "curation_record_id",
            "retain_with_flag",
            "substrate_for_km_norm",
            "target_log10_km",
            "km_millimolar",
        ]
    ].copy()
    rows["tuned_catboost_prediction"] = prediction
    rows["tuned_catboost_abs_error"] = np.abs(y_external.to_numpy() - prediction)
    rows["substrate_median_prediction"] = baseline_prediction
    rows["substrate_median_abs_error"] = np.abs(y_external.to_numpy() - baseline_prediction)
    rows["global_training_median"] = global_median
    rows["training_substrate_medians"] = json.dumps(medians, sort_keys=True)
    low, high, bootstrap_se = cluster_bootstrap_skill(
        rows,
        model_error_column="tuned_catboost_abs_error",
        cluster_column="external_source_cluster_id",
        n_boot=N_BOOT,
        seed=SEED,
    )
    model_mae = float(rows["tuned_catboost_abs_error"].mean())
    baseline_mae = float(rows["substrate_median_abs_error"].mean())
    summary = pd.DataFrame(
        [
            {
                **selected,
                "n_train": len(internal),
                "n_external": len(external),
                "n_external_publications": external["external_source_cluster_id"].nunique(),
                "model_mae": model_mae,
                "substrate_median_mae": baseline_mae,
                "skill_vs_substrate_median": skill(model_mae, baseline_mae),
                "skill_ci95_low": low,
                "skill_ci95_high": high,
                "bootstrap_skill_se": bootstrap_se,
                "target_sd": float(y_external.std(ddof=1)),
                "mae_over_target_sd": model_mae / float(y_external.std(ddof=1)),
                "selection_labels_used": "internal source-grouped CV only",
                "external_labels_used_for_selection": False,
            }
        ]
    )
    rows.to_csv(output_dir / "package_d_tuned_catboost_external_predictions.csv", index=False)
    summary.to_csv(output_dir / "package_d_tuned_catboost_external_summary.csv", index=False)
    return rows, summary


def curation_sensitivity(
    fixed_rows: pd.DataFrame,
    tuned_rows: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    fixed = fixed_rows.loc[
        fixed_rows["domain"].eq("strict_external_restricted_shared_schema")
    ].copy()
    fixed["row_id"] = fixed["row_id"].astype(int)
    fixed["retain_with_flag"] = fixed["retain_with_flag"].astype(bool)
    tuned = tuned_rows[["row_id", "tuned_catboost_abs_error"]].copy()
    tuned["row_id"] = tuned["row_id"].astype(int)
    frame = fixed.merge(tuned, on="row_id", validate="one_to_one")
    km_millimolar = np.power(10.0, frame["target_log10_km"].astype(float)) * 1e3
    low_q, high_q = km_millimolar.quantile([0.025, 0.975])
    masks = {
        "All curated records": pd.Series(True, index=frame.index),
        "Unflagged records": ~frame["retain_with_flag"],
        "Central 95% of Kmapp": km_millimolar.between(low_q, high_q, inclusive="both"),
        "Hard extremes excluded": km_millimolar.between(0.001, 100.0, inclusive="both"),
    }
    expected_n = {
        "All curated records": 322,
        "Unflagged records": 305,
        "Central 95% of Kmapp": 304,
        "Hard extremes excluded": 310,
    }
    model_columns = {**FIXED_ERROR_COLUMNS, "CatBoost (grouped-CV tuned)": "tuned_catboost_abs_error"}
    rows: list[dict[str, object]] = []
    for scenario_index, (scenario, mask) in enumerate(masks.items()):
        subset = frame.loc[mask].reset_index(drop=True)
        if len(subset) != expected_n[scenario]:
            raise AssertionError(f"Unexpected {scenario} row count: {len(subset)}")
        baseline_mae = float(subset["substrate_median_abs_error"].mean())
        for model, error_column in model_columns.items():
            model_mae = float(subset[error_column].mean())
            low, high, bootstrap_se = cluster_bootstrap_skill(
                subset,
                model_error_column=error_column,
                cluster_column="cluster_id",
                n_boot=N_BOOT,
                seed=SEED + scenario_index,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "n_rows": len(subset),
                    "n_publications": subset["cluster_id"].nunique(),
                    "model": model,
                    "model_mae": model_mae,
                    "substrate_median_mae": baseline_mae,
                    "skill_vs_substrate_median": skill(model_mae, baseline_mae),
                    "skill_ci95_low": low,
                    "skill_ci95_high": high,
                    "bootstrap_skill_se": bootstrap_se,
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(output_dir / "package_d_shared_schema_curation_sensitivity.csv", index=False)
    return output


def external_grouped_cv(
    external: pd.DataFrame,
    X_external: pd.DataFrame,
    selected: dict[str, float | int],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = external["target_log10_km"].astype(float).reset_index(drop=True)
    external = external.reset_index(drop=True)
    X_external = X_external.reset_index(drop=True)
    prediction = np.full(len(external), np.nan, dtype=float)
    baseline_prediction = np.full(len(external), np.nan, dtype=float)
    fold_assignment = np.zeros(len(external), dtype=int)
    fold_rows: list[dict[str, object]] = []
    for fold_index, (train_index, test_index) in enumerate(
        grouped_folds(external["external_source_cluster_id"]), start=1
    ):
        fold_prediction = fit_predict_catboost(
            X_external.iloc[train_index],
            y.iloc[train_index],
            X_external.iloc[test_index],
            config=selected,
            seed=SEED + fold_index - 1,
        )
        fold_baseline, _, _, _ = training_only_substrate_median_predictions(
            y.iloc[train_index],
            external.iloc[train_index]["substrate_for_km_norm"],
            external.iloc[test_index]["substrate_for_km_norm"],
        )
        prediction[test_index] = fold_prediction
        baseline_prediction[test_index] = fold_baseline
        fold_assignment[test_index] = fold_index
        model_mae = float(np.mean(np.abs(y.iloc[test_index].to_numpy() - fold_prediction)))
        baseline_mae = float(np.mean(np.abs(y.iloc[test_index].to_numpy() - fold_baseline)))
        fold_rows.append(
            {
                "fold": fold_index,
                "n_train": len(train_index),
                "n_test": len(test_index),
                "n_train_publications": external.iloc[train_index]["external_source_cluster_id"].nunique(),
                "n_test_publications": external.iloc[test_index]["external_source_cluster_id"].nunique(),
                "model_mae": model_mae,
                "substrate_median_mae": baseline_mae,
                "skill_vs_substrate_median": skill(model_mae, baseline_mae),
                "target_sd": float(y.iloc[test_index].std(ddof=1)),
            }
        )
    if np.isnan(prediction).any() or np.isnan(baseline_prediction).any():
        raise AssertionError("External grouped-CV did not assign every row")
    rows = external[
        ["row_id", "external_source_cluster_id", "substrate_for_km_norm", "target_log10_km"]
    ].copy()
    rows["fold"] = fold_assignment
    rows["prediction"] = prediction
    rows["model_abs_error"] = np.abs(y.to_numpy() - prediction)
    rows["substrate_median_prediction"] = baseline_prediction
    rows["substrate_median_abs_error"] = np.abs(y.to_numpy() - baseline_prediction)
    low, high, bootstrap_se = cluster_bootstrap_skill(
        rows,
        model_error_column="model_abs_error",
        cluster_column="external_source_cluster_id",
        n_boot=N_BOOT,
        seed=SEED,
    )
    model_mae = float(rows["model_abs_error"].mean())
    baseline_mae = float(rows["substrate_median_abs_error"].mean())
    summary = pd.DataFrame(
        [
            {
                "fold": "Pooled OOF",
                "n_train": pd.NA,
                "n_test": len(external),
                "n_train_publications": pd.NA,
                "n_test_publications": external["external_source_cluster_id"].nunique(),
                "model_mae": model_mae,
                "substrate_median_mae": baseline_mae,
                "skill_vs_substrate_median": skill(model_mae, baseline_mae),
                "skill_ci95_low": low,
                "skill_ci95_high": high,
                "bootstrap_skill_se": bootstrap_se,
                "target_sd": float(y.std(ddof=1)),
            }
        ]
    )
    folds = pd.DataFrame(fold_rows)
    folds["skill_ci95_low"] = np.nan
    folds["skill_ci95_high"] = np.nan
    folds["bootstrap_skill_se"] = np.nan
    combined = pd.concat([folds, summary], ignore_index=True)
    rows.to_csv(output_dir / "package_d_external_grouped_cv_predictions.csv", index=False)
    combined.to_csv(output_dir / "package_d_external_grouped_cv_summary.csv", index=False)
    return rows, combined


def table4_dispersion(
    fixed_rows: pd.DataFrame,
    tuned_summary: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    internal = fixed_rows.loc[
        fixed_rows["domain"].eq("internal_restricted_shared_schema")
        & fixed_rows["split_type"].isin(["random", "leakage_graph"])
    ].copy()
    rows: list[dict[str, object]] = []
    for split_type, label in [("random", "Internal random"), ("leakage_graph", "Internal component")]:
        split = internal.loc[internal["split_type"].eq(split_type)]
        repeat_sd = split.groupby("repeat")["target_log10_km"].std(ddof=1)
        target_sd = float(repeat_sd.mean())
        for model, error_column in {
            "Substrate median (baseline)": "substrate_median_abs_error",
            **FIXED_ERROR_COLUMNS,
        }.items():
            repeat_mae = split.groupby("repeat")[error_column].mean()
            rows.append(
                {
                    "regime": label,
                    "model": model,
                    "target_sd": target_sd,
                    "model_mae": float(repeat_mae.mean()),
                    "mae_over_target_sd": float((repeat_mae / repeat_sd).mean()),
                    "aggregation": "mean of per-repeat values across 20 outer splits",
                }
            )
    external = fixed_rows.loc[
        fixed_rows["domain"].eq("strict_external_restricted_shared_schema")
    ].copy()
    target_sd = float(external["target_log10_km"].std(ddof=1))
    for model, error_column in {
        "Substrate median (baseline)": "substrate_median_abs_error",
        **FIXED_ERROR_COLUMNS,
    }.items():
        model_mae = float(external[error_column].mean())
        rows.append(
            {
                "regime": "External",
                "model": model,
                "target_sd": target_sd,
                "model_mae": model_mae,
                "mae_over_target_sd": model_mae / target_sd,
                "aggregation": "full external corpus",
            }
        )
    tuned = tuned_summary.iloc[0]
    rows.append(
        {
            "regime": "External",
            "model": "CatBoost (grouped-CV tuned)",
            "target_sd": target_sd,
            "model_mae": float(tuned["model_mae"]),
            "mae_over_target_sd": float(tuned["mae_over_target_sd"]),
            "aggregation": "full external corpus",
        }
    )
    output = pd.DataFrame(rows)
    output.to_csv(output_dir / "package_d_table4_dispersion_metrics.csv", index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", default="data/nanozymes.csv")
    parser.add_argument(
        "--external-data",
        default="outputs/tables/external_validation/external_validation_strict_model_ready.csv",
    )
    parser.add_argument(
        "--fixed-row-predictions",
        default="outputs/tables/editor_major_revision/m4_substrate_baseline_row_predictions.csv",
    )
    parser.add_argument("--output-dir", default="outputs/tables/package_d")
    parser.add_argument("--reuse-grid", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    internal_full, _ = prepare_modeling_data(read_table(args.train_data))
    internal = _prepare_internal_shared_substrate(internal_full)
    external = _prepare_external_shared_substrate(pd.read_csv(args.external_data))
    X_internal, _ = _restricted_features(internal)
    X_external, _ = _restricted_features(external)
    X_external = X_external.reindex(columns=X_internal.columns)
    y_internal = internal["target_log10_km"].astype(float)
    fixed_rows = pd.read_csv(args.fixed_row_predictions)
    primary_external_clusters = (
        fixed_rows.loc[
            fixed_rows["domain"].eq("strict_external_restricted_shared_schema"),
            ["row_id", "cluster_id"],
        ]
        .drop_duplicates("row_id")
        .rename(columns={"cluster_id": "primary_publication_cluster_id"})
    )
    primary_external_clusters["row_id"] = primary_external_clusters["row_id"].astype(int)
    external = external.merge(primary_external_clusters, on="row_id", validate="one_to_one")
    if external["primary_publication_cluster_id"].nunique() != 127:
        raise AssertionError("Expected the primary external analysis to use 127 publication clusters")
    # Use the same frozen publication-cluster unit as the primary Table 4 analysis.
    # This keeps tuning and learnability controls directly comparable to its bootstrap.
    external["external_source_cluster_id"] = external["primary_publication_cluster_id"].astype(str)

    grid_path = output_dir / "package_d_catboost_tuning_grid.csv"
    fold_path = output_dir / "package_d_catboost_tuning_fold_metrics.csv"
    if args.reuse_grid and grid_path.exists() and fold_path.exists():
        grid = pd.read_csv(grid_path)
        folds = pd.read_csv(fold_path)
        selected_row = grid.sort_values("rank").iloc[0]
        selected = {
            key: int(selected_row[key]) if key in {"depth", "iterations", "l2_leaf_reg"}
            else float(selected_row[key])
            for key in GRID
        }
    else:
        selected, grid, folds = tune_catboost(internal, X_internal, y_internal, output_dir)

    tuned_rows, tuned_summary = prospective_tuned_evaluation(
        internal,
        external,
        X_internal,
        X_external,
        selected,
        output_dir,
    )
    curation = curation_sensitivity(fixed_rows, tuned_rows, output_dir)
    _, external_cv = external_grouped_cv(external, X_external, selected, output_dir)
    dispersion = table4_dispersion(fixed_rows, tuned_summary, output_dir)

    manifest = {
        "analysis": "Package D robustness analyses excluding inter-curator agreement",
        "seed": SEED,
        "n_splits": N_SPLITS,
        "n_boot": N_BOOT,
        "internal_rows": len(internal),
        "internal_source_groups": int(internal["source_group_id"].nunique()),
        "external_rows": len(external),
        "external_publication_groups": int(external["external_source_cluster_id"].nunique()),
        "grid": GRID,
        "selected_config": selected,
        "selection_criterion": "minimum unweighted mean MAE across five source-grouped folds",
        "inter_curator_analysis_performed": False,
    }
    (output_dir / "package_d_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\nSelected configuration:", selected)
    print("\nTuned prospective summary:\n", tuned_summary.to_string(index=False))
    print("\nCuration sensitivity:\n", curation.to_string(index=False))
    print("\nExternal grouped CV:\n", external_cv.to_string(index=False))
    print("\nTable 4 dispersion:\n", dispersion.to_string(index=False))


if __name__ == "__main__":
    main()
