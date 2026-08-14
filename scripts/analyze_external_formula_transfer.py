#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import prepare_modeling_data, read_table


def cluster_bootstrap_skill_draws(
    paired: pd.DataFrame, n_boot: int = 5000, seed: int = 2026
) -> tuple[float, np.ndarray]:
    clusters = paired["curation_record_id"].astype(str)
    unique_clusters = clusters.unique()
    cluster_rows = {
        cluster: np.flatnonzero(clusters.to_numpy() == cluster) for cluster in unique_clusters
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        rows = np.concatenate([cluster_rows[cluster] for cluster in sampled])
        median_mae = paired["median"].to_numpy()[rows].mean()
        catboost_mae = paired["catboost"].to_numpy()[rows].mean()
        draws[index] = 1.0 - catboost_mae / median_mae
    point = 1.0 - paired["catboost"].mean() / paired["median"].mean()
    return point, draws


def joint_cluster_bootstrap_skill_difference(
    paired: pd.DataFrame, n_boot: int = 5000, seed: int = 2026
) -> tuple[float, np.ndarray]:
    clusters = paired["curation_record_id"].astype(str)
    unique_clusters = clusters.unique()
    cluster_rows = {
        cluster: np.flatnonzero(clusters.to_numpy() == cluster) for cluster in unique_clusters
    }
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        rows = np.concatenate([cluster_rows[cluster] for cluster in sampled])
        draw = paired.iloc[rows]
        skills = {}
        for stratum, group in draw.groupby("formula_transfer_stratum"):
            skills[stratum] = 1.0 - group["catboost"].mean() / group["median"].mean()
        if {"seen_formula", "unseen_formula"}.issubset(skills):
            draws.append(skills["seen_formula"] - skills["unseen_formula"])
    point_skills = {}
    for stratum, group in paired.groupby("formula_transfer_stratum"):
        point_skills[stratum] = 1.0 - group["catboost"].mean() / group["median"].mean()
    point = point_skills["seen_formula"] - point_skills["unseen_formula"]
    return point, np.asarray(draws, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal", default="data/nanozymes.csv")
    parser.add_argument(
        "--predictions",
        default=(
            "outputs/tables/restricted_feature_validation/"
            "restricted_feature_external_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/tables/publication/table27_external_seen_formula_transfer.csv",
    )
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    internal, _ = prepare_modeling_data(read_table(args.internal))
    seen_formulas = set(internal["standardized_formula"].astype(str))
    predictions = pd.read_csv(args.predictions)
    predictions["formula_transfer_stratum"] = np.where(
        predictions["standardized_formula"].astype(str).isin(seen_formulas),
        "seen_formula",
        "unseen_formula",
    )

    rows = []
    skill_draws: dict[str, np.ndarray] = {}
    for stratum, stratum_rows in predictions.groupby("formula_transfer_stratum"):
        for model, group in stratum_rows.groupby("model"):
            y_true = group["target_log10_km"].astype(float).to_numpy()
            y_pred = group["pred_log10_km"].astype(float).to_numpy()
            rho = (
                spearmanr(y_true, y_pred).statistic
                if len(group) > 1 and np.std(y_pred) > 1e-12 and np.std(y_true) > 1e-12
                else np.nan
            )
            rows.append(
                {
                    "formula_transfer_stratum": stratum,
                    "model": model,
                    "n_records": int(group["row_id"].nunique()),
                    "n_publications": int(group["curation_record_id"].nunique()),
                    "n_formulas": int(group["standardized_formula"].nunique()),
                    "mae": float(mean_absolute_error(y_true, y_pred)),
                    "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
                    "r2": float(r2_score(y_true, y_pred)) if len(group) > 1 else np.nan,
                    "spearman": float(rho) if np.isfinite(rho) else np.nan,
                }
            )

        paired = stratum_rows.pivot_table(
            index=[
                "row_id",
                "curation_record_id",
                "standardized_formula",
                "target_log10_km",
            ],
            columns="model",
            values="abs_error_log10_km",
            aggfunc="first",
        ).reset_index()
        point, draws = cluster_bootstrap_skill_draws(paired, args.n_boot, args.seed)
        skill_draws[stratum] = draws
        rows.append(
            {
                "formula_transfer_stratum": stratum,
                "model": "catboost_vs_median_skill",
                "n_records": int(len(paired)),
                "n_publications": int(paired["curation_record_id"].nunique()),
                "n_formulas": int(paired["standardized_formula"].nunique()),
                "mae": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "spearman": np.nan,
                "skill": point,
                "skill_ci95_low": float(np.quantile(draws, 0.025)),
                "skill_ci95_high": float(np.quantile(draws, 0.975)),
                "bootstrap_unit": "curation_record_id",
                "n_boot": args.n_boot,
                "bootstrap_seed": args.seed,
            }
        )

    if {"seen_formula", "unseen_formula"}.issubset(skill_draws):
        full_paired = predictions.pivot_table(
            index=[
                "row_id",
                "curation_record_id",
                "standardized_formula",
                "formula_transfer_stratum",
                "target_log10_km",
            ],
            columns="model",
            values="abs_error_log10_km",
            aggfunc="first",
        ).reset_index()
        point_difference, difference = joint_cluster_bootstrap_skill_difference(
            full_paired, args.n_boot, args.seed
        )
        rows.append(
            {
                "formula_transfer_stratum": "seen_minus_unseen",
                "model": "catboost_vs_median_skill_difference",
                "skill": point_difference,
                "skill_ci95_low": float(np.quantile(difference, 0.025)),
                "skill_ci95_high": float(np.quantile(difference, 0.975)),
                "bootstrap_unit": "joint curation_record_id bootstrap",
                "n_boot": args.n_boot,
                "bootstrap_seed": args.seed,
            }
        )

    output = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
