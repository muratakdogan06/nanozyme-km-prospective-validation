#!/usr/bin/env python3
"""Build the three-model external rows inserted into manuscript Table 4."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ERROR_COLUMNS = {
    "CatBoost": "catboost_abs_error",
    "ExtraTrees": "extratrees_y",
    "ElasticNet": "elasticnet_y",
}
EXPECTED_ROUNDED = {
    "CatBoost": (0.776, -0.027, -0.098, 0.042),
    "ExtraTrees": (0.787, -0.040, -0.127, 0.041),
    "ElasticNet": (0.789, -0.044, -0.136, 0.031),
}
Z_975 = 1.959963984540054
Z_80 = 0.8416212335729143


def build_metrics(frame: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    external = frame.loc[
        frame["domain"].eq("strict_external_restricted_shared_schema")
    ].reset_index(drop=True)
    if len(external) != 322 or external["cluster_id"].nunique() != 127:
        raise ValueError("Expected 322 external records from 127 publication clusters")
    cluster_rows = [
        group.index.to_numpy()
        for _, group in external.groupby("cluster_id", sort=False)
    ]
    rng = np.random.default_rng(seed)
    draw_indices = []
    for _ in range(n_boot):
        sampled = rng.integers(0, len(cluster_rows), size=len(cluster_rows))
        draw_indices.append(np.concatenate([cluster_rows[index] for index in sampled]))

    baseline_mae = float(external["substrate_median_abs_error"].mean())
    rows = []
    for model, error_column in MODEL_ERROR_COLUMNS.items():
        model_mae = float(external[error_column].mean())
        skill = float(1.0 - model_mae / baseline_mae)
        draws = np.empty(n_boot, dtype=float)
        for draw, indices in enumerate(draw_indices):
            sampled = external.loc[indices]
            sampled_baseline = float(sampled["substrate_median_abs_error"].mean())
            sampled_model = float(sampled[error_column].mean())
            draws[draw] = 1.0 - sampled_model / sampled_baseline
        ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
        bootstrap_skill_se = float(draws.std(ddof=1))
        approximate_mde = float((Z_975 + Z_80) * bootstrap_skill_se)
        observed_rounded = tuple(
            round(value, 3) for value in (model_mae, skill, ci_low, ci_high)
        )
        if observed_rounded != EXPECTED_ROUNDED[model]:
            raise RuntimeError(
                f"Table 4 regression mismatch for {model}: {observed_rounded}"
            )
        rows.append(
            {
                "model": model,
                "n_external": len(external),
                "n_publication_clusters": external["cluster_id"].nunique(),
                "model_mae": model_mae,
                "substrate_median_mae": baseline_mae,
                "skill_vs_substrate_median": skill,
                "skill_ci95_low": float(ci_low),
                "skill_ci95_high": float(ci_high),
                "bootstrap_skill_se": bootstrap_skill_se,
                "approx_mde_skill_80pct_two_sided_alpha_0_05": approximate_mde,
                "mde_method": (
                    "normal approximation: (z_0.975 + z_0.80) * "
                    "publication-cluster bootstrap SE"
                ),
                "bootstrap_unit": "external_source_cluster_id (publication)",
                "n_boot": n_boot,
                "bootstrap_seed": seed,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/tables/editor_major_revision/"
            "m4_substrate_baseline_row_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/tables/publication/table4_external_three_model_metrics.csv"
        ),
    )
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    metrics = build_metrics(pd.read_csv(args.input), n_boot=args.n_boot, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output, index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
