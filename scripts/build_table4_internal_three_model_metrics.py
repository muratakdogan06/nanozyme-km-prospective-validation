#!/usr/bin/env python3
"""Build internal three-model rows for manuscript Table 4."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODEL_ERROR_COLUMNS = {
    "CatBoost": "catboost_abs_error",
    "ExtraTrees": "extratrees_y",
    "ElasticNet": "elasticnet_y",
}
SPLITS = {
    "random": "Random",
    "leakage_graph": "Component holdout",
}
EXPECTED_SUBSTRATE_ROUNDED = {
    ("Random", "CatBoost"): (0.804, 0.053, -0.014, 0.105),
    ("Random", "ExtraTrees"): (0.790, 0.070, 0.003, 0.129),
    ("Random", "ElasticNet"): (0.885, -0.041, -0.095, 0.000),
    ("Component holdout", "CatBoost"): (0.850, -0.044, -0.120, 0.025),
    ("Component holdout", "ExtraTrees"): (0.856, -0.050, -0.126, 0.032),
    ("Component holdout", "ElasticNet"): (0.935, -0.148, -0.285, -0.067),
}


def build_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    internal = frame.loc[
        frame["domain"].eq("internal_restricted_shared_schema")
    ].reset_index(drop=True)
    if internal["repeat"].nunique() != 20:
        raise ValueError("Expected 20 internal outer-split repeats")

    rows: list[dict[str, object]] = []
    for split_type, split_label in SPLITS.items():
        split = internal.loc[internal["split_type"].eq(split_type)]
        if split.empty:
            raise ValueError(f"Missing internal split {split_type}")
        for model, error_column in MODEL_ERROR_COLUMNS.items():
            repeat_metrics = split.groupby("repeat", sort=True).agg(
                model_mae=(error_column, "mean"),
                substrate_median_mae=("substrate_median_abs_error", "mean"),
                global_median_mae=("global_median_abs_error", "mean"),
                n_test=(error_column, "size"),
            )
            repeat_metrics["skill_vs_substrate_median"] = (
                1.0
                - repeat_metrics["model_mae"]
                / repeat_metrics["substrate_median_mae"]
            )
            repeat_metrics["skill_vs_global_median"] = (
                1.0
                - repeat_metrics["model_mae"]
                / repeat_metrics["global_median_mae"]
            )
            substrate_ci = repeat_metrics["skill_vs_substrate_median"].quantile(
                [0.025, 0.975]
            )
            global_ci = repeat_metrics["skill_vs_global_median"].quantile(
                [0.025, 0.975]
            )
            observed = (
                round(float(repeat_metrics["model_mae"].mean()), 3),
                round(float(repeat_metrics["skill_vs_substrate_median"].mean()), 3),
                round(float(substrate_ci.loc[0.025]), 3),
                round(float(substrate_ci.loc[0.975]), 3),
            )
            if observed != EXPECTED_SUBSTRATE_ROUNDED[(split_label, model)]:
                raise RuntimeError(
                    f"Table 4 internal regression mismatch for "
                    f"{split_label}/{model}: {observed}"
                )
            rows.append(
                {
                    "split_type": split_type,
                    "split_label": split_label,
                    "model": model,
                    "n_repeats": len(repeat_metrics),
                    "mean_n_test": float(repeat_metrics["n_test"].mean()),
                    "model_mae_mean": float(repeat_metrics["model_mae"].mean()),
                    "substrate_median_mae_mean": float(
                        repeat_metrics["substrate_median_mae"].mean()
                    ),
                    "skill_vs_substrate_median_mean": float(
                        repeat_metrics["skill_vs_substrate_median"].mean()
                    ),
                    "skill_vs_substrate_median_ci95_low": float(
                        substrate_ci.loc[0.025]
                    ),
                    "skill_vs_substrate_median_ci95_high": float(
                        substrate_ci.loc[0.975]
                    ),
                    "global_median_mae_mean": float(
                        repeat_metrics["global_median_mae"].mean()
                    ),
                    "skill_vs_global_median_mean": float(
                        repeat_metrics["skill_vs_global_median"].mean()
                    ),
                    "skill_vs_global_median_ci95_low": float(global_ci.loc[0.025]),
                    "skill_vs_global_median_ci95_high": float(global_ci.loc[0.975]),
                    "uncertainty_unit": "outer split repeat",
                    "interval_type": (
                        "empirical 2.5th-97.5th percentile across 20 repeats"
                    ),
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
            "outputs/tables/publication/table4_internal_three_model_metrics.csv"
        ),
    )
    args = parser.parse_args()
    metrics = build_metrics(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output, index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
