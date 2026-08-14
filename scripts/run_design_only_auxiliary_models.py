#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import prepare_modeling_data, read_table
from src.diagnostics import source_formula_protocol_component_ids
from src.evaluate import regression_metrics
from src.features import build_design_only_feature_frame
from src.models import FeatureSpec, fit_predict, model_suite
from src.splits import make_split


SPLIT_TYPES = [
    "random",
    "source_group",
    "formula_group",
    "composition_cluster",
    "leakage_graph",
]


def main() -> None:
    out = Path("outputs/tables/reviewer3_controls")
    out.mkdir(parents=True, exist_ok=True)
    df, _ = prepare_modeling_data(read_table("data/nanozymes.csv"))
    df["leakage_graph_component_id"] = source_formula_protocol_component_ids(df)
    X, meta = build_design_only_feature_frame(df)
    y = df["target_log10_km"].astype(float)
    spec = FeatureSpec(meta["numeric_columns"], meta["categorical_columns"])
    rows: list[dict] = []
    for repeat in range(20):
        seed = 2026 + repeat
        models = model_suite(spec, seed)
        for split_type in SPLIT_TYPES:
            train_idx, test_idx, _ = make_split(
                df,
                split_type,
                seed,
                test_size=0.2,
                composition=X[meta["composition_columns"]],
                min_test_n=20,
            )
            for model_name in ["elasticnet", "extratrees"]:
                pred = fit_predict(
                    models[model_name], X.loc[train_idx], y.loc[train_idx], X.loc[test_idx]
                )
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
    raw.to_csv(out / "design_only_auxiliary_model_metrics.csv", index=False)
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
    primary = pd.read_csv(
        "outputs/tables/reviewer2_sensitivity/design_only_vs_post_characterization.csv"
    )
    primary = primary.loc[
        primary["feature_schema"].eq("design_only_primary"),
        [
            "split_type",
            "model",
            "mae_mean",
            "mae_sd",
            "r2_mean",
            "r2_sd",
            "spearman_mean",
        ],
    ].copy()
    combined = pd.concat([primary, summary[primary.columns]], ignore_index=True)
    order = pd.Categorical(
        combined["split_type"], categories=SPLIT_TYPES, ordered=True
    )
    combined = combined.assign(_order=order).sort_values(["_order", "model"]).drop(columns="_order")
    combined.to_csv(out / "design_only_model_split_matrix.csv", index=False)


if __name__ == "__main__":
    main()
