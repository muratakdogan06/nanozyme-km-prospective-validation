#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import prepare_modeling_data, read_table
from src.diagnostics import external_ood_severity_rows, source_formula_protocol_component_ids
from src.features import build_feature_frame
from src.models import FeatureSpec, fit_predict, model_suite, require_catboost
from src.reporting import ensure_output_dirs


SPLIT_TYPES = ["random", "source_group", "formula_group", "composition_cluster", "leakage_graph"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct internal row-level CatBoost errors for existing split manifests "
            "and score each held-out row against its paired training fold."
        )
    )
    parser.add_argument("--data", default="data/nanozymes.csv")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--missingness-indicators", action="store_true")
    args = parser.parse_args()

    require_catboost()
    dirs = ensure_output_dirs()
    table_dir = dirs["tables"]
    out_dir = table_dir / "internal_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_table(args.data)
    df, _ = prepare_modeling_data(raw)
    df["leakage_graph_component_id"] = source_formula_protocol_component_ids(df)
    X, meta = build_feature_frame(df, include_missing_indicators=args.missingness_indicators)
    y = df["target_log10_km"].astype(float)
    feature_spec = FeatureSpec(meta["numeric_columns"], meta["categorical_columns"])
    catboost = model_suite(feature_spec, args.seed)["catboost"]
    row_id_to_index = df.reset_index().set_index("row_id")["index"].to_dict()

    predictions = []
    for repeat in range(args.repeats):
        for split_type in SPLIT_TYPES:
            manifest_path = dirs["split_manifests"] / f"{split_type}_repeat_{repeat}.csv"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing split manifest: {manifest_path}")
            manifest = pd.read_csv(manifest_path)
            train_row_ids = manifest.loc[manifest["split"].eq("train"), "row_id"].astype(int)
            test_row_ids = manifest.loc[manifest["split"].eq("test"), "row_id"].astype(int)
            train_idx = np.array([row_id_to_index[row_id] for row_id in train_row_ids], dtype=int)
            test_idx = np.array([row_id_to_index[row_id] for row_id in test_row_ids], dtype=int)

            pred = fit_predict(catboost, X.loc[train_idx], y.loc[train_idx], X.loc[test_idx])
            severity = external_ood_severity_rows(
                df.loc[train_idx],
                df.loc[test_idx],
                X.loc[train_idx],
                X.loc[test_idx],
                meta["composition_columns"],
            )
            pred_table = df.loc[
                test_idx,
                [
                    "row_id",
                    "formula",
                    "standardized_formula",
                    "source_group_id",
                    "km_value",
                    "km_unit",
                    "km_common_molar",
                    "target_log10_km",
                ],
            ].copy()
            pred_table = pred_table.merge(severity, on="row_id", how="left")
            pred_table["repeat"] = repeat
            pred_table["seed"] = args.seed + repeat
            pred_table["split_type"] = split_type
            pred_table["model"] = "catboost"
            pred_table["pred_log10_km"] = pred
            pred_table["abs_error_log10_km"] = (
                pred_table["target_log10_km"] - pred_table["pred_log10_km"]
            ).abs()
            predictions.append(pred_table)

    row_level = pd.concat(predictions, ignore_index=True)
    row_level.to_csv(out_dir / "internal_row_level_catboost_predictions.csv", index=False)
    association = row_level_severity_error_association(row_level)
    association.to_csv(out_dir / "internal_row_level_severity_error_association.csv", index=False)
    print(f"Wrote {out_dir / 'internal_row_level_catboost_predictions.csv'}")
    print(f"Wrote {out_dir / 'internal_row_level_severity_error_association.csv'}")


def row_level_severity_error_association(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_type, split_group in predictions.groupby("split_type", dropna=False):
        rows.extend(_association_rows(split_group, split_type))
    rows.extend(_association_rows(predictions, "all_internal_test_rows"))
    return pd.DataFrame(rows)


def _association_rows(group: pd.DataFrame, split_type: str) -> list[dict]:
    rows = []
    unique = (
        group.groupby("row_id", as_index=False)
        .agg(
            ood_severity_index=("ood_severity_index", "mean"),
            source_formula_free_severity_index=("source_formula_free_severity_index", "mean"),
            graph_free_composition_severity_index=("graph_free_composition_severity_index", "mean"),
            abs_error_log10_km=("abs_error_log10_km", "mean"),
        )
    )
    for severity_col in [
        "ood_severity_index",
        "source_formula_free_severity_index",
        "graph_free_composition_severity_index",
    ]:
        valid = unique[[severity_col, "abs_error_log10_km"]].dropna()
        statistic = np.nan
        p_value = np.nan
        if (
            len(valid) > 2
            and valid[severity_col].nunique() > 1
            and valid["abs_error_log10_km"].nunique() > 1
        ):
            result = spearmanr(valid[severity_col], valid["abs_error_log10_km"])
            statistic = float(result.statistic) if np.isfinite(result.statistic) else np.nan
            p_value = float(result.pvalue) if np.isfinite(result.pvalue) else np.nan
        rows.append(
            {
                "model": "catboost",
                "split_type": split_type,
                "severity_score": severity_col,
                "error_metric": "abs_error_log10_km",
                "aggregation_unit": "unique_row_mean_across_repeats",
                "n_test_rows_before_aggregation": int(len(group[[severity_col, "abs_error_log10_km"]].dropna())),
                "n_unique_rows": int(len(valid)),
                "spearman_r": statistic,
                "spearman_p": p_value,
                "interpretation": (
                    "internal association after averaging repeated held-out predictions "
                    "for each unique row to avoid pseudo-replication"
                ),
            }
        )
    return rows


if __name__ == "__main__":
    main()
