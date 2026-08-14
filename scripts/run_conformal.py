#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conformal import (
    absolute_residual_intervals,
    few_shot_recalibration,
    interval_metrics,
    residual_normalized_intervals,
)
from src.data import prepare_modeling_data, read_table
from src.diagnostics import source_formula_protocol_component_ids
from src.features import build_feature_frame
from src.models import FeatureSpec, model_suite, require_catboost
from src.reporting import ensure_output_dirs, make_main_tables
from src.splits import make_split


SPLIT_TYPES = ["random", "source_group", "formula_group", "composition_cluster", "leakage_graph"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--few-shot-repeats", type=int, default=100)
    args = parser.parse_args()

    if not args.data.strip():
        parser.error(
            '--data was empty. If using --data "$DATA", set DATA first '
            "or pass a file path such as --data data/nanozymes.csv."
        )

    try:
        require_catboost()
    except ImportError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    dirs = ensure_output_dirs()
    raw = read_table(args.data)
    df, _ = prepare_modeling_data(raw)
    df["leakage_graph_component_id"] = source_formula_protocol_component_ids(df)
    X, meta = build_feature_frame(df)
    y = df["target_log10_km"].astype(float)
    feature_spec = FeatureSpec(meta["numeric_columns"], meta["categorical_columns"])
    rows = []
    for rep in range(args.repeats):
        rep_seed = args.seed + rep
        base_model = model_suite(feature_spec, rep_seed)["catboost"]
        for split_type in SPLIT_TYPES:
            train_idx, test_idx, _ = make_split(
                df,
                split_type,
                rep_seed,
                composition=X[meta["composition_columns"]],
            )
            proper_idx, cal_idx = train_test_split(
                train_idx, test_size=0.25, random_state=rep_seed, shuffle=True
            )
            methods = {
                "absolute_residual": absolute_residual_intervals,
                "residual_normalized": residual_normalized_intervals,
            }
            for method_name, method in methods.items():
                result = method(
                    base_model,
                    X.loc[proper_idx],
                    y.loc[proper_idx],
                    X.loc[cal_idx],
                    y.loc[cal_idx],
                    X.loc[test_idx],
                    alpha=args.alpha,
                )
                metrics = interval_metrics(
                    y.loc[test_idx],
                    result.y_pred,
                    result.lower,
                    result.upper,
                    alpha=args.alpha,
                )
                metrics.update(
                    {
                        "feature_schema": "post_characterization_full_992",
                        "repeat": rep,
                        "split_type": split_type,
                        "method": method_name,
                        "shots": 0,
                        "underpowered": bool(not np.isfinite(result.qhat)),
                        "n_test": len(test_idx),
                    }
                )
                rows.append(metrics)
                recal = few_shot_recalibration(
                    y.loc[test_idx],
                    result.y_pred,
                    result.lower,
                    result.upper,
                    repeats=args.few_shot_repeats,
                    alpha=args.alpha,
                    seed=rep_seed,
                )
                recal["repeat"] = rep
                recal["feature_schema"] = "post_characterization_full_992"
                recal["split_type"] = split_type
                recal["method"] = method_name
                recal["n_test"] = len(test_idx)
                rows.extend(recal[recal["shots"] > 0].to_dict("records"))

    conformal = pd.DataFrame(rows)
    conformal.to_csv(dirs["metrics"] / "conformal_metrics.csv", index=False)
    conformal.to_csv(dirs["tables"] / "supp_table_s6_full_few_shot_recalibration.csv", index=False)
    conformal_summary(conformal).to_csv(
        dirs["tables"] / "supp_table_s19_conformal_summary.csv", index=False
    )
    calibration_fragility_index(conformal).to_csv(
        dirs["tables"] / "supp_table_s25_calibration_fragility_index.csv", index=False
    )
    conformal_interpretation_note(args.alpha).to_csv(
        dirs["tables"] / "supp_table_s18_conformal_interpretation_note.csv", index=False
    )
    make_main_tables(
        dirs["tables"] / "audit",
        pd.read_csv(dirs["metrics"] / "benchmark_metrics.csv")
        if (dirs["metrics"] / "benchmark_metrics.csv").exists()
        else pd.DataFrame(),
        conformal,
        dirs["tables"],
    )
    print(f"Conformal analysis complete: {dirs['metrics'] / 'conformal_metrics.csv'}")


def conformal_summary(conformal: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "mae",
        "empirical_coverage",
        "coverage_gap",
        "mean_interval_width",
        "median_interval_width",
        "interval_score",
    ]
    rows = []
    group_cols = ["feature_schema", "split_type", "method", "shots"]
    for keys, group in conformal.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_outer_repeats"] = int(group["repeat"].nunique())
        row["n_rows"] = int(len(group))
        row["n_underpowered"] = (
            int(group.loc[group["underpowered"].astype(bool), "repeat"].nunique())
            if "underpowered" in group
            else 0
        )
        repeat_level = group.groupby("repeat", as_index=False)[metric_cols].mean(numeric_only=True)
        row["summary_unit"] = (
            "outer-repeat mean over few-shot resamples"
            if int(row["shots"]) > 0
            else "outer split repeat"
        )
        for metric in metric_cols:
            vals = repeat_level[metric].dropna().astype(float)
            row[f"{metric}_mean"] = vals.mean() if len(vals) else pd.NA
            row[f"{metric}_ci95_low"] = vals.quantile(0.025) if len(vals) > 1 else pd.NA
            row[f"{metric}_ci95_high"] = vals.quantile(0.975) if len(vals) > 1 else pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_fragility_index(conformal: pd.DataFrame) -> pd.DataFrame:
    summary = conformal_summary(conformal)
    rows = []
    for (split_type, method), group in summary.groupby(["split_type", "method"], dropna=False):
        base = group[group["shots"] == 0]
        if base.empty:
            continue
        base_row = base.iloc[0]
        base_gap = float(base_row["coverage_gap_mean"])
        base_width = float(base_row["mean_interval_width_mean"])
        base_coverage = float(base_row["empirical_coverage_mean"])
        for _, row in group.iterrows():
            if pd.isna(row["coverage_gap_mean"]):
                continue
            gap = float(row["coverage_gap_mean"])
            width = float(row["mean_interval_width_mean"])
            coverage = float(row["empirical_coverage_mean"])
            rows.append(
                {
                    "split_type": split_type,
                    "method": method,
                    "shots": int(row["shots"]),
                    "calibration_fragility_index": abs(gap),
                    "baseline_shots0_fragility": abs(base_gap),
                    "recalibration_gain_vs_shots0": abs(base_gap) - abs(gap),
                    "coverage_change_vs_shots0": coverage - base_coverage,
                    "width_change_vs_shots0": width - base_width,
                    "interpretation": (
                        "positive recalibration_gain means fewer absolute coverage-gap points "
                        "relative to the no-target-domain-calibration baseline"
                    ),
                }
            )
    return pd.DataFrame(rows)


def conformal_interpretation_note(alpha: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "analysis_component": "split conformal intervals",
                "nominal_coverage": 1 - alpha,
                "validity_scope": "exchangeability/approximately in-distribution calibration",
                "ood_interpretation": "under source, formula, or composition shift, coverage is empirical stress-test evidence rather than a formal guarantee",
                "recommended_manuscript_language": "We report empirical coverage and interval width under predefined OOD splits; conformal guarantees are not claimed under covariate shift.",
            },
            {
                "analysis_component": "few-shot recalibration",
                "nominal_coverage": 1 - alpha,
                "validity_scope": "limited target-domain calibration subset",
                "ood_interpretation": "few-shot rows quantify how much target-domain calibration may repair coverage gaps",
                "recommended_manuscript_language": "Few-shot recalibration is presented as an empirical adaptation experiment, not as evidence of universal OOD validity.",
            },
        ]
    )


if __name__ == "__main__":
    main()
