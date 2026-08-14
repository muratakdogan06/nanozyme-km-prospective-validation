#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from statistics import NormalDist

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import prepare_modeling_data, read_table
from src.diagnostics import source_formula_protocol_component_ids
from src.features import build_feature_frame
from src.models import FeatureSpec, fit_predict, model_suite, require_catboost
from src.reporting import ensure_output_dirs
from src.splits import make_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the external validation sample size needed to bound an MAE "
            "estimate before curation begins."
        )
    )
    parser.add_argument(
        "--benchmark-metrics",
        default="outputs/metrics/benchmark_metrics.csv",
        help="Existing benchmark metrics table.",
    )
    parser.add_argument(
        "--data",
        default="",
        help=(
            "Optional frozen modeling table. When provided, held-out residuals are recomputed "
            "for the selected model and split and used for the external row-count estimate."
        ),
    )
    parser.add_argument(
        "--benchmark-predictions",
        default="",
        help=(
            "Optional held-out prediction table with abs_error_log10_km, model, and split_type "
            "columns. If omitted, repeat-level MAE variability is used as a conservative planning proxy."
        ),
    )
    parser.add_argument("--model", default="catboost")
    parser.add_argument("--split-type", default="leakage_graph")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-composition-test-n", type=int, default=20)
    parser.add_argument("--missingness-indicators", action="store_true")
    parser.add_argument("--effect-size", type=float, default=0.12)
    parser.add_argument(
        "--target-half-width",
        type=float,
        default=None,
        help=(
            "Desired 95%% CI half-width for the external MAE estimate. "
            "Defaults to --effect-size."
        ),
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    if args.target_half_width is None:
        args.target_half_width = args.effect_size

    rows = []
    if args.data:
        rows.append(_estimate_from_internal_residuals(args))
    if args.benchmark_predictions:
        prediction_path = Path(args.benchmark_predictions)
        if not prediction_path.exists():
            raise FileNotFoundError(f"Benchmark predictions not found: {prediction_path}")
        predictions = pd.read_csv(prediction_path)
        rows.append(_estimate_from_predictions(predictions, args))
    metrics_path = Path(args.benchmark_metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Benchmark metrics not found: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    rows.append(_estimate_from_repeat_metrics(metrics, args))

    out = pd.DataFrame(rows)
    output_dir = ensure_output_dirs()["tables"] / "external_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "external_sample_size_planning.csv"
    out.to_csv(output_path, index=False)
    print(f"Wrote external sample-size planning to {output_path}")


def _estimate_from_predictions(predictions: pd.DataFrame, args) -> dict:
    required = {"model", "split_type"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Benchmark prediction table missing columns: {sorted(missing)}")
    filtered = predictions[
        (predictions["model"] == args.model)
        & (predictions["split_type"] == args.split_type)
    ].copy()
    if "abs_error_log10_km" not in filtered.columns:
        if {"target_log10_km", "pred_log10_km"} <= set(filtered.columns):
            filtered["abs_error_log10_km"] = (
                filtered["target_log10_km"] - filtered["pred_log10_km"]
            ).abs()
        else:
            raise ValueError(
                "Prediction table needs abs_error_log10_km or target_log10_km/pred_log10_km."
            )
    values = filtered["abs_error_log10_km"].dropna().astype(float)
    return _sample_size_row(
        values,
        args,
        variability_source="held_out_absolute_errors",
        limitation=(
            "uses row-level held-out absolute-error variability from the selected internal split"
        ),
        estimate_external_rows=True,
    )


def _estimate_from_internal_residuals(args) -> dict:
    require_catboost()
    raw = read_table(args.data)
    df, _ = prepare_modeling_data(raw)
    if args.split_type == "leakage_graph":
        df["leakage_graph_component_id"] = source_formula_protocol_component_ids(df)
    X, meta = build_feature_frame(df, include_missing_indicators=args.missingness_indicators)
    y = df["target_log10_km"].astype(float)
    feature_spec = FeatureSpec(meta["numeric_columns"], meta["categorical_columns"])
    models = model_suite(feature_spec, args.seed)
    if args.model not in models:
        raise ValueError(f"Unknown model {args.model!r}; available: {sorted(models)}")

    abs_errors = []
    for repeat in range(args.repeats):
        rep_seed = args.seed + repeat
        train_idx, test_idx, _ = make_split(
            df,
            args.split_type,
            rep_seed,
            test_size=args.test_size,
            composition=X[meta["composition_columns"]],
            min_test_n=args.min_composition_test_n,
        )
        pred = fit_predict(models[args.model], X.loc[train_idx], y.loc[train_idx], X.loc[test_idx])
        abs_errors.extend((y.loc[test_idx].to_numpy(dtype=float) - pred).tolist())
    values = pd.Series(abs_errors, dtype=float).abs()
    return _sample_size_row(
        values,
        args,
        variability_source="recomputed_internal_held_out_absolute_errors",
        limitation=(
            "uses pooled held-out absolute errors from repeated internal splits; repeated "
            "test-set overlap may make the effective residual sample smaller than the row count"
        ),
        estimate_external_rows=True,
    )


def _estimate_from_repeat_metrics(metrics: pd.DataFrame, args) -> dict:
    required = {"model", "split_type", "mae"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Benchmark metrics table missing columns: {sorted(missing)}")
    values = metrics[
        (metrics["model"] == args.model) & (metrics["split_type"] == args.split_type)
    ]["mae"].dropna().astype(float)
    return _sample_size_row(
        values,
        args,
        variability_source="repeat_level_mae",
        limitation=(
            "planning proxy only; repeat-level MAE variability is not the same as "
            "row-level external absolute-error variability"
        ),
        estimate_external_rows=False,
    )


def _sample_size_row(
    values: pd.Series,
    args,
    variability_source: str,
    limitation: str,
    estimate_external_rows: bool,
) -> dict:
    if len(values) < 2:
        sd = math.nan
        n_required = math.nan
    else:
        sd = float(values.std(ddof=1))
        z = NormalDist().inv_cdf(0.5 + args.confidence / 2.0)
        n_required = (
            math.ceil((z * sd / args.target_half_width) ** 2)
            if estimate_external_rows
            else math.nan
        )
    return {
        "model": args.model,
        "split_type": args.split_type,
        "variability_source": variability_source,
        "n_observations_used_for_sd": int(len(values)),
        "mean_mae_or_abs_error": float(values.mean()) if len(values) else math.nan,
        "sd_used_for_planning": sd,
        "effect_size_log10_mae": float(args.effect_size),
        "target_ci_half_width_log10_mae": float(args.target_half_width),
        "confidence": float(args.confidence),
        "minimum_external_rows": n_required,
        "recommendation": (
            "curate at least this many retained external rows before interpreting "
            "a log10-MAE difference of the requested size"
            if estimate_external_rows
            else "provide --data or --benchmark-predictions for a row-level external sample-size target"
        ),
        "limitation": limitation,
    }


if __name__ == "__main__":
    main()
