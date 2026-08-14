#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

_tmp_cache = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(_tmp_cache / "nanozyme_mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(_tmp_cache / "nanozyme_xdg_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import prepare_modeling_data, read_table
from src.diagnostics import source_formula_protocol_component_ids
from src.evaluate import regression_metrics
from src.features import MISSING_TOKEN, build_feature_frame
from src.models import FeatureSpec, fit_predict, model_suite, require_catboost
from src.reporting import ensure_output_dirs
from src.splits import make_split


SPLIT_TYPES = [
    "random",
    "eligible_pool_random",
    "source_group",
    "formula_group",
    "leakage_graph",
    "composition_cluster",
]
SPLIT_LABELS = {
    "random": "Random",
    "eligible_pool_random": "Eligible-pool random",
    "source_group": "Source",
    "formula_group": "Formula",
    "composition_cluster": "Composition cluster",
    "leakage_graph": "Component",
}
MODEL_LABELS = {
    "median": "Median",
    "elasticnet": "ElasticNet",
    "extratrees": "ExtraTrees",
    "catboost": "CatBoost",
}
MODEL_COLORS = {
    "median": "#9c9c9c",
    "elasticnet": "#5b8db8",
    "extratrees": "#7ca66a",
    "catboost": "#c2755b",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fair restricted-feature validation using only descriptors available "
            "for both the internal training cohort and the external Km/Vmax validation schema."
        )
    )
    parser.add_argument("--train-data", default="data/nanozymes.csv")
    parser.add_argument(
        "--external-data",
        default="outputs/tables/external_validation/external_validation_strict_model_ready.csv",
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-composition-test-n", type=int, default=20)
    args = parser.parse_args()

    require_catboost()
    dirs = ensure_output_dirs()
    out_dir = dirs["tables"] / "restricted_feature_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = dirs["figures"] / "publication"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pub_table_dir = dirs["tables"] / "publication"
    pub_table_dir.mkdir(parents=True, exist_ok=True)

    train_raw = read_table(args.train_data)
    train_full, _ = prepare_modeling_data(train_raw)
    train = _prepare_internal_shared_substrate(train_full)
    train["leakage_graph_component_id"] = source_formula_protocol_component_ids(train)
    external = _prepare_external_shared_substrate(pd.read_csv(args.external_data))

    X_train, comp_cols = _restricted_features(train)
    X_external, _ = _restricted_features(external)
    X_external = X_external.reindex(columns=X_train.columns)
    y_train = train["target_log10_km"].astype(float)
    y_external = external["target_log10_km"].astype(float)

    internal_metrics, split_diagnostics, internal_predictions = _run_internal_restricted_benchmark(
        train,
        X_train,
        y_train,
        comp_cols,
        args,
    )
    external_metrics, external_predictions = _run_external_restricted_validation(
        train,
        external,
        X_train,
        y_train,
        X_external,
        y_external,
        args.seed,
    )

    internal_metrics.to_csv(out_dir / "restricted_feature_internal_metrics.csv", index=False)
    split_diagnostics.to_csv(out_dir / "restricted_feature_split_diagnostics.csv", index=False)
    internal_predictions.to_csv(out_dir / "restricted_feature_internal_predictions.csv", index=False)
    external_metrics.to_csv(out_dir / "restricted_feature_external_metrics.csv", index=False)
    external_predictions.to_csv(out_dir / "restricted_feature_external_predictions.csv", index=False)
    _write_publication_tables(
        internal_metrics,
        split_diagnostics,
        internal_predictions,
        external_metrics,
        external_predictions,
        train,
        external,
        X_train,
        X_external,
        comp_cols,
        pub_table_dir,
    )
    _plot_restricted_feature_validation(internal_metrics, external_metrics, external_predictions, fig_dir)
    _plot_connection_dose_response(pub_table_dir / "table26_graph_connection_dose_response.csv", fig_dir)
    _update_publication_manifests(pub_table_dir, fig_dir)
    print(f"Wrote restricted-feature outputs to {out_dir}")


def _prepare_internal_shared_substrate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["substrate_for_km_norm"] = out["reaction_type"].map(_substrate_from_reaction)
    out = out.loc[out["substrate_for_km_norm"].isin(["TMB", "H2O2"])].copy()
    out["substrate_for_km_norm"] = out["substrate_for_km_norm"].astype(str)
    return out.reset_index(drop=True)


def _prepare_external_shared_substrate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "external_row_id" in out.columns:
        out["row_id"] = out["external_row_id"].astype(int)
    elif "row_id" not in out.columns:
        out["row_id"] = np.arange(len(out), dtype=int)
    required = ["standardized_formula", "formula_group_id", "target_log10_km", "substrate_for_km_norm"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"External restricted validation missing columns: {missing}")
    out["substrate_for_km_norm"] = out["substrate_for_km_norm"].astype("string").fillna("").astype(str)
    out = out.loc[out["substrate_for_km_norm"].isin(["TMB", "H2O2"])].copy()
    return out.reset_index(drop=True)


def _restricted_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    X_full, meta = build_feature_frame(df)
    comp_cols = list(meta["composition_columns"])
    X = X_full[comp_cols].copy()
    X["substrate_for_km_norm"] = (
        df["substrate_for_km_norm"].astype("string").fillna(MISSING_TOKEN).astype(str).to_numpy()
    )
    return X, comp_cols


def _run_internal_restricted_benchmark(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    comp_cols: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    split_rows = []
    pred_rows = []
    for repeat in range(args.repeats):
        seed = args.seed + repeat
        models = _restricted_models(seed, comp_cols, ["substrate_for_km_norm"])
        for split_type in SPLIT_TYPES:
            if split_type == "eligible_pool_random":
                train_idx, test_idx, split_info = _eligible_pool_random_split(df, seed, args.test_size)
            else:
                train_idx, test_idx, split_info = make_split(
                    df,
                    split_type,
                    seed,
                    test_size=args.test_size,
                    composition=X[comp_cols],
                    min_test_n=args.min_composition_test_n,
                )
            split_rows.append(_split_diagnostic_row(df, y, train_idx, test_idx, repeat, seed, split_type, split_info))
            fold_row_diagnostics = _fold_row_diagnostics(df, X, comp_cols, train_idx, test_idx)
            for model_name, model in models.items():
                pred = fit_predict(model, X.loc[train_idx], y.loc[train_idx], X.loc[test_idx])
                rows.append(
                    {
                        "domain": "internal_restricted",
                        "repeat": repeat,
                        "seed": seed,
                        "split_type": split_type,
                        "model": model_name,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        **regression_metrics(y.loc[test_idx], pred),
                        "pred_sd": float(np.std(pred, ddof=1)) if len(pred) > 1 else np.nan,
                        "obs_sd": float(np.std(y.loc[test_idx], ddof=1)) if len(test_idx) > 1 else np.nan,
                    }
                )
                rows[-1]["mae_over_obs_sd"] = (
                    rows[-1]["mae"] / rows[-1]["obs_sd"] if rows[-1]["obs_sd"] and rows[-1]["obs_sd"] > 0 else np.nan
                )
                pred_table = fold_row_diagnostics.copy()
                pred_table["domain"] = "internal_restricted"
                pred_table["repeat"] = repeat
                pred_table["seed"] = seed
                pred_table["split_type"] = split_type
                pred_table["split_label"] = SPLIT_LABELS[split_type]
                pred_table["model"] = model_name
                pred_table["target_log10_km"] = y.loc[test_idx].astype(float).to_numpy()
                pred_table["pred_log10_km"] = pred
                pred_table["abs_error_log10_km"] = (pred_table["target_log10_km"] - pred_table["pred_log10_km"]).abs()
                pred_rows.append(pred_table)
    return pd.DataFrame(rows), pd.DataFrame(split_rows), pd.concat(pred_rows, ignore_index=True)


def _run_external_restricted_validation(
    train: pd.DataFrame,
    external: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_external: pd.DataFrame,
    y_external: pd.Series,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    preds = []
    numeric_columns = [col for col in X_train.columns if col != "substrate_for_km_norm"]
    for model_name, model in _restricted_models(seed, numeric_columns, ["substrate_for_km_norm"]).items():
        pred = fit_predict(model, X_train, y_train, X_external)
        metrics = regression_metrics(y_external, pred)
        rows.append(
            {
                "domain": "external_restricted",
                "scenario": "strict_external_shared_schema",
                "model": model_name,
                "n_train": int(len(train)),
                "n_external": int(len(external)),
                **metrics,
                "pred_sd": float(np.std(pred, ddof=1)) if len(pred) > 1 else np.nan,
                "obs_sd": float(np.std(y_external, ddof=1)) if len(y_external) > 1 else np.nan,
                "pred_obs_sd_ratio": float(np.std(pred, ddof=1) / np.std(y_external, ddof=1))
                if len(pred) > 1 and np.std(y_external, ddof=1) > 0
                else np.nan,
            }
        )
        rows[-1]["mae_over_obs_sd"] = (
            rows[-1]["mae"] / rows[-1]["obs_sd"] if rows[-1]["obs_sd"] and rows[-1]["obs_sd"] > 0 else np.nan
        )
        pred_table = external[
            [
                "row_id",
                "formula",
                "standardized_formula",
                "substrate_for_km_norm",
                "target_log10_km",
                "km_value",
                "km_unit",
            ]
        ].copy()
        for col in ["curation_record_id", "retain_with_flag", "manual_audit_decision", "journal", "year"]:
            if col in external.columns:
                pred_table[col] = external[col]
        pred_table["model"] = model_name
        pred_table["pred_log10_km"] = pred
        pred_table["abs_error_log10_km"] = (pred_table["target_log10_km"] - pred).abs()
        preds.append(pred_table)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def _restricted_models(seed: int, numeric_columns: list[str], categorical_columns: list[str]) -> dict[str, object]:
    return model_suite(FeatureSpec(numeric_columns=numeric_columns, categorical_columns=categorical_columns), seed)


def _fold_row_diagnostics(
    df: pd.DataFrame,
    X: pd.DataFrame,
    comp_cols: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    component_sizes = df.groupby("leakage_graph_component_id", dropna=False).size().sort_values(ascending=False)
    largest_component = component_sizes.index[0]
    train_component_counts = df.loc[train_idx, "leakage_graph_component_id"].value_counts(dropna=False)
    out = df.loc[
        test_idx,
        [
            "formula",
            "standardized_formula",
            "formula_group_id",
            "source_group_id",
            "leakage_graph_component_id",
            "substrate_for_km_norm",
        ],
    ].copy()
    out.insert(0, "internal_row_id", test_idx)
    out["leakage_graph_component_id"] = out["leakage_graph_component_id"].astype(str)
    out["is_largest_leakage_component"] = df.loc[test_idx, "leakage_graph_component_id"].eq(largest_component).to_numpy()
    out["component_size"] = df.loc[test_idx, "leakage_graph_component_id"].map(component_sizes).astype(int).to_numpy()
    out["train_same_component_n"] = (
        df.loc[test_idx, "leakage_graph_component_id"].map(train_component_counts).fillna(0).astype(int).to_numpy()
    )
    out["has_train_component_neighbor"] = out["train_same_component_n"] > 0
    out["nearest_train_feature_distance"] = _nearest_train_feature_distance(X, comp_cols, train_idx, test_idx)
    return out.reset_index(drop=True)


def _nearest_train_feature_distance(
    X: pd.DataFrame,
    comp_cols: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    chunk_size: int = 256,
) -> np.ndarray:
    X_train_num = X.loc[train_idx, comp_cols].apply(pd.to_numeric, errors="coerce")
    medians = X_train_num.median(numeric_only=True)
    scale = X_train_num.std(ddof=0).replace(0, 1.0).fillna(1.0)
    train_num = ((X_train_num.fillna(medians).fillna(0.0) - medians) / scale).to_numpy(dtype=float)
    test_num = (
        (X.loc[test_idx, comp_cols].apply(pd.to_numeric, errors="coerce").fillna(medians).fillna(0.0) - medians)
        / scale
    ).to_numpy(dtype=float)
    train_substrate = X.loc[train_idx, "substrate_for_km_norm"].astype(str).to_numpy()
    test_substrate = X.loc[test_idx, "substrate_for_km_norm"].astype(str).to_numpy()
    mins = np.empty(len(test_idx), dtype=float)
    for start in range(0, len(test_idx), chunk_size):
        stop = min(start + chunk_size, len(test_idx))
        diff = test_num[start:stop, None, :] - train_num[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        substrate_mismatch = (test_substrate[start:stop, None] != train_substrate[None, :]).astype(float)
        mins[start:stop] = np.sqrt(np.min(dist2 + substrate_mismatch, axis=1))
    return mins


def _split_diagnostic_row(
    df: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    repeat: int,
    seed: int,
    split_type: str,
    split_info: dict,
) -> dict:
    test = df.loc[test_idx]
    train = df.loc[train_idx]
    target = y.loc[test_idx].astype(float)
    component_overlap = _group_overlap_count(train, test, "leakage_graph_component_id")
    test_component_ids = sorted(test["leakage_graph_component_id"].astype(str).unique())
    row = {
        "repeat": int(repeat),
        "seed": int(seed),
        "split_type": split_type,
        "split_label": SPLIT_LABELS[split_type],
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_fraction": float(len(test_idx) / len(df)) if len(df) else np.nan,
        "test_unique_formulas": _nunique_if_present(test, "formula_group_id"),
        "train_unique_formulas": _nunique_if_present(train, "formula_group_id"),
        "test_unique_sources": _nunique_if_present(test, "source_group_id"),
        "train_unique_sources": _nunique_if_present(train, "source_group_id"),
        "test_unique_leakage_components": _nunique_if_present(test, "leakage_graph_component_id"),
        "train_unique_leakage_components": _nunique_if_present(train, "leakage_graph_component_id"),
        "leakage_graph_component_train_test_overlap_count": component_overlap,
        "leakage_graph_component_integrity_pass": bool(component_overlap == 0),
        "test_leakage_component_ids": ";".join(test_component_ids) if split_type == "leakage_graph" else "",
        "source_group_train_test_overlap_count": _group_overlap_count(train, test, "source_group_id"),
        "formula_group_train_test_overlap_count": _group_overlap_count(train, test, "formula_group_id"),
        "target_mean": float(target.mean()) if len(target) else np.nan,
        "target_sd": float(target.std(ddof=1)) if len(target) > 1 else np.nan,
        "target_iqr": float(target.quantile(0.75) - target.quantile(0.25)) if len(target) else np.nan,
        "target_min": float(target.min()) if len(target) else np.nan,
        "target_max": float(target.max()) if len(target) else np.nan,
    }
    for key, value in split_info.items():
        if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)) or value is None:
            row[f"split_info_{key}"] = value
    return row


def _nunique_if_present(df: pd.DataFrame, col: str) -> int:
    return int(df[col].nunique(dropna=False)) if col in df.columns else 0


def _group_overlap_count(train: pd.DataFrame, test: pd.DataFrame, col: str) -> int:
    if col not in train.columns or col not in test.columns:
        return 0
    return int(len(set(train[col].astype(str)) & set(test[col].astype(str))))


def _eligible_pool_random_split(df: pd.DataFrame, seed: int, test_size: float) -> tuple[np.ndarray, np.ndarray, dict]:
    component_sizes = df.groupby("leakage_graph_component_id", dropna=False).size().sort_values(ascending=False)
    largest_component = component_sizes.index[0]
    largest_component_n = int(component_sizes.iloc[0])
    eligible_idx = df.index[~df["leakage_graph_component_id"].eq(largest_component)].to_numpy()
    target_n = min(len(eligible_idx), int(round(len(df) * test_size)))
    rng = np.random.default_rng(seed)
    test_idx = np.sort(rng.choice(eligible_idx, size=target_n, replace=False))
    train_idx = np.setdiff1d(df.index.to_numpy(), test_idx, assume_unique=False)
    return train_idx, test_idx, {
        "split_type": "eligible_pool_random",
        "held_in_train_component_id": str(largest_component),
        "held_in_train_component_n": largest_component_n,
        "eligible_pool_n": int(len(eligible_idx)),
        "target_test_n": int(target_n),
        "effective_test_fraction_of_eligible_pool": float(target_n / len(eligible_idx)) if len(eligible_idx) else np.nan,
        "control_rationale": (
            "keep the largest leakage-graph component in training and randomly sample the same test budget "
            "from the remaining eligible component pool without enforcing component integrity"
        ),
    }


def _write_publication_tables(
    internal_metrics: pd.DataFrame,
    split_diagnostics: pd.DataFrame,
    internal_predictions: pd.DataFrame,
    external_metrics: pd.DataFrame,
    external_predictions: pd.DataFrame,
    train: pd.DataFrame,
    external: pd.DataFrame,
    X_train: pd.DataFrame,
    X_external: pd.DataFrame,
    comp_cols: list[str],
    table_dir: Path,
) -> None:
    internal_summary = _summarize_internal(internal_metrics)
    external_summary = _add_external_mae_ci(external_metrics, external_predictions)
    delta_rows = []
    for split_type, group in internal_metrics.groupby("split_type"):
        paired = group.pivot_table(index="repeat", columns="model", values="mae", aggfunc="first")
        diff = (paired["catboost"] - paired["median"]).dropna()
        delta_rows.append(_delta_row("internal", SPLIT_LABELS[split_type], diff.to_numpy(), len(diff)))
    paired_external = external_predictions.pivot_table(
        index="row_id", columns="model", values="abs_error_log10_km", aggfunc="first"
    )
    external_diff = (paired_external["catboost"] - paired_external["median"]).dropna()
    external_diff_clusters = (
        external_predictions.drop_duplicates("row_id")
        .set_index("row_id")
        .loc[external_diff.index]
        .pipe(_external_bootstrap_clusters)
    )
    delta_rows.append(
        _delta_row(
            "external",
            "Strict external shared schema",
            external_diff.to_numpy(),
            len(external_diff),
            clusters=external_diff_clusters,
        )
    )
    delta = pd.DataFrame(delta_rows)
    split_summary = _summarize_split_diagnostics(split_diagnostics, internal_metrics)
    normalized = _normalized_generalization_summary(internal_metrics, external_summary, external_predictions, delta)
    informative_null = _informative_null_table(delta, internal_metrics, external_predictions)
    component_summary, component_map = _leakage_component_structure_tables(train, split_diagnostics)
    matched_pool_decomposition = _matched_pool_skill_decomposition_table(internal_metrics)
    giant_component_random = _random_giant_component_skill_table(internal_predictions)
    connection_dose_response = _connection_dose_response_table(
        internal_predictions,
        external_predictions,
        X_train,
        X_external,
        comp_cols,
    )
    model_connection_dose_response = _model_connection_dose_response_table(internal_predictions, external_predictions)
    schema = pd.DataFrame(
        [
            {
                "analysis": "restricted_feature_validation",
                "internal_training_rows": int(len(train)),
                "external_rows": int(len(external)),
                "included_internal_substrates": "TMB;H2O2",
                "included_feature_groups": "formula-derived composition descriptors;substrate_for_km_norm",
                "excluded_feature_groups": "protocol numeric fields; morphology; surface/coating; source metadata; Vmax",
                "interpretation": (
                    "Fair external-validation schema using only feature groups available by "
                    "construction in both internal and external data."
                ),
            }
        ]
    )
    internal_summary.to_csv(table_dir / "table15_restricted_feature_internal_metrics.csv", index=False)
    external_summary.to_csv(table_dir / "table16_restricted_feature_external_metrics.csv", index=False)
    delta.to_csv(table_dir / "table17_restricted_feature_delta_mae.csv", index=False)
    schema.to_csv(table_dir / "table18_restricted_feature_schema.csv", index=False)
    split_summary.to_csv(table_dir / "table19_restricted_feature_split_diagnostics.csv", index=False)
    informative_null.to_csv(table_dir / "table20_restricted_external_informative_null.csv", index=False)
    normalized.to_csv(table_dir / "table21_restricted_normalized_generalization_summary.csv", index=False)
    component_summary.to_csv(table_dir / "table22_restricted_leakage_component_summary.csv", index=False)
    component_map.to_csv(table_dir / "table23_restricted_leakage_component_holdout_map.csv", index=False)
    matched_pool_decomposition.to_csv(
        table_dir / "table24_restricted_matched_pool_skill_decomposition.csv", index=False
    )
    giant_component_random.to_csv(table_dir / "table25_random_giant_component_skill_diagnostics.csv", index=False)
    connection_dose_response.to_csv(table_dir / "table26_graph_connection_dose_response.csv", index=False)
    model_connection_dose_response.to_csv(
        table_dir.parent / "supp_table_s32_model_connection_dose_response.csv", index=False
    )


def _summarize_internal(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_type, model), group in metrics.groupby(["split_type", "model"]):
        row = {
            "split_type": split_type,
            "split_label": SPLIT_LABELS[split_type],
            "model": model,
            "model_label": MODEL_LABELS[model],
            "n_repeats": int(group["repeat"].nunique()),
            "mean_n_train": float(group["n_train"].mean()),
            "mean_n_test": float(group["n_test"].mean()),
        }
        for metric in ["mae", "rmse", "r2", "spearman", "pred_sd", "obs_sd", "mae_over_obs_sd"]:
            vals = group[metric].dropna().astype(float)
            row[metric] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_ci95_low"] = float(vals.quantile(0.025)) if len(vals) > 1 else np.nan
            row[f"{metric}_ci95_high"] = float(vals.quantile(0.975)) if len(vals) > 1 else np.nan
        row["pred_obs_sd_ratio"] = row["pred_sd"] / row["obs_sd"] if row["obs_sd"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split_type", "model"])


def _add_external_mae_ci(external_metrics: pd.DataFrame, external_predictions: pd.DataFrame) -> pd.DataFrame:
    out = external_metrics.copy()
    ci_rows = []
    for model, group in external_predictions.groupby("model"):
        low, high = _bootstrap_cluster_mean_ci(
            group["abs_error_log10_km"].to_numpy(),
            _external_bootstrap_clusters(group),
        )
        ci_rows.append({"model": model, "mae_ci95_low": low, "mae_ci95_high": high})
    ci = pd.DataFrame(ci_rows)
    return out.merge(ci, on="model", how="left")


def _summarize_split_diagnostics(split_diagnostics: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    paired = metrics.pivot_table(index=["split_type", "repeat"], columns="model", values="mae", aggfunc="first")
    paired = paired.reset_index()
    if {"catboost", "median"}.issubset(paired.columns):
        paired["delta_mae_catboost_minus_median"] = paired["catboost"] - paired["median"]
    merged = split_diagnostics.merge(paired, on=["split_type", "repeat"], how="left")
    rows = []
    summary_cols = [
        "n_test",
        "test_fraction",
        "test_unique_formulas",
        "test_unique_sources",
        "test_unique_leakage_components",
        "leakage_graph_component_train_test_overlap_count",
        "target_sd",
        "target_iqr",
        "median",
        "catboost",
        "delta_mae_catboost_minus_median",
    ]
    for split_type, group in merged.groupby("split_type"):
        row = {
            "split_type": split_type,
            "split_label": SPLIT_LABELS[split_type],
            "n_repeats": int(group["repeat"].nunique()),
        }
        for col in summary_cols:
            vals = group[col].dropna().astype(float) if col in group.columns else pd.Series(dtype=float)
            label = {
                "median": "median_mae",
                "catboost": "catboost_mae",
            }.get(col, col)
            row[f"mean_{label}"] = float(vals.mean()) if len(vals) else np.nan
            row[f"min_{label}"] = float(vals.min()) if len(vals) else np.nan
            row[f"max_{label}"] = float(vals.max()) if len(vals) else np.nan
        row["median_mae_drop_vs_random"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and "random" in set(out["split_type"]):
        random_median = float(out.loc[out["split_type"].eq("random"), "mean_median_mae"].iloc[0])
        random_target_sd = float(out.loc[out["split_type"].eq("random"), "mean_target_sd"].iloc[0])
        out["median_mae_drop_vs_random"] = out["mean_median_mae"] - random_median
        out["target_sd_shift_vs_random"] = out["mean_target_sd"] - random_target_sd
    return out.sort_values("split_type")


def _normalized_generalization_summary(
    internal_metrics: pd.DataFrame,
    external_summary: pd.DataFrame,
    external_predictions: pd.DataFrame,
    delta: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for split_type in SPLIT_TYPES:
        paired = internal_metrics.loc[internal_metrics["split_type"].eq(split_type)].pivot_table(
            index="repeat", columns="model", values=["mae", "obs_sd"], aggfunc="first"
        )
        if ("mae", "median") not in paired.columns or ("mae", "catboost") not in paired.columns:
            continue
        median_mae = paired[("mae", "median")].astype(float)
        catboost_mae = paired[("mae", "catboost")].astype(float)
        target_sd = paired[("obs_sd", "catboost")].astype(float)
        median_mae_over_sd = median_mae / target_sd
        catboost_mae_over_sd = catboost_mae / target_sd
        skill = 1.0 - (catboost_mae / median_mae)
        delta_row = delta.loc[(delta["domain"].eq("internal")) & (delta["scenario"].eq(SPLIT_LABELS[split_type]))]
        delta_value = float(delta_row["delta_mae_catboost_minus_median"].iloc[0]) if not delta_row.empty else np.nan
        cat_low, cat_high = _bootstrap_mean_ci(catboost_mae_over_sd.to_numpy())
        med_low, med_high = _bootstrap_mean_ci(median_mae_over_sd.to_numpy())
        skill_low, skill_high = _bootstrap_mean_ci(skill.to_numpy())
        rows.append(
            {
                "domain": "internal",
                "scenario": SPLIT_LABELS[split_type],
                "catboost_mae": float(catboost_mae.mean()),
                "target_sd": float(target_sd.mean()),
                "catboost_mae_over_target_sd": float(catboost_mae_over_sd.mean()),
                "catboost_mae_over_target_sd_ci95_low": cat_low,
                "catboost_mae_over_target_sd_ci95_high": cat_high,
                "median_mae": float(median_mae.mean()),
                "median_mae_over_target_sd": float(median_mae_over_sd.mean()),
                "median_mae_over_target_sd_ci95_low": med_low,
                "median_mae_over_target_sd_ci95_high": med_high,
                "delta_mae_catboost_minus_median": delta_value,
                "delta_mae_fraction_of_median_mae": delta_value / float(median_mae.mean())
                if float(median_mae.mean()) != 0
                else np.nan,
                "skill_score_catboost_vs_median": float(skill.mean()),
                "skill_score_ci95_low": skill_low,
                "skill_score_ci95_high": skill_high,
                "primary_use": "internal_proxy_assessment",
            }
        )
    median_ext = external_summary.loc[external_summary["model"].eq("median")]
    catboost_ext = external_summary.loc[external_summary["model"].eq("catboost")]
    delta_ext = delta.loc[delta["domain"].eq("external")]
    if not median_ext.empty and not catboost_ext.empty and not delta_ext.empty:
        median_ext = median_ext.iloc[0]
        catboost_ext = catboost_ext.iloc[0]
        delta_value = float(delta_ext["delta_mae_catboost_minus_median"].iloc[0])
        ext_norm = _external_normalized_bootstrap_summary(external_predictions)
        rows.append(
            {
                "domain": "external",
                "scenario": "Strict external shared schema",
                "catboost_mae": float(catboost_ext["mae"]),
                "target_sd": float(catboost_ext["obs_sd"]),
                "catboost_mae_over_target_sd": float(ext_norm["catboost_mae_over_target_sd"]),
                "catboost_mae_over_target_sd_ci95_low": float(ext_norm["catboost_mae_over_target_sd_ci95_low"]),
                "catboost_mae_over_target_sd_ci95_high": float(ext_norm["catboost_mae_over_target_sd_ci95_high"]),
                "median_mae": float(median_ext["mae"]),
                "median_mae_over_target_sd": float(ext_norm["median_mae_over_target_sd"]),
                "median_mae_over_target_sd_ci95_low": float(ext_norm["median_mae_over_target_sd_ci95_low"]),
                "median_mae_over_target_sd_ci95_high": float(ext_norm["median_mae_over_target_sd_ci95_high"]),
                "delta_mae_catboost_minus_median": delta_value,
                "delta_mae_fraction_of_median_mae": delta_value / float(median_ext["mae"])
                if float(median_ext["mae"]) != 0
                else np.nan,
                "skill_score_catboost_vs_median": float(ext_norm["skill_score_catboost_vs_median"]),
                "skill_score_ci95_low": float(ext_norm["skill_score_ci95_low"]),
                "skill_score_ci95_high": float(ext_norm["skill_score_ci95_high"]),
                "primary_use": "independent_external_result",
            }
        )
    return pd.DataFrame(rows)


def _external_normalized_bootstrap_summary(
    external_predictions: pd.DataFrame, n_boot: int = 5000, seed: int = 2026
) -> dict:
    paired = _external_paired_error_frame(external_predictions)
    median_abs = paired["median"].astype(float).to_numpy()
    cat_abs = paired["catboost"].astype(float).to_numpy()
    y = paired["target_log10_km"].astype(float).to_numpy()
    boot = _external_cluster_bootstrap_indices(paired, n_boot=n_boot, seed=seed)
    med_mae = np.array([median_abs[idx].mean() for idx in boot])
    cat_mae = np.array([cat_abs[idx].mean() for idx in boot])
    target_sd = np.array([np.std(y[idx], ddof=1) for idx in boot])
    valid = target_sd > 0
    med_over_sd = med_mae[valid] / target_sd[valid]
    cat_over_sd = cat_mae[valid] / target_sd[valid]
    skill = 1.0 - (cat_mae / med_mae)
    return {
        "catboost_mae_over_target_sd": float(np.mean(cat_abs) / np.std(y, ddof=1)),
        "catboost_mae_over_target_sd_ci95_low": float(np.quantile(cat_over_sd, 0.025)),
        "catboost_mae_over_target_sd_ci95_high": float(np.quantile(cat_over_sd, 0.975)),
        "median_mae_over_target_sd": float(np.mean(median_abs) / np.std(y, ddof=1)),
        "median_mae_over_target_sd_ci95_low": float(np.quantile(med_over_sd, 0.025)),
        "median_mae_over_target_sd_ci95_high": float(np.quantile(med_over_sd, 0.975)),
        "skill_score_catboost_vs_median": float(1.0 - (np.mean(cat_abs) / np.mean(median_abs))),
        "skill_score_ci95_low": float(np.quantile(skill, 0.025)),
        "skill_score_ci95_high": float(np.quantile(skill, 0.975)),
        "bootstrap_unit": _external_bootstrap_unit(external_predictions),
        "n_bootstrap_clusters": int(paired["bootstrap_cluster"].nunique()),
    }


def _leakage_component_structure_tables(
    train: pd.DataFrame, split_diagnostics: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    component_map = (
        train.groupby("leakage_graph_component_id", dropna=False)
        .size()
        .rename("component_n")
        .reset_index()
    )
    component_map["leakage_graph_component_id"] = component_map["leakage_graph_component_id"].astype(str)
    leakage_rows = split_diagnostics.loc[split_diagnostics["split_type"].eq("leakage_graph")]
    heldout_counts: dict[str, int] = {}
    for value in leakage_rows["test_leakage_component_ids"].fillna(""):
        for component_id in str(value).split(";"):
            if component_id:
                heldout_counts[component_id] = heldout_counts.get(component_id, 0) + 1
    component_map["heldout_repeat_count"] = (
        component_map["leakage_graph_component_id"].map(heldout_counts).fillna(0).astype(int)
    )
    component_map["ever_heldout_by_leakage_graph"] = component_map["heldout_repeat_count"] > 0
    component_map = component_map.sort_values("component_n", ascending=False).reset_index(drop=True)
    component_map["component_size_rank"] = np.arange(1, len(component_map) + 1)
    total_rows = int(len(train))
    component_map["row_fraction"] = component_map["component_n"] / total_rows if total_rows else np.nan
    test_budget_n = int(leakage_rows["n_test"].median()) if len(leakage_rows) else 0
    mean_train_n = float(leakage_rows["n_train"].mean()) if len(leakage_rows) else np.nan
    component_map["larger_than_leakage_test_budget"] = component_map["component_n"] > test_budget_n
    never = component_map.loc[~component_map["ever_heldout_by_leakage_graph"]]
    held = component_map.loc[component_map["ever_heldout_by_leakage_graph"]]
    largest_n = int(component_map["component_n"].max()) if len(component_map) else 0
    summary = pd.DataFrame(
        [
            {
                "analysis": "restricted_leakage_graph_component_structure",
                "n_rows": total_rows,
                "n_components": int(len(component_map)),
                "test_budget_n": test_budget_n,
                "largest_component_n": largest_n,
                "largest_component_fraction": float(largest_n / total_rows) if total_rows else np.nan,
                "eligible_pool_n_excluding_largest_component": int(total_rows - largest_n),
                "effective_test_fraction_of_eligible_pool": float(test_budget_n / (total_rows - largest_n))
                if total_rows > largest_n
                else np.nan,
                "largest_component_fraction_of_mean_training_set": float(largest_n / mean_train_n)
                if mean_train_n and mean_train_n > 0
                else np.nan,
                "largest_component_testable_under_20pct_budget": bool(largest_n <= test_budget_n),
                "n_components_larger_than_test_budget": int(component_map["larger_than_leakage_test_budget"].sum()),
                "n_components_ever_heldout": int(component_map["ever_heldout_by_leakage_graph"].sum()),
                "n_components_never_heldout": int((~component_map["ever_heldout_by_leakage_graph"]).sum()),
                "n_rows_ever_heldout": int(held["component_n"].sum()) if len(held) else 0,
                "n_rows_never_heldout": int(never["component_n"].sum()) if len(never) else 0,
                "fraction_rows_never_heldout": float(never["component_n"].sum() / total_rows) if total_rows else np.nan,
                "mean_heldout_component_n": float(held["component_n"].mean()) if len(held) else np.nan,
                "max_heldout_component_n": int(held["component_n"].max()) if len(held) else 0,
                "interpretation": (
                    "Leakage-graph test folds are assembled from components that fit the test-size budget; "
                    "oversized connected components remain structurally untested under this holdout design."
                ),
            }
        ]
    )
    return summary, component_map


def _matched_pool_skill_decomposition_table(internal_metrics: pd.DataFrame) -> pd.DataFrame:
    skill = _internal_skill_by_repeat(internal_metrics)
    required = {"random", "eligible_pool_random", "leakage_graph"}
    if not required.issubset(set(skill.columns)):
        return pd.DataFrame()
    random_skill = skill["random"].dropna().to_numpy()
    eligible_skill = skill["eligible_pool_random"].dropna().to_numpy()
    leakage_skill = skill["leakage_graph"].dropna().to_numpy()
    random_minus_eligible = (skill["random"] - skill["eligible_pool_random"]).dropna().to_numpy()
    eligible_minus_leakage = (skill["eligible_pool_random"] - skill["leakage_graph"]).dropna().to_numpy()
    random_minus_leakage = (skill["random"] - skill["leakage_graph"]).dropna().to_numpy()
    re_low, re_high = _bootstrap_mean_ci(random_minus_eligible)
    el_low, el_high = _bootstrap_mean_ci(eligible_minus_leakage)
    rl_low, rl_high = _bootstrap_mean_ci(random_minus_leakage)
    total = float(np.mean(random_minus_leakage))
    row = {
        "analysis": "eligible_pool_matched_random_control",
        "random_skill_score": float(np.mean(random_skill)),
        "eligible_pool_random_skill_score": float(np.mean(eligible_skill)),
        "leakage_graph_skill_score": float(np.mean(leakage_skill)),
        "drop_random_to_eligible_pool": float(np.mean(random_minus_eligible)),
        "drop_random_to_eligible_pool_ci95_low": re_low,
        "drop_random_to_eligible_pool_ci95_high": re_high,
        "drop_eligible_pool_to_leakage_graph": float(np.mean(eligible_minus_leakage)),
        "drop_eligible_pool_to_leakage_graph_ci95_low": el_low,
        "drop_eligible_pool_to_leakage_graph_ci95_high": el_high,
        "total_drop_random_to_leakage_graph": total,
        "total_drop_random_to_leakage_graph_ci95_low": rl_low,
        "total_drop_random_to_leakage_graph_ci95_high": rl_high,
        "fraction_total_drop_attributable_to_pool_restriction": float(np.mean(random_minus_eligible) / total)
        if total
        else np.nan,
        "fraction_total_drop_attributable_to_component_integrity": float(np.mean(eligible_minus_leakage) / total)
        if total
        else np.nan,
        "fraction_total_drop_attributable_to_dense_graph_connection_loss": float(np.mean(random_minus_eligible) / total)
        if total
        else np.nan,
        "fraction_total_drop_attributable_to_weak_pool_component_integrity": float(
            np.mean(eligible_minus_leakage) / total
        )
        if total
        else np.nan,
        "interpretation": (
            "Eligible-pool random keeps the largest component in training and samples the same test budget "
            "from the non-giant component pool. The random-to-eligible drop should be interpreted as the "
            "loss of dense graph connection to training records, not as a non-leakage population effect; "
            "the eligible-to-leakage drop estimates the additional loss from enforcing component integrity "
            "within that weakly connected pool."
        ),
    }
    return pd.DataFrame([row])


def _internal_skill_by_repeat(internal_metrics: pd.DataFrame) -> pd.DataFrame:
    paired = internal_metrics.pivot_table(index=["split_type", "repeat"], columns="model", values="mae", aggfunc="first")
    paired = paired.reset_index()
    paired["skill_score_catboost_vs_median"] = 1.0 - (paired["catboost"] / paired["median"])
    return paired.pivot(index="repeat", columns="split_type", values="skill_score_catboost_vs_median")


def _random_giant_component_skill_table(internal_predictions: pd.DataFrame) -> pd.DataFrame:
    random_preds = internal_predictions.loc[internal_predictions["split_type"].eq("random")].copy()
    paired = random_preds.pivot_table(
        index=["repeat", "internal_row_id"],
        columns="model",
        values="abs_error_log10_km",
        aggfunc="first",
    ).reset_index()
    meta_cols = [
        "repeat",
        "internal_row_id",
        "is_largest_leakage_component",
        "train_same_component_n",
        "nearest_train_feature_distance",
    ]
    meta = random_preds.drop_duplicates(["repeat", "internal_row_id"])[meta_cols]
    paired = paired.merge(meta, on=["repeat", "internal_row_id"], how="left")
    repeat_rows = []
    for (repeat, is_largest), group in paired.groupby(["repeat", "is_largest_leakage_component"]):
        repeat_rows.append(
            {
                "repeat": int(repeat),
                "random_test_subset": "giant_component" if is_largest else "non_giant_components",
                "n_test_rows": int(len(group)),
                "median_mae": float(group["median"].mean()),
                "catboost_mae": float(group["catboost"].mean()),
                "skill_score_catboost_vs_median": float(1.0 - group["catboost"].mean() / group["median"].mean()),
                "mean_train_same_component_n": float(group["train_same_component_n"].mean()),
                "mean_nearest_train_feature_distance": float(group["nearest_train_feature_distance"].mean()),
            }
        )
    repeat_df = pd.DataFrame(repeat_rows)
    rows = []
    for subset, group in repeat_df.groupby("random_test_subset"):
        row = {
            "analysis": "random_split_giant_component_diagnostic",
            "random_test_subset": subset,
            "n_repeats": int(group["repeat"].nunique()),
            "mean_n_test_rows": float(group["n_test_rows"].mean()),
            "mean_median_mae": float(group["median_mae"].mean()),
            "mean_catboost_mae": float(group["catboost_mae"].mean()),
            "mean_train_same_component_n": float(group["mean_train_same_component_n"].mean()),
            "mean_nearest_train_feature_distance": float(group["mean_nearest_train_feature_distance"].mean()),
        }
        for metric in ["skill_score_catboost_vs_median", "mean_nearest_train_feature_distance"]:
            low, high = _bootstrap_mean_ci(group[metric].to_numpy())
            row[metric] = float(group[metric].mean())
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out) == 2:
        repeat_wide = repeat_df.pivot(index="repeat", columns="random_test_subset", values="skill_score_catboost_vs_median")
        if {"giant_component", "non_giant_components"}.issubset(repeat_wide.columns):
            diff = (repeat_wide["giant_component"] - repeat_wide["non_giant_components"]).dropna().to_numpy()
            low, high = _bootstrap_mean_ci(diff)
            out["giant_minus_non_giant_skill_difference"] = float(np.mean(diff))
            out["giant_minus_non_giant_skill_difference_ci95_low"] = low
            out["giant_minus_non_giant_skill_difference_ci95_high"] = high
    return out.sort_values("random_test_subset")


def _connection_dose_response_table(
    internal_predictions: pd.DataFrame,
    external_predictions: pd.DataFrame,
    X_train: pd.DataFrame,
    X_external: pd.DataFrame,
    comp_cols: list[str],
) -> pd.DataFrame:
    selected = internal_predictions.loc[
        internal_predictions["split_type"].isin(["random", "eligible_pool_random", "leakage_graph"])
    ].copy()
    paired = selected.pivot_table(
        index=["split_type", "repeat", "internal_row_id"],
        columns="model",
        values="abs_error_log10_km",
        aggfunc="first",
    ).reset_index()
    meta_cols = [
        "split_type",
        "repeat",
        "internal_row_id",
        "leakage_graph_component_id",
        "train_same_component_n",
        "nearest_train_feature_distance",
        "target_log10_km",
    ]
    meta = selected.drop_duplicates(["split_type", "repeat", "internal_row_id"])[meta_cols]
    paired = paired.merge(meta, on=["split_type", "repeat", "internal_row_id"], how="left")
    paired["connection_bin"] = pd.cut(
        paired["train_same_component_n"],
        bins=[-0.1, 0, 5, 25, 100, np.inf],
        labels=["0", "1-5", "6-25", "26-100", ">100"],
    ).astype(str)
    rows = []
    for bin_label, group in paired.groupby("connection_bin", sort=False):
        if group.empty:
            continue
        skill = 1.0 - (group["catboost"].mean() / group["median"].mean())
        boot = _bootstrap_skill_from_abs_errors_clustered(
            group["median"].to_numpy(),
            group["catboost"].to_numpy(),
            group["leakage_graph_component_id"].to_numpy(),
        )
        target_sd = float(group["target_log10_km"].std(ddof=1)) if len(group) > 1 else np.nan
        median_mae = float(group["median"].mean())
        catboost_mae = float(group["catboost"].mean())
        rows.append(
            {
                "domain": "internal",
                "connection_bin": bin_label,
                "n_test_predictions": int(len(group)),
                "n_unique_rows": int(group["internal_row_id"].nunique()),
                "mean_train_same_component_n": float(group["train_same_component_n"].mean()),
                "mean_nearest_train_feature_distance": float(group["nearest_train_feature_distance"].mean()),
                "target_sd": target_sd,
                "median_mae": median_mae,
                "catboost_mae": catboost_mae,
                "median_mae_over_target_sd": float(median_mae / target_sd) if target_sd > 0 else np.nan,
                "catboost_mae_over_target_sd": float(catboost_mae / target_sd) if target_sd > 0 else np.nan,
                "skill_score_catboost_vs_median": float(skill),
                "skill_score_ci95_low": boot[0],
                "skill_score_ci95_high": boot[1],
                "included_splits": "random;eligible_pool_random;leakage_graph",
            }
        )
    ext_boot = _external_normalized_bootstrap_summary(external_predictions)
    external_distance = _nearest_train_feature_distance(
        X=pd.concat([X_train, X_external], ignore_index=True),
        comp_cols=comp_cols,
        train_idx=np.arange(len(X_train), dtype=int),
        test_idx=np.arange(len(X_train), len(X_train) + len(X_external), dtype=int),
    )
    ext_target = (
        external_predictions.loc[external_predictions["model"].eq("median")]
        .drop_duplicates("row_id")["target_log10_km"]
        .astype(float)
    )
    ext_target_sd = float(ext_target.std(ddof=1)) if len(ext_target) > 1 else np.nan
    ext_median_mae = float(
        external_predictions.loc[external_predictions["model"].eq("median"), "abs_error_log10_km"].mean()
    )
    ext_catboost_mae = float(
        external_predictions.loc[external_predictions["model"].eq("catboost"), "abs_error_log10_km"].mean()
    )
    rows.append(
        {
            "domain": "external",
            "connection_bin": "external_disconnected",
            "n_test_predictions": int(external_predictions["row_id"].nunique()),
            "n_unique_rows": int(external_predictions["row_id"].nunique()),
            "mean_train_same_component_n": 0.0,
            "mean_nearest_train_feature_distance": float(np.mean(external_distance)),
            "target_sd": ext_target_sd,
            "median_mae": ext_median_mae,
            "catboost_mae": ext_catboost_mae,
            "median_mae_over_target_sd": float(ext_median_mae / ext_target_sd) if ext_target_sd > 0 else np.nan,
            "catboost_mae_over_target_sd": float(ext_catboost_mae / ext_target_sd) if ext_target_sd > 0 else np.nan,
            "skill_score_catboost_vs_median": float(ext_boot["skill_score_catboost_vs_median"]),
            "skill_score_ci95_low": float(ext_boot["skill_score_ci95_low"]),
            "skill_score_ci95_high": float(ext_boot["skill_score_ci95_high"]),
            "included_splits": "strict_external_shared_schema",
        }
    )
    out = pd.DataFrame(rows)
    order = {"external_disconnected": 0, "0": 1, "1-5": 2, "6-25": 3, "26-100": 4, ">100": 5}
    out["plot_order"] = out["connection_bin"].map(order).fillna(99).astype(int)
    return out.sort_values("plot_order")


def _model_connection_dose_response_table(
    internal_predictions: pd.DataFrame,
    external_predictions: pd.DataFrame,
) -> pd.DataFrame:
    selected = internal_predictions.loc[
        internal_predictions["split_type"].isin(["random", "eligible_pool_random", "leakage_graph"])
    ].copy()
    paired = selected.pivot_table(
        index=["split_type", "repeat", "internal_row_id"],
        columns="model",
        values="abs_error_log10_km",
        aggfunc="first",
    ).reset_index()
    meta = selected.drop_duplicates(["split_type", "repeat", "internal_row_id"])[
        [
            "split_type",
            "repeat",
            "internal_row_id",
            "leakage_graph_component_id",
            "train_same_component_n",
            "nearest_train_feature_distance",
            "target_log10_km",
        ]
    ]
    paired = paired.merge(meta, on=["split_type", "repeat", "internal_row_id"], how="left")
    paired["connection_bin"] = pd.cut(
        paired["train_same_component_n"],
        bins=[-0.1, 0, 5, 25, 100, np.inf],
        labels=["0", "1-5", "6-25", "26-100", ">100"],
    ).astype(str)
    rows = []
    model_names = [model for model in ["elasticnet", "extratrees", "catboost"] if model in paired.columns]
    for model in model_names:
        for bin_label, group in paired.groupby("connection_bin", sort=False):
            if group.empty:
                continue
            skill = 1.0 - (group[model].mean() / group["median"].mean())
            low, high = _bootstrap_skill_from_abs_errors_clustered(
                group["median"].to_numpy(),
                group[model].to_numpy(),
                group["leakage_graph_component_id"].to_numpy(),
            )
            rows.append(
                {
                    "domain": "internal",
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "connection_bin": bin_label,
                    "n_test_predictions": int(len(group)),
                    "n_unique_rows": int(group["internal_row_id"].nunique()),
                    "mean_train_same_component_n": float(group["train_same_component_n"].mean()),
                    "mean_nearest_train_feature_distance": float(group["nearest_train_feature_distance"].mean()),
                    "target_sd": float(group["target_log10_km"].std(ddof=1)) if len(group) > 1 else np.nan,
                    "median_mae": float(group["median"].mean()),
                    "model_mae": float(group[model].mean()),
                    "skill_score_vs_median": float(skill),
                    "skill_score_ci95_low": low,
                    "skill_score_ci95_high": high,
                    "bootstrap_unit": "leakage_graph_component_id",
                    "included_splits": "random;eligible_pool_random;leakage_graph",
                }
            )
    ext_paired = _external_paired_error_frame(external_predictions)
    for model in [model for model in model_names if model in ext_paired.columns]:
        model_abs = ext_paired[model].astype(float).to_numpy()
        median_abs = ext_paired["median"].astype(float).to_numpy()
        boot_idx = _external_cluster_bootstrap_indices(ext_paired)
        boot_skill = np.array([1.0 - (model_abs[idx].mean() / median_abs[idx].mean()) for idx in boot_idx])
        rows.append(
            {
                "domain": "external",
                "model": model,
                "model_label": MODEL_LABELS[model],
                "connection_bin": "external_disconnected",
                "n_test_predictions": int(len(ext_paired)),
                "n_unique_rows": int(len(ext_paired)),
                "mean_train_same_component_n": 0.0,
                "mean_nearest_train_feature_distance": np.nan,
                "target_sd": float(ext_paired["target_log10_km"].std(ddof=1)),
                "median_mae": float(median_abs.mean()),
                "model_mae": float(model_abs.mean()),
                "skill_score_vs_median": float(1.0 - (model_abs.mean() / median_abs.mean())),
                "skill_score_ci95_low": float(np.quantile(boot_skill, 0.025)),
                "skill_score_ci95_high": float(np.quantile(boot_skill, 0.975)),
                "bootstrap_unit": _external_bootstrap_unit(external_predictions),
                "included_splits": "strict_external_shared_schema",
            }
        )
    order = {"external_disconnected": 0, "0": 1, "1-5": 2, "6-25": 3, "26-100": 4, ">100": 5}
    out = pd.DataFrame(rows)
    out["plot_order"] = out["connection_bin"].map(order).fillna(99).astype(int)
    return out.sort_values(["model", "plot_order"]).reset_index(drop=True)


def _bootstrap_skill_from_abs_errors(
    median_abs: np.ndarray, catboost_abs: np.ndarray, n_boot: int = 5000, seed: int = 2026
) -> tuple[float, float]:
    median_abs = np.asarray(median_abs, dtype=float)
    catboost_abs = np.asarray(catboost_abs, dtype=float)
    valid = np.isfinite(median_abs) & np.isfinite(catboost_abs)
    median_abs = median_abs[valid]
    catboost_abs = catboost_abs[valid]
    if len(median_abs) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(np.arange(len(median_abs)), size=(n_boot, len(median_abs)), replace=True)
    med = median_abs[draws].mean(axis=1)
    cat = catboost_abs[draws].mean(axis=1)
    skill = 1.0 - (cat / med)
    return float(np.quantile(skill, 0.025)), float(np.quantile(skill, 0.975))


def _bootstrap_skill_from_abs_errors_clustered(
    median_abs: np.ndarray,
    model_abs: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 5000,
    seed: int = 2026,
) -> tuple[float, float]:
    median_abs = np.asarray(median_abs, dtype=float)
    model_abs = np.asarray(model_abs, dtype=float)
    clusters = pd.Series(clusters).astype(str).to_numpy()
    valid = np.isfinite(median_abs) & np.isfinite(model_abs)
    median_abs = median_abs[valid]
    model_abs = model_abs[valid]
    clusters = clusters[valid]
    unique_clusters = np.unique(clusters)
    if len(unique_clusters) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate(
            [np.flatnonzero(clusters == cluster) for cluster in sampled]
        )
        draws.append(1.0 - model_abs[idx].mean() / median_abs[idx].mean())
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _informative_null_table(
    delta: pd.DataFrame, internal_metrics: pd.DataFrame, external_predictions: pd.DataFrame
) -> pd.DataFrame:
    external = delta.loc[delta["domain"].eq("external")]
    internal = delta.loc[delta["domain"].eq("internal")]
    if external.empty:
        return pd.DataFrame()
    ext = external.iloc[0]
    paired_external = external_predictions.pivot_table(
        index="row_id", columns="model", values="abs_error_log10_km", aggfunc="first"
    )
    external_diff = (paired_external["catboost"] - paired_external["median"]).dropna().to_numpy()
    external_diff_clusters = (
        external_predictions.drop_duplicates("row_id")
        .set_index("row_id")
        .loc[paired_external.dropna(subset=["catboost", "median"]).index]
        .pipe(_external_bootstrap_clusters)
    )
    rows = []
    for _, row in internal.iterrows():
        split_type = _split_type_from_label(str(row["scenario"]))
        paired_internal = internal_metrics.loc[internal_metrics["split_type"].eq(split_type)].pivot_table(
            index="repeat", columns="model", values="mae", aggfunc="first"
        )
        internal_diff = (paired_internal["catboost"] - paired_internal["median"]).dropna().to_numpy()
        internal_skill = (1.0 - (paired_internal["catboost"] / paired_internal["median"])).dropna().to_numpy()
        internal_delta = float(row["delta_mae_catboost_minus_median"])
        diff_mean, diff_low, diff_high = _bootstrap_difference_ci(
            external_diff, internal_diff, external_clusters=external_diff_clusters
        )
        skill_diff_mean, skill_diff_low, skill_diff_high = _bootstrap_external_internal_skill_difference_ci(
            external_predictions, internal_skill
        )
        internal_low = float(row["delta_mae_ci95_low"])
        internal_high = float(row["delta_mae_ci95_high"])
        external_low = float(ext["delta_mae_ci95_low"])
        external_high = float(ext["delta_mae_ci95_high"])
        ci_overlap = bool(max(external_low, internal_low) <= min(external_high, internal_high))
        rows.append(
            {
                "comparison": f"External delta vs {row['scenario']}",
                "external_bootstrap_unit": _external_bootstrap_unit(external_predictions),
                "internal_bootstrap_unit": "repeat",
                "external_delta_mae": float(ext["delta_mae_catboost_minus_median"]),
                "external_ci95_low": external_low,
                "external_ci95_high": external_high,
                "internal_delta_mae": internal_delta,
                "internal_ci95_low": internal_low,
                "internal_ci95_high": internal_high,
                "external_internal_ci_overlap": ci_overlap,
                "difference_of_deltas_external_minus_internal": diff_mean,
                "difference_of_deltas_ci95_low": diff_low,
                "difference_of_deltas_ci95_high": diff_high,
                "difference_ci_excludes_zero": bool(diff_low > 0 or diff_high < 0),
                "difference_of_skill_external_minus_internal": skill_diff_mean,
                "difference_of_skill_ci95_low": skill_diff_low,
                "difference_of_skill_ci95_high": skill_diff_high,
                "difference_of_skill_ci_excludes_zero": bool(skill_diff_low > 0 or skill_diff_high < 0),
                "interpretation": (
                    "External and internal delta estimates are compatible"
                    if ci_overlap and not (diff_low > 0 or diff_high < 0) and not (skill_diff_low > 0 or skill_diff_high < 0)
                    else "External delta differs from this internal split estimate"
                ),
            }
        )
    return pd.DataFrame(rows)


def _split_type_from_label(label: str) -> str:
    inverse = {value: key for key, value in SPLIT_LABELS.items()}
    if label not in inverse:
        raise ValueError(f"Unknown split label: {label}")
    return inverse[label]


def _delta_row(
    domain: str,
    scenario: str,
    diff: np.ndarray,
    n: int,
    clusters: pd.Series | np.ndarray | None = None,
) -> dict:
    if clusters is None:
        low, high = _bootstrap_mean_ci(diff)
        inference_basis = "paired/repeat bootstrap CI; Wilcoxon is reported as a sensitivity check"
    else:
        low, high = _bootstrap_cluster_mean_ci(diff, clusters)
        inference_basis = "paired source-cluster bootstrap CI; Wilcoxon is reported as a sensitivity check"
    p_value = np.nan
    if len(diff) > 1 and np.nanstd(diff) > 0:
        try:
            p_value = float(wilcoxon(diff).pvalue)
        except ValueError:
            p_value = np.nan
    return {
        "domain": domain,
        "scenario": scenario,
        "delta_mae_catboost_minus_median": float(np.nanmean(diff)),
        "delta_mae_ci95_low": low,
        "delta_mae_ci95_high": high,
        "n": int(n),
        "bootstrap_ci_excludes_zero": bool(low > 0 or high < 0) if np.isfinite(low) and np.isfinite(high) else False,
        "wilcoxon_p_sensitivity": p_value,
        "primary_inference_basis": inference_basis,
        "interpretation": "negative values indicate lower CatBoost MAE than median baseline",
    }


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 2026) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _bootstrap_cluster_mean_ci(
    values: np.ndarray, clusters: pd.Series | np.ndarray, n_boot: int = 5000, seed: int = 2026
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    clusters = pd.Series(clusters).astype(str).to_numpy()
    valid = np.isfinite(values) & pd.notna(clusters)
    values = values[valid]
    clusters = clusters[valid]
    if len(values) == 0:
        return np.nan, np.nan
    unique_clusters = np.unique(clusters)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([np.flatnonzero(clusters == cluster) for cluster in sampled])
        draws.append(float(values[idx].mean()))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _external_bootstrap_unit(external_predictions: pd.DataFrame) -> str:
    return "curation_record_id" if "curation_record_id" in external_predictions.columns else "row_id"


def _external_bootstrap_clusters(external_predictions: pd.DataFrame) -> pd.Series:
    cluster_col = _external_bootstrap_unit(external_predictions)
    row_ids = external_predictions["row_id"] if "row_id" in external_predictions.columns else external_predictions.index.to_series()
    return external_predictions[cluster_col].fillna(row_ids).astype(str)


def _external_paired_error_frame(external_predictions: pd.DataFrame) -> pd.DataFrame:
    paired = external_predictions.pivot_table(
        index="row_id", columns="model", values="abs_error_log10_km", aggfunc="first"
    ).reset_index()
    meta_cols = ["row_id", "target_log10_km"]
    if "curation_record_id" in external_predictions.columns:
        meta_cols.append("curation_record_id")
    meta = external_predictions.drop_duplicates("row_id")[meta_cols]
    paired = paired.merge(meta, on="row_id", how="left")
    paired["bootstrap_cluster"] = _external_bootstrap_clusters(paired)
    return paired


def _external_cluster_bootstrap_indices(
    paired: pd.DataFrame, n_boot: int = 5000, seed: int = 2026
) -> list[np.ndarray]:
    clusters = paired["bootstrap_cluster"].astype(str).to_numpy()
    unique_clusters = np.unique(clusters)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        draws.append(np.concatenate([np.flatnonzero(clusters == cluster) for cluster in sampled]))
    return draws


def _bootstrap_difference_ci(
    external_diff: np.ndarray,
    internal_diff: np.ndarray,
    external_clusters: pd.Series | np.ndarray | None = None,
    n_boot: int = 5000,
    seed: int = 2026,
) -> tuple[float, float, float]:
    external_diff = np.asarray(external_diff, dtype=float)
    if external_clusters is not None:
        external_clusters = np.asarray(pd.Series(external_clusters).astype(str))
        valid_external = np.isfinite(external_diff) & pd.notna(external_clusters)
        external_diff = external_diff[valid_external]
        external_clusters = external_clusters[valid_external]
    else:
        external_diff = external_diff[np.isfinite(external_diff)]
    internal_diff = np.asarray(internal_diff, dtype=float)
    internal_diff = internal_diff[np.isfinite(internal_diff)]
    if len(external_diff) == 0 or len(internal_diff) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    if external_clusters is not None:
        unique_clusters = np.unique(external_clusters)
        ext_draws = []
        for _ in range(n_boot):
            sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
            idx = np.concatenate([np.flatnonzero(external_clusters == cluster) for cluster in sampled])
            ext_draws.append(float(external_diff[idx].mean()))
        ext_draws = np.asarray(ext_draws)
    else:
        ext_draws = rng.choice(external_diff, size=(n_boot, len(external_diff)), replace=True).mean(axis=1)
    int_draws = rng.choice(internal_diff, size=(n_boot, len(internal_diff)), replace=True).mean(axis=1)
    diff = ext_draws - int_draws
    return float(diff.mean()), float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))


def _bootstrap_external_internal_skill_difference_ci(
    external_predictions: pd.DataFrame,
    internal_skill: np.ndarray,
    n_boot: int = 5000,
    seed: int = 2026,
) -> tuple[float, float, float]:
    paired = _external_paired_error_frame(external_predictions)
    median_abs = paired["median"].astype(float).to_numpy()
    cat_abs = paired["catboost"].astype(float).to_numpy()
    internal_skill = np.asarray(internal_skill, dtype=float)
    internal_skill = internal_skill[np.isfinite(internal_skill)]
    if len(median_abs) == 0 or len(internal_skill) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    ext_draws = _external_cluster_bootstrap_indices(paired, n_boot=n_boot, seed=seed)
    ext_skill = np.array([1.0 - (cat_abs[idx].mean() / median_abs[idx].mean()) for idx in ext_draws])
    int_skill = rng.choice(internal_skill, size=(n_boot, len(internal_skill)), replace=True).mean(axis=1)
    diff = ext_skill - int_skill
    return float(diff.mean()), float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))


def _skill_score_plot_rows(internal_metrics: pd.DataFrame, external_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    plot_split_order = [
        "random",
        "eligible_pool_random",
        "source_group",
        "formula_group",
        "leakage_graph",
        "composition_cluster",
    ]
    for split_type in plot_split_order:
        paired = internal_metrics.loc[internal_metrics["split_type"].eq(split_type)].pivot_table(
            index="repeat", columns="model", values="mae", aggfunc="first"
        )
        skill = 1.0 - (paired["catboost"].astype(float) / paired["median"].astype(float))
        low, high = _bootstrap_mean_ci(skill.to_numpy())
        rows.append(
            {
                "domain": "internal",
                "scenario": SPLIT_LABELS[split_type],
                "skill_score_catboost_vs_median": float(skill.mean()),
                "skill_score_ci95_low": low,
                "skill_score_ci95_high": high,
            }
        )
    ext = _external_normalized_bootstrap_summary(external_predictions)
    rows.append(
        {
            "domain": "external",
            "scenario": "External",
            "skill_score_catboost_vs_median": float(ext["skill_score_catboost_vs_median"]),
            "skill_score_ci95_low": float(ext["skill_score_ci95_low"]),
            "skill_score_ci95_high": float(ext["skill_score_ci95_high"]),
        }
    )
    return pd.DataFrame(rows)


def _plot_restricted_feature_validation(
    internal_metrics: pd.DataFrame,
    external_metrics: pd.DataFrame,
    external_predictions: pd.DataFrame,
    fig_dir: Path,
) -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=1.05)
    internal_summary = _summarize_internal(internal_metrics)
    external_plot = _add_external_mae_ci(external_metrics, external_predictions)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    ax = axes[0]
    for model in ["median", "catboost"]:
        group = internal_summary.loc[internal_summary["model"].eq(model)].copy()
        group["split_label"] = pd.Categorical(
            group["split_label"],
            categories=[SPLIT_LABELS[s] for s in SPLIT_TYPES],
            ordered=True,
        )
        group = group.sort_values("split_label")
        x = np.arange(len(group))
        yerr = np.vstack([group["mae"] - group["mae_ci95_low"], group["mae_ci95_high"] - group["mae"]])
        ax.errorbar(
            x,
            group["mae"],
            yerr=yerr,
            fmt="o",
            linestyle="none",
            elinewidth=1.4,
            capsize=3,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    ax.set_xticks(np.arange(len(SPLIT_TYPES)))
    internal_tick_labels = [SPLIT_LABELS[s] for s in SPLIT_TYPES]
    internal_tick_labels[-1] = "Composition cluster*"
    ax.set_xticklabels(internal_tick_labels, rotation=25, ha="right")
    ax.set_ylabel(r"MAE (log$_{10}$ units)")
    ax.set_title("A.", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    ax = axes[1]
    external_plot = external_plot.loc[external_plot["model"].isin(["median", "catboost"])].copy()
    external_plot["model"] = pd.Categorical(external_plot["model"], ["median", "catboost"], ordered=True)
    external_plot = external_plot.sort_values("model")
    x = np.arange(len(external_plot))
    for xpos, (_, row) in zip(x, external_plot.iterrows()):
        ax.errorbar(
            xpos,
            row["mae"],
            yerr=np.array(
                [[row["mae"] - row["mae_ci95_low"]], [row["mae_ci95_high"] - row["mae"]]]
            ),
            fmt="o",
            linewidth=1.4,
            capsize=3,
            color=MODEL_COLORS[str(row["model"])],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(external_plot["model"].map(MODEL_LABELS))
    ax.set_xlim(-0.45, 1.45)
    ax.set_title("B.", loc="left", fontweight="bold")
    ax.set_ylabel("")
    for xpos, (_, row) in zip(x, external_plot.iterrows()):
        label = f"MAE = {row['mae']:.2f}"
        if row["model"] == "catboost":
            label += f"\nSD ratio = {row['pred_obs_sd_ratio']:.2f}"
        y_offset = 0.09 if row["model"] == "catboost" else 0.06
        ax.text(
            xpos,
            row["mae"] + y_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=7.2,
        )
    ax.text(0.5, 0.04, f"n = {int(external_plot['n_external'].iloc[0])}", transform=ax.transAxes, ha="center")
    ymax = max(
        1.25,
        float(internal_summary["mae_ci95_high"].max()) * 1.15,
        float(external_plot["mae_ci95_high"].max()) * 1.18,
    )
    axes[0].set_ylim(0, ymax)
    axes[1].set_ylim(0, ymax * 1.05)

    ax = axes[2]
    skill_rows = _skill_score_plot_rows(internal_metrics, external_predictions)
    skill_x = np.arange(len(skill_rows))
    for xpos, (_, row) in zip(skill_x, skill_rows.iterrows()):
        color = "#333333" if row["domain"] == "external" else MODEL_COLORS["catboost"]
        markerface = "white" if row["scenario"] == "Composition cluster" else color
        ax.errorbar(
            xpos,
            row["skill_score_catboost_vs_median"],
            yerr=np.array(
                [
                    [row["skill_score_catboost_vs_median"] - row["skill_score_ci95_low"]],
                    [row["skill_score_ci95_high"] - row["skill_score_catboost_vs_median"]],
                ]
            ),
            fmt="o",
            linewidth=1.4,
            capsize=3,
            color=color,
            markerfacecolor=markerface,
            markeredgecolor=color,
            zorder=3,
        )
    ext_skill = float(skill_rows.loc[skill_rows["domain"].eq("external"), "skill_score_catboost_vs_median"].iloc[0])
    ax.axhline(ext_skill, color="#333333", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.5)
    skill_tick_labels = skill_rows["scenario"].astype(str).tolist()
    skill_tick_labels = [
        "Composition cluster*" if label == "Composition cluster" else label
        for label in skill_tick_labels
    ]
    ax.set_xticks(skill_x)
    ax.set_xticklabels(skill_tick_labels, rotation=25, ha="right")
    ax.set_ylabel("Skill score relative to median baseline")
    ax.set_title("C.", loc="left", fontweight="bold")
    ax.set_ylim(-0.10, 0.28)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=MODEL_COLORS["catboost"], linestyle="", label="Internal"),
            Line2D([0], [0], marker="o", color="#333333", linestyle="", label="External"),
            Line2D(
                [0],
                [0],
                marker="o",
                color=MODEL_COLORS["catboost"],
                markerfacecolor="white",
                linestyle="",
                label="* Reduced test fraction",
            ),
        ],
        frameon=False,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    sns.despine(fig=fig)
    fig.subplots_adjust(wspace=0.34)
    _save_figure(fig, fig_dir / "fig10_restricted_feature_validation")


def _plot_connection_dose_response(table26_path: Path, fig_dir: Path) -> None:
    dose = pd.read_csv(table26_path).sort_values("plot_order")
    sns.set_theme(context="paper", style="ticks", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(6.2, 3.85))
    internal = dose.loc[dose["domain"].eq("internal")]
    bin_order = ["0", "1-5", "6-25", "26-100", ">100"]
    bin_positions = {label: index for index, label in enumerate(bin_order)}
    for _, row in internal.iterrows():
        bin_label = str(row["connection_bin"])
        xpos = bin_positions[bin_label]
        skill = float(row["skill_score_catboost_vs_median"])
        if np.isfinite(row["skill_score_ci95_low"]) and np.isfinite(row["skill_score_ci95_high"]):
            ax.errorbar(
                xpos,
                skill,
                yerr=np.array(
                    [
                        [skill - float(row["skill_score_ci95_low"])],
                        [float(row["skill_score_ci95_high"]) - skill],
                    ]
                ),
                fmt="o",
                color=MODEL_COLORS["catboost"],
                capsize=3,
                elinewidth=1.4,
                markersize=5.2,
            )
        else:
            ax.plot(
                xpos,
                skill,
                marker="D",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=MODEL_COLORS["catboost"],
                markeredgewidth=1.4,
                markersize=5.6,
            )
            ax.annotate(
                "1 component\ninterval not estimable",
                xy=(xpos, skill),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color="0.28",
            )
    ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.5)
    tick_labels = []
    for bin_label in bin_order:
        row = internal.loc[internal["connection_bin"].astype(str).eq(bin_label)]
        if row.empty:
            tick_labels.append("26–100\n(no observations)")
        else:
            nn_distance = float(row["mean_nearest_train_feature_distance"].iloc[0])
            tick_labels.append(f"{bin_label}\nNN={nn_distance:.2f}")
    ax.set_xticks(np.arange(len(bin_order)))
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_xlabel("Same-component training records (internal predictions)")
    ax.set_ylabel("Skill score relative to median baseline")
    finite_high = internal["skill_score_ci95_high"].dropna().astype(float)
    ax.set_ylim(-0.10, max(0.43, float(finite_high.max()) * 1.08))
    sns.despine(fig=fig)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.97, bottom=0.30)
    _save_figure(fig, fig_dir / "fig11_graph_connection_dose_response")


def _update_publication_manifests(table_dir: Path, fig_dir: Path) -> None:
    figure_manifest = table_dir / "publication_figure_manifest.csv"
    if figure_manifest.exists():
        figures = pd.read_csv(figure_manifest)
        figures = figures.loc[
            ~figures["figure_id"].isin(["fig10_restricted_feature_validation", "fig11_graph_connection_dose_response"])
        ]
        figures = pd.concat(
            [
                figures,
                pd.DataFrame(
                    [
                        {
                            "figure_id": "fig10_restricted_feature_validation",
                            "description": (
                                "Fair shared-schema validation using formula-derived composition "
                                "descriptors and substrate identity only, with raw MAE and "
                                "baseline-normalized skill-score diagnostics."
                            ),
                            "png": str((fig_dir / "fig10_restricted_feature_validation.png").resolve()),
                            "pdf": str((fig_dir / "fig10_restricted_feature_validation.pdf").resolve()),
                            "svg": str((fig_dir / "fig10_restricted_feature_validation.svg").resolve()),
                        }
                    ],
                ),
                pd.DataFrame(
                    [
                        {
                            "figure_id": "fig11_graph_connection_dose_response",
                            "description": (
                                "Dose-response relationship between graph connection to training data "
                                "and baseline-normalized CatBoost skill."
                            ),
                            "png": str((fig_dir / "fig11_graph_connection_dose_response.png").resolve()),
                            "pdf": str((fig_dir / "fig11_graph_connection_dose_response.pdf").resolve()),
                            "svg": str((fig_dir / "fig11_graph_connection_dose_response.svg").resolve()),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        figures.to_csv(figure_manifest, index=False)
    table_rows = [{"table_file": path.name, "path": str(path.resolve())} for path in sorted(table_dir.glob("table*.csv"))]
    pd.DataFrame(table_rows).to_csv(table_dir / "publication_table_manifest.csv", index=False)


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def _substrate_from_reaction(value: object) -> str:
    reaction = str(value).strip().upper().replace(" ", "")
    if reaction.startswith("TMB+H2O2"):
        return "TMB"
    if reaction.startswith("H2O2+TMB"):
        return "H2O2"
    return "OTHER"


if __name__ == "__main__":
    main()
