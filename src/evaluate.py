from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if _is_constant(y_true) or _is_constant(y_pred):
        spearman = np.nan
    else:
        spear = spearmanr(y_true, y_pred, nan_policy="omit")
        spearman = float(spear.statistic) if np.isfinite(spear.statistic) else np.nan
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": spearman,
    }


def _is_constant(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return len(finite) < 2 or np.all(finite == finite[0])


def summarize_with_ci(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric in metric_cols:
            vals = group[metric].dropna().astype(float)
            row[f"{metric}_mean"] = vals.mean() if len(vals) else np.nan
            row[f"{metric}_ci95_low"] = vals.quantile(0.025) if len(vals) > 1 else np.nan
            row[f"{metric}_ci95_high"] = vals.quantile(0.975) if len(vals) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_with_cluster_bootstrap(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
    cluster_col: str = "repeat",
    n_boot: int = 2000,
    seed: int = 2026,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        clusters = group[cluster_col].dropna().unique() if cluster_col in group.columns else np.array([])
        row["n_clusters"] = int(len(clusters))
        for metric in metric_cols:
            vals = group[[cluster_col, metric]].dropna() if cluster_col in group.columns else pd.DataFrame()
            metric_values = group[metric].dropna().astype(float)
            row[f"{metric}_mean"] = metric_values.mean() if len(metric_values) else np.nan
            if len(clusters) <= 1 or vals.empty:
                row[f"{metric}_cluster_boot_ci95_low"] = np.nan
                row[f"{metric}_cluster_boot_ci95_high"] = np.nan
                continue
            boot_means = []
            for _ in range(n_boot):
                sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
                sampled = pd.concat(
                    [vals[vals[cluster_col] == cluster] for cluster in sampled_clusters],
                    ignore_index=True,
                )
                boot_means.append(float(sampled[metric].mean()))
            row[f"{metric}_cluster_boot_ci95_low"] = float(np.quantile(boot_means, 0.025))
            row[f"{metric}_cluster_boot_ci95_high"] = float(np.quantile(boot_means, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)
