from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


@dataclass
class IntervalResult:
    lower: np.ndarray
    upper: np.ndarray
    y_pred: np.ndarray
    qhat: float


def conformal_rank(n_scores: int, alpha: float) -> int:
    """Return the 1-indexed split-conformal order statistic rank."""
    return int(np.ceil((n_scores + 1) * (1 - alpha)))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return np.nan
    rank = conformal_rank(len(scores), alpha)
    if rank > len(scores):
        return np.nan
    return float(np.sort(scores)[rank - 1])


def absolute_residual_intervals(model, X_train, y_train, X_cal, y_cal, X_test, alpha=0.1):
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    cal_pred = fitted.predict(X_cal)
    qhat = conformal_quantile(np.abs(np.asarray(y_cal) - cal_pred), alpha)
    y_pred = fitted.predict(X_test)
    return IntervalResult(y_pred - qhat, y_pred + qhat, y_pred, qhat)


class _ScaleFramer:
    def fit(self, X):
        frame = self._coerce(X)
        self.columns_ = list(frame.columns)
        self.medians_ = frame.median(numeric_only=True)
        return self

    def transform(self, X) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        frame = frame.reindex(columns=self.columns_)
        frame = self._coerce(frame)
        return frame.fillna(self.medians_).fillna(0.0)

    @staticmethod
    def _coerce(X) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        for col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame


def residual_normalized_intervals(
    model,
    X_train,
    y_train,
    X_cal,
    y_cal,
    X_test,
    alpha=0.1,
    n_scale_folds=5,
    random_state=17,
):
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    y_train_array = np.asarray(y_train, dtype=float)
    train_residuals = _out_of_fold_residuals(
        model, X_train, y_train_array, n_splits=n_scale_folds, random_state=random_state
    )
    cal_pred = fitted.predict(X_cal)
    residuals = np.abs(np.asarray(y_cal) - cal_pred)
    scale_model = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=1,
    )
    # The scale model must be fixed before conformal calibration. Fitting it on
    # calibration residuals makes the normalized scores in-sample and can
    # severely under-cover even under exchangeable random splits.
    scale_framer = _ScaleFramer().fit(X_train)
    scale_model.fit(scale_framer.transform(X_train), np.maximum(train_residuals, 1e-9))
    cal_scale = np.maximum(scale_model.predict(scale_framer.transform(X_cal)), 1e-6)
    qhat = conformal_quantile(residuals / cal_scale, alpha)
    y_pred = fitted.predict(X_test)
    test_scale = np.maximum(scale_model.predict(scale_framer.transform(X_test)), 1e-6)
    return IntervalResult(y_pred - qhat * test_scale, y_pred + qhat * test_scale, y_pred, qhat)


def _out_of_fold_residuals(model, X, y: np.ndarray, n_splits=5, random_state=17) -> np.ndarray:
    n = len(y)
    if n == 0:
        return np.array([], dtype=float)
    if n_splits < 2 or n < 2:
        fitted = clone(model)
        fitted.fit(X, y)
        return np.abs(y - fitted.predict(X))
    n_splits = min(n_splits, n)
    residuals = np.empty(n, dtype=float)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_pos, valid_pos in splitter.split(np.arange(n)):
        fitted = clone(model)
        fitted.fit(_take_rows(X, train_pos), y[train_pos])
        residuals[valid_pos] = np.abs(y[valid_pos] - fitted.predict(_take_rows(X, valid_pos)))
    return residuals


def _take_rows(X, positions: np.ndarray):
    if hasattr(X, "iloc"):
        return X.iloc[positions]
    return np.asarray(X)[positions]


def interval_metrics(y_true, y_pred, lower, upper, alpha=0.1) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    finite_intervals = np.isfinite(lower) & np.isfinite(upper)
    if not finite_intervals.all():
        return {
            "mae": mae,
            "empirical_coverage": np.nan,
            "target_coverage": float(1 - alpha),
            "coverage_gap": np.nan,
            "mean_interval_width": np.nan,
            "median_interval_width": np.nan,
            "interval_score": np.nan,
        }
    covered = (y_true >= lower) & (y_true <= upper)
    width = upper - lower
    below = y_true < lower
    above = y_true > upper
    interval_score = width + (2 / alpha) * (lower - y_true) * below + (2 / alpha) * (
        y_true - upper
    ) * above
    return {
        "mae": mae,
        "empirical_coverage": float(np.mean(covered)),
        "target_coverage": float(1 - alpha),
        "coverage_gap": float(np.mean(covered) - (1 - alpha)),
        "mean_interval_width": float(np.mean(width)),
        "median_interval_width": float(np.median(width)),
        "interval_score": float(np.mean(interval_score)),
    }


def few_shot_recalibration(
    y_true,
    y_pred,
    lower,
    upper,
    shots=(0, 5, 10, 20),
    repeats=100,
    min_remaining_n=50,
    alpha=0.1,
    seed=0,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    half_width = (np.asarray(upper) - np.asarray(lower)) / 2
    residual = np.abs(y_true - y_pred)
    rows = []
    rng = np.random.default_rng(seed)
    n = len(y_true)
    for shot in shots:
        if shot == 0:
            metrics = interval_metrics(y_true, y_pred, y_pred - half_width, y_pred + half_width, alpha)
            metrics.update({"shots": 0, "few_shot_repeat": 0, "underpowered": False})
            rows.append(metrics)
            continue
        if n - shot < min_remaining_n or conformal_rank(shot, alpha) > shot:
            rows.append(
                {
                    "shots": shot,
                    "few_shot_repeat": 0,
                    "underpowered": True,
                    "mae": np.nan,
                    "empirical_coverage": np.nan,
                    "coverage_gap": np.nan,
                    "mean_interval_width": np.nan,
                    "median_interval_width": np.nan,
                    "interval_score": np.nan,
                }
            )
            continue
        for rep in range(repeats):
            recal_idx = rng.choice(n, size=shot, replace=False)
            eval_mask = np.ones(n, dtype=bool)
            eval_mask[recal_idx] = False
            q_local = conformal_quantile(residual[recal_idx] / np.maximum(half_width[recal_idx], 1e-6), alpha)
            new_half_width = half_width[eval_mask] * q_local
            metrics = interval_metrics(
                y_true[eval_mask],
                y_pred[eval_mask],
                y_pred[eval_mask] - new_half_width,
                y_pred[eval_mask] + new_half_width,
                alpha,
            )
            metrics.update({"shots": shot, "few_shot_repeat": rep, "underpowered": False})
            rows.append(metrics)
    return pd.DataFrame(rows)
