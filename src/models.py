from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import MISSING_TOKEN


MAIN_MODEL_NAME = "catboost"


@dataclass(frozen=True)
class FeatureSpec:
    numeric_columns: list[str]
    categorical_columns: list[str]


class CatBoostFrameRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        random_seed: int = 0,
        iterations: int = 500,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        loss_function: str = "RMSE",
    ):
        self.random_seed = random_seed
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.loss_function = loss_function

    def fit(self, X, y):
        if importlib.util.find_spec("catboost") is None:
            raise ImportError(
                "CatBoost is mandatory for the main model suite. Install catboost==1.2.5."
            )
        from catboost import CatBoostRegressor

        Xp = self._prepare_fit_frame(X)
        self.feature_names_in_ = list(Xp.columns)
        self.numeric_medians_ = Xp[self.numeric_columns_].median(numeric_only=True)
        Xp[self.numeric_columns_] = Xp[self.numeric_columns_].fillna(self.numeric_medians_)
        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function=self.loss_function,
            random_seed=self.random_seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
        self.model_.fit(Xp, y, cat_features=self.categorical_columns_)
        return self

    def predict(self, X):
        Xp = self._prepare_predict_frame(X)
        return self.model_.predict(Xp)

    def _prepare_fit_frame(self, X):
        Xp = pd.DataFrame(X).copy()
        self.categorical_columns_ = [
            c for c in Xp.columns if Xp[c].dtype == object or str(Xp[c].dtype).startswith("string")
        ]
        self.numeric_columns_ = [c for c in Xp.columns if c not in self.categorical_columns_]
        for col in self.categorical_columns_:
            Xp[col] = Xp[col].astype("string").fillna(MISSING_TOKEN).astype(str)
        for col in self.numeric_columns_:
            Xp[col] = pd.to_numeric(Xp[col], errors="coerce")
        return Xp

    def _prepare_predict_frame(self, X):
        Xp = pd.DataFrame(X).copy()
        Xp = Xp.reindex(columns=self.feature_names_in_)
        for col in self.categorical_columns_:
            Xp[col] = Xp[col].astype("string").fillna(MISSING_TOKEN).astype(str)
        for col in self.numeric_columns_:
            Xp[col] = pd.to_numeric(Xp[col], errors="coerce")
        Xp[self.numeric_columns_] = Xp[self.numeric_columns_].fillna(self.numeric_medians_)
        return Xp


def require_catboost() -> None:
    if importlib.util.find_spec("catboost") is None:
        raise ImportError(
            "CatBoostRegressor is mandatory and pinned in requirements.txt. "
            "Install dependencies before running the benchmark."
        )


def sklearn_preprocessor(feature_spec: FeatureSpec, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_TOKEN)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, feature_spec.numeric_columns),
            ("cat", cat_pipe, feature_spec.categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model_suite(feature_spec: FeatureSpec, seed: int) -> dict[str, BaseEstimator]:
    return {
        "median": DummyRegressor(strategy="median"),
        "elasticnet": Pipeline(
            [
                ("preprocess", sklearn_preprocessor(feature_spec, scale_numeric=True)),
                (
                    "model",
                    ElasticNet(
                        alpha=0.01,
                        l1_ratio=0.5,
                        random_state=seed,
                        max_iter=10000,
                        selection="cyclic",
                    ),
                ),
            ]
        ),
        "extratrees": Pipeline(
            [
                ("preprocess", sklearn_preprocessor(feature_spec, scale_numeric=False)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        criterion="squared_error",
                        random_state=seed,
                        min_samples_leaf=2,
                        max_features=1.0,
                        bootstrap=False,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "catboost": CatBoostFrameRegressor(random_seed=seed),
    }


def fit_predict(model: BaseEstimator, X_train, y_train, X_test) -> np.ndarray:
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    return fitted.predict(X_test)


def unseen_category_rates(
    X_train: pd.DataFrame, X_test: pd.DataFrame, categorical_columns: list[str]
) -> pd.DataFrame:
    rows = []
    for col in categorical_columns:
        train_vocab = set(X_train[col].astype("string").fillna(MISSING_TOKEN).astype(str))
        test_vals = X_test[col].astype("string").fillna(MISSING_TOKEN).astype(str)
        unseen = ~test_vals.isin(train_vocab)
        rows.append(
            {
                "feature_name": col,
                "number_unseen_test_categories": int(test_vals[unseen].nunique()),
                "fraction_test_rows_affected": float(unseen.mean()) if len(unseen) else 0.0,
                "handling_rule": "OneHotEncoder(handle_unknown='ignore') or CatBoost string token",
            }
        )
    return pd.DataFrame(rows)
