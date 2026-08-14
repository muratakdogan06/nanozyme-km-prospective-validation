from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor

import src.conformal as conformal
from src.data import convert_km_to_molar, prepare_modeling_data, read_table
from src.diagnostics import (
    external_ood_severity_rows,
    generalization_stress_index,
    ood_severity_row,
    protocol_granularity_sensitivity,
    severity_error_association,
    split_repeat_variability,
    source_formula_protocol_component_ids,
)
from src.evaluate import regression_metrics
from src.features import (
    build_design_only_feature_frame,
    build_feature_frame,
    parse_formula,
    standardize_formula,
)
from src.splits import LeakageGraphSplit, make_split
from src.validation import validate_feature_columns, validate_modeling_frame, validate_split
from scripts.run_editor_major_revision_analyses import (
    component_aware_calibration_split,
    mse_bias_variance_metrics,
    training_only_substrate_median_predictions,
)


def synthetic_raw(n: int = 48) -> pd.DataFrame:
    formulas = ["Fe3O4", "CeO2", "Au", "Pt", "MnO2", "CuO", "Co3O4", "ZnO"]
    rows = []
    for i in range(n):
        rows.append(
            {
                "formula": formulas[i % len(formulas)],
                "activity": "peroxidase-like",
                "syngony": ["cubic", "tetragonal", "unknown"][i % 3],
                "length": 10 + i % 5,
                "width": 8 + i % 4,
                "depth": 5 + i % 3,
                "surface": ["bare", "PEG", "citrate"][i % 3],
                "km_value": 0.1 + 0.01 * i,
                "km_unit": "mM",
                "vmax_value": 1.0 + i,
                "vmax_unit": "uM/s",
                "reaction_type": ["TMB/H2O2", "OPD/H2O2"][i % 2],
                "c_min": 0.01,
                "c_max": 1.0,
                "c_const": 0.1,
                "c_const_unit": "mM",
                "ccat_value": 0.001 * i,
                "ccat_unit": "mg/mL",
                "ph": 7.0 + (i % 4) * 0.1,
                "temperature": 25,
                "doi": f"10.0000/test{i // 4}",
            }
        )
    return pd.DataFrame(rows)


def test_formula_parser_and_standardization():
    assert parse_formula("Fe3O4") == {"Fe": 3.0, "O": 4.0}
    assert parse_formula("CeO2") == {"Ce": 1.0, "O": 2.0}
    assert standardize_formula("O2Ce") == "CeO2"


def test_km_unit_conversion_variants():
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    units = pd.Series(["mM", "umol L-1", "nM", "M"])
    converted = convert_km_to_molar(values, units)
    assert converted.tolist() == pytest.approx([1e-3, 2e-6, 3e-9, 4.0])


def test_regression_metrics_handle_constant_predictions_without_warnings():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [2.0, 2.0, 2.0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics = regression_metrics(y_true, y_pred)
    assert not caught
    assert metrics["rmse"] == pytest.approx((2 / 3) ** 0.5)
    assert pd.isna(metrics["spearman"])


def test_residual_normalized_conformal_scale_model_excludes_calibration(monkeypatch):
    fit_shapes = []

    class RecordingScaleModel:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, X, y):
            fit_shapes.append((len(X), len(y)))
            self.scale_ = float(np.mean(y)) if len(y) else 1.0
            return self

        def predict(self, X):
            return np.full(len(X), max(self.scale_, 1e-6))

    monkeypatch.setattr(conformal, "ExtraTreesRegressor", RecordingScaleModel)
    X_train = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y_train = pd.Series(np.arange(12, dtype=float))
    X_cal = pd.DataFrame({"x": np.arange(100, 105, dtype=float)})
    y_cal = pd.Series(np.arange(100, 105, dtype=float))
    X_test = pd.DataFrame({"x": np.arange(200, 203, dtype=float)})

    conformal.residual_normalized_intervals(
        DummyRegressor(strategy="mean"),
        X_train,
        y_train,
        X_cal,
        y_cal,
        X_test,
        alpha=0.1,
    )

    assert fit_shapes == [(len(X_train), len(y_train))]
    assert fit_shapes[0][0] != len(X_cal)


def test_conformal_quantile_uses_one_indexed_order_statistic():
    assert conformal.conformal_rank(5, 0.1) == 6
    assert np.isnan(conformal.conformal_quantile(np.arange(1, 6), 0.1))
    assert conformal.conformal_rank(10, 0.1) == 10
    assert conformal.conformal_quantile(np.arange(1, 11), 0.1) == 10
    assert conformal.conformal_rank(20, 0.1) == 19
    assert conformal.conformal_quantile(np.arange(1, 21), 0.1) == 19


def test_scale_framer_uses_training_medians_for_calibration_and_test():
    framer = conformal._ScaleFramer().fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    transformed = framer.transform(pd.DataFrame({"x": [np.nan, 100.0]}))
    assert transformed["x"].tolist() == pytest.approx([2.0, 100.0])


def test_scale_residual_targets_are_out_of_fold_not_in_sample():
    X = pd.DataFrame({"x": np.arange(20, dtype=float)})
    y = np.arange(20, dtype=float)
    residuals = conformal._out_of_fold_residuals(
        DecisionTreeRegressor(random_state=0), X, y, n_splits=5, random_state=17
    )
    in_sample_model = DecisionTreeRegressor(random_state=0).fit(X, y)
    in_sample_residuals = np.abs(y - in_sample_model.predict(X))

    assert np.allclose(in_sample_residuals, 0.0)
    assert residuals.mean() > 0.0


def test_read_table_rejects_empty_or_directory_paths(tmp_path):
    with pytest.raises(ValueError, match="Data path is empty"):
        read_table("")
    with pytest.raises(ValueError, match="directory"):
        read_table(tmp_path)


def test_embedded_unit_headers_are_normalized():
    raw = pd.DataFrame(
        {
            "formula": ["Fe3O4"],
            "activity": ["peroxidase"],
            "Km, mM": [2.5],
            "Vmax, mM/s": [1.2],
            "ReactionType": ["TMB + H2O2"],
            "temp, °C": [37],
            "link": ["https://doi.org/10.0000/example"],
        }
    )
    df, _ = prepare_modeling_data(raw)
    assert df.loc[0, "km_unit"] == "mM"
    assert df.loc[0, "km_common_molar"] == pytest.approx(2.5e-3)
    assert df.loc[0, "vmax_unit"] == "mM/s"
    assert df.loc[0, "reaction_type"] == "TMB + H2O2"
    assert df.loc[0, "temperature"] == 37


def test_reaction_order_is_preserved_as_varied_substrate_semantics():
    raw = pd.DataFrame(
        {
            "formula": ["Fe3O4", "Fe3O4"],
            "activity": ["peroxidase", "peroxidase"],
            "Km, mM": [0.2, 4.0],
            "ReactionType": ["TMB + H2O2", "H2O2 + TMB"],
            "link": ["https://example.org/a", "https://example.org/a"],
        }
    )
    df, _ = prepare_modeling_data(raw)
    assert df["varied_substrate"].tolist() == ["TMB", "H2O2"]
    assert df["canonical_reaction_system"].nunique() == 1
    assert df["canonical_reaction_system"].iloc[0] == "H2O2 + TMB"


def test_substrate_median_baseline_is_training_only_with_global_fallback():
    predictions, fallback, global_median, medians = (
        training_only_substrate_median_predictions(
            pd.Series([1.0, 3.0, 10.0]),
            pd.Series(["TMB", "TMB", "H2O2"]),
            pd.Series(["TMB", "H2O2", "ABTS"]),
        )
    )
    assert predictions.tolist() == pytest.approx([2.0, 10.0, 3.0])
    assert fallback.tolist() == [False, False, True]
    assert global_median == pytest.approx(3.0)
    assert medians == pytest.approx({"TMB": 2.0, "H2O2": 10.0})


def test_mse_bias_variance_decomposition_is_exact_on_population_scale():
    metrics = mse_bias_variance_metrics(np.array([-1.0, 1.0, 3.0]))
    assert metrics["mse"] == pytest.approx(
        metrics["bias_squared"] + metrics["residual_variance"]
    )
    assert metrics["decomposition_roundoff"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["signed_bias"] == pytest.approx(1.0)


def test_component_aware_calibration_split_preserves_group_integrity():
    train_idx = np.arange(20)
    components = pd.Series(
        ["giant"] * 8 + ["a"] * 3 + ["b"] * 3 + ["c"] * 2 + ["d"] * 2 + ["e"] * 2
    )
    proper, calibration, diagnostics = component_aware_calibration_split(
        train_idx,
        components,
        seed=2026,
        calibration_fraction=0.25,
        candidates=500,
    )
    assert set(components.loc[proper]).isdisjoint(set(components.loc[calibration]))
    assert "giant" in set(components.loc[proper])
    assert diagnostics["proper_calibration_component_overlap"] == 0
    assert diagnostics["component_integrity_pass"]
    assert len(proper) + len(calibration) == len(train_idx)


def test_design_only_feature_frame_excludes_post_characterization_fields():
    df = pd.DataFrame(
        {
            "standardized_formula": ["Fe3O4"],
            "ph": [4.0],
            "reaction_type": ["TMB + H2O2"],
            "km_unit": ["mM"],
            "vmax_value": [1.2],
            "vmax_unit": ["mM/s"],
        }
    )
    X, meta = build_design_only_feature_frame(df)
    assert "ph" in X.columns
    assert "reaction_type" in X.columns
    assert "km_unit" not in X.columns
    assert "vmax_value" not in X.columns
    assert "vmax_unit" not in X.columns
    assert meta["excluded_columns"] == ["km_unit", "vmax_unit", "vmax_value"]


def test_ood_severity_and_generalization_stress_are_computed():
    df, _ = prepare_modeling_data(synthetic_raw())
    X, meta = build_feature_frame(df)
    train_idx = df.index[df["source_group_id"] != df.loc[0, "source_group_id"]]
    test_idx = df.index[df["source_group_id"] == df.loc[0, "source_group_id"]]
    severity = ood_severity_row(
        df,
        X,
        meta["composition_columns"],
        train_idx,
        test_idx,
        "source_group",
        repeat=0,
        seed=2026,
    )
    assert severity["unseen_source_fraction"] == pytest.approx(1.0)
    assert 0.0 <= severity["ood_severity_index"] <= 1.0
    assert 0.0 <= severity["source_formula_free_severity_index"] <= 1.0
    assert 0.0 <= severity["graph_free_composition_severity_index"] <= 1.0

    metrics = pd.DataFrame(
        [
            {"repeat": 0, "seed": 2026, "split_type": "random", "model": "catboost", "mae": 0.5, "r2": 0.7},
            {
                "repeat": 0,
                "seed": 2026,
                "split_type": "source_group",
                "model": "catboost",
                "mae": 0.8,
                "r2": 0.4,
            },
        ]
    )
    stress = generalization_stress_index(metrics)
    assert stress.loc[0, "generalization_stress_index"] == pytest.approx(0.3)
    assert stress.loc[0, "relative_generalization_stress_index"] == pytest.approx(0.6)

    variability = split_repeat_variability(
        df,
        [
            {"repeat": 0, "seed": 2026, "tests": {"source_group": test_idx}},
            {"repeat": 1, "seed": 2027, "tests": {"source_group": train_idx[: len(test_idx)]}},
        ],
    )
    assert variability.loc[0, "split_type"] == "source_group"
    assert "mean_pairwise_test_jaccard" in variability.columns


def test_data_preparation_and_feature_validation():
    df, _ = prepare_modeling_data(synthetic_raw())
    X, _ = build_feature_frame(df)
    results = validate_modeling_frame(df) + validate_feature_columns(X.columns)
    assert all(r.passed for r in results)
    assert "doi" not in X.columns
    assert "target_log10_km" not in X.columns


def test_group_splits_do_not_leak_groups():
    df, _ = prepare_modeling_data(synthetic_raw())
    df["leakage_graph_component_id"] = source_formula_protocol_component_ids(df)
    X, meta = build_feature_frame(df)
    for split_type in ["source_group", "formula_group", "composition_cluster", "leakage_graph"]:
        train_idx, test_idx, _ = make_split(
            df,
            split_type,
            seed=2026,
            composition=X[meta["composition_columns"]],
            min_test_n=3,
        )
        results = validate_split(df, train_idx, test_idx, split_type)
        assert all(r.passed for r in results)


def test_protocol_granularity_sensitivity_reports_graph_effects():
    df, _ = prepare_modeling_data(synthetic_raw())
    sensitivity = protocol_granularity_sensitivity(df)
    assert set(sensitivity["protocol_tokenization"]) == {
        "three_significant_figures",
        "coarse_ph0.5_temp5_two_sigfig",
        "very_coarse_ph1_temp10_one_sigfig",
    }
    assert "n_source_formula_components" in sensitivity.columns
    assert "n_tripartite_components" in sensitivity.columns
    assert "n_protocol_nodes_with_multiple_rows" in sensitivity.columns
    assert "n_cross_component_protocol_nodes" in sensitivity.columns
    assert "n_redundant_shared_protocol_nodes" in sensitivity.columns
    assert "rows_in_protocol_merged_source_formula_components" in sensitivity.columns


def test_external_severity_scores_can_be_linked_to_errors():
    train_df, _ = prepare_modeling_data(synthetic_raw())
    external_raw = synthetic_raw(12).copy()
    external_raw["doi"] = [f"10.9999/ext{i // 3}" for i in range(len(external_raw))]
    external_raw["formula"] = ["Ag"] * len(external_raw)
    external_df, _ = prepare_modeling_data(external_raw)
    X_train, meta = build_feature_frame(train_df)
    X_external, _ = build_feature_frame(external_df)
    X_external = X_external.reindex(columns=X_train.columns)

    severity = external_ood_severity_rows(
        train_df, external_df, X_train, X_external, meta["composition_columns"]
    )
    assert len(severity) == len(external_df)
    assert "ood_severity_index" in severity.columns
    assert severity["unseen_source"].eq(1.0).all()

    predictions = severity[["row_id", "ood_severity_index"]].copy()
    predictions["model"] = "catboost"
    predictions["abs_error_log10_km"] = predictions["ood_severity_index"]
    association = severity_error_association(predictions)
    assert association.loc[0, "severity_score"] == "ood_severity_index"
    assert association.loc[0, "n_external_rows"] == len(external_df)


def test_generic_leakage_graph_split_is_sklearn_compatible():
    frame = pd.DataFrame(
        {
            "source": ["s1", "s1", "s2", "s3", "s4", "s5"],
            "formula": ["a", "b", "b", "c", "d", "e"],
            "protocol": ["p1", "p2", "p2", "p3", "p4", "p5"],
        }
    )
    splitter = LeakageGraphSplit(
        node_cols=["source", "formula", "protocol"],
        n_splits=2,
        random_state=2026,
        min_test_n=1,
    )
    splits = list(splitter.split(frame))
    assert splitter.get_n_splits() == 2
    assert len(splits) == 2
    for train_idx, test_idx in splits:
        assert set(train_idx).isdisjoint(set(test_idx))
        assert len(train_idx) > 0
        assert len(test_idx) > 0


def test_leakage_graph_split_rejects_single_component_degeneracy():
    df, _ = prepare_modeling_data(synthetic_raw())
    df["leakage_graph_component_id"] = 0
    with pytest.raises(ValueError, match="at least two connected components"):
        make_split(df, "leakage_graph", seed=2026, min_test_n=3)

    results = validate_split(df, df.index[:0], df.index, "leakage_graph")
    nonempty = [r for r in results if r.check_name == "leakage_graph_nonempty_train_test"][0]
    assert not nonempty.passed
