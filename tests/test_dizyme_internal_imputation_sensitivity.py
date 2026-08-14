from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_dizyme_internal_imputation_sensitivity import (
    SUBSTRATE_DESCRIPTOR_7,
    TWO_FOLD_LOG10,
    cross_imputed_arms,
    descriptor_integrity_diagnostics,
    donor_mask_for_query_groups,
    read_numeric_training_table,
    recover_publication_groups,
    summarize_predictions,
)


def synthetic_dizyme_frame(n: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(17)
    X = pd.DataFrame(
        {
            "huge_scale": np.linspace(1.0, 1000.0, n),
            "small_scale": rng.normal(size=n),
            "ph": 3.0 + (np.arange(n) % 6),
            "temp": 20.0 + (np.arange(n) % 5) * 5.0,
            "lgvolume": -1.0 + (np.arange(n) % 7) * 0.25,
        }
    )
    X.loc[np.arange(0, n, 7), "small_scale"] = np.nan
    metadata = pd.DataFrame(
        {
            "formula": [f"F{i % 8}" for i in range(n)],
            "id": np.arange(n),
            "publication_id": [f"doi:10.0000/{i // 2}" for i in range(n)],
        }
    )
    return X, metadata


@pytest.mark.parametrize("rule", ["raw_knn", "standardized_knn", "training_median"])
def test_cross_imputed_arms_change_only_protocol_fields_between_arms(rule):
    X, metadata = synthetic_dizyme_frame()
    protocol = ["ph", "temp", "lgvolume"]
    eligible, observed, imputed, folds = cross_imputed_arms(
        X,
        metadata,
        protocol,
        rule,
        n_splits=5,
        seed=2026,
        split_unit="row",
        n_neighbors=3,
    )

    protocol_columns = [X.columns.get_loc(column) for column in protocol]
    non_protocol_columns = [
        position for position, column in enumerate(X.columns) if column not in protocol
    ]
    assert len(eligible) == len(X)
    assert set(folds) == set(range(5))
    assert np.isfinite(observed).all()
    assert np.isfinite(imputed).all()
    assert observed[:, protocol_columns] == pytest.approx(
        X.iloc[eligible][protocol].to_numpy()
    )
    assert observed[:, non_protocol_columns] == pytest.approx(
        imputed[:, non_protocol_columns]
    )


def test_standardized_knn_protocol_imputation_is_invariant_to_rescaling():
    X, metadata = synthetic_dizyme_frame()
    rescaled = X.copy()
    rescaled["huge_scale"] *= 1_000_000.0
    protocol = ["ph", "temp", "lgvolume"]

    _, _, imputed_original, _ = cross_imputed_arms(
        X,
        metadata,
        protocol,
        "standardized_knn",
        n_splits=5,
        seed=2026,
        split_unit="row",
        n_neighbors=3,
    )
    _, _, imputed_rescaled, _ = cross_imputed_arms(
        rescaled,
        metadata,
        protocol,
        "standardized_knn",
        n_splits=5,
        seed=2026,
        split_unit="row",
        n_neighbors=3,
    )
    protocol_columns = [X.columns.get_loc(column) for column in protocol]
    assert imputed_original[:, protocol_columns] == pytest.approx(
        imputed_rescaled[:, protocol_columns]
    )


def test_protocol_rule_comparison_holds_observed_arm_fixed():
    X, metadata = synthetic_dizyme_frame()
    protocol = ["ph", "temp", "lgvolume"]
    observed_arms = []
    for rule in ("raw_knn", "standardized_knn", "training_median"):
        _, observed, _, _ = cross_imputed_arms(
            X,
            metadata,
            protocol,
            rule,
            n_splits=5,
            seed=2026,
            split_unit="row",
            n_neighbors=3,
        )
        observed_arms.append(observed)

    assert observed_arms[0] == pytest.approx(observed_arms[1])
    assert observed_arms[0] == pytest.approx(observed_arms[2])


def test_mask_set_comparison_holds_common_reference_arm_fixed():
    X, metadata = synthetic_dizyme_frame()
    shared_eligible = np.flatnonzero(X[["small_scale"]].notna().all(axis=1).to_numpy())
    _, observed_protocol, _, _ = cross_imputed_arms(
        X,
        metadata,
        ["ph", "temp"],
        "raw_knn",
        n_splits=5,
        split_unit="row",
        n_neighbors=3,
        eligible_override=shared_eligible,
    )
    _, observed_comparator, _, _ = cross_imputed_arms(
        X,
        metadata,
        ["huge_scale", "small_scale"],
        "raw_knn",
        n_splits=5,
        split_unit="row",
        n_neighbors=3,
        eligible_override=shared_eligible,
    )
    assert observed_protocol == pytest.approx(observed_comparator)


def test_publication_donor_mask_excludes_all_sibling_rows():
    _, metadata = synthetic_dizyme_frame(12)
    query_rows = np.array([0])
    mask = donor_mask_for_query_groups(metadata, query_rows, "publication")
    assert not mask[0]
    assert not mask[1]
    assert mask[2:].all()


def test_publication_mapping_resolves_duplicate_ids_by_formula_and_km(tmp_path):
    peroxidase = pd.DataFrame(
        {
            "id": [1, 2, 2, 3],
            "formula": ["Fe3O4", "Pt", "Au", "N4-TiO2"],
            "Km": [0.1, 0.2, 0.3, "no"],
        }
    )
    source = pd.DataFrame(
        {
            "#": [1, 2, 2, 3],
            "formula": ["Fe3O4", "Pt", "Au", "N4-TiO2"],
            "Km, mM": [0.1, 0.2, 0.3, "no"],
            "link": [
                "https://doi.org/10.0000/a",
                "https://doi.org/10.0000/b",
                "https://doi.org/10.0000/c",
                "https://doi.org/10.0000/d",
            ],
        }
    )
    path = tmp_path / "nanozymes.csv"
    source.to_csv(path, index=False)
    mapping, qc = recover_publication_groups(peroxidase, path)
    assert mapping["doi_normalized"].tolist() == [
        "10.0000/a",
        "10.0000/b",
        "10.0000/c",
        "10.0000/d",
    ]
    assert qc["source_duplicated_ids"] == 1
    assert qc["cross_publication_duplicate_ids"] == [2]
    assert qc["same_publication_duplicate_ids"] == []
    duplicated = mapping.loc[mapping["processed_id"].eq(2)]
    assert set(duplicated["duplicate_id_classification"]) == {
        "cross_publication_collision_resolved"
    }
    assert duplicated["duplicate_id_note"].str.contains("multiple DOI groups").all()
    assert qc["qc_passed"]


def test_publication_mapping_names_same_publication_duplicate(tmp_path):
    peroxidase = pd.DataFrame(
        {"id": [9], "formula": ["Pt"], "Km": [0.2]}
    )
    source = pd.DataFrame(
        {
            "#": [9, 9],
            "formula": ["Pt", "Pt"],
            "Km, mM": [0.2, 0.3],
            "link": [
                "https://doi.org/10.0000/shared",
                "https://doi.org/10.0000/shared",
            ],
        }
    )
    path = tmp_path / "nanozymes.csv"
    source.to_csv(path, index=False)
    mapping, qc = recover_publication_groups(peroxidase, path)
    assert qc["same_publication_duplicate_ids"] == [9]
    assert qc["cross_publication_duplicate_ids"] == []
    assert mapping.loc[0, "duplicate_id_classification"] == (
        "same_publication_duplicate"
    )


def test_descriptor_integrity_diagnostics_stratifies_sentinel_and_xlogp():
    X = pd.DataFrame(
        {
            "MolWt": [0.0, 34.01, 106.121],
            "TPSA": [0.0, 40.0, 52.99],
            "XLogP": [0.0, 2.0, 53.0],
            "PEOE_VSA7": [0.0, 1.0, 1.0],
            "PEOE_VSA9": [0.0, 1.0, 1.0],
            "Kappa2": [0.0, 1.0, 1.0],
            "BalabanJ": [0.0, 1.0, 1.0],
        }
    )
    non_sentinel, sentinel, flag, audit, counts = descriptor_integrity_diagnostics(
        X, np.arange(len(X))
    )
    assert list(X.columns) == SUBSTRATE_DESCRIPTOR_7
    assert sentinel.tolist() == [0]
    assert non_sentinel.tolist() == [1, 2]
    assert flag.tolist() == [True, False, False]
    assert counts == {
        "primary_7_field_rows": 3,
        "descriptor_sentinel_all_zero_rows": 1,
        "descriptor_non_sentinel_rows": 2,
        "xlogp_above_10_rows": 1,
        "xlogp_above_10_within_1_of_tpsa_rows": 1,
    }
    assert set(audit["parser_action"]) == {"recorded_not_corrected"}


def test_known_placeholder_is_coerced_but_unknown_token_fails(tmp_path):
    path = tmp_path / "training.csv"
    pd.DataFrame({"a": [1.0, "#????"], "b": [2.0, 3.0]}).to_csv(path, index=False)
    _, numeric, tokens = read_numeric_training_table(path, ["a", "b"])
    assert pd.isna(numeric.loc[1, "a"])
    assert tokens == {"a": ["#????"]}

    pd.DataFrame({"a": [1.0, "unexpected"], "b": [2.0, 3.0]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="Unexpected non-numeric"):
        read_numeric_training_table(path, ["a", "b"])


def test_prediction_summary_uses_prespecified_twofold_threshold():
    predictions = pd.DataFrame(
        {
            "mask_set": ["primary"] * 3,
            "imputation_rule": ["raw_knn"] * 3,
            "split_unit": ["row"] * 3,
            "publication_id": ["doi:a", "doi:a", "doi:b"],
            "absolute_prediction_shift_log10": [0.0, TWO_FOLD_LOG10, 0.6],
            "signed_prediction_shift_log10": [0.0, TWO_FOLD_LOG10, -0.6],
            "reference_prediction_sd_log10": [0.5] * 3,
            "absolute_shift_over_reference_prediction_sd": [
                0.0,
                TWO_FOLD_LOG10 / 0.5,
                1.2,
            ],
        }
    )
    summary = summarize_predictions(predictions, bootstrap_draws=100).iloc[0]
    assert summary["n_rows"] == 3
    assert summary["median_absolute_shift_log10"] == pytest.approx(TWO_FOLD_LOG10)
    assert summary["fraction_at_least_twofold"] == pytest.approx(2 / 3)
    assert summary["median_absolute_shift_over_reference_sd"] == pytest.approx(
        TWO_FOLD_LOG10 / 0.5
    )
    assert summary["n_publication_clusters"] == 2
