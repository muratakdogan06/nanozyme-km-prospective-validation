#!/usr/bin/env python3
"""Build the publication-ready five-protocol substrate-stratified SI table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SPLIT_LABELS = {
    "random": "Random",
    "composition_cluster": "Composition-cluster holdout",
    "formula_group": "Formula holdout",
    "source_group": "Source holdout",
    "leakage_graph": "Component holdout",
}
SPLIT_ORDER = list(SPLIT_LABELS)
SUBSTRATE_ORDER = ["TMB", "H2O2"]


def build_table(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "feature_schema",
        "split_type",
        "varied_substrate",
        "n_repeats",
        "mean_n_test",
        "min_n_test",
        "max_n_test",
        "mae_mean",
        "mae_sd",
        "r2_mean",
        "r2_sd",
        "spearman_mean",
        "spearman_sd",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Substrate summary is missing columns: {missing}")

    selected = summary.loc[
        summary["feature_schema"].eq("design_only_primary")
        & summary["split_type"].isin(SPLIT_ORDER)
        & summary["varied_substrate"].isin(SUBSTRATE_ORDER)
    ].copy()
    selected["split_type"] = pd.Categorical(
        selected["split_type"], categories=SPLIT_ORDER, ordered=True
    )
    selected["varied_substrate"] = pd.Categorical(
        selected["varied_substrate"], categories=SUBSTRATE_ORDER, ordered=True
    )
    selected = selected.sort_values(["split_type", "varied_substrate"]).reset_index(drop=True)

    expected = pd.MultiIndex.from_product(
        [SPLIT_ORDER, SUBSTRATE_ORDER], names=["split_type", "varied_substrate"]
    )
    observed = pd.MultiIndex.from_frame(
        selected[["split_type", "varied_substrate"]].astype(str)
    )
    absent = expected.difference(observed)
    duplicates = observed[observed.duplicated()].tolist()
    if len(absent) or duplicates:
        raise ValueError(
            f"Expected one row per protocol/substrate; absent={list(absent)}, "
            f"duplicates={duplicates}"
        )
    if not selected["n_repeats"].eq(20).all():
        raise ValueError("Expected 20 outer-split repeats for every TMB/H2O2 row")

    numeric = selected[
        [
            "split_type",
            "varied_substrate",
            "n_repeats",
            "mean_n_test",
            "min_n_test",
            "max_n_test",
            "mae_mean",
            "mae_sd",
            "r2_mean",
            "r2_sd",
            "spearman_mean",
            "spearman_sd",
        ]
    ].copy()
    numeric.insert(0, "protocol", numeric["split_type"].astype(str).map(SPLIT_LABELS))
    numeric["substrate"] = numeric["varied_substrate"].astype(str).replace({"H2O2": "H₂O₂"})
    numeric = numeric.drop(columns=["varied_substrate"])

    display = pd.DataFrame(
        {
            "Protocol": numeric["protocol"],
            "Substrate": numeric["substrate"],
            "Repeats": numeric["n_repeats"].astype(int),
            "Test records, mean (range)": numeric.apply(
                lambda row: (
                    f"{row['mean_n_test']:.2f} "
                    f"({int(row['min_n_test'])}–{int(row['max_n_test'])})"
                ),
                axis=1,
            ),
            "MAE, mean (SD)": numeric.apply(
                lambda row: f"{row['mae_mean']:.3f} ({row['mae_sd']:.3f})", axis=1
            ),
            "R², mean (SD)": numeric.apply(
                lambda row: f"{row['r2_mean']:.3f} ({row['r2_sd']:.3f})", axis=1
            ),
            "Spearman ρ, mean (SD)": numeric.apply(
                lambda row: f"{row['spearman_mean']:.3f} ({row['spearman_sd']:.3f})", axis=1
            ),
        }
    )
    return numeric, display


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=(
            "outputs/tables/reviewer2_sensitivity/"
            "design_only_substrate_stratified_performance.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/si_substrate_stratified_20260810",
    )
    args = parser.parse_args()

    summary = pd.read_csv(args.input)
    numeric, display = build_table(summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric.to_csv(output_dir / "substrate_stratified_si_numeric.csv", index=False)
    display.to_csv(output_dir / "substrate_stratified_si_table.csv", index=False)

    component = numeric.loc[numeric["split_type"].eq("leakage_graph")].set_index("substrate")
    observed = {
        "TMB": round(float(component.loc["TMB", "mae_mean"]), 3),
        "H₂O₂": round(float(component.loc["H₂O₂", "mae_mean"]), 3),
    }
    expected = {"TMB": 0.538, "H₂O₂": 0.865}
    if observed != expected:
        raise RuntimeError(f"Component-holdout MAE regression mismatch: {observed}")

    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
