#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import prepare_modeling_data, read_table


def main() -> None:
    df, _ = prepare_modeling_data(read_table("data/nanozymes.csv"))
    work = df.set_index("row_id")
    rows: list[dict] = []
    for repeat in range(20):
        manifest = pd.read_csv(f"outputs/split_manifests/random_repeat_{repeat}.csv")
        train_ids = manifest.loc[manifest["split"].eq("train"), "row_id"].astype(int)
        test_ids = manifest.loc[manifest["split"].eq("test"), "row_id"].astype(int)
        train = work.loc[train_ids]
        test = work.loc[test_ids]
        eligible = test.loc[test["varied_substrate"].isin(["TMB", "H2O2"])]

        for definition in ["source_formula_pair", "protocol_matched_pair"]:
            supported = 0
            for _, row in eligible.iterrows():
                counterpart = "H2O2" if row["varied_substrate"] == "TMB" else "TMB"
                candidates = train.loc[
                    train["source_group_id"].eq(row["source_group_id"])
                    & train["standardized_formula"].eq(row["standardized_formula"])
                    & train["canonical_reaction_system"].eq(
                        row["canonical_reaction_system"]
                    )
                    & train["varied_substrate"].eq(counterpart)
                ]
                if definition == "protocol_matched_pair":
                    for column in ["ph", "temperature", "length", "width", "depth"]:
                        value = pd.to_numeric(
                            pd.Series([row.get(column)]), errors="coerce"
                        ).iloc[0]
                        if pd.notna(value):
                            candidate_values = pd.to_numeric(
                                candidates[column], errors="coerce"
                            )
                            candidates = candidates[np.isclose(candidate_values, value)]
                    surface = str(row.get("surface", "")).strip().lower()
                    if surface and surface != "nan":
                        candidates = candidates.loc[
                            candidates["surface"]
                            .astype(str)
                            .str.strip()
                            .str.lower()
                            .eq(surface)
                        ]
                supported += int(len(candidates) > 0)
            rows.append(
                {
                    "repeat": repeat,
                    "pair_definition": definition,
                    "n_test_rows": int(len(test)),
                    "n_tmb_or_h2o2_test_rows": int(len(eligible)),
                    "n_test_rows_with_training_counterpart": int(supported),
                    "fraction_with_training_counterpart": supported / len(eligible),
                }
            )

    raw = pd.DataFrame(rows)
    summary = raw.groupby("pair_definition", as_index=False).agg(
        n_repeats=("repeat", "nunique"),
        mean_test_rows=("n_test_rows", "mean"),
        mean_tmb_or_h2o2_test_rows=("n_tmb_or_h2o2_test_rows", "mean"),
        sd_tmb_or_h2o2_test_rows=("n_tmb_or_h2o2_test_rows", "std"),
        min_tmb_or_h2o2_test_rows=("n_tmb_or_h2o2_test_rows", "min"),
        max_tmb_or_h2o2_test_rows=("n_tmb_or_h2o2_test_rows", "max"),
        mean_supported_test_rows=("n_test_rows_with_training_counterpart", "mean"),
        sd_supported_test_rows=("n_test_rows_with_training_counterpart", "std"),
        mean_fraction_with_training_counterpart=(
            "fraction_with_training_counterpart",
            "mean",
        ),
        sd_fraction_with_training_counterpart=(
            "fraction_with_training_counterpart",
            "std",
        ),
    )
    summary["mean_eligible_fraction_of_test"] = (
        summary["mean_tmb_or_h2o2_test_rows"] / summary["mean_test_rows"]
    )
    summary["eligibility_definition"] = (
        "test row whose varied substrate is TMB or H2O2; full-corpus counterpart "
        "existence is not required"
    )
    out = Path("outputs/tables/reviewer3_controls")
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "paired_substrate_denominator_by_repeat.csv", index=False)
    summary.to_csv(out / "paired_substrate_denominator_summary.csv", index=False)


if __name__ == "__main__":
    main()
