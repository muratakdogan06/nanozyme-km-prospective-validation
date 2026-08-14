#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_reviewer3_controls import bias_corrected_cramers_v
from src.data import prepare_modeling_data, read_table


def main() -> None:
    df, _ = prepare_modeling_data(read_table("data/nanozymes.csv"))
    table = pd.crosstab(df["canonical_reaction_system"], df["source_group_id"])
    columns = table.to_numpy(float).T
    observed = bias_corrected_cramers_v(table.to_numpy(float))
    rows: list[dict] = []
    for seed in range(2026, 2036):
        rng = np.random.default_rng(seed)
        boot = np.array(
            [
                bias_corrected_cramers_v(
                    columns[rng.integers(0, len(columns), size=len(columns))].T
                )
                for _ in range(5000)
            ]
        )
        rows.append(
            {
                "categorical_variable": "canonical_reaction_system",
                "n_rows": int(table.to_numpy().sum()),
                "n_levels": int(table.shape[0]),
                "n_source_groups": int(table.shape[1]),
                "observed_bias_corrected_cramers_v": observed,
                "seed": seed,
                "n_boot": 5000,
                "bootstrap_median": float(np.median(boot)),
                "bootstrap_95_ci_low": float(np.quantile(boot, 0.025)),
                "bootstrap_95_ci_high": float(np.quantile(boot, 0.975)),
                "bootstrap_mean": float(np.mean(boot)),
                "bootstrap_unit": "source group",
            }
        )
    out = Path("outputs/tables/reviewer3_controls")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        out / "reaction_source_association_seed_sensitivity.csv", index=False
    )


if __name__ == "__main__":
    main()
