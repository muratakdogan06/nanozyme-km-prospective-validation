#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import prepare_modeling_data, read_table
from scripts.run_reviewer3_controls import restricted_nearest_neighbor_sensitivity


def main() -> None:
    df, _ = prepare_modeling_data(read_table("data/nanozymes.csv"))
    raw, summary = restricted_nearest_neighbor_sensitivity(df)
    out = Path("outputs/tables/reviewer3_controls")
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "restricted_knn_metrics.csv", index=False)
    summary.to_csv(out / "restricted_knn_summary.csv", index=False)


if __name__ == "__main__":
    main()
