#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import prepare_modeling_data, read_table
from src.diagnostics import source_formula_protocol_component_ids
from src.features import build_design_only_feature_frame
from src.reporting import ensure_output_dirs
from src.splits import make_split
from src.validation import (
    raise_if_failed,
    validate_feature_columns,
    validate_modeling_frame,
    validate_split,
    validation_results_to_frame,
)


SPLIT_TYPES = ["random", "source_group", "formula_group", "composition_cluster", "leakage_graph"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-composition-test-n", type=int, default=20)
    args = parser.parse_args()

    if not args.data.strip():
        parser.error(
            '--data was empty. If using --data "$DATA", set DATA first '
            "or pass a file path such as --data data/nanozymes.csv."
        )

    dirs = ensure_output_dirs()
    raw = read_table(args.data)
    df, _ = prepare_modeling_data(raw)
    df["leakage_graph_component_id"] = source_formula_protocol_component_ids(df)
    df = df.drop(columns=["vmax_value", "vmax_unit"], errors="ignore")
    X, meta = build_design_only_feature_frame(df)
    results = []
    results.extend(validate_modeling_frame(df))
    results.extend(validate_feature_columns(X.columns))

    for split_type in SPLIT_TYPES:
        train_idx, test_idx, _ = make_split(
            df,
            split_type,
            args.seed,
            test_size=args.test_size,
            composition=X[meta["composition_columns"]],
            min_test_n=args.min_composition_test_n,
        )
        results.extend(validate_split(df, train_idx, test_idx, split_type))

    out = dirs["tables"] / "preflight_validation.csv"
    validation_results_to_frame(results).to_csv(out, index=False)
    raise_if_failed(results)
    print(f"Preflight validation passed: {out}")


if __name__ == "__main__":
    main()
