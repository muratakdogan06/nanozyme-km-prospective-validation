#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import audit_data, prepare_modeling_data, read_table, write_audit_outputs
from src.features import elemental_coverage_table
from src.reporting import ensure_output_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to DiZyme/ChemX-like table")
    parser.add_argument("--output", default="outputs/tables/audit", help="Audit output directory")
    args = parser.parse_args()

    if not args.data.strip():
        parser.error(
            '--data was empty. If using --data "$DATA", set DATA first '
            "or pass a file path such as --data data/nanozymes.csv."
        )

    raw = read_table(args.data)
    modeling, filter_log = prepare_modeling_data(raw)
    audit = audit_data(raw, modeling, filter_log)
    ensure_output_dirs()
    out = Path(args.output)
    write_audit_outputs(audit, out)
    elemental_coverage_table(modeling["standardized_formula"]).to_csv(
        "outputs/tables/supp_table_s10_elemental_property_coverage.csv", index=False
    )
    print(f"Audit complete: {out}")


if __name__ == "__main__":
    main()
