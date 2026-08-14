#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import regression_metrics
from src.reporting import ensure_output_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge manual audit decisions into the model-ready external validation table, "
            "write the strict external subset, and recompute strict external metrics."
        )
    )
    parser.add_argument(
        "--model-ready",
        default="outputs/tables/external_validation/external_validation_model_ready.csv",
    )
    parser.add_argument(
        "--manual-audit",
        default="outputs/tables/external_validation/external_validation_priority20_manual_audit.csv",
    )
    parser.add_argument(
        "--predictions",
        default="outputs/tables/external_validation/external_validation_predictions.csv",
    )
    args = parser.parse_args()

    dirs = ensure_output_dirs()
    external_dir = dirs["tables"] / "external_validation"
    external_dir.mkdir(parents=True, exist_ok=True)

    model_ready = pd.read_csv(args.model_ready)
    manual = pd.read_csv(args.manual_audit)
    predictions = pd.read_csv(args.predictions)

    audited = merge_manual_audit(model_ready, manual)

    with_audit_path = external_dir / "external_validation_model_ready_with_audit.csv"
    strict_path = external_dir / "external_validation_strict_model_ready.csv"
    strict_metrics_path = external_dir / "external_validation_strict_metrics.csv"
    strict_predictions_path = external_dir / "external_validation_strict_predictions.csv"
    strict_summary_path = external_dir / "external_validation_strict_summary.csv"

    audited.to_csv(with_audit_path, index=False)
    strict = audited.loc[audited["strict_external_recommended"].astype(bool)].copy()
    strict.to_csv(strict_path, index=False)

    strict_predictions = filter_predictions(predictions, strict)
    strict_predictions.to_csv(strict_predictions_path, index=False)

    strict_metrics = metric_table(strict_predictions, strict)
    strict_metrics.to_csv(strict_metrics_path, index=False)

    strict_summary = summary_table(audited, strict)
    strict_summary.to_csv(strict_summary_path, index=False)

    print(f"Wrote {with_audit_path}")
    print(f"Wrote {strict_path}")
    print(f"Wrote {strict_predictions_path}")
    print(f"Wrote {strict_metrics_path}")
    print(f"Wrote {strict_summary_path}")
    print("\nStrict summary:")
    print(strict_summary.to_string(index=False))
    print("\nStrict metrics:")
    print(strict_metrics.to_string(index=False))


def merge_manual_audit(model_ready: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    audit_cols = [
        "row_id",
        "audit_decision",
        "strict_include_recommended",
        "manual_flags",
        "strict_exclusion_reason",
        "audit_rationale",
        "recommended_manuscript_handling",
    ]
    available = [c for c in audit_cols if c in manual.columns]
    manual = manual[available].rename(
        columns={
            "row_id": "external_row_id",
            "audit_decision": "manual_audit_decision",
            "strict_include_recommended": "strict_external_recommended",
        }
    )
    out = model_ready.merge(manual, on="external_row_id", how="left")
    out["manual_audit_decision"] = out["manual_audit_decision"].fillna("not_priority_audited")
    out["strict_external_recommended"] = (
        out["strict_external_recommended"].where(out["strict_external_recommended"].notna(), True)
    ).astype(bool)
    out["manual_flags"] = out["manual_flags"].fillna("")
    out["strict_exclusion_reason"] = out["strict_exclusion_reason"].fillna("")
    out["audit_rationale"] = out["audit_rationale"].fillna("")
    out["recommended_manuscript_handling"] = out["recommended_manuscript_handling"].fillna("")
    out["priority20_audited"] = out["manual_audit_decision"].ne("not_priority_audited")
    out["retain_with_flag"] = out["manual_audit_decision"].eq("retain_with_flag")
    out["exclude_from_strict"] = ~out["strict_external_recommended"]

    ordered_front = [
        "external_row_id",
        "curation_record_id",
        "strict_external_recommended",
        "manual_audit_decision",
        "priority20_audited",
        "retain_with_flag",
        "exclude_from_strict",
        "manual_flags",
        "strict_exclusion_reason",
        "audit_rationale",
        "recommended_manuscript_handling",
    ]
    ordered_front = [c for c in ordered_front if c in out.columns]
    remaining = [c for c in out.columns if c not in ordered_front]
    return out[ordered_front + remaining]


def filter_predictions(predictions: pd.DataFrame, strict: pd.DataFrame) -> pd.DataFrame:
    keep_ids = set(strict["external_row_id"].astype(int))
    out = predictions.loc[predictions["row_id"].astype(int).isin(keep_ids)].copy()
    meta_cols = [
        "external_row_id",
        "curation_record_id",
        "strict_external_recommended",
        "manual_audit_decision",
        "manual_flags",
        "priority20_audited",
        "retain_with_flag",
        "exclude_from_strict",
        "substrate_for_km_norm",
        "km_millimolar",
        "target_log10_km_millimolar",
    ]
    meta = strict[[c for c in meta_cols if c in strict.columns]].rename(
        columns={"external_row_id": "row_id"}
    )
    out = out.merge(meta, on="row_id", how="left")
    out["pred_km_millimolar"] = 10 ** out["pred_log10_km"] * 1e3
    out["fold_error"] = np.maximum(
        out["pred_km_millimolar"] / out["km_millimolar"],
        out["km_millimolar"] / out["pred_km_millimolar"],
    )
    return out


def metric_table(strict_predictions: pd.DataFrame, strict: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in strict_predictions.groupby("model"):
        rows.append(metric_row("strict_external", model, "all", group, strict))
        for substrate, sub_group in group.groupby("substrate_for_km_norm"):
            rows.append(metric_row("strict_external", model, substrate, sub_group, strict))
        unflagged = group.loc[~group["retain_with_flag"].astype(bool)]
        if len(unflagged):
            rows.append(metric_row("strict_external_unflagged_only", model, "all", unflagged, strict))
            for substrate, sub_group in unflagged.groupby("substrate_for_km_norm"):
                rows.append(
                    metric_row(
                        "strict_external_unflagged_only",
                        model,
                        substrate,
                        sub_group,
                        strict,
                    )
                )
    return pd.DataFrame(rows).sort_values(["scenario", "model", "substrate"])


def metric_row(
    scenario: str,
    model: str,
    substrate: str,
    group: pd.DataFrame,
    strict: pd.DataFrame,
) -> dict:
    metrics = regression_metrics(group["target_log10_km"], group["pred_log10_km"])
    return {
        "scenario": scenario,
        "model": model,
        "substrate": substrate,
        "n_rows": int(len(group)),
        "n_unique_records": int(group["curation_record_id"].nunique())
        if "curation_record_id" in group.columns
        else np.nan,
        "n_priority20_audited_rows": int(group["priority20_audited"].sum())
        if "priority20_audited" in group.columns
        else 0,
        "n_retain_with_flag_rows": int(group["retain_with_flag"].sum())
        if "retain_with_flag" in group.columns
        else 0,
        **metrics,
    }


def summary_table(audited: pd.DataFrame, strict: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "summary_item": "all_model_ready_rows",
            "value": int(len(audited)),
            "note": "Rows before applying strict_external_recommended",
        },
        {
            "summary_item": "strict_external_rows",
            "value": int(len(strict)),
            "note": "Rows retained in strict external benchmark",
        },
        {
            "summary_item": "excluded_from_strict_rows",
            "value": int((~audited["strict_external_recommended"]).sum()),
            "note": "Rows removed by manual strict audit",
        },
        {
            "summary_item": "priority20_audited_rows",
            "value": int(audited["priority20_audited"].sum()),
            "note": "Rows manually audited from CatBoost priority list",
        },
        {
            "summary_item": "retain_with_flag_rows",
            "value": int(audited["retain_with_flag"].sum()),
            "note": "Rows retained but flagged for manuscript interpretation",
        },
        {
            "summary_item": "strict_tmb_rows",
            "value": int(strict["substrate_for_km_norm"].eq("TMB").sum()),
            "note": "TMB rows retained in strict external benchmark",
        },
        {
            "summary_item": "strict_h2o2_rows",
            "value": int(strict["substrate_for_km_norm"].eq("H2O2").sum()),
            "note": "H2O2 rows retained in strict external benchmark",
        },
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
