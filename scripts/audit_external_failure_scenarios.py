#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import regression_metrics
from src.reporting import ensure_output_dirs


SUSPICIOUS_NOTE_PATTERN = re.compile(
    r"verify|unit mismatch|unit should|unit unclear|unclear|not preserve|"
    r"extreme|outlier|flag|saturation|ambiguous",
    flags=re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit external validation failure cases and sensitivity scenarios."
    )
    parser.add_argument(
        "--predictions",
        default="outputs/tables/external_validation/external_validation_predictions.csv",
    )
    parser.add_argument(
        "--model-ready",
        default="outputs/tables/external_validation/external_validation_model_ready.csv",
    )
    args = parser.parse_args()

    dirs = ensure_output_dirs()
    external_dir = dirs["tables"] / "external_validation"
    external_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    model_ready = pd.read_csv(args.model_ready)
    meta = model_ready[
        [
            "external_row_id",
            "curation_record_id",
            "surface",
            "substrate_for_km_norm",
            "km_millimolar",
            "target_log10_km_millimolar",
            "vmax_value",
            "vmax_unit",
            "vmax_molar_per_s",
            "doi_normalized",
            "doi_missing",
            "external_source_cluster_id",
            "source_cluster_row_count",
            "source_location",
        ]
    ].rename(columns={"external_row_id": "row_id"})

    merged = predictions.merge(meta, on="row_id", how="left")
    merged = add_audit_flags(merged)

    audit_path = external_dir / "external_validation_failure_audit.csv"
    scenario_path = external_dir / "external_validation_scenario_metrics.csv"
    catboost_priority_path = external_dir / "external_validation_catboost_priority_audit.csv"

    audit_cols = [
        "row_id",
        "curation_record_id",
        "model",
        "formula",
        "standardized_formula",
        "surface",
        "substrate_for_km_norm",
        "km_millimolar",
        "target_log10_km",
        "pred_log10_km",
        "abs_error_log10_km",
        "pred_km_millimolar",
        "fold_error",
        "doi",
        "doi_normalized",
        "doi_missing",
        "title",
        "journal",
        "year",
        "ood_severity_index",
        "unseen_source",
        "unseen_formula",
        "composition_nearest_train_distance",
        "source_cluster_row_count",
        "km_extreme_hard",
        "km_tail_95pct",
        "note_suspicious",
        "vmax_unconverted",
        "manual_audit_recommended",
        "audit_reasons",
        "curation_source",
        "source_location",
        "curation_note",
    ]
    merged[audit_cols].to_csv(audit_path, index=False)

    scenario_metrics = scenario_metric_table(merged)
    scenario_metrics.to_csv(scenario_path, index=False)

    cat = merged[merged["model"].eq("catboost")].copy()
    cat["priority_score"] = (
        cat["abs_error_log10_km"]
        + 0.5 * cat["km_extreme_hard"].astype(float)
        + 0.25 * cat["note_suspicious"].astype(float)
    )
    cat.sort_values(["priority_score", "abs_error_log10_km"], ascending=False)[
        audit_cols + ["priority_score"]
    ].head(50).to_csv(catboost_priority_path, index=False)

    print(f"Wrote {audit_path}")
    print(f"Wrote {scenario_path}")
    print(f"Wrote {catboost_priority_path}")
    print("\nScenario metrics:")
    print(scenario_metrics.to_string(index=False))


def add_audit_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pred_km_millimolar"] = 10 ** out["pred_log10_km"] * 1e3
    out["fold_error"] = np.maximum(
        out["pred_km_millimolar"] / out["km_millimolar"],
        out["km_millimolar"] / out["pred_km_millimolar"],
    )
    low_q, high_q = out.drop_duplicates("row_id")["km_millimolar"].quantile([0.025, 0.975])
    out["km_extreme_hard"] = out["km_millimolar"].lt(0.001) | out["km_millimolar"].gt(100)
    out["km_tail_95pct"] = out["km_millimolar"].lt(low_q) | out["km_millimolar"].gt(high_q)
    out["note_suspicious"] = out["curation_note"].fillna("").str.contains(SUSPICIOUS_NOTE_PATTERN)
    out["vmax_unconverted"] = out["vmax_value"].notna() & out["vmax_molar_per_s"].isna()
    out["manual_audit_recommended"] = (
        out["km_extreme_hard"]
        | out["note_suspicious"]
        | out["vmax_unconverted"]
        | out["doi_normalized"].fillna("").eq("")
    )
    out["audit_reasons"] = out.apply(audit_reasons, axis=1)
    return out


def audit_reasons(row: pd.Series) -> str:
    reasons = []
    if bool(row.get("km_extreme_hard")):
        reasons.append("hard_km_extreme")
    if bool(row.get("km_tail_95pct")):
        reasons.append("km_distribution_tail")
    if bool(row.get("note_suspicious")):
        reasons.append("curation_note_suspicious")
    if bool(row.get("vmax_unconverted")):
        reasons.append("vmax_unit_unconverted")
    if str(row.get("doi_normalized", "") or "") == "":
        reasons.append("missing_doi")
    return ";".join(reasons)


def scenario_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    scenarios = {
        "all_model_ready": pd.Series(True, index=df.index),
        "doi_bearing_only": df["doi_normalized"].fillna("").ne(""),
        "no_hard_km_extremes": ~df["km_extreme_hard"],
        "central_95pct_km": ~df["km_tail_95pct"],
        "no_suspicious_notes": ~df["note_suspicious"],
        "strict_core": (
            (~df["km_extreme_hard"])
            & (~df["note_suspicious"])
            & df["doi_normalized"].fillna("").ne("")
        ),
        "strict_core_or_missing_doi_allowed": (
            (~df["km_extreme_hard"]) & (~df["note_suspicious"])
        ),
    }
    rows = []
    for scenario, mask in scenarios.items():
        subset = df.loc[mask].copy()
        for model, group in subset.groupby("model"):
            rows.append(metric_row(scenario, model, "all", group))
            for substrate, sub_group in group.groupby("substrate_for_km_norm"):
                rows.append(metric_row(scenario, model, substrate, sub_group))
    return pd.DataFrame(rows).sort_values(["scenario", "model", "substrate"])


def metric_row(scenario: str, model: str, substrate: str, group: pd.DataFrame) -> dict:
    metrics = regression_metrics(group["target_log10_km"], group["pred_log10_km"])
    return {
        "scenario": scenario,
        "model": model,
        "substrate": substrate,
        "n_rows": int(len(group)),
        "n_unique_records": int(group["curation_record_id"].nunique()),
        "n_unique_source_clusters": int(group["external_source_cluster_id"].nunique()),
        **metrics,
    }


if __name__ == "__main__":
    main()
