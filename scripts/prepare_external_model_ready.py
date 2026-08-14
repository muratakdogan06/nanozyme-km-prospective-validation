#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import prepare_modeling_data, read_table
from src.features import parse_formula
from src.reporting import ensure_output_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the curated external validation CSV into a model-ready table."
    )
    parser.add_argument(
        "--input",
        default="outputs/tables/external_validation/curated_external_validation_records_final_clean.csv",
        help="Validated curated external CSV.",
    )
    parser.add_argument(
        "--output",
        default="outputs/tables/external_validation/external_validation_model_ready.csv",
        help="Model-ready external validation CSV to write.",
    )
    parser.add_argument(
        "--summary",
        default="outputs/tables/external_validation/external_validation_model_ready_summary.json",
        help="JSON summary report to write.",
    )
    parser.add_argument(
        "--filter-log",
        default="outputs/tables/external_validation/external_validation_model_ready_filter_log.csv",
        help="Deterministic target-cleaning log to write.",
    )
    args = parser.parse_args()

    dirs = ensure_output_dirs()
    external_dir = dirs["tables"] / "external_validation"
    external_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    filter_log_path = Path(args.filter_log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    filter_log_path.parent.mkdir(parents=True, exist_ok=True)

    raw = read_table(input_path)
    if "include_in_final" in raw.columns:
        raw = raw.loc[raw["include_in_final"].astype("string").str.lower().eq("yes")].copy()

    raw.insert(0, "curation_row_index", np.arange(len(raw), dtype=int))
    raw["curation_record_id"] = raw.get("record_id", pd.Series(index=raw.index, dtype=object))
    raw["doi_normalized"] = raw.get("doi", pd.Series(index=raw.index, dtype=object)).map(
        normalize_doi
    )
    raw["title_normalized"] = raw.get("title", pd.Series(index=raw.index, dtype=object)).map(
        normalize_text_token
    )
    raw["external_source_cluster_id"] = [
        source_cluster_id(doi, title, record_id)
        for doi, title, record_id in zip(
            raw["doi_normalized"],
            raw["title_normalized"],
            raw["curation_record_id"].astype("string").fillna(""),
        )
    ]

    modeling, filter_log = prepare_modeling_data(raw)
    filter_log.to_csv(filter_log_path, index=False)

    model_ready = modeling.copy()
    model_ready["external_row_id"] = np.arange(len(model_ready), dtype=int)
    model_ready["km_millimolar"] = model_ready["km_common_molar"].astype(float) * 1e3
    model_ready["target_log10_km_millimolar"] = np.log10(
        model_ready["km_millimolar"].astype(float)
    )
    model_ready["substrate_for_km_norm"] = model_ready.get(
        "substrate_for_km", pd.Series(index=model_ready.index, dtype=object)
    ).map(normalize_substrate)
    model_ready["is_tmb_target"] = (model_ready["substrate_for_km_norm"] == "TMB").astype(int)
    model_ready["is_h2o2_target"] = (model_ready["substrate_for_km_norm"] == "H2O2").astype(int)
    model_ready["vmax_molar_per_s"] = convert_vmax_to_molar_per_s(
        model_ready.get("vmax_value"), model_ready.get("vmax_unit")
    )
    model_ready["has_vmax"] = model_ready["vmax_molar_per_s"].notna().astype(int)
    model_ready["formula_parse_success_external"] = model_ready["formula"].map(
        lambda value: bool(parse_formula(value))
    )
    model_ready["doi_missing"] = model_ready["doi_normalized"].eq("").astype(int)
    model_ready["source_cluster_row_count"] = model_ready.groupby("external_source_cluster_id")[
        "external_source_cluster_id"
    ].transform("size")

    ordered = [
        "external_row_id",
        "curation_row_index",
        "curation_record_id",
        "include_in_final",
        "formula",
        "standardized_formula",
        "formula_group_id",
        "formula_parse_success_external",
        "activity",
        "surface",
        "substrate_for_km",
        "substrate_for_km_norm",
        "is_tmb_target",
        "is_h2o2_target",
        "km_value",
        "km_unit",
        "km_millimolar",
        "km_common_molar",
        "target_log10_km",
        "target_log10_km_millimolar",
        "vmax_value",
        "vmax_unit",
        "vmax_molar_per_s",
        "has_vmax",
        "reaction_type",
        "doi",
        "doi_normalized",
        "doi_missing",
        "title",
        "title_normalized",
        "journal",
        "year",
        "source_group_id",
        "external_source_cluster_id",
        "source_cluster_row_count",
        "source_location",
        "curation_source",
        "curation_note",
    ]
    ordered = [col for col in ordered if col in model_ready.columns]
    remaining = [col for col in model_ready.columns if col not in ordered and col != "row_id"]
    model_ready = model_ready[ordered + remaining]
    model_ready.to_csv(output_path, index=False)

    summary = build_summary(raw, model_ready, filter_log, input_path, output_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


def normalize_substrate(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    compact = text.lower().replace("₂", "2").replace(" ", "")
    if compact in {"tmb", "3,3',5,5'-tetramethylbenzidine"}:
        return "TMB"
    if compact in {"h2o2", "h₂o₂", "hydrogenperoxide"}:
        return "H2O2"
    return text


def normalize_doi(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if not text or text == "nan":
        return ""
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text.strip().rstrip(".")


def normalize_text_token(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def source_cluster_id(doi: str, title: str, record_id: object) -> str:
    if doi:
        return f"doi:{doi}"
    if title:
        return f"title:{hashlib.sha1(title.encode('utf-8')).hexdigest()[:12]}"
    return f"record:{record_id}"


def convert_vmax_to_molar_per_s(
    values: pd.Series | None, units: pd.Series | None
) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    numeric = pd.to_numeric(values, errors="coerce")
    if units is None:
        return pd.Series(np.nan, index=numeric.index, dtype=float)
    unit = (
        units.astype("string")
        .fillna("")
        .str.lower()
        .str.replace("μ", "u", regex=False)
        .str.replace("µ", "u", regex=False)
        .str.replace("−", "-", regex=False)
        .str.replace("·", " ", regex=False)
        .str.replace(".", " ", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip()
    )
    factor = pd.Series(np.nan, index=numeric.index, dtype=float)
    per_min = unit.str.contains("min", na=False)
    has_1e_minus_8 = unit.str.contains("10\\^-8|10-8", regex=True, na=False)
    has_nm = unit.str.contains("nm", na=False)
    has_um = unit.str.contains("um", na=False)
    has_mm = unit.str.contains("mm", na=False)
    has_plain_m = (
        unit.str.contains("m/s|ms-1|msec-1|mpersec|m/min|mmin-1", regex=True, na=False)
        & ~has_nm
        & ~has_um
        & ~has_mm
    )
    factor.loc[has_nm] = 1e-9
    factor.loc[has_um] = 1e-6
    factor.loc[has_mm] = 1e-3
    factor.loc[has_plain_m & factor.isna()] = 1.0
    factor.loc[has_1e_minus_8] = factor.loc[has_1e_minus_8].fillna(1.0) * 1e-8
    factor.loc[per_min] = factor.loc[per_min] / 60.0
    return numeric * factor


def build_summary(
    raw: pd.DataFrame,
    model_ready: pd.DataFrame,
    filter_log: pd.DataFrame,
    input_path: Path,
    output_path: Path,
) -> dict:
    doi_values = model_ready.get("doi_normalized", pd.Series(dtype=object)).astype("string")
    vmax = model_ready.get("vmax_molar_per_s", pd.Series(dtype=float))
    return {
        "input": str(input_path),
        "output": str(output_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "raw_yes_rows": int(len(raw)),
        "model_ready_rows": int(len(model_ready)),
        "unique_curation_record_ids": int(model_ready["curation_record_id"].nunique()),
        "unique_external_source_clusters": int(model_ready["external_source_cluster_id"].nunique()),
        "unique_standardized_formulas": int(model_ready["standardized_formula"].nunique()),
        "substrate_counts": model_ready["substrate_for_km_norm"].value_counts().to_dict(),
        "km_millimolar_min": float(model_ready["km_millimolar"].min()),
        "km_millimolar_median": float(model_ready["km_millimolar"].median()),
        "km_millimolar_max": float(model_ready["km_millimolar"].max()),
        "target_log10_km_molar_min": float(model_ready["target_log10_km"].min()),
        "target_log10_km_molar_max": float(model_ready["target_log10_km"].max()),
        "rows_with_vmax_raw": int(model_ready.get("vmax_value", pd.Series(dtype=float)).notna().sum()),
        "rows_with_vmax_molar_per_s": int(vmax.notna().sum()),
        "vmax_units": sorted(
            str(v) for v in model_ready.get("vmax_unit", pd.Series(dtype=object)).dropna().unique()
        ),
        "missing_doi_rows": int(doi_values.eq("").sum()),
        "unique_dois": int(doi_values[doi_values.ne("")].nunique()),
        "formula_parse_success_rows": int(model_ready["formula_parse_success_external"].sum()),
        "formula_parse_success_fraction": float(model_ready["formula_parse_success_external"].mean()),
        "filter_log": filter_log.to_dict(orient="records"),
    }


if __name__ == "__main__":
    main()
