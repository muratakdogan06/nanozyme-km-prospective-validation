#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import prepare_modeling_data, read_table
from src.diagnostics import external_ood_severity_rows, severity_error_association
from src.evaluate import regression_metrics
from src.features import build_feature_frame
from src.models import FeatureSpec, fit_predict, model_suite, require_catboost
from src.reporting import ensure_output_dirs
from src.validation import (
    raise_if_failed,
    validate_feature_columns,
    validate_modeling_frame,
    validation_results_to_frame,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train predefined models on the full internal modeling dataset and evaluate them "
            "on an independently curated external nanozyme Km table."
        )
    )
    parser.add_argument("--train-data", required=True, help="Internal training table.")
    parser.add_argument("--external-data", required=True, help="Curated external validation table.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--missingness-indicators", action="store_true")
    args = parser.parse_args()

    if not args.train_data.strip():
        parser.error("--train-data was empty.")
    if not args.external_data.strip():
        parser.error("--external-data was empty.")

    require_catboost()
    dirs = ensure_output_dirs()
    external_dir = dirs["tables"] / "external_validation"
    external_dir.mkdir(parents=True, exist_ok=True)

    train_raw = read_table(args.train_data)
    external_raw = read_table(args.external_data)
    train_df, train_filter_log = prepare_modeling_data(train_raw)
    external_df, external_filter_log = prepare_modeling_data(external_raw)

    validation_results = []
    validation_results.extend(validate_modeling_frame(train_df))
    validation_results.extend(validate_modeling_frame(external_df))

    X_train, meta = build_feature_frame(
        train_df, include_missing_indicators=args.missingness_indicators
    )
    X_external, _ = build_feature_frame(
        external_df, include_missing_indicators=args.missingness_indicators
    )
    X_external = X_external.reindex(columns=X_train.columns)
    validation_results.extend(validate_feature_columns(X_train.columns))
    raise_if_failed(validation_results)
    validation_results_to_frame(validation_results).to_csv(
        external_dir / "external_validation_checks.csv", index=False
    )

    y_train = train_df["target_log10_km"].astype(float)
    y_external = external_df["target_log10_km"].astype(float)
    feature_spec = FeatureSpec(meta["numeric_columns"], meta["categorical_columns"])
    models = model_suite(feature_spec, args.seed)
    severity = external_ood_severity_rows(
        train_df,
        external_df,
        X_train,
        X_external,
        meta["composition_columns"],
    )
    severity.to_csv(external_dir / "external_ood_severity_scores.csv", index=False)

    metric_rows = []
    prediction_tables = []
    for model_name, model in models.items():
        pred = fit_predict(model, X_train, y_train, X_external)
        metrics = regression_metrics(y_external, pred)
        metric_rows.append(
            {
                "validation_type": "external",
                "model": model_name,
                "n_train": len(train_df),
                "n_external": len(external_df),
                **metrics,
            }
        )
        pred_table = external_df[
            [
                "row_id",
                "formula",
                "standardized_formula",
                "source_group_id",
                "km_value",
                "km_unit",
                "km_common_molar",
                "target_log10_km",
            ]
        ].copy()
        for col in ["doi", "title", "journal", "year", "curation_source", "curation_note"]:
            if col in external_df.columns:
                pred_table[col] = external_df[col]
        pred_table = pred_table.merge(severity, on="row_id", how="left")
        pred_table["model"] = model_name
        pred_table["pred_log10_km"] = pred
        pred_table["abs_error_log10_km"] = (pred_table["target_log10_km"] - pred).abs()
        prediction_tables.append(pred_table)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    metrics.to_csv(dirs["metrics"] / "external_validation_metrics.csv", index=False)
    predictions.to_csv(external_dir / "external_validation_predictions.csv", index=False)
    severity_error_association(predictions).to_csv(
        external_dir / "external_severity_error_association.csv", index=False
    )
    external_overlap_diagnostics(train_df, external_df).to_csv(
        external_dir / "external_overlap_diagnostics.csv", index=False
    )
    external_analysis_commitment().to_csv(
        external_dir / "external_analysis_commitment.csv", index=False
    )
    train_filter_log.assign(dataset="internal_train").to_csv(
        external_dir / "internal_train_filter_log.csv", index=False
    )
    external_filter_log.assign(dataset="external_validation").to_csv(
        external_dir / "external_filter_log.csv", index=False
    )
    print(f"Wrote external metrics to {dirs['metrics'] / 'external_validation_metrics.csv'}")


def external_overlap_diagnostics(train_df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
    train_sources = set(train_df["source_group_id"].astype(str))
    train_formulas = set(train_df["formula_group_id"].astype(str))
    external_sources = external_df["source_group_id"].astype(str)
    external_formulas = external_df["formula_group_id"].astype(str)
    train_dois = _doi_tokens(train_df)
    external_dois = _doi_tokens(external_df)
    train_doi_set = set(train_dois[train_dois != ""])
    external_doi_values = external_dois[external_dois != ""]
    overlapping_dois = sorted(set(external_doi_values) & train_doi_set)
    train_stable_ids = _stable_source_tokens(train_df)
    external_stable_ids = _stable_source_tokens(external_df)
    train_stable_id_set = set(train_stable_ids[train_stable_ids != ""])
    external_stable_id_values = external_stable_ids[external_stable_ids != ""]
    overlapping_stable_ids = sorted(set(external_stable_id_values) & train_stable_id_set)
    return pd.DataFrame(
        [
            {
                "diagnostic": "external_rows",
                "value": int(len(external_df)),
                "interpretation": "retained rows after external target cleaning",
            },
            {
                "diagnostic": "external_source_overlap_fraction",
                "value": float(external_sources.isin(train_sources).mean())
                if len(external_sources)
                else 0.0,
                "interpretation": "fraction of external rows sharing a source_group_id with training data",
            },
            {
                "diagnostic": "external_formula_overlap_fraction",
                "value": float(external_formulas.isin(train_formulas).mean())
                if len(external_formulas)
                else 0.0,
                "interpretation": "fraction of external rows sharing a standardized formula with training data",
            },
            {
                "diagnostic": "external_unique_formulas",
                "value": int(external_df["formula_group_id"].nunique()),
                "interpretation": "number of unique standardized formulas in external validation data",
            },
            {
                "diagnostic": "train_unique_dois",
                "value": int(len(train_doi_set)),
                "interpretation": "unique DOI tokens found in the internal training table",
            },
            {
                "diagnostic": "external_unique_dois",
                "value": int(external_doi_values.nunique()),
                "interpretation": "unique DOI tokens found in retained external rows",
            },
            {
                "diagnostic": "external_rows_missing_doi",
                "value": int((external_dois == "").sum()),
                "interpretation": "external rows without a DOI token; manual source audit required",
            },
            {
                "diagnostic": "external_rows_with_training_doi_overlap",
                "value": int(external_dois.isin(train_doi_set).sum()) if train_doi_set else 0,
                "interpretation": "must be zero for a strictly external DOI-level validation claim",
            },
            {
                "diagnostic": "external_doi_overlap_fraction_all_rows",
                "value": float(external_dois.isin(train_doi_set).mean())
                if len(external_dois) and train_doi_set
                else 0.0,
                "interpretation": "fraction of all external rows sharing a DOI with training data",
            },
            {
                "diagnostic": "external_doi_overlap_fraction_among_doi_rows",
                "value": float(external_doi_values.isin(train_doi_set).mean())
                if len(external_doi_values) and train_doi_set
                else 0.0,
                "interpretation": "fraction of DOI-bearing external rows sharing a DOI with training data",
            },
            {
                "diagnostic": "external_overlapping_dois",
                "value": ";".join(overlapping_dois),
                "interpretation": "empty value indicates no DOI-level overlap was detected",
            },
            {
                "diagnostic": "train_unique_stable_source_ids",
                "value": int(len(train_stable_id_set)),
                "interpretation": "unique DOI or arXiv source identifiers found in the internal training table",
            },
            {
                "diagnostic": "external_unique_stable_source_ids",
                "value": int(external_stable_id_values.nunique()),
                "interpretation": "unique DOI or arXiv source identifiers found in retained external rows",
            },
            {
                "diagnostic": "external_rows_missing_stable_source_id",
                "value": int((external_stable_ids == "").sum()),
                "interpretation": "external rows without DOI or arXiv source identifiers; manual source audit required",
            },
            {
                "diagnostic": "external_rows_with_training_stable_source_overlap",
                "value": int(external_stable_ids.isin(train_stable_id_set).sum())
                if train_stable_id_set
                else 0,
                "interpretation": "must be zero for a strictly external stable-source validation claim",
            },
            {
                "diagnostic": "external_stable_source_overlap_fraction_all_rows",
                "value": float(external_stable_ids.isin(train_stable_id_set).mean())
                if len(external_stable_ids) and train_stable_id_set
                else 0.0,
                "interpretation": "fraction of all external rows sharing a DOI or arXiv identifier with training data",
            },
            {
                "diagnostic": "external_overlapping_stable_source_ids",
                "value": ";".join(overlapping_stable_ids),
                "interpretation": "empty value indicates no DOI/arXiv stable-source overlap was detected",
            },
        ]
    )


def external_analysis_commitment() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "commitment": "external_evaluation_reporting",
                "statement": (
                    "The external evaluation was specified before model refitting; "
                    "results are reported regardless of direction."
                ),
            },
            {
                "commitment": "external_success_scenarios",
                "statement": (
                    "If external MAE is close to leakage-graph MAE, the split is interpreted as "
                    "a deployment proxy; if external MAE is larger, the strictest internal split "
                    "is interpreted as still optimistic for real deployment."
                ),
            },
            {
                "commitment": "severity_validation",
                "statement": (
                    "External rows are scored by label-free OOD severity before error analysis; "
                    "severity-error association is reported for every predefined model."
                ),
            },
        ]
    )


def _doi_tokens(df: pd.DataFrame) -> pd.Series:
    candidates = [col for col in ["doi", "link", "url"] if col in df.columns]
    if not candidates:
        return pd.Series([""] * len(df), index=df.index)
    tokens = []
    for _, row in df[candidates].iterrows():
        token = ""
        for value in row:
            token = _extract_doi(value)
            if token:
                break
        tokens.append(token)
    return pd.Series(tokens, index=df.index)


def _stable_source_tokens(df: pd.DataFrame) -> pd.Series:
    candidates = [col for col in ["doi", "link", "url"] if col in df.columns]
    if not candidates:
        return pd.Series([""] * len(df), index=df.index)
    tokens = []
    for _, row in df[candidates].iterrows():
        token = ""
        for value in row:
            token = _extract_stable_source_id(value)
            if token:
                break
        tokens.append(token)
    return pd.Series(tokens, index=df.index)


def _extract_doi(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    text = text.replace("https://doi.org/", "").replace("http://doi.org/", "")
    match = re.search(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", text)
    if not match:
        return ""
    return match.group(0).rstrip(".,;)")


def _extract_stable_source_id(value: object) -> str:
    doi = _extract_doi(value)
    if doi:
        return f"doi:{doi}"
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    arxiv = re.search(r"(?:arxiv\.org/abs/|arxiv:)(\d{4}\.\d{4,5})(v\d+)?", text)
    if arxiv:
        return f"arxiv:{arxiv.group(1)}"
    return ""


if __name__ == "__main__":
    main()
