from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FORBIDDEN_FEATURE_TOKENS = [
    "doi",
    "source",
    "article",
    "paper",
    "title",
    "journal",
    "year",
    "pdf",
    "filename",
    "file",
]


@dataclass(frozen=True)
class ValidationResult:
    check_name: str
    passed: bool
    detail: str


def validate_modeling_frame(df: pd.DataFrame) -> list[ValidationResult]:
    results = []
    required = [
        "row_id",
        "target_log10_km",
        "km_common_molar",
        "source_group_id",
        "formula_group_id",
        "standardized_formula",
    ]
    missing = [col for col in required if col not in df.columns]
    results.append(
        ValidationResult(
            "required_modeling_columns",
            not missing,
            "missing=" + ",".join(missing) if missing else "all required modeling columns present",
        )
    )
    valid_target = (
        "target_log10_km" in df.columns
        and np.isfinite(pd.to_numeric(df["target_log10_km"], errors="coerce")).all()
    )
    results.append(
        ValidationResult(
            "finite_target_no_imputation",
            bool(valid_target),
            "target_log10_km is finite for every retained row"
            if valid_target
            else "non-finite target values remain",
        )
    )
    positive_km = (
        "km_common_molar" in df.columns
        and (pd.to_numeric(df["km_common_molar"], errors="coerce") > 0).all()
    )
    results.append(
        ValidationResult(
            "positive_common_km",
            bool(positive_km),
            "all retained Km values are positive after unit conversion"
            if positive_km
            else "invalid common-unit Km values remain",
        )
    )
    return results


def validate_feature_columns(columns: list[str] | pd.Index) -> list[ValidationResult]:
    bad = []
    for col in columns:
        lower = str(col).lower()
        if any(token == lower or lower.startswith(f"{token}_") for token in FORBIDDEN_FEATURE_TOKENS):
            bad.append(str(col))
    return [
        ValidationResult(
            "forbidden_metadata_not_used_as_features",
            not bad,
            "forbidden feature columns=" + ",".join(bad)
            if bad
            else "DOI/source/title/journal/year/PDF metadata absent from feature matrix",
        )
    ]


def validate_split(
    df: pd.DataFrame,
    train_idx,
    test_idx,
    split_type: str,
) -> list[ValidationResult]:
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    results = [
        ValidationResult(
            f"{split_type}_nonempty_train_test",
            len(train_idx) > 0 and len(test_idx) > 0,
            f"train={len(train_idx)}, test={len(test_idx)}",
        ),
        ValidationResult(
            f"{split_type}_row_disjointness",
            len(set(train_idx) & set(test_idx)) == 0,
            "train and test row indices are disjoint",
        )
    ]
    if split_type == "source_group":
        overlap = set(df.loc[train_idx, "source_group_id"]) & set(df.loc[test_idx, "source_group_id"])
        results.append(
            ValidationResult(
                "source_group_no_overlap",
                not overlap,
                f"{len(overlap)} source groups overlap" if overlap else "no source-group overlap",
            )
        )
    if split_type == "formula_group":
        overlap = set(df.loc[train_idx, "formula_group_id"]) & set(df.loc[test_idx, "formula_group_id"])
        results.append(
            ValidationResult(
                "formula_group_no_overlap",
                not overlap,
                f"{len(overlap)} formula groups overlap" if overlap else "no formula-group overlap",
            )
        )
    if split_type == "leakage_graph":
        overlap = set(df.loc[train_idx, "leakage_graph_component_id"]) & set(
            df.loc[test_idx, "leakage_graph_component_id"]
        )
        results.append(
            ValidationResult(
                "leakage_graph_component_no_overlap",
                not overlap,
                f"{len(overlap)} graph components overlap" if overlap else "no graph-component overlap",
            )
        )
    return results


def validation_results_to_frame(results: list[ValidationResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])


def raise_if_failed(results: list[ValidationResult]) -> None:
    failed = [r for r in results if not r.passed]
    if failed:
        detail = "; ".join(f"{r.check_name}: {r.detail}" for r in failed)
        raise ValueError(f"Validation failed: {detail}")
