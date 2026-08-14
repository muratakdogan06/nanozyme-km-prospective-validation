#!/usr/bin/env python3
"""Run a leakage-controlled internal DiZyme protocol-imputation sensitivity audit.

This script evaluates the frozen DiZyme 2.0 peroxidase Km artifact. It does not
use Km labels as outcomes. For each eligible row, selected fields are masked
and cross-imputed from donor folds that exclude the query publication. The
observed and imputed arms share one common completed feature vector, so their
prediction difference isolates replacement of the prespecified mask set.

The primary protocol set contains seven assay variables, including reaction
volume. The earlier six-variable definition is retained as a sensitivity
analysis. Three mask-imputation rules are evaluated: the notebook's
raw-space five-neighbour geometry, standardized-space five-neighbour geometry,
and training-fold medians. A matched-size substrate-descriptor mask is an
active comparator, not a null or negative control. Because all seven selected
substrate descriptors are encoded as zero in many released rows, the
comparator is stratified by this sentinel state. The primary comparator and
its paired protocol re-analysis use only descriptor-non-sentinel rows;
sentinel rows are retained as a data-coding diagnostic. Non-masked missing
values use the raw-space rule in every arm so rule comparisons alter
replacements in the selected mask set only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


AUDITED_COMMIT = "30cbf8345e2f40ce8ee3596d9d9da959efb686d7"
EXPECTED_HASHES = {
    "Km_model_stack_perox.pkl": (
        "804d98af7d790da0e44a1c1cb471e15a27c9dcf05a99b365605cfdb879a29cbf"
    ),
    "Km_scaler_stack_perox.pkl": (
        "b9024d7e83ff032961013b0449fc8a6943e7c056b42b7ca2aa06516a1c77b635"
    ),
    "perox_final.csv": (
        "454dbab5efe6780953379de620f0e0a56d2f2ee00fb2b2acde718d79880efd19"
    ),
    "features_perox.txt": (
        "80e62eb11316ebeae32d54f51c9b798d9c29bfc82373b395588873098901fce3"
    ),
    "stack_perox_Km.ipynb": (
        "8cf33f3082ffed3e5f7a75529dee54abfbd9846325f7d25d4ec910e41b093ac6"
    ),
    "nanozymes.csv": (
        "6af8d3ddabf56508e36c8c9cc97e4b74c7b7562eb305dab9a9aface2ba3fdb56"
    ),
}
EXPECTED_VERSIONS = {
    "python": "3.11",
    "scikit-learn": "1.2.2",
    "numpy": "1.23.5",
    "pandas": "2.0.1",
    "joblib": "1.2.0",
    "catboost": "1.2.1",
    "lightgbm": "4.0.0",
    "scipy": "1.10.1",
    "threadpoolctl": "3.1.0",
}

PROTOCOL_6 = ["ph", "temp", "lgCmin", "lgCmax", "Cconst", "lgCcat"]
PROTOCOL_7 = PROTOCOL_6 + ["lgvolume"]
SUBSTRATE_DESCRIPTOR_7 = [
    "MolWt",
    "TPSA",
    "XLogP",
    "PEOE_VSA7",
    "PEOE_VSA9",
    "Kappa2",
    "BalabanJ",
]
MASK_SETS = {
    "primary_protocol_7_fields": PROTOCOL_7,
    "sensitivity_protocol_6_fields": PROTOCOL_6,
    "paired_protocol_7_fields_descriptor_non_sentinel": PROTOCOL_7,
    "active_comparator_substrate_7_fields_descriptor_non_sentinel": (
        SUBSTRATE_DESCRIPTOR_7
    ),
    "diagnostic_protocol_7_fields_descriptor_sentinel": PROTOCOL_7,
    "diagnostic_substrate_7_fields_descriptor_sentinel": SUBSTRATE_DESCRIPTOR_7,
}
IMPUTATION_RULES = ("raw_knn", "standardized_knn", "training_median")
TWO_FOLD_LOG10 = float(np.log10(2.0))
KNOWN_MISSING_TOKENS = {"#????"}
PUBLICATION_QC_MIN_ID_COVERAGE = 0.99
PUBLICATION_QC_MIN_FORMULA_AGREEMENT = 0.98
PUBLICATION_QC_MIN_NUMERIC_KM_AGREEMENT = 0.98
PUBLICATION_QC_MIN_DOI_COVERAGE = 0.95


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def runtime_versions() -> dict[str, str]:
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_full": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": package_version("joblib"),
        "catboost": package_version("catboost"),
        "lightgbm": package_version("lightgbm"),
        "scipy": package_version("scipy"),
        "threadpoolctl": package_version("threadpoolctl"),
    }


def validate_runtime(allow_mismatch: bool = False) -> dict[str, str]:
    observed = runtime_versions()
    mismatches = {
        name: {"expected": expected, "observed": observed.get(name)}
        for name, expected in EXPECTED_VERSIONS.items()
        if observed.get(name) != expected
    }
    if mismatches and not allow_mismatch:
        details = "; ".join(
            f"{name}: expected {values['expected']}, observed {values['observed']}"
            for name, values in mismatches.items()
        )
        raise RuntimeError(
            "The frozen DiZyme artifact must be run in its prespecified compatibility "
            f"environment. Version mismatch: {details}"
        )
    observed["version_check"] = "overridden" if mismatches else "passed"
    observed["version_mismatches"] = json.dumps(mismatches, sort_keys=True)
    return observed


def git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "not-a-git-checkout"


def artifact_paths(repo_root: Path) -> dict[str, Path]:
    peroxidase = repo_root / "models" / "peroxidase"
    return {
        "Km_model_stack_perox.pkl": peroxidase / "Km_model_stack_perox.pkl",
        "Km_scaler_stack_perox.pkl": peroxidase / "Km_scaler_stack_perox.pkl",
        "perox_final.csv": peroxidase / "perox_final.csv",
        "features_perox.txt": peroxidase / "features_perox.txt",
        "stack_perox_Km.ipynb": peroxidase / "stack_perox_Km.ipynb",
        "nanozymes.csv": repo_root / "data" / "nanozymes.csv",
    }


def validate_artifacts(
    repo_root: Path, allow_mismatch: bool = False
) -> tuple[dict[str, Path], dict[str, str], str]:
    paths = artifact_paths(repo_root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing DiZyme artifact(s): " + ", ".join(missing))

    observed_hashes = {name: sha256_file(path) for name, path in paths.items()}
    hash_mismatches = {
        name: {"expected": EXPECTED_HASHES[name], "observed": observed_hashes[name]}
        for name in EXPECTED_HASHES
        if observed_hashes[name] != EXPECTED_HASHES[name]
    }
    commit = git_commit(repo_root)
    commit_mismatch = commit != AUDITED_COMMIT
    if (hash_mismatches or commit_mismatch) and not allow_mismatch:
        raise RuntimeError(
            "Repository state does not match the audited DiZyme artifact. "
            f"commit={commit}; hash_mismatches={json.dumps(hash_mismatches, sort_keys=True)}"
        )
    return paths, observed_hashes, commit


def read_feature_contract(path: Path) -> list[str]:
    features = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(features) != 37 or len(set(features)) != 37:
        raise ValueError(f"Expected 37 unique features, found {len(features)}")
    return features


def read_numeric_training_table(
    csv_path: Path, features: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    raw = pd.read_csv(csv_path)
    missing_columns = sorted(set(features) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Training table lacks feature columns: {missing_columns}")

    numeric = pd.DataFrame(index=raw.index)
    coerced_tokens: dict[str, list[str]] = {}
    unknown_tokens: dict[str, list[str]] = {}
    for column in features:
        converted = pd.to_numeric(raw[column], errors="coerce")
        invalid = raw[column].notna() & converted.isna()
        if invalid.any():
            tokens = sorted(raw.loc[invalid, column].astype(str).unique().tolist())
            coerced_tokens[column] = tokens
            unknown = sorted(set(tokens) - KNOWN_MISSING_TOKENS)
            if unknown:
                unknown_tokens[column] = unknown
        numeric[column] = converted.astype(float)

    if unknown_tokens:
        raise ValueError(
            "Unexpected non-numeric feature tokens: "
            + json.dumps(unknown_tokens, sort_keys=True)
        )
    return raw, numeric, coerced_tokens


def normalize_formula_token(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"\s+", "", text)


def normalize_doi(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    match = re.search(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", text)
    return match.group(0).rstrip(".,;)") if match else ""


def km_tokens_agree(left: object, right: object) -> tuple[bool, bool]:
    """Return (agreement, both_numeric) for publication-map quality control."""

    left_numeric = pd.to_numeric(pd.Series([left]), errors="coerce").iloc[0]
    right_numeric = pd.to_numeric(pd.Series([right]), errors="coerce").iloc[0]
    both_numeric = pd.notna(left_numeric) and pd.notna(right_numeric)
    if both_numeric:
        return (
            bool(
                np.isclose(
                    float(left_numeric),
                    float(right_numeric),
                    rtol=1e-9,
                    atol=1e-12,
                )
            ),
            True,
        )
    left_text = "" if pd.isna(left) else str(left).strip().lower()
    right_text = "" if pd.isna(right) else str(right).strip().lower()
    return bool(left_text and left_text == right_text), False


def recover_publication_groups(
    peroxidase: pd.DataFrame, source_csv: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Recover DOI groups through the processed ``id`` to source ``#`` mapping.

    Duplicate source identifiers are resolved only when formula/Km concordance
    selects one candidate or all best candidates share a DOI. Rows with no
    defensible DOI are assigned singleton groups. Global quality thresholds are
    evaluated before publication-grouped analysis is allowed.
    """

    required_peroxidase = {"id", "formula", "Km"}
    missing_peroxidase = sorted(required_peroxidase - set(peroxidase.columns))
    if missing_peroxidase:
        raise ValueError(
            f"Peroxidase table lacks publication-map fields: {missing_peroxidase}"
        )
    source = pd.read_csv(source_csv)
    required_source = {"#", "formula", "Km, mM", "link"}
    missing_source = sorted(required_source - set(source.columns))
    if missing_source:
        raise ValueError(f"Source table lacks publication-map fields: {missing_source}")

    source = source.copy()
    source["_source_id"] = pd.to_numeric(source["#"], errors="coerce").astype("Int64")
    source["_doi_normalized"] = source["link"].map(normalize_doi)
    duplicate_mask = source["_source_id"].duplicated(keep=False)
    duplicate_id_classes: dict[int, str] = {}
    for source_id, group in source.loc[duplicate_mask].groupby("_source_id"):
        doi_count = group["_doi_normalized"].replace("", np.nan).nunique(dropna=True)
        duplicate_id_classes[int(source_id)] = (
            "cross_publication_collision_resolved"
            if doi_count > 1
            else "same_publication_duplicate"
        )
    records: list[dict[str, object]] = []

    for row_position, row in peroxidase.reset_index(drop=True).iterrows():
        processed_id = pd.to_numeric(pd.Series([row["id"]]), errors="coerce").iloc[0]
        candidates = (
            source.loc[source["_source_id"] == int(processed_id)]
            if pd.notna(processed_id)
            else source.iloc[0:0]
        )
        evaluations: list[dict[str, object]] = []
        for source_row, candidate in candidates.iterrows():
            formula_match = normalize_formula_token(row["formula"]) == normalize_formula_token(
                candidate["formula"]
            )
            km_match, km_numeric_pair = km_tokens_agree(row["Km"], candidate["Km, mM"])
            evaluations.append(
                {
                    "source_row": int(source_row),
                    "formula_match": bool(formula_match),
                    "km_match": bool(km_match),
                    "km_numeric_pair": bool(km_numeric_pair),
                    "doi_normalized": normalize_doi(candidate["link"]),
                    "score": int(formula_match) + int(km_match),
                }
            )

        selected: dict[str, object] | None = None
        if not evaluations:
            resolution = "missing_source_id"
        else:
            best_score = max(int(candidate["score"]) for candidate in evaluations)
            best = [candidate for candidate in evaluations if candidate["score"] == best_score]
            if len(best) == 1:
                selected = best[0]
                resolution = (
                    "unique_source_id"
                    if len(evaluations) == 1
                    else "duplicate_id_resolved_by_formula_km"
                )
            else:
                best_dois = {
                    str(candidate["doi_normalized"])
                    for candidate in best
                    if candidate["doi_normalized"]
                }
                if len(best_dois) == 1:
                    selected = best[0]
                    resolution = "duplicate_id_shared_doi"
                else:
                    resolution = "ambiguous_duplicate_id"

        defensible = bool(
            selected
            and selected["doi_normalized"]
            and (selected["formula_match"] or selected["km_match"])
        )
        doi = str(selected["doi_normalized"]) if defensible and selected else ""
        publication_id = f"doi:{doi}" if doi else f"unknown:row-{row_position}"
        processed_id_int = int(processed_id) if pd.notna(processed_id) else None
        duplicate_classification = (
            duplicate_id_classes.get(processed_id_int, "not_duplicated")
            if processed_id_int is not None
            else "missing_source_id"
        )
        candidate_publication_count = len(
            {
                str(candidate["doi_normalized"])
                for candidate in evaluations
                if candidate["doi_normalized"]
            }
        )
        if duplicate_classification == "cross_publication_collision_resolved":
            duplicate_note = (
                f"source #{processed_id_int} has candidates from multiple DOI groups; "
                "the selected publication was resolved by formula and Km-token agreement"
            )
        elif duplicate_classification == "same_publication_duplicate":
            duplicate_note = (
                f"source #{processed_id_int} is duplicated within one normalized DOI group"
            )
        else:
            duplicate_note = ""
        records.append(
            {
                "row_position": row_position,
                "processed_id": row["id"],
                "publication_id": publication_id,
                "doi_normalized": doi,
                "source_candidate_count": len(evaluations),
                "source_candidate_publication_count": candidate_publication_count,
                "source_row": selected["source_row"] if selected else np.nan,
                "formula_match": bool(selected and selected["formula_match"]),
                "km_match": bool(selected and selected["km_match"]),
                "km_numeric_pair": bool(selected and selected["km_numeric_pair"]),
                "resolution": resolution if doi else f"{resolution}_singleton",
                "duplicate_id_classification": duplicate_classification,
                "duplicate_id_note": duplicate_note,
            }
        )

    mapping = pd.DataFrame(records)
    matched_id = mapping["source_candidate_count"] > 0
    numeric_pairs = mapping["km_numeric_pair"]
    id_coverage = float(matched_id.mean())
    formula_agreement = float(mapping.loc[matched_id, "formula_match"].mean())
    numeric_km_agreement = float(mapping.loc[numeric_pairs, "km_match"].mean())
    doi_coverage = float(mapping["doi_normalized"].ne("").mean())
    qc = {
        "processed_rows": len(peroxidase),
        "source_rows": len(source),
        "source_unique_ids": int(source["_source_id"].nunique(dropna=True)),
        "source_duplicated_ids": int(
            source.loc[duplicate_mask, "_source_id"].nunique(dropna=True)
        ),
        "source_rows_with_duplicated_ids": int(duplicate_mask.sum()),
        "same_publication_duplicate_ids": sorted(
            source_id
            for source_id, classification in duplicate_id_classes.items()
            if classification == "same_publication_duplicate"
        ),
        "cross_publication_duplicate_ids": sorted(
            source_id
            for source_id, classification in duplicate_id_classes.items()
            if classification == "cross_publication_collision_resolved"
        ),
        "matched_source_id_rows": int(matched_id.sum()),
        "id_coverage": id_coverage,
        "normalized_formula_matches": int(mapping["formula_match"].sum()),
        "normalized_formula_agreement": formula_agreement,
        "numeric_km_pairs": int(numeric_pairs.sum()),
        "numeric_km_matches": int(mapping.loc[numeric_pairs, "km_match"].sum()),
        "numeric_km_agreement": numeric_km_agreement,
        "all_token_km_matches": int(mapping["km_match"].sum()),
        "doi_recovered_rows": int(mapping["doi_normalized"].ne("").sum()),
        "doi_coverage": doi_coverage,
        "unique_recovered_dois": int(
            mapping.loc[mapping["doi_normalized"].ne(""), "doi_normalized"].nunique()
        ),
        "singleton_unknown_rows": int(mapping["doi_normalized"].eq("").sum()),
        "thresholds": {
            "minimum_id_coverage": PUBLICATION_QC_MIN_ID_COVERAGE,
            "minimum_formula_agreement": PUBLICATION_QC_MIN_FORMULA_AGREEMENT,
            "minimum_numeric_km_agreement": PUBLICATION_QC_MIN_NUMERIC_KM_AGREEMENT,
            "minimum_doi_coverage": PUBLICATION_QC_MIN_DOI_COVERAGE,
        },
    }
    qc_passed = bool(
        id_coverage >= PUBLICATION_QC_MIN_ID_COVERAGE
        and formula_agreement >= PUBLICATION_QC_MIN_FORMULA_AGREEMENT
        and numeric_km_agreement >= PUBLICATION_QC_MIN_NUMERIC_KM_AGREEMENT
        and doi_coverage >= PUBLICATION_QC_MIN_DOI_COVERAGE
    )
    qc["qc_passed"] = qc_passed
    if not qc_passed:
        raise RuntimeError(
            "Publication mapping failed prespecified quality thresholds; use "
            "formula-grouped folds and report the mapping failure. QC="
            + json.dumps(qc, sort_keys=True)
        )
    return mapping, qc


def validate_frozen_pipeline(model: object, scaler: object, features: Sequence[str]) -> None:
    check_is_fitted(model)
    check_is_fitted(scaler)
    scaler_width = getattr(scaler, "n_features_in_", None)
    if scaler_width != len(features):
        raise ValueError(
            f"Released scaler expects {scaler_width} features; contract has {len(features)}"
        )
    names = getattr(scaler, "feature_names_in_", None)
    if names is not None and list(names) != list(features):
        raise ValueError("Released scaler feature_names_in_ does not match features_perox.txt")
    model_width = getattr(model, "n_features_in_", None)
    if model_width is not None and model_width != len(features):
        raise ValueError(
            f"Released estimator expects {model_width} features; contract has {len(features)}"
        )


def eligible_positions(X: pd.DataFrame, mask_fields: Sequence[str]) -> np.ndarray:
    missing = sorted(set(mask_fields) - set(X.columns))
    if missing:
        raise ValueError(f"Mask fields are absent from the feature matrix: {missing}")
    return np.flatnonzero(X[list(mask_fields)].notna().all(axis=1).to_numpy())


def descriptor_integrity_diagnostics(
    X: pd.DataFrame, primary_eligible: np.ndarray
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.DataFrame, dict[str, int]]:
    """Classify the seven-descriptor zero sentinel and record XLogP anomalies.

    Values are audited but never corrected. The returned positions keep the
    primary active comparator chemically interpretable while preserving the
    sentinel subset as a separate data-coding diagnostic.
    """

    missing = sorted(set(SUBSTRATE_DESCRIPTOR_7) - set(X.columns))
    if missing:
        raise ValueError(f"Substrate descriptor fields are absent: {missing}")
    primary_eligible = np.asarray(primary_eligible, dtype=int)
    descriptors = X.iloc[primary_eligible][SUBSTRATE_DESCRIPTOR_7]
    sentinel_local = descriptors.eq(0).all(axis=1).to_numpy()
    sentinel_positions = primary_eligible[sentinel_local]
    non_sentinel_positions = primary_eligible[~sentinel_local]

    sentinel_flag = pd.Series(False, index=X.index, dtype=bool)
    sentinel_flag.iloc[sentinel_positions] = True
    xlogp_high = descriptors["XLogP"].gt(10)
    xlogp_near_tpsa = xlogp_high & (
        descriptors["XLogP"] - descriptors["TPSA"]
    ).abs().le(1.0)
    counts = {
        "primary_7_field_rows": int(len(primary_eligible)),
        "descriptor_sentinel_all_zero_rows": int(sentinel_local.sum()),
        "descriptor_non_sentinel_rows": int((~sentinel_local).sum()),
        "xlogp_above_10_rows": int(xlogp_high.sum()),
        "xlogp_above_10_within_1_of_tpsa_rows": int(xlogp_near_tpsa.sum()),
    }
    audit = pd.DataFrame(
        [
            {
                "audit_item": "seven_descriptor_all_zero_sentinel",
                "population_rows": len(primary_eligible),
                "flagged_rows": counts["descriptor_sentinel_all_zero_rows"],
                "parser_action": "recorded_not_corrected",
                "analysis_action": (
                    "excluded from primary active comparator; retained as a "
                    "data-coding diagnostic stratum"
                ),
            },
            {
                "audit_item": "seven_descriptor_non_sentinel",
                "population_rows": len(primary_eligible),
                "flagged_rows": counts["descriptor_non_sentinel_rows"],
                "parser_action": "recorded_not_corrected",
                "analysis_action": (
                    "primary active comparator and matched protocol re-analysis"
                ),
            },
            {
                "audit_item": "xlogp_above_10",
                "population_rows": len(primary_eligible),
                "flagged_rows": counts["xlogp_above_10_rows"],
                "parser_action": "recorded_not_corrected",
                "analysis_action": "no value correction; frozen artifact preserved",
            },
            {
                "audit_item": "xlogp_above_10_and_within_1_of_tpsa",
                "population_rows": len(primary_eligible),
                "flagged_rows": counts[
                    "xlogp_above_10_within_1_of_tpsa_rows"
                ],
                "parser_action": "recorded_not_corrected",
                "analysis_action": (
                    "reported as consistent with possible parsing or column-shift "
                    "contamination; no causal correction attempted"
                ),
            },
        ]
    )
    return (
        non_sentinel_positions,
        sentinel_positions,
        sentinel_flag,
        audit,
        counts,
    )


def make_cross_imputation_folds(
    eligible: np.ndarray,
    metadata: pd.DataFrame,
    n_splits: int,
    seed: int,
    split_unit: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2 or n_splits > len(eligible):
        raise ValueError("n_splits must be between 2 and the eligible-row count")

    positions = np.arange(len(eligible))
    if split_unit == "row":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(train, test) for train, test in splitter.split(positions)]
    if split_unit == "formula":
        if "formula" not in metadata.columns:
            raise ValueError("Formula-grouped cross-imputation requires a formula column")
        groups = metadata.iloc[eligible]["formula"].fillna("<missing-formula>").astype(str)
        if groups.nunique() < n_splits:
            raise ValueError("Fewer formula groups than requested folds")
        splitter = GroupKFold(n_splits=n_splits)
        return [
            (train, test)
            for train, test in splitter.split(positions, groups=groups.to_numpy())
        ]
    if split_unit == "publication":
        if "publication_id" not in metadata.columns:
            raise ValueError(
                "Publication-grouped cross-imputation requires a publication_id column"
            )
        groups = metadata.iloc[eligible]["publication_id"].astype(str)
        if groups.nunique() < n_splits:
            raise ValueError("Fewer publication groups than requested folds")
        splitter = GroupKFold(n_splits=n_splits)
        return [
            (train, test)
            for train, test in splitter.split(positions, groups=groups.to_numpy())
        ]
    raise ValueError(f"Unknown split unit: {split_unit}")


def _fit_transform_rule(
    donor: np.ndarray,
    query: np.ndarray,
    rule: str,
    n_neighbors: int,
) -> np.ndarray:
    if rule == "raw_knn":
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
        return imputer.fit(donor).transform(query)
    if rule == "standardized_knn":
        geometry_scaler = StandardScaler()
        donor_scaled = geometry_scaler.fit_transform(donor)
        query_scaled = geometry_scaler.transform(query)
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
        imputed_scaled = imputer.fit(donor_scaled).transform(query_scaled)
        return geometry_scaler.inverse_transform(imputed_scaled)
    if rule == "training_median":
        imputer = SimpleImputer(strategy="median")
        return imputer.fit(donor).transform(query)
    raise ValueError(f"Unknown imputation rule: {rule}")


def donor_mask_for_query_groups(
    metadata: pd.DataFrame, query_rows: np.ndarray, split_unit: str
) -> np.ndarray:
    donor_mask = np.ones(len(metadata), dtype=bool)
    if split_unit == "row":
        donor_mask[query_rows] = False
        return donor_mask
    group_column = {"formula": "formula", "publication": "publication_id"}.get(
        split_unit
    )
    if group_column is None or group_column not in metadata.columns:
        raise ValueError(f"Cannot construct donor exclusion for split unit: {split_unit}")
    query_groups = set(metadata.iloc[query_rows][group_column].astype(str))
    return ~metadata[group_column].astype(str).isin(query_groups).to_numpy()


def cross_imputed_arms(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    mask_fields: Sequence[str],
    rule: str,
    *,
    n_splits: int = 10,
    seed: int = 2026,
    split_unit: str = "publication",
    n_neighbors: int = 5,
    eligible_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return eligible indices, observed arm, imputed arm, and fold labels.

    The query fold is always absent from the imputer donor pool. Both arms use
    one common completed vector. The imputed arm replaces only ``mask_fields``.
    """

    eligible = (
        eligible_positions(X, mask_fields)
        if eligible_override is None
        else np.asarray(eligible_override, dtype=int)
    )
    if not X.iloc[eligible][list(mask_fields)].notna().to_numpy().all():
        raise ValueError("eligible_override contains rows incomplete in the mask fields")
    folds = make_cross_imputation_folds(
        eligible, metadata, n_splits=n_splits, seed=seed, split_unit=split_unit
    )
    matrix = X.to_numpy(dtype=float)
    mask_columns = [X.columns.get_loc(column) for column in mask_fields]
    imputed_arm = np.full((len(eligible), X.shape[1]), np.nan, dtype=float)
    observed_arm = np.full((len(eligible), X.shape[1]), np.nan, dtype=float)
    fold_labels = np.full(len(eligible), -1, dtype=int)

    for fold_number, (_, test_positions) in enumerate(folds):
        query_rows = eligible[test_positions]
        donor_mask = donor_mask_for_query_groups(metadata, query_rows, split_unit)
        donor = matrix[donor_mask]
        observed_query = matrix[query_rows].copy()
        common_completed = _fit_transform_rule(
            donor, observed_query, "raw_knn", n_neighbors
        )
        masked_query = observed_query.copy()
        masked_query[:, mask_columns] = np.nan
        rule_imputed = (
            _fit_transform_rule(donor, masked_query, "raw_knn", n_neighbors)
            if rule == "raw_knn"
            else _fit_transform_rule(donor, masked_query, rule, n_neighbors)
        )
        transformed = common_completed.copy()
        transformed[:, mask_columns] = rule_imputed[:, mask_columns]
        if (
            transformed.shape != observed_query.shape
            or rule_imputed.shape != observed_query.shape
        ):
            raise ValueError(
                "Imputer changed the feature width, likely because a donor-fold column "
                "was entirely missing"
            )
        imputed_arm[test_positions] = transformed
        observed_arm[test_positions] = common_completed
        fold_labels[test_positions] = fold_number

    if not np.isfinite(observed_arm).all() or not np.isfinite(imputed_arm).all():
        raise ValueError("Non-finite values remain after cross-imputation")
    return eligible, observed_arm, imputed_arm, fold_labels


def predict_log10_km(
    model: object,
    scaler: object,
    values: np.ndarray,
    features: Sequence[str],
) -> np.ndarray:
    frame = pd.DataFrame(values, columns=features)
    transformed = scaler.transform(frame)
    prediction = np.asarray(model.predict(transformed), dtype=float).reshape(-1)
    if len(prediction) != len(frame) or not np.isfinite(prediction).all():
        raise ValueError("Frozen estimator returned invalid predictions")
    return prediction


def prediction_rows(
    metadata: pd.DataFrame,
    eligible: np.ndarray,
    fold_labels: np.ndarray,
    observed_prediction: np.ndarray,
    imputed_prediction: np.ndarray,
    *,
    mask_set: str,
    rule: str,
    split_unit: str,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "row_position": eligible,
            "cross_imputation_fold": fold_labels,
            "mask_set": mask_set,
            "imputation_rule": rule,
            "split_unit": split_unit,
            "prediction_reference_log10_km_mM": observed_prediction,
            "prediction_mask_imputed_log10_km_mM": imputed_prediction,
        }
    )
    metadata_columns = [
        column
        for column in (
            "id",
            "formula",
            "activity",
            "publication_id",
            "doi_normalized",
            "substrate_descriptor_sentinel_all_zero",
        )
        if column in metadata.columns
    ]
    for insertion_position, column in enumerate(metadata_columns, start=1):
        result.insert(
            insertion_position,
            column,
            metadata.iloc[eligible][column].to_numpy(),
        )
    result["signed_prediction_shift_log10"] = (
        result["prediction_mask_imputed_log10_km_mM"]
        - result["prediction_reference_log10_km_mM"]
    )
    result["absolute_prediction_shift_log10"] = result[
        "signed_prediction_shift_log10"
    ].abs()
    result["shift_at_least_twofold"] = (
        result["absolute_prediction_shift_log10"] >= TWO_FOLD_LOG10
    )
    reference_sd = float(np.std(observed_prediction, ddof=1))
    if not np.isfinite(reference_sd) or reference_sd <= 0:
        raise ValueError("Reference predictions have zero or invalid standard deviation")
    result["reference_prediction_sd_log10"] = reference_sd
    result["absolute_shift_over_reference_prediction_sd"] = (
        result["absolute_prediction_shift_log10"] / reference_sd
    )
    return result


def publication_cluster_bootstrap_median_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    unique_clusters = pd.unique(clusters)
    if draws <= 0 or len(unique_clusters) < 2:
        return np.nan, np.nan, len(unique_clusters)
    cluster_values = [values[clusters == cluster] for cluster in unique_clusters]
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        selected = rng.integers(0, len(cluster_values), size=len(cluster_values))
        sample = np.concatenate([cluster_values[index] for index in selected])
        estimates[draw] = np.median(sample)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high), len(unique_clusters)


def summarize_predictions(
    predictions: pd.DataFrame,
    *,
    bootstrap_draws: int = 10_000,
    seed: int = 2026,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["mask_set", "imputation_rule", "split_unit"]
    rng = np.random.default_rng(seed)
    for keys, group in predictions.groupby(group_columns, sort=False):
        absolute = group["absolute_prediction_shift_log10"].to_numpy()
        signed = group["signed_prediction_shift_log10"].to_numpy()
        normalized = group["absolute_shift_over_reference_prediction_sd"].to_numpy()
        reference_sd = float(group["reference_prediction_sd_log10"].iloc[0])
        if "publication_id" in group.columns:
            clusters = group["publication_id"].astype(str).to_numpy()
            ci_low, ci_high, n_clusters = publication_cluster_bootstrap_median_ci(
                absolute, clusters, draws=bootstrap_draws, rng=rng
            )
            normalized_low = ci_low / reference_sd
            normalized_high = ci_high / reference_sd
        else:
            ci_low = ci_high = normalized_low = normalized_high = np.nan
            n_clusters = 0
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_rows": len(group),
                "n_publication_clusters": n_clusters,
                "median_absolute_shift_log10": float(np.median(absolute)),
                "median_absolute_shift_cluster_ci_low": ci_low,
                "median_absolute_shift_cluster_ci_high": ci_high,
                "mean_absolute_shift_log10": float(np.mean(absolute)),
                "p90_absolute_shift_log10": float(np.quantile(absolute, 0.90)),
                "fraction_at_least_twofold": float(np.mean(absolute >= TWO_FOLD_LOG10)),
                "reference_prediction_sd_log10": reference_sd,
                "median_absolute_shift_over_reference_sd": float(
                    np.median(normalized)
                ),
                "median_normalized_shift_cluster_ci_low": normalized_low,
                "median_normalized_shift_cluster_ci_high": normalized_high,
                "mean_signed_shift_log10": float(np.mean(signed)),
                "median_signed_shift_log10": float(np.median(signed)),
                "cluster_bootstrap_draws": bootstrap_draws,
            }
        )
    return pd.DataFrame(rows)


def compare_imputation_rules(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (mask_set, split_unit), group in predictions.groupby(
        ["mask_set", "split_unit"], sort=False
    ):
        pivot = group.pivot(
            index="row_position",
            columns="imputation_rule",
            values="prediction_mask_imputed_log10_km_mM",
        )
        for left, right in itertools.combinations(IMPUTATION_RULES, 2):
            difference = (pivot[left] - pivot[right]).abs().dropna().to_numpy()
            rows.append(
                {
                    "mask_set": mask_set,
                    "split_unit": split_unit,
                    "rule_a": left,
                    "rule_b": right,
                    "n_rows": len(difference),
                    "median_absolute_prediction_difference_log10": float(
                        np.median(difference)
                    ),
                    "p90_absolute_prediction_difference_log10": float(
                        np.quantile(difference, 0.90)
                    ),
                    "fraction_difference_at_least_twofold": float(
                        np.mean(difference >= TWO_FOLD_LOG10)
                    ),
                }
            )
    return pd.DataFrame(rows)


def compare_protocol_to_substrate_mask(
    predictions: pd.DataFrame,
    *,
    bootstrap_draws: int = 10_000,
    seed: int = 2026,
) -> pd.DataFrame:
    mask_pairs = (
        (
            "descriptor_non_sentinel",
            "paired_protocol_7_fields_descriptor_non_sentinel",
            "active_comparator_substrate_7_fields_descriptor_non_sentinel",
            "primary_active_comparator",
        ),
        (
            "descriptor_all_zero_sentinel",
            "diagnostic_protocol_7_fields_descriptor_sentinel",
            "diagnostic_substrate_7_fields_descriptor_sentinel",
            "data_coding_diagnostic_only",
        ),
    )
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for (rule, split_unit), group in predictions.groupby(
        ["imputation_rule", "split_unit"], sort=False
    ):
        for descriptor_stratum, protocol_name, comparator_name, use in mask_pairs:
            selected = group[
                group["mask_set"].isin([protocol_name, comparator_name])
            ]
            pivot = selected.pivot(
                index="row_position",
                columns="mask_set",
                values="absolute_prediction_shift_log10",
            ).dropna()
            if pivot.empty:
                continue
            metadata = (
                selected.drop_duplicates("row_position")
                .set_index("row_position")
                .loc[pivot.index]
            )
            paired_difference = (
                pivot[comparator_name] - pivot[protocol_name]
            ).to_numpy()
            clusters = metadata["publication_id"].astype(str).to_numpy()
            ci_low, ci_high, n_clusters = publication_cluster_bootstrap_median_ci(
                paired_difference,
                clusters,
                draws=bootstrap_draws,
                rng=rng,
            )
            rows.append(
                {
                    "descriptor_stratum": descriptor_stratum,
                    "analysis_use": use,
                    "imputation_rule": rule,
                    "split_unit": split_unit,
                    "n_rows": len(pivot),
                    "n_publication_clusters": n_clusters,
                    "median_protocol_absolute_shift_log10": float(
                        np.median(pivot[protocol_name])
                    ),
                    "median_substrate_comparator_absolute_shift_log10": float(
                        np.median(pivot[comparator_name])
                    ),
                    "median_paired_substrate_minus_protocol_shift_log10": float(
                        np.median(paired_difference)
                    ),
                    "paired_median_difference_cluster_ci_low": ci_low,
                    "paired_median_difference_cluster_ci_high": ci_high,
                    "fraction_substrate_shift_larger": float(
                        np.mean(paired_difference > 0)
                    ),
                    "cluster_bootstrap_draws": bootstrap_draws,
                }
            )
    return pd.DataFrame(rows)


def feature_scale_table(X: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "feature": X.columns,
            "n_observed": X.notna().sum().to_numpy(),
            "sd_raw": X.std(ddof=1).to_numpy(),
        }
    )
    ph_sd = float(result.loc[result["feature"] == "ph", "sd_raw"].iloc[0])
    result["sd_relative_to_ph"] = result["sd_raw"] / ph_sd
    return result.sort_values("sd_raw", ascending=False, ignore_index=True)


def donor_geometry_diagnostics(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    mask_fields: Sequence[str],
    *,
    n_splits: int,
    seed: int,
    split_unit: str,
    n_neighbors: int,
    eligible_override: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compare raw- and standardized-space five-neighbour donor sets."""

    eligible = (
        eligible_positions(X, mask_fields)
        if eligible_override is None
        else np.asarray(eligible_override, dtype=int)
    )
    folds = make_cross_imputation_folds(
        eligible, metadata, n_splits=n_splits, seed=seed, split_unit=split_unit
    )
    matrix = X.to_numpy(dtype=float)
    mask_columns = [X.columns.get_loc(column) for column in mask_fields]
    mcoat_column = X.columns.get_loc("Mcoat")
    records: list[dict[str, object]] = []

    for fold_number, (_, test_positions) in enumerate(folds):
        query_rows = eligible[test_positions]
        donor_mask = donor_mask_for_query_groups(metadata, query_rows, split_unit)
        donor_rows = np.flatnonzero(donor_mask)
        donor = matrix[donor_rows]
        query = matrix[query_rows].copy()
        query[:, mask_columns] = np.nan

        raw_distance = nan_euclidean_distances(query, donor)
        geometry_scaler = StandardScaler()
        donor_scaled = geometry_scaler.fit_transform(donor)
        query_scaled = geometry_scaler.transform(query)
        standardized_distance = nan_euclidean_distances(query_scaled, donor_scaled)

        for mask_field in mask_fields:
            field_column = X.columns.get_loc(mask_field)
            valid_donor_positions = np.flatnonzero(np.isfinite(donor[:, field_column]))
            for query_offset, query_row in enumerate(query_rows):
                raw_order = valid_donor_positions[
                    np.argsort(raw_distance[query_offset, valid_donor_positions])[:n_neighbors]
                ]
                standardized_order = valid_donor_positions[
                    np.argsort(
                        standardized_distance[query_offset, valid_donor_positions]
                    )[:n_neighbors]
                ]
                overlap = len(set(raw_order) & set(standardized_order)) / n_neighbors

                mcoat_shares: list[float] = []
                for donor_position in raw_order:
                    jointly_observed = np.isfinite(query[query_offset]) & np.isfinite(
                        donor[donor_position]
                    )
                    squared = np.square(
                        query[query_offset, jointly_observed]
                        - donor[donor_position, jointly_observed]
                    )
                    total = float(np.sum(squared))
                    if jointly_observed[mcoat_column] and total > 0:
                        mcoat_difference = (
                            query[query_offset, mcoat_column]
                            - donor[donor_position, mcoat_column]
                        )
                        mcoat_shares.append(float(mcoat_difference**2 / total))

                records.append(
                    {
                        "fold": fold_number,
                        "row_position": int(query_row),
                        "masked_field": mask_field,
                        "top5_raw_standardized_overlap": overlap,
                        "median_mcoat_share_raw_top5": (
                            float(np.median(mcoat_shares)) if mcoat_shares else np.nan
                        ),
                    }
                )

    detailed = pd.DataFrame(records)
    summary = (
        detailed.groupby("masked_field", as_index=False)
        .agg(
            n_rows=("row_position", "size"),
            mean_top5_raw_standardized_overlap=(
                "top5_raw_standardized_overlap",
                "mean",
            ),
            median_top5_raw_standardized_overlap=(
                "top5_raw_standardized_overlap",
                "median",
            ),
            median_mcoat_share_raw_top5=("median_mcoat_share_raw_top5", "median"),
        )
        .assign(split_unit=split_unit)
    )
    return summary


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    versions = validate_runtime(args.allow_version_mismatch)
    paths, hashes, commit = validate_artifacts(
        repo_root, allow_mismatch=args.allow_artifact_mismatch
    )
    features = read_feature_contract(paths["features_perox.txt"])
    raw, X, coerced_tokens = read_numeric_training_table(
        paths["perox_final.csv"], features
    )
    publication_mapping, publication_qc = recover_publication_groups(
        raw, paths["nanozymes.csv"]
    )
    metadata = raw.copy()
    metadata["publication_id"] = publication_mapping["publication_id"].to_numpy()
    metadata["doi_normalized"] = publication_mapping["doi_normalized"].to_numpy()
    primary_eligible = eligible_positions(X, PROTOCOL_7)
    (
        descriptor_non_sentinel,
        descriptor_sentinel,
        descriptor_sentinel_flag,
        descriptor_audit,
        descriptor_counts,
    ) = descriptor_integrity_diagnostics(X, primary_eligible)
    metadata["substrate_descriptor_sentinel_all_zero"] = (
        descriptor_sentinel_flag.to_numpy()
    )
    primary_publications = metadata.iloc[primary_eligible]["publication_id"].astype(str)
    primary_publication_sizes = primary_publications.value_counts()
    publication_qc["primary_7_field_rows"] = len(primary_eligible)
    publication_qc["primary_7_field_publications"] = int(
        primary_publications.nunique()
    )
    publication_qc["primary_publications_with_multiple_rows"] = int(
        (primary_publication_sizes > 1).sum()
    )
    publication_qc["primary_median_rows_per_publication"] = float(
        primary_publication_sizes.median()
    )
    publication_qc["primary_max_rows_per_publication"] = int(
        primary_publication_sizes.max()
    )

    scaler = joblib.load(paths["Km_scaler_stack_perox.pkl"])
    model = joblib.load(paths["Km_model_stack_perox.pkl"])
    validate_frozen_pipeline(model, scaler, features)

    prediction_tables: list[pd.DataFrame] = []
    geometry_tables: list[pd.DataFrame] = []
    eligible_overrides = {
        "paired_protocol_7_fields_descriptor_non_sentinel": (
            descriptor_non_sentinel
        ),
        "active_comparator_substrate_7_fields_descriptor_non_sentinel": (
            descriptor_non_sentinel
        ),
        "diagnostic_protocol_7_fields_descriptor_sentinel": descriptor_sentinel,
        "diagnostic_substrate_7_fields_descriptor_sentinel": descriptor_sentinel,
    }
    for mask_name, mask_fields in MASK_SETS.items():
        eligible_override = eligible_overrides.get(mask_name)
        for rule in IMPUTATION_RULES:
            eligible, observed_arm, imputed_arm, fold_labels = cross_imputed_arms(
                X,
                metadata,
                mask_fields,
                rule,
                n_splits=args.n_splits,
                seed=args.seed,
                split_unit=args.split_unit,
                n_neighbors=args.n_neighbors,
                eligible_override=eligible_override,
            )
            observed_prediction = predict_log10_km(
                model, scaler, observed_arm, features
            )
            imputed_prediction = predict_log10_km(
                model, scaler, imputed_arm, features
            )
            prediction_tables.append(
                prediction_rows(
                    metadata,
                    eligible,
                    fold_labels,
                    observed_prediction,
                    imputed_prediction,
                    mask_set=mask_name,
                    rule=rule,
                    split_unit=args.split_unit,
                )
            )

        if not args.skip_geometry_diagnostics:
            geometry = donor_geometry_diagnostics(
                X,
                metadata,
                mask_fields,
                n_splits=args.n_splits,
                seed=args.seed,
                split_unit=args.split_unit,
                n_neighbors=args.n_neighbors,
                eligible_override=eligible_override,
            )
            geometry.insert(0, "mask_set", mask_name)
            geometry_tables.append(geometry)

    predictions = pd.concat(prediction_tables, ignore_index=True)
    summary = summarize_predictions(
        predictions, bootstrap_draws=args.bootstrap_draws, seed=args.seed
    )
    rule_comparison = compare_imputation_rules(predictions)
    mask_comparison = compare_protocol_to_substrate_mask(
        predictions, bootstrap_draws=args.bootstrap_draws, seed=args.seed
    )
    scales = feature_scale_table(X)
    geometry_summary = (
        pd.concat(geometry_tables, ignore_index=True)
        if geometry_tables
        else pd.DataFrame()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "internal_prediction_shifts.csv", index=False)
    summary.to_csv(output_dir / "internal_prediction_shift_summary.csv", index=False)
    rule_comparison.to_csv(output_dir / "imputation_rule_comparison.csv", index=False)
    mask_comparison.to_csv(output_dir / "mask_set_comparison.csv", index=False)
    scales.to_csv(output_dir / "feature_scales.csv", index=False)
    publication_mapping.to_csv(output_dir / "publication_mapping_audit.csv", index=False)
    descriptor_audit.to_csv(output_dir / "descriptor_integrity_audit.csv", index=False)
    if not geometry_summary.empty:
        geometry_summary.to_csv(output_dir / "donor_geometry_summary.csv", index=False)

    manifest: dict[str, object] = {
        "analysis": "DiZyme internal mask-imputation model-behavior control",
        "outcome_labels_used_for_prediction_or_eligibility": False,
        "km_used_only_for_publication_mapping_qc": True,
        "audited_commit_expected": AUDITED_COMMIT,
        "repository_commit_observed": commit,
        "artifact_sha256": hashes,
        "runtime": versions,
        "parameters": {
            "seed": args.seed,
            "n_splits": args.n_splits,
            "split_unit": args.split_unit,
            "n_neighbors": args.n_neighbors,
            "cluster_bootstrap_draws": args.bootstrap_draws,
            "twofold_log10_threshold": TWO_FOLD_LOG10,
            "mask_sets": MASK_SETS,
            "imputation_rules": list(IMPUTATION_RULES),
            "query_fold_excluded_from_donor_pool": True,
            "non_masked_values_identical_between_arms": True,
            "common_reference_completion_rule": "raw_knn",
            "primary_split_unit": "publication",
            "active_comparator_primary_stratum": "descriptor_non_sentinel",
            "sentinel_stratum_use": "data_coding_diagnostic_only",
            "frozen_descriptor_values_corrected": False,
        },
        "data_counts": {
            "training_rows": len(X),
            "features": len(features),
            "complete_37_feature_rows": int(X.notna().all(axis=1).sum()),
            "primary_7_field_complete_rows": int(
                X[PROTOCOL_7].notna().all(axis=1).sum()
            ),
            "sensitivity_6_field_complete_rows": int(
                X[PROTOCOL_6].notna().all(axis=1).sum()
            ),
            "active_comparator_rows": int(len(descriptor_non_sentinel)),
            "active_comparator_sentinel_diagnostic_rows": int(
                len(descriptor_sentinel)
            ),
        },
        "descriptor_integrity_audit": descriptor_counts,
        "publication_mapping_qc": publication_qc,
        "coerced_known_missing_tokens": coerced_tokens,
        "interpretation_limit": (
            "D_i is an internal perturbational property of the frozen artifact; it is "
            "not external validation, an error estimate, or a guaranteed lower bound "
            "on external sensitivity."
        ),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Path to the audited acid-design-lab/DiZyme checkout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dizyme_internal_imputation_sensitivity"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--n-neighbors", type=int, default=5)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument(
        "--split-unit",
        choices=("publication", "formula", "row"),
        default="publication",
        help="Primary is publication-grouped; formula and row are sensitivity analyses.",
    )
    parser.add_argument("--skip-geometry-diagnostics", action="store_true")
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="Testing/debugging only; do not use for manuscript results.",
    )
    parser.add_argument(
        "--allow-artifact-mismatch",
        action="store_true",
        help="Testing/debugging only; do not use for manuscript results.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_analysis(args)
    print(json.dumps(manifest["data_counts"], indent=2, sort_keys=True))
    print(f"Results written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
