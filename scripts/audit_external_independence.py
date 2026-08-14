#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import prepare_modeling_data, read_table


USER_AGENT = "nanozymes-ood-benchmark/0.1 (external-independence audit)"


def normalize_doi(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text.strip().rstrip(".,;")


def normalize_title(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def first_nonempty(values: pd.Series) -> str:
    for value in values:
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return ""


def crossref_metadata(doi: str, retries: int = 3) -> dict:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                message = json.load(response)["message"]
            titles = message.get("title") or []
            year = np.nan
            for field in ("published-print", "published-online", "published", "issued", "created"):
                date_parts = (message.get(field) or {}).get("date-parts") or []
                if date_parts and date_parts[0]:
                    year = int(date_parts[0][0])
                    break
            return {
                "doi_normalized": doi,
                "crossref_title": str(titles[0]) if titles else "",
                "crossref_year": year,
                "crossref_status": "resolved",
            }
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError) as exc:
            if attempt == retries - 1:
                return {
                    "doi_normalized": doi,
                    "crossref_title": "",
                    "crossref_year": np.nan,
                    "crossref_status": f"unresolved:{type(exc).__name__}",
                }
            time.sleep(0.5 * (attempt + 1))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal", default="data/nanozymes.csv")
    parser.add_argument(
        "--external",
        default="outputs/tables/external_validation/external_validation_strict_model_ready.csv",
    )
    parser.add_argument(
        "--metadata-cache",
        default="outputs/tables/external_validation/internal_crossref_metadata.csv",
    )
    parser.add_argument(
        "--audit-output",
        default="outputs/tables/external_validation/external_independence_audit.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="outputs/tables/external_validation/external_independence_summary.csv",
    )
    args = parser.parse_args()

    internal, _ = prepare_modeling_data(read_table(args.internal))
    external = pd.read_csv(args.external)
    internal["doi_normalized"] = internal["link"].map(normalize_doi)
    unique_dois = sorted(set(internal["doi_normalized"]) - {""})

    cache_path = Path(args.metadata_cache)
    if cache_path.exists():
        cache = pd.read_csv(cache_path).fillna("")
        cache["doi_normalized"] = cache["doi_normalized"].map(normalize_doi)
    else:
        cache = pd.DataFrame(
            columns=["doi_normalized", "crossref_title", "crossref_year", "crossref_status"]
        )
    cached = set(cache["doi_normalized"].astype(str))
    missing = [doi for doi in unique_dois if doi not in cached]
    if missing:
        resolved = []
        for index, doi in enumerate(missing, start=1):
            resolved.append(crossref_metadata(doi))
            if index % 25 == 0 or index == len(missing):
                print(f"Crossref metadata: {index}/{len(missing)}", flush=True)
            time.sleep(0.05)
        cache = pd.concat([cache, pd.DataFrame(resolved)], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(cache_path, index=False)

    cache["doi_normalized"] = cache["doi_normalized"].astype(str)
    cache["crossref_year"] = pd.to_numeric(cache["crossref_year"], errors="coerce")
    cache.to_csv(cache_path, index=False)
    internal = internal.merge(cache, on="doi_normalized", how="left")

    external["title_normalized_audit"] = external.get(
        "title", pd.Series(index=external.index, dtype=object)
    ).map(normalize_title)
    publication = (
        external.groupby("curation_record_id", dropna=False)
        .agg(
            external_title=("title", first_nonempty),
            external_title_normalized=("title_normalized_audit", first_nonempty),
            external_doi=("doi_normalized", first_nonempty),
            n_external_records=("external_row_id", "size"),
        )
        .reset_index()
    )
    internal_titles = cache.loc[cache["crossref_title"].astype(str).ne("")].copy()
    internal_titles["internal_title_normalized"] = internal_titles["crossref_title"].map(
        normalize_title
    )
    title_to_dois = (
        internal_titles.groupby("internal_title_normalized")["doi_normalized"]
        .agg(lambda values: ";".join(sorted(set(values))))
        .to_dict()
    )
    publication["normalized_title_match_internal"] = publication[
        "external_title_normalized"
    ].map(lambda title: bool(title) and title in title_to_dois)
    publication["matched_internal_dois"] = publication["external_title_normalized"].map(
        title_to_dois
    ).fillna("")

    internal_formulas = set(internal["standardized_formula"].astype(str))
    external_formula = external["standardized_formula"].astype(str)
    unique_external_formulas = set(external_formula)
    row_formula_overlap = external_formula.isin(internal_formulas)

    row_year = pd.to_numeric(internal["crossref_year"], errors="coerce")
    publication_year = (
        internal[["doi_normalized", "crossref_year"]]
        .drop_duplicates("doi_normalized")["crossref_year"]
        .pipe(pd.to_numeric, errors="coerce")
    )
    external_doi_publications = int(
        publication["external_doi"].astype("string").fillna("").str.strip().ne("").sum()
    )
    external_unique_doi_tokens = int(
        publication["external_doi"]
        .astype("string")
        .fillna("")
        .str.strip()
        .loc[lambda values: values.ne("")]
        .nunique()
    )
    summary = pd.DataFrame(
        [
            ("internal_records", len(internal), "modeling rows"),
            ("internal_unique_publications", len(unique_dois), "unique normalized DOI tokens"),
            ("internal_records_missing_year", int(row_year.isna().sum()), "Crossref year unresolved"),
            ("internal_publications_missing_year", int(publication_year.isna().sum()), "Crossref year unresolved"),
            ("internal_record_year_min", float(row_year.min()), "row-weighted Crossref year"),
            ("internal_record_year_median", float(row_year.median()), "row-weighted Crossref year"),
            ("internal_record_year_max", float(row_year.max()), "row-weighted Crossref year"),
            ("internal_publication_year_min", float(publication_year.min()), "unique-publication Crossref year"),
            ("internal_publication_year_median", float(publication_year.median()), "unique-publication Crossref year"),
            ("internal_publication_year_max", float(publication_year.max()), "unique-publication Crossref year"),
            ("internal_records_2024_or_later", int((row_year >= 2024).sum()), "tests temporal separation"),
            ("internal_publications_2024_or_later", int((publication_year >= 2024).sum()), "tests temporal separation"),
            ("external_records", len(external), "strict external rows"),
            ("external_publications", int(publication["curation_record_id"].nunique()), "curated publication units"),
            ("external_publications_with_doi", external_doi_publications, "DOI audit denominator"),
            ("external_unique_doi_tokens", external_unique_doi_tokens, "unique normalized external DOI tokens"),
            (
                "external_publications_with_normalized_title",
                int(publication["external_title_normalized"].astype(str).ne("").sum()),
                "title-audit denominator",
            ),
            (
                "external_publications_matching_internal_title",
                int(publication["normalized_title_match_internal"].sum()),
                "exact normalized-title matches",
            ),
            ("external_records_with_seen_formula", int(row_formula_overlap.sum()), "record-level overlap numerator"),
            ("external_record_formula_overlap_fraction", float(row_formula_overlap.mean()), "record-level overlap"),
            (
                "external_unique_formulas_with_internal_overlap",
                int(len(unique_external_formulas & internal_formulas)),
                "unique-formula overlap numerator",
            ),
            ("external_unique_formulas", int(len(unique_external_formulas)), "unique-formula denominator"),
            (
                "external_unique_formula_overlap_fraction",
                float(len(unique_external_formulas & internal_formulas) / len(unique_external_formulas)),
                "unique-formula overlap",
            ),
        ],
        columns=["diagnostic", "value", "interpretation"],
    )

    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    publication.to_csv(audit_path, index=False)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
