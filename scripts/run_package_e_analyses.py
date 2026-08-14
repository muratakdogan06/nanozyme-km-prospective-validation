#!/usr/bin/env python3
"""Build Package E chemistry-context summaries from frozen Package D inputs.

The material-family comparison is deliberately descriptive. It classifies the
nominal formula strings carried by the two corpora; it does not infer crystal
structure, surface termination, oxidation state, or catalytic-site identity.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import prepare_modeling_data, read_table
from src.features import ELEMENTS


ROOT = Path(__file__).resolve().parents[1]
INTERNAL_PATH = ROOT / "data" / "nanozymes.csv"
EXTERNAL_PATH = (
    ROOT / "outputs" / "tables" / "external_validation"
    / "external_validation_strict_model_ready.csv"
)
BIAS_PATH = (
    ROOT / "outputs" / "tables" / "editor_major_revision"
    / "m5_external_bias_variance_decomposition.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "tables" / "package_e"

ARCHITECTURE_TOKEN_PATTERN = re.compile(r"(?:MOF|COF|ZIF|MIL)")
NONMETALS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "Te", "I"}
METALS = set(ELEMENTS).difference(NONMETALS)
FAMILY_ORDER = [
    "Framework-labeled",
    "C/N-containing",
    "Chalcogenide",
    "Oxide/ferrite",
    "Elemental metal/alloy",
    "Other/mixed",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recognized_element_tokens(value: object) -> set[str]:
    """Return valid element symbols after removing explicit architecture acronyms.

    The scanner treats a valid two-character symbol preferentially, but falls
    back to a valid one-character symbol. This preserves O in labels such as
    ``MoOx`` while preventing ``Ox`` from becoming an unknown pseudo-element.
    """

    text = ARCHITECTURE_TOKEN_PATTERN.sub("", str(value))
    found: set[str] = set()
    position = 0
    while position < len(text):
        character = text[position]
        if character.isupper():
            two_character = ""
            if position + 1 < len(text) and text[position + 1].islower():
                two_character = text[position : position + 2]
            if two_character in ELEMENTS:
                found.add(two_character)
                position += 2
                continue
            if character in ELEMENTS:
                found.add(character)
        position += 1
    return found


def nominal_formula_family(value: object) -> str:
    text = str(value)
    elements = recognized_element_tokens(text)
    if ARCHITECTURE_TOKEN_PATTERN.search(text):
        return "Framework-labeled"
    if elements.intersection({"C", "N"}):
        return "C/N-containing"
    if elements.intersection({"S", "Se", "Te"}):
        return "Chalcogenide"
    if "O" in elements:
        return "Oxide/ferrite"
    if elements and elements.issubset(METALS):
        return "Elemental metal/alloy"
    return "Other/mixed"


def row_assignments(internal: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    internal_rows = internal[
        ["row_id", "source_group_id", "formula", "standardized_formula"]
    ].copy()
    internal_rows.insert(0, "cohort", "Internal, 2007–2023")
    internal_rows = internal_rows.rename(columns={"row_id": "record_id", "source_group_id": "group_id"})

    external_rows = external[
        ["external_row_id", "external_source_cluster_id", "formula", "standardized_formula"]
    ].copy()
    external_rows.insert(0, "cohort", "External, 2024–2026")
    external_rows = external_rows.rename(
        columns={"external_row_id": "record_id", "external_source_cluster_id": "group_id"}
    )

    combined = pd.concat([internal_rows, external_rows], ignore_index=True)
    combined["nominal_formula_family"] = combined["formula"].map(nominal_formula_family)
    combined["recognized_element_tokens"] = combined["formula"].map(
        lambda value: ";".join(sorted(recognized_element_tokens(value)))
    )
    return combined


def family_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    denominators = assignments.groupby("cohort").size().to_dict()
    for family in FAMILY_ORDER:
        record: dict[str, object] = {"nominal_formula_family": family}
        for cohort, prefix in [
            ("Internal, 2007–2023", "internal"),
            ("External, 2024–2026", "external"),
        ]:
            count = int(
                assignments.loc[
                    assignments["cohort"].eq(cohort), "nominal_formula_family"
                ].eq(family).sum()
            )
            record[f"{prefix}_records"] = count
            record[f"{prefix}_percent"] = 100.0 * count / int(denominators[cohort])
        record["external_minus_internal_percentage_points"] = (
            float(record["external_percent"]) - float(record["internal_percent"])
        )
        rows.append(record)
    summary = pd.DataFrame(rows)
    if int(summary["internal_records"].sum()) != 992:
        raise AssertionError("Internal family counts do not sum to 992")
    if int(summary["external_records"].sum()) != 322:
        raise AssertionError("External family counts do not sum to 322")
    return summary


def element_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    internal = assignments.loc[assignments["cohort"].eq("Internal, 2007–2023")]
    external = assignments.loc[assignments["cohort"].eq("External, 2024–2026")]
    rows: list[dict[str, object]] = []
    for element in ELEMENTS:
        internal_count = int(
            internal["recognized_element_tokens"].str.split(";").map(lambda x: element in x).sum()
        )
        external_count = int(
            external["recognized_element_tokens"].str.split(";").map(lambda x: element in x).sum()
        )
        rows.append(
            {
                "element": element,
                "internal_records": internal_count,
                "internal_percent": 100.0 * internal_count / len(internal),
                "external_records": external_count,
                "external_percent": 100.0 * external_count / len(external),
                "external_minus_internal_percentage_points": (
                    100.0 * external_count / len(external)
                    - 100.0 * internal_count / len(internal)
                ),
                "pooled_records": internal_count + external_count,
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["pooled_records", "element"], ascending=[False, True], kind="mergesort"
    )
    summary.insert(0, "pooled_prevalence_rank", range(1, len(summary) + 1))
    return summary.head(10).reset_index(drop=True)


def chemistry_context_summary() -> pd.DataFrame:
    bias = pd.read_csv(BIAS_PATH)
    selected = bias.loc[bias["stratum"].isin(["pooled", "H2O2", "TMB"])].copy()
    selected["geometric_prediction_observation_ratio"] = 10.0 ** selected["signed_bias"]
    return selected[
        [
            "stratum",
            "n_rows",
            "n_publication_clusters",
            "signed_bias",
            "signed_bias_ci95_low",
            "signed_bias_ci95_high",
            "geometric_prediction_observation_ratio",
        ]
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    internal, _ = prepare_modeling_data(read_table(INTERNAL_PATH))
    external = pd.read_csv(EXTERNAL_PATH)
    if len(internal) != 992 or len(external) != 322:
        raise AssertionError(f"Unexpected cohort sizes: {len(internal)}, {len(external)}")

    assignments = row_assignments(internal, external)
    families = family_summary(assignments)
    elements = element_summary(assignments)
    chemistry = chemistry_context_summary()

    assignments.to_csv(OUTPUT_DIR / "package_e_nominal_formula_family_assignments.csv", index=False)
    families.to_csv(OUTPUT_DIR / "package_e_nominal_formula_family_summary.csv", index=False)
    elements.to_csv(OUTPUT_DIR / "package_e_element_token_prevalence.csv", index=False)
    chemistry.to_csv(OUTPUT_DIR / "package_e_h2o2_bias_chemistry_context.csv", index=False)

    manifest = {
        "package": "E",
        "purpose": "Descriptive chemistry-context analysis; no causal mechanism test",
        "inputs": {
            str(INTERNAL_PATH.relative_to(ROOT)): sha256(INTERNAL_PATH),
            str(EXTERNAL_PATH.relative_to(ROOT)): sha256(EXTERNAL_PATH),
            str(BIAS_PATH.relative_to(ROOT)): sha256(BIAS_PATH),
        },
        "cohort_sizes": {"internal": len(internal), "external": len(external)},
        "family_priority": FAMILY_ORDER,
        "architecture_tokens_removed_before_element_scanning": ["MOF", "COF", "ZIF", "MIL"],
        "interpretation": (
            "Counts describe nominal formula labels at record level. They include paired substrate "
            "rows and are not independent material counts or definitive structural assignments."
        ),
    }
    (OUTPUT_DIR / "package_e_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("Nominal formula families:\n", families.to_string(index=False))
    print("\nTop element tokens:\n", elements.to_string(index=False))
    print("\nBias chemistry context:\n", chemistry.to_string(index=False))


if __name__ == "__main__":
    main()
