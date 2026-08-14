from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


MISSING_TOKEN = "__MISSING__"
UNKNOWN_TOKEN = "__UNKNOWN__"


@dataclass(frozen=True)
class Element:
    z: int
    mass: float
    electronegativity: float | None
    group: int | None
    period: int
    atomic_radius: float | None
    covalent_radius: float | None


# Lightweight pinned property table. Values are conventional reference values sufficient for
# deterministic descriptors; missing values are handled explicitly and reported.
ELEMENTS: dict[str, Element] = {
    "H": Element(1, 1.008, 2.20, 1, 1, 53, 31),
    "C": Element(6, 12.011, 2.55, 14, 2, 67, 76),
    "N": Element(7, 14.007, 3.04, 15, 2, 56, 71),
    "O": Element(8, 15.999, 3.44, 16, 2, 48, 66),
    "F": Element(9, 18.998, 3.98, 17, 2, 42, 57),
    "Na": Element(11, 22.990, 0.93, 1, 3, 190, 166),
    "Mg": Element(12, 24.305, 1.31, 2, 3, 145, 141),
    "Al": Element(13, 26.982, 1.61, 13, 3, 118, 121),
    "Si": Element(14, 28.085, 1.90, 14, 3, 111, 111),
    "P": Element(15, 30.974, 2.19, 15, 3, 98, 107),
    "S": Element(16, 32.06, 2.58, 16, 3, 88, 105),
    "Cl": Element(17, 35.45, 3.16, 17, 3, 79, 102),
    "K": Element(19, 39.098, 0.82, 1, 4, 243, 203),
    "Ca": Element(20, 40.078, 1.00, 2, 4, 194, 176),
    "Ti": Element(22, 47.867, 1.54, 4, 4, 176, 160),
    "V": Element(23, 50.942, 1.63, 5, 4, 171, 153),
    "Cr": Element(24, 51.996, 1.66, 6, 4, 166, 139),
    "Mn": Element(25, 54.938, 1.55, 7, 4, 161, 139),
    "Fe": Element(26, 55.845, 1.83, 8, 4, 156, 132),
    "Co": Element(27, 58.933, 1.88, 9, 4, 152, 126),
    "Ni": Element(28, 58.693, 1.91, 10, 4, 149, 124),
    "Cu": Element(29, 63.546, 1.90, 11, 4, 145, 132),
    "Zn": Element(30, 65.38, 1.65, 12, 4, 142, 122),
    "Ga": Element(31, 69.723, 1.81, 13, 4, 136, 122),
    "Ge": Element(32, 72.630, 2.01, 14, 4, 125, 120),
    "Se": Element(34, 78.971, 2.55, 16, 4, 103, 120),
    "Br": Element(35, 79.904, 2.96, 17, 4, 94, 120),
    "Sr": Element(38, 87.62, 0.95, 2, 5, 219, 195),
    "Y": Element(39, 88.906, 1.22, 3, 5, 212, 190),
    "Zr": Element(40, 91.224, 1.33, 4, 5, 206, 175),
    "Nb": Element(41, 92.906, 1.60, 5, 5, 198, 164),
    "Mo": Element(42, 95.95, 2.16, 6, 5, 190, 154),
    "Ru": Element(44, 101.07, 2.20, 8, 5, 178, 146),
    "Rh": Element(45, 102.906, 2.28, 9, 5, 173, 142),
    "Pd": Element(46, 106.42, 2.20, 10, 5, 169, 139),
    "Ag": Element(47, 107.868, 1.93, 11, 5, 165, 145),
    "Cd": Element(48, 112.414, 1.69, 12, 5, 161, 144),
    "In": Element(49, 114.818, 1.78, 13, 5, 156, 142),
    "Sn": Element(50, 118.710, 1.96, 14, 5, 145, 139),
    "Sb": Element(51, 121.760, 2.05, 15, 5, 133, 139),
    "Te": Element(52, 127.60, 2.10, 16, 5, 123, 138),
    "I": Element(53, 126.904, 2.66, 17, 5, 115, 139),
    "Ba": Element(56, 137.327, 0.89, 2, 6, 253, 215),
    "La": Element(57, 138.905, 1.10, 3, 6, 195, 207),
    "Ce": Element(58, 140.116, 1.12, 3, 6, 185, 204),
    "Pr": Element(59, 140.908, 1.13, 3, 6, 247, 203),
    "Nd": Element(60, 144.242, 1.14, 3, 6, 206, 201),
    "Sm": Element(62, 150.36, 1.17, 3, 6, 238, 198),
    "Eu": Element(63, 151.964, 1.20, 3, 6, 231, 198),
    "Gd": Element(64, 157.25, 1.20, 3, 6, 233, 196),
    "Tb": Element(65, 158.925, 1.20, 3, 6, 225, 194),
    "Dy": Element(66, 162.500, 1.22, 3, 6, 228, 192),
    "Ho": Element(67, 164.930, 1.23, 3, 6, 226, 192),
    "Er": Element(68, 167.259, 1.24, 3, 6, 226, 189),
    "Yb": Element(70, 173.045, 1.10, 3, 6, 222, 187),
    "Lu": Element(71, 174.967, 1.27, 3, 6, 217, 187),
    "Hf": Element(72, 178.49, 1.30, 4, 6, 208, 175),
    "Ta": Element(73, 180.948, 1.50, 5, 6, 200, 170),
    "W": Element(74, 183.84, 2.36, 6, 6, 193, 162),
    "Re": Element(75, 186.207, 1.90, 7, 6, 188, 151),
    "Os": Element(76, 190.23, 2.20, 8, 6, 185, 144),
    "Ir": Element(77, 192.217, 2.20, 9, 6, 180, 141),
    "Pt": Element(78, 195.084, 2.28, 10, 6, 177, 136),
    "Au": Element(79, 196.967, 2.54, 11, 6, 174, 136),
    "Hg": Element(80, 200.592, 2.00, 12, 6, 171, 132),
    "Pb": Element(82, 207.2, 2.33, 14, 6, 154, 146),
    "Bi": Element(83, 208.980, 2.02, 15, 6, 143, 148),
}


NUMERIC_COLUMNS = [
    "length",
    "width",
    "depth",
    "ph",
    "temperature",
    "c_min",
    "c_max",
    "c_const",
    "vmax_value",
    "ccat_value",
]

CATEGORICAL_COLUMNS = [
    "activity",
    "syngony",
    "surface",
    "reaction_type",
    "km_unit",
    "vmax_unit",
    "c_const_unit",
    "ccat_unit",
]

DESIGN_ONLY_EXCLUDED_COLUMNS = {
    "km_unit",
    "vmax_value",
    "vmax_unit",
}


def normalize_formula_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = text.replace("·", ".").replace("•", ".")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[\+\-]+$", "", text)
    return text


def parse_formula(formula: object) -> dict[str, float]:
    text = normalize_formula_text(formula)
    if not text:
        return {}
    text = re.sub(r"^[A-Za-z0-9_/-]+@", "", text)
    parts = [p for p in re.split(r"[.]", text) if p]
    total: dict[str, float] = defaultdict(float)
    for part in parts:
        multiplier = 1.0
        leading = re.match(r"^([0-9]+(?:\.[0-9]+)?)([A-Z].*)$", part)
        if leading:
            multiplier = float(leading.group(1))
            part = leading.group(2)
        parsed, pos = _parse_segment(part, 0)
        if pos != len(part):
            return {}
        for elem, amount in parsed.items():
            total[elem] += multiplier * amount
    return dict(total)


def _parse_segment(text: str, pos: int) -> tuple[dict[str, float], int]:
    counts: dict[str, float] = defaultdict(float)
    while pos < len(text):
        char = text[pos]
        if char == "(":
            inner, pos = _parse_segment(text, pos + 1)
            multiplier, pos = _read_number(text, pos)
            for elem, amount in inner.items():
                counts[elem] += amount * multiplier
        elif char == ")":
            return dict(counts), pos + 1
        elif char.isupper():
            elem = char
            pos += 1
            if pos < len(text) and text[pos].islower():
                elem += text[pos]
                pos += 1
            amount, pos = _read_number(text, pos)
            counts[elem] += amount
        else:
            return {}, len(text) + 1
    return dict(counts), pos


def _read_number(text: str, pos: int) -> tuple[float, int]:
    match = re.match(r"([0-9]+(?:\.[0-9]+)?)", text[pos:])
    if not match:
        return 1.0, pos
    return float(match.group(1)), pos + len(match.group(1))


def standardize_formula(formula: object) -> str:
    counts = parse_formula(formula)
    if not counts:
        return normalize_formula_text(formula).lower()
    return "".join(f"{elem}{_format_amount(counts[elem])}" for elem in sorted(counts))


def _format_amount(value: float) -> str:
    if math.isclose(value, 1.0):
        return ""
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:g}"


def composition_descriptors(formula: object) -> dict[str, float]:
    counts = parse_formula(formula)
    total = float(sum(counts.values()))
    features: dict[str, float] = {
        "comp_parse_success": float(bool(counts)),
        "comp_num_elements": float(len(counts)),
        "comp_total_atoms": total if total else np.nan,
    }
    for elem in ELEMENTS:
        features[f"frac_{elem}"] = 0.0
    if not counts or total <= 0:
        _add_empty_property_stats(features)
        return features

    fractions = {elem: amount / total for elem, amount in counts.items()}
    for elem, fraction in fractions.items():
        if elem in ELEMENTS:
            features[f"frac_{elem}"] = fraction

    amounts = np.array(list(counts.values()), dtype=float)
    features.update(
        {
            "stoich_min": float(np.min(amounts)),
            "stoich_max": float(np.max(amounts)),
            "stoich_mean": float(np.mean(amounts)),
            "stoich_std": float(np.std(amounts)),
            "stoich_entropy": float(-sum(f * math.log(f) for f in fractions.values() if f > 0)),
            "unknown_element_fraction": float(
                sum(frac for elem, frac in fractions.items() if elem not in ELEMENTS)
            ),
        }
    )
    for prop in [
        "z",
        "mass",
        "electronegativity",
        "group",
        "period",
        "atomic_radius",
        "covalent_radius",
    ]:
        _add_property_stats(features, prop, fractions)
    return features


def _add_empty_property_stats(features: dict[str, float]) -> None:
    for prop in [
        "z",
        "mass",
        "electronegativity",
        "group",
        "period",
        "atomic_radius",
        "covalent_radius",
    ]:
        for stat in ["mean", "min", "max", "std", "coverage"]:
            features[f"{prop}_{stat}"] = np.nan if stat != "coverage" else 0.0
    for name in [
        "stoich_min",
        "stoich_max",
        "stoich_mean",
        "stoich_std",
        "stoich_entropy",
        "unknown_element_fraction",
    ]:
        features[name] = np.nan


def _add_property_stats(features: dict[str, float], prop: str, fractions: dict[str, float]) -> None:
    values = []
    weights = []
    covered = 0.0
    for elem, frac in fractions.items():
        ref = ELEMENTS.get(elem)
        value = getattr(ref, prop) if ref else None
        if value is None:
            continue
        values.append(float(value))
        weights.append(float(frac))
        covered += float(frac)
    if not values:
        for stat in ["mean", "min", "max", "std"]:
            features[f"{prop}_{stat}"] = np.nan
        features[f"{prop}_coverage"] = 0.0
        return
    arr = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    mean = float(np.sum(arr * w))
    features[f"{prop}_mean"] = mean
    features[f"{prop}_min"] = float(np.min(arr))
    features[f"{prop}_max"] = float(np.max(arr))
    features[f"{prop}_std"] = float(np.sqrt(np.sum(w * (arr - mean) ** 2)))
    features[f"{prop}_coverage"] = covered


def build_feature_frame(
    df: pd.DataFrame,
    include_missing_indicators: bool = False,
    exclude_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    excluded = set(exclude_columns or ())
    numeric_cols = [c for c in NUMERIC_COLUMNS if c in df.columns and c not in excluded]
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns and c not in excluded]
    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce") if numeric_cols else pd.DataFrame()
    if include_missing_indicators and numeric_cols:
        indicators = numeric.isna().astype(float).add_prefix("missing_")
        numeric = pd.concat([numeric, indicators], axis=1)

    categoricals = pd.DataFrame(index=df.index)
    for col in categorical_cols:
        categoricals[col] = df[col].astype("string").fillna(MISSING_TOKEN).astype(str)

    comp = pd.DataFrame([composition_descriptors(v) for v in df["standardized_formula"]], index=df.index)
    X = pd.concat([numeric, categoricals, comp], axis=1)
    metadata = {
        "numeric_columns": list(numeric.columns) + list(comp.columns),
        "categorical_columns": categorical_cols,
        "composition_columns": list(comp.columns),
        "excluded_columns": sorted(excluded),
    }
    return X, metadata


def build_design_only_feature_frame(
    df: pd.DataFrame, include_missing_indicators: bool = False
) -> tuple[pd.DataFrame, dict]:
    """Build the prospective schema without Km/Vmax-derived fields."""
    return build_feature_frame(
        df,
        include_missing_indicators=include_missing_indicators,
        exclude_columns=DESIGN_ONLY_EXCLUDED_COLUMNS,
    )


def elemental_coverage_table(formulas: pd.Series) -> pd.DataFrame:
    rows = []
    props = [
        "z",
        "mass",
        "electronegativity",
        "group",
        "period",
        "atomic_radius",
        "covalent_radius",
    ]
    parsed = [parse_formula(v) for v in formulas]
    for prop in props:
        complete = 0
        partial = 0
        missing = 0
        for counts in parsed:
            if not counts:
                missing += 1
                continue
            known = 0
            for elem in counts:
                ref = ELEMENTS.get(elem)
                if ref and getattr(ref, prop) is not None:
                    known += 1
            if known == len(counts):
                complete += 1
            elif known > 0:
                partial += 1
            else:
                missing += 1
        rows.append(
            {
                "elemental_property": prop,
                "formulas_complete_coverage": complete,
                "formulas_partial_or_missing_coverage": partial + missing,
                "fallback_rule": "train-set median imputation for descriptor columns",
            }
        )
    return pd.DataFrame(rows)
