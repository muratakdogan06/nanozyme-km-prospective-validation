from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .features import standardize_formula


SOURCE_COLUMNS = [
    "doi",
    "link",
    "url",
    "source",
    "article",
    "paper",
    "title",
    "journal",
    "pdf",
    "filename",
    "file",
]


def read_table(path: str | Path) -> pd.DataFrame:
    raw_path = str(path).strip()
    if not raw_path:
        raise ValueError(
            'Data path is empty. If using --data "$DATA", set DATA first '
            "or pass a file path such as --data data/nanozymes.csv."
        )
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.is_dir():
        raise ValueError(f"Data path is a directory, not a table file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(
        f"Unsupported data extension for {path}: {suffix or '<none>'}. "
        "Supported extensions are .csv, .tsv, .txt, .xlsx, .xls, and .parquet."
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        re.sub(r"[^0-9a-zA-Z]+", "_", str(c).strip().lower()).strip("_") for c in out.columns
    ]
    has_km_mm = "km_mm" in out.columns
    has_vmax_mm_s = "vmax_mm_s" in out.columns
    has_c_const_mm = "c_const_mm" in out.columns
    has_ccat_mg_ml = "ccat_mg_ml" in out.columns
    aliases = {
        "km": "km_value",
        "kmval": "km_value",
        "k_m": "km_value",
        "km_mm": "km_value",
        "km_units": "km_unit",
        "reaction": "reaction_type",
        "reactiontype": "reaction_type",
        "ph_value": "ph",
        "temp": "temperature",
        "temp_c": "temperature",
        "temperature_c": "temperature",
        "formula_pretty": "formula",
        "chemical_formula": "formula",
        "coating": "surface",
        "length_nm": "length",
        "width_nm": "width",
        "depth_nm": "depth",
        "vmax_mm_s": "vmax_value",
        "c_min_mm": "c_min",
        "c_max_mm": "c_max",
        "c_const_mm": "c_const",
        "ccat_mg_ml": "ccat_value",
    }
    out = out.rename(columns={c: aliases[c] for c in out.columns if c in aliases})
    if has_km_mm and "km_unit" not in out.columns:
        out["km_unit"] = "mM"
    if has_vmax_mm_s and "vmax_unit" not in out.columns:
        out["vmax_unit"] = "mM/s"
    if has_c_const_mm and "c_const_unit" not in out.columns:
        out["c_const_unit"] = "mM"
    if has_ccat_mg_ml and "ccat_unit" not in out.columns:
        out["ccat_unit"] = "mg/mL"
    return out


def prepare_modeling_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_columns(raw)
    log_rows = []
    log_rows.append({"step": "raw_rows", "rows": len(df), "dropped": 0, "reason": "input"})

    if "activity" in df.columns:
        activity = df["activity"].astype("string").str.lower()
        keep = activity.str.contains("peroxidase|pod|peroxid", na=False)
        df = df.loc[keep].copy()
        log_rows.append(
            {
                "step": "retain_peroxidase_like",
                "rows": len(df),
                "dropped": int((~keep).sum()),
                "reason": "activity label must indicate peroxidase-like behavior",
            }
        )
    else:
        log_rows.append(
            {
                "step": "retain_peroxidase_like",
                "rows": len(df),
                "dropped": 0,
                "reason": "activity column absent; no activity filtering applied",
            }
        )

    for required in ["formula", "km_value"]:
        if required not in df.columns:
            raise ValueError(f"Required column missing after normalization: {required}")

    df["km_common_molar"] = convert_km_to_molar(df["km_value"], df.get("km_unit"))
    valid = np.isfinite(df["km_common_molar"]) & (df["km_common_molar"] > 0)
    dropped = int((~valid).sum())
    df = df.loc[valid].copy()
    log_rows.append(
        {
            "step": "valid_km",
            "rows": len(df),
            "dropped": dropped,
            "reason": "Km must convert to positive molar concentration",
        }
    )

    df["target_log10_km"] = np.log10(df["km_common_molar"].astype(float))
    if "reaction_type" in df.columns:
        df["varied_substrate"] = df["reaction_type"].map(varied_substrate_from_reaction)
        df["canonical_reaction_system"] = df["reaction_type"].map(
            canonical_reaction_system
        )
    df["standardized_formula"] = df["formula"].map(standardize_formula)
    df["formula_group_id"] = df["standardized_formula"].replace("", "__NO_FORMULA__")
    df["source_group_id"] = build_source_group_id(df)
    df["row_id"] = np.arange(len(df), dtype=int)
    log_rows.append(
        {
            "step": "final_modeling_rows",
            "rows": len(df),
            "dropped": 0,
            "reason": "final cohort after deterministic target cleaning",
        }
    )
    return df.reset_index(drop=True), pd.DataFrame(log_rows)


def varied_substrate_from_reaction(value: object) -> str:
    """Return the first reaction token, which denotes the substrate varied for Km."""
    if pd.isna(value):
        return "OTHER"
    tokens = [
        token.strip().upper()
        for token in re.split(r"\s*\+\s*|/", str(value))
        if token.strip()
    ]
    return tokens[0] if tokens else "OTHER"


def canonical_reaction_system(value: object) -> str:
    """Return an order-invariant reaction-pair label without losing varied_substrate."""
    if pd.isna(value):
        return "OTHER"
    tokens = [
        token.strip().upper()
        for token in re.split(r"\s*\+\s*|/", str(value))
        if token.strip()
    ]
    return " + ".join(sorted(tokens)) if tokens else "OTHER"


def convert_km_to_molar(values: pd.Series, units: pd.Series | None) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if units is None:
        return numeric
    unit = (
        units.astype("string")
        .str.lower()
        .str.replace("μ", "u", regex=False)
        .str.replace("µ", "u", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("−", "-", regex=False)
        .str.strip()
    )
    unit = unit.str.replace("moll-1", "mol/l", regex=False)
    unit = unit.str.replace("mol/liter", "mol/l", regex=False)
    unit = unit.str.replace("mmoll-1", "mmol/l", regex=False)
    unit = unit.str.replace("umoll-1", "umol/l", regex=False)
    unit = unit.str.replace("nmoll-1", "nmol/l", regex=False)
    factors = {
        "m": 1.0,
        "mol/l": 1.0,
        "mol/litre": 1.0,
        "mol/liter": 1.0,
        "molar": 1.0,
        "mm": 1e-3,
        "mmol/l": 1e-3,
        "mmolar": 1e-3,
        "millimolar": 1e-3,
        "um": 1e-6,
        "umol/l": 1e-6,
        "umolar": 1e-6,
        "micromolar": 1e-6,
        "nm": 1e-9,
        "nmol/l": 1e-9,
        "nmolar": 1e-9,
        "nanomolar": 1e-9,
    }
    factor = unit.map(factors)
    return numeric * factor


def build_source_group_id(df: pd.DataFrame) -> pd.Series:
    candidates = [c for c in SOURCE_COLUMNS if c in df.columns]
    if not candidates:
        return pd.Series([f"row_source_{i}" for i in range(len(df))], index=df.index)
    values = df[candidates].astype("string").fillna("").agg(" | ".join, axis=1)
    ids = []
    for idx, value in values.items():
        cleaned = re.sub(r"\s+", " ", value.strip().lower())
        if not cleaned or set(cleaned) <= {"|", " "}:
            ids.append(f"missing_source_{idx}")
        else:
            ids.append(hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:12])
    return pd.Series(ids, index=df.index)


def audit_data(raw: pd.DataFrame, modeling: pd.DataFrame, filter_log: pd.DataFrame) -> dict:
    missingness = raw.isna().mean().sort_values(ascending=False).reset_index()
    missingness.columns = ["column", "missing_fraction"]
    reaction_dist = (
        modeling.get("reaction_type", pd.Series(dtype=object)).value_counts(dropna=False).head(50)
    )
    formula_freq = modeling["standardized_formula"].value_counts().reset_index()
    formula_freq.columns = ["standardized_formula", "n_rows"]
    source_freq = modeling["source_group_id"].value_counts().reset_index()
    source_freq.columns = ["source_group_id", "n_rows"]
    components = source_formula_components(modeling)
    duplicates = near_duplicate_clusters(modeling)
    protocol_counts = protocol_variation_counts(modeling)
    return {
        "summary": {
            "raw_rows": int(len(raw)),
            "final_modeling_rows": int(len(modeling)),
            "unique_source_groups": int(modeling["source_group_id"].nunique()),
            "unique_standardized_formulas": int(modeling["standardized_formula"].nunique()),
            "dominant_reaction_system": str(reaction_dist.index[0]) if len(reaction_dist) else "",
            "largest_source_formula_component_rows": int(
                components["component_n_rows"].max() if len(components) else 0
            ),
            "near_duplicate_clusters": int((duplicates["cluster_size"] > 1).sum())
            if len(duplicates)
            else 0,
        },
        "missingness": missingness,
        "reaction_distribution": reaction_dist.reset_index(name="n_rows").rename(
            columns={"index": "reaction_type"}
        ),
        "formula_frequency": formula_freq,
        "source_frequency": source_freq,
        "source_formula_components": components,
        "near_duplicate_clusters": duplicates,
        "protocol_variation_counts": protocol_counts,
        "filter_log": filter_log,
    }


def source_formula_components(df: pd.DataFrame) -> pd.DataFrame:
    graph = nx.Graph()
    for _, row in df.iterrows():
        source = f"s:{row['source_group_id']}"
        formula = f"f:{row['standardized_formula']}"
        graph.add_edge(source, formula)
    rows = []
    for cid, nodes in enumerate(nx.connected_components(graph)):
        sources = [n[2:] for n in nodes if n.startswith("s:")]
        formulas = [n[2:] for n in nodes if n.startswith("f:")]
        mask = df["source_group_id"].isin(sources) & df["standardized_formula"].isin(formulas)
        rows.append(
            {
                "component_id": cid,
                "n_sources": len(sources),
                "n_formulas": len(formulas),
                "component_n_rows": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("component_n_rows", ascending=False)


def near_duplicate_clusters(df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "source_group_id",
        "standardized_formula",
        "activity",
        "reaction_type",
        "ph",
        "temperature",
        "length",
        "width",
        "depth",
        "surface",
        "c_min",
        "c_max",
        "c_const",
    ]
    available = [c for c in keys if c in df.columns]
    if not available:
        return pd.DataFrame()
    temp = df[available].copy()
    for col in temp.columns:
        if pd.api.types.is_numeric_dtype(temp[col]):
            temp[col] = pd.to_numeric(temp[col], errors="coerce").round(3)
        else:
            temp[col] = temp[col].astype("string").fillna("__MISSING__").str.lower().str.strip()
    grouped = temp.groupby(available, dropna=False).size().reset_index(name="cluster_size")
    return grouped[grouped["cluster_size"] > 1].sort_values("cluster_size", ascending=False)


def protocol_variation_counts(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["standardized_formula", "source_group_id"]
    rows = []
    for col in ["ph", "temperature", "c_const"]:
        if col not in df.columns:
            continue
        grouped = df.groupby(group_cols)[col].nunique(dropna=True)
        rows.append(
            {
                "protocol_field": col,
                "groups_with_multiple_values": int((grouped > 1).sum()),
                "max_values_within_group": int(grouped.max()) if len(grouped) else 0,
            }
        )
    return pd.DataFrame(rows)


def write_audit_outputs(audit: dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, value in audit.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output_dir / f"{key}.csv", index=False)
    serializable = {
        key: value for key, value in audit.items() if not isinstance(value, pd.DataFrame)
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(serializable, indent=2))
    summary = audit["summary"]
    lines = ["# Data Audit Summary", ""]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Filtering log is saved in `filter_log.csv`; no near-duplicates are removed by default.")
    (output_dir / "audit_summary.md").write_text("\n".join(lines) + "\n")
