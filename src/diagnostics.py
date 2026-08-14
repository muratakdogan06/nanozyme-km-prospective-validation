from __future__ import annotations

import math
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROTOCOL_CATEGORICAL_COLUMNS = [
    "activity",
    "reaction_type",
    "syngony",
    "surface",
]

PROTOCOL_NUMERIC_COLUMNS = [
    "ph",
    "temperature",
    "c_min",
    "c_max",
    "c_const",
    "ccat_value",
]


def ood_severity_row(
    df: pd.DataFrame,
    X: pd.DataFrame,
    composition_columns: list[str],
    train_idx,
    test_idx,
    split_type: str,
    repeat: int,
    seed: int,
) -> dict:
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    source_unseen = _unseen_fraction(df, train_idx, test_idx, "source_group_id")
    formula_unseen = _unseen_fraction(df, train_idx, test_idx, "formula_group_id")
    protocol_categorical_unseen = _mean_unseen_fraction(
        df, train_idx, test_idx, PROTOCOL_CATEGORICAL_COLUMNS
    )
    protocol_numeric_shift = _mean_standardized_numeric_shift(
        df, train_idx, test_idx, PROTOCOL_NUMERIC_COLUMNS
    )
    comp = _composition_shift(X, composition_columns, train_idx, test_idx)
    bounded_components = [
        source_unseen,
        formula_unseen,
        protocol_categorical_unseen,
        _squash(protocol_numeric_shift),
        _squash(comp["composition_nearest_train_distance_mean"]),
    ]
    source_formula_free_components = [
        protocol_categorical_unseen,
        _squash(protocol_numeric_shift),
        _squash(comp["composition_nearest_train_distance_mean"]),
    ]
    graph_free_components = [
        _squash(comp["composition_nearest_train_distance_mean"]),
        _squash(comp["composition_centroid_distance"]),
    ]
    return {
        "repeat": repeat,
        "seed": seed,
        "split_type": split_type,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "unseen_source_fraction": source_unseen,
        "unseen_formula_fraction": formula_unseen,
        "unseen_protocol_category_fraction": protocol_categorical_unseen,
        "protocol_numeric_shift": protocol_numeric_shift,
        **comp,
        "ood_severity_index": float(np.nanmean(bounded_components)),
        "source_formula_free_severity_index": float(np.nanmean(source_formula_free_components)),
        "graph_free_composition_severity_index": float(np.nanmean(graph_free_components)),
        "severity_definition": (
            "mean of unseen source, unseen formula, unseen protocol categories, "
            "bounded protocol numeric shift, and bounded nearest-train composition distance"
        ),
        "source_formula_free_severity_definition": (
            "mean of unseen protocol categories, bounded protocol numeric shift, "
            "and bounded nearest-train composition distance"
        ),
        "graph_free_composition_severity_definition": (
            "mean of bounded nearest-train and centroid composition distances only"
        ),
    }


def generalization_stress_index(metrics: pd.DataFrame) -> pd.DataFrame:
    random = metrics[metrics["split_type"] == "random"].set_index(["model", "repeat"])
    rows = []
    for _, row in metrics[metrics["split_type"] != "random"].iterrows():
        key = (row["model"], row["repeat"])
        if key not in random.index:
            continue
        ref = random.loc[key]
        delta_mae = float(row["mae"] - ref["mae"])
        delta_r2 = float(ref["r2"] - row["r2"])
        rows.append(
            {
                "repeat": int(row["repeat"]),
                "seed": int(row["seed"]),
                "model": row["model"],
                "split_type": row["split_type"],
                "mae_random": float(ref["mae"]),
                "mae_ood": float(row["mae"]),
                "generalization_stress_index": delta_mae,
                "relative_generalization_stress_index": delta_mae / max(abs(float(ref["mae"])), 1e-12),
                "r2_drop": delta_r2,
                "interpretation": "positive values indicate degradation relative to paired random split",
            }
        )
    return pd.DataFrame(rows)


def severity_performance_association(severity: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    joined = metrics.merge(
        severity[["repeat", "seed", "split_type", "ood_severity_index"]],
        on=["repeat", "seed", "split_type"],
        how="left",
    )
    rows = []
    for model, group in joined.groupby("model", dropna=False):
        for metric in ["mae", "rmse", "r2", "spearman"]:
            valid = group[["ood_severity_index", metric]].dropna()
            statistic = np.nan
            p_value = np.nan
            if len(valid) > 2 and valid["ood_severity_index"].nunique() > 1 and valid[metric].nunique() > 1:
                result = spearmanr(valid["ood_severity_index"], valid[metric])
                statistic = float(result.statistic) if np.isfinite(result.statistic) else np.nan
                p_value = float(result.pvalue) if np.isfinite(result.pvalue) else np.nan
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_points": int(len(valid)),
                    "spearman_r": statistic,
                    "spearman_p": p_value,
                    "interpretation": "association between OOD severity index and benchmark metric",
                }
            )
    return pd.DataFrame(rows)


def split_overlap_diagnostics(df: pd.DataFrame, repeat_tests: list[dict]) -> pd.DataFrame:
    rows = []
    for item in repeat_tests:
        rep = item["repeat"]
        seed = item["seed"]
        tests = item["tests"]
        if "leakage_graph" not in tests:
            continue
        leakage_rows = _row_id_set(df, tests["leakage_graph"])
        source_rows = _row_id_set(df, tests.get("source_group", []))
        formula_rows = _row_id_set(df, tests.get("formula_group", []))
        composition_rows = _row_id_set(df, tests.get("composition_cluster", []))
        random_rows = _row_id_set(df, tests.get("random", []))
        source_formula_union = source_rows | formula_rows
        source_formula_intersection = source_rows & formula_rows
        rows.append(
            {
                "repeat": rep,
                "seed": seed,
                "leakage_graph_test_n": len(leakage_rows),
                "source_group_test_n": len(source_rows),
                "formula_group_test_n": len(formula_rows),
                "source_formula_union_n": len(source_formula_union),
                "source_formula_intersection_n": len(source_formula_intersection),
                "jaccard_leakage_graph_vs_source_group": _jaccard(leakage_rows, source_rows),
                "jaccard_leakage_graph_vs_formula_group": _jaccard(leakage_rows, formula_rows),
                "jaccard_leakage_graph_vs_source_formula_union": _jaccard(
                    leakage_rows, source_formula_union
                ),
                "jaccard_leakage_graph_vs_composition_cluster": _jaccard(
                    leakage_rows, composition_rows
                ),
                "jaccard_leakage_graph_vs_random": _jaccard(leakage_rows, random_rows),
                "fraction_leakage_graph_test_outside_source_formula_union": _fraction(
                    leakage_rows - source_formula_union, leakage_rows
                ),
                "fraction_source_formula_union_captured_by_leakage_graph": _fraction(
                    source_formula_union & leakage_rows, source_formula_union
                ),
                "interpretation": (
                    "low Jaccard and nonzero outside-union fraction indicate graph component holdout "
                    "is not reducible to the paired source/formula holdout selections"
                ),
            }
        )
    return pd.DataFrame(rows)


def split_repeat_variability(df: pd.DataFrame, repeat_tests: list[dict]) -> pd.DataFrame:
    by_split: dict[str, list[tuple[int, set[int]]]] = defaultdict(list)
    all_rows = set(df["row_id"].astype(int))
    for item in repeat_tests:
        for split_type, idx in item["tests"].items():
            by_split[split_type].append((int(item["repeat"]), _row_id_set(df, idx)))

    rows = []
    for split_type, sets in sorted(by_split.items()):
        pairwise = []
        for i, (rep_a, rows_a) in enumerate(sets):
            for rep_b, rows_b in sets[i + 1 :]:
                pairwise.append(
                    {
                        "repeat_a": rep_a,
                        "repeat_b": rep_b,
                        "jaccard": _jaccard(rows_a, rows_b),
                        "overlap_fraction_of_smaller_test": _fraction(
                            rows_a & rows_b, rows_a if len(rows_a) <= len(rows_b) else rows_b
                        ),
                    }
                )
        union = set().union(*(rows for _, rows in sets)) if sets else set()
        test_sizes = [len(rows) for _, rows in sets]
        rows.append(
            {
                "split_type": split_type,
                "n_repeats": len(sets),
                "mean_test_n": float(np.mean(test_sizes)) if test_sizes else np.nan,
                "min_test_n": int(np.min(test_sizes)) if test_sizes else 0,
                "max_test_n": int(np.max(test_sizes)) if test_sizes else 0,
                "unique_test_rows_across_repeats": len(union),
                "fraction_dataset_tested_across_repeats": len(union) / max(1, len(all_rows)),
                "mean_pairwise_test_jaccard": float(np.mean([p["jaccard"] for p in pairwise]))
                if pairwise
                else np.nan,
                "max_pairwise_test_jaccard": float(np.max([p["jaccard"] for p in pairwise]))
                if pairwise
                else np.nan,
                "mean_overlap_fraction_of_smaller_test": float(
                    np.mean([p["overlap_fraction_of_smaller_test"] for p in pairwise])
                )
                if pairwise
                else np.nan,
                "interpretation": (
                    "high pairwise Jaccard indicates split repeats reuse similar held-out rows; "
                    "low values indicate split-source variability"
                ),
            }
        )
    return pd.DataFrame(rows)


def source_formula_protocol_graph_diagnostics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    graph, row_nodes = _source_formula_protocol_graph(df)
    sf_graph, sf_row_nodes = _source_formula_graph(df)
    bridge_summary = _protocol_bridge_summary(row_nodes, sf_graph, sf_row_nodes)
    components = list(nx.connected_components(graph))
    node_to_component = {}
    for component_id, nodes in enumerate(components):
        for node in nodes:
            node_to_component[node] = component_id

    component_rows = defaultdict(int)
    for nodes in row_nodes:
        component_ids = [node_to_component[node] for node in nodes]
        # Each row contributes a source, formula, and protocol node; the graph
        # construction connects those nodes, so all three map to one component.
        component_rows[component_ids[0]] += 1

    detail_rows = []
    for component_id, nodes in enumerate(components):
        node_types = [graph.nodes[n].get("node_type", "unknown") for n in nodes]
        detail_rows.append(
            {
                "component_id": component_id,
                "n_nodes": len(nodes),
                "n_sources": node_types.count("source"),
                "n_formulas": node_types.count("formula"),
                "n_protocols": node_types.count("protocol"),
                "n_rows": component_rows.get(component_id, 0),
            }
        )
    components_df = pd.DataFrame(detail_rows).sort_values(
        ["n_rows", "n_nodes"], ascending=False
    )

    node_type_counts = pd.Series(nx.get_node_attributes(graph, "node_type")).value_counts()
    largest_rows = int(components_df["n_rows"].max()) if len(components_df) else 0
    summary = pd.DataFrame(
        [
            {
                "n_rows": int(len(df)),
                "n_nodes": int(graph.number_of_nodes()),
                "n_edges": int(graph.number_of_edges()),
                "n_source_nodes": int(node_type_counts.get("source", 0)),
                "n_formula_nodes": int(node_type_counts.get("formula", 0)),
                "n_protocol_nodes": int(node_type_counts.get("protocol", 0)),
                "n_connected_components": int(len(components)),
                "largest_component_rows": largest_rows,
                "largest_component_row_fraction": largest_rows / max(1, len(df)),
                **bridge_summary,
                "methodological_use": (
                    "tripartite source-formula-protocol graph supports a component-held-out "
                    "split and quantifies hidden dependency structure beyond row-level checks"
                ),
                "protocol_topology_interpretation": _protocol_bridge_interpretation(
                    bridge_summary
                ),
            }
        ]
    )
    return summary, components_df


def protocol_granularity_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Compare graph connectivity under alternative protocol-token resolutions."""
    schemes = [
        ("three_significant_figures", _format_numeric_protocol_value),
        ("coarse_ph0.5_temp5_two_sigfig", _coarse_protocol_formatter),
        ("very_coarse_ph1_temp10_one_sigfig", _very_coarse_protocol_formatter),
    ]
    sf_graph, sf_row_nodes = _source_formula_graph(df)
    sf_summary = _graph_component_summary(sf_graph, sf_row_nodes)
    rows = []
    for scheme_name, formatter in schemes:
        graph, row_nodes = _source_formula_protocol_graph(df, numeric_formatter=formatter)
        summary = _graph_component_summary(graph, row_nodes)
        merge = _source_formula_merge_summary(df, graph, row_nodes, sf_row_nodes)
        protocol_counts = _protocol_node_counts(row_nodes)
        protocol_bridges = _protocol_bridge_summary(row_nodes, sf_graph, sf_row_nodes)
        rows.append(
            {
                "protocol_tokenization": scheme_name,
                "n_rows": int(len(df)),
                "n_source_formula_components": int(sf_summary["n_connected_components"]),
                "source_formula_largest_component_rows": int(sf_summary["largest_component_rows"]),
                "n_tripartite_components": int(summary["n_connected_components"]),
                "tripartite_largest_component_rows": int(summary["largest_component_rows"]),
                "component_count_delta_vs_source_formula": int(
                    summary["n_connected_components"] - sf_summary["n_connected_components"]
                ),
                "largest_component_delta_vs_source_formula": int(
                    summary["largest_component_rows"] - sf_summary["largest_component_rows"]
                ),
                "n_protocol_nodes": int(protocol_counts["n_protocol_nodes"]),
                "n_shared_protocol_nodes": int(protocol_counts["n_shared_protocol_nodes"]),
                "max_rows_per_protocol_node": int(protocol_counts["max_rows_per_protocol_node"]),
                **protocol_bridges,
                **merge,
            }
        )
    out = pd.DataFrame(rows)
    out["interpretation"] = (
        "Sensitivity diagnostic only: the benchmark uses three-significant-figure protocol tokens; "
        "coarser rows test whether protocol-context granularity changes component topology."
    )
    return out


def external_ood_severity_rows(
    train_df: pd.DataFrame,
    external_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_external: pd.DataFrame,
    composition_columns: list[str],
) -> pd.DataFrame:
    """Score each external row against the internal training set before labels are inspected."""
    if len(external_df) == 0:
        return pd.DataFrame()

    source_unseen = _row_unseen_indicator(train_df, external_df, "source_group_id")
    formula_unseen = _row_unseen_indicator(train_df, external_df, "formula_group_id")
    protocol_category = _row_protocol_category_unseen_fraction(
        train_df, external_df, PROTOCOL_CATEGORICAL_COLUMNS
    )
    protocol_numeric = _row_protocol_numeric_shift(
        train_df, external_df, PROTOCOL_NUMERIC_COLUMNS
    )
    comp = _row_composition_shift(X_train, X_external, composition_columns)

    bounded_numeric = protocol_numeric.map(_squash)
    bounded_nearest = comp["composition_nearest_train_distance"].map(_squash)
    bounded_centroid = comp["composition_centroid_distance"].map(_squash)
    severity = pd.DataFrame(
        {
            "row_id": external_df.get(
                "row_id", pd.Series(np.arange(len(external_df)), index=external_df.index)
            ).to_numpy(),
            "unseen_source": source_unseen.to_numpy(dtype=float),
            "unseen_formula": formula_unseen.to_numpy(dtype=float),
            "unseen_protocol_category_fraction": protocol_category.to_numpy(dtype=float),
            "protocol_numeric_shift": protocol_numeric.to_numpy(dtype=float),
            "composition_nearest_train_distance": comp[
                "composition_nearest_train_distance"
            ].to_numpy(dtype=float),
            "composition_centroid_distance": comp["composition_centroid_distance"].to_numpy(
                dtype=float
            ),
        },
        index=external_df.index,
    )
    severity["ood_severity_index"] = np.nanmean(
        np.column_stack(
            [
                severity["unseen_source"],
                severity["unseen_formula"],
                severity["unseen_protocol_category_fraction"],
                bounded_numeric,
                bounded_nearest,
            ]
        ),
        axis=1,
    )
    severity["source_formula_free_severity_index"] = np.nanmean(
        np.column_stack(
            [
                severity["unseen_protocol_category_fraction"],
                bounded_numeric,
                bounded_nearest,
            ]
        ),
        axis=1,
    )
    severity["graph_free_composition_severity_index"] = np.nanmean(
        np.column_stack([bounded_nearest, bounded_centroid]),
        axis=1,
    )
    severity["severity_reference"] = (
        "computed for each external row against the full internal training set before "
        "external labels or errors are used"
    )
    return severity.reset_index(drop=True)


def severity_error_association(
    predictions: pd.DataFrame,
    error_col: str = "abs_error_log10_km",
    severity_cols: list[str] | None = None,
) -> pd.DataFrame:
    if severity_cols is None:
        severity_cols = [
            "ood_severity_index",
            "source_formula_free_severity_index",
            "graph_free_composition_severity_index",
        ]
    rows = []
    for model, group in predictions.groupby("model", dropna=False):
        for severity_col in severity_cols:
            if severity_col not in group.columns or error_col not in group.columns:
                continue
            valid = group[[severity_col, error_col]].dropna()
            statistic = np.nan
            p_value = np.nan
            if (
                len(valid) > 2
                and valid[severity_col].nunique() > 1
                and valid[error_col].nunique() > 1
            ):
                result = spearmanr(valid[severity_col], valid[error_col])
                statistic = (
                    float(result.statistic) if np.isfinite(result.statistic) else np.nan
                )
                p_value = float(result.pvalue) if np.isfinite(result.pvalue) else np.nan
            rows.append(
                {
                    "model": model,
                    "severity_score": severity_col,
                    "error_metric": error_col,
                    "n_external_rows": int(len(valid)),
                    "spearman_r": statistic,
                    "spearman_p": p_value,
                    "interpretation": (
                        "external validation of whether label-free severity scores "
                        "rank true external prediction errors"
                    ),
                }
            )
    return pd.DataFrame(rows)


def source_formula_protocol_component_ids(df: pd.DataFrame) -> pd.Series:
    graph, row_nodes = _source_formula_protocol_graph(df)
    node_to_component = {}
    for component_id, nodes in enumerate(nx.connected_components(graph)):
        for node in nodes:
            node_to_component[node] = component_id
    ids = []
    for nodes in row_nodes:
        component_ids = [node_to_component[node] for node in nodes]
        # Source, formula, and protocol nodes are connected within each row.
        ids.append(component_ids[0])
    return pd.Series(ids, index=df.index, name="leakage_graph_component_id")


def _source_formula_protocol_graph(
    df: pd.DataFrame,
    numeric_formatter=None,
) -> tuple[nx.Graph, list[tuple[str, str, str]]]:
    if numeric_formatter is None:
        numeric_formatter = _format_numeric_protocol_value
    graph = nx.Graph()
    row_nodes = []
    for idx, row in df.iterrows():
        source = f"source:{row.get('source_group_id', f'missing_source_{idx}')}"
        formula = f"formula:{row.get('formula_group_id', f'missing_formula_{idx}')}"
        protocol = f"protocol:{_protocol_signature(row, numeric_formatter=numeric_formatter)}"
        graph.add_node(source, node_type="source")
        graph.add_node(formula, node_type="formula")
        graph.add_node(protocol, node_type="protocol")
        graph.add_edge(source, formula)
        graph.add_edge(formula, protocol)
        graph.add_edge(source, protocol)
        row_nodes.append((source, formula, protocol))
    return graph, row_nodes


def _source_formula_graph(df: pd.DataFrame) -> tuple[nx.Graph, list[tuple[str, str]]]:
    graph = nx.Graph()
    row_nodes = []
    for idx, row in df.iterrows():
        source = f"source:{row.get('source_group_id', f'missing_source_{idx}')}"
        formula = f"formula:{row.get('formula_group_id', f'missing_formula_{idx}')}"
        graph.add_node(source, node_type="source")
        graph.add_node(formula, node_type="formula")
        graph.add_edge(source, formula)
        row_nodes.append((source, formula))
    return graph, row_nodes


def _graph_component_summary(graph: nx.Graph, row_nodes: list[tuple[str, ...]]) -> dict[str, int]:
    components = list(nx.connected_components(graph))
    node_to_component = {}
    for component_id, nodes in enumerate(components):
        for node in nodes:
            node_to_component[node] = component_id
    component_rows = defaultdict(int)
    for nodes in row_nodes:
        component_rows[node_to_component[nodes[0]]] += 1
    largest_rows = max(component_rows.values()) if component_rows else 0
    return {
        "n_connected_components": len(components),
        "largest_component_rows": int(largest_rows),
    }


def _source_formula_merge_summary(
    df: pd.DataFrame,
    tripartite_graph: nx.Graph,
    tripartite_row_nodes: list[tuple[str, str, str]],
    source_formula_row_nodes: list[tuple[str, str]],
) -> dict[str, int]:
    sf_graph, _ = _source_formula_graph(df)
    sf_components = list(nx.connected_components(sf_graph))
    sf_node_to_component = {
        node: component_id
        for component_id, nodes in enumerate(sf_components)
        for node in nodes
    }
    tri_components = list(nx.connected_components(tripartite_graph))
    tri_node_to_component = {
        node: component_id
        for component_id, nodes in enumerate(tri_components)
        for node in nodes
    }
    tri_to_sf_components: dict[int, set[int]] = defaultdict(set)
    sf_component_rows = defaultdict(int)
    for sf_nodes, tri_nodes in zip(source_formula_row_nodes, tripartite_row_nodes):
        sf_component = sf_node_to_component[sf_nodes[0]]
        tri_component = tri_node_to_component[tri_nodes[0]]
        sf_component_rows[sf_component] += 1
        tri_to_sf_components[tri_component].add(sf_component)
    merged_tri = {
        tri_component: sf_components
        for tri_component, sf_components in tri_to_sf_components.items()
        if len(sf_components) > 1
    }
    merged_sf = set().union(*merged_tri.values()) if merged_tri else set()
    rows_affected = sum(sf_component_rows[component] for component in merged_sf)
    return {
        "n_tripartite_components_merging_multiple_source_formula_components": int(
            len(merged_tri)
        ),
        "n_source_formula_components_merged_by_protocol": int(len(merged_sf)),
        "rows_in_protocol_merged_source_formula_components": int(rows_affected),
    }


def _protocol_bridge_summary(
    tripartite_row_nodes: list[tuple[str, str, str]],
    source_formula_graph: nx.Graph,
    source_formula_row_nodes: list[tuple[str, str]],
) -> dict[str, int]:
    sf_components = list(nx.connected_components(source_formula_graph))
    sf_node_to_component = {
        node: component_id
        for component_id, nodes in enumerate(sf_components)
        for node in nodes
    }
    protocol_to_rows: dict[str, set[int]] = defaultdict(set)
    protocol_to_sf_components: dict[str, set[int]] = defaultdict(set)
    for row_pos, (tri_nodes, sf_nodes) in enumerate(
        zip(tripartite_row_nodes, source_formula_row_nodes)
    ):
        protocol = tri_nodes[2]
        sf_component = sf_node_to_component[sf_nodes[0]]
        protocol_to_rows[protocol].add(row_pos)
        protocol_to_sf_components[protocol].add(sf_component)

    multirow_protocols = {
        protocol for protocol, row_ids in protocol_to_rows.items() if len(row_ids) > 1
    }
    cross_component_protocols = {
        protocol
        for protocol, components in protocol_to_sf_components.items()
        if len(components) > 1
    }
    rows_incident_to_cross = set().union(
        *(protocol_to_rows[protocol] for protocol in cross_component_protocols)
    ) if cross_component_protocols else set()
    sf_components_spanned = set().union(
        *(protocol_to_sf_components[protocol] for protocol in cross_component_protocols)
    ) if cross_component_protocols else set()
    return {
        "n_protocol_nodes_with_multiple_rows": int(len(multirow_protocols)),
        "n_cross_component_protocol_nodes": int(len(cross_component_protocols)),
        "n_redundant_shared_protocol_nodes": int(
            len(multirow_protocols - cross_component_protocols)
        ),
        "rows_incident_to_cross_component_protocol_nodes": int(len(rows_incident_to_cross)),
        "n_source_formula_components_spanned_by_cross_component_protocol_nodes": int(
            len(sf_components_spanned)
        ),
    }


def _protocol_bridge_interpretation(summary: dict[str, int]) -> str:
    if summary.get("n_cross_component_protocol_nodes", 0) == 0:
        return (
            "shared protocol nodes are redundant with source-formula components; "
            "the split operates on source-formula component topology in this dataset"
        )
    return (
        "at least one protocol node bridges otherwise separate source-formula components; "
        "the protocol layer changes connected-component topology"
    )


def _protocol_node_counts(row_nodes: list[tuple[str, str, str]]) -> dict[str, int]:
    counts = pd.Series([nodes[2] for nodes in row_nodes]).value_counts()
    return {
        "n_protocol_nodes": int(len(counts)),
        "n_shared_protocol_nodes": int((counts > 1).sum()) if len(counts) else 0,
        "max_rows_per_protocol_node": int(counts.max()) if len(counts) else 0,
    }


def _row_unseen_indicator(
    train_df: pd.DataFrame, test_df: pd.DataFrame, column: str
) -> pd.Series:
    if column not in train_df.columns or column not in test_df.columns:
        return pd.Series(0.0, index=test_df.index)
    train_values = set(
        train_df[column].astype("string").fillna("__MISSING__").astype(str)
    )
    test_values = test_df[column].astype("string").fillna("__MISSING__").astype(str)
    return (~test_values.isin(train_values)).astype(float)


def _row_protocol_category_unseen_fraction(
    train_df: pd.DataFrame, test_df: pd.DataFrame, columns: list[str]
) -> pd.Series:
    available = [col for col in columns if col in train_df.columns and col in test_df.columns]
    if not available:
        return pd.Series(0.0, index=test_df.index)
    indicators = []
    for col in available:
        indicators.append(_row_unseen_indicator(train_df, test_df, col))
    return pd.concat(indicators, axis=1).mean(axis=1).fillna(0.0)


def _row_protocol_numeric_shift(
    train_df: pd.DataFrame, test_df: pd.DataFrame, columns: list[str]
) -> pd.Series:
    shifts = pd.DataFrame(index=test_df.index)
    for col in columns:
        if col not in train_df.columns or col not in test_df.columns:
            continue
        train = pd.to_numeric(train_df[col], errors="coerce")
        test = pd.to_numeric(test_df[col], errors="coerce")
        if train.notna().sum() < 2:
            continue
        scale = float(train.std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-12:
            continue
        shifts[col] = (test - float(train.mean())).abs() / scale
    if shifts.empty:
        return pd.Series(0.0, index=test_df.index)
    return shifts.mean(axis=1, skipna=True).fillna(0.0)


def _row_composition_shift(
    X_train: pd.DataFrame, X_test: pd.DataFrame, composition_columns: list[str]
) -> pd.DataFrame:
    if not composition_columns or len(X_train) == 0 or len(X_test) == 0:
        return pd.DataFrame(
            {
                "composition_centroid_distance": np.zeros(len(X_test)),
                "composition_nearest_train_distance": np.zeros(len(X_test)),
            },
            index=X_test.index,
        )
    train = X_train.reindex(columns=composition_columns).apply(pd.to_numeric, errors="coerce")
    test = X_test.reindex(columns=composition_columns).apply(pd.to_numeric, errors="coerce")
    med = train.median(numeric_only=True)
    scale = train.std(ddof=0).replace(0, np.nan)
    train_scaled = ((train.fillna(med) - med) / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    test_scaled = ((test.fillna(med) - med) / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_arr = train_scaled.to_numpy(dtype=float)
    test_arr = test_scaled.to_numpy(dtype=float)
    centroid = train_arr.mean(axis=0)
    centroid_distance = np.sqrt(np.mean((test_arr - centroid) ** 2, axis=1))
    nearest = []
    for start in range(0, len(test_arr), 128):
        chunk = test_arr[start : start + 128]
        diff = chunk[:, None, :] - train_arr[None, :, :]
        distances = np.sqrt(np.mean(diff * diff, axis=2))
        nearest.extend(np.min(distances, axis=1).tolist())
    return pd.DataFrame(
        {
            "composition_centroid_distance": centroid_distance,
            "composition_nearest_train_distance": nearest,
        },
        index=X_test.index,
    )


def _unseen_fraction(df: pd.DataFrame, train_idx, test_idx, column: str) -> float:
    if column not in df.columns or len(test_idx) == 0:
        return 0.0
    train_values = set(df.loc[train_idx, column].astype("string").fillna("__MISSING__").astype(str))
    test_values = df.loc[test_idx, column].astype("string").fillna("__MISSING__").astype(str)
    return float((~test_values.isin(train_values)).mean())


def _mean_unseen_fraction(df: pd.DataFrame, train_idx, test_idx, columns: list[str]) -> float:
    values = [_unseen_fraction(df, train_idx, test_idx, col) for col in columns if col in df.columns]
    return float(np.mean(values)) if values else 0.0


def _mean_standardized_numeric_shift(
    df: pd.DataFrame, train_idx, test_idx, columns: list[str]
) -> float:
    shifts = []
    for col in columns:
        if col not in df.columns:
            continue
        train = pd.to_numeric(df.loc[train_idx, col], errors="coerce")
        test = pd.to_numeric(df.loc[test_idx, col], errors="coerce")
        if train.notna().sum() < 2 or test.notna().sum() < 1:
            continue
        scale = float(train.std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-12:
            continue
        shifts.append(abs(float(test.mean()) - float(train.mean())) / scale)
    return float(np.mean(shifts)) if shifts else 0.0


def _composition_shift(
    X: pd.DataFrame, composition_columns: list[str], train_idx, test_idx
) -> dict[str, float]:
    if not composition_columns or len(train_idx) == 0 or len(test_idx) == 0:
        return {
            "composition_centroid_distance": 0.0,
            "composition_nearest_train_distance_mean": 0.0,
        }
    train = X.loc[train_idx, composition_columns].apply(pd.to_numeric, errors="coerce")
    test = X.loc[test_idx, composition_columns].apply(pd.to_numeric, errors="coerce")
    med = train.median(numeric_only=True)
    scale = train.std(ddof=0).replace(0, np.nan)
    train_scaled = ((train.fillna(med) - med) / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    test_scaled = ((test.fillna(med) - med) / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_arr = train_scaled.to_numpy(dtype=float)
    test_arr = test_scaled.to_numpy(dtype=float)
    centroid_distance = float(np.linalg.norm(test_arr.mean(axis=0) - train_arr.mean(axis=0)))
    nearest_distances = []
    for start in range(0, len(test_arr), 128):
        chunk = test_arr[start : start + 128]
        diff = chunk[:, None, :] - train_arr[None, :, :]
        distances = np.sqrt(np.mean(diff * diff, axis=2))
        nearest_distances.extend(np.min(distances, axis=1).tolist())
    return {
        "composition_centroid_distance": centroid_distance,
        "composition_nearest_train_distance_mean": float(np.mean(nearest_distances)),
    }


def _squash(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 0.0
    return float(value / (1.0 + value))


def _row_id_set(df: pd.DataFrame, idx) -> set[int]:
    if len(idx) == 0:
        return set()
    return set(df.loc[np.asarray(idx), "row_id"].astype(int).tolist())


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _fraction(numerator: set[int], denominator: set[int]) -> float:
    if not denominator:
        return 0.0
    return len(numerator) / len(denominator)


def _protocol_signature(row: pd.Series, numeric_formatter=None) -> str:
    if numeric_formatter is None:
        numeric_formatter = _format_numeric_protocol_value
    parts = []
    for col in PROTOCOL_CATEGORICAL_COLUMNS:
        if col in row.index:
            value = row[col]
            parts.append(f"{col}={_clean_token(value)}")
    for col in PROTOCOL_NUMERIC_COLUMNS:
        if col in row.index:
            value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.isna(value):
                parts.append(f"{col}=missing")
            else:
                parts.append(f"{col}={numeric_formatter(float(value), col)}")
    return "|".join(parts) if parts else "__NO_PROTOCOL__"


def _clean_token(value: object) -> str:
    if pd.isna(value):
        return "missing"
    return str(value).strip().lower().replace(" ", "_") or "missing"


def _format_numeric_protocol_value(value: float, col: str | None = None) -> str:
    if not math.isfinite(value):
        return "missing"
    return f"{value:.3g}"


def _coarse_protocol_formatter(value: float, col: str | None = None) -> str:
    if not math.isfinite(value):
        return "missing"
    if col == "ph":
        return _format_binned_value(value, 0.5)
    if col == "temperature":
        return _format_binned_value(value, 5.0)
    return f"{value:.2g}"


def _very_coarse_protocol_formatter(value: float, col: str | None = None) -> str:
    if not math.isfinite(value):
        return "missing"
    if col == "ph":
        return _format_binned_value(value, 1.0)
    if col == "temperature":
        return _format_binned_value(value, 10.0)
    return f"{value:.1g}"


def _format_binned_value(value: float, width: float) -> str:
    if width <= 0:
        return f"{value:.3g}"
    return f"{round(value / width) * width:.3g}"
