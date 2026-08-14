from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import BaseCrossValidator, GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler


class LeakageGraphSplit(BaseCrossValidator):
    """Sklearn-compatible component holdout over arbitrary row-level node columns."""

    def __init__(
        self,
        node_cols: list[str],
        n_splits: int = 5,
        test_size: float = 0.2,
        random_state: int | None = None,
        min_test_n: int = 20,
    ):
        self.node_cols = node_cols
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.min_test_n = min_test_n

    def split(self, X, y=None, groups=None):
        data = pd.DataFrame(X).reset_index(drop=True)
        missing = [col for col in self.node_cols if col not in data.columns]
        if missing:
            raise ValueError(f"LeakageGraphSplit missing node_cols: {missing}")
        component_ids = leakage_graph_component_ids(data, self.node_cols)
        work = pd.DataFrame(
            {
                "row_id": np.arange(len(data), dtype=int),
                "leakage_graph_component_id": component_ids,
            }
        )
        base_seed = 0 if self.random_state is None else int(self.random_state)
        for split_id in range(self.n_splits):
            train_idx, test_idx, _ = _component_balanced_group_split(
                work,
                "leakage_graph_component_id",
                base_seed + split_id,
                self.test_size,
                "leakage_graph",
                self.min_test_n,
            )
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return int(self.n_splits)


def leakage_graph_component_ids(df: pd.DataFrame, node_cols: list[str]) -> pd.Series:
    """Return connected-component ids for a graph built from arbitrary node columns."""
    graph = nx.Graph()
    row_nodes: list[list[str]] = []
    for row_pos, (_, row) in enumerate(df.iterrows()):
        nodes = [
            f"{col}:{_generic_node_token(row[col])}"
            for col in node_cols
            if col in row.index
        ]
        if not nodes:
            nodes = [f"row:{row_pos}"]
        graph.add_nodes_from(nodes)
        anchor = nodes[0]
        for node in nodes[1:]:
            graph.add_edge(anchor, node)
        row_nodes.append(nodes)

    node_to_component = {}
    for component_id, nodes in enumerate(nx.connected_components(graph)):
        for node in nodes:
            node_to_component[node] = component_id
    return pd.Series(
        [node_to_component[nodes[0]] for nodes in row_nodes],
        index=df.index,
        name="leakage_graph_component_id",
    )


def _generic_node_token(value: object) -> str:
    if pd.isna(value):
        return "__MISSING__"
    text = str(value).strip().lower()
    return text if text else "__MISSING__"


def make_split(
    df: pd.DataFrame,
    split_type: str,
    seed: int,
    test_size: float = 0.2,
    composition: pd.DataFrame | None = None,
    min_test_n: int = 20,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if split_type == "random":
        train_idx, test_idx = train_test_split(
            df.index.to_numpy(), test_size=test_size, random_state=seed, shuffle=True
        )
        return train_idx, test_idx, {"split_type": split_type}
    if split_type == "source_group":
        return _group_split(df, "source_group_id", seed, test_size, split_type)
    if split_type == "formula_group":
        return _group_split(df, "formula_group_id", seed, test_size, split_type)
    if split_type == "leakage_graph":
        if "leakage_graph_component_id" not in df.columns:
            raise ValueError("leakage_graph_component_id is required for leakage_graph split")
        return _component_balanced_group_split(
            df,
            "leakage_graph_component_id",
            seed,
            test_size,
            split_type,
            min_test_n,
        )
    if split_type == "composition_cluster":
        if composition is None:
            raise ValueError("composition descriptors are required for composition_cluster split")
        return _composition_cluster_split(df, composition, seed, test_size, min_test_n)
    raise ValueError(f"Unknown split_type: {split_type}")


def _group_split(df, group_col, seed, test_size, split_type):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_pos, test_pos = next(splitter.split(df, groups=df[group_col]))
    train_idx = df.index.to_numpy()[train_pos]
    test_idx = df.index.to_numpy()[test_pos]
    return train_idx, test_idx, {"split_type": split_type, "grouping_variable": group_col}


def _component_balanced_group_split(df, group_col, seed, test_size, split_type, min_test_n):
    groups = df.groupby(group_col, dropna=False).size().sort_values(ascending=False)
    if len(groups) < 2:
        raise ValueError(
            f"{split_type} split requires at least two connected components; "
            f"found {len(groups)} in {group_col}"
        )
    target_n = max(min_test_n, int(round(len(df) * test_size)))
    rng = np.random.default_rng(seed)
    eligible = groups[groups <= target_n]
    if eligible.empty:
        if len(groups) < 2:
            raise ValueError(
                f"{split_type} split cannot hold out a component without emptying train"
            )
        chosen_groups = [groups.index[-1]]
    else:
        shuffled = list(rng.permutation(eligible.index.to_numpy()))
        chosen_groups = []
        current_n = 0
        for group in shuffled:
            group_n = int(groups.loc[group])
            if current_n + group_n <= target_n or current_n < min_test_n:
                chosen_groups.append(group)
                current_n += group_n
            if current_n >= target_n:
                break
        if current_n < min_test_n:
            remaining = [g for g in eligible.sort_values().index if g not in set(chosen_groups)]
            for group in remaining:
                chosen_groups.append(group)
                current_n += int(groups.loc[group])
                if current_n >= min_test_n:
                    break
    test_mask = df[group_col].isin(chosen_groups)
    test_idx = df.index[test_mask].to_numpy()
    train_idx = df.index[~test_mask].to_numpy()
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"{split_type} split produced an empty train/test partition "
            f"(train={len(train_idx)}, test={len(test_idx)}, groups={len(groups)})"
        )
    oversized = groups[groups > target_n]
    info = {
        "split_type": split_type,
        "grouping_variable": group_col,
        "target_test_n": int(target_n),
        "test_n": int(len(test_idx)),
        "component_balancing_rule": (
            "hold out graph components without train/test overlap; components larger than target test "
            "size are kept in train when smaller components can satisfy the test-size constraint"
        ),
        "n_groups_total": int(len(groups)),
        "n_groups_test": int(len(chosen_groups)),
        "largest_group_n": int(groups.max()),
        "n_oversized_groups_kept_train": int(len(oversized)),
    }
    return train_idx, test_idx, info


def _composition_cluster_split(df, composition, seed, test_size, min_test_n):
    base_train_idx, base_test_idx = train_test_split(
        df.index.to_numpy(), test_size=test_size, random_state=seed, shuffle=True
    )
    X_fit = composition.loc[base_train_idx].copy()
    X_assign = composition.copy()
    medians = X_fit.median(numeric_only=True)
    X_fit = X_fit.fillna(medians).fillna(0.0)
    X_assign = X_assign.fillna(medians).fillna(0.0)
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)
    X_assign_scaled = scaler.transform(X_assign)
    candidate_k = [k for k in [3, 4, 5, 6, 8] if k < len(base_train_idx)]
    selected_k = candidate_k[0] if candidate_k else 2
    best_score = -np.inf
    diagnostics = []
    for k in candidate_k:
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = km.fit_predict(X_fit_scaled)
        counts = pd.Series(labels).value_counts()
        min_cluster = int(counts.min())
        if len(set(labels)) < 2:
            score = -np.inf
        else:
            score = float(silhouette_score(X_fit_scaled, labels))
        diagnostics.append({"k": k, "silhouette": score, "min_train_cluster_size": min_cluster})
        if min_cluster >= min_test_n and score > best_score:
            selected_k = k
            best_score = score
    if not diagnostics and len(base_train_idx) >= 2:
        selected_k = 2
    km = KMeans(n_clusters=selected_k, random_state=seed, n_init=20)
    fit_labels = km.fit_predict(X_fit_scaled)
    all_labels = km.predict(X_assign_scaled)
    test_labels = pd.Series(all_labels, index=df.index).loc[base_test_idx]
    cluster_sizes = test_labels.value_counts().sort_values(ascending=False)
    eligible = cluster_sizes[cluster_sizes >= min_test_n]
    if len(eligible):
        chosen_cluster = int(eligible.index[0])
    else:
        chosen_cluster = int(cluster_sizes.index[0])
    test_idx = test_labels[test_labels == chosen_cluster].index.to_numpy()
    train_idx = np.setdiff1d(base_train_idx, test_idx, assume_unique=False)
    info = {
        "split_type": "composition_cluster",
        "algorithm": "KMeans",
        "distance_metric": "euclidean_on_standardized_composition_descriptors",
        "candidate_k_values": candidate_k,
        "selected_k": int(selected_k),
        "k_selection_rule": "highest train-only silhouette among k with min train cluster size constraint",
        "fallback_rule": "if no k satisfies min size, use best available/default and largest assigned test cluster",
        "chosen_ood_cluster": chosen_cluster,
        "test_n": int(len(test_idx)),
        "underpowered_fold": bool(len(test_idx) < min_test_n),
        "cluster_diagnostics": diagnostics,
        "train_cluster_sizes": pd.Series(fit_labels).value_counts().to_dict(),
        "heldout_assigned_cluster_sizes": cluster_sizes.to_dict(),
    }
    return train_idx, test_idx, info


def save_manifest(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    info: dict,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_cols = ["row_id", "source_group_id", "formula_group_id"]
    if "leakage_graph_component_id" in df.columns:
        manifest_cols.append("leakage_graph_component_id")
    manifest = df.loc[np.r_[train_idx, test_idx], manifest_cols].copy()
    manifest["split"] = "train"
    manifest.loc[manifest["row_id"].isin(df.loc[test_idx, "row_id"]), "split"] = "test"
    manifest.to_csv(output_path.with_suffix(".csv"), index=False)
    output_path.with_suffix(".json").write_text(json.dumps(info, indent=2, default=str))


def split_overlap_report(df: pd.DataFrame, formula_test_idx, composition_test_idx) -> pd.DataFrame:
    formula_rows = set(df.loc[formula_test_idx, "row_id"])
    comp_rows = set(df.loc[composition_test_idx, "row_id"])
    formula_formulas = set(df.loc[formula_test_idx, "formula_group_id"])
    comp_formulas = set(df.loc[composition_test_idx, "formula_group_id"])
    row_union = formula_rows | comp_rows
    formula_union = formula_formulas | comp_formulas
    return pd.DataFrame(
        [
            {
                "fraction_samples_overlapping": len(formula_rows & comp_rows) / max(1, len(row_union)),
                "fraction_formulas_overlapping": len(formula_formulas & comp_formulas)
                / max(1, len(formula_union)),
                "distinct_ood_tests": "yes"
                if len(formula_rows & comp_rows) / max(1, len(row_union)) < 0.5
                else "high_overlap_present_as_sensitivity",
            }
        ]
    )
