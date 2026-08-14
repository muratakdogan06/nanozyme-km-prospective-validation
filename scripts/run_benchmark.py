#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import audit_data, prepare_modeling_data, read_table, write_audit_outputs
from src.diagnostics import (
    generalization_stress_index,
    ood_severity_row,
    protocol_granularity_sensitivity,
    severity_performance_association,
    split_overlap_diagnostics,
    split_repeat_variability,
    source_formula_protocol_graph_diagnostics,
    source_formula_protocol_component_ids,
)
from src.evaluate import regression_metrics, summarize_with_ci, summarize_with_cluster_bootstrap
from src.features import (
    build_design_only_feature_frame,
    build_feature_frame,
    elemental_coverage_table,
)
from src.models import FeatureSpec, fit_predict, model_suite, require_catboost, unseen_category_rates
from src.reporting import ensure_output_dirs, make_main_tables
from src.splits import make_split, save_manifest, split_overlap_report
from src.validation import (
    raise_if_failed,
    validate_feature_columns,
    validate_modeling_frame,
    validate_split,
    validation_results_to_frame,
)


SPLIT_TYPES = ["random", "source_group", "formula_group", "composition_cluster", "leakage_graph"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-composition-test-n", type=int, default=20)
    parser.add_argument("--missingness-indicators", action="store_true")
    parser.add_argument("--skip-missingness-analysis", action="store_true")
    parser.add_argument("--skip-feature-ablation", action="store_true")
    parser.add_argument(
        "--post-characterization",
        action="store_true",
        help="Include Km/Vmax reporting fields; default is the prospective design-only schema.",
    )
    args = parser.parse_args()

    if not args.data.strip():
        parser.error(
            '--data was empty. If using --data "$DATA", set DATA first '
            "or pass a file path such as --data data/nanozymes.csv."
        )

    try:
        require_catboost()
    except ImportError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    dirs = ensure_output_dirs()
    raw = read_table(args.data)
    df, filter_log = prepare_modeling_data(raw)
    if not args.post_characterization:
        df = df.drop(columns=["vmax_value", "vmax_unit"], errors="ignore")
    df["leakage_graph_component_id"] = source_formula_protocol_component_ids(df)
    audit = audit_data(raw, df, filter_log)
    write_audit_outputs(audit, dirs["tables"] / "audit")
    audit["missingness"].to_csv(dirs["tables"] / "supp_table_s1_missingness_data_quality.csv", index=False)
    audit["near_duplicate_clusters"].to_csv(
        dirs["tables"] / "supp_table_s8_near_duplicate_audit.csv", index=False
    )

    feature_builder = (
        build_feature_frame
        if args.post_characterization
        else build_design_only_feature_frame
    )
    X, meta = feature_builder(
        df, include_missing_indicators=args.missingness_indicators
    )
    y = df["target_log10_km"].astype(float)
    validation_results = []
    validation_results.extend(validate_modeling_frame(df))
    validation_results.extend(validate_feature_columns(X.columns))
    raise_if_failed(validation_results)
    feature_spec = FeatureSpec(meta["numeric_columns"], meta["categorical_columns"])
    elemental_coverage_table(df["standardized_formula"]).to_csv(
        dirs["tables"] / "supp_table_s10_elemental_property_coverage.csv", index=False
    )

    rows = []
    severity_rows = []
    unseen_rows = []
    clustering_diags = []
    overlap_reports = []
    repeat_test_sets = []
    for rep in range(args.repeats):
        rep_seed = args.seed + rep
        models = model_suite(feature_spec, rep_seed)
        rep_tests = {}
        for split_type in SPLIT_TYPES:
            train_idx, test_idx, info = make_split(
                df,
                split_type,
                rep_seed,
                test_size=args.test_size,
                composition=X[meta["composition_columns"]],
                min_test_n=args.min_composition_test_n,
            )
            split_validation = validate_split(df, train_idx, test_idx, split_type)
            validation_results.extend(split_validation)
            raise_if_failed(split_validation)
            rep_tests[split_type] = test_idx
            save_manifest(
                df,
                train_idx,
                test_idx,
                info | {"repeat": rep, "seed": rep_seed},
                dirs["split_manifests"] / f"{split_type}_repeat_{rep}",
            )
            if split_type == "composition_cluster":
                clustering_diags.append({"repeat": rep, **info})
            severity_rows.append(
                ood_severity_row(
                    df,
                    X,
                    meta["composition_columns"],
                    train_idx,
                    test_idx,
                    split_type,
                    rep,
                    rep_seed,
                )
            )
            unseen = unseen_category_rates(
                X.loc[train_idx], X.loc[test_idx], meta["categorical_columns"]
            )
            unseen["split_type"] = split_type
            unseen["repeat"] = rep
            unseen_rows.append(unseen)
            for model_name, model in models.items():
                pred = fit_predict(model, X.loc[train_idx], y.loc[train_idx], X.loc[test_idx])
                metrics = regression_metrics(y.loc[test_idx], pred)
                rows.append(
                    {
                        "repeat": rep,
                        "seed": rep_seed,
                        "split_type": split_type,
                        "model": model_name,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        **metrics,
                    }
                )
        overlap_reports.append(
            split_overlap_report(df, rep_tests["formula_group"], rep_tests["composition_cluster"]).assign(
                repeat=rep
            )
        )
        repeat_test_sets.append({"repeat": rep, "seed": rep_seed, "tests": rep_tests.copy()})

    metrics = pd.DataFrame(rows)
    severity = pd.DataFrame(severity_rows)
    graph_summary, graph_components = source_formula_protocol_graph_diagnostics(df)
    validation_results_to_frame(validation_results).to_csv(
        dirs["tables"] / "supp_table_split_and_leakage_validation.csv", index=False
    )
    metrics.to_csv(dirs["metrics"] / "benchmark_metrics.csv", index=False)
    severity.to_csv(dirs["tables"] / "supp_table_s20_ood_severity_diagnostics.csv", index=False)
    generalization_stress_index(metrics).to_csv(
        dirs["tables"] / "supp_table_s21_generalization_stress_index.csv", index=False
    )
    severity_performance_association(severity, metrics).to_csv(
        dirs["tables"] / "supp_table_s22_ood_severity_performance_association.csv", index=False
    )
    graph_summary.to_csv(
        dirs["tables"] / "supp_table_s23_source_formula_protocol_graph_diagnostics.csv",
        index=False,
    )
    graph_components.to_csv(
        dirs["tables"] / "supp_table_s24_source_formula_protocol_graph_components.csv",
        index=False,
    )
    protocol_granularity_sensitivity(df).to_csv(
        dirs["tables"] / "supp_table_s31_protocol_granularity_sensitivity.csv",
        index=False,
    )
    split_overlap_diagnostics(df, repeat_test_sets).to_csv(
        dirs["tables"] / "supp_table_s26_leakage_graph_overlap_diagnostics.csv",
        index=False,
    )
    split_repeat_variability(df, repeat_test_sets).to_csv(
        dirs["tables"] / "supp_table_s27_split_repeat_variability.csv",
        index=False,
    )
    summary = summarize_with_ci(metrics, ["split_type", "model"], ["mae", "rmse", "r2", "spearman"])
    summary.to_csv(dirs["tables"] / "supp_table_s2_full_model_benchmark.csv", index=False)
    bootstrap_summary = summarize_with_cluster_bootstrap(
        metrics,
        ["split_type", "model"],
        ["mae", "rmse", "r2", "spearman"],
    )
    bootstrap_summary.to_csv(
        dirs["tables"] / "supp_table_s28_cluster_bootstrap_performance_ci.csv", index=False
    )
    ci_method_comparison(summary, bootstrap_summary).to_csv(
        dirs["tables"] / "supp_table_s30_ci_method_comparison.csv", index=False
    )
    ci_dependency_note().to_csv(
        dirs["tables"] / "supp_table_s29_ci_dependency_limitation_note.csv", index=False
    )
    if unseen_rows:
        pd.concat(unseen_rows, ignore_index=True).to_csv(
            dirs["tables"] / "supp_table_s12_unseen_category_diagnostics.csv", index=False
        )
    if overlap_reports:
        pd.concat(overlap_reports, ignore_index=True).to_csv(
            dirs["tables"] / "supp_table_s7_formula_vs_composition_overlap.csv", index=False
        )
    pd.DataFrame(clustering_diags).to_csv(
        dirs["tables"] / "supp_table_s11_composition_clustering_diagnostics.csv", index=False
    )
    if not args.skip_missingness_analysis:
        run_missingness_analysis(df, y, meta, args, dirs)
    if not args.skip_feature_ablation:
        run_feature_group_ablation(df, y, X, meta, metrics, args, dirs)
    filter_log.to_csv(dirs["tables"] / "supp_table_s5_unit_conversion_filtering_log.csv", index=False)
    feature_decisions(meta).to_csv(
        dirs["tables"] / "supp_table_s4_feature_preprocessing_decisions.csv", index=False
    )
    model_selection_log(args.seed).to_csv(
        dirs["tables"] / "supp_table_s13_train_only_model_selection_log.csv", index=False
    )
    literature_reproduction_note(metrics).to_csv(
        dirs["tables"] / "supp_table_s9_literature_aligned_reproduction.csv", index=False
    )
    model_stability_diagnostics(metrics).to_csv(
        dirs["tables"] / "supp_table_s14_model_stability_diagnostics.csv", index=False
    )
    primary_comparison_table(metrics).to_csv(
        dirs["tables"] / "supp_table_s15_primary_random_vs_ood_comparisons.csv", index=False
    )
    optimism_gap_table(metrics).to_csv(
        dirs["tables"] / "supp_table_s3_optimism_gap.csv", index=False
    )
    q1_readiness_checklist(args).to_csv(
        dirs["tables"] / "supp_table_s16_q1_readiness_checklist.csv", index=False
    )
    make_main_tables(dirs["tables"] / "audit", metrics, None, dirs["tables"])
    (dirs["metrics"] / "primary_comparisons.json").write_text(
        json.dumps(
            {
                "primary_metric": "mae",
                "target": "log10(Km_common_molar)",
                "comparisons": [
                    "random_vs_source_group",
                    "random_vs_formula_group",
                    "random_vs_composition_cluster",
                    "random_vs_leakage_graph",
                ],
                "main_model": "catboost predefined before test evaluation",
            },
            indent=2,
        )
    )
    print(f"Benchmark complete: {dirs['metrics'] / 'benchmark_metrics.csv'}")


def feature_decisions(meta: dict) -> pd.DataFrame:
    rows = []
    for col in meta["numeric_columns"]:
        rows.append(
            {
                "feature_name_or_group": col,
                "transformation": "numeric median imputation fitted on training data only",
                "imputation_method": "train median",
                "leakage_risk": "low",
                "included_excluded": "included",
                "reason": "experimental or deterministic composition descriptor",
            }
        )
    for col in meta["categorical_columns"]:
        rows.append(
            {
                "feature_name_or_group": col,
                "transformation": "train-fit one-hot for sklearn; string token for CatBoost",
                "imputation_method": "__MISSING__ token",
                "leakage_risk": "controlled with train-only vocabularies",
                "included_excluded": "included",
                "reason": "experimental/material descriptor, not DOI/title/journal/year/source metadata",
            }
        )
    return pd.DataFrame(rows)


def model_selection_log(seed: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_models": "median; elasticnet; extratrees; catboost",
                "inner_validation_metric": "not used for main-model selection",
                "selected_model": "catboost",
                "selection_rule": "predefined before final test evaluation",
                "random_seed": seed,
            }
        ]
    )


def literature_reproduction_note(metrics: pd.DataFrame) -> pd.DataFrame:
    random_perf = metrics[metrics["split_type"] == "random"].groupby("model")["mae"].mean()
    rows = []
    for model, mae in random_perf.items():
        rows.append(
            {
                "model_family": model,
                "reproduced_settings": "random split; log10(Km) target; standard tabular regressors",
                "approximated_settings": "exact original DiZyme preprocessing only if matching columns are present",
                "unavailable_settings": "any undocumented train/test seeds, hyperparameter grids, or feature filters",
                "random_split_mae": mae,
                "leakage_controlled_performance": "see source/formula/composition split rows in S2",
            }
        )
    return pd.DataFrame(rows)


def model_stability_diagnostics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_type, model), group in metrics.groupby(["split_type", "model"], dropna=False):
        mae = group["mae"].astype(float)
        r2 = group["r2"].astype(float)
        worst = group.loc[mae.idxmax()]
        instability_flag = bool((mae.max() > 5.0) or (r2.min() < -10.0))
        rows.append(
            {
                "split_type": split_type,
                "model": model,
                "n_repeats": int(group["repeat"].nunique()),
                "mae_mean": float(mae.mean()),
                "mae_sd": float(mae.std(ddof=1)) if len(mae) > 1 else pd.NA,
                "mae_min": float(mae.min()),
                "mae_max": float(mae.max()),
                "r2_min": float(r2.min()),
                "worst_repeat": int(worst["repeat"]),
                "worst_seed": int(worst["seed"]),
                "instability_flag": instability_flag,
                "interpretation": "inspect as unstable baseline" if instability_flag else "no catastrophic repeat",
            }
        )
    return pd.DataFrame(rows)


def primary_comparison_table(metrics: pd.DataFrame, main_model: str = "catboost") -> pd.DataFrame:
    main = metrics[metrics["model"] == main_model].copy()
    random_mae = main[main["split_type"] == "random"].set_index("repeat")["mae"]
    rows = []
    for split_type in ["source_group", "formula_group", "composition_cluster", "leakage_graph"]:
        ood_mae = main[main["split_type"] == split_type].set_index("repeat")["mae"]
        paired = pd.concat([random_mae.rename("random_mae"), ood_mae.rename("ood_mae")], axis=1).dropna()
        delta = paired["ood_mae"] - paired["random_mae"]
        p_value = pd.NA
        if len(delta) > 1 and not (delta == 0).all():
            try:
                p_value = float(wilcoxon(delta).pvalue)
            except ValueError:
                p_value = pd.NA
        n_positive = int((delta > 0).sum()) if len(delta) else 0
        sign_test_p = (
            float(binomtest(n_positive, len(delta), 0.5, alternative="greater").pvalue)
            if len(delta)
            else pd.NA
        )
        boot_low = pd.NA
        boot_high = pd.NA
        if len(delta) > 1:
            rng = np.random.default_rng(2026)
            boot_means = [
                float(rng.choice(delta.to_numpy(dtype=float), size=len(delta), replace=True).mean())
                for _ in range(20000)
            ]
            boot_low = float(np.quantile(boot_means, 0.025))
            boot_high = float(np.quantile(boot_means, 0.975))
        rows.append(
            {
                "comparison": f"random_vs_{split_type}",
                "model": main_model,
                "primary_metric": "mae",
                "n_paired_repeats": int(len(delta)),
                "n_positive_delta_repeats": n_positive,
                "n_nonpositive_delta_repeats": int(len(delta) - n_positive),
                "random_mae_mean": float(paired["random_mae"].mean()) if len(paired) else pd.NA,
                "ood_mae_mean": float(paired["ood_mae"].mean()) if len(paired) else pd.NA,
                "delta_mae_mean": float(delta.mean()) if len(delta) else pd.NA,
                "delta_mae_ci95_low": float(delta.quantile(0.025)) if len(delta) > 1 else pd.NA,
                "delta_mae_ci95_high": float(delta.quantile(0.975)) if len(delta) > 1 else pd.NA,
                "delta_mae_bootstrap_mean_ci95_low": boot_low,
                "delta_mae_bootstrap_mean_ci95_high": boot_high,
                "paired_wilcoxon_p": p_value,
                "one_sided_positive_sign_test_p": sign_test_p,
                "interpretation": "positive delta means worse OOD performance than random split",
            }
        )
    return pd.DataFrame(rows)


def optimism_gap_table(metrics: pd.DataFrame) -> pd.DataFrame:
    random_ref = metrics[metrics["split_type"] == "random"].groupby("model")[["mae", "r2"]].mean()
    rows = []
    for _, row in metrics.iterrows():
        if row["split_type"] == "random" or row["model"] not in random_ref.index:
            continue
        rows.append(
            {
                "repeat": row["repeat"],
                "seed": row["seed"],
                "model": row["model"],
                "split_type": row["split_type"],
                "delta_mae": row["mae"] - random_ref.loc[row["model"], "mae"],
                "delta_r2": random_ref.loc[row["model"], "r2"] - row["r2"],
                "interpretation": "positive delta_mae indicates degradation relative to the random split reference",
            }
        )
    return pd.DataFrame(rows)


def q1_readiness_checklist(args) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "q1_requirement": "ElasticNet instability is visible, not hidden",
                "status": "implemented",
                "pipeline_evidence": "outputs/tables/supp_table_s14_model_stability_diagnostics.csv",
                "manuscript_use": "report unstable linear baseline repeats transparently",
            },
            {
                "q1_requirement": "Train-only preprocessing/model selection is documented",
                "status": "implemented",
                "pipeline_evidence": "outputs/tables/supp_table_s4_feature_preprocessing_decisions.csv; outputs/tables/supp_table_s13_train_only_model_selection_log.csv",
                "manuscript_use": "state that imputation, one-hot vocabularies, CatBoost medians, and model choice are train-only/predefined",
            },
            {
                "q1_requirement": "Feature-group ablations quantify where signal comes from",
                "status": "skipped_by_flag" if args.skip_feature_ablation else "implemented",
                "pipeline_evidence": "outputs/tables/supp_table_s17_feature_group_ablation.csv",
                "manuscript_use": "separate composition, morphology, protocol/context, and full-feature contributions",
            },
            {
                "q1_requirement": "OOD emphasis is primary, random split is optimistic reference",
                "status": "implemented",
                "pipeline_evidence": "outputs/metrics/primary_comparisons.json; outputs/tables/supp_table_s15_primary_random_vs_ood_comparisons.csv; outputs/tables/supp_table_s3_optimism_gap.csv",
                "manuscript_use": "frame random split as reference and source/formula/composition splits as primary stress tests",
            },
            {
                "q1_requirement": "Uncertainty analysis is interpreted cautiously under shift",
                "status": "implemented_in_conformal_step",
                "pipeline_evidence": "outputs/tables/supp_table_s18_conformal_interpretation_note.csv; outputs/tables/supp_table_s6_full_few_shot_recalibration.csv",
                "manuscript_use": "avoid claiming formal conformal validity under covariate shift",
            },
            {
                "q1_requirement": "Source metadata leakage is audited",
                "status": "implemented",
                "pipeline_evidence": "outputs/tables/supp_table_split_and_leakage_validation.csv; outputs/split_manifests/",
                "manuscript_use": "show DOI/source/title/journal/year/PDF metadata are excluded from features and used only for grouping",
            },
            {
                "q1_requirement": "Primary statistical comparisons are predefined and paired by repeat",
                "status": "implemented",
                "pipeline_evidence": "outputs/tables/supp_table_s15_primary_random_vs_ood_comparisons.csv",
                "manuscript_use": "report paired random-vs-OOD MAE deltas with uncertainty and Wilcoxon p-values",
            },
            {
                "q1_requirement": "Methodological novelty is anchored in connectivity-aware leakage auditing",
                "status": "implemented",
                "pipeline_evidence": "outputs/tables/supp_table_s20_ood_severity_diagnostics.csv; outputs/tables/supp_table_s21_generalization_stress_index.csv; outputs/tables/supp_table_s23_source_formula_protocol_graph_diagnostics.csv; outputs/tables/supp_table_s26_leakage_graph_overlap_diagnostics.csv; outputs/tables/supp_table_s31_protocol_granularity_sensitivity.csv",
                "manuscript_use": "present the source-formula-protocol graph as an audited component-holdout framework; explicitly report whether protocol nodes alter source-formula topology; present severity, stress, and fragility tables as reporting diagnostics",
            },
            {
                "q1_requirement": "Split-repeat dependence is quantified for uncertainty reporting",
                "status": "implemented",
                "pipeline_evidence": "outputs/tables/supp_table_s27_split_repeat_variability.csv; outputs/tables/supp_table_s28_cluster_bootstrap_performance_ci.csv; outputs/tables/supp_table_s29_ci_dependency_limitation_note.csv",
                "manuscript_use": "report leakage_graph uncertainty with caution because repeat-level test sets are partially overlapping; compare empirical repeat quantiles and repeat-cluster bootstrap intervals",
            },
        ]
    )


def ci_method_comparison(naive: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = ["mae", "rmse", "r2", "spearman"]
    merged = naive.merge(bootstrap, on=["split_type", "model"], suffixes=("_naive", "_bootstrap"))
    for _, row in merged.iterrows():
        base = {"split_type": row["split_type"], "model": row["model"]}
        for metric in metric_cols:
            naive_low = row.get(f"{metric}_ci95_low")
            naive_high = row.get(f"{metric}_ci95_high")
            boot_low = row.get(f"{metric}_cluster_boot_ci95_low")
            boot_high = row.get(f"{metric}_cluster_boot_ci95_high")
            naive_width = naive_high - naive_low if pd.notna(naive_low) and pd.notna(naive_high) else pd.NA
            boot_width = boot_high - boot_low if pd.notna(boot_low) and pd.notna(boot_high) else pd.NA
            conservative_low = _nan_safe_min([naive_low, boot_low])
            conservative_high = _nan_safe_max([naive_high, boot_high])
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "naive_repeat_quantile_low": naive_low,
                    "naive_repeat_quantile_high": naive_high,
                    "naive_repeat_quantile_width": naive_width,
                    "cluster_bootstrap_low": boot_low,
                    "cluster_bootstrap_high": boot_high,
                    "cluster_bootstrap_width": boot_width,
                    "bootstrap_wider_than_naive": bool(boot_width > naive_width)
                    if pd.notna(naive_width) and pd.notna(boot_width)
                    else pd.NA,
                    "conservative_low": conservative_low,
                    "conservative_high": conservative_high,
                    "conservative_width": conservative_high - conservative_low
                    if pd.notna(conservative_low) and pd.notna(conservative_high)
                    else pd.NA,
                    "recommended_use": (
                        "report conservative interval spanning both empirical repeat quantiles "
                        "and repeat-cluster bootstrap percentiles"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _nan_safe_min(values: list[float]) -> float:
    finite = [value for value in values if pd.notna(value)]
    return min(finite) if finite else pd.NA


def _nan_safe_max(values: list[float]) -> float:
    finite = [value for value in values if pd.notna(value)]
    return max(finite) if finite else pd.NA


def ci_dependency_note() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "issue": "repeat-level test-set dependence",
                "affected_split": "especially leakage_graph",
                "reason": "component-balanced graph holdout reuses some held-out rows across repeats more than random/source/formula splits",
                "reporting_response": "provide empirical repeat-quantile intervals, repeat-cluster bootstrap intervals, and a conservative interval spanning both",
                "recommended_manuscript_language": (
                    "Because graph-component holdout repeats are partially overlapping, confidence intervals "
                    "for the leakage_graph split should be interpreted descriptively. We therefore report "
                    "repeat-level empirical intervals alongside repeat-cluster bootstrap intervals and use "
                    "their combined span as a conservative uncertainty summary."
                ),
            }
        ]
    )


def run_missingness_analysis(df, y, meta, args, dirs) -> None:
    X_no, meta_no = build_design_only_feature_frame(
        df, include_missing_indicators=False
    )
    X_yes, meta_yes = build_design_only_feature_frame(
        df, include_missing_indicators=True
    )
    rows = []
    for include_indicators, X_cur, meta_cur in [
        (False, X_no, meta_no),
        (True, X_yes, meta_yes),
    ]:
        feature_spec = FeatureSpec(meta_cur["numeric_columns"], meta_cur["categorical_columns"])
        model = model_suite(feature_spec, args.seed)["catboost"]
        for rep in range(args.repeats):
            rep_seed = args.seed + rep
            for split_type in SPLIT_TYPES:
                train_idx, test_idx, _ = make_split(
                    df,
                    split_type,
                    rep_seed,
                    test_size=args.test_size,
                    composition=X_cur[meta_cur["composition_columns"]],
                    min_test_n=args.min_composition_test_n,
                )
                pred = fit_predict(model, X_cur.loc[train_idx], y.loc[train_idx], X_cur.loc[test_idx])
                rows.append(
                    {
                        "repeat": rep,
                        "split_type": split_type,
                        "model": "catboost",
                        "missingness_indicators": include_indicators,
                        **regression_metrics(y.loc[test_idx], pred),
                    }
                )
    pd.DataFrame(rows).to_csv(dirs["tables"] / "supp_table_missingness_indicator_ablation.csv", index=False)


def run_feature_group_ablation(df, y, X, meta, metrics, args, dirs) -> None:
    rows = []
    full = metrics[metrics["model"] == "catboost"].copy()
    full["feature_set"] = "full"
    rows.extend(full.to_dict("records"))

    for feature_set, selected_cols in feature_ablation_sets(meta).items():
        numeric_cols = [c for c in meta["numeric_columns"] if c in selected_cols]
        categorical_cols = [c for c in meta["categorical_columns"] if c in selected_cols]
        feature_spec = FeatureSpec(numeric_cols, categorical_cols)
        model = model_suite(feature_spec, args.seed)["catboost"]
        X_cur = X[selected_cols]
        for rep in range(args.repeats):
            rep_seed = args.seed + rep
            for split_type in SPLIT_TYPES:
                train_idx, test_idx, _ = make_split(
                    df,
                    split_type,
                    rep_seed,
                    test_size=args.test_size,
                    composition=X[meta["composition_columns"]],
                    min_test_n=args.min_composition_test_n,
                )
                pred = fit_predict(model, X_cur.loc[train_idx], y.loc[train_idx], X_cur.loc[test_idx])
                rows.append(
                    {
                        "repeat": rep,
                        "seed": rep_seed,
                        "split_type": split_type,
                        "model": "catboost",
                        "feature_set": feature_set,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        **regression_metrics(y.loc[test_idx], pred),
                    }
                )
    ablation = pd.DataFrame(rows)
    ablation.to_csv(dirs["tables"] / "supp_table_s17_feature_group_ablation.csv", index=False)
    summarize_with_ci(ablation, ["split_type", "feature_set"], ["mae", "rmse", "r2", "spearman"]).to_csv(
        dirs["tables"] / "supp_table_s17_feature_group_ablation_summary.csv", index=False
    )


def feature_ablation_sets(meta: dict) -> dict[str, list[str]]:
    composition = list(meta["composition_columns"])
    morphology = ["length", "width", "depth", "syngony", "surface"]
    protocol = [
        "activity",
        "reaction_type",
        "ph",
        "temperature",
        "c_min",
        "c_max",
        "c_const",
        "c_const_unit",
    ]

    def existing(cols: list[str]) -> list[str]:
        allowed = set(meta["numeric_columns"]) | set(meta["categorical_columns"])
        return [c for c in cols if c in allowed]

    return {
        "composition_only": existing(composition),
        "composition_plus_morphology": existing(composition + morphology),
        "composition_plus_protocol": existing(composition + protocol),
    }


if __name__ == "__main__":
    main()
