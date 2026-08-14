#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, write_run_manifest
from src.reporting import ensure_output_dirs


def run_step(args: list[str]) -> None:
    print("Running:", " ".join(args))
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/default_config.json")
    parser.add_argument("--quick", action="store_true", help="Use one repeat and 10 few-shot repeats.")
    parser.add_argument("--skip-conformal", action="store_true")
    parser.add_argument("--skip-missingness-analysis", action="store_true")
    parser.add_argument("--skip-feature-ablation", action="store_true")
    args = parser.parse_args()

    if not args.data.strip():
        parser.error(
            '--data was empty. If using --data "$DATA", set DATA first '
            "or pass a file path such as --data data/nanozymes.csv."
        )

    cfg = load_config(args.config)
    dirs = ensure_output_dirs()
    write_run_manifest(cfg, dirs["metrics"] / "run_manifest.json", args.data)
    repeats = 1 if args.quick else cfg.repeats
    few_shot_repeats = 10 if args.quick else cfg.few_shot_repeats

    py = sys.executable
    run_step(
        [
            py,
            "scripts/preflight.py",
            "--data",
            args.data,
            "--seed",
            str(cfg.random_seed),
            "--test-size",
            str(cfg.test_size),
            "--min-composition-test-n",
            str(cfg.min_composition_test_n),
        ]
    )
    run_step([py, "scripts/audit_data.py", "--data", args.data])
    benchmark_cmd = [
        py,
        "scripts/run_benchmark.py",
        "--data",
        args.data,
        "--repeats",
        str(repeats),
        "--seed",
        str(cfg.random_seed),
        "--test-size",
        str(cfg.test_size),
        "--min-composition-test-n",
        str(cfg.min_composition_test_n),
    ]
    if args.skip_missingness_analysis:
        benchmark_cmd.append("--skip-missingness-analysis")
    if args.skip_feature_ablation:
        benchmark_cmd.append("--skip-feature-ablation")
    run_step(benchmark_cmd)
    if not args.skip_conformal:
        run_step(
            [
                py,
                "scripts/run_conformal.py",
                "--data",
                args.data,
                "--repeats",
                str(repeats),
                "--seed",
                str(cfg.random_seed),
                "--alpha",
                str(cfg.conformal_alpha),
                "--few-shot-repeats",
                str(few_shot_repeats),
            ]
        )
    run_step([py, "scripts/make_figures.py"])
    print("Full pipeline complete.")


if __name__ == "__main__":
    main()
