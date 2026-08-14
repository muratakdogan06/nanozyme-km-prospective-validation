#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting import ensure_output_dirs, make_figures


def main() -> None:
    dirs = ensure_output_dirs()
    make_figures(
        Path(dirs["metrics"] / "benchmark_metrics.csv"),
        Path(dirs["metrics"] / "conformal_metrics.csv"),
        dirs["figures"],
    )
    print(f"Figures written to {dirs['figures']}")


if __name__ == "__main__":
    main()
