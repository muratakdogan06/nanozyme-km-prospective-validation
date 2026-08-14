#!/usr/bin/env python
"""Combine the two retained legacy full-schema diagnostics into one SI figure."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_tmp_cache = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(_tmp_cache / "nanozyme_mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(_tmp_cache / "nanozyme_xdg_cache"))

import matplotlib.pyplot as plt
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "outputs" / "figures" / "publication"
PANEL_PATHS = (
    FIGURE_DIR / "fig6_external_predicted_vs_observed_catboost.png",
    FIGURE_DIR / "figS1_external_scenario_metrics.png",
)
OUTPUT_STEM = FIGURE_DIR / "figS2_legacy_schema_mismatch_audit"


def crop_white_margin(image: Image.Image, padding: int = 16) -> Image.Image:
    """Remove only the uniform outer white margin while preserving plot content."""
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, "white")
    bbox = ImageChops.difference(rgb, background).getbbox()
    if bbox is None:
        return rgb
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(rgb.width, right + padding)
    bottom = min(rgb.height, bottom + padding)
    return rgb.crop((left, top, right, bottom))


def main() -> None:
    images = [crop_white_margin(Image.open(path)) for path in PANEL_PATHS]
    figure = plt.figure(figsize=(11.8, 5.25), facecolor="white")
    grid = figure.add_gridspec(1, 2, width_ratios=(1.0, 1.55), wspace=0.035)

    for label, image, cell in zip(("A", "B"), images, grid):
        axis = figure.add_subplot(cell)
        axis.imshow(image)
        axis.set_axis_off()
        axis.text(
            0.006,
            0.995,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.9},
        )

    figure.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT_STEM.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {OUTPUT_STEM.with_suffix('.png')}")


if __name__ == "__main__":
    main()
