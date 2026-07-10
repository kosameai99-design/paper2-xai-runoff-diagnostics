#!/usr/bin/env python3
"""Reproduce main-text Figure 3 from the released Group A table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "pr": "Precipitation",
    "soil_water": "Soil water",
    "snow_physics": "Snow physics",
    "temp": "Air temperature",
    "radiation": "Radiation",
    "pet": "Potential evapotranspiration",
    "soil_temp": "Soil temperature",
    "wind": "Wind",
}
COLORS = {
    "pr": "#2f65d9",
    "soil_water": "#1f78b4",
    "snow_physics": "#6f58c9",
    "temp": "#f28e2b",
    "radiation": "#edc948",
    "pet": "#e15759",
    "soil_temp": "#9c755f",
    "wind": "#59a14f",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.package_root
    output = args.output or root / "figures/reproduced/Figure3.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.4,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.65,
            "savefig.dpi": 400,
            "figure.dpi": 300,
        }
    )

    source = root / "derived_results/attribution/groupA_global/global_feature_importance_aggregated.csv"
    frame = pd.read_csv(source)
    frame["label"] = frame["group"].map(LABELS)
    frame = frame.sort_values("mean_importance", ascending=True)
    fig, ax = plt.subplots(figsize=(6.8, 3.45))
    y = np.arange(len(frame))
    ax.barh(
        y,
        frame["mean_importance"],
        xerr=frame["std_importance"],
        color=[COLORS[group] for group in frame["group"]],
        edgecolor="0.25",
        linewidth=0.34,
        capsize=2.0,
    )
    ax.set_yticks(y, frame["label"])
    ax.set_xlabel("Mean unsigned IG contribution (m³/s)")
    ax.grid(axis="x", color="0.90", lw=0.45)
    ax.set_axisbelow(True)
    for index, value in enumerate(frame["mean_importance"]):
        ax.text(value + 0.06, index + 0.18, f"{value:.2f}", fontsize=6.9)
    ax.set_xlim(0, max(frame["mean_importance"] + frame["std_importance"]) * 1.18)
    fig.tight_layout(pad=0.45)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
