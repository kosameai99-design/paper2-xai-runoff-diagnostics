#!/usr/bin/env python3
"""Recompute compact Paper 2 performance and response-consistency summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau, spearmanr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.package_root
    output = args.output or root / "derived_results" / "reproduced_summary.json"

    metrics = pd.read_csv(root / "derived_results/performance/test_metrics_epoch020.csv")
    ranks = pd.read_csv(root / "derived_results/validation/response_consistency/ig_vs_ablation_ranks.csv")
    summary = {
        "checkpoint": 20,
        "n_basins": int(len(metrics)),
        "NSE_mean": float(metrics["NSE"].mean()),
        "NSE_median": float(metrics["NSE"].median()),
        "KGE_mean": float(metrics["KGE"].mean()),
        "KGE_median": float(metrics["KGE"].median()),
        "IG_vs_ablation_NSE_spearman": float(spearmanr(ranks["IG_rank"], ranks["rank_minus_dNSE"]).statistic),
        "IG_vs_ablation_NSE_kendall": float(kendalltau(ranks["IG_rank"], ranks["rank_minus_dNSE"]).statistic),
        "IG_vs_ablation_KGE_spearman": float(spearmanr(ranks["IG_rank"], ranks["rank_minus_dKGE"]).statistic),
        "IG_vs_ablation_KGE_kendall": float(kendalltau(ranks["IG_rank"], ranks["rank_minus_dKGE"]).statistic),
    }
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

