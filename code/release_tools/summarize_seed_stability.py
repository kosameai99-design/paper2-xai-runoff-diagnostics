#!/usr/bin/env python3
"""Recompute pairwise grouped-importance stability across four seeds."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import kendalltau, spearmanr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.package_root
    source = root / "derived_results/validation/stability/si5_2_per_seed_group_unsigned_ig_importance.csv"
    frame = pd.read_csv(source)
    value_columns = [column for column in frame.columns if column not in {"seed", "run"}]
    rows = []
    for left, right in itertools.combinations(range(len(frame)), 2):
        x = frame.loc[left, value_columns].to_numpy(dtype=float)
        y = frame.loc[right, value_columns].to_numpy(dtype=float)
        rows.append(
            {
                "seed_pair": f"{frame.loc[left, 'seed']} vs {frame.loc[right, 'seed']}",
                "cosine_similarity": 1.0 - cosine(x, y),
                "spearman_rho": spearmanr(x, y).statistic,
                "kendall_tau": kendalltau(x, y).statistic,
            }
        )
    output = args.output or root / "derived_results/validation/stability/reproduced_pairwise_group_stability.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

