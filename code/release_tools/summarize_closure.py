#!/usr/bin/env python3
"""Check the released closure headline and supporting tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    directory = args.package_root / "derived_results/validation/closure"
    headline = pd.read_csv(directory / "closure_headline_metrics.csv").set_index("metric")
    basin = pd.read_csv(directory / "si5_1_per_basin_closure_distribution_summary.csv").set_index("metric")
    result = {
        "forward_global_mean_mismatch_percent": float(headline.loc["forward_global_mean_mismatch", "value"]),
        "sample_level_NMAE_by_mean_Qsim": float(headline.loc["sample_level_NMAE_by_mean_Qsim", "value"]),
        "reverse_Qbase_consistency_percent": float(headline.loc["reverse_Qbase_consistency", "value"]),
        "per_basin_MAE_median_m3s": float(basin.loc["mae_abs_e_m3s", "basin_distribution_median"]),
    }
    if result["forward_global_mean_mismatch_percent"] != 0.112:
        raise RuntimeError("Forward closure headline mismatch")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

