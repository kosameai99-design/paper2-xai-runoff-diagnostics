#!/usr/bin/env python3
"""Build compact Group B and Group C tables from released JSON sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


RAW_TO_GROUP = {
    "Precipitation": "Precipitation",
    "Temperature": "Air temperature",
    "Solar Radiation": "Radiation",
    "Snow Depth Water Equivalent": "Snow physics",
    "Snow Cover": "Snow physics",
    "Snowfall": "Snow physics",
    "Snow Melt": "Snow physics",
    "Volumetric Soil Water Layer 1": "Soil water",
    "Volumetric Soil Water Deep": "Soil water",
    "Soil Temperature": "Soil temperature",
    "Wind Speed": "Wind",
    "Potential Evapotranspiration": "Potential evapotranspiration",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.package_root

    group_b_dir = root / "derived_results/attribution/groupB_temporal"
    seasonal = json.loads((group_b_dir / "seasonal_patterns.json").read_text())
    seasonal_rows = []
    for variable, seasons in seasonal.items():
        for season, metrics in seasons.items():
            seasonal_rows.append({"variable": variable, "season": season, **metrics})
    pd.DataFrame(seasonal_rows).to_csv(group_b_dir / "groupB_seasonal_raw_variables.csv", index=False)

    group_c_dir = root / "derived_results/attribution/groupC_basin"
    basin_rows = []
    for path in sorted(group_c_dir.glob("basin_*/detailed_analysis.json")):
        data = json.loads(path.read_text())
        grouped = {name: 0.0 for name in sorted(set(RAW_TO_GROUP.values()))}
        for variable, metrics in data["variable_stats"].items():
            group = RAW_TO_GROUP.get(variable)
            if group:
                grouped[group] += float(metrics.get("mean_total_abs_physical", 0.0))
        basin_rows.append(
            {
                "basin_id": str(data["basin_id"]).zfill(3),
                "dominant_raw_variable": data.get("dominant_variable"),
                **grouped,
            }
        )
    pd.DataFrame(basin_rows).to_csv(group_c_dir / "groupC_basin_fingerprints.csv", index=False)
    print(f"Group B rows: {len(seasonal_rows)}")
    print(f"Group C basins: {len(basin_rows)}")


if __name__ == "__main__":
    main()

