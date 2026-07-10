#!/usr/bin/env python3
"""
Generate single-basin IG summaries (Groups D/E/F)
================================================

This utility extends the existing ABC analysis by producing seasonal,
multi-year daily and full daily series summaries for selected basins.
It relies entirely on the processed IG arrays created by
``preprocess_ig_data.py`` and does not recompute gradients.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - fallback for environments without matplotlib
    plt = None


VARIABLE_NAMES = [
    'Precipitation',
    'Temperature',
    'Solar Radiation',
    'Snow Depth Water Equivalent',
    'Snow Cover',
    'Snowfall',
    'Volumetric Soil Water Layer 1',
    'Volumetric Soil Water Deep',
    'Soil Temperature',
    'Wind Speed',
    'Potential Evapotranspiration'
]

# Default colors reused across plots for visual consistency
DEFAULT_COLORS = [
    "#31688e",
    "#35b779",
    "#fde725",
    "#440154",
    "#ff7f0e",
    "#bc3754",
    "#8c564b",
    "#17becf",
    "#7f7f7f",
]

# Variable grouping for aggregation (swe merged into snow_physics)
VARIABLE_GROUPS = {
    "Precipitation": "pr",
    "Temperature": "temp",
    "Solar Radiation": "radiation",
    "Snow Depth Water Equivalent": "snow_physics",
    "Snow Cover": "snow_physics",
    "Snowfall": "snow_physics",
    "Volumetric Soil Water Layer 1": "soil_water",
    "Volumetric Soil Water Deep": "soil_water",
    "Soil Temperature": "soil_temp",
    "Wind Speed": "wind",
    "Potential Evapotranspiration": "pet",
}

# Aggregated variable names (in display order; swe merged into snow_physics)
AGGREGATED_NAMES = [
    "pr",
    "temp",
    "snow_physics",
    "radiation",
    "soil_temp",
    "soil_water",
    "wind",
    "pet"
]

# Colors for aggregated variables (intuitive natural colors)
AGGREGATED_COLORS = {
    "pr": "#1e40af",  # Precipitation
    "temp": "#f97316",  # Temperature
    "snow_physics": "#60a5fa",  # Snow Physics (incl. SWE, Snow Cover, Snowfall)
    "radiation": "#fbbf24",  # Solar Radiation
    "soil_temp": "#facc15",  # Soil Temperature
    "soil_water": "#92400e",  # Soil Water
    "wind": "#22c55e",  # Wind
    "pet": "#ec4899",  # Potential Evapotranspiration
}

# Formal names for aggregated variables (for display in legends and labels)
AGGREGATED_FORMAL_NAMES = {
    "pr": "Precipitation",
    "temp": "Temperature",
    "snow_physics": "Snow Physics",
    "radiation": "Solar Radiation",
    "soil_temp": "Soil Temperature",
    "soil_water": "Soil Water",
    "wind": "Wind",
    "pet": "Potential Evapotranspiration",
}


@dataclass
class BasinIGData:
    basin_id: str
    array: np.ndarray  # shape: (samples, timesteps, variables)
    dates: pd.DatetimeIndex

    @property
    def sample_importance(self) -> np.ndarray:
        """Sum of absolute IG per variable per sample (collapse time axis).
        
        CORRECTED: Sum over time dimension first (axis=1), matching Case A/D/E/F methodology.
        This gives total IG contribution per sample, which will be averaged later.
        """
        return np.sum(np.abs(self.array), axis=1)


class SingleBasinViewGenerator:
    """Create seasonal / daily summaries for multiple basins."""

    def __init__(
        self,
        run_dir: Path,
        basins: list[str],
        output_dir: Path,
        start_date: pd.Timestamp | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.basins = basins
        self.output_dir = output_dir
        self.start_date = start_date  # May be None, will be inferred per basin

        # For test period, processed data is in test/processed_data
        # Check if we're in test directory structure
        test_processed_dir = self.run_dir / "ig_outputs" / "test" / "processed_data"
        validation_processed_dir = self.run_dir / "ig_outputs" / "processed_data"
        
        if test_processed_dir.exists():
            self.processed_dir = test_processed_dir
            print(f"[INFO] Using TEST period processed data: {self.processed_dir}")
        elif validation_processed_dir.exists():
            self.processed_dir = validation_processed_dir
            print(f"[INFO] Using VALIDATION period processed data: {self.processed_dir}")
        else:
            raise FileNotFoundError(
                f"Processed IG data not found. Checked:\n"
                f"  - {test_processed_dir}\n"
                f"  - {validation_processed_dir}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config for date inference
        self.config_path = self.run_dir / "config.yml"
        self.config = None
        if self.config_path.exists() and yaml is not None:
            with open(self.config_path, "r", encoding="utf-8") as fp:
                self.config = yaml.safe_load(fp)
        
        # Load sigma_QObs for unit conversion (IG -> m^3/s)
        self.sigma_qobs = self._try_load_sigma_qobs()
        if self.sigma_qobs is not None:
            print(f"[UNIT] sigma_QObs loaded: {self.sigma_qobs:.6f} (IG -> m^3/s)")
        else:
            print("[UNIT] sigma_QObs not found. IG values will remain in normalized units.")
    
    def _try_load_sigma_qobs(self):
        """Best-effort loader for sigma_QObs from train_data_scaler.yml (run directory)."""
        scaler_candidates = []
        try:
            # Check from run_dir
            for cand in [
                self.run_dir / "train_data_scaler.yml",
                self.run_dir / "train_data" / "train_data_scaler.yml",
            ]:
                if cand.exists():
                    scaler_candidates.append(cand)
                    break
        except Exception:
            pass
        
        # Also check from script location upwards
        try:
            this_file = Path(__file__).resolve()
            for parent in [this_file.parent] + list(this_file.parents):
                for cand in [
                    parent / "train_data_scaler.yml",
                    parent / "train_data" / "train_data_scaler.yml",
                ]:
                    if cand.exists():
                        scaler_candidates.append(cand)
                        break
                if scaler_candidates:
                    break
        except Exception:
            pass

        scaler_path = scaler_candidates[0] if scaler_candidates else None
        if scaler_path is None:
            return None

        # Load YAML
        try:
            if yaml is not None:
                with scaler_path.open("r", encoding="utf-8") as f:
                    scaler = yaml.safe_load(f)
            else:
                import yaml as pyyaml
                with scaler_path.open("r", encoding="utf-8") as f:
                    scaler = pyyaml.safe_load(f)
        except Exception:
            return None

        # Extract sigma_QObs from scaler
        for key_path in [
            ("xarray_feature_scale", "data_vars", "QObs", "data"),
            ("xarray_feature_scale", "QObs", "data"),
        ]:
            try:
                cur = scaler
                for k in key_path:
                    cur = cur[k]
                return float(cur)
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------ #
    # Variable aggregation helper
    # ------------------------------------------------------------------ #
    def _aggregate_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate variables into groups."""
        aggregated = {}
        for group_name in AGGREGATED_NAMES:
            # Find all variables in this group
            group_vars = [var for var, group in VARIABLE_GROUPS.items() if group == group_name]
            if group_vars:
                # Sum the IG values for all variables in this group
                aggregated[group_name] = df[group_vars].sum(axis=1)
        return pd.DataFrame(aggregated, index=df.index)

    # ------------------------------------------------------------------ #
    # Loading & preparation helpers
    # ------------------------------------------------------------------ #
    def _infer_period_start(self, num_samples: int) -> pd.Timestamp:
        """Infer period start date based on sample count and config."""
        if self.start_date is not None:
            return self.start_date
        
        # Auto-detect based on sample count
        if num_samples == 730:
            # Test period: 2002-01-01 to 2003-12-31 (730 days)
            if self.config and "test_start_date" in self.config:
                date_str = self.config["test_start_date"]
                return pd.to_datetime(date_str, format="%d/%m/%Y")
            return pd.Timestamp("2002-01-01")
        elif num_samples == 731:
            # Validation period: 2000-01-01 to 2001-12-31 (731 days, includes leap year)
            if self.config and "validation_start_date" in self.config:
                date_str = self.config["validation_start_date"]
                return pd.to_datetime(date_str, format="%d/%m/%Y")
            return pd.Timestamp("2000-01-01")
        else:
            # Try validation first, then test
            if self.config and "validation_start_date" in self.config:
                date_str = self.config["validation_start_date"]
                return pd.to_datetime(date_str, format="%d/%m/%Y")
            if self.config and "test_start_date" in self.config:
                date_str = self.config["test_start_date"]
                return pd.to_datetime(date_str, format="%d/%m/%Y")
        
        # Fallback default (validation period)
        return pd.Timestamp("2000-01-01")
    def load_basin(self, basin_id: str) -> BasinIGData:
        basin_id = f"{int(basin_id):03d}"
        basin_dir = self.processed_dir / f"basin_{basin_id}"
        array_path = basin_dir / "ig_data_combined.npy"
        stats_path = basin_dir / "summary_stats.json"

        if not array_path.exists():
            raise FileNotFoundError(f"Missing IG array for basin {basin_id}")

        data = np.load(array_path)
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            num_samples = stats.get("total_samples", data.shape[0])
        else:
            num_samples = data.shape[0]

        # Validate that num_samples matches actual data shape
        if num_samples != data.shape[0]:
            print(f"[WARN] Basin {basin_id}: stats.json reports {num_samples} samples, "
                  f"but array has {data.shape[0]} samples. Using array shape.")
            num_samples = data.shape[0]

        # Infer period start date based on sample count and config
        # 731 samples = validation period (2000-2001), 730 samples = test period (2002-2003)
        period_start = self._infer_period_start(num_samples)
        dates = pd.date_range(
            start=period_start, periods=num_samples, freq="D"
        )
        print(f"[INFO] Basin {basin_id}: Using period {period_start.date()} to {dates[-1].date()} ({num_samples} days)")
        return BasinIGData(basin_id=basin_id, array=data, dates=dates)

    # ------------------------------------------------------------------ #
    # Group D: single basin seasonal stats
    # ------------------------------------------------------------------ #
    def compute_seasonal(self, basin: BasinIGData) -> pd.DataFrame:
        sample_imp = basin.sample_importance  # (samples, variables)
        df = pd.DataFrame(sample_imp, columns=VARIABLE_NAMES, index=basin.dates)
        # Aggregate variables into groups
        df_agg = self._aggregate_variables(df)
        seasons = pd.Series(pd.Categorical(
            np.select(
                [
                    df_agg.index.month.isin([12, 1, 2]),
                    df_agg.index.month.isin([3, 4, 5]),
                    df_agg.index.month.isin([6, 7, 8]),
                    df_agg.index.month.isin([9, 10, 11]),
                ],
                ["Winter", "Spring", "Summer", "Autumn"],
                default="Unknown",
            ),
            categories=["Spring", "Summer", "Autumn", "Winter"],
            ordered=True,
        ), index=df_agg.index)

        # Explicit observed=False to keep current pandas behavior and silence warning
        seasonal = df_agg.groupby(seasons, observed=False).mean()
        return seasonal

    def plot_seasonal(self, seasonal: pd.DataFrame, out_path: Path, ylabel: str = "Mean |IG|") -> None:
        """Plot seasonal contributions with customizable ylabel."""
        if plt is None:
            print("[WARN] matplotlib not available, skipping seasonal plot.")
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        # Get top 5 variables and combine the rest as "other"
        top_vars = seasonal.sum().nlargest(5).index.tolist()
        other_vars = [v for v in seasonal.columns if v not in top_vars]
        
        # Create a new dataframe with top 5 + other
        seasonal_plot = seasonal[top_vars].copy()
        if other_vars:
            seasonal_plot["other"] = seasonal[other_vars].sum(axis=1)
        
        # Plot as stacked bars, with each season sorted by importance (descending)
        x = np.arange(len(seasonal_plot.index))
        # Collect all variables that appear in any season for legend
        legend_handles = {}
        
        for season_idx, season in enumerate(seasonal_plot.index):
            # Get values for this season
            season_values = seasonal_plot.loc[season]
            
            # Separate "other" from top5 variables
            other_value = season_values.get("other", 0)
            top5_values = season_values.drop("other") if "other" in season_values.index else season_values
            
            # Sort top5 by importance (descending: most important at bottom)
            top5_sorted = top5_values.sort_values(ascending=False)
            
            bottom = 0
            # First, plot top5 variables (most important at bottom)
            for var in top5_sorted.index:
                color = AGGREGATED_COLORS.get(var, "#999999")
                bar = ax.bar(
                    x[season_idx],
                    top5_sorted[var],
                    0.85,
                    bottom=bottom,
                    label=var,  # Always set label
                    color=color,
                    alpha=0.9,
                )
                # Store handle for legend (use first occurrence, and set label explicitly)
                if var not in legend_handles:
                    legend_handles[var] = (bar[0], var)
                bottom += top5_sorted[var]
            
            # Finally, plot "other" on top (always gray)
            if other_value > 0:
                bar = ax.bar(
                    x[season_idx],
                    other_value,
                    0.85,
                    bottom=bottom,
                    label="other",  # Always set label
                    color="#808080",  # Gray for other
                    alpha=0.9,
                )
                if "other" not in legend_handles:
                    legend_handles["other"] = (bar[0], "other")
        
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Season")
        ax.set_title("Seasonal variable importance")
        ax.set_xticks(x)
        ax.set_xticklabels(list(seasonal_plot.index))
        # Create legend from collected handles with explicit labels
        from matplotlib.patches import Patch
        legend_elements = []
        for var_name in legend_handles.keys():  # Use keys() to get variable names directly
            color = AGGREGATED_COLORS.get(var_name, "#999999") if var_name != "other" else "#808080"
            formal_name = AGGREGATED_FORMAL_NAMES.get(var_name, var_name)
            legend_elements.append(Patch(facecolor=color, label=formal_name))
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=2)
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Group E: multi-year mean daily profile (per DOY)
    # ------------------------------------------------------------------ #
    def compute_doy_profile(self, basin: BasinIGData) -> pd.DataFrame:
        sample_imp = basin.sample_importance
        df = pd.DataFrame(sample_imp, columns=VARIABLE_NAMES, index=basin.dates)
        # Aggregate variables into groups
        df_agg = self._aggregate_variables(df)
        df_agg["doy"] = df_agg.index.dayofyear
        doy_profile = df_agg.groupby("doy").mean().sort_index()
        return doy_profile

    def plot_doy_profile(self, doy_profile: pd.DataFrame, out_path: Path, ylabel: str = "Mean |IG|") -> None:
        if plt is None:
            print("[WARN] matplotlib not available, skipping DOY plot.")
            return
        total_series = doy_profile.sum(axis=1)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            doy_profile.index, total_series, color="#1f77b4", label="Total importance"
        )
        # Plot top 3 aggregated variables
        top_vars = doy_profile.sum().nlargest(3).index.tolist()
        for var in top_vars:
            color = AGGREGATED_COLORS.get(var, "#999999")
            formal_name = AGGREGATED_FORMAL_NAMES.get(var, var)
            ax.plot(doy_profile.index, doy_profile[var], label=formal_name, alpha=0.7, color=color, linewidth=1.6)
        ax.set_xlabel("Day of year")
        ax.set_ylabel(ylabel)
        ax.set_title("Multi-year daily mean importance")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Group F: full daily series across the validation period
    # ------------------------------------------------------------------ #
    def compute_daily_series(self, basin: BasinIGData) -> pd.DataFrame:
        sample_imp = basin.sample_importance
        df = pd.DataFrame(sample_imp, columns=VARIABLE_NAMES, index=basin.dates)
        # Aggregate variables into groups
        df_agg = self._aggregate_variables(df)
        df_agg["total"] = df_agg.sum(axis=1)
        return df_agg

    def plot_daily_series(self, df: pd.DataFrame, out_path: Path, ylabel: str = "|IG|") -> None:
        if plt is None:
            print("[WARN] matplotlib not available, skipping daily series plot.")
            return
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(df.index, df["total"], label="Total |IG|", color="#1f77b4", linewidth=1.6)
        # Plot top 3 aggregated variables
        top_vars = df.drop("total", axis=1).sum().nlargest(3).index.tolist()
        for var in top_vars:
            color = AGGREGATED_COLORS.get(var, "#999999")
            formal_name = AGGREGATED_FORMAL_NAMES.get(var, var)
            ax.plot(df.index, df[var], label=formal_name, alpha=0.7, color=color, linewidth=1.0)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Date")
        ax.set_title("Daily IG series")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        # Fix x-axis label crowding: format dates and rotate labels
        if hasattr(df.index, 'to_pydatetime'):
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Every 3 months
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    def plot_daily_series_enhanced(self, df: pd.DataFrame, out_path: Path, ylabel: str = "|IG|") -> None:
        """Two-row daily series: top=total+top 3 vars, bottom=all aggregated variables."""
        if plt is None:
            print("[WARN] matplotlib not available, skipping enhanced daily series plot.")
            return
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8.5), sharex=True)
        # Top: total + top 3 aggregated vars
        ax1.plot(df.index, df["total"], label="Total |IG|", color="#1f77b4", linewidth=1.6, alpha=0.9)
        top_vars = df.drop("total", axis=1).sum().nlargest(3).index.tolist()
        for var in top_vars:
            color = AGGREGATED_COLORS.get(var, "#999999")
            formal_name = AGGREGATED_FORMAL_NAMES.get(var, var)
            ax1.plot(df.index, df[var], label=formal_name, color=color, linewidth=1.0, alpha=0.8)
        ax1.set_ylabel(ylabel)
        ax1.set_title("Case F: Daily IG Series – Total and Main Variables")
        ax1.legend(fontsize=9, loc="upper left")
        ax1.grid(alpha=0.25)
        # Bottom: all aggregated variables
        for var in AGGREGATED_NAMES:
            if var in df.columns:
                color = AGGREGATED_COLORS.get(var, "#999999")
                formal_name = AGGREGATED_FORMAL_NAMES.get(var, var)
                ax2.plot(df.index, df[var], label=formal_name, color=color, linewidth=0.9, alpha=0.7)
        ax2.set_xlabel("Date")
        ax2.set_ylabel(ylabel)
        ax2.set_title("All Aggregated Variables")
        ax2.legend(fontsize=8, ncol=4, loc="upper left")
        ax2.grid(alpha=0.25)
        # Fix x-axis label crowding: format dates and rotate labels (only on bottom axis since sharex=True)
        if hasattr(df.index, 'to_pydatetime'):
            import matplotlib.dates as mdates
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Every 3 months
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Composite: three-panel D/E/F summary
    # ------------------------------------------------------------------ #
    def plot_def_summary(
        self,
        basin_id: str,
        seasonal: pd.DataFrame,
        doy_profile: pd.DataFrame,
        daily_series: pd.DataFrame,
        out_path: Path,
    ) -> None:
        if plt is None:
            print("[WARN] matplotlib not available, skipping D/E/F summary plot.")
            return

        # Use aggregated colors
        color_map = AGGREGATED_COLORS.copy()

        # Use a 2x3 grid; bottom row spans all columns for the "all variables" plot
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(16, 8.5))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.3)

        # Left: Case D stacked bars (top 5 + other), sorted by importance per season
        ax = fig.add_subplot(gs[0, 0])
        # Get top 5 aggregated variables and combine the rest as "other"
        top_vars = seasonal.sum().nlargest(5).index.tolist()
        other_vars = [v for v in seasonal.columns if v not in top_vars]
        
        # Create a new dataframe with top 5 + other
        seasonal_plot = seasonal[top_vars].copy()
        if other_vars:
            seasonal_plot["other"] = seasonal[other_vars].sum(axis=1)
        
        # Plot as stacked bars, with each season sorted by importance (descending)
        x = np.arange(len(seasonal_plot.index))
        # Collect all variables that appear in any season for legend
        legend_handles = {}
        
        for season_idx, season in enumerate(seasonal_plot.index):
            # Get values for this season
            season_values = seasonal_plot.loc[season]
            
            # Separate "other" from top4 variables
            other_value = season_values.get("other", 0)
            top4_values = season_values.drop("other") if "other" in season_values.index else season_values
            
            # Sort top4 by importance (descending: most important at bottom)
            top4_sorted = top4_values.sort_values(ascending=False)
            
            bottom = 0
            # First, plot top4 variables (most important at bottom)
            for var in top4_sorted.index:
                color = color_map.get(var, "#999999")
                bar = ax.bar(
                    x[season_idx],
                    top4_sorted[var],
                    0.75,
                    bottom=bottom,
                    label=var,  # Always set label
                    color=color,
                    alpha=0.9,
                )
                # Store handle for legend (use first occurrence, and set label explicitly)
                if var not in legend_handles:
                    legend_handles[var] = (bar[0], var)
                bottom += top4_sorted[var]
            
            # Finally, plot "other" on top (always gray)
            if other_value > 0:
                bar = ax.bar(
                    x[season_idx],
                    other_value,
                    0.75,
                    bottom=bottom,
                    label="other",  # Always set label
                    color="#808080",  # Gray for other
                    alpha=0.9,
                )
                if "other" not in legend_handles:
                    legend_handles["other"] = (bar[0], "other")
        ax.set_title("Case D: Seasonal Importance")
        ax.set_xlabel("Season")
        ax.set_ylabel("Mean |IG|")
        ax.set_xticks(x)
        ax.set_xticklabels(list(seasonal_plot.index))
        # Create legend from collected handles with explicit labels
        from matplotlib.patches import Patch
        legend_elements = []
        for var_name in legend_handles.keys():  # Use keys() to get variable names directly
            color = color_map.get(var_name, "#999999") if var_name != "other" else "#808080"
            formal_name = AGGREGATED_FORMAL_NAMES.get(var_name, var_name)
            legend_elements.append(Patch(facecolor=color, label=formal_name))
        ax.legend(handles=legend_elements, fontsize=8)
        ax.grid(alpha=0.25, axis="y")

        # Middle: Case E DOY profile (top 3 aggregated variables)
        ax = fig.add_subplot(gs[0, 1])
        top_vars_e = doy_profile.sum().nlargest(3).index.tolist()
        for var in top_vars_e:
            formal_name = AGGREGATED_FORMAL_NAMES.get(var, var)
            ax.plot(
                doy_profile.index,
                doy_profile[var],
                label=formal_name,
                color=color_map.get(var, "#999999"),
                alpha=0.9,
                linewidth=1.6,
            )
        ax.set_title("Case E: Multi-year Daily Mean")
        ax.set_xlabel("Day of year")
        ax.set_ylabel("Mean |IG|")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

        # Right: Case F daily series (total + top 3 aggregated variables)
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(
            daily_series.index,
            daily_series["total"],
            label="Total",
            color="#1f77b4",
            linewidth=1.6,
        )
        top_vars_f = daily_series.drop("total", axis=1).sum().nlargest(3).index.tolist()
        for var in top_vars_f:
            formal_name = AGGREGATED_FORMAL_NAMES.get(var, var)
            ax.plot(
                daily_series.index,
                daily_series[var],
                label=formal_name,
                color=color_map.get(var, "#999999"),
                alpha=0.8,
                linewidth=1.0,
            )
        ax.set_title("Case F: Daily Series")
        ax.set_xlabel("Date")
        ax.set_ylabel("|IG|")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        # Fix x-axis label crowding: format dates and rotate labels
        if hasattr(daily_series.index, 'to_pydatetime'):
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Every 3 months
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)

        # Bottom: All aggregated variables time series
        ax = fig.add_subplot(gs[1, :])
        for var in AGGREGATED_NAMES:
            if var in daily_series.columns:
                formal_name = AGGREGATED_FORMAL_NAMES.get(var, var)
                ax.plot(
                    daily_series.index,
                    daily_series[var],
                    label=formal_name,
                    color=color_map.get(var, "#999999"),
                    linewidth=0.9,
                    alpha=0.7,
                )
        ax.set_title("All Aggregated Variables")
        ax.set_xlabel("Date")
        ax.set_ylabel("|IG|")
        ax.legend(fontsize=8, ncol=4, loc="upper left")
        ax.grid(alpha=0.25)
        # Fix x-axis label crowding: format dates and rotate labels
        if hasattr(daily_series.index, 'to_pydatetime'):
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Every 3 months
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

        fig.suptitle(f"Basin {basin_id} – D/E/F Analysis Summary", y=1.02, fontsize=13)
        fig.tight_layout()
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    def plot_def_summary_normalized(
        self,
        basin_id: str,
        seasonal: pd.DataFrame,
        doy_profile: pd.DataFrame,
        daily_series: pd.DataFrame,
        out_path: Path,
    ) -> None:
        """Plot D/E/F summary with physical units (m³/s) - same as plot_def_summary but with unit conversion."""
        if plt is None:
            print("[WARN] matplotlib not available, skipping D/E/F normalized summary plot.")
            return

        # Use aggregated colors
        color_map = AGGREGATED_COLORS.copy()

        # Use a 2x3 grid; bottom row spans all columns for the "all variables" plot
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(16, 8.5))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.3)

        # Left: Case D stacked bars (top 5 + other), sorted by importance per season
        ax = fig.add_subplot(gs[0, 0])
        # Get top 5 aggregated variables and combine the rest as "other"
        top_vars = seasonal.sum().nlargest(5).index.tolist()
        other_vars = [v for v in seasonal.columns if v not in top_vars]
        
        # Create a new dataframe with top 5 + other
        seasonal_plot = seasonal[top_vars].copy()
        if other_vars:
            seasonal_plot["other"] = seasonal[other_vars].sum(axis=1)
        
        # Plot as stacked bars, with each season sorted by importance (descending)
        x = np.arange(len(seasonal_plot.index))
        # Collect all variables that appear in any season for legend
        legend_handles = {}
        
        for season_idx, season in enumerate(seasonal_plot.index):
            # Get values for this season
            season_values = seasonal_plot.loc[season]
            
            # Separate "other" from top4 variables
            other_value = season_values.get("other", 0)
            top4_values = season_values.drop("other") if "other" in season_values.index else season_values
            
            # Sort top4 by importance (descending: most important at bottom)
            top4_sorted = top4_values.sort_values(ascending=False)
            
            bottom = 0
            # First, plot top4 variables (most important at bottom)
            for var in top4_sorted.index:
                color = color_map.get(var, "#999999")
                bar = ax.bar(
                    x[season_idx],
                    top4_sorted[var],
                    0.75,
                    bottom=bottom,
                    label=var,  # Always set label
                    color=color,
                    alpha=0.9,
                )
                # Store handle for legend (use first occurrence, and set label explicitly)
                if var not in legend_handles:
                    legend_handles[var] = (bar[0], var)
                bottom += top4_sorted[var]
            
            # Finally, plot "other" on top (always gray)
            if other_value > 0:
                bar = ax.bar(
                    x[season_idx],
                    other_value,
                    0.75,
                    bottom=bottom,
                    label="other",  # Always set label
                    color="#808080",  # Gray for other
                    alpha=0.9,
                )
                if "other" not in legend_handles:
                    legend_handles["other"] = (bar[0], "other")
        ax.set_title("Case D: Seasonal Importance")
        ax.set_xlabel("Season")
        ax.set_ylabel("IG_q (m³/s)")
        ax.set_xticks(x)
        ax.set_xticklabels(list(seasonal_plot.index))
        # Create legend from collected handles with explicit labels
        from matplotlib.patches import Patch
        legend_elements = []
        for var_name in legend_handles.keys():  # Use keys() to get variable names directly
            color = color_map.get(var_name, "#999999") if var_name != "other" else "#808080"
            formal_name = AGGREGATED_FORMAL_NAMES.get(var_name, var_name)
            legend_elements.append(Patch(facecolor=color, label=formal_name))
        ax.legend(handles=legend_elements, fontsize=8)
        ax.grid(alpha=0.25, axis="y")

        # Middle: Case E DOY profile (top 3 aggregated variables)
        ax = fig.add_subplot(gs[0, 1])
        top_vars_e = doy_profile.sum().nlargest(3).index.tolist()
        for var in top_vars_e:
            formal_name = AGGREGATED_FORMAL_NAMES.get(var, var)
            ax.plot(
                doy_profile.index,
                doy_profile[var],
                label=formal_name,
                color=color_map.get(var, "#999999"),
                alpha=0.9,
                linewidth=1.6,
            )
        ax.set_title("Case E: Multi-year Daily Mean")
        ax.set_xlabel("Day of year")
        ax.set_ylabel("IG_q (m³/s)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

        # Right: Case F daily series (total + top 3 aggregated variables)
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(
            daily_series.index,
            daily_series["total"],
            label="Total",
            color="#1f77b4",
            linewidth=1.6,
        )
        top_vars_f = daily_series.drop("total", axis=1).sum().nlargest(3).index.tolist()
        for var in top_vars_f:
            formal_name = AGGREGATED_FORMAL_NAMES.get(var, var)
            ax.plot(
                daily_series.index,
                daily_series[var],
                label=formal_name,
                color=color_map.get(var, "#999999"),
                alpha=0.8,
                linewidth=1.0,
            )
        ax.set_title("Case F: Daily Series")
        ax.set_xlabel("Date")
        ax.set_ylabel("IG_q (m³/s)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        # Fix x-axis label crowding: format dates and rotate labels
        if hasattr(daily_series.index, 'to_pydatetime'):
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Every 3 months
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)

        # Bottom: All aggregated variables time series
        ax = fig.add_subplot(gs[1, :])
        for var in AGGREGATED_NAMES:
            if var in daily_series.columns:
                formal_name = AGGREGATED_FORMAL_NAMES.get(var, var)
                ax.plot(
                    daily_series.index,
                    daily_series[var],
                    label=formal_name,
                    color=color_map.get(var, "#999999"),
                    linewidth=0.9,
                    alpha=0.7,
                )
        ax.set_title("All Aggregated Variables")
        ax.set_xlabel("Date")
        ax.set_ylabel("IG_q (m³/s)")
        ax.legend(fontsize=8, ncol=4, loc="upper left")
        ax.grid(alpha=0.25)
        # Fix x-axis label crowding: format dates and rotate labels
        if hasattr(daily_series.index, 'to_pydatetime'):
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Every 3 months
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

        fig.suptitle(f"Basin {basin_id} – D/E/F Analysis Summary (Normalized)", y=1.02, fontsize=13)
        fig.tight_layout()
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------ #
    def process_basin(self, basin_id: str) -> None:
        basin = self.load_basin(basin_id)
        basin_dir = self.output_dir / f"basin_{basin.basin_id}"
        basin_dir.mkdir(exist_ok=True, parents=True)

        # Group D
        seasonal = self.compute_seasonal(basin)
        seasonal.to_csv(basin_dir / "seasonal_contributions.csv")
        self.plot_seasonal(seasonal, basin_dir / "seasonal_contributions.png")
        
        # Group D - Physical units (if sigma_QObs available)
        if self.sigma_qobs is not None:
            seasonal_physical = seasonal * self.sigma_qobs
            seasonal_physical.to_csv(basin_dir / "seasonal_contributions_physical.csv")
            self.plot_seasonal(seasonal_physical, basin_dir / "seasonal_contributions_physical.png", 
                             ylabel="IG_q (m³/s)")

        # Group E
        doy_profile = self.compute_doy_profile(basin)
        doy_profile.to_csv(basin_dir / "doy_mean_contributions.csv")
        self.plot_doy_profile(doy_profile, basin_dir / "doy_mean_contributions.png")
        
        # Group E - Physical units (if sigma_QObs available)
        if self.sigma_qobs is not None:
            doy_profile_physical = doy_profile * self.sigma_qobs
            doy_profile_physical.to_csv(basin_dir / "doy_mean_contributions_physical.csv")
            self.plot_doy_profile(doy_profile_physical, basin_dir / "doy_mean_contributions_physical.png",
                                ylabel="IG_q (m³/s)")

        # Group F
        daily_series = self.compute_daily_series(basin)
        daily_series.to_csv(basin_dir / "daily_series_importance.csv")
        self.plot_daily_series(daily_series, basin_dir / "daily_series_importance.png")
        # Enhanced two-row daily plot
        self.plot_daily_series_enhanced(
            daily_series,
            basin_dir / "daily_series_importance_enhanced.png",
        )
        
        # Group F - Physical units (if sigma_QObs available)
        if self.sigma_qobs is not None:
            daily_series_physical = daily_series * self.sigma_qobs
            daily_series_physical.to_csv(basin_dir / "daily_series_physical.csv")
            self.plot_daily_series(daily_series_physical, basin_dir / "daily_series_physical.png",
                                  ylabel="IG_q (m³/s)")
            self.plot_daily_series_enhanced(
                daily_series_physical,
                basin_dir / "daily_series_physical_enhanced.png",
                ylabel="IG_q (m³/s)"
            )

        # Composite figure with three panels (normalized)
        self.plot_def_summary(
            basin_id=basin.basin_id,
            seasonal=seasonal,
            doy_profile=doy_profile,
            daily_series=daily_series,
            out_path=basin_dir / "def_summary_three_panels.png",
        )
        
        # Composite figure with physical units (def_summary_normalized.png)
        if self.sigma_qobs is not None:
            self.plot_def_summary_normalized(
                basin_id=basin.basin_id,
                seasonal=seasonal * self.sigma_qobs,
                doy_profile=doy_profile * self.sigma_qobs,
                daily_series=daily_series * self.sigma_qobs,
                out_path=basin_dir / "def_summary_normalized.png",
            )

        print(f"[DONE] Basin {basin.basin_id}: outputs stored in {basin_dir}")

    def run(self) -> None:
        for basin in self.basins:
            try:
                self.process_basin(basin)
            except FileNotFoundError as exc:
                print(f"[WARN] {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Groups D/E/F IG summaries for selected basins."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("."),
        help="Path to run directory (default: current directory).",
    )
    parser.add_argument(
        "--basins",
        type=str,
        nargs="+",
        default=["001", "050", "100"],
        help="List of basin IDs (numeric or zero-padded).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all basins found under <run-dir>/ig_outputs/processed_data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for figures/CSVs (default: ig_outputs/analysis/single_basin_extended).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to config.yml to infer period start date (validation or test).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Override period start date (format YYYY-MM-DD). Auto-detected from sample count if not provided.",
    )
    return parser.parse_args()




def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    # For test period, output to test/analysis/single_basin_extended
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        # Check if test directory exists
        test_output_dir = run_dir / "ig_outputs" / "test" / "analysis" / "single_basin_extended"
        validation_output_dir = run_dir / "ig_outputs" / "analysis" / "single_basin_extended"
        
        if (run_dir / "ig_outputs" / "test" / "processed_data").exists():
            output_dir = test_output_dir
            print(f"[INFO] Output directory: {output_dir} (TEST period)")
        else:
            output_dir = validation_output_dir
            print(f"[INFO] Output directory: {output_dir} (VALIDATION period)")

    # Start date will be inferred per basin based on sample count
    # If explicitly provided, it will be used for all basins
    start_date = None
    if args.start_date:
        start_date = pd.Timestamp(args.start_date)
    
    # Resolve basin list
    basins = args.basins
    if args.all:
        # Check both test and validation directories
        test_processed_dir = run_dir / "ig_outputs" / "test" / "processed_data"
        validation_processed_dir = run_dir / "ig_outputs" / "processed_data"
        
        if test_processed_dir.exists():
            processed_dir = test_processed_dir
        elif validation_processed_dir.exists():
            processed_dir = validation_processed_dir
        else:
            raise FileNotFoundError(
                f"Processed IG data not found. Checked:\n"
                f"  - {test_processed_dir}\n"
                f"  - {validation_processed_dir}"
            )
        # Find all basin_* folders and extract numeric id
        found = []
        for p in processed_dir.glob("basin_*"):
            name = p.name
            try:
                num = int(name.split("_")[-1])
                found.append(f"{num:03d}")
            except ValueError:
                continue
        basins = sorted(set(found))
        if not basins:
            raise FileNotFoundError("No basin_* folders found under processed_data.")

    gen = SingleBasinViewGenerator(
        run_dir=run_dir,
        basins=basins,
        output_dir=output_dir,
        start_date=start_date,
    )
    gen.run()


if __name__ == "__main__":
    main()
