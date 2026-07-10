#!/usr/bin/env python3
"""
Global IG Analysis Script
========================

This script performs global-level analysis on the preprocessed IG data,
including overall statistics, feature importance ranking, and global patterns.

Author: AI Assistant
Date: 2025-10-17
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class GlobalAnalyzer:
    """Global analysis of IG data across all basins"""
    
    def __init__(self, data_dir=None, output_dir=None):
        # Auto-detect test or validation directory
        # Check if we're running from analysis_scripts directory
        current_dir = Path.cwd()
        is_in_scripts_dir = current_dir.name == "analysis_scripts"
        
        if data_dir is None:
            if is_in_scripts_dir:
                # Running from analysis_scripts/, data is in parent
                self.data_dir = (Path("..") / "processed_data").resolve()
            else:
                # Running from test/, data is in current dir
                self.data_dir = Path("processed_data").resolve()
            print(f"[INFO] Data directory: {self.data_dir}")
        else:
            self.data_dir = Path(data_dir)
        
        if output_dir is None:
            if is_in_scripts_dir:
                # Running from analysis_scripts/, output to parent
                self.output_dir = (Path("..") / "analysis" / "global_analysis").resolve()
            else:
                # Running from test/, output to current dir
                self.output_dir = (Path("analysis") / "global_analysis").resolve()
        else:
            self.output_dir = Path(output_dir) / "global_analysis"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Variable names (17 variables: 9 original + 8 new add_era5land variables)
        self.variable_names = [
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
        
        # Variable grouping for aggregation (swe merged into snow_physics, same as Case D/E/F)
        self.variable_groups = {
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
        
        # Aggregated variable names and colors (swe merged into snow_physics)
        self.aggregated_names = ["pr", "temp", "snow_physics", "radiation", "soil_temp", "soil_water", "wind", "pet"]
        self.aggregated_colors = {
            "pr": "#1e40af",  # Precipitation
            "temp": "#f97316",  # Temperature
            "snow_physics": "#60a5fa",  # Snow Physics (incl. SWE, Snow Cover, Snowfall)
            "radiation": "#fbbf24",  # Solar Radiation
            "soil_temp": "#facc15",  # Soil Temperature
            "soil_water": "#92400e",  # Soil Water
            "wind": "#22c55e",  # Wind
            "pet": "#ec4899",  # Potential Evapotranspiration
        }
        self.aggregated_formal_names = {
            "pr": "Precipitation",
            "temp": "Temperature",
            "snow_physics": "Snow Physics",
            "radiation": "Solar Radiation",
            "soil_temp": "Soil Temperature",
            "soil_water": "Soil Water",
            "wind": "Wind",
            "pet": "Potential Evapotranspiration",
        }

        # Try to load de-normalization coefficient for QObs (sigma_QObs) to convert IG to m^3/s.
        self.sigma_qobs = self._try_load_sigma_qobs()
        if self.sigma_qobs is not None:
            print(f"[UNIT] sigma_QObs loaded: {self.sigma_qobs:.6f} (IG -> m^3/s)")
        else:
            print("[UNIT] sigma_QObs not found. Feature importance will remain in normalized units.")
        
        print("[GLOBAL] Global IG Analysis")
        print(f"[DIR] Data directory: {self.data_dir}")
        print(f"[OUT] Output directory: {self.output_dir}")
        print("=" * 50)

    def _try_load_sigma_qobs(self):
        """Best-effort loader for sigma_QObs from train_data_scaler.yml (run directory)."""
        # Script is typically located at: <run_dir>/ig_outputs/test/analysis_scripts/global_analysis.py
        # The scaler is typically at: <run_dir>/train_data/train_data_scaler.yml
        scaler_candidates = []
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

        # Also check from current working directory upwards
        try:
            cwd = Path.cwd().resolve()
            for parent in [cwd] + list(cwd.parents):
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

        # Load YAML using ruamel.yaml if available, else PyYAML
        try:
            from ruamel.yaml import YAML
            yaml = YAML(typ="safe")
            with scaler_path.open("r", encoding="utf-8") as f:
                scaler = yaml.load(f)
        except Exception:
            try:
                import yaml as pyyaml
                with scaler_path.open("r", encoding="utf-8") as f:
                    scaler = pyyaml.safe_load(f)
            except Exception:
                return None

        # train_data_scaler.yml stores xarray info under xarray_feature_scale/data_vars
        for key_path in [
            ("xarray_feature_scale", "data_vars", "QObs", "data"),
            ("xarray_feature_scale", "QObs", "data"),  # fallback for other formats
        ]:
            try:
                cur = scaler
                for k in key_path:
                    cur = cur[k]
                return float(cur)
            except Exception:
                continue
        return None
    
    def load_global_data(self):
        """Load global summary data"""
        global_summary_path = self.data_dir / "global_summary.json"
        basin_summaries_path = self.data_dir / "basin_summaries.csv"
        
        if not global_summary_path.exists():
            raise FileNotFoundError(f"Global summary not found: {global_summary_path}")
        
        # Load global summary
        with open(global_summary_path, 'r') as f:
            self.global_summary = json.load(f)
        
        # Load basin summaries
        self.basin_df = pd.read_csv(basin_summaries_path)
        
        print(f"[LOAD] Loaded data for {self.global_summary['total_basins']} basins")
        print(f"[LOAD] Total samples: {self.global_summary['total_samples']:,}")
        
        return self.global_summary, self.basin_df
    
    def calculate_global_statistics(self):
        """Calculate comprehensive global statistics"""
        print("[STATS] Calculating global statistics...")
        
        # Basic statistics
        stats = {
            'total_basins': self.global_summary['total_basins'],
            'total_samples': self.global_summary['total_samples'],
            'total_batches': self.global_summary['total_batches'],
            'time_steps': self.global_summary['time_steps'],
            'variables': self.global_summary['variables']
        }
        
        # Global IG statistics
        global_stats = self.global_summary['global_statistics']
        stats.update({
            'global_mean': global_stats['mean'],
            'global_std': global_stats['std'],
            'global_min': global_stats['min'],
            'global_max': global_stats['max'],
            'global_median': global_stats['median']
        })
        
        # Basin-level statistics
        basin_means = self.basin_df['overall_mean'].values
        basin_stds = self.basin_df['overall_std'].values
        
        stats.update({
            'basin_mean_mean': float(np.mean(basin_means)),
            'basin_mean_std': float(np.std(basin_means)),
            'basin_std_mean': float(np.mean(basin_stds)),
            'basin_std_std': float(np.std(basin_stds)),
            'basin_mean_min': float(np.min(basin_means)),
            'basin_mean_max': float(np.max(basin_means))
        })
        
        # Save statistics
        stats_path = self.output_dir / "global_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"[SAVE] Global statistics saved to: {stats_path}")
        return stats
    
    def analyze_feature_importance(self):
        """Analyze global feature importance patterns"""
        print("[FEATURES] Analyzing global feature importance...")
        
        # Use ALL basins in processed_data (not just a handful of samples)
        basin_dirs = sorted([p for p in self.data_dir.glob("basin_*") if p.is_dir()])
        if not basin_dirs:
            raise FileNotFoundError(f"No basin directories found in {self.data_dir}")

        per_basin_importance_abs = []  # Absolute values
        per_basin_importance_signed = []  # Signed values
        used_basins = 0

        for basin_dir in basin_dirs:
            basin_data_path = basin_dir / "ig_data_combined.npy"
            if not basin_data_path.exists():
                continue

            data = np.load(basin_data_path)
            # data shape: [n_samples, seq_length(365), n_vars(17)]
            # We want: mean over samples of (sum over time of abs(IG)) for each variable.
            # Also compute signed version: mean over samples of (sum over time of IG).
            # To limit peak memory, compute in sample chunks.
            chunk_size = 64
            sums_abs = []
            sums_signed = []
            for start in range(0, data.shape[0], chunk_size):
                chunk = data[start:start + chunk_size]  # [chunk, 365, 17]
                sums_abs.append(np.sum(np.abs(chunk), axis=1))  # [chunk, 17]
                sums_signed.append(np.sum(chunk, axis=1))  # [chunk, 17] - signed
            per_sample_sum_abs = np.concatenate(sums_abs, axis=0)  # [n_samples, 17]
            per_sample_sum_signed = np.concatenate(sums_signed, axis=0)  # [n_samples, 17]
            basin_var_importance_abs = np.mean(per_sample_sum_abs, axis=0)  # [17]
            basin_var_importance_signed = np.mean(per_sample_sum_signed, axis=0)  # [17]

            per_basin_importance_abs.append(basin_var_importance_abs)
            per_basin_importance_signed.append(basin_var_importance_signed)
            used_basins += 1

        if used_basins == 0:
            raise FileNotFoundError(f"No ig_data_combined.npy files found under {self.data_dir}")

        per_basin_importance_abs = np.asarray(per_basin_importance_abs)  # [n_basins, 17]
        per_basin_importance_signed = np.asarray(per_basin_importance_signed)  # [n_basins, 17]

        # Convert to physical units if available
        if self.sigma_qobs is not None:
            per_basin_importance_abs = per_basin_importance_abs * self.sigma_qobs
            per_basin_importance_signed = per_basin_importance_signed * self.sigma_qobs

        # Aggregate across basins
        mean_importance = np.mean(per_basin_importance_abs, axis=0)
        std_importance = np.std(per_basin_importance_abs, axis=0)
        mean_importance_signed = np.mean(per_basin_importance_signed, axis=0)
        std_importance_signed = np.std(per_basin_importance_signed, axis=0)

        # Build feature importance dict (absolute)
        feature_importance = {}
        for i, var_name in enumerate(self.variable_names):
            feature_importance[var_name] = {
                'mean_importance': float(mean_importance[i]),
                'std_importance': float(std_importance[i]),
                'min_importance': float(np.min(per_basin_importance_abs[:, i])),
                'max_importance': float(np.max(per_basin_importance_abs[:, i]))
            }
        
        # Build signed feature importance dict
        feature_importance_signed = {}
        for i, var_name in enumerate(self.variable_names):
            feature_importance_signed[var_name] = {
                'mean_importance': float(mean_importance_signed[i]),
                'std_importance': float(std_importance_signed[i]),
                'min_importance': float(np.min(per_basin_importance_signed[:, i])),
                'max_importance': float(np.max(per_basin_importance_signed[:, i]))
            }
        
        # Store signed data for plotting
        self.per_basin_importance_signed = per_basin_importance_signed
        self.mean_importance_signed = mean_importance_signed
        self.std_importance_signed = std_importance_signed
        self.feature_importance_signed = feature_importance_signed
        
        # Sort by mean importance
        sorted_features = sorted(feature_importance.items(), 
                              key=lambda x: x[1]['mean_importance'], 
                              reverse=True)
        
        # Save feature importance
        importance_path = self.output_dir / "global_feature_importance.json"
        with open(importance_path, 'w') as f:
            json.dump(feature_importance, f, indent=2)
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame([
            {
                'variable': var,
                'mean_importance': stats['mean_importance'],
                'std_importance': stats['std_importance'],
                'rank': i+1
            }
            for i, (var, stats) in enumerate(sorted_features)
        ])
        
        importance_df.to_csv(self.output_dir / "global_feature_importance.csv", index=False)
        
        print(f"[SAVE] Feature importance saved to: {importance_path}")
        print(f"[RANK] Top 3 most important variables:")
        for i, (var, stats) in enumerate(sorted_features[:3]):
            print(f"  {i+1}. {var}: {stats['mean_importance']:.6f}")
        
        return feature_importance, importance_df
    
    def create_global_plots(self):
        """Create global visualization plots"""
        print("[PLOTS] Creating global plots...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Basin statistics distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Global IG Statistics Distribution', fontsize=16, fontweight='bold')
        
        # Basin mean distribution
        axes[0, 0].hist(self.basin_df['overall_mean'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Distribution of Basin Mean IG Values', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Mean IG Value', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Basins', fontsize=12, fontweight='bold')
        axes[0, 0].tick_params(axis='both', which='major', labelsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Basin std distribution
        axes[0, 1].hist(self.basin_df['overall_std'], bins=30, alpha=0.7, edgecolor='black', color='orange')
        axes[0, 1].set_title('Distribution of Basin IG Standard Deviation', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('IG Standard Deviation', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Number of Basins', fontsize=12, fontweight='bold')
        axes[0, 1].tick_params(axis='both', which='major', labelsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Basin min distribution
        axes[1, 0].hist(self.basin_df['overall_min'], bins=30, alpha=0.7, edgecolor='black', color='green')
        axes[1, 0].set_title('Distribution of Basin Minimum IG Values', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Minimum IG Value', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Number of Basins', fontsize=12, fontweight='bold')
        axes[1, 0].tick_params(axis='both', which='major', labelsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Basin max distribution
        axes[1, 1].hist(self.basin_df['overall_max'], bins=30, alpha=0.7, edgecolor='black', color='red')
        axes[1, 1].set_title('Distribution of Basin Maximum IG Values', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Maximum IG Value', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Number of Basins', fontsize=12, fontweight='bold')
        axes[1, 1].tick_params(axis='both', which='major', labelsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "global_statistics_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature importance bar plot
        if hasattr(self, 'feature_importance_df'):
            plt.figure(figsize=(12, 8))
            bars = plt.bar(range(len(self.feature_importance_df)), 
                          self.feature_importance_df['mean_importance'],
                          yerr=self.feature_importance_df['std_importance'],
                          capsize=5, alpha=0.7, edgecolor='black')
            
            plt.title('Global Feature Importance Ranking', fontsize=16, fontweight='bold')
            plt.xlabel('Variables', fontsize=14, fontweight='bold')
            if self.sigma_qobs is not None:
                plt.ylabel('Mean Total |IG| per Prediction (m$^3$/s)', fontsize=14, fontweight='bold')
            else:
                plt.ylabel('Mean Total |IG| per Prediction (normalized)', fontsize=14, fontweight='bold')
            plt.xticks(range(len(self.feature_importance_df)), 
                       self.feature_importance_df['variable'], rotation=45, ha='right', fontsize=12, fontweight='bold')
            plt.yticks(fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "global_feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Aggregated feature importance bar plot (new plot, not overwriting original)
            self._create_aggregated_plot()
            
            # 4. Signed feature importance plots (both original and aggregated)
            self._create_signed_plots()
        
        print(f"[SAVE] Global plots saved to: {self.output_dir}")
    
    def _create_aggregated_plot(self):
        """Create aggregated feature importance plot (grouped variables)"""
        if not hasattr(self, 'feature_importance_df'):
            return
        
        # Aggregate variables by group
        aggregated_importance = {}
        aggregated_std = {}
        
        for var_name, group_name in self.variable_groups.items():
            if var_name not in self.feature_importance_df['variable'].values:
                continue
            var_row = self.feature_importance_df[self.feature_importance_df['variable'] == var_name].iloc[0]
            
            if group_name not in aggregated_importance:
                aggregated_importance[group_name] = []
                aggregated_std[group_name] = []
            
            aggregated_importance[group_name].append(var_row['mean_importance'])
            aggregated_std[group_name].append(var_row['std_importance'])
        
        # Sum within each group
        aggregated_data = []
        for group_name in self.aggregated_names:
            if group_name in aggregated_importance:
                mean_sum = sum(aggregated_importance[group_name])
                # For std, use sqrt of sum of squares (assuming independence)
                std_sum = np.sqrt(sum(s**2 for s in aggregated_std[group_name]))
                aggregated_data.append({
                    'group': group_name,
                    'mean_importance': mean_sum,
                    'std_importance': std_sum,
                    'formal_name': self.aggregated_formal_names[group_name]
                })
        
        # Sort by importance
        aggregated_data.sort(key=lambda x: x['mean_importance'], reverse=True)
        agg_df = pd.DataFrame(aggregated_data)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        colors = [self.aggregated_colors[group] for group in agg_df['group']]
        bars = plt.bar(range(len(agg_df)), 
                      agg_df['mean_importance'],
                      yerr=agg_df['std_importance'],
                      capsize=5, alpha=0.7, edgecolor='black',
                      color=colors)
        
        plt.title('Global Feature Importance Ranking (Aggregated)', fontsize=16, fontweight='bold')
        plt.xlabel('Variable Groups', fontsize=14, fontweight='bold')
        if self.sigma_qobs is not None:
            plt.ylabel('Mean Total |IG| per Prediction (m$^3$/s)', fontsize=14, fontweight='bold')
        else:
            plt.ylabel('Mean Total |IG| per Prediction (normalized)', fontsize=14, fontweight='bold')
        plt.xticks(range(len(agg_df)), 
                   agg_df['formal_name'], rotation=45, ha='right', fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars (2 decimal places)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "global_feature_importance_aggregated.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save aggregated CSV
        agg_df.to_csv(self.output_dir / "global_feature_importance_aggregated.csv", index=False)
        print(f"[SAVE] Aggregated feature importance plot saved")
    
    def _create_signed_plots(self):
        """Create signed (with positive/negative) feature importance plots"""
        if not hasattr(self, 'feature_importance_signed'):
            return
        
        # 1. Signed original plot (17 variables)
        sorted_features_signed = sorted(self.feature_importance_signed.items(), 
                                       key=lambda x: abs(x[1]['mean_importance']), 
                                       reverse=True)
        
        importance_df_signed = pd.DataFrame([
            {
                'variable': var,
                'mean_importance': stats['mean_importance'],
                'std_importance': stats['std_importance'],
                'rank': i+1
            }
            for i, (var, stats) in enumerate(sorted_features_signed)
        ])
        
        plt.figure(figsize=(12, 8))
        # Color bars based on sign: positive = blue, negative = red
        colors = ['#2e7d32' if val >= 0 else '#c62828' for val in importance_df_signed['mean_importance']]
        bars = plt.bar(range(len(importance_df_signed)), 
                      importance_df_signed['mean_importance'],
                      yerr=importance_df_signed['std_importance'],
                      capsize=5, alpha=0.7, edgecolor='black',
                      color=colors)
        
        plt.title('Global Feature Importance Ranking (Signed IG)', fontsize=16, fontweight='bold')
        plt.xlabel('Variables', fontsize=14, fontweight='bold')
        if self.sigma_qobs is not None:
            plt.ylabel('Mean Total IG per Prediction (m$^3$/s)', fontsize=14, fontweight='bold')
        else:
            plt.ylabel('Mean Total IG per Prediction (normalized)', fontsize=14, fontweight='bold')
        plt.xticks(range(len(importance_df_signed)), 
                   importance_df_signed['variable'], rotation=45, ha='right', fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add value labels on bars (2 decimal places)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            y_pos = height + (height*0.02 if height >= 0 else height*0.02 - abs(height)*0.05)
            plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{height:.2f}', ha='center', 
                    va='bottom' if height >= 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "global_feature_importance_signed.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save signed CSV
        importance_df_signed.to_csv(self.output_dir / "global_feature_importance_signed.csv", index=False)
        
        # 2. Signed aggregated plot (7 groups)
        aggregated_importance_signed = {}
        aggregated_std_signed = {}
        
        for var_name, group_name in self.variable_groups.items():
            if var_name not in self.feature_importance_signed:
                continue
            var_stats = self.feature_importance_signed[var_name]
            
            if group_name not in aggregated_importance_signed:
                aggregated_importance_signed[group_name] = []
                aggregated_std_signed[group_name] = []
            
            aggregated_importance_signed[group_name].append(var_stats['mean_importance'])
            aggregated_std_signed[group_name].append(var_stats['std_importance'])
        
        # Sum within each group (signed sum)
        aggregated_data_signed = []
        for group_name in self.aggregated_names:
            if group_name in aggregated_importance_signed:
                mean_sum = sum(aggregated_importance_signed[group_name])
                # For std, use sqrt of sum of squares (assuming independence)
                std_sum = np.sqrt(sum(s**2 for s in aggregated_std_signed[group_name]))
                aggregated_data_signed.append({
                    'group': group_name,
                    'mean_importance': mean_sum,
                    'std_importance': std_sum,
                    'formal_name': self.aggregated_formal_names[group_name]
                })
        
        # Sort by absolute importance
        aggregated_data_signed.sort(key=lambda x: abs(x['mean_importance']), reverse=True)
        agg_df_signed = pd.DataFrame(aggregated_data_signed)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        # Color bars based on sign: use group color if positive, lighter/reddish if negative
        colors_signed = []
        for group in agg_df_signed['group']:
            val = agg_df_signed[agg_df_signed['group'] == group]['mean_importance'].iloc[0]
            if val >= 0:
                colors_signed.append(self.aggregated_colors[group])
            else:
                # Light red for negative values
                colors_signed.append('#ff6b6b')
        
        bars = plt.bar(range(len(agg_df_signed)), 
                      agg_df_signed['mean_importance'],
                      yerr=agg_df_signed['std_importance'],
                      capsize=5, alpha=0.7, edgecolor='black',
                      color=colors_signed)
        
        plt.title('Global Feature Importance Ranking (Signed IG, Aggregated)', fontsize=16, fontweight='bold')
        plt.xlabel('Variable Groups', fontsize=14, fontweight='bold')
        if self.sigma_qobs is not None:
            plt.ylabel('Mean Total IG per Prediction (m$^3$/s)', fontsize=14, fontweight='bold')
        else:
            plt.ylabel('Mean Total IG per Prediction (normalized)', fontsize=14, fontweight='bold')
        plt.xticks(range(len(agg_df_signed)), 
                   agg_df_signed['formal_name'], rotation=45, ha='right', fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add value labels on bars (2 decimal places)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            y_pos = height + (height*0.02 if height >= 0 else height*0.02 - abs(height)*0.05)
            plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{height:.2f}', ha='center', 
                    va='bottom' if height >= 0 else 'top', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "global_feature_importance_signed_aggregated.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save signed aggregated CSV
        agg_df_signed.to_csv(self.output_dir / "global_feature_importance_signed_aggregated.csv", index=False)
        print(f"[SAVE] Signed feature importance plots saved")
    
    def run_global_analysis(self):
        """Run complete global analysis"""
        print("[START] Starting global analysis...")
        
        # Load data
        self.load_global_data()
        
        # Calculate statistics
        stats = self.calculate_global_statistics()
        
        # Analyze feature importance
        feature_importance, importance_df = self.analyze_feature_importance()
        self.feature_importance_df = importance_df
        
        # Create plots
        self.create_global_plots()
        
        print("[SUCCESS] Global analysis completed!")
        print(f"[SAVE] Results saved to: {self.output_dir}")
        
        return {
            'statistics': stats,
            'feature_importance': feature_importance,
            'importance_dataframe': importance_df
        }

def main():
    """Main function"""
    analyzer = GlobalAnalyzer()
    results = analyzer.run_global_analysis()
    
    print("\n[SUMMARY] Global Analysis Results:")
    print(f"- Total basins: {results['statistics']['total_basins']}")
    print(f"- Total samples: {results['statistics']['total_samples']:,}")
    print(f"- Global mean IG: {results['statistics']['global_mean']:.6f}")
    print(f"- Top feature: {results['importance_dataframe'].iloc[0]['variable']}")

if __name__ == "__main__":
    main()
