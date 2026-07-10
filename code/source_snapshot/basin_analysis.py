#!/usr/bin/env python3
"""
Basin-level IG Analysis Script
=============================

This script performs basin-level analysis on the preprocessed IG data,
including individual basin statistics, clustering, and comparative analysis.

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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings
warnings.filterwarnings('ignore')

class BasinAnalyzer:
    """Basin-level analysis of IG data"""
    
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
                self.output_dir = (Path("..") / "analysis" / "basin_analysis").resolve()
            else:
                # Running from test/, output to current dir
                self.output_dir = (Path("analysis") / "basin_analysis").resolve()
        else:
            self.output_dir = Path(output_dir) / "basin_analysis"
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
        
        # Variable grouping for aggregation (swe merged into snow_physics, same as dominance map & global)
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
        
        # Aggregated variable names and colors (8 groups; swe in snow_physics)
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
        
        print("[BASIN] Basin-level IG Analysis")
        print(f"[DIR] Data directory: {self.data_dir}")
        print(f"[OUT] Output directory: {self.output_dir}")
        print("=" * 50)
    
    def load_basin_data(self, basin_id):
        """Load data for a specific basin"""
        basin_data_path = self.data_dir / f"basin_{basin_id}" / "ig_data_combined.npy"
        basin_stats_path = self.data_dir / f"basin_{basin_id}" / "summary_stats.json"
        
        if not basin_data_path.exists():
            raise FileNotFoundError(f"Basin data not found: {basin_data_path}")
        
        # Load IG data
        ig_data = np.load(basin_data_path)
        
        # Load statistics
        with open(basin_stats_path, 'r') as f:
            stats = json.load(f)
        
        return ig_data, stats
    
    def analyze_individual_basin(self, basin_id):
        """Analyze a single basin in detail"""
        print(f"[BASIN] Analyzing basin {basin_id}...")
        
        try:
            ig_data, stats = self.load_basin_data(basin_id)
        except FileNotFoundError:
            print(f"[WARNING] Basin {basin_id} data not found, skipping...")
            return None
        
        # Calculate detailed statistics
        analysis = {
            'basin_id': basin_id,
            'data_shape': ig_data.shape,
            'aggregation_method': 'sum_over_time_then_mean_over_samples',  # Corrected method
            'sigma_QObs': float(self.sigma_qobs) if self.sigma_qobs is not None else None,
            'units': {
                'normalized': 'standardized units (IG raw values)',
                'physical': 'm³/s' if self.sigma_qobs is not None else None
            },
            'overall_stats': {
                'mean': float(np.mean(ig_data)),
                'std': float(np.std(ig_data)),
                'min': float(np.min(ig_data)),
                'max': float(np.max(ig_data)),
                'median': float(np.median(ig_data))
            },
            'variable_stats': {}
        }
        
        # Calculate statistics for each variable
        # CORRECTED: Sum over time dimension (axis=1) first, then mean over samples (axis=0)
        # This matches Case A and Case D/E/F methodology
        for i, var_name in enumerate(self.variable_names):
            var_data = ig_data[:, :, i]  # All samples, all time steps, variable i
            # Sum over time (365 days), then mean over samples
            total_abs_per_sample = np.sum(np.abs(var_data), axis=1)  # Shape: (n_samples,)
            total_signed_per_sample = np.sum(var_data, axis=1)  # Shape: (n_samples,)
            
            mean_total_abs = float(np.mean(total_abs_per_sample))
            std_total_abs = float(np.std(total_abs_per_sample))
            mean_total_signed = float(np.mean(total_signed_per_sample))
            std_total_signed = float(np.std(total_signed_per_sample))
            
            # Apply unit conversion if available
            if self.sigma_qobs is not None:
                mean_total_abs_physical = mean_total_abs * self.sigma_qobs
                std_total_abs_physical = std_total_abs * self.sigma_qobs
                mean_total_signed_physical = mean_total_signed * self.sigma_qobs
                std_total_signed_physical = std_total_signed * self.sigma_qobs
            else:
                mean_total_abs_physical = None
                std_total_abs_physical = None
                mean_total_signed_physical = None
                std_total_signed_physical = None
            
            analysis['variable_stats'][var_name] = {
                'mean': float(np.mean(var_data)),
                'std': float(np.std(var_data)),
                'min': float(np.min(var_data)),
                'max': float(np.max(var_data)),
                # Corrected aggregation: sum over time, then mean over samples
                'mean_total_abs': mean_total_abs,  # Normalized units
                'std_total_abs': std_total_abs,
                'mean_total_signed': mean_total_signed,
                'std_total_signed': std_total_signed,
                # Physical units (m³/s) if sigma_QObs available
                'mean_total_abs_physical': mean_total_abs_physical,
                'std_total_abs_physical': std_total_abs_physical,
                'mean_total_signed_physical': mean_total_signed_physical,
                'std_total_signed_physical': std_total_signed_physical,
                # Legacy fields (kept for backward compatibility, but deprecated)
                'mean_abs': float(np.mean(np.abs(var_data))),  # DEPRECATED: This averages over time
                'max_abs': float(np.max(np.abs(var_data)))
            }
        
        # Find dominant variable (highest mean total absolute IG)
        # Use corrected aggregation method
        dominant_var = max(analysis['variable_stats'].items(), 
                          key=lambda x: x[1]['mean_total_abs'])
        analysis['dominant_variable'] = dominant_var[0]
        analysis['dominant_importance'] = dominant_var[1]['mean_total_abs']
        if self.sigma_qobs is not None:
            analysis['dominant_importance_physical'] = dominant_var[1]['mean_total_abs_physical']
        else:
            analysis['dominant_importance_physical'] = None
        
        # Temporal analysis (across time steps)
        temporal_stats = []
        for t in range(ig_data.shape[1]):  # For each time step
            time_data = ig_data[:, t, :]  # All samples, time t, all variables
            temporal_stats.append({
                'timestep': t,
                'mean': float(np.mean(time_data)),
                'std': float(np.std(time_data)),
                'max_abs': float(np.max(np.abs(time_data)))
            })
        
        analysis['temporal_stats'] = temporal_stats
        
        # Save individual basin analysis
        basin_output_dir = self.output_dir / f"basin_{basin_id}"
        basin_output_dir.mkdir(exist_ok=True)
        
        with open(basin_output_dir / "detailed_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"[SAVE] Basin {basin_id} analysis saved to: {basin_output_dir}")
        return analysis
    
    def _try_load_sigma_qobs(self):
        """Best-effort loader for sigma_QObs from train_data_scaler.yml (run directory)."""
        # Script is typically located at: <run_dir>/ig_outputs/test/analysis_scripts/basin_analysis.py
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
    
    def cluster_basins(self, n_clusters=5):
        """Cluster basins based on their IG characteristics"""
        print(f"[CLUSTER] Clustering basins into {n_clusters} groups...")
        
        # Load basin summaries
        basin_summaries_path = self.data_dir / "basin_summaries.csv"
        basin_df = pd.read_csv(basin_summaries_path)
        
        # Prepare features for clustering (numerical only)
        features = ['overall_mean', 'overall_std', 'overall_min', 'overall_max']
        feature_matrix = basin_df[features].values
        
        # Standardize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
        
        # Add cluster labels to dataframe
        basin_df['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = []
        for cluster_id in range(n_clusters):
            cluster_basins = basin_df[basin_df['cluster'] == cluster_id]
            cluster_stats.append({
                'cluster_id': cluster_id,
                'num_basins': len(cluster_basins),
                'mean_mean': float(cluster_basins['overall_mean'].mean()),
                'mean_std': float(cluster_basins['overall_std'].mean()),
                'basin_ids': cluster_basins['basin_id'].tolist()
            })
        
        # Save clustering results
        clustering_results = {
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'cluster_stats': cluster_stats,
            'basin_clusters': basin_df[['basin_id', 'cluster']].to_dict('records')
        }
        
        with open(self.output_dir / "basin_clustering.json", 'w') as f:
            json.dump(clustering_results, f, indent=2)
        
        # Save cluster dataframe
        basin_df.to_csv(self.output_dir / "basin_clusters.csv", index=False)
        
        print(f"[SAVE] Clustering results saved to: {self.output_dir}")
        return clustering_results, basin_df
    
    def create_basin_plots(self, basin_id):
        """Create plots for a specific basin"""
        print(f"[PLOTS] Creating plots for basin {basin_id}...")
        
        try:
            ig_data, stats = self.load_basin_data(basin_id)
        except FileNotFoundError:
            print(f"[WARNING] Basin {basin_id} data not found, skipping plots...")
            return
        
        # Create output directory
        basin_output_dir = self.output_dir / f"basin_{basin_id}"
        basin_output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        
        # 1. Variable importance bar plot (original variables, absolute values, SORTED)
        plt.figure(figsize=(12, 8))
        var_data_list = []
        for i, var_name in enumerate(self.variable_names):
            var_data = ig_data[:, :, i]
            # CORRECTED: Sum over time (365 days), then mean over samples
            total_abs_per_sample = np.sum(np.abs(var_data), axis=1)
            importance = np.mean(total_abs_per_sample)
            std_importance = np.std(total_abs_per_sample)
            # Apply unit conversion if available
            if self.sigma_qobs is not None:
                importance *= self.sigma_qobs
                std_importance *= self.sigma_qobs
            var_data_list.append({
                'name': var_name,
                'importance': importance,
                'std': std_importance
            })
        
        # Sort by importance (high to low)
        var_data_list.sort(key=lambda x: x['importance'], reverse=True)
        sorted_names = [d['name'] for d in var_data_list]
        sorted_importance = [d['importance'] for d in var_data_list]
        sorted_std = [d['std'] for d in var_data_list]
        
        bars = plt.bar(range(len(sorted_names)), sorted_importance, 
                       alpha=0.7, edgecolor='black')
        plt.title(f'Variable Importance - Basin {basin_id}', fontsize=16, fontweight='bold')
        plt.xlabel('Variables', fontsize=14, fontweight='bold')
        ylabel = 'Mean Total |IG| per Prediction'
        if self.sigma_qobs is not None:
            ylabel += ' (m$^3$/s)'
        plt.ylabel(ylabel, fontsize=14, fontweight='bold')
        plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right', fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Adjust y-axis limits to leave space for value labels
        max_value = max(sorted_importance) if sorted_importance else 1.0
        y_max = max_value * 1.15  # Add 15% space at top for labels
        plt.ylim(0, y_max)
        
        # Add value labels (2 decimal places) with better spacing
        label_offset = y_max * 0.03  # 3% of y-axis range
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + label_offset,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout(rect=[0, 0, 1, 0.88])  # Adjust top margin to prevent title and y-label cutoff
        plt.subplots_adjust(top=0.90)  # Additional top margin adjustment
        plt.savefig(basin_output_dir / f"variable_importance_basin_{basin_id}.png", 
                   dpi=300, bbox_inches='tight', pad_inches=0.4)
        plt.close()
        
        # 1a. Variable importance bar plot (original variables, SIGNED values, SORTED)
        plt.figure(figsize=(12, 8))
        var_data_list_signed = []
        for i, var_name in enumerate(self.variable_names):
            var_data = ig_data[:, :, i]
            # CORRECTED: Sum over time (365 days), then mean over samples (with sign)
            total_signed_per_sample = np.sum(var_data, axis=1)
            importance = np.mean(total_signed_per_sample)
            std_importance = np.std(total_signed_per_sample)
            # Apply unit conversion if available
            if self.sigma_qobs is not None:
                importance *= self.sigma_qobs
                std_importance *= self.sigma_qobs
            var_data_list_signed.append({
                'name': var_name,
                'importance': importance,
                'std': std_importance
            })
        
        # Sort by absolute importance (high to low)
        var_data_list_signed.sort(key=lambda x: abs(x['importance']), reverse=True)
        sorted_names_signed = [d['name'] for d in var_data_list_signed]
        sorted_importance_signed = [d['importance'] for d in var_data_list_signed]
        sorted_std_signed = [d['std'] for d in var_data_list_signed]
        
        # Use different colors for positive and negative values
        colors_signed = ['#1e40af' if val >= 0 else '#dc2626' for val in sorted_importance_signed]
        
        bars = plt.bar(range(len(sorted_names_signed)), sorted_importance_signed, 
                       color=colors_signed, alpha=0.7, edgecolor='black')
        plt.title(f'Variable Importance (Signed) - Basin {basin_id}', fontsize=16, fontweight='bold')
        plt.xlabel('Variables', fontsize=14, fontweight='bold')
        ylabel = 'Mean Total IG per Prediction'
        if self.sigma_qobs is not None:
            ylabel += ' (m$^3$/s)'
        plt.ylabel(ylabel, fontsize=14, fontweight='bold')
        plt.xticks(range(len(sorted_names_signed)), sorted_names_signed, rotation=45, ha='right', fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Adjust y-axis limits to leave space for value labels
        max_val = max(sorted_importance_signed) if sorted_importance_signed else 1.0
        min_val = min(sorted_importance_signed) if sorted_importance_signed else -1.0
        y_range = max_val - min_val
        y_max = max_val + y_range * 0.15 if max_val > 0 else max_val * 1.15
        y_min = min_val - y_range * 0.15 if min_val < 0 else min_val * 1.15
        plt.ylim(y_min, y_max)
        
        # Add value labels (2 decimal places) with better spacing
        label_offset = y_range * 0.03 if y_range > 0 else abs(max_val) * 0.03
        for i, bar in enumerate(bars):
            height = bar.get_height()
            y_pos = height + label_offset if height >= 0 else height - label_offset
            plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        plt.tight_layout(rect=[0, 0, 1, 0.88])  # Adjust top margin to prevent title and y-label cutoff
        plt.subplots_adjust(top=0.90)  # Additional top margin adjustment
        plt.savefig(basin_output_dir / f"variable_importance_signed_basin_{basin_id}.png", 
                   dpi=300, bbox_inches='tight', pad_inches=0.4)
        plt.close()
        
        # 1b. Aggregated variable importance bar plot (grouped variables)
        plt.figure(figsize=(10, 6))
        aggregated_importance = {}
        aggregated_std = {}
        
        for i, var_name in enumerate(self.variable_names):
            var_data = ig_data[:, :, i]
            # CORRECTED: Sum over time, then mean over samples
            total_abs_per_sample = np.sum(np.abs(var_data), axis=1)
            importance = np.mean(total_abs_per_sample)
            std_importance = np.std(total_abs_per_sample)
            # Apply unit conversion if available
            if self.sigma_qobs is not None:
                importance *= self.sigma_qobs
                std_importance *= self.sigma_qobs
            
            group_name = self.variable_groups[var_name]
            if group_name not in aggregated_importance:
                aggregated_importance[group_name] = []
                aggregated_std[group_name] = []
            aggregated_importance[group_name].append(importance)
            aggregated_std[group_name].append(std_importance)
        
        # Sum within each group
        agg_data = []
        for group_name in self.aggregated_names:
            if group_name in aggregated_importance:
                mean_sum = sum(aggregated_importance[group_name])
                # For std, use sqrt of sum of squares (assuming independence)
                std_sum = np.sqrt(sum(s**2 for s in aggregated_std[group_name]))
                agg_data.append({
                    'group': group_name,
                    'importance': mean_sum,
                    'std': std_sum,
                    'formal_name': self.aggregated_formal_names[group_name]
                })
        
        # Sort by importance
        agg_data.sort(key=lambda x: x['importance'], reverse=True)
        agg_groups = [d['group'] for d in agg_data]
        agg_importance = [d['importance'] for d in agg_data]
        agg_std = [d['std'] for d in agg_data]
        agg_formal_names = [d['formal_name'] for d in agg_data]
        colors = [self.aggregated_colors[g] for g in agg_groups]
        
        bars = plt.bar(range(len(agg_data)), agg_importance,
                      color=colors, alpha=0.7, edgecolor='black')
        plt.title(f'Aggregated Variable Importance - Basin {basin_id}', fontsize=16, fontweight='bold')
        plt.xlabel('Variable Groups', fontsize=14, fontweight='bold')
        ylabel = 'Mean Total |IG| per Prediction'
        if self.sigma_qobs is not None:
            ylabel += ' (m$^3$/s)'
        plt.ylabel(ylabel, fontsize=14, fontweight='bold')
        plt.xticks(range(len(agg_data)), agg_formal_names, rotation=45, ha='right', fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Adjust y-axis limits to leave space for value labels
        max_value = max(agg_importance) if agg_importance else 1.0
        y_max = max_value * 1.15  # Add 15% space at top for labels
        plt.ylim(0, y_max)
        
        # Add value labels (2 decimal places) with better spacing
        for i, bar in enumerate(bars):
            height = bar.get_height()
            # Use fixed offset based on y-axis range to avoid overlap with grid lines
            label_offset = y_max * 0.03  # 3% of y-axis range
            plt.text(bar.get_x() + bar.get_width()/2., height + label_offset,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.88])  # Adjust top margin to prevent title and y-label cutoff
        plt.subplots_adjust(top=0.90)  # Additional top margin adjustment
        plt.savefig(basin_output_dir / f"variable_importance_aggregated_basin_{basin_id}.png", 
                   dpi=300, bbox_inches='tight', pad_inches=0.4)
        plt.close()
        
        # 1c. Aggregated variable importance bar plot (grouped variables, SIGNED values)
        plt.figure(figsize=(10, 6))
        aggregated_importance_signed = {}
        aggregated_std_signed = {}
        
        for i, var_name in enumerate(self.variable_names):
            var_data = ig_data[:, :, i]
            # CORRECTED: Sum over time, then mean over samples (with sign)
            total_signed_per_sample = np.sum(var_data, axis=1)
            importance = np.mean(total_signed_per_sample)
            std_importance = np.std(total_signed_per_sample)
            # Apply unit conversion if available
            if self.sigma_qobs is not None:
                importance *= self.sigma_qobs
                std_importance *= self.sigma_qobs
            
            group_name = self.variable_groups[var_name]
            if group_name not in aggregated_importance_signed:
                aggregated_importance_signed[group_name] = []
                aggregated_std_signed[group_name] = []
            aggregated_importance_signed[group_name].append(importance)
            aggregated_std_signed[group_name].append(std_importance)
        
        # Sum within each group
        agg_data_signed = []
        for group_name in self.aggregated_names:
            if group_name in aggregated_importance_signed:
                mean_sum = sum(aggregated_importance_signed[group_name])
                # For std, use sqrt of sum of squares (assuming independence)
                std_sum = np.sqrt(sum(s**2 for s in aggregated_std_signed[group_name]))
                agg_data_signed.append({
                    'group': group_name,
                    'importance': mean_sum,
                    'std': std_sum,
                    'formal_name': self.aggregated_formal_names[group_name]
                })
        
        # Sort by absolute importance
        agg_data_signed.sort(key=lambda x: abs(x['importance']), reverse=True)
        agg_groups_signed = [d['group'] for d in agg_data_signed]
        agg_importance_signed = [d['importance'] for d in agg_data_signed]
        agg_std_signed = [d['std'] for d in agg_data_signed]
        agg_formal_names_signed = [d['formal_name'] for d in agg_data_signed]
        # Use different colors for positive and negative values
        colors_signed_agg = ['#1e40af' if val >= 0 else '#dc2626' for val in agg_importance_signed]
        
        bars = plt.bar(range(len(agg_data_signed)), agg_importance_signed,
                      color=colors_signed_agg, alpha=0.7, edgecolor='black')
        plt.title(f'Aggregated Variable Importance (Signed) - Basin {basin_id}', fontsize=16, fontweight='bold')
        plt.xlabel('Variable Groups', fontsize=14, fontweight='bold')
        ylabel = 'Mean Total IG per Prediction'
        if self.sigma_qobs is not None:
            ylabel += ' (m$^3$/s)'
        plt.ylabel(ylabel, fontsize=14, fontweight='bold')
        plt.xticks(range(len(agg_data_signed)), agg_formal_names_signed, rotation=45, ha='right', fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Adjust y-axis limits to leave space for value labels
        max_val = max(agg_importance_signed) if agg_importance_signed else 1.0
        min_val = min(agg_importance_signed) if agg_importance_signed else -1.0
        y_range = max_val - min_val
        y_max = max_val + y_range * 0.15 if max_val > 0 else max_val * 1.15
        y_min = min_val - y_range * 0.15 if min_val < 0 else min_val * 1.15
        plt.ylim(y_min, y_max)
        
        # Add value labels (2 decimal places) with better spacing
        label_offset = y_range * 0.03 if y_range > 0 else abs(max_val) * 0.03
        for i, bar in enumerate(bars):
            height = bar.get_height()
            y_pos = height + label_offset if height >= 0 else height - label_offset
            plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.88])  # Adjust top margin to prevent title and y-label cutoff
        plt.subplots_adjust(top=0.90)  # Additional top margin adjustment
        plt.savefig(basin_output_dir / f"variable_importance_aggregated_signed_basin_{basin_id}.png", 
                   dpi=300, bbox_inches='tight', pad_inches=0.4)
        plt.close()
        
        # 2. Temporal pattern heatmap (sample of data)
        plt.figure(figsize=(15, 8))
        
        # Use full data for visualization (every 10th sample, all timesteps)
        sample_data = ig_data[::10, :, :]  # Shape: (samples, timesteps, variables)
        
        # Calculate mean absolute IG across samples for each time-variable combination
        heatmap_data = np.mean(np.abs(sample_data), axis=0)  # Shape: (timesteps, variables)
        
        # Reverse the data so that left side shows recent data (1 day ago) and right side shows old data (365 days ago)
        heatmap_data_reversed = np.flip(heatmap_data, axis=0)  # Reverse along time axis
        
        # Let matplotlib automatically handle x-axis labels
        sns.heatmap(heatmap_data_reversed.T, 
                   yticklabels=self.variable_names,
                   cmap='viridis', cbar_kws={'label': 'Mean Absolute IG'})
        
        # Set x-axis labels manually after creating the heatmap
        ax = plt.gca()
        n_timesteps = sample_data.shape[1]
        # Set x-axis ticks and labels
        ax.set_xticks(range(0, n_timesteps, n_timesteps//10))  # 10 evenly spaced ticks
        ax.set_xticklabels(range(1, 366, 36))  # Labels from 1 to 361, step 36
        
        plt.title(f'Temporal IG Pattern - Basin {basin_id}', fontsize=16, fontweight='bold')
        plt.xlabel('Days Before Prediction (1 to 365)', fontsize=14, fontweight='bold')
        plt.ylabel('Variables', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(basin_output_dir / f"temporal_pattern_basin_{basin_id}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[SAVE] Basin {basin_id} plots saved to: {basin_output_dir}")
    
    def create_clustering_plots(self, basin_df):
        """Create plots for basin clustering results"""
        print("[PLOTS] Creating clustering plots...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Cluster distribution
        plt.figure(figsize=(10, 6))
        cluster_counts = basin_df['cluster'].value_counts().sort_index()
        bars = plt.bar(cluster_counts.index, cluster_counts.values, 
                      alpha=0.7, edgecolor='black')
        plt.title('Basin Cluster Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Cluster ID', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Basins', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "basin_cluster_distribution.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Cluster characteristics scatter plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(basin_df['overall_mean'], basin_df['overall_std'], 
                            c=basin_df['cluster'], cmap='tab10', alpha=0.7, s=50)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title('Basin Clusters: Mean vs Standard Deviation', fontsize=16, fontweight='bold')
        plt.xlabel('Overall Mean IG Value', fontsize=14, fontweight='bold')
        plt.ylabel('Overall Standard Deviation', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "basin_cluster_scatter.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[SAVE] Clustering plots saved to: {self.output_dir}")
    
    def run_basin_analysis(self, sample_basins=None, n_clusters=5):
        """Run complete basin analysis"""
        print("[START] Starting basin analysis...")
        
        # If no sample basins specified, analyze first 10 basins
        if sample_basins is None:
            sample_basins = [f"{i:03d}" for i in range(1, 11)]
        
        # Analyze individual basins
        basin_analyses = []
        for basin_id in sample_basins:
            analysis = self.analyze_individual_basin(basin_id)
            if analysis:
                basin_analyses.append(analysis)
                # Create plots for this basin
                self.create_basin_plots(basin_id)
        
        # Cluster all basins
        clustering_results, basin_df = self.cluster_basins(n_clusters)
        
        # Create clustering plots
        self.create_clustering_plots(basin_df)
        
        print("[SUCCESS] Basin analysis completed!")
        print(f"[SAVE] Results saved to: {self.output_dir}")
        
        return {
            'basin_analyses': basin_analyses,
            'clustering_results': clustering_results,
            'basin_dataframe': basin_df
        }

def main():
    """Main function"""
    analyzer = BasinAnalyzer()
    
    # Analyze all 135 basins
    sample_basins = [f"{i:03d}" for i in range(1, 136)]
    results = analyzer.run_basin_analysis(sample_basins=sample_basins, n_clusters=5)
    
    print("\n[SUMMARY] Basin Analysis Results:")
    print(f"- Analyzed {len(results['basin_analyses'])} individual basins")
    print(f"- Created {results['clustering_results']['n_clusters']} clusters")
    print(f"- Total basins clustered: {len(results['basin_dataframe'])}")

if __name__ == "__main__":
    main()
