#!/usr/bin/env python3
"""
Temporal IG Analysis Script
=========================

This script performs temporal analysis on the preprocessed IG data,
including seasonal patterns, time series analysis, and temporal clustering.

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
from scipy import signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class TemporalAnalyzer:
    """Temporal analysis of IG data"""
    
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
                self.output_dir = (Path("..") / "analysis" / "temporal_analysis").resolve()
            else:
                # Running from test/, output to current dir
                self.output_dir = (Path("analysis") / "temporal_analysis").resolve()
        else:
            self.output_dir = Path(output_dir) / "temporal_analysis"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Variable names (11 variables: optimized; must match ig_data_combined.npy last dimension)
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
        
        print("[TEMPORAL] Temporal IG Analysis")
        print(f"[DIR] Data directory: {self.data_dir}")
        print(f"[OUT] Output directory: {self.output_dir}")
        print("=" * 50)
    
    def load_temporal_data(self, sample_basins=None):
        """Load data for temporal analysis"""
        if sample_basins is None:
            # Use all 135 basins for comprehensive analysis
            sample_basins = [f"{i:03d}" for i in range(1, 136)]  # All 135 basins
        
        print(f"[LOAD] Loading temporal data from {len(sample_basins)} basins...")
        
        temporal_data = {}
        basin_info = []
        
        for basin_id in sample_basins:
            basin_data_path = self.data_dir / f"basin_{basin_id}" / "ig_data_combined.npy"
            if basin_data_path.exists():
                data = np.load(basin_data_path)  # Shape: (samples, timesteps, variables)
                
                # Store the raw data for seasonal analysis
                # We need to analyze by samples (prediction dates), not timesteps
                temporal_data[basin_id] = data  # Keep raw data: (samples, timesteps, variables)
                basin_info.append({
                    'basin_id': basin_id,
                    'shape': data.shape,
                    'samples': data.shape[0],
                    'timesteps': data.shape[1],
                    'variables': data.shape[2]
                })
        
        print(f"[LOAD] Loaded temporal data from {len(temporal_data)} basins")
        return temporal_data, basin_info
    
    def analyze_seasonal_patterns(self, temporal_data):
        """Analyze seasonal patterns in IG data - Prediction Target Seasonality"""
        print("[SEASONAL] Analyzing seasonal patterns by prediction target season...")
        
        seasonal_stats = {}
        
        for var_idx, var_name in enumerate(self.variable_names):
            print(f"  Analyzing {var_name} seasonal patterns...")
            
            # Collect data for this variable across all basins
            # Data shape: (samples, timesteps, variables)
            # We need to group by prediction target season (samples dimension)
            
            # For now, let's assume we have 731 samples representing 2 years of daily predictions
            # We'll group them by day-of-year to determine season
            all_var_data = []
            
            for basin_id, basin_data in temporal_data.items():
                # basin_data shape: (samples, timesteps, variables)
                var_data = basin_data[:, :, var_idx]  # Shape: (samples, timesteps)
                
                # Calculate mean IG across all timesteps for each sample
                # This gives us the overall importance of this variable for each prediction
                sample_importance = np.mean(np.abs(var_data), axis=1)  # Shape: (samples,)
                all_var_data.append(sample_importance)
            
            # Combine all basin data
            combined_data = np.concatenate(all_var_data)  # Shape: (total_samples,)
            
            # Group by prediction target season
            # We have 731 samples per basin representing daily predictions over 2 years (2000-2001)
            # Each sample represents a prediction date
            # We need to determine which season each prediction belongs to
            
            n_samples = len(combined_data)
            print(f"    Total samples: {n_samples}")
            
            # Define seasons based on day-of-year for 2 years (2000-2001)
            # Year 1 (2000): samples 0-364, Year 2 (2001): samples 365-730
            # Spring: March-May (days 60-151, 425-516)
            # Summer: June-August (days 152-243, 517-608)  
            # Autumn: September-November (days 244-334, 609-699)
            # Winter: December-February (days 335-364+0-59, 700-730+0-59)
            
            spring_samples_base = list(range(60, 152)) + list(range(425, 517))
            summer_samples_base = list(range(152, 244)) + list(range(517, 609))
            autumn_samples_base = list(range(244, 335)) + list(range(609, 700))
            winter_samples_base = list(range(335, 365)) + list(range(0, 60)) + list(range(700, 731))
            
            # Expand indices for all basins
            n_basins = len(temporal_data)
            samples_per_basin = 731
            
            spring_samples = []
            summer_samples = []
            autumn_samples = []
            winter_samples = []
            
            for basin_idx in range(n_basins):
                offset = basin_idx * samples_per_basin
                spring_samples.extend([idx + offset for idx in spring_samples_base])
                summer_samples.extend([idx + offset for idx in summer_samples_base])
                autumn_samples.extend([idx + offset for idx in autumn_samples_base])
                winter_samples.extend([idx + offset for idx in winter_samples_base])
            
            seasons = {
                'Spring': spring_samples,
                'Summer': summer_samples,
                'Autumn': autumn_samples,
                'Winter': winter_samples
            }
            
            seasonal_analysis = {}
            for season_name, sample_indices in seasons.items():
                # Extract data for this season
                season_data = []
                for idx in sample_indices:
                    if idx < len(combined_data):
                        season_data.append(combined_data[idx])
                
                if season_data:
                    season_data = np.array(season_data)
                    
                    seasonal_analysis[season_name] = {
                        'mean': float(np.mean(season_data)),
                        'std': float(np.std(season_data)),
                        'mean_abs': float(np.mean(np.abs(season_data))),
                        'max_abs': float(np.max(np.abs(season_data))),
                        'min': float(np.min(season_data)),
                        'max': float(np.max(season_data))
                    }
                else:
                    seasonal_analysis[season_name] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'mean_abs': 0.0,
                        'max_abs': 0.0,
                        'min': 0.0,
                        'max': 0.0
                    }
            
            seasonal_stats[var_name] = seasonal_analysis
        
        # Save seasonal analysis
        seasonal_path = self.output_dir / "seasonal_patterns.json"
        with open(seasonal_path, 'w') as f:
            json.dump(seasonal_stats, f, indent=2)
        
        print(f"[SAVE] Seasonal patterns saved to: {seasonal_path}")
        return seasonal_stats
    
    def analyze_temporal_correlations(self, temporal_data):
        """Analyze temporal correlations between variables"""
        print("[CORRELATION] Analyzing temporal correlations...")
        
        temporal_correlations = {}
        
        # Calculate correlations between all variable pairs
        for i, var1 in enumerate(self.variable_names):
            temporal_correlations[var1] = {}
            
            for j, var2 in enumerate(self.variable_names):
                if i != j:
                    print(f"  Calculating correlation: {var1} vs {var2}")
                    
                    # Collect data for both variables across all basins
                    var1_data = []
                    var2_data = []
                    
                    for basin_id, basin_data in temporal_data.items():
                        # basin_data shape: (samples, timesteps, variables)
                        var1_samples = basin_data[:, :, i]  # Shape: (samples, timesteps)
                        var2_samples = basin_data[:, :, j]  # Shape: (samples, timesteps)
                        
                        # Calculate mean importance across timesteps for each sample
                        var1_importance = np.mean(np.abs(var1_samples), axis=1)  # Shape: (samples,)
                        var2_importance = np.mean(np.abs(var2_samples), axis=1)  # Shape: (samples,)
                        
                        var1_data.extend(var1_importance)
                        var2_data.extend(var2_importance)
                    
                    # Calculate correlation
                    if len(var1_data) > 1 and len(var2_data) > 1:
                        correlation, p_value = pearsonr(var1_data, var2_data)
                        
                        temporal_correlations[var1][var2] = {
                            'correlation': float(correlation),
                            'p_value': float(p_value),
                            'sample_size': len(var1_data)
                        }
                    else:
                        temporal_correlations[var1][var2] = {
                            'correlation': 0.0,
                            'p_value': 1.0,
                            'sample_size': 0
                        }
                else:
                    # Same variable
                    temporal_correlations[var1][var2] = {
                        'correlation': 1.0,
                        'p_value': 0.0,
                        'sample_size': 0
                    }
        
        # Calculate summary statistics
        all_correlations = []
        for var1 in self.variable_names:
            for var2 in self.variable_names:
                if var1 != var2:
                    all_correlations.append(temporal_correlations[var1][var2]['correlation'])
        
        if all_correlations:
            summary_stats = {
                'mean_correlation': float(np.mean(all_correlations)),
                'std_correlation': float(np.std(all_correlations)),
                'min_correlation': float(np.min(all_correlations)),
                'max_correlation': float(np.max(all_correlations)),
                'total_pairs': len(all_correlations)
            }
        else:
            summary_stats = {
                'mean_correlation': 0.0,
                'std_correlation': 0.0,
                'min_correlation': 0.0,
                'max_correlation': 0.0,
                'total_pairs': 0
            }
        
        # Add summary to results
        temporal_correlations['summary'] = summary_stats
        
        # Save temporal correlations
        corr_path = self.output_dir / "temporal_correlations.json"
        with open(corr_path, 'w') as f:
            json.dump(temporal_correlations, f, indent=2)
        
        print(f"[SAVE] Temporal correlations saved to: {corr_path}")
        print(f"[STATS] Mean correlation: {summary_stats['mean_correlation']:.4f}")
        return temporal_correlations
    
    def analyze_temporal_clustering(self, temporal_data):
        """Cluster time steps based on IG patterns"""
        print("[CLUSTERING] Analyzing temporal clustering...")
        
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Collect temporal patterns for clustering
        # We'll cluster based on the average IG patterns across all basins
        temporal_patterns = []
        
        # For each time step, calculate average IG across all basins and variables
        n_timesteps = 365  # Fixed timesteps per basin
        n_basins = len(temporal_data)
        
        for t in range(n_timesteps):
            timestep_pattern = []
            
            for basin_id, basin_data in temporal_data.items():
                # basin_data shape: (samples, timesteps, variables)
                # Get all samples for this timestep
                timestep_data = basin_data[:, t, :]  # Shape: (samples, variables)
                
                # Calculate mean importance for this timestep across all samples
                timestep_importance = np.mean(np.abs(timestep_data), axis=0)  # Shape: (variables,)
                timestep_pattern.extend(timestep_importance)
            
            temporal_patterns.append(timestep_pattern)
        
        # Convert to numpy array
        temporal_patterns = np.array(temporal_patterns)  # Shape: (timesteps, variables * basins)
        
        # Standardize the patterns
        scaler = StandardScaler()
        temporal_patterns_scaled = scaler.fit_transform(temporal_patterns)
        
        # Perform K-means clustering
        n_clusters = 4  # 4 clusters for different temporal patterns
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(temporal_patterns_scaled)
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_timesteps = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_timesteps) > 0:
                cluster_data = temporal_patterns[cluster_timesteps]
                
                cluster_analysis[f'cluster_{cluster_id}'] = {
                    'timesteps': cluster_timesteps.tolist(),
                    'size': len(cluster_timesteps),
                    'start_day': int(cluster_timesteps.min()),
                    'end_day': int(cluster_timesteps.max()),
                    'characteristics': {}
                }
                
                # Calculate characteristics for each variable
                for i, var_name in enumerate(self.variable_names):
                    var_values = []
                    for timestep_idx in cluster_timesteps:
                        # Extract values for this variable across all basins
                        var_timestep_values = []
                        for basin_id, basin_data in temporal_data.items():
                            timestep_data = basin_data[:, timestep_idx, i]  # Shape: (samples,)
                            var_timestep_values.extend(np.abs(timestep_data))
                        var_values.extend(var_timestep_values)
                    
                    if var_values:
                        cluster_analysis[f'cluster_{cluster_id}']['characteristics'][var_name] = {
                            'mean': float(np.mean(var_values)),
                            'std': float(np.std(var_values)),
                            'max': float(np.max(var_values)),
                            'min': float(np.min(var_values))
                        }
                    else:
                        cluster_analysis[f'cluster_{cluster_id}']['characteristics'][var_name] = {
                            'mean': 0.0,
                            'std': 0.0,
                            'max': 0.0,
                            'min': 0.0
                        }
        
        # Save temporal clustering
        clustering_path = self.output_dir / "temporal_clustering.json"
        with open(clustering_path, 'w') as f:
            json.dump(cluster_analysis, f, indent=2)
        
        print(f"[SAVE] Temporal clustering saved to: {clustering_path}")
        print(f"[STATS] Created {n_clusters} temporal clusters")
        return cluster_analysis, cluster_labels
    
    def create_seasonal_plots(self, seasonal_stats):
        """Create seasonal visualization plots"""
        print("[PLOTS] Creating seasonal plots...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Seasonal patterns heatmap
        plt.figure(figsize=(12, 8))
        
        # Create seasonal data matrix
        seasonal_matrix = []
        season_names = ['Spring', 'Summer', 'Autumn', 'Winter']
        
        for var_name in self.variable_names:
            var_seasonal = []
            for season in season_names:
                mean_abs = seasonal_stats[var_name][season]['mean_abs']
                var_seasonal.append(mean_abs)
            seasonal_matrix.append(var_seasonal)
        
        seasonal_matrix = np.array(seasonal_matrix)
        
        # Create heatmap
        sns.heatmap(seasonal_matrix, 
                   xticklabels=season_names,
                   yticklabels=self.variable_names,
                   annot=True, fmt='.4f', cmap='viridis',
                   cbar_kws={'label': 'Mean Absolute IG Value'})
        
        plt.title('Seasonal Importance by Prediction Target Season', fontsize=16, fontweight='bold')
        plt.xlabel('Prediction Target Seasons')
        plt.ylabel('Variables')
        plt.tight_layout()
        plt.savefig(self.output_dir / "seasonal_patterns_heatmap.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Top variables seasonal comparison
        plt.figure(figsize=(15, 10))
        
        # Get top 6 variables by overall importance
        var_importance = {}
        for var_name in self.variable_names:
            total_importance = 0
            for season in season_names:
                total_importance += seasonal_stats[var_name][season]['mean_abs']
            var_importance[var_name] = total_importance
        
        top_6_vars = sorted(var_importance.items(), key=lambda x: x[1], reverse=True)[:6]
        
        # Create subplots for top 6 variables
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Seasonal Patterns of Top 6 Variables', fontsize=16, fontweight='bold')
        
        for i, (var_name, _) in enumerate(top_6_vars):
            row = i // 3
            col = i % 3
            
            seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
            values = [seasonal_stats[var_name][season]['mean_abs'] for season in seasons]
            
            bars = axes[row, col].bar(seasons, values, color=['#2E8B57', '#FF6347', '#FF8C00', '#4682B4'])
            axes[row, col].set_title(f'{var_name}', fontsize=14, fontweight='bold')
            axes[row, col].set_ylabel('Mean Absolute IG Value', fontsize=12, fontweight='bold')
            axes[row, col].tick_params(axis='both', which='major', labelsize=10)
            axes[row, col].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[row, col].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                   f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "seasonal_patterns_top6.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[SAVE] Seasonal plots saved to: {self.output_dir}")
    
    def create_temporal_plots(self, temporal_data, seasonal_stats, temporal_clusters):
        """Create temporal visualization plots"""
        print("[PLOTS] Creating temporal plots...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Temporal series for top variables
        print("  Creating temporal series plots...")
        
        # Get top 3 variables by overall importance
        var_importance = {}
        season_names = ['Spring', 'Summer', 'Autumn', 'Winter']
        for var_name in self.variable_names:
            total_importance = 0
            for season in season_names:
                total_importance += seasonal_stats[var_name][season]['mean_abs']
            var_importance[var_name] = total_importance
        
        top_3_vars = sorted(var_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Temporal Patterns of Top 3 Variables', fontsize=16, fontweight='bold')
        
        for i, (var_name, _) in enumerate(top_3_vars):
            var_idx = self.variable_names.index(var_name)
            
            # Calculate mean temporal series across all basins and samples
            all_temporal_data = []
            for basin_id, basin_data in temporal_data.items():
                # basin_data shape: (samples, timesteps, variables)
                var_data = basin_data[:, :, var_idx]  # Shape: (samples, timesteps)
                
                # Calculate mean across samples for each timestep
                timestep_means = np.mean(np.abs(var_data), axis=0)  # Shape: (timesteps,)
                all_temporal_data.append(timestep_means)
            
            # Combine across basins
            combined_temporal = np.array(all_temporal_data)  # Shape: (basins, timesteps)
            mean_series = np.mean(combined_temporal, axis=0)  # Shape: (timesteps,)
            std_series = np.std(combined_temporal, axis=0)   # Shape: (timesteps,)
            
            # Plot with confidence interval
            timesteps = range(len(mean_series))
            axes[i].plot(timesteps, mean_series, linewidth=2, label=f'{var_name} Mean')
            axes[i].fill_between(timesteps, 
                               mean_series - std_series, 
                               mean_series + std_series, 
                               alpha=0.3, label='±1 Std')
            
            axes[i].set_title(f'{var_name} - Temporal Pattern', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Time Steps', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Mean Absolute IG Value', fontsize=12, fontweight='bold')
            axes[i].tick_params(axis='both', which='major', labelsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "temporal_patterns_top3.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Temporal clustering visualization
        if temporal_clusters is not None:
            print("  Creating temporal clustering visualization...")
            plt.figure(figsize=(15, 8))
            
            # Create a plot showing temporal clusters
            n_timesteps = len(temporal_clusters)
            timesteps = range(n_timesteps)
            
            # Plot clusters as colored regions
            unique_clusters = np.unique(temporal_clusters)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
            
            for i, cluster_id in enumerate(unique_clusters):
                cluster_timesteps = np.where(temporal_clusters == cluster_id)[0]
                if len(cluster_timesteps) > 0:
                    plt.axvspan(cluster_timesteps[0], cluster_timesteps[-1], 
                              alpha=0.3, color=colors[i], 
                              label=f'Cluster {cluster_id}')
            
            plt.title('Temporal Clustering of IG Patterns', fontsize=16, fontweight='bold')
            plt.xlabel('Time Steps', fontsize=14, fontweight='bold')
            plt.ylabel('Cluster ID', fontsize=14, fontweight='bold')
            plt.xticks(fontsize=12, fontweight='bold')
            plt.yticks(fontsize=12, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / "temporal_clustering_visualization.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Cross-variable temporal correlations heatmap
        print("  Creating correlation heatmap...")
        plt.figure(figsize=(12, 10))
        
        # Calculate correlation matrix for temporal patterns
        n_vars = len(self.variable_names)
        corr_matrix = np.zeros((n_vars, n_vars))
        
        for i, var1 in enumerate(self.variable_names):
            for j, var2 in enumerate(self.variable_names):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    # Calculate correlation between temporal patterns
                    var1_data = []
                    var2_data = []
                    
                    for basin_id, basin_data in temporal_data.items():
                        # basin_data shape: (samples, timesteps, variables)
                        var1_timestep_means = np.mean(np.abs(basin_data[:, :, i]), axis=0)  # Shape: (timesteps,)
                        var2_timestep_means = np.mean(np.abs(basin_data[:, :, j]), axis=0)  # Shape: (timesteps,)
                        
                        var1_data.extend(var1_timestep_means)
                        var2_data.extend(var2_timestep_means)
                    
                    # Calculate correlation
                    if len(var1_data) > 1 and len(var2_data) > 1:
                        corr, _ = pearsonr(var1_data, var2_data)
                        corr_matrix[i, j] = corr
                    else:
                        corr_matrix[i, j] = 0.0
        
        # Create correlation heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.3f', cbar_kws={'label': 'Temporal Correlation'},
                   xticklabels=self.variable_names, yticklabels=self.variable_names)
        
        plt.title('Temporal Correlations Between Variables', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "temporal_correlations_heatmap.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Variable importance over time
        print("  Creating variable importance over time plot...")
        plt.figure(figsize=(15, 10))
        
        # Calculate mean importance for each variable over time
        for i, var_name in enumerate(self.variable_names):
            all_temporal_data = []
            for basin_id, basin_data in temporal_data.items():
                var_data = basin_data[:, :, i]  # Shape: (samples, timesteps)
                timestep_means = np.mean(np.abs(var_data), axis=0)  # Shape: (timesteps,)
                all_temporal_data.append(timestep_means)
            
            combined_temporal = np.array(all_temporal_data)
            mean_series = np.mean(combined_temporal, axis=0)
            
            # Reverse the data so that left side shows recent data (1 day ago) and right side shows old data (365 days ago)
            mean_series_reversed = np.flip(mean_series)
            x_values = range(1, len(mean_series_reversed) + 1)  # From 1 to 365
            
            plt.plot(x_values, mean_series_reversed, 
                    linewidth=2, label=var_name, alpha=0.8)
        
        plt.title('Variable Importance Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Days Before Prediction (1 to 365)', fontsize=14, fontweight='bold')
        plt.ylabel('Mean Absolute IG Value', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "variable_importance_over_time.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[SAVE] Temporal plots saved to: {self.output_dir}")
    
    def run_temporal_analysis(self, sample_basins=None):
        """Run complete temporal analysis"""
        print("[START] Starting temporal analysis...")
        
        # Load data
        temporal_data, basin_info = self.load_temporal_data(sample_basins)
        
        # Analyze seasonal patterns
        seasonal_stats = self.analyze_seasonal_patterns(temporal_data)
        
        # Analyze temporal correlations
        temporal_correlations = self.analyze_temporal_correlations(temporal_data)
        
        # Analyze temporal clustering
        cluster_analysis, temporal_clusters = self.analyze_temporal_clustering(temporal_data)
        
        # Create plots
        self.create_seasonal_plots(seasonal_stats)
        self.create_temporal_plots(temporal_data, seasonal_stats, temporal_clusters)
        
        print("[SUCCESS] Temporal analysis completed!")
        print(f"[SAVE] Results saved to: {self.output_dir}")
        
        return {
            'seasonal_stats': seasonal_stats,
            'temporal_correlations': temporal_correlations,
            'cluster_analysis': cluster_analysis,
            'temporal_clusters': temporal_clusters
        }

def main():
    """Main function"""
    analyzer = TemporalAnalyzer()
    
    # Analyze temporal patterns using all 135 basins
    sample_basins = [f"{i:03d}" for i in range(1, 136)]  # All 135 basins
    results = analyzer.run_temporal_analysis(sample_basins=sample_basins)
    
    print("\n[SUMMARY] Temporal Analysis Results:")
    print(f"- Analyzed {len(sample_basins)} basins")
    print(f"- Created {len(results['cluster_analysis'])} temporal clusters")
    print(f"- Seasonal patterns identified for {len(results['seasonal_stats'])} variables")

if __name__ == "__main__":
    main()
