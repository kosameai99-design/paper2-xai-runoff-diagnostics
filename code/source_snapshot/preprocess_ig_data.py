#!/usr/bin/env python3
"""
IG Data Preprocessing Script (TEST PERIOD)
==========================================

This script processes the raw IG batch data for TEST period and creates consolidated files
for easier analysis. It merges multiple batches per basin into single files
and creates summary statistics.

Author: AI Assistant
Date: 2026-01-05
Period: TEST
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class IGDataPreprocessor:
    """Preprocess IG batch data for easier analysis"""
    
    def __init__(self, data_dir, epoch="018"):
        self.data_dir = Path(data_dir)
        self.epoch = epoch
        # For test period, data is in epoch_* subdirectory (name can be padded, e.g., epoch_020 / epoch_000)
        self.epoch_dir = self._resolve_epoch_dir(epoch)
        # Output should be in the same directory as epoch_dir (test/processed_data)
        self.output_dir = self.data_dir / "processed_data"
        self.output_dir.mkdir(exist_ok=True)
        
        # Variable names (11 variables: optimized)
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
        
        print(f"[IG] IG Data Preprocessor - TEST PERIOD")
        print(f"[DIR] Data directory: {self.data_dir}")
        print(f"[EPOCH] Epoch: {epoch}")
        print(f"[EPOCH_DIR] Using epoch directory: {self.epoch_dir}")
        print(f"[OUT] Output directory: {self.output_dir}")
        print("=" * 50)

    def _resolve_epoch_dir(self, epoch):
        """Resolve epoch directory name robustly.

        Supports common naming variants like:
        - epoch_20
        - epoch_020 (3-digit padding)
        - epoch_0020 (4-digit padding)
        If none exist, falls back to existing epoch_* directories under data_dir.
        """
        # Fast path: user provided full directory name or Path-like
        if isinstance(epoch, Path):
            candidate = epoch
            if candidate.is_absolute():
                if candidate.exists():
                    return candidate
            else:
                candidate = self.data_dir / candidate
                if candidate.exists():
                    return candidate

        epoch_str = str(epoch)
        # If user already passed "epoch_XXX"
        if epoch_str.startswith("epoch_"):
            candidate = self.data_dir / epoch_str
            if candidate.exists():
                return candidate

        # Try common patterns
        candidates = [self.data_dir / f"epoch_{epoch_str}"]
        try:
            epoch_int = int(epoch_str)
            candidates.extend([
                self.data_dir / f"epoch_{epoch_int:03d}",
                self.data_dir / f"epoch_{epoch_int:04d}",
            ])
        except ValueError:
            # Non-numeric epoch strings: nothing to pad
            pass

        for c in candidates:
            if c.exists():
                return c

        # Fallback: pick from existing epoch_* directories
        epoch_dirs = sorted([p for p in self.data_dir.glob("epoch_*") if p.is_dir()])
        if len(epoch_dirs) == 1:
            return epoch_dirs[0]

        # If multiple exist, prefer ones that end with padded numeric epoch
        if epoch_dirs:
            preferred = []
            for c in candidates:
                # candidates are absolute paths; compare by name
                name = c.name
                preferred.extend([p for p in epoch_dirs if p.name == name])
            if preferred:
                return preferred[0]

        available = [p.name for p in epoch_dirs]
        raise FileNotFoundError(
            f"Epoch directory not found. Tried: {[p.name for p in candidates]} under {self.data_dir}. "
            f"Available epoch_* dirs: {available}"
        )
    
    def find_ig_files(self):
        """Find all IG batch files"""
        if not self.epoch_dir.exists():
            raise FileNotFoundError(f"Epoch directory not found after resolution: {self.epoch_dir}")
        
        # Find all *_batch*.npy files
        ig_files = list(self.epoch_dir.glob("*_batch*.npy"))
        print(f"[FOUND] Found {len(ig_files)} IG batch files")
        
        # Group by basin
        basin_files = {}
        for file in ig_files:
            # Extract basin ID from filename like "001_batch0.npy"
            parts = file.stem.split('_')
            if len(parts) >= 2:
                basin_id = parts[0]  # e.g., "001"
                batch_num = int(parts[1].replace('batch', ''))  # e.g., 0
                
                if basin_id not in basin_files:
                    basin_files[basin_id] = []
                basin_files[basin_id].append((batch_num, file))
        
        # Sort batches by number
        for basin_id in basin_files:
            basin_files[basin_id].sort(key=lambda x: x[0])
        
        print(f"[BASINS] Found data for {len(basin_files)} basins")
        return basin_files
    
    def process_basin_data(self, basin_id, batch_files):
        """Process all batches for a single basin"""
        print(f"[PROCESS] Processing basin {basin_id}...")
        
        # Load all batches for this basin
        batch_data = []
        batch_info = []
        
        for batch_num, file_path in batch_files:
            try:
                data = np.load(file_path)
                batch_data.append(data)
                batch_info.append({
                    'batch_num': batch_num,
                    'file_path': str(file_path),
                    'shape': data.shape,
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data))
                })
            except Exception as e:
                print(f"[WARNING] Error loading {file_path}: {e}")
                continue
        
        if not batch_data:
            print(f"[ERROR] No valid data for basin {basin_id}")
            return None
        
        # Concatenate all batches along the first dimension (samples)
        combined_data = np.concatenate(batch_data, axis=0)
        
        # Calculate summary statistics
        summary_stats = {
            'basin_id': basin_id,
            'num_batches': len(batch_data),
            'total_samples': combined_data.shape[0],
            'time_steps': combined_data.shape[1],
            'variables': combined_data.shape[2],
            'data_shape': list(combined_data.shape),
            'overall_mean': float(np.mean(combined_data)),
            'overall_std': float(np.std(combined_data)),
            'overall_min': float(np.min(combined_data)),
            'overall_max': float(np.max(combined_data)),
            'batch_info': batch_info,
            'period': 'test'  # Mark as test period
        }
        
        # Save consolidated data
        basin_output_dir = self.output_dir / f"basin_{basin_id}"
        basin_output_dir.mkdir(exist_ok=True)
        
        # Save the combined array
        np.save(basin_output_dir / f"ig_data_combined.npy", combined_data)
        
        # Save summary statistics
        with open(basin_output_dir / "summary_stats.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Create a more readable CSV format (sample of data)
        # Take every 10th sample to keep file size manageable
        sample_data = combined_data[::10]  # Every 10th sample
        
        # Create DataFrame with time series structure
        n_samples, n_timesteps, n_vars = sample_data.shape
        
        # Flatten to create a long-format DataFrame
        data_rows = []
        for sample_idx in range(min(100, n_samples)):  # Limit to 100 samples for CSV
            for timestep in range(n_timesteps):
                row = {
                    'sample_id': sample_idx,
                    'timestep': timestep,
                    'basin_id': basin_id
                }
                for var_idx, var_name in enumerate(self.variable_names):
                    row[var_name] = sample_data[sample_idx, timestep, var_idx]
                data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        df.to_csv(basin_output_dir / "ig_data_sample.csv", index=False)
        
        print(f"[SUCCESS] Basin {basin_id}: {combined_data.shape[0]} samples, {len(batch_data)} batches")
        return summary_stats
    
    def create_global_summary(self, all_summaries):
        """Create global summary of all basins"""
        print("[SUMMARY] Creating global summary...")
        
        # Aggregate statistics
        total_samples = sum(s['total_samples'] for s in all_summaries)
        total_batches = sum(s['num_batches'] for s in all_summaries)
        
        # Calculate global statistics
        all_means = [s['overall_mean'] for s in all_summaries]
        all_stds = [s['overall_std'] for s in all_summaries]
        all_mins = [s['overall_min'] for s in all_summaries]
        all_maxs = [s['overall_max'] for s in all_summaries]
        
        global_summary = {
            'preprocessing_timestamp': datetime.now().isoformat(),
            'period': 'test',
            'total_basins': len(all_summaries),
            'total_samples': total_samples,
            'total_batches': total_batches,
            'time_steps': all_summaries[0]['time_steps'] if all_summaries else 0,
            'variables': all_summaries[0]['variables'] if all_summaries else 0,
            'global_statistics': {
                'mean': float(np.mean(all_means)),
                'std': float(np.std(all_means)),
                'min': float(np.min(all_mins)),
                'max': float(np.max(all_maxs)),
                'median': float(np.median(all_means))
            },
            'basin_summaries': all_summaries
        }
        
        # Save global summary
        with open(self.output_dir / "global_summary.json", 'w') as f:
            json.dump(global_summary, f, indent=2)
        
        # Create a summary DataFrame
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(self.output_dir / "basin_summaries.csv", index=False)
        
        print(f"[SUCCESS] Global summary created")
        print(f"[STATS] Total basins: {len(all_summaries)}")
        print(f"[STATS] Total samples: {total_samples:,}")
        print(f"[STATS] Total batches: {total_batches}")
        
        return global_summary
    
    def run_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        print("[START] Starting IG data preprocessing for TEST period...")
        
        # Find all IG files
        basin_files = self.find_ig_files()
        
        if not basin_files:
            print("[ERROR] No IG files found!")
            return
        
        # Process each basin
        all_summaries = []
        for basin_id, batch_files in basin_files.items():
            summary = self.process_basin_data(basin_id, batch_files)
            if summary:
                all_summaries.append(summary)
        
        # Create global summary
        if all_summaries:
            global_summary = self.create_global_summary(all_summaries)
            print("[SUCCESS] Preprocessing completed successfully!")
            return global_summary
        else:
            print("[ERROR] No valid data processed!")
            return None

def main():
    """Main function"""
    # Configuration - run from test/analysis_scripts directory
    # Need to go up one level to reach test/ directory where epoch_20 is located
    data_dir = Path("..")  # Go up one level to test/ directory
    # Note: IG outputs may be named epoch_20 / epoch_020 / epoch_000 depending on your pipeline.
    # You can set epoch="20" or epoch="000"; if it doesn't exist, the script will fall back to existing epoch_*.
    epoch = "20"
    
    # Create preprocessor
    preprocessor = IGDataPreprocessor(data_dir, epoch)
    
    # Run preprocessing
    result = preprocessor.run_preprocessing()
    
    if result:
        print("\n[SUCCESS] Preprocessing completed for TEST period!")
        print(f"[SAVE] Processed data saved to: {preprocessor.output_dir}")
        print(f"[STATS] Processed {result['total_basins']} basins")
        print(f"[STATS] Total samples: {result['total_samples']:,}")
    else:
        print("[ERROR] Preprocessing failed!")

if __name__ == "__main__":
    main()

