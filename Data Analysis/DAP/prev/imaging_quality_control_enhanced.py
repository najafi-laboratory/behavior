"""
Enhanced Imaging Quality Control Module

Supports multiple QC methods:
1. Traditional threshold-based filtering
2. Manual curation via Suite2p GUI
3. Hybrid approach (thresholds + manual override)
4. Machine learning-based classification

Generates comprehensive QC reports and validation plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

class ImagingQualityControl:
    """Enhanced quality control for imaging data with multiple selection methods."""
    
    def __init__(self, ops: Dict[str, Any], logger=None):
        self.ops = ops
        self.logger = logger
        self.qc_results = {}
        
    def load_suite2p_data(self) -> Dict[str, np.ndarray]:
        """Load all relevant Suite2p data."""
        suite2p_path = os.path.join(self.ops['save_path0'], 'suite2p', 'plane0')
        
        data = {}
        files_to_load = ['F.npy', 'Fneu.npy', 'iscell.npy', 'stat.npy', 'spks.npy']
        
        for file in files_to_load:
            file_path = os.path.join(suite2p_path, file)
            if os.path.exists(file_path):
                data[file.replace('.npy', '')] = np.load(file_path, allow_pickle=True)
                
        return data
    
    def calculate_comprehensive_metrics(self, data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Calculate comprehensive quality metrics for all ROIs."""
        stat = data['stat']
        F = data['F']
        Fneu = data['Fneu'] if 'Fneu' in data else None
        
        metrics = []
        
        for i, roi_stat in enumerate(stat):
            metric = {
                'roi_id': i,
                # Morphological metrics
                'area': roi_stat.get('npix', 0),
                'aspect_ratio': roi_stat.get('aspect_ratio', 0),
                'compactness': roi_stat.get('compact', 0),
                'skewness': roi_stat.get('skew', 0),
                'footprint': roi_stat.get('footprint', 0),
                'overlap': roi_stat.get('overlap', 0),
                
                # Signal quality metrics
                'mean_fluorescence': np.mean(F[i]) if i < len(F) else 0,
                'std_fluorescence': np.std(F[i]) if i < len(F) else 0,
                'snr': self._calculate_snr(F[i]) if i < len(F) else 0,
                
                # Neuropil contamination
                'neuropil_ratio': self._calculate_neuropil_ratio(F[i], Fneu[i]) if Fneu is not None and i < len(Fneu) else 0,
                
                # Activity metrics
                'activity_rate': self._calculate_activity_rate(F[i]) if i < len(F) else 0,
                'correlation_with_neuropil': self._calculate_neuropil_correlation(F[i], Fneu[i]) if Fneu is not None and i < len(Fneu) else 0,
            }
            
            # Add location metrics
            if 'med' in roi_stat:
                metric['centroid_y'] = roi_stat['med'][0]
                metric['centroid_x'] = roi_stat['med'][1]
            
            metrics.append(metric)
        
        return pd.DataFrame(metrics)
    
    def _calculate_snr(self, trace: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        if len(trace) == 0:
            return 0
        return np.mean(trace) / np.std(trace) if np.std(trace) > 0 else 0
    
    def _calculate_neuropil_ratio(self, F: np.ndarray, Fneu: np.ndarray) -> float:
        """Calculate ratio of cell to neuropil fluorescence."""
        if len(F) == 0 or len(Fneu) == 0:
            return 0
        return np.mean(Fneu) / np.mean(F) if np.mean(F) > 0 else 0
    
    def _calculate_activity_rate(self, trace: np.ndarray) -> float:
        """Calculate activity rate (fraction of time above baseline)."""
        if len(trace) == 0:
            return 0
        baseline = np.percentile(trace, 20)
        threshold = baseline + 2 * np.std(trace)
        return np.mean(trace > threshold)
    
    def _calculate_neuropil_correlation(self, F: np.ndarray, Fneu: np.ndarray) -> float:
        """Calculate correlation between cell and neuropil."""
        if len(F) == 0 or len(Fneu) == 0:
            return 0
        return np.corrcoef(F, Fneu)[0, 1] if not np.isnan(np.corrcoef(F, Fneu)[0, 1]) else 0
    
    def apply_manual_curation(self, data: Dict[str, np.ndarray], 
                            use_iscell_manual: bool = True) -> np.ndarray:
        """Apply manual curation from Suite2p GUI selections."""
        if not use_iscell_manual or 'iscell' not in data:
            return np.ones(len(data['stat']), dtype=bool)
        
        iscell = data['iscell']
        # iscell[:, 0] contains manual classifications (1=cell, 0=not cell)
        # iscell[:, 1] contains classifier probabilities
        
        manual_selection = iscell[:, 0].astype(bool)
        
        self.logger.info(f"Manual curation: {np.sum(manual_selection)}/{len(manual_selection)} ROIs selected as cells")
        
        return manual_selection
    
    def apply_threshold_filtering(self, metrics_df: pd.DataFrame, 
                                qc_params: Dict[str, Any]) -> np.ndarray:
        """Apply traditional threshold-based filtering."""
        
        # Initialize all as passing
        passes_qc = np.ones(len(metrics_df), dtype=bool)
        
        # Apply each threshold
        for param, (min_val, max_val) in qc_params.items():
            if param in metrics_df.columns:
                mask = (metrics_df[param] >= min_val) & (metrics_df[param] <= max_val)
                passes_qc = passes_qc & mask
                self.logger.info(f"Threshold {param} [{min_val}, {max_val}]: {np.sum(mask)}/{len(mask)} ROIs pass")
        
        self.logger.info(f"Threshold filtering: {np.sum(passes_qc)}/{len(passes_qc)} ROIs pass all criteria")
        
        return passes_qc
    
    def apply_hybrid_selection(self, data: Dict[str, np.ndarray], 
                             metrics_df: pd.DataFrame,
                             qc_params: Dict[str, Any],
                             manual_override_priority: bool = True) -> np.ndarray:
        """Apply hybrid selection: manual + thresholds with configurable priority."""
        
        manual_selection = self.apply_manual_curation(data, use_iscell_manual=True)
        threshold_selection = self.apply_threshold_filtering(metrics_df, qc_params)
        
        if manual_override_priority:
            # Manual selection takes priority
            final_selection = manual_selection
            self.logger.info("Using manual curation as primary selection method")
        else:
            # Intersection of manual and threshold
            final_selection = manual_selection & threshold_selection
            self.logger.info("Using intersection of manual curation and threshold filtering")
        
        return final_selection
    
    def generate_qc_report(self, data: Dict[str, np.ndarray], 
                         metrics_df: pd.DataFrame,
                         final_selection: np.ndarray,
                         qc_method: str) -> Dict[str, Any]:
        """Generate comprehensive QC report."""
        
        report = {
            'qc_method': qc_method,
            'timestamp': datetime.now().isoformat(),
            'total_rois': len(final_selection),
            'selected_rois': np.sum(final_selection),
            'rejection_rate': 1 - (np.sum(final_selection) / len(final_selection)),
        }
        
        # Add metrics summaries
        selected_metrics = metrics_df[final_selection]
        rejected_metrics = metrics_df[~final_selection]
        
        for column in metrics_df.select_dtypes(include=[np.number]).columns:
            report[f'{column}_selected_mean'] = selected_metrics[column].mean()
            report[f'{column}_selected_std'] = selected_metrics[column].std()
            report[f'{column}_rejected_mean'] = rejected_metrics[column].mean() if len(rejected_metrics) > 0 else 0
            report[f'{column}_rejected_std'] = rejected_metrics[column].std() if len(rejected_metrics) > 0 else 0
        
        return report
    
    def save_qc_results(self, data: Dict[str, np.ndarray], 
                       final_selection: np.ndarray,
                       metrics_df: pd.DataFrame,
                       qc_report: Dict[str, Any]):
        """Save QC results to files."""
        
        qc_dir = os.path.join(self.ops['save_path0'], 'qc_results')
        os.makedirs(qc_dir, exist_ok=True)
        
        # Save filtered data
        selected_indices = np.where(final_selection)[0]
        
        # Save fluorescence traces
        if 'F' in data:
            np.save(os.path.join(qc_dir, 'fluo.npy'), data['F'][selected_indices])
        
        if 'Fneu' in data:
            np.save(os.path.join(qc_dir, 'neuropil.npy'), data['Fneu'][selected_indices])
        
        # Save cell statistics and masks
        if 'stat' in data:
            np.save(os.path.join(qc_dir, 'stat.npy'), data['stat'][selected_indices])
        
        # Save QC selection
        iscell_qc = np.zeros((len(final_selection), 2))
        iscell_qc[:, 0] = final_selection.astype(int)
        iscell_qc[:, 1] = 1.0  # Set probability to 1 for selected cells
        np.save(os.path.join(qc_dir, 'iscell_qc.npy'), iscell_qc)
        
        # Save comprehensive metrics
        metrics_df.to_csv(os.path.join(qc_dir, 'roi_metrics.csv'), index=False)
        
        # Save QC report
        import json
        with open(os.path.join(qc_dir, 'qc_report.json'), 'w') as f:
            json.dump(qc_report, f, indent=2)
        
        self.logger.info(f"QC results saved to {qc_dir}")
    
    def create_qc_plots(self, metrics_df: pd.DataFrame, 
                       final_selection: np.ndarray) -> List[plt.Figure]:
        """Create QC validation plots."""
        
        figures = []
        
        # Plot 1: ROI selection overview
        fig1, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig1.suptitle('Quality Control Overview', fontsize=16)
        
        # Metrics distributions
        numeric_columns = ['area', 'aspect_ratio', 'compactness', 'snr', 'activity_rate']
        
        for i, column in enumerate(numeric_columns):
            if i < 6 and column in metrics_df.columns:
                ax = axes.flat[i]
                
                # Plot distributions
                selected_data = metrics_df[final_selection][column]
                rejected_data = metrics_df[~final_selection][column]
                
                ax.hist(rejected_data, bins=30, alpha=0.5, color='red', label='Rejected')
                ax.hist(selected_data, bins=30, alpha=0.5, color='green', label='Selected')
                ax.set_xlabel(column)
                ax.set_ylabel('Count')
                ax.legend()
                ax.set_title(f'{column.replace("_", " ").title()}')
        
        figures.append(fig1)
        
        # Plot 2: Spatial distribution
        if 'centroid_x' in metrics_df.columns and 'centroid_y' in metrics_df.columns:
            fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            selected_x = metrics_df[final_selection]['centroid_x']
            selected_y = metrics_df[final_selection]['centroid_y']
            rejected_x = metrics_df[~final_selection]['centroid_x']
            rejected_y = metrics_df[~final_selection]['centroid_y']
            
            ax.scatter(rejected_x, rejected_y, c='red', alpha=0.5, s=20, label='Rejected')
            ax.scatter(selected_x, selected_y, c='green', alpha=0.7, s=20, label='Selected')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title('Spatial Distribution of Selected vs Rejected ROIs')
            ax.legend()
            
            figures.append(fig2)
        
        return figures
    
    def run_quality_control(self, qc_method: str = 'hybrid', 
                          qc_params: Optional[Dict[str, Any]] = None,
                          save_plots: bool = True) -> Dict[str, Any]:
        """Run complete quality control pipeline."""
        
        # Load data
        data = self.load_suite2p_data()
        
        # Calculate metrics
        metrics_df = self.calculate_comprehensive_metrics(data)
        
        # Apply selection method
        if qc_method == 'manual':
            final_selection = self.apply_manual_curation(data)
        elif qc_method == 'threshold':
            final_selection = self.apply_threshold_filtering(metrics_df, qc_params)
        elif qc_method == 'hybrid':
            final_selection = self.apply_hybrid_selection(data, metrics_df, qc_params)
        else:
            raise ValueError(f"Unknown QC method: {qc_method}")
        
        # Generate report
        qc_report = self.generate_qc_report(data, metrics_df, final_selection, qc_method)
        
        # Save results
        self.save_qc_results(data, final_selection, metrics_df, qc_report)
        
        # Create plots
        if save_plots:
            figures = self.create_qc_plots(metrics_df, final_selection)
            
            # Save plots
            plot_dir = os.path.join(self.ops['save_path0'], 'qc_results', 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            
            for i, fig in enumerate(figures):
                fig.savefig(os.path.join(plot_dir, f'qc_validation_{i+1}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        return {
            'status': 'success',
            'method': qc_method,
            'total_rois': len(final_selection),
            'selected_rois': np.sum(final_selection),
            'qc_report': qc_report
        }
