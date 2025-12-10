"""
Imaging Pipeline Manager

Separate pipeline manager for imaging preprocessing (Pipeline 2b).
Handles Suite2p → processed imaging data with validation checkpoints.

This is a standalone utility that can be run independently or integrated
into the main pipeline later.
"""

import os
import logging
import importlib
from datetime import datetime
from typing import Dict, Any, List, Optional

from modules.core.config_manager import ConfigManager
import modules.utils as utils


class ImagingPipelineManager:
    """
    Pipeline manager specifically for imaging preprocessing.
    
    Handles the complete Pipeline 2b workflow:
    1. Load Suite2p results
    2. Apply quality control filtering  
    3. Run cell type labeling
    4. Extract ΔF/F traces
    5. Generate validation reports
    6. Create QC checkpoint
    """
    
    def __init__(self, config_path: str, experiment_config: str, 
                 subject_selection: str = None, run_id: str = None, logger=None):
        """
        Initialize imaging pipeline manager.
        
        Args:
            config_path: Path to configuration YAML file
            experiment_config: Name of experiment configuration to use
            subject_selection: Subject selection (same format as main pipeline)
            run_id: Unique run identifier
            logger: Logger instance
        """
        # Initialize ConfigManager with experiment name
        self.config_manager = ConfigManager(config_path, experiment_name=experiment_config)
        self.config = self.config_manager.config
        self.experiment_name = experiment_config
        self.experiment_config = self.config['experiment_configs'][experiment_config]
        self.run_id = run_id or f"imaging_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = logger or logging.getLogger(__name__)
        
        # Get subjects with imaging sessions
        self.subject_list = self._get_imaging_subjects(subject_selection)
        
        # Initialize components
        self.imaging_preprocessor = None
        
        self.logger.info(f"IMP: Imaging pipeline manager initialized")
        self.logger.info(f"IMP: Experiment: {self.experiment_name}")
        self.logger.info(f"IMP: Subjects with imaging: {self.subject_list}")
    
    def _get_imaging_subjects(self, subject_selection: str = None) -> List[str]:
        """Get list of subjects that have imaging sessions."""
        # Start with all subjects for this experiment
        experiment_subjects = list(self.experiment_config.get('subject_configs', {}).keys())
        
        # Filter to only subjects with imaging sessions
        imaging_subjects = []
        for subject_id in experiment_subjects:
            subject_config = self.config.get('subjects', {}).get(subject_id, {})
            if 'imaging_sessions' in subject_config:
                imaging_subjects.append(subject_id)
        
        # Apply subject selection if provided
        if subject_selection:
            if ',' in subject_selection:
                # Comma-separated list
                selected_subjects = [s.strip() for s in subject_selection.split(',')]
                imaging_subjects = [s for s in imaging_subjects if s in selected_subjects]
            elif subject_selection in self.config.get('subject_groups', {}):
                # Subject group
                group_subjects = self.config['subject_groups'][subject_selection]
                imaging_subjects = [s for s in imaging_subjects if s in group_subjects]
            else:
                # Single subject
                imaging_subjects = [s for s in imaging_subjects if s == subject_selection]
        
        return imaging_subjects
    
    def initialize_imaging_preprocessor(self, force: bool = False) -> bool:
        """Initialize the imaging preprocessor."""
        self.logger.info("IMP: Initializing imaging preprocessor...")
        
        try:
            # Import the general imaging preprocessor
            module = importlib.import_module('modules.general_imaging_preprocessor')
            preprocessor_class = getattr(module, 'GeneralImagingPreprocessor')
            
            self.imaging_preprocessor = preprocessor_class(
                config_manager=self.config_manager,
                subject_list=self.subject_list,
                logger=self.logger
            )
            
            self.logger.info("IMP: Imaging preprocessor initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"IMP: Failed to initialize imaging preprocessor: {e}")
            return False
    
    def preprocess_imaging_sessions(self, force: bool = False) -> Dict[str, Any]:
        """
        Run complete imaging preprocessing pipeline.
        
        Args:
            force: Force reprocessing even if outputs exist
            
        Returns:
            Dictionary with preprocessing results
        """
        self.logger.info("IMP: Starting imaging preprocessing pipeline...")
        
        if not self.imaging_preprocessor:
            if not self.initialize_imaging_preprocessor(force):
                raise RuntimeError("Failed to initialize imaging preprocessor")
        
        # Get imaging preprocessing configuration
        imaging_config = self.experiment_config.get('imaging_preprocessing', {})
        
        if not imaging_config.get('enabled', False):
            raise ValueError(f"Imaging preprocessing not enabled for experiment {self.experiment_name}")
        
        # Run preprocessing for all subjects
        preprocessing_results = self.imaging_preprocessor.preprocess_imaging_data(
            experiment_config=imaging_config,
            force=force
        )
        
        self.logger.info("IMP: Imaging preprocessing pipeline completed")
        
        return {
            'pipeline_type': 'imaging_preprocessing',
            'experiment_config': self.experiment_name,
            'run_id': self.run_id,
            'subjects_processed': len(self.subject_list),
            'preprocessing_results': preprocessing_results,
            'completed_at': datetime.now().isoformat()
        }
    
    def generate_qc_checkpoint_report(self, preprocessing_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a checkpoint report for reviewing QC results.
        
        Args:
            preprocessing_results: Results from preprocessing pipeline
            
        Returns:
            Dictionary with checkpoint report information
        """
        self.logger.info("IMP: Generating QC checkpoint report...")
        
        # Create checkpoint summary
        checkpoint_report = {
            'checkpoint_type': 'imaging_preprocessing_qc',
            'experiment': self.experiment_name,
            'run_id': self.run_id,
            'generated_at': datetime.now().isoformat(),
            'subjects_summary': {},
            'overall_status': 'unknown',
            'recommendations': []
        }
        
        total_sessions = 0
        successful_sessions = 0
        
        # Get the actual preprocessing results - they're nested under 'preprocessing_results'
        actual_results = preprocessing_results.get('preprocessing_results', {})
        
        # Analyze results per subject
        for subject_id in self.subject_list:
            subject_results = actual_results.get('subject_results', {}).get(subject_id, {})
            
            sessions_processed = subject_results.get('sessions_processed', 0)
            sessions_successful = subject_results.get('sessions_successful', 0)
            
            # Debug logging
            self.logger.info(f"IMP: Subject {subject_id}: {sessions_processed} processed, {sessions_successful} successful")
            
            total_sessions += sessions_processed
            successful_sessions += sessions_successful
            
            checkpoint_report['subjects_summary'][subject_id] = {
                'sessions_processed': sessions_processed,
                'sessions_successful': sessions_successful,
                'success_rate': sessions_successful / sessions_processed if sessions_processed > 0 else 0,
                'status': 'good' if sessions_successful == sessions_processed and sessions_processed > 0 else 'needs_review'
            }
        
        # Debug logging
        self.logger.info(f"IMP: Total sessions: {total_sessions}, successful: {successful_sessions}")
        
        # Overall assessment
        overall_success_rate = successful_sessions / total_sessions if total_sessions > 0 else 0
        
        if total_sessions == 0:
            checkpoint_report['overall_status'] = 'no_sessions_found'
            checkpoint_report['recommendations'].append("✗ No imaging sessions found to process")
        elif overall_success_rate >= 0.9:
            checkpoint_report['overall_status'] = 'ready_for_analysis'
            checkpoint_report['recommendations'].append("✓ Preprocessing completed successfully - ready for neural analysis")
        elif overall_success_rate >= 0.7:
            checkpoint_report['overall_status'] = 'review_failures'
            checkpoint_report['recommendations'].append("⚠ Most sessions processed successfully - review failed sessions")
        else:
            checkpoint_report['overall_status'] = 'needs_attention'
            checkpoint_report['recommendations'].append("✗ Multiple processing failures - check configurations and data quality")
        
        # Add specific recommendations
        if any(s['status'] == 'needs_review' for s in checkpoint_report['subjects_summary'].values()):
            checkpoint_report['recommendations'].append("• Review QC validation plots in qc_results/plots/ directories")
            checkpoint_report['recommendations'].append("• Check QC reports in qc_results/qc_report.json files")
            checkpoint_report['recommendations'].append("• Verify imaging session paths and Suite2p data quality")
        
        # Add success-specific recommendations
        if checkpoint_report['overall_status'] == 'ready_for_analysis':
            checkpoint_report['recommendations'].append("• Check generated dff.h5 files for ΔF/F traces")
            checkpoint_report['recommendations'].append("• Review quality control metrics in qc_results/ directories")
            checkpoint_report['recommendations'].append("• Proceed with neural analysis when ready")
        
        self.logger.info("IMP: QC checkpoint report generated")
        
        return checkpoint_report
    
    def save_checkpoint_report(self, checkpoint_report: Dict[str, Any], 
                              output_dir: str = "output/imaging_checkpoints") -> str:
        """Save checkpoint report to file."""
        os.makedirs(output_dir, exist_ok=True)
        
        import json
        report_filename = f"imaging_qc_checkpoint_{self.run_id}.json"
        report_path = os.path.join(output_dir, report_filename)
        
        with open(report_path, 'w') as f:
            json.dump(checkpoint_report, f, indent=2)
        
        self.logger.info(f"IMP: Checkpoint report saved to {report_path}")
        return report_path
    
    def print_checkpoint_summary(self, checkpoint_report: Dict[str, Any]):
        """Print a formatted summary of the checkpoint report."""
        print("\n" + "="*60)
        print("IMAGING PREPROCESSING CHECKPOINT")
        print("="*60)
        
        print(f"Experiment: {checkpoint_report['experiment']}")
        print(f"Run ID: {checkpoint_report['run_id']}")
        print(f"Overall Status: {checkpoint_report['overall_status'].upper()}")
        
        print(f"\nSUBJECT SUMMARY:")
        for subject_id, summary in checkpoint_report['subjects_summary'].items():
            status_icon = "✓" if summary['status'] == 'good' else "⚠"
            print(f"  {status_icon} {subject_id}: {summary['sessions_successful']}/{summary['sessions_processed']} sessions")
        
        print(f"\nRECOMMENDATIONS:")
        for rec in checkpoint_report['recommendations']:
            print(f"  {rec}")
        
        print(f"\nNEXT STEPS:")
        if checkpoint_report['overall_status'] == 'ready_for_analysis':
            print("  → Proceed with neural analysis pipeline")
        else:
            print("  → Review failed sessions and QC reports")
            print("  → Fix issues and re-run preprocessing if needed")
            print("  → Check QC validation plots for data quality")
        
        print("="*60)
    
    def run_complete_pipeline(self, force: bool = False) -> Dict[str, Any]:
        """
        Run the complete imaging preprocessing pipeline with checkpoint.
        
        Args:
            force: Force reprocessing
            
        Returns:
            Complete pipeline results including checkpoint
        """
        self.logger.info("IMP: Running complete imaging preprocessing pipeline...")
        
        # Step 1: Run preprocessing
        preprocessing_results = self.preprocess_imaging_sessions(force=force)
        
        # Step 2: Generate checkpoint report
        checkpoint_report = self.generate_qc_checkpoint_report(preprocessing_results)
        
        # Step 3: Save checkpoint report
        report_path = self.save_checkpoint_report(checkpoint_report)
        
        # Step 4: Print summary
        self.print_checkpoint_summary(checkpoint_report)
        
        # Return complete results
        return {
            'preprocessing_results': preprocessing_results,
            'checkpoint_report': checkpoint_report,
            'checkpoint_report_path': report_path,
            'pipeline_completed_at': datetime.now().isoformat()
        }
