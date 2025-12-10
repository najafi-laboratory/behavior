"""
General Imaging Preprocessor

Generic imaging preprocessor that handles suite2p data processing for any experiment.
This module provides the core imaging preprocessing functionality that can be used
across different experiment types.

Processing Pipeline:
1. Load suite2p data (ops.npy, F.npy, iscell.npy, etc.)
2. Quality control filtering of ROIs/cells
3. Cell type labeling (excitatory/inhibitory classification)
4. ΔF/F trace extraction
5. Save processed results

This is the base class that experiment-specific imaging preprocessors can inherit from.
"""

import os
import sys
import numpy as np
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import importlib.util

# Add the 2p_post_process_module to path for importing
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '2p_post_process_module_202404'))
print(f"Adding to path: {module_dir}")

if os.path.exists(module_dir):
    sys.path.insert(0, module_dir)  # Use insert(0, ...) to prioritize this path
    print(f"✓ 2p_post_process_module_202404 directory found")
else:
    print(f"✗ 2p_post_process_module_202404 directory not found at: {module_dir}")

# Check what's actually in the modules directory
if os.path.exists(module_dir):
    modules_subdir = os.path.join(module_dir, 'modules')
    if os.path.exists(modules_subdir):
        print(f"Contents of {modules_subdir}:")
        for item in os.listdir(modules_subdir):
            print(f"  - {item}")
    else:
        print(f"No 'modules' subdirectory found in {module_dir}")
        print(f"Contents of {module_dir}:")
        for item in os.listdir(module_dir):
            print(f"  - {item}")

# Import with specific module names to avoid conflicts
QualControlDataIO = None
LabelExcInh = None
DffTraces = None

try:
    print("Attempting to import QualControlDataIO...")
    # Import the specific module files directly
    qc_spec = importlib.util.spec_from_file_location(
        "QualControlDataIO", 
        os.path.join(module_dir, "modules", "QualControlDataIO.py")
    )
    QualControlDataIO = importlib.util.module_from_spec(qc_spec)
    qc_spec.loader.exec_module(QualControlDataIO)
    print("✓ QualControlDataIO imported successfully")
    
except Exception as e:
    print(f"✗ Failed to import QualControlDataIO: {e}")

try:
    print("Attempting to import LabelExcInh...")
    
    label_spec = importlib.util.spec_from_file_location(
        "LabelExcInh", 
        os.path.join(module_dir, "modules", "LabelExcInh.py")
    )
    LabelExcInh = importlib.util.module_from_spec(label_spec)
    label_spec.loader.exec_module(LabelExcInh)
    print("✓ LabelExcInh imported successfully")
    
except Exception as e:
    print(f"✗ Failed to import LabelExcInh: {e}")

try:
    print("Attempting to import DffTraces...")
    
    dff_spec = importlib.util.spec_from_file_location(
        "DffTraces", 
        os.path.join(module_dir, "modules", "DffTraces.py")
    )
    DffTraces = importlib.util.module_from_spec(dff_spec)
    dff_spec.loader.exec_module(DffTraces)
    print("✓ DffTraces imported successfully")
    
except Exception as e:
    print(f"✗ Failed to import DffTraces: {e}")

# Check what's actually in the modules directory
if os.path.exists(module_dir):
    modules_subdir = os.path.join(module_dir, 'modules')
    if os.path.exists(modules_subdir):
        print(f"Contents of {modules_subdir}:")
        for item in os.listdir(modules_subdir):
            print(f"  - {item}")
    else:
        print(f"No 'modules' subdirectory found in {module_dir}")
        print(f"Contents of {module_dir}:")
        for item in os.listdir(module_dir):
            print(f"  - {item}")


class GeneralImagingPreprocessor:
    """
    Generic imaging preprocessor for suite2p data processing.
    
    This class handles the core imaging preprocessing pipeline that is common
    across different experiment types. Experiment-specific preprocessors can
    inherit from this class and override methods as needed.
    """
    
    def __init__(self, config_manager, subject_list, logger=None):
        """
        Initialize the imaging preprocessor.
        
        Args:
            config_manager: ConfigManager instance
            subject_list: List of subject IDs to process
            logger: Logger instance
        """
        print(dir)
        
        self.config_manager = config_manager
        self.config = config_manager.config
        self.subject_list = subject_list
        self.logger = logger or logging.getLogger(__name__)
        
        # Get paths from config
        self.imaging_data_base = self.config.get('paths', {}).get('imaging_data_base', '')
        
        # Default quality control parameters (can be overridden by experiment-specific configs)
        self.default_qc_params = {
            'range_skew': [0, 2],           # For dendrites: [0,2], for neurons: [-5,5]
            'max_connect': 2,               # For dendrites: 2, for neurons: 1
            'range_aspect': [1.2, 5],       # For dendrites: [1.2,5], for neurons: [0,5]
            'range_footprint': [1, 2],      # Same for both
            'range_compact': [1.06, 5],     # For dendrites: [1.06,5], for neurons: [0,1.06]
            'diameter': 6                   # Cellpose diameter
        }
        
        self.logger.info("GIP: Initializing GeneralImagingPreprocessor...")
        dependencies_ok = self._validate_dependencies()
        if dependencies_ok:
            self.logger.info("GIP: GeneralImagingPreprocessor initialized successfully")
        else:
            self.logger.warning("GIP: GeneralImagingPreprocessor initialized with missing dependencies")
    
    def _validate_dependencies(self):
        """Validate that required 2p processing modules are available."""
        missing_modules = []
        
        if QualControlDataIO is None:
            missing_modules.append("QualControlDataIO")
        if LabelExcInh is None:
            missing_modules.append("LabelExcInh")
        if DffTraces is None:
            missing_modules.append("DffTraces")
        
        if missing_modules:
            error_msg = (
                f"Required 2p processing modules not available: {', '.join(missing_modules)}. "
                f"Check that 2p_post_process_module_202404 is accessible and contains the modules directory."
            )
            print(f"Dependency validation failed: {error_msg}")
            # Don't raise exception during testing - just warn
            self.logger.warning(f"GIP: {error_msg}")
            return False
        
        return True

    def get_imaging_session_path(self, subject_id: str, imaging_session: Dict[str, str]) -> str:
        """
        Construct the full path to an imaging session's suite2p results.
        
        Args:
            subject_id: Subject identifier
            imaging_session: Dict with imaging session info
            
        Returns:
            Full path to suite2p results directory
        """
        experiment_type = imaging_session.get('experiment_type', '2afc')
        imaging_folder = imaging_session.get('imaging_folder', '')
        subject_name = self.config['subjects'][subject_id].get('subject_name', subject_id)
        
        # Construct path: {imaging_data_base}/{experiment_type}/{subject_name}/{imaging_folder}/suite2p/plane0/
        path = os.path.join(
            self.imaging_data_base,
            experiment_type,
            subject_name,
            imaging_folder,
            'suite2p',
            'plane0'
        )
        
        return path
    
    def load_suite2p_data(self, suite2p_path: str) -> Dict[str, Any]:
        """
        Load suite2p data files.
        
        Args:
            suite2p_path: Path to suite2p plane0 directory
            
        Returns:
            Dictionary containing loaded suite2p data
        """
        self.logger.info(f"GIP: Loading suite2p data from {suite2p_path}")
        
        if not os.path.exists(suite2p_path):
            raise FileNotFoundError(f"Suite2p path not found: {suite2p_path}")
        
        # Load ops.npy (required)
        ops_path = os.path.join(suite2p_path, 'ops.npy')
        if not os.path.exists(ops_path):
            raise FileNotFoundError(f"ops.npy not found in {suite2p_path}")
        
        ops = np.load(ops_path, allow_pickle=True).item()
        
        # Set save_path0 to the imaging session root directory 
        # suite2p_path is: .../imaging_folder/suite2p/plane0/
        # We want save_path0 to be: .../imaging_folder/
        # So we need to go up 2 levels: plane0 -> suite2p -> imaging_folder
        imaging_session_root = os.path.dirname(os.path.dirname(suite2p_path))
        ops['save_path0'] = imaging_session_root
        
        # Create output directory for QC results (only one actually used)
        qc_results_dir = os.path.join(imaging_session_root, 'qc_results')
        os.makedirs(qc_results_dir, exist_ok=True)
        self.logger.info(f"GIP: Created/verified QC results directory: {qc_results_dir}")
        
        # Note: Main processed files (dff.h5, masks.h5, etc.) are saved to root level
        # This matches the original standalone processing behavior
        
        self.logger.info(f"GIP: Successfully loaded ops.npy with {ops.get('ncells', 'unknown')} cells")
        self.logger.info(f"GIP: Output will be saved to: {ops['save_path0']}")
        
        # Debug: Show the path structure
        self.logger.info(f"GIP: suite2p_path: {suite2p_path}")
        self.logger.info(f"GIP: imaging_session_root: {imaging_session_root}")
        
        return {
            'ops': ops,
            'suite2p_path': suite2p_path,
            'imaging_session_root': imaging_session_root,
            'loaded_at': datetime.now().isoformat()
        }
    
    def get_quality_control_params(self, experiment_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get quality control parameters for this experiment.
        
        Args:
            experiment_config: Optional experiment-specific configuration
            
        Returns:
            Dictionary of QC parameters
        """
        # Start with defaults
        qc_params = self.default_qc_params.copy()
        
        # Override with experiment-specific parameters if provided
        if experiment_config and 'imaging_preprocessing' in experiment_config:
            imaging_config = experiment_config['imaging_preprocessing']
            if 'quality_control' in imaging_config:
                qc_params.update(imaging_config['quality_control'])
        
        self.logger.info(f"GIP: Using QC parameters: {qc_params}")
        return qc_params
    
    def run_quality_control(self, ops: Dict[str, Any], qc_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run quality control filtering on ROIs/cells.
        
        Args:
            ops: Suite2p ops dictionary
            qc_params: Quality control parameters
            
        Returns:
            Results from quality control step
        """
        self.logger.info("GIP: Running quality control...")
        
        try:
            QualControlDataIO.run(
                ops,
                qc_params['range_skew'],
                qc_params['max_connect'],
                qc_params['range_aspect'],
                qc_params['range_compact'],
                qc_params['range_footprint']
            )
            
            self.logger.info("GIP: Quality control completed successfully")
            return {
                'status': 'success',
                'parameters_used': qc_params,
                'completed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"GIP: Quality control failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'parameters_used': qc_params,
                'failed_at': datetime.now().isoformat()
            }
    
    def run_cell_labeling(self, ops: Dict[str, Any], diameter: float = 6) -> Dict[str, Any]:
        """
        Run cell type labeling (excitatory/inhibitory classification).
        
        Args:
            ops: Suite2p ops dictionary
            diameter: Cellpose diameter parameter
            
        Returns:
            Results from cell labeling step
        """
        self.logger.info(f"GIP: Running cell labeling with diameter={diameter}...")
        
        try:
            LabelExcInh.run(ops, diameter)
            
            self.logger.info("GIP: Cell labeling completed successfully")
            return {
                'status': 'success',
                'diameter_used': diameter,
                'completed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"GIP: Cell labeling failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'diameter_used': diameter,
                'failed_at': datetime.now().isoformat()
            }
    
    def run_trace_extraction(self, ops: Dict[str, Any], correct_pmt: bool = False) -> Dict[str, Any]:
        """
        Run ΔF/F trace extraction.
        
        Args:
            ops: Suite2p ops dictionary
            correct_pmt: Whether to apply PMT correction
            
        Returns:
            Results from trace extraction step
        """
        self.logger.info(f"GIP: Running trace extraction with correct_pmt={correct_pmt}...")
        
        try:
            DffTraces.run(ops, correct_pmt=correct_pmt)
            
            self.logger.info("GIP: Trace extraction completed successfully")
            return {
                'status': 'success',
                'correct_pmt_used': correct_pmt,
                'completed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"GIP: Trace extraction failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'correct_pmt_used': correct_pmt,
                'failed_at': datetime.now().isoformat()
            }
    
    def process_imaging_session(self, subject_id: str, imaging_session: Dict[str, str], 
                               experiment_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a single imaging session through the complete pipeline.
        
        Args:
            subject_id: Subject identifier
            imaging_session: Imaging session configuration
            experiment_config: Optional experiment-specific configuration
            
        Returns:
            Processing results for this session
        """
        session_id = f"{subject_id}_{imaging_session.get('imaging_folder', 'unknown')}"
        self.logger.info(f"GIP: Processing imaging session {session_id}")
        
        results = {
            'subject_id': subject_id,
            'imaging_session': imaging_session,
            'processing_started': datetime.now().isoformat(),
            'steps_completed': [],
            'steps_failed': []
        }
        
        try:
            # Step 1: Load suite2p data
            suite2p_path = self.get_imaging_session_path(subject_id, imaging_session)
            suite2p_data = self.load_suite2p_data(suite2p_path)
            ops = suite2p_data['ops']
            results['suite2p_data_loaded'] = True
            results['steps_completed'].append('load_suite2p_data')
            
            # Step 2: Quality control
            qc_params = self.get_quality_control_params(experiment_config)
            qc_results = self.run_quality_control(ops, qc_params)
            results['quality_control'] = qc_results
            if qc_results['status'] == 'success':
                results['steps_completed'].append('quality_control')
            else:
                results['steps_failed'].append('quality_control')
            
            # Step 3: Cell labeling
            diameter = qc_params.get('diameter', 6)
            labeling_results = self.run_cell_labeling(ops, diameter)
            results['cell_labeling'] = labeling_results
            if labeling_results['status'] == 'success':
                results['steps_completed'].append('cell_labeling')
            else:
                results['steps_failed'].append('cell_labeling')
            
            # Step 4: Trace extraction
            correct_pmt = experiment_config.get('imaging_preprocessing', {}).get('correct_pmt', False) if experiment_config else False
            trace_results = self.run_trace_extraction(ops, correct_pmt)
            results['trace_extraction'] = trace_results
            if trace_results['status'] == 'success':
                results['steps_completed'].append('trace_extraction')
            else:
                results['steps_failed'].append('trace_extraction')
            
            # Overall status
            if results['steps_failed']:
                results['overall_status'] = 'partial_success'
                self.logger.warning(f"GIP: Session {session_id} completed with failures in: {results['steps_failed']}")
            else:
                results['overall_status'] = 'success'
                self.logger.info(f"GIP: Session {session_id} completed successfully")
            
        except Exception as e:
            results['overall_status'] = 'failed'
            results['error'] = str(e)
            self.logger.error(f"GIP: Session {session_id} failed: {e}")
        
        results['processing_completed'] = datetime.now().isoformat()
        return results
    
    def preprocess_subject_imaging(self, subject_id: str, experiment_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process all imaging sessions for a subject.
        
        Args:
            subject_id: Subject identifier
            experiment_config: Optional experiment-specific configuration
            
        Returns:
            Results for all sessions for this subject
        """
        self.logger.info(f"GIP: Starting imaging preprocessing for subject {subject_id}")
        
        subject_config = self.config.get('subjects', {}).get(subject_id, {})
        imaging_sessions = subject_config.get('imaging_sessions', [])
        
        if not imaging_sessions:
            self.logger.warning(f"GIP: No imaging sessions found for subject {subject_id}")
            return {
                'subject_id': subject_id,
                'status': 'no_imaging_sessions',
                'sessions_processed': 0,
                'sessions_successful': 0,
                'sessions_failed': 0
            }
        
        results = {
            'subject_id': subject_id,
            'sessions_processed': 0,
            'sessions_successful': 0,
            'sessions_failed': 0,
            'session_results': {},
            'started_at': datetime.now().isoformat()
        }
        
        for imaging_session in imaging_sessions:
            session_results = self.process_imaging_session(subject_id, imaging_session, experiment_config)
            
            session_key = imaging_session.get('imaging_folder', f'session_{results["sessions_processed"]}')
            results['session_results'][session_key] = session_results
            results['sessions_processed'] += 1
            
            if session_results['overall_status'] in ['success', 'partial_success']:
                results['sessions_successful'] += 1
            else:
                results['sessions_failed'] += 1
        
        results['completed_at'] = datetime.now().isoformat()
        results['status'] = 'completed'
        
        self.logger.info(f"GIP: Completed imaging preprocessing for subject {subject_id}: "
                        f"{results['sessions_successful']}/{results['sessions_processed']} sessions successful")
        
        return results
    
    def preprocess_imaging_data(self, experiment_config: Dict[str, Any] = None, force: bool = False) -> Dict[str, Any]:
        """
        Process imaging data for all subjects.
        
        Args:
            experiment_config: Optional experiment-specific configuration
            force: Whether to force reprocessing
            
        Returns:
            Overall preprocessing results
        """
        self.logger.info("GIP: Starting imaging preprocessing for all subjects")
        
        results = {
            'preprocessing_type': 'imaging',
            'subjects_processed': 0,
            'subjects_successful': 0,
            'subjects_failed': 0,
            'subject_results': {},
            'started_at': datetime.now().isoformat()
        }
        
        for subject_id in self.subject_list:
            self.logger.info(f"GIP: Processing subject {subject_id}")
            
            subject_results = self.preprocess_subject_imaging(subject_id, experiment_config)
            results['subject_results'][subject_id] = subject_results
            results['subjects_processed'] += 1
            
            if subject_results.get('status') == 'completed' and subject_results.get('sessions_successful', 0) > 0:
                results['subjects_successful'] += 1
            else:
                results['subjects_failed'] += 1
        
        results['completed_at'] = datetime.now().isoformat()
        results['status'] = 'completed'
        
        self.logger.info(f"GIP: Imaging preprocessing completed: "
                        f"{results['subjects_successful']}/{results['subjects_processed']} subjects successful")
        
        return results


# Placeholder for experiment-specific imaging preprocessors
class ExperimentSpecificImagingPreprocessor(GeneralImagingPreprocessor):
    """
    Placeholder base class for experiment-specific imaging preprocessors.
    
    Experiment-specific preprocessors should inherit from this class and override
    methods as needed to handle experiment-specific requirements.
    
    Example usage:
    
    class SingleIntervalDiscriminationImagingPreprocessor(ExperimentSpecificImagingPreprocessor):
        def get_quality_control_params(self, experiment_config=None):
            # Override with experiment-specific QC parameters
            params = super().get_quality_control_params(experiment_config)
            params['range_skew'] = [-5, 5]  # Different for this experiment
            return params
    """
    
    def __init__(self, config_manager, subject_list, logger=None):
        """Initialize experiment-specific imaging preprocessor."""
        super().__init__(config_manager, subject_list, logger)
        self.logger.info("Experiment-specific imaging preprocessor initialized")
    
    # Methods can be overridden by specific experiment preprocessors
    # Examples:
    # - get_quality_control_params(): Override QC parameters
    # - process_imaging_session(): Add experiment-specific processing steps
    # - preprocess_subject_imaging(): Add subject-specific logic
    # - process_imaging_session(): Add experiment-specific processing steps
    # - preprocess_subject_imaging(): Add subject-specific logic
