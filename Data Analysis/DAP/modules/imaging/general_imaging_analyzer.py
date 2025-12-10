"""
General Imaging Analyzer

Base class for imaging analysis that loads experiment-specific analyzers.
Follows the same pattern as other pipeline components.
"""

import os
import logging
import importlib
from typing import Dict, Any, Optional, List


class GeneralImagingAnalyzer:
    """
    General imaging analyzer that coordinates experiment-specific analysis workflows.
    """
    
    def __init__(self, config_manager, subject_list, logger=None):
        """
        Initialize the general imaging analyzer.
        
        Args:
            config_manager: Configuration manager instance
            subject_list: List of subject IDs to analyze
            logger: Logger instance (optional)
        """
        self.config_manager = config_manager
        self.subject_list = subject_list
        self.config = config_manager.config
        self.logger = logger or logging.getLogger(__name__)
        
        # Get imaging data path from config
        self.imaging_data_base = self.config.get('paths', {}).get('imaging_data_base', '')        
        
        # Load experiment-specific analyzer
        self.experiment_analyzer = self._load_experiment_analyzer()
        
        self.logger.info("IMG_ANALYZER: General imaging analyzer initialized")
    
    def _load_experiment_analyzer(self):
        """Load experiment-specific analyzer if configured."""
        try:
            # Get experiment config
            experiment_config = self.config.get('experiment_configs', {}).get(self.config_manager.experiment_name, {})
            analyzer_module_name = experiment_config.get('imaging_analyzer_module')
            
            if analyzer_module_name is None:
                self.logger.info("IMG_ANALYZER: No experiment-specific analyzer configured")
                return None
            
            # Get analyzer info
            available_analyzers = self.config.get('available_imaging_analyzers', {})
            analyzer_info = available_analyzers.get(analyzer_module_name)
            
            if not analyzer_info:
                self.logger.error(f"IMG_ANALYZER: Analyzer module '{analyzer_module_name}' not found")
                return None
            
            class_name = analyzer_info.get('class')
            if not class_name:
                self.logger.error(f"IMG_ANALYZER: No class specified for analyzer module '{analyzer_module_name}'")
                return None
            
            self.logger.info(f"IMG_ANALYZER: Loading experiment-specific analyzer: {class_name}")
            
            # Import and initialize
            module = importlib.import_module(f"modules.experiments.{self.config_manager.experiment_name}.{analyzer_module_name}")
            analyzer_class = getattr(module, class_name)
            
            experiment_analyzer = analyzer_class(self.config_manager, self.subject_list, self.logger)
            self.logger.info(f"IMG_ANALYZER: Successfully loaded: {class_name}")
            
            return experiment_analyzer
            
        except Exception as e:
            self.logger.error(f"IMG_ANALYZER: Failed to load experiment-specific analyzer: {e}")
            return None
    
    def _get_subject_imaging_paths(self, subject_id):
        """
        Construct imaging paths for a subject based on config.
        
        Args:
            subject_id: Subject identifier
            
        Returns:
            Dict with suite2p_path and output_path, or None if session mapping not found
        """
        try:
            # Get subject config
            subject_config = self.config.get('subjects', {}).get(subject_id, {})
            imaging_sessions = subject_config.get('imaging_sessions', [])
            
            if not imaging_sessions:
                self.logger.warning(f"IMG_ANALYZER: No imaging sessions found for subject {subject_id}")
                return None
            
            # For now, use the first imaging session (could be enhanced to handle multiple)
            imaging_session = imaging_sessions[0]
            imaging_folder = imaging_session.get('imaging_folder', '')
            experiment_dir = imaging_session.get('experiment_dir', '')
            
            if not imaging_folder:
                self.logger.warning(f"IMG_ANALYZER: No imaging folder specified for subject {subject_id}")
                return None
            
            if not experiment_dir:
                self.logger.warning(f"IMG_ANALYZER: No experiment directory specified for subject {subject_id}")
                return None
            
            # Construct paths using experiment_dir from session mapping
            base_path = os.path.join(self.imaging_data_base, experiment_dir, subject_id, imaging_folder)
            
            suite2p_path = os.path.join(base_path, 'suite2p', 'plane0')
            output_path = os.path.join(base_path)
            
            self.logger.info(f"IMG_ANALYZER: Constructed paths for {subject_id}: suite2p={suite2p_path}")
            
            return {
                'suite2p_path': suite2p_path,
                'output_path': output_path,
                'imaging_folder': imaging_folder,
                'experiment_dir': experiment_dir
            }
            
        except Exception as e:
            self.logger.error(f"IMG_ANALYZER: Failed to construct imaging paths for {subject_id}: {e}")
            return None

    def analyze_subject(self, subject_id: str, suite2p_path: str, output_path: str, force: bool = False) -> Dict[str, Any]:
        """
        Analyze imaging sessions for a single subject.
        
        Args:
            subject_id: Subject identifier
            force: Force reanalysis even if results exist
            
        Returns:
            Analysis results dictionary
        """
        try:
            self.logger.info(f"SID_IMG_ANALYZER: Analyzing imaging sessions for subject {subject_id}")
                        
        
            # Load preprocessed data
            loaded_data = self._load_preprocessed_data(subject_id, suite2p_path, output_path)

            if not loaded_data['success']:
                return loaded_data  # Return the error from loading
            
            # For now, just return the loaded data
            return {
                'success': True,
                'subject_id': subject_id,
                'loaded_data': loaded_data
            }
            
        except Exception as e:
            self.logger.error(f"SID_IMG_ANALYZER: Failed to analyze subject {subject_id}: {e}")
            return {'success': False, 'error': str(e)}

    
    def analyze_imaging_sessions(self, force: bool = False) -> Dict[str, Any]:
        """
        Analyze imaging sessions for all subjects using experiment-specific analyzer.
        
        Args:
            force: Force reanalysis even if results exist
            
        Returns:
            Analysis results dictionary
        """
        self.logger.info(f"IMG_ANALYZER: Starting imaging session analysis for {len(self.subject_list)} subjects...")
        
        
        
        if self.experiment_analyzer is not None:
            results = {'success': True, 'subjects_processed': 0, 'subjects_failed': 0, 'results': {}}
            
            for subject_id in self.subject_list:
                try:
                    self.logger.info(f"IMG_ANALYZER: Analyzing imaging sessions for subject {subject_id}")
                    
                    # Get imaging paths for this subject
                    imaging_paths = self._get_subject_imaging_paths(subject_id)
                    if imaging_paths is None:
                        raise ValueError(f"Could not construct imaging paths for subject {subject_id}")

                    subject_results = self.experiment_analyzer.analyze_subject(subject_id, imaging_paths['suite2p_path'], imaging_paths['output_path'], force)
                    results['results'][subject_id] = subject_results
                    
                    if subject_results.get('success', False):
                        results['subjects_processed'] += 1
                    else:
                        results['subjects_failed'] += 1
                        
                except Exception as e:
                    self.logger.error(f"IMG_ANALYZER: Failed to analyze subject {subject_id}: {e}")
                    results['results'][subject_id] = {'success': False, 'error': str(e)}
                    results['subjects_failed'] += 1
            
            return results
        else:
            self.logger.error("IMG_ANALYZER: No experiment-specific analyzer available")
            return {'success': False, 'error': 'No experiment-specific analyzer available'}    