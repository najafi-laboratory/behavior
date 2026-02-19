"""
General Imaging Suite2p Preprocessor

Minimal preprocessing component for Suite2p output processing.
Follows the same pattern as other pipeline components (session_extractor, session_preprocessor, etc.).

This handles the generic Suite2p processing pipeline:
1. Load Suite2p data (ops.npy, F.npy, iscell.npy, etc.)
2. Quality control filtering
3. Cell type labeling
4. dF/F trace extraction
"""

import os
import sys
import logging
from datetime import datetime
import importlib
from typing import Dict, Any, List

# Import 2p processing modules
from modules.imaging.suite2p_qc import Suite2pQC
from modules.imaging.suite2p_labeling import Suite2pLabeling
from modules.imaging.suite2p_dff_traces import Suite2pDffTraces
from modules.utils.utils import clean_memmap_path


class GeneralImagingSuite2pPreprocessor:
    """
    Generic Suite2p preprocessor following pipeline component pattern.
    
    Minimal implementation that handles Suite2p → processed data pipeline.
    Similar structure to SessionExtractor, SessionPreprocessor, etc.
    """
    
    def __init__(self, config_manager, subject_list, logger=None):
        """
        Initialize the Suite2p preprocessor.
        
        Args:
            config_manager: ConfigManager instance
            subject_list: List of subject IDs to process
            logger: Logger instance
        """
        self.config_manager = config_manager
        self.config = config_manager.config
        self.subject_list = subject_list
        self.logger = logger or logging.getLogger(__name__)
        
        # Get imaging data path from config
        self.imaging_data_base = self.config.get('paths', {}).get('imaging_data_base', '')
        
        self.logger.info("IMG_PRE: GeneralImagingSuite2pPreprocessor initialized")
        
        # Initialize 2p processing modules
        self._init_2p_modules()
        
        # Initialize experiment-specific imaging preprocessor if specified
        self.experiment_imaging_preprocessor = self._load_experiment_imaging_preprocessor()
    
    def _init_2p_modules(self):
        """Initialize 2p processing modules."""
        try:
            # Initialize QC module
            self.qc_processor = Suite2pQC(
                config_manager=self.config_manager,
                subject_list=self.subject_list,
                logger=self.logger
            )
            
            # Initialize Labeling module
            self.labeling_processor = Suite2pLabeling(
                config_manager=self.config_manager,
                subject_list=self.subject_list,
                logger=self.logger
            )
            
            # Initialize DFF trace extraction
            self.dff_processor = Suite2pDffTraces(
                config_manager=self.config_manager,
                subject_list=self.subject_list,
                logger=self.logger
            )
            
            self.logger.info("IMG_PRE: Suite2p QC and Labeling modules initialized successfully")
            
        except Exception as e:
            self.logger.error(f"IMG_PRE: Failed to initialize 2p modules: {e}")
            self.qc_processor = None
            self.labeling_processor = None
    
    def _load_experiment_imaging_preprocessor(self):
        """Load experiment-specific imaging preprocessor if configured."""
        try:
            # Get experiment config
            experiment_config = self.config.get('experiment_configs', {}).get(self.config_manager.experiment_name, {})
            
            
            # Look for experiment-specific imaging preprocessor module
            imaging_preprocessor_module = experiment_config.get('imaging_preprocessor_module')
            
            if imaging_preprocessor_module is None:
                self.logger.info("IMG_PRE: No experiment-specific imaging preprocessor configured, using general processing only")
                return None
            
            # Get imaging preprocessor info from available list
            available_imaging_preprocessors = self.config.get('available_imaging_preprocessors', {})
            preprocessor_info = available_imaging_preprocessors.get(imaging_preprocessor_module)
            
            if not preprocessor_info:
                self.logger.error(f"IMG_PRE: Imaging preprocessor module '{imaging_preprocessor_module}' not found in available_imaging_preprocessors")
                return None
            
            class_name = preprocessor_info.get('class')
            if not class_name:
                self.logger.error(f"IMG_PRE: No class specified for imaging preprocessor module '{imaging_preprocessor_module}'")
                return None
            
            self.logger.info(f"IMG_PRE: Loading experiment-specific imaging preprocessor: {class_name} from {imaging_preprocessor_module}")
            
            # Direct import of the specific module and class
            module = importlib.import_module(f"modules.experiments.{self.config_manager.experiment_name}.{imaging_preprocessor_module}")
            preprocessor_class = getattr(module, class_name)
            
            # Initialize the experiment-specific imaging preprocessor
            experiment_imaging_preprocessor = preprocessor_class(self.config_manager, self.subject_list, self.logger)
            self.logger.info(f"IMG_PRE: Successfully loaded experiment-specific imaging preprocessor: {class_name}")
            
            return experiment_imaging_preprocessor
            
        except Exception as e:
            self.logger.error(f"IMG_PRE: Failed to load experiment-specific imaging preprocessor: {e}")
            self.logger.info("IMG_PRE: Falling back to general imaging processing only")
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
                self.logger.warning(f"IMG_PRE: No imaging sessions found for subject {subject_id}")
                return None
            
            # For now, use the first imaging session (could be enhanced to handle multiple)
            imaging_session = imaging_sessions[0]
            imaging_folder = imaging_session.get('imaging_folder', '')
            experiment_dir = imaging_session.get('experiment_dir', '')
            
            if not imaging_folder:
                self.logger.warning(f"IMG_PRE: No imaging folder specified for subject {subject_id}")
                return None
            
            if not experiment_dir:
                self.logger.warning(f"IMG_PRE: No experiment directory specified for subject {subject_id}")
                return None
            
            # Construct paths using experiment_dir from session mapping
            base_path = os.path.join(self.imaging_data_base, experiment_dir, subject_id, imaging_folder)
            
            suite2p_path = os.path.join(base_path, 'suite2p', 'plane0')
            output_path = os.path.join(base_path)
            
            self.logger.info(f"IMG_PRE: Constructed paths for {subject_id}: suite2p={suite2p_path}")
            
            return {
                'suite2p_path': suite2p_path,
                'output_path': output_path,
                'imaging_folder': imaging_folder,
                'experiment_dir': experiment_dir
            }
            
        except Exception as e:
            self.logger.error(f"IMG_PRE: Failed to construct imaging paths for {subject_id}: {e}")
            return None
    
    def preprocess_imaging_sessions(self, force=False):
        """
        Run the Suite2p preprocessing pipeline for all subjects.
        
        Args:
            force: If True, reprocess even if output already exists
            
        Returns:
            Dict with processing results and metadata
        """
        self.logger.info("IMG_PRE: Starting Suite2p preprocessing pipeline")
        
        results = {
            'subjects_processed': [],
            'subjects_failed': [],
            'total_sessions_processed': 0,
            'processing_metadata': {}
        }
        
        for subject_id in self.subject_list:
            try:
                self.logger.info(f"IMG_PRE: Processing subject {subject_id}")
                
                # Process this subject's imaging sessions
                subject_results = self._process_subject_imaging(subject_id, force=force)
                
                if subject_results['success']:
                    results['subjects_processed'].append(subject_id)
                    results['total_sessions_processed'] += subject_results.get('sessions_processed', 0)
                    results['processing_metadata'][subject_id] = subject_results
                    self.logger.info(f"IMG_PRE: Successfully processed subject {subject_id}")
                else:
                    results['subjects_failed'].append(subject_id)
                    self.logger.warning(f"IMG_PRE: Failed to process subject {subject_id}")
                
            except Exception as e:
                self.logger.error(f"IMG_PRE: Error processing subject {subject_id}: {e}")
                results['subjects_failed'].append(subject_id)
        
        # Log summary
        total_subjects = len(self.subject_list)
        processed_subjects = len(results['subjects_processed'])
        self.logger.info(f"IMG_PRE: Pipeline completed. Processed {processed_subjects}/{total_subjects} subjects")
        
        return results
    
    def _process_subject_imaging(self, subject_id, force=False):
        """
        Process imaging data for a single subject.
        
        Args:
            subject_id: Subject identifier
            force: If True, reprocess even if output exists
            
        Returns:
            Dict with processing results for this subject
        """
        self.logger.info(f"IMG_PRE: Processing imaging data for subject {subject_id}")
        
        results = {
            'success': False,
            'sessions_processed': 0,
            'error_message': None,
            'qc_results': None,
            'labeling_results': None,
            'dff_results': None,
            'experiment_imaging_results': None
        }
        
        try:
            # Check if we have processors available
            if self.qc_processor is None:
                raise ValueError("QC processor not initialized")
            if self.labeling_processor is None:
                raise ValueError("Labeling processor not initialized")
            if self.dff_processor is None:
                raise ValueError("DFF processor not initialized")
            
            # Get imaging paths for this subject
            imaging_paths = self._get_subject_imaging_paths(subject_id)
            if imaging_paths is None:
                raise ValueError(f"Could not construct imaging paths for subject {subject_id}")
            
        
            # Step 1: Run QC processing
            self.logger.info(f"IMG_PRE: Running QC processing for {subject_id}")
            qc_results = self.qc_processor.process_subject(
                subject_id, 
                imaging_paths['suite2p_path'],
                imaging_paths['output_path'],
                force=force
            )
            results['qc_results'] = qc_results
            
            if not qc_results.get('success', False):
                results['error_message'] = f"QC processing failed: {qc_results.get('error_message', 'Unknown QC error')}"
                self.logger.error(f"IMG_PRE: QC processing failed for {subject_id}")
                return results
            
            self.logger.info(f"IMG_PRE: QC processing completed for {subject_id}")
            
            # Step 2: Run labeling processing
            self.logger.info(f"IMG_PRE: Running labeling processing for {subject_id}")
            labeling_results = self.labeling_processor.process_subject(
                subject_id,
                imaging_paths['suite2p_path'],
                imaging_paths['output_path'],
                force=force
            )
            results['labeling_results'] = labeling_results
            
            if not labeling_results.get('success', False):
                results['error_message'] = f"Labeling processing failed: {labeling_results.get('error_message', 'Unknown labeling error')}"
                self.logger.error(f"IMG_PRE: Labeling processing failed for {subject_id}")
                return results
            
            self.logger.info(f"IMG_PRE: Labeling processing completed for {subject_id}")
            
            #TODO: Add more in-depth mask-processing
            
            # Step 3: Run DFF trace extraction
            self.logger.info(f"IMG_PRE: Running DFF processing for {subject_id}")
            dff_results = self.dff_processor.process_subject(
                subject_id,
                imaging_paths['suite2p_path'],
                imaging_paths['output_path'],
                force=force
            )
            results['dff_results'] = dff_results
            
            if not dff_results.get('success', False):
                results['error_message'] = f"DFF processing failed: {dff_results.get('error_message', 'Unknown DFF error')}"
                self.logger.error(f"IMG_PRE: DFF processing failed for subject {subject_id}")
                return results
            
            self.logger.info(f"IMG_PRE: DFF processing completed for {subject_id}")
            
            # Step 4: Run experiment-specific imaging processing if available
            if self.experiment_imaging_preprocessor is not None:
                self.logger.info(f"IMG_PRE: Running experiment-specific imaging processing for {subject_id}")
                experiment_imaging_results = self.experiment_imaging_preprocessor.process_subject(
                    subject_id,
                    imaging_paths['suite2p_path'],
                    imaging_paths['output_path'],
                    force=force
                )
                results['experiment_imaging_results'] = experiment_imaging_results
                
                self.experiment_imaging_preprocessor.clean_output(subject_id, imaging_paths['output_path'])               
                
                if not experiment_imaging_results.get('success', False):
                    results['error_message'] = f"Experiment-specific imaging processing failed: {experiment_imaging_results.get('error_message', 'Unknown experiment imaging error')}"
                    self.logger.error(f"IMG_PRE: Experiment-specific imaging processing failed for {subject_id}")
                    return results
                                                
                self.logger.info(f"IMG_PRE: Experiment-specific imaging processing completed for {subject_id}")
            else:
                self.logger.info(f"IMG_PRE: No experiment-specific imaging processing configured for {subject_id}")
            
            # All processing steps succeeded
            results['success'] = True
            results['sessions_processed'] = qc_results.get('sessions_processed', 0)
            results['imaging_paths'] = imaging_paths
            
            self.logger.info(f"IMG_PRE: Successfully completed all processing steps for {subject_id}")
            
        except Exception as e:
            results['error_message'] = str(e)
            self.logger.error(f"IMG_PRE: Error in _process_subject_imaging for {subject_id}: {e}")
        
        return results
