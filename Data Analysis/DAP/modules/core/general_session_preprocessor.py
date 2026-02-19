import os
import pickle
from tqdm import tqdm
import importlib
import logging

import modules.utils as utils

class GeneralSessionPreprocessor:
    """
    General session preprocessor that handles common preprocessing tasks.
    Works with ConfigManager for configuration and validation.
    """
    
    def __init__(self, config_manager, subject_list, logger, force=False):
        """
        Initialize the GeneralSessionPreprocessor with ConfigManager and subject list.
        ConfigManager has already validated everything, so we just set up for preprocessing.
        """
        self.config_manager = config_manager
        self.subject_list = subject_list
        self.force = force
        self.logger = logger
        
        self.logger.info("SP: Initializing GeneralSessionPreprocessor...")
        
        # Get paths from ConfigManager's config
        self.config = config_manager.config
        paths = self.config.get('paths', {})
        self.extracted_root_dir = paths.get('extracted_data', '')
        self.preprocessed_root_dir = paths.get('preprocessed_data', '')
        
        # Initialize experiment-specific preprocessor if specified
        self.experiment_preprocessor = self._load_experiment_preprocessor()
        
        self.logger.info("SP: GeneralSessionPreprocessor initialized successfully")

    def _load_experiment_preprocessor(self):
        """Load experiment-specific preprocessor if configured."""
        try:
            # Get experiment config
            experiment_config = self.config.get('experiment_configs', {}).get(self.config_manager.experiment_name, {})
            preprocessor_module_name = experiment_config.get('preprocessor_module')
            
            if preprocessor_module_name is None:
                self.logger.info("SP: No experiment-specific preprocessor configured, using general preprocessing only")
                return None
            
            # Get preprocessor info from available list
            available_preprocessors = self.config.get('available_preprocessors', {})
            preprocessor_info = available_preprocessors.get(preprocessor_module_name)
            
            if not preprocessor_info:
                self.logger.error(f"SP: Preprocessor module '{preprocessor_module_name}' not found in available_preprocessors")
                return None
            
            class_name = preprocessor_info.get('class')
            if not class_name:
                self.logger.error(f"SP: No class specified for preprocessor module '{preprocessor_module_name}'")
                return None
            
            self.logger.info(f"SP: Loading experiment-specific preprocessor: {class_name} from {preprocessor_module_name}")
            
            # Direct import of the specific module and class
            module = importlib.import_module(f"modules.experiments.{self.config_manager.experiment_name}.{preprocessor_module_name}")
            preprocessor_class = getattr(module, class_name)
            
            # Initialize the experiment-specific preprocessor
            experiment_preprocessor = preprocessor_class(self.config_manager, logger=self.logger)
            self.logger.info(f"SP: Successfully loaded experiment-specific preprocessor: {class_name}")
            
            return experiment_preprocessor
            
        except Exception as e:
            self.logger.error(f"SP: Failed to load experiment-specific preprocessor: {e}")
            self.logger.info("SP: Falling back to general preprocessing only")
            return None

    def load_extracted_session(self, session_path):
        """
        Load extracted session data from pickle file.
        Returns the session data dictionary or None if loading fails.
        """
        try:
            with open(session_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            self.logger.error(f"SP: Failed to load extracted session: {session_path} — {e}")
            return None

    def save_preprocessed_session(self, preprocessed_data, save_path):
        """
        Save preprocessed session data to a pickle file.
        Logs errors if saving fails.
        """
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(preprocessed_data, f)
        except Exception as e:
            self.logger.error(f"SP: Failed to save preprocessed session: {save_path} — {e}")

    def session_already_preprocessed(self, preprocessed_dir, session_id):
        """
        Check if the session has already been preprocessed and saved.
        Returns True if the pickle file exists.
        """
        session_file = os.path.join(preprocessed_dir, f"{session_id}_preprocessed.pkl")
        return os.path.isfile(session_file)

    def preprocess_session_data(self, session_data, session_id):
        """
        Apply preprocessing to session data.
        Uses experiment-specific preprocessor if available, otherwise general processing.
        """
        if self.experiment_preprocessor is not None:
            # Use experiment-specific preprocessing
            return self.experiment_preprocessor.preprocess_session_data(session_data, session_id)
        else:
            # Use general preprocessing only
            self.logger.info("SP: Applying general preprocessing...")
            preprocessed_data = session_data.copy()
            preprocessed_data['general_preprocessing_applied'] = True
            return preprocessed_data

    def preprocess_and_store_subject(self, subject_id, force=False):
        """
        Preprocess sessions for a subject based on sessions_to_process list.

        - Only processes sessions listed in subject config
        - Skips already preprocessed sessions unless 'force' is True.
        - Logs errors for failed preprocessing.

        Args:
            subject_id (str): The subject identifier.
            force (bool): If True, re-preprocess even if already preprocessed.
        """
        # Get subject config from ConfigManager
        subject_config = self.config['subjects'][subject_id]
        sessions_to_process = subject_config.get('sessions_to_process', [])
        
        if not sessions_to_process:
            self.logger.warning(f"SP: No sessions_to_process found for subject {subject_id}")
            return

        extracted_dir = os.path.join(self.extracted_root_dir, subject_id)
        preprocessed_dir = os.path.join(self.preprocessed_root_dir, subject_id)
        
        # ConfigManager has already created these directories
        
        # Check if extracted directory exists
        if not os.path.isdir(extracted_dir):
            self.logger.error(f"SP: Extracted directory does not exist: {extracted_dir}")
            return

        for session_name in tqdm(sessions_to_process, desc=f"Preprocessing {subject_id}"):
            session_id = session_name  # Use full session name as ID
            
            # Skip if already preprocessed and not forcing re-preprocessing
            if not force and self.session_already_preprocessed(preprocessed_dir, session_id):
                continue

            # Look for extracted session file
            extracted_session_path = os.path.join(extracted_dir, f"{session_id}_extracted.pkl")
            
            if not os.path.isfile(extracted_session_path):
                self.logger.warning(f"SP: Extracted session not found: {extracted_session_path}")
                self.logger.warning(f"SP: This could indicate:")
                self.logger.warning(f"SP:   1. Session '{session_name}' was not found in the original session folder during extraction")
                self.logger.warning(f"SP:   2. Session extraction was skipped or failed for this session")
                self.logger.warning(f"SP:   3. Session name in config doesn't match actual .mat file name")
                self.logger.warning(f"SP: Check: session folder contents, extraction logs, and config sessions_to_process list")
                continue

            # Load extracted session data
            session_data = self.load_extracted_session(extracted_session_path)
            if session_data is None:
                continue

            # Apply preprocessing
            preprocessed_data = self.preprocess_session_data(session_data, session_id)
            
            # Save preprocessed data
            save_path = os.path.join(preprocessed_dir, f"{session_id}_preprocessed.pkl")
            self.save_preprocessed_session(preprocessed_data, save_path)

    def batch_preprocess_sessions(self, force=None):
        """
        Batch preprocess session data for the subjects in self.subject_list.
        Uses the instance's force flag if not provided.
        """
        if force is None:
            force = self.force
            
        self.logger.info(f"SP: Starting batch preprocessing for {len(self.subject_list)} subjects")
        
        for subject_id in self.subject_list:
            self.logger.info(f"SP: Processing subject {subject_id}")
            self.preprocess_and_store_subject(subject_id, force=force)
            
        self.logger.info("SP: Batch preprocessing completed")