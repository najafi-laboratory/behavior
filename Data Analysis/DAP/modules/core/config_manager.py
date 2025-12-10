import logging
from typing import Dict, Any, List
import yaml
from pathlib import Path
import os

import modules.utils.utils as utils

class ConfigManager:
    """Manages configuration loading and access."""
    
    def __init__(self, config_path: str = None, experiment_name: str = None, subject_selection: str = None):
        """Initialize with basic config and validate specified experiment/subjects."""
        # Use the shared root logger that was configured with run_id
        self.logger = logging.getLogger()
        self.config_path = config_path
        self.experiment_name = experiment_name
        self.subject_selection = subject_selection
        
        # Store experiment config for easy access by modules
        self.experiment_config = {}
        
        self.logger.info(f"CM: Initializing ConfigManager...")
        
        # For now, just use a minimal default config
        self.config = {
            'subjects': {},
            'experiment_configs': {}
        }
        
        # Store final results - only what we need
        self.subject_list = []  # Final validated subject list
        self.initialization_successful = False
        
        # Load YAML config if path provided
        if config_path:
            if self.load_config(config_path):
                self.logger.info("CM: Config loaded successfully")
            else:
                self.logger.error("CM: Failed to load config file")
                return
        else:
            self.logger.info("CM: Using default config (no config path provided)")
        
        # Parse subject selection once and validate
        parsed_subjects = None
        if subject_selection is not None:
            parsed_subjects = self.parse_subject_selection(subject_selection)
            if not parsed_subjects:
                self.logger.error(f"CM: Failed to parse subject selection: '{subject_selection}'")
                return
        
        # Run validation and store final subject list
        self.initialization_successful = self.validate_experiment_config(experiment_name, parsed_subjects)
        
        # Load experiment config if initialization was successful
        if self.initialization_successful and experiment_name:
            self.experiment_config = self.config.get('experiment_configs', {}).get(experiment_name, {})
            self.logger.info(f"CM: Loaded experiment config for '{experiment_name}'")
        
        if self.initialization_successful:
            self.logger.info(f"CM: ConfigManager initialized successfully")
        else:
            self.logger.error(f"CM: ConfigManager initialization failed")
        
    def load_config(self, config_path: str) -> bool:
        """Load configuration from YAML file."""
        try:
            config_file = Path(config_path)
            
            if not config_file.exists():
                self.logger.warning(f"CM: Config file not found: {config_path}, using default config")
                return False
            
            with open(config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            if loaded_config:
                self.config.update(loaded_config)
                self.logger.info(f"CM: Loaded config from: {config_path}")
                return True
            else:
                self.logger.warning(f"CM: Empty config file: {config_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"CM: Error loading config: {e}")
            return False        
        
    def validate_experiment_config(self, experiment_name: str, subject_list: List[str] = None) -> bool:
        """
        Validate that experiment configuration exists and is complete for specified subjects.
        Only validates the subjects we're actually going to process.
        
        Args:
            experiment_name: Name of experiment to validate (required)
            subject_list: List of subjects to validate (if None, validates all in experiment)
        """
        # Currently require experiment to be specified
        if experiment_name is None:
            self.logger.error("CM: No experiment specified - experiment_name is required")
            return False
            
        try:
            self.logger.info(f"CM: Validating experiment config '{experiment_name}'...")
            
            # Check if experiment exists
            if not self._experiment_exists(experiment_name):
                return False
            
            # Get experiment config
            experiment_config = self.config['experiment_configs'][experiment_name]
            
            # Check if experiment has subject configuration
            if not self._experiment_has_subjects(experiment_config, experiment_name):
                return False
            
            # Get subjects to validate
            if subject_list is None:
                # If no subject list provided, validate all subjects in experiment
                subjects_to_validate = self._get_experiment_subjects(experiment_config)
            else:
                # Only validate the subjects we're actually processing
                subjects_to_validate = subject_list
                self.logger.info(f"CM: Validating only selected subjects: {subjects_to_validate}")
            
            # Check that selected subjects are actually in the experiment
            experiment_subjects = self._get_experiment_subjects(experiment_config)
            for subject_id in subjects_to_validate:
                if subject_id not in experiment_subjects:
                    error_msg = (
                        f"Subject '{subject_id}' is not configured for experiment '{experiment_name}'. "
                        f"Available subjects: {experiment_subjects}"
                    )
                    self.logger.error(error_msg)
                    return False
            
            # Validate each selected subject exists and has required fields
            for subject_id in subjects_to_validate:
                if not self._validate_subject(subject_id):
                    return False
                
                # Check session folder exists
                if not self._validate_subject_session_folder(subject_id):
                    return False
            
            # Set up and validate config paths
            self._setup_paths()
            
            # Store the final validated subject list
            self.subject_list = subjects_to_validate
            
            # Set up subject-specific directories after successful validation
            self._setup_subject_directories()            
            
            self.logger.info(f"CM: Experiment config '{experiment_name}' validation successful for {len(subjects_to_validate)} subjects!")
            return True
            
        except Exception as e:
            self.logger.error(f"CM: Exception during validation: {str(e)}")
            return False
    
    def is_initialized_successfully(self) -> bool:
        """Check if ConfigManager was initialized successfully."""
        return self.initialization_successful
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """
        Get the loaded experiment configuration.
        
        Returns:
            Dictionary containing the experiment configuration
        """
        return self.experiment_config
    
    def get_validated_subjects(self) -> List[str]:
        """Get the subjects that were validated during initialization."""
        return self.subject_list
        
    def _experiment_exists(self, experiment_name: str) -> bool:
        """Check if experiment exists in config."""
        if experiment_name not in self.config.get('experiment_configs', {}):
            self.logger.error(f"CM: Experiment '{experiment_name}' not found in experiment_configs")
            return False
        
        self.logger.info(f"CM: Experiment '{experiment_name}' found in config")
        return True

    def _experiment_has_subjects(self, experiment_config: Dict[str, Any], experiment_name: str) -> bool:
        """Check if experiment has subject configuration."""
        # Check for either 'subject_configs' or 'list_config' (support both formats)
        if 'subject_configs' not in experiment_config and 'list_config' not in experiment_config:
            self.logger.error(
                f"CM: Neither 'subject_configs' nor 'list_config' found in experiment '{experiment_name}'. "
                "Please check your experiment configuration."
            )
            return False
        
        self.logger.info(f"CM: Subject configuration found for experiment '{experiment_name}'")
        return True

    def _get_experiment_subjects(self, experiment_config: Dict[str, Any]) -> List[str]:
        """Get list of subjects from experiment config."""
        # Get 'subject_configs' if present
        if 'subject_configs' in experiment_config:
            return list(experiment_config['subject_configs'].keys())
        else:
            return []

    def _validate_subject(self, subject_id: str) -> bool:
        """Validate that subject exists and has required fields."""
        # Check if subject exists in subjects section
        if subject_id not in self.config.get('subjects', {}):
            self.logger.error(
                f"CM: Subject '{subject_id}' not found in subjects section. "
                "Please check your subject configuration."
            )
            return False
        
        self.logger.info(f"CM: Found config for subject_id '{subject_id}'")
        
        # Get subject config
        subject_config = self.config['subjects'][subject_id]
        
        # Check for required keys
        required_keys = ["subject_name", "session_folder"]
        for key in required_keys:
            if not subject_config.get(key):
                self.logger.error(
                    f"CM: '{key}' missing for subject_id '{subject_id}'. "
                    f"Please check your subject config and ensure '{key}' is set."
                )
                return False
            
            self.logger.info(f"CM: '{key}' for subject_id '{subject_id}' is '{subject_config.get(key)}'")
        
        return True

    def _validate_subject_session_folder(self, subject_id: str) -> bool:
        """Check if the session folder exists for the subject."""
        subject_config = self.config['subjects'][subject_id]
        session_folder = subject_config["session_folder"]
        
        # Get session data path from config
        session_data_path = self.config.get('paths', {}).get('session_data')
        
        if not session_data_path:
            self.logger.error("CM: 'session_data' path not found in config paths section")
            return False
        
        # Build full session folder path
        session_folder_path = os.path.join(session_data_path, session_folder)
        
        if not os.path.isdir(session_folder_path):
            self.logger.error(
                f"CM: Session folder does not exist for subject_id '{subject_id}': {session_folder_path}. "
                "Please check your directory structure."
            )
            return False
        
        self.logger.info(f"CM: Session folder exists for subject_id '{subject_id}': {session_folder_path}")
        return True

    def parse_subject_selection(self, selection: str) -> List[str]:
        """Parse subject selection - for now just split by comma."""
        if not selection or not selection.strip():
            self.logger.error("CM: Empty subject selection provided")
            return []
            
        subjects = [s.strip() for s in selection.split(',')]
        self.logger.info(f"CM: Parsed subject selection: {subjects}")
        return subjects

    def _setup_paths(self):
        """Sanitize and create all paths defined in the config."""
        paths = self.config.get('paths', {})
        
        if not paths:
            self.logger.warning("CM: No paths section found in config")
            return
        
        self.logger.info("CM: Setting up and validating paths...")
        
        # Sanitize and create directories for all paths
        sanitized_paths = {}
        for path_name, path_value in paths.items():
            if path_value:
                try:
                    # Sanitize and create directory in one step
                    sanitized_path = utils.sanitize_and_create_dir(path_value)
                    
                    sanitized_paths[path_name] = sanitized_path
                    self.logger.info(f"CM: Path '{path_name}' set up: {sanitized_path}")
                    
                except Exception as e:
                    self.logger.error(f"CM: Failed to set up path '{path_name}': {path_value} - {e}")
                    # Keep original path as fallback
                    sanitized_paths[path_name] = path_value
            else:
                self.logger.warning(f"CM: Empty path value for '{path_name}'")
                sanitized_paths[path_name] = path_value
        
        # Update config with sanitized paths
        self.config['paths'] = sanitized_paths
        self.logger.info("CM: Path setup completed")
        
    def _setup_subject_directories(self):
        """Create subject-specific directories for extracted and preprocessed data."""
        self.logger.info("CM: Setting up subject-specific directories...")
        
        paths = self.config.get('paths', {})
        extracted_root = paths.get('extracted_data', '')
        preprocessed_root = paths.get('preprocessed_data', '')
        
        for subject_id in self.subject_list:
            try:
                # Create extracted data directory for subject
                if extracted_root:
                    extracted_subject_dir = os.path.join(extracted_root, subject_id)
                    utils.sanitize_and_create_dir(extracted_subject_dir)
                    
                # Create preprocessed data directory for subject
                if preprocessed_root:
                    preprocessed_subject_dir = os.path.join(preprocessed_root, subject_id)
                    utils.sanitize_and_create_dir(preprocessed_subject_dir)
                    
                self.logger.info(f"CM: Created extracted and preprocessed directories for subject {subject_id}")
                
            except Exception as e:
                self.logger.error(f"CM: Failed to create extracted and preprocessed directories for subject {subject_id}: {e}")
        
        self.logger.info("CM: Subject directory setup completed")