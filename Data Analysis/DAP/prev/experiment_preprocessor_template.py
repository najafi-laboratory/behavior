import logging

class ExperimentPreprocessorTemplate:
    """
    Template for experiment-specific preprocessors.
    Copy this file and modify for your specific experiment type.
    
    Steps to create a new experiment preprocessor:
    1. Copy this file and rename to match your experiment
    2. Rename the class to match your experiment (e.g., TwoAlternativeForcedChoicePreprocessor)
    3. Update the logging prefix in __init__ and preprocess_session_data
    4. Implement experiment-specific preprocessing logic in preprocess_session_data
    5. Add your preprocessor to config.yaml under available_preprocessors
    6. Set preprocessor_class in your experiment config
    """
    
    def __init__(self, config_manager):
        """Initialize with ConfigManager for accessing experiment config."""
        self.config_manager = config_manager
        self.logger = logging.getLogger()
        
        # TODO: Update logging prefix to match your experiment (e.g., "2AFC-SP:")
        self.logger.info("TEMPLATE-SP: Initializing ExperimentPreprocessorTemplate...")
        
        # Get experiment-specific preprocessing config
        experiment_config = config_manager.config.get('experiment_configs', {}).get(config_manager.experiment_name, {})
        self.preprocessing_config = experiment_config.get('preprocessing', {})
        
        # TODO: Update logging prefix to match your experiment
        self.logger.info("TEMPLATE-SP: ExperimentPreprocessorTemplate initialized successfully")
    
    def preprocess_session_data(self, session_data):
        """
        Apply experiment-specific preprocessing to session data.
        
        Args:
            session_data: Raw session data from SessionExtractor
            
        Returns:
            Processed session data with experiment-specific transformations
        """
        # TODO: Update logging prefix and message to match your experiment
        self.logger.info("TEMPLATE-SP: Applying template preprocessing...")
        
        # Start with copy of original data
        processed_data = session_data.copy()
        
        # TODO: Replace with your experiment-specific preprocessing steps
        processed_data['experiment_type'] = 'template_experiment'  # Update this
        processed_data['template_preprocessing_applied'] = True  # Update this
        
        # TODO: Add your specific preprocessing steps here:
        # Examples:
        # - Extract trial structure specific to your experiment
        # - Process behavioral responses according to your paradigm
        # - Handle stimulus parameters unique to your experiment
        # - Clean trial data based on your experiment's criteria
        # - Calculate derived measures specific to your experiment
        # - Apply quality control checks for your experiment type
        
        # Example of accessing preprocessing config:
        # if self.preprocessing_config.get('remove_artifacts', {}).get('enabled', False):
        #     processed_data = self._remove_artifacts(processed_data)
        
        # TODO: Update logging prefix and message to match your experiment
        self.logger.info("TEMPLATE-SP: Template preprocessing completed")
        return processed_data
    
    # TODO: Add experiment-specific helper methods here
    # Example:
    # def _extract_trial_structure(self, session_data):
    #     """Extract trial structure specific to this experiment."""
    #     pass
    #
    # def _process_behavioral_responses(self, session_data):
    #     """Process behavioral responses for this experiment type."""
    #     pass
    #
    # def _remove_artifacts(self, session_data):
    #     """Remove artifacts based on experiment-specific criteria."""
    #     pass
