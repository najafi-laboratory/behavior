import logging
from datetime import datetime
import random
from typing import List, Dict, Any

from .core.config_manager import ConfigManager
from .core.session_extractor import SessionExtractor
from .core.session_preprocessor import GeneralSessionPreprocessor
from .core.data_loader import DataLoader
from .data_analyzer import GeneralDataAnalyzer

class PipelineManager:
    """Pipeline orchestrator - minimal version."""
    
    def __init__(self, config_path: str = None, experiment_config: str = None, subject_selection: str = None, run_id: str = None):
        """Initialize the pipeline.
        
        Args:
            config_path: Path to config file
            experiment_config: Required experiment configuration name
            subject_selection: Optional subject selection. If None, uses all subjects from experiment config
            run_id: Optional shared run ID. If None, generates a new one
        """
        # Use the shared root logger that was configured with run_id
        self.logger = logging.getLogger()
        
        # Validate required parameters using logger
        if run_id is None:
            error_msg = "run_id is required"
            self.logger.error(f"PM: {error_msg}")
            raise ValueError(error_msg)
        self.run_id = run_id
        
        self.logger.info(f"PM: Initializing PipelineManager with run ID: {self.run_id}...")
        
        if experiment_config is None:
            error_msg = "experiment_config is required"
            self.logger.error(f"PM: {error_msg}")
            raise ValueError(error_msg)
            
        # Create ConfigManager with experiment and subject parameters
        # Let CM handle all validation internally
        self.config_manager = ConfigManager(config_path, experiment_config, subject_selection)
        
        # Check if ConfigManager initialized successfully
        if not self.config_manager.is_initialized_successfully():
            error_msg = f"ConfigManager failed to initialize for experiment '{experiment_config}'"
            raise ValueError(error_msg)
        
        self.config = self.config_manager.config
        
        # Store experiment config
        self.experiment_config = experiment_config
        
        # Get the validated subjects from ConfigManager
        self.subject_list = self.config_manager.get_validated_subjects()
        
        # ConfigManager returns empty list if validation failed
        if not self.subject_list:
            error_msg = f"ConfigManager failed to provide valid subjects for experiment '{experiment_config}'"
            self.logger.error(f"PM: {error_msg}")
            raise ValueError(error_msg)
        
        self.logger.info(f"PM: Experiment: {self.experiment_config}")
        self.logger.info(f"PM: Subjects: {self.subject_list}")
        
        # Initialize SessionExtractor, Preprocessor, DataLoader, and Analyzer
        self.session_extractor = None
        self.session_preprocessor = None
        self.data_loader = None
        self.data_analyzer = None
        
        self.logger.info(f"PM: PipelineManager initialized successfully with run ID: {self.run_id}")
    
    def get_subject_list(self) -> List[str]:
        """Get the subject list determined during initialization."""
        return self.subject_list
    
    def initialize_session_extractor(self, force=False):
        """Initialize SessionExtractor with ConfigManager."""
        try:
            # Direct initialization - no adapter needed
            self.session_extractor = SessionExtractor(
                config_manager=self.config_manager,
                subject_list=self.subject_list,
                force=force
            )
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize SessionExtractor: {str(e)}"
            self.logger.error(f"PM: {error_msg}")
            return False
    
    def extract_sessions(self, force=None):
        """Extract session data using SessionExtractor."""
        if self.session_extractor is None:
            if not self.initialize_session_extractor():
                raise RuntimeError("SessionExtractor not initialized")
        
        # SessionExtractor will log its own progress
        self.session_extractor.batch_extract_sessions(force=force)
    
    def initialize_session_preprocessor(self, force=False):
        """Initialize GeneralSessionPreprocessor with ConfigManager."""
        try:
            self.session_preprocessor = GeneralSessionPreprocessor(
                config_manager=self.config_manager,
                subject_list=self.subject_list,
                force=force
            )
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize GeneralSessionPreprocessor: {str(e)}"
            self.logger.error(f"PM: {error_msg}")
            return False
    
    def preprocess_sessions(self, force=None):
        """Preprocess session data using GeneralSessionPreprocessor."""
        if self.session_preprocessor is None:
            if not self.initialize_session_preprocessor():
                raise RuntimeError("GeneralSessionPreprocessor not initialized")
        
        self.session_preprocessor.batch_preprocess_sessions(force=force)
    
    def initialize_data_loader(self):
        """Initialize DataLoader with ConfigManager."""
        try:
            self.data_loader = DataLoader(
                config_manager=self.config_manager,
                subject_list=self.subject_list
            )
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize DataLoader: {str(e)}"
            self.logger.error(f"PM: {error_msg}")
            return False
    
    def load_data(self):
        """Load preprocessed data using DataLoader."""
        if self.data_loader is None:
            if not self.initialize_data_loader():
                raise RuntimeError("DataLoader not initialized")
        
        return self.data_loader.load_all_subjects()
    
    def initialize_data_analyzer(self, loaded_data):
        """Initialize GeneralDataAnalyzer with ConfigManager and loaded data."""
        try:
            self.data_analyzer = GeneralDataAnalyzer(
                config_manager=self.config_manager,
                subject_list=self.subject_list,
                loaded_data=loaded_data
            )
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize GeneralDataAnalyzer: {str(e)}"
            self.logger.error(f"PM: {error_msg}")
            return False
    
    def analyze_data(self, loaded_data):
        """Analyze data using GeneralDataAnalyzer."""
        if self.data_analyzer is None:
            if not self.initialize_data_analyzer(loaded_data):
                raise RuntimeError("GeneralDataAnalyzer not initialized")
        
        return self.data_analyzer.analyze_data()