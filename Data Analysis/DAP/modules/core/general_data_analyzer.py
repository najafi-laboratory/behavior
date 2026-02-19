import os
import logging
import importlib
from typing import Dict, Any, Optional

import modules.utils as utils

class GeneralDataAnalyzer:
    """
    General data analyzer that handles common analysis tasks.
    Works with ConfigManager and loads experiment-specific analyzers.
    """
    
    def __init__(self, config_manager, subject_list, loaded_data, logger):
        """
        Initialize the GeneralDataAnalyzer with ConfigManager, subject list, and loaded data.
        """
        self.config_manager = config_manager
        self.subject_list = subject_list
        self.loaded_data = loaded_data
        self.logger = logger
        
        self.logger.info("DA: Initializing GeneralDataAnalyzer...")
        
        # Get config
        self.config = config_manager.config
        
        # Initialize experiment-specific analyzer if specified
        self.experiment_analyzer = self._load_experiment_analyzer()
        
        self.logger.info("DA: GeneralDataAnalyzer initialized successfully")

    def _load_experiment_analyzer(self):
        """Load experiment-specific analyzer if configured."""
        try:
            # Get experiment config
            experiment_config = self.config.get('experiment_configs', {}).get(self.config_manager.experiment_name, {})
            analyzer_module_name = experiment_config.get('analyzer_module')
            
            if analyzer_module_name is None:
                self.logger.info("DA: No experiment-specific analyzer configured, using general analysis only")
                return None
            
            # Get analyzer info from available list
            available_analyzers = self.config.get('available_analyzers', {})
            analyzer_info = available_analyzers.get(analyzer_module_name)
            
            if not analyzer_info:
                self.logger.error(f"DA: Analyzer module '{analyzer_module_name}' not found in available_analyzers")
                return None
            
            class_name = analyzer_info.get('class')
            if not class_name:
                self.logger.error(f"DA: No class specified for analyzer module '{analyzer_module_name}'")
                return None
            
            self.logger.info(f"DA: Loading experiment-specific analyzer: {class_name} from {analyzer_module_name}")
            
            # Direct import of the specific module and class
            module = importlib.import_module(f"modules.experiments.{self.config_manager.experiment_name}.{analyzer_module_name}")
            analyzer_class = getattr(module, class_name)
            
            # Initialize the experiment-specific analyzer
            experiment_analyzer = analyzer_class(self.config_manager, self.subject_list, self.loaded_data, self.logger)
            self.logger.info(f"DA: Successfully loaded experiment-specific analyzer: {class_name}")
            
            return experiment_analyzer
            
        except Exception as e:
            self.logger.error(f"DA: Failed to load experiment-specific analyzer: {e}")
            self.logger.info("DA: Falling back to general analysis only")
            return None

    def analyze_data(self) -> Dict[str, Any]:
        """
        Perform data analysis using experiment-specific analyzer if available.
        
        Returns:
            Analysis results dictionary
        """
        self.logger.info("DA: Starting data analysis...")
        
        if self.experiment_analyzer is not None:
            # Use experiment-specific analysis
            return self.experiment_analyzer.analyze_data()
        else:
            # Use general analysis only
            self.logger.info("DA: Applying general analysis...")
            return self._general_analysis()

    def _general_analysis(self) -> Dict[str, Any]:
        """
        Perform basic general analysis when no experiment-specific analyzer is available.
        """
        self.logger.info("DA: Performing general analysis...")
        
        # Basic analysis - count sessions and trials
        analysis_results = {
            'analysis_type': 'general',
            'experiment_config': self.config_manager.experiment_name,
            'subjects_analyzed': len(self.subject_list),
            'total_sessions': self.loaded_data['metadata']['total_sessions_loaded'],
            'summary': {}
        }
        
        # Add per-subject summary
        for subject_id, subject_data in self.loaded_data['subjects'].items():
            analysis_results['summary'][subject_id] = {
                'sessions_analyzed': subject_data['metadata']['sessions_loaded'],
                'sessions_requested': subject_data['metadata']['sessions_requested']
            }
        
        self.logger.info("DA: General analysis completed")
        return analysis_results
