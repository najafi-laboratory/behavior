import os
import pickle
from typing import List, Dict, Any, Optional
import pandas as pd

import modules.utils.utils as utils

class DataLoader:
    """
    General data loader that handles loading preprocessed session data.
    Works with ConfigManager for configuration and validation.
    """
    
    def __init__(self, config_manager, subject_list, logger):
        """
        Initialize the DataLoader with ConfigManager and subject list.
        ConfigManager has already validated everything, so we just set up for loading.
        
        Args:
            config_manager: Instance of ConfigManager with validated configuration
            subject_list: List of subject IDs to be processed
            logger: Logger instance for logging messages
        """
        self.config_manager = config_manager
        self.subject_list = subject_list
        self.logger = logger
        
        self.logger.info("DL: Initializing DataLoader...")
        
        # Get paths from ConfigManager's config
        self.config = config_manager.config
        paths = self.config.get('paths', {})
        self.preprocessed_root_dir = paths.get('preprocessed_data', '')
        
        self.logger.info("DL: DataLoader initialized successfully")

    def load_session(self, subject_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a single preprocessed session.
        
        Args:
            subject_id: Subject identifier
            session_id: Session identifier
            
        Returns:
            Session data dictionary or None if loading fails
        """
        preprocessed_dir = os.path.join(self.preprocessed_root_dir, subject_id)
        session_path = os.path.join(preprocessed_dir, f"{session_id}_preprocessed.pkl")
        
        if not os.path.isfile(session_path):
            self.logger.warning(f"DL: Preprocessed session not found: {session_path}")
            return None
        
        try:
            with open(session_path, 'rb') as f:
                session_data = pickle.load(f)
            return session_data
        except Exception as e:
            self.logger.error(f"DL: Failed to load session {session_id} for subject {subject_id}: {e}")
            return None

    def load_subject_sessions(self, subject_id: str) -> Dict[str, Any]:
        """
        Load all preprocessed sessions for a subject.
        
        Args:
            subject_id: Subject identifier
            
        Returns:
            Dictionary with individual sessions and metadata (sessions ordered chronologically)
        """
        self.logger.info(f"DL: Loading sessions for subject {subject_id}")
        
        # Get subject config from ConfigManager
        subject_config = self.config['subjects'][subject_id]
        sessions_to_process = subject_config.get('sessions_to_process', [])
        
        if not sessions_to_process:
            self.logger.warning(f"DL: No sessions_to_process found for subject {subject_id}")
            return {
                'sessions': {},
                'metadata': {'subject_id': subject_id, 'sessions_loaded': 0}
            }
        
        # Load each session
        loaded_sessions = {}
        
        for session_name in sessions_to_process:
            session_data = self.load_session(subject_id, session_name)
            if session_data is not None:
                loaded_sessions[session_name] = session_data
        
        # Order sessions chronologically by date
        ordered_sessions = self._order_sessions_by_date(loaded_sessions)
        
        # Create result structure
        result = {
            'sessions': ordered_sessions,
            'metadata': {
                'subject_id': subject_id,
                'sessions_requested': len(sessions_to_process),
                'sessions_loaded': len(ordered_sessions),
                'missing_sessions': [s for s in sessions_to_process if s not in ordered_sessions]
            }
        }
        
        self.logger.info(f"DL: Loaded {len(ordered_sessions)}/{len(sessions_to_process)} sessions for subject {subject_id}")
        
        if result['metadata']['missing_sessions']:
            self.logger.warning(f"DL: Missing sessions for {subject_id}: {result['metadata']['missing_sessions']}")
        
        return result

    def load_all_subjects(self) -> Dict[str, Any]:
        """
        Load all preprocessed sessions for all subjects.
        
        Returns:
            Dictionary with per-subject data and overall metadata
        """
        self.logger.info(f"DL: Loading data for {len(self.subject_list)} subjects")
        
        all_data = {}
        total_sessions_requested = 0
        total_sessions_loaded = 0
        
        for subject_id in self.subject_list:
            subject_data = self.load_subject_sessions(subject_id)
            all_data[subject_id] = subject_data
            
            total_sessions_requested += subject_data['metadata']['sessions_requested']
            total_sessions_loaded += subject_data['metadata']['sessions_loaded']
        
        # Create overall result
        result = {
            'subjects': all_data,
            'metadata': {
                'experiment_config': self.config_manager.experiment_name,
                'subjects_requested': len(self.subject_list),
                'subjects_loaded': len([s for s in all_data.values() if s['metadata']['sessions_loaded'] > 0]),
                'total_sessions_requested': total_sessions_requested,
                'total_sessions_loaded': total_sessions_loaded
            }
        }
        
        self.logger.info(f"DL: Data loading completed. Loaded {total_sessions_loaded}/{total_sessions_requested} sessions across {len(self.subject_list)} subjects")
        
        return result

    def _order_sessions_by_date(self, sessions_dict):
        """Order sessions chronologically by date from session_info."""
        if not sessions_dict:
            return {}
        
        # Extract sessions with their dates
        sessions_with_dates = []
        for session_name, session_data in sessions_dict.items():
            # Get date from session_info, fallback to '99999999' for missing dates
            date = session_data.get('session_info', {}).get('date', '99999999')
            sessions_with_dates.append((date, session_name, session_data))
        
        # Just in case session file names are not in chronological order due to versioning or other reasons,
        # we will sort them by date.
        # Log the original order
        original_dates = [date for date, _, _ in sessions_with_dates]
        self.logger.info(f"DL: Session dates before sorting: {original_dates}")
        
        # Sort by date (yyyymmdd format sorts correctly as strings)
        sessions_with_dates.sort(key=lambda x: x[0])
        
        # Log the ordered dates for verification
        ordered_dates = [date for date, _, _ in sessions_with_dates]
        self.logger.info(f"DL: Session dates after sorting: {ordered_dates}")
        
        # Rebuild ordered dictionary
        ordered_sessions = {}
        for date, session_name, session_data in sessions_with_dates:
            ordered_sessions[session_name] = session_data
        
        self.logger.info(f"DL: Ordered {len(ordered_sessions)} sessions chronologically")
        return ordered_sessions


