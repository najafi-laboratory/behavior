import os
import numpy as np
import scipy.io as sio
import pickle
from tqdm import tqdm

import modules.utils.utils as utils

class SessionExtractor:
    """
    Class for extracting session data from .mat files and saving them.
    Works with ConfigManager for configuration and validation.
    """
    def __init__(self, config_manager, subject_list, logger, force=False):
        """
        Initialize the SessionExtractor with ConfigManager and subject list.
        ConfigManager has already validated everything, so we just set up for extraction.
        """
        self.config_manager = config_manager
        self.subject_list = subject_list
        self.force = force
        self.logger = logger
        
        self.logger.info("SE: Initializing SessionExtractor...")
        
        # Get paths from ConfigManager's config (already sanitized)
        self.config = config_manager.config
        paths = self.config.get('paths', {})
        self.subjects_root_dir = paths.get('session_data', '')
        self.extracted_root_dir = paths.get('extracted_data', '')
        
        self.logger.info("SE: SessionExtractor initialized successfully")

    def load_mat(self, fname):
        """
        Load a MATLAB .mat file and convert its contents to a Python dictionary.
        Handles nested mat_struct objects and arrays.
        Returns the 'SessionData' key from the .mat file.
        """
        def _check_keys(d):
            """
            Recursively convert mat_struct objects in dictionary to nested dicts.
            """
            for key in d:
                if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
                    d[key] = _todict(d[key])
            return d

        def _todict(matobj):
            """
            Convert a mat_struct object to a Python dictionary.
            """
            d = {}
            for strg in matobj._fieldnames:
                elem = matobj.__dict__[strg]
                if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                    d[strg] = _todict(elem)
                elif isinstance(elem, np.ndarray):
                    d[strg] = _tolist(elem)
                else:
                    d[strg] = elem
            return d

        def _tolist(ndarray):
            """
            Convert a numpy ndarray to a list, handling nested mat_structs.
            """
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, sio.matlab.mio5_params.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_tolist(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return elem_list
        
        data = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
        data = _check_keys(data)
        if 'SessionData' in data:
            data = data['SessionData']
        else:
            raise KeyError(f"'SessionData' key not found in {fname}")
        return data

    def extract_session_data(self, mat_file_path):
        """
        Extract session data from a .mat file and handle errors.
        Returns the session data dictionary or None if loading fails.
        """
        try:
            data = self.load_mat(mat_file_path)
            return data
        except Exception as e:
            self.logger.error(f"SE: Failed to load .mat file: {mat_file_path} — {e}")
            return None

    def save_extracted_session(self, extracted_data, save_path):
        """
        Save extracted session data to a pickle file.
        Logs errors if saving fails.
        """
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(extracted_data, f)
        except Exception as e:
            self.logger.error(f"SE: Failed to save extracted session: {save_path} — {e}")

    def session_already_extracted(self, extracted_dir, session_id):
        """
        Check if the session has already been extracted and saved.
        Returns True if the pickle file exists.
        """
        session_file = os.path.join(extracted_dir, f"{session_id}_extracted.pkl")
        return os.path.isfile(session_file)

    def extract_and_store_subject(self, subject_id, force=False):
        """
        Extract all session .mat files for a subject and store them as pickles.

        - Skips already extracted sessions unless 'force' is True.
        - Logs errors for failed extractions.
        - Adds 'subject_name' to each extracted session data.

        Args:
            subject_id (str): The subject identifier.
            force (bool): If True, re-extract even if already extracted.
        """
        # Get subject config from ConfigManager
        subject_config = self.config['subjects'][subject_id]
        session_folder = subject_config['session_folder']
        subject_name = subject_config['subject_name']
        
        subject_dir = utils.sanitize_path(os.path.join(self.subjects_root_dir, session_folder))
        extracted_dir = utils.sanitize_path(os.path.join(self.extracted_root_dir, subject_id))

        # # Check if subject directory exists (operational check)
        # if not os.path.isdir(subject_dir):
        #     self.logger.error(f"SE: Subject directory does not exist: {subject_dir}")
        #     return

        # List all .mat files in the subject directory
        try:
            mat_files = [f for f in os.listdir(subject_dir) if f.endswith('.mat')]
        except Exception as e:
            self.logger.error(f"SE: Failed to list files in {subject_dir}: {e}")
            return

        for mat_file in tqdm(mat_files, desc=f"Extracting {subject_id}"):
            session_id = os.path.splitext(mat_file)[0]

            # Skip if already extracted and not forcing re-extraction
            if not force and self.session_already_extracted(extracted_dir, session_id):
                continue

            mat_file_path = os.path.join(subject_dir, mat_file)
            extracted_data = self.extract_session_data(mat_file_path)
            if extracted_data is None:
                # Extraction failed, error already logged
                continue

            # Add subject_name to extracted data
            extracted_data['subject_name'] = subject_name
            save_path = utils.sanitize_path(os.path.join(extracted_dir, f"{session_id}_extracted.pkl"))
            self.save_extracted_session(extracted_data, save_path)

    def batch_extract_sessions(self, force=None):
        """
        Batch extract session data for the subjects in self.subject_list.
        Uses the instance's force flag if not provided.
        Handles and logs errors encountered during extraction for each subject.
        """
        if force is None:
            force = self.force
            
        self.logger.info(f"SE: Starting batch extraction for {len(self.subject_list)} subjects")
        
        for subject_id in self.subject_list:
            # self.logger.info(f"SE: Processing subject {subject_id}")
            self.extract_and_store_subject(subject_id, force=force)
            
        self.logger.info("SE: Batch extraction completed")