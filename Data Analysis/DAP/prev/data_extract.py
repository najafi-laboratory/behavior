# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 10:59:51 2025

@author: timst
"""

import os
import numpy as np
import scipy.io as sio
import pickle
from tqdm import tqdm

import modules.utils as utils  # import the whole utils module

# --- MAT file loading utilities ---

def load_mat(fname):
    """
    Load a MATLAB .mat file and convert its contents to a Python dictionary.
    Handles nested mat_struct objects and arrays.
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

# --- Session extraction and saving ---

def extract_session_data(mat_file_path, error_log_path):
    """
    Extract session data from a .mat file and handle errors.
    Returns the session data dictionary or None if loading fails.
    """
    mat_file_path = utils.sanitize_path(mat_file_path)

    try:
        # load .mat session data as python dictionary
        data = load_mat(mat_file_path)
    except Exception as e:              
        error_msg = f"[ERROR] Failed to load .mat file: {mat_file_path} — {e}"
        print(error_msg)
        utils.log_error(error_msg, error_log_path)
        return None  # or {}, up to you 
    return data

def save_extracted_session(extracted_data, save_path, error_log_path):
    """
    Save extracted session data to a pickle file.
    Logs errors if saving fails.
    """
    save_path = utils.sanitize_path(save_path)
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(extracted_data, f)
    except Exception as e:
        error_msg = f"[ERROR] Failed to save extracted session: {save_path} — {e}"
        print(error_msg)
        utils.log_error(error_msg, error_log_path)

def session_already_extracted(extracted_dir, session_id):
    """
    Check if the session has already been extracted and saved.
    """
    extracted_dir = utils.sanitize_path(extracted_dir)
    session_file = os.path.join(extracted_dir, f"{session_id}_extracted.pkl")
    return os.path.isfile(session_file)

def extract_and_store_subject(subject_dir, extracted_dir, config, selected_config, subject_id, force=False):
    """
    Extract all session .mat files for a subject and store them as pickles.
    Skips already extracted sessions unless 'force' is True.
    Logs errors for failed extractions.
    """
    subject_dir = utils.sanitize_path(subject_dir)
    extracted_dir = utils.sanitize_and_create_dir(extracted_dir)
    error_log_path = utils.sanitize_path(config.paths['error_log_path'])

    mat_files = [f for f in os.listdir(subject_dir) if f.endswith('.mat')]

    for mat_file in tqdm(mat_files, desc=f"Extracting {os.path.basename(subject_dir)}"):
        session_id = os.path.splitext(mat_file)[0]

        if not force and session_already_extracted(extracted_dir, session_id):
            continue

        mat_file_path = os.path.join(subject_dir, mat_file)
        extracted_data = extract_session_data(mat_file_path, error_log_path)
        
        # If failed to load
        if extracted_data is None:
            error_msg = f"[SKIP] Could not load file: {mat_file_path}"
            print(error_msg)
            utils.log_error(error_msg, error_log_path)       
        else:
            # add subject name to session data
            subject_config = selected_config["list_config"].get(subject_id, None)
            if subject_config is None:
                error_msg = (
                    f"[ERROR] Subject config not found for subject_id '{subject_id}'. "
                    "Please check your subject config file."
                )
                print(error_msg)
                utils.log_error(error_msg, error_log_path)
                raise RuntimeError(error_msg)
            extracted_data['subject_name'] = subject_config["subject_name"]
            save_path = os.path.join(extracted_dir, f"{session_id}_extracted.pkl")
            save_extracted_session(extracted_data, save_path, error_log_path)

# --- Batch extraction for multiple subjects ---

def batch_extract(subject_list, config, selected_config, force=False):    
    """
    Batch extract session data for a list of subjects.
    Calls extract_and_store_subject for each subject.
    """
    subjects_root_dir = utils.sanitize_path(config.paths['session_data'])
    extracted_root_dir = utils.sanitize_and_create_dir(config.paths['extracted_data'])

    # optional clear error log
    # if os.path.exists(config.paths['error_log_path']):
    #     os.remove(config.paths['error_log_path'])

    for subject_id in subject_list:
        subject_dir = os.path.join(subjects_root_dir, subject_id)
        subject_extracted_dir = os.path.join(extracted_root_dir, subject_id)
        extract_and_store_subject(subject_dir, subject_extracted_dir, config, selected_config, subject_id, force)
