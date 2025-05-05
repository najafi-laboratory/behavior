# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 10:59:51 2025

@author: timst
"""

import os
import numpy as np
import scipy.io as sio
import pandas as pd
import pickle
from tqdm import tqdm

from modules.utils import sanitize_path, sanitize_and_create_dir  # assume your utilities live here


# read .mat to dict
def load_mat(fname):
    def _check_keys(d):
        for key in d:
            if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
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
    data = data['SessionData']
    return data

# def build_trial_dataframe(trial_data):
#     rows = []
    
#     events_to_get = ['Port1In',
#                    'Port1Out',
#                    'Port3In',
#                    'Port3Out',                   
#                    ]

#     for idx, trial in enumerate(trial_data):
#         row = {'trial_index': idx}  # Include index in the row

#         states = trial['States']
#         for state in states:
#             try:
#                 row[state] = states[state]
#             except Exception as e:
#                 print(f"Warning: Failed to extract '{state}' in trial {idx}: {e}")
#                 row[state] = None

#         events = trial['Events']
#         for event in events_to_get:
#             if event in events:
#                 row[event] = events[event]
#             else:
#                 # row[event] = [np.nan]
#                 row[event] = np.nan
#         rows.append(row)

#     return pd.DataFrame(rows)

def extract_session_data(mat_file_path):
    mat_file_path = sanitize_path(mat_file_path)

    try:
        # load .mat session data as python dictionary
        data = load_mat(mat_file_path)
    except Exception as e:
        print(f"[ERROR] Failed to load .mat file: {mat_file_path} — {e}")
        return None  # or {}, up to you        

    print('')
    

    
    return data

# def extract_session_data(mat_file_path):
#     mat_file_path = sanitize_path(mat_file_path)

#     try:
#         # load .mat session data as python dictionary
#         data = load_mat(mat_file_path)
#     except Exception as e:
#         print(f"[ERROR] Failed to load .mat file: {mat_file_path} — {e}")
#         return None  # or {}, up to you        

#     print('')
    
#     # --- Session Data ---
#     # get gui settings from first trial
#     GUISettings = data['TrialSettings'][0]['GUI']
    
#     # get session data
#     session_info = {
#         'ComputerHostName' : data['ComputerHostName'],        
#         'RigName' : data['RigName'],
#         'nTrials' : data['nTrials'],        
#         'SessionDate' : data['Info']['SessionDate'],
#         'SessionStartTime_MATLAB' : data['Info']['SessionStartTime_MATLAB'],
#         'SessionStartTime_UTC' : data['Info']['SessionStartTime_UTC'],  
#         'TrialTypes' : data['TrialTypes'],
#         'OptoType' : data.get('OptoType', 0),
#         'GUISettings': GUISettings,
#     }
    
#     # --- Trial Data ---
#     # get trial data variable
#     trial_data = data['RawEvents']['Trial']

#     # check we have the right number of trials    
#     assert data['nTrials'] == len(trial_data), "Mismatch: expected ntrials does not match length of trial_data"

#     # get df for trial
#     trials_df = build_trial_dataframe(trial_data)

#     # compile extracted data
#     extracted = {
#         "session_info": session_info,
#         "df": trials_df,    
#     }

#     print('')
    
#     return extracted

def save_extracted_session(extracted_data, save_path):
    save_path = sanitize_path(save_path)

    with open(save_path, 'wb') as f:
        pickle.dump(extracted_data, f)

def session_already_extracted(extracted_dir, session_id):
    extracted_dir = sanitize_path(extracted_dir)
    session_file = os.path.join(extracted_dir, f"{session_id}_extracted.pkl")
    return os.path.isfile(session_file)

def extract_and_store_subject(subject_dir, extracted_dir, config, subject_id, force=False):
    subject_dir = sanitize_path(subject_dir)
    extracted_dir = sanitize_and_create_dir(extracted_dir)
    error_log_path = sanitize_path(config.paths['error_log_path'])

    mat_files = [f for f in os.listdir(subject_dir) if f.endswith('.mat')]

    for mat_file in tqdm(mat_files, desc=f"Extracting {os.path.basename(subject_dir)}"):
        session_id = os.path.splitext(mat_file)[0]

        if not force and session_already_extracted(extracted_dir, session_id):
            continue

        mat_file_path = os.path.join(subject_dir, mat_file)
        extracted_data = extract_session_data(mat_file_path)
        
        # If failed to load
        if extracted_data is None:
            error_msg = f"[SKIP] Could not load file: {mat_file_path}"
            print(error_msg)

            # Log to file
            # err_file_path = os.path.join(error_log_path, "error_log_data.txt")
            # with open(err_file_path, 'a') as log_file:
            #     log_file.write(error_msg + '\n')
            # continue        
        else:
            # add subject name to session data
            subject_config = config.session_config_list_2AFC["list_config"].get(subject_id, None)
            extracted_data['subject_name'] = subject_config["subject_name"]
            save_path = os.path.join(extracted_dir, f"{session_id}_extracted.pkl")
            save_extracted_session(extracted_data, save_path)

# def batch_extract(subjects_root_dir, extracted_root_dir, subject_list):
def batch_extract(subject_list, config, force=False):    
    subjects_root_dir = sanitize_path(config.paths['session_data'])
    extracted_root_dir = sanitize_and_create_dir(config.paths['extracted_data'])

    # optional clear error log
    # if os.path.exists(config.paths['error_log_path']):
    #     os.remove(config.paths['error_log_path'])

    for subject_id in subject_list:
        subject_dir = os.path.join(subjects_root_dir, subject_id)
        subject_extracted_dir = os.path.join(extracted_root_dir, subject_id)
        extract_and_store_subject(subject_dir, subject_extracted_dir, config, subject_id, force)
