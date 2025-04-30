# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:56:09 2025

@author: timst
"""
import os
import pickle
import pandas as pd

from modules.utils import sanitize_path

def load_extracted_session(session_file):
    """
    Load a single extracted session.
    
    Args:
        session_file (str): Full path to the extracted session file (.pkl).
        
    Returns:
        dict: Dictionary with "df" and "session_info".
    """
    session_file = sanitize_path(session_file)

    with open(session_file, 'rb') as f:
        session_data = pickle.load(f)

    return session_data

def load_subject_sessions(extracted_subject_dir, session_list):
    """
    Load multiple extracted sessions for a subject.
    
    Args:
        extracted_subject_dir (str): Directory containing extracted session files for subject.
        session_list (list of str): List of session IDs to load (no _extracted.pkl needed).
        
    Returns:
        list of dict: List of session data {"df", "session_info"} dictionaries.
    """
    extracted_subject_dir = sanitize_path(extracted_subject_dir)

    sessions = []

    for session_id in session_list:
        session_file = os.path.join(extracted_subject_dir, f"{session_id}_extracted.pkl")
        if not os.path.isfile(session_file):
            raise FileNotFoundError(f"Extracted session not found: {session_file}")
        
        session_data = load_extracted_session(session_file)
        sessions.append(session_data)

    return sessions

def load_batch_sessions(subject_list, config):
    """
    Load all sessions listed in a config file.

    Args:
        config (dict): Config dictionary, expected format:
            {
                "subject_list": ["subject1", "subject2"],
                "sessions_per_subject": {
                    "subject1": ["session1", "session2"],
                    "subject2": ["session3"],
                }
            }
        
    Returns:
        dict: Loaded data organized by subject.
            {
                "subject1": [session_data1, session_data2],
                "subject2": [session_data3],
            }
    """
    extracted_root_dir = sanitize_path(config.session_config_list_2AFC['paths']['extracted_data'])

    all_data = {}

    for subject_id in subject_list:
        extracted_subject_dir = os.path.join(extracted_root_dir, subject_id)
        
        subject_config = config.session_config_list_2AFC["list_config"].get(subject_id, None)
        session_list = subject_config["list_session_name"]

        subject_sessions = load_subject_sessions(extracted_subject_dir, session_list)

        all_data[subject_id] = subject_sessions

    return all_data

