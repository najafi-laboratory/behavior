# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 08:46:06 2025

@author: timst
"""

import os
import random
from datetime import datetime


import modules.utils as utils
import modules.config as config
import modules.data_extract as extract
import modules.data_load as load_data
import modules.data_preprocess as preprocess

# import warnings
# warnings.filterwarnings('ignore')

def remove_substrings(s, substrings):
    for sub in substrings:
        s = s.replace(sub, "")
    return s

def flip_underscore_parts(s):
    parts = s.split("_", 1)  # Split into two parts at the first underscore
    if len(parts) < 2:
        return s  # Return original string if no underscore is found
    if "TS" in parts[1]:  # Check if second part contains "TS"
        return f"{parts[1]}_{parts[0]}"
    else:
        return s

def lowercase_h(s):
    return s.replace('H', 'h')

def filter_sessions(M, session_config_list):
    for config in session_config_list['list_config']:
        target_subject = config['subject_name']
        session_names = list(config['list_session_name'])
    
        # Find the subject index in M
        subject_idx = None
        for idx, subject_data in enumerate(M):
            # You can customize this match logic depending on your JSON structure
            if target_subject in subject_data['subject']:
                subject_idx = idx
                break
    
        if subject_idx is None:
            print(f"Subject {target_subject} not found in loaded data.")
            continue

        filenames = M[subject_idx]['session_filenames']
    
        # Get indices of matches
        matched_indices = [i for i, fname in enumerate(filenames) if fname in session_names]

        print(f"Matched indices for subject {target_subject}: {matched_indices}")      
        
        for key in M[subject_idx].keys():
            if key not in ['answer', 'correct', 'name', 'subject', 'total_sessions', 'y']:
                M[subject_idx][key] = [M[subject_idx][key][i] for i in matched_indices]
                
        print(f"Filtered sessions for subject {target_subject}")
        for date in M[subject_idx]['dates']:
            print(date)
        print('')
   
    return M

def assign_grid_position(index, grid_size, block_size):
    grid_rows, grid_cols = grid_size
    block_rows, block_cols = block_size

    figs_per_row = grid_cols // block_cols
    figs_per_col = grid_rows // block_rows
    figs_per_page = figs_per_row * figs_per_col

    page_idx = index // figs_per_page
    index_in_page = index % figs_per_page

    row = (index_in_page // figs_per_row) * block_rows
    col = (index_in_page % figs_per_row) * block_cols

    # page_key = f"{base_page_key}_p{page_idx}"
    return page_idx, row, col

def generate_paged_pdf_spec(
    config_dict,
    total_items,
    grid_size=(4, 8),
    fig_size=(30, 15),
    block_size=(2, 2),
    dpi=300,
    margins=None,
    start_index=0    
):
    if margins is None:
        margins = {
            "left": 0,
            "right": 0,
            "top": 0,
            "bottom": 0,
            "wspace": 0,
            "hspace": 0,
        }
    figs_per_row = grid_size[1] // block_size[1]
    figs_per_col = grid_size[0] // block_size[0]
    figs_per_page = figs_per_row * figs_per_col
    num_pages = (total_items + figs_per_page - 1) // figs_per_page

    return num_pages, grid_size, block_size  # if you want to track how many were created

if __name__ == "__main__":
    # Get the current date
    current_date = datetime.now()
    # Format the date as 'yyyymmdd'
    formatted_date = current_date.strftime('%Y%m%d')
    
    # random num
    num_str = f"{random.randint(0, 9999):04d}"
    

    
    # session_data_path = directories.SESSION_DATA_PATH
    # figure_dir_local = config.FIGURE_DIR_LOCAL
    # output_dir_onedrive = config.OUTPUT_DIR_ONEDRIVE
    # output_dir_local = config.OUTPUT_DIR_LOCAL

    # session_data_path = directories.SESSION_DATA_PATH

    # subject_list = ['LCHR_TS01', 'LCHR_TS02', 'SCHR_TS06', 'SCHR_TS07', 'SCHR_TS08', 'SCHR_TS09']
    # subject_list = ['LCHR_TS01']
    # subject_list = ['LCHR_TS02']
    # subject_list = ['SCHR_TS06']
    # subject_list = ['SCHR_TS07']
    # subject_list = ['SCHR_TS08']
    # subject_list = ['SCHR_TS09']
    # subject_list = ['TS03', 'YH24']
    # subject_list = ['TS03']
    # subject_list = ['YH24']
    
    # for subject in subject_list:
    #     update_cache_from_mat_files(subject, config.paths['session_data'], 'result.json')
    # extract_data(subject_list, config.paths['session_data'])

    # session_configs = session_config_list_2AFC

    # M = load_json_to_dict('result.json')
    
    

    # M = filter_sessions(M, config.session_config_list_2AFC)
    
    # Settings
    # subjects_root_dir = utils.sanitize_path(config.paths['session_data'])
    # extracted_root_dir = utils.sanitize_and_create_dir(config.paths['extracted_data'])
    
    
    
    subject_list = ["LCHR_TS01", "LCHR_TS02"]
    subjects = ", ".join([s for s in subject_list])
    print(f"Extracting data for subjects {subjects}...")

    # Extract and store sessions
    # extract.batch_extract(subjects_root_dir, extracted_root_dir, subject_list)
    extract.batch_extract(config, subject_list, force=False)
    
    subject_list = ["LCHR_TS01", "LCHR_TS02"]
    subjects = ", ".join([s for s in subject_list])
    print(f"Preprocessing data for subjects {subjects}...")    
    
    # datist - data extraction, data bridging, data cleaning, phantom data, data snoopingecho
    all_subjects_data = load_data.load_batch_sessions(subject_list, config)
    
    preprocessed_data = preprocess.batch_preprocess_sessions(subject_list, config, force=True)


    print(f"Running data analysis for subjects {subjects}...") 
    # Access a session
    df = preprocessed_data["LCHR_TS01"][0]["df"]
    session_info = preprocessed_data["LCHR_TS01"][0]["session_info"]
    
    
    
    print("")



