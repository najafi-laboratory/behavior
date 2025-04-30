# data_extraction.py
import os
import json
import numpy as np
from DataIOPsyTrack import run  # Your extraction function
from utils import config
from bisect import bisect_right

# JSON helper: custom encoder to handle NumPy arrays and scalars
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)

def save_dict_to_json(data, filename='result.json'):
    with open(filename, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder)

def load_json_to_dict(filename='result.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
        return {}

def extract_data(subject_list, session_data_path, fname=None, output_filename='result.json'):
    """
    Always extracts data for the given subject list and writes it to a JSON file.
    """
    print("Extracting data and saving to JSON...")
    data = run(subject_list, session_data_path, fname)
    save_dict_to_json(data, output_filename)
    return data

def update_cache_from_mat_files(subject_list, mat_dir, json_path):
    cache = load_json_to_dict(json_path)

    if not isinstance(subject_list, list):
        subject_list = [subject_list]
    subjectFound = False
    for subjectIdx in range(len(cache)):
        print(subjectIdx)
        if cache[subjectIdx]['subject'] == subject_list[0]:
            cache = [cache[subjectIdx]]
            subjectFound = True
            continue
        
    if subjectFound == False:
        print('subject not found')
        return cache
        
    # Collect .mat files in the directory
    session_data_path = mat_dir
    mat_dir = os.path.join(mat_dir, subject_list[0])
    mat_files = [f for f in os.listdir(mat_dir) if f.endswith('.mat')]

    new_data_added = False
    for fname in mat_files:
        session_id = os.path.splitext(fname)[0]
        if session_id not in cache:
            print(f"Loading new .mat file: {fname}")
            # mat_path = os.path.join(mat_dir, fname)
            session_data = extract_data(subject_list, session_data_path, fname)
            
        
            new_date = session_data[0]['dates'][0]
            dates = cache[0]['dates']
            
            if new_date in dates:
                return cache
            
            # Find index where new date should be inserted
            insert_idx = bisect_right(dates, new_date)
            
            # Insert date into list
            dates.insert(insert_idx, new_date)            
            
            for key in session_data[0].keys():
                if key not in ['dates', 'answer', 'correct', 'name', 'subject', 'total_sessions', 'y']:
                    session_data[key] = [session_data[key][i] for i in matched_indices]
             
            # For each field in session_data[0], insert corresponding value at the same index
            for key in session_data[0]:
                if key == 'dates':
                    continue
                if key not in cache[0]:
                    cache[0][key] = []
                cache[0][key].insert(insert_idx, session_data[0][key])                    
             
            print(f"Filtered sessions for subject {target_subject}")
            for date in M[subject_idx]['dates']:
                print(date)
            print('')            
            
            # cache[session_id] = session_data
            new_data_added = True

    if new_data_added:
        # save_cached_data(cache, json_path)
        print("Cache updated with new sessions.")
    else:
        print("No new .mat files to load.")

    return cache

if __name__ == "__main__":
    # Set your subject list and session folder path as needed.
    subject_list = ['LCHR_TS01']
    session_data_path = config.SESSION_DATA_PATH
    extract_data(subject_list, session_data_path)