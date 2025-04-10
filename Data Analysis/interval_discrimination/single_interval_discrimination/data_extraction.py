# data_extraction.py
import os
import json
import numpy as np
from DataIOPsyTrack import run  # Your extraction function
from utils import config

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
    with open(filename, 'r') as f:
        return json.load(f)

def extract_data(subject_list, session_data_path, output_filename='result.json'):
    """
    Always extracts data for the given subject list and writes it to a JSON file.
    """
    print("Extracting data and saving to JSON...")
    data = run(subject_list, session_data_path)
    save_dict_to_json(data, output_filename)
    return data

if __name__ == "__main__":
    # Set your subject list and session folder path as needed.
    subject_list = ['LCHR_TS01']
    session_data_path = config.SESSION_DATA_PATH
    extract_data(subject_list, session_data_path)