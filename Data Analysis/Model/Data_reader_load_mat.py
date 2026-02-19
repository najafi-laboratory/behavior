import os
import re
import scipy
import scipy.io as sio
from scipy.io.matlab import mat_struct
import numpy as np

from Data_reader_session_properties import extract_session_properties
from Data_reader_licking_properties import extract_lick_properties


def parse_behavior_file_path(path):
    """
    Parses a behavioral data file path to extract:
    - Subject name
    - Version info
    - Session date

    Parameters:
    - path (str): Full path to the .mat file

    Returns:
    - subject (str): e.g., "YH24LG"
    - version (str): e.g., "V_1"
    - session_date (str): e.g., "20250628"
    """
    # Get just the filename
    filename = os.path.basename(path)

    # Remove the extension
    name_without_ext = os.path.splitext(filename)[0]

    # Use regex to extract parts
    match = re.search(r"(?P<subject>[A-Za-z0-9]+)_block.*?(?P<version>V_\d+).*?(?P<date>\d{8})", name_without_ext)
    if match:
        subject = match.group("subject")
        version = match.group("version")
        session_date = match.group("date")
        return subject, version, session_date
    else:
        raise ValueError("Filename format did not match expected pattern.")
    

def load_mat(fname):
    """
    Load MATLAB .mat file into a Python dictionary with appropriate type conversions.
    
    Args:
        fname (str): Path to the .mat file
        
    Returns:
        dict: Dictionary containing all data from the 'SessionData' field
    """
    
    def _check_keys(d):
        """
        Checks if entries in dictionary are mat-objects. If yes, converts them to nested dictionaries.
        
        Args:
            d (dict): Dictionary to check
            
        Returns:
            dict: Dictionary with mat-objects converted to dictionaries
        """
        for key in d:
            if isinstance(d[key], mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        """
        Converts a mat-object (struct) to a nested dictionary.
        
        Args:
            matobj (mat_struct): MATLAB structure object
            
        Returns:
            dict: Dictionary representation of the MATLAB structure
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        """
        Converts a numpy ndarray to a nested list with appropriate type conversion.
        
        Args:
            ndarray (numpy.ndarray): NumPy array to convert
            
        Returns:
            list: Nested list representation of the array with converted types
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    
    try:
        # Load the .mat file with specific parameters to preserve structure information
        data = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
        
        # Convert MATLAB structures to Python dictionaries
        data = _check_keys(data)
        
        # Extract only the SessionData part (main content)
        if 'SessionData' in data:
            return data['SessionData']
        else:
            raise KeyError("'SessionData' field not found in the .mat file")
            
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return None
    
def load_session_data(path):
    """
    Loads session data from a .mat file and extracts relevant properties.
    Args:
        path (str): Path to the .mat file containing session data.
    Returns:
        dict: A dictionary containing session data, subject info, version, date,
              trial properties, and lick properties.    
    """
    session = load_mat(path)
    subject, version, date = parse_behavior_file_path(path)
    # Uses the local extract_session_properties with contingency logic
    return {
        'session': session,
        'subject': subject,
        'version': version,
        'date': date,
        'trial_properties': extract_session_properties(session, subject, version, date),
        'lick_properties': extract_lick_properties(session, subject, version, date)
    }

def prepare_session_data(data_paths):
    """
    Aggregates session data from multiple data files into a structured dictionary.
    
    This function iterates through a list of data file paths, loads session data from each file,
    and collects various trial and behavioral properties into organized lists within a dictionary.
    If a file fails to load, a warning is printed and processing continues with remaining files.
    
    Args:
        data_paths (list): A list of file paths (str) pointing to session data files to be loaded.
    
    Returns:
        dict: A dictionary containing the following keys, each mapped to a list of aggregated values:
            - 'dates' (list): Session dates extracted from each loaded file.
            - 'outcomes' (list): Trial outcomes from trial_properties.
            - 'opto_tags' (list): Optogenetic tags from trial_properties.
            - 'trial_types' (list): Trial types from trial_properties.
            - 'trial_isi' (list): Inter-stimulus intervals from trial_properties.
            - 'lick_properties' (list): Lick-related behavioral properties.
            - 'block_type' (list): Block types from trial_properties.
    
    Raises:
        No exceptions are raised; errors during file loading are caught and logged as warnings.
    
    Example:
        >>> data_paths = ['/path/to/session1.mat', '/path/to/session2.mat']
        >>> sessions = prepare_session_data(data_paths)
    """
    
    sessions_data = {
        'dates': [], 'outcomes': [], 'opto_tags': [], 'trial_types': [], 'trial_isi': [],
        'lick_properties': [], 'block_type': []
    }
    for path in data_paths:
        try:
            data = load_session_data(path)
            sessions_data['dates'].append(data['date'])
            sessions_data['outcomes'].append(data['trial_properties']['outcome'])
            sessions_data['opto_tags'].append(data['trial_properties']['opto_tag'])
            sessions_data['trial_types'].append(data['trial_properties']['trial_type'])
            sessions_data['trial_isi'].append(data['trial_properties']['trial_isi'])
            sessions_data['lick_properties'].append(data['lick_properties'])
            sessions_data['block_type'].append(data['trial_properties']['block_type'])
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
    return sessions_data