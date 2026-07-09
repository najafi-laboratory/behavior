import scipy.io as sio
from scipy.io.matlab import mat_struct
import numpy as np

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