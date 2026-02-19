"""
Convert MATLAB .mat files to Python .pkl files for GPFA analysis.

This script converts sample_dat.mat to sample_dat.pkl with proper
data structure formatting for the Python GPFA implementation.
"""

import scipy.io
import pickle
import numpy as np
import os
from typing import Dict, Any, List


def convert_matlab_struct_to_dict(matlab_struct_array) -> List[Dict[str, Any]]:
    """
    Convert MATLAB struct array to Python list of dictionaries.
    
    INPUTS:
    matlab_struct_array - numpy array of mat_struct objects from scipy.io.loadmat
    
    OUTPUTS:
    data_list - List of dictionaries with proper field names
    """
    
    # Check if it's an array of mat_struct objects
    if not isinstance(matlab_struct_array, np.ndarray):
        return matlab_struct_array
    
    # Check if the first element is a mat_struct
    if len(matlab_struct_array) == 0:
        return []
    
    first_element = matlab_struct_array[0]
    
    # Check if it's a mat_struct object
    if not hasattr(first_element, '_fieldnames'):
        print(f"  Not a mat_struct array, element type: {type(first_element)}")
        return matlab_struct_array
    
    # Get field names from the first struct
    field_names = first_element._fieldnames
    print(f"  Struct field names: {field_names}")
    print(f"  Processing {len(matlab_struct_array)} struct entries...")
    
    data_list = []
    
    # Iterate through each struct in the array
    for i, struct_item in enumerate(matlab_struct_array):
        entry = {}
        
        for field_name in field_names:
            try:
                # Access field data from the mat_struct object
                field_data = getattr(struct_item, field_name)
                
                # Convert field name (keeping original for now)
                python_field = convert_field_name(field_name)
                
                # Handle different data types
                if isinstance(field_data, np.ndarray):
                    if field_data.size == 1:
                        # Scalar values
                        entry[python_field] = field_data.item()
                    else:
                        # Arrays - keep as numpy arrays
                        entry[python_field] = field_data
                elif isinstance(field_data, (int, float, str)):
                    # Simple types
                    entry[python_field] = field_data
                else:
                    # Other types - try to extract if possible
                    if hasattr(field_data, 'item') and callable(field_data.item):
                        entry[python_field] = field_data.item()
                    else:
                        entry[python_field] = field_data
                        
            except Exception as e:
                print(f"    Error processing field '{field_name}' in entry {i}: {e}")
                continue
        
        data_list.append(entry)
        
        # Print progress for first few entries
        if i < 3:
            print(f"    Entry {i}: {list(entry.keys())}")
            if 'trialId' in entry:
                print(f"      trialId: {entry['trialId']}")
            if 'spikes' in entry:
                spikes_shape = entry['spikes'].shape if hasattr(entry['spikes'], 'shape') else 'unknown'
                print(f"      spikes: shape {spikes_shape}")
    
    return data_list


def convert_field_name(matlab_name: str) -> str:
    """
    Convert MATLAB field names to Python convention.
    
    Keep original field names for maximum compatibility.
    """
    
    # Keep original field names for compatibility
    return matlab_name


def mat_to_pkl(mat_filename: str, pkl_filename: str = None) -> str:
    """
    Convert a MATLAB .mat file to Python .pkl file.
    
    INPUTS:
    mat_filename - Path to input .mat file
    pkl_filename - Path to output .pkl file (optional, auto-generated if None)
    
    OUTPUTS:
    pkl_filename - Path to created .pkl file
    """
    
    if not os.path.exists(mat_filename):
        raise FileNotFoundError(f"Input file not found: {mat_filename}")
    
    # Generate output filename if not provided
    if pkl_filename is None:
        base_name = os.path.splitext(mat_filename)[0]
        pkl_filename = f"{base_name}.pkl"
    
    print(f"Loading MATLAB file: {mat_filename}")
    
    # Load MATLAB file with struct_as_record=False to preserve structure
    try:
        mat_data = scipy.io.loadmat(mat_filename, struct_as_record=False, squeeze_me=True)
    except Exception as e:
        print(f"Error loading MATLAB file: {e}")
        raise
    
    # Remove MATLAB metadata
    clean_data = {k: v for k, v in mat_data.items() 
                  if not k.startswith('__')}
    
    print(f"Found variables: {list(clean_data.keys())}")
    
    # The main conversion logic
    if len(clean_data) == 1:
        # If there's only one variable, extract it directly
        # This handles the case where the .mat file contains just 'dat'
        var_name, var_data = next(iter(clean_data.items()))
        print(f"Single variable found: {var_name}")
        print(f"Variable type: {type(var_data)}")
        
        if hasattr(var_data, 'dtype'):
            print(f"Variable dtype: {var_data.dtype}")
            if hasattr(var_data, 'shape'):
                print(f"Variable shape: {var_data.shape}")
        
        # Check for mat_struct objects
        if (isinstance(var_data, np.ndarray) and 
            len(var_data) > 0 and 
            hasattr(var_data[0], '_fieldnames')):
            # It's an array of mat_struct objects
            print("Converting mat_struct array...")
            converted_data = convert_matlab_struct_to_dict(var_data)
        elif hasattr(var_data, 'dtype') and var_data.dtype.names:
            # It's a structured numpy array
            print("Converting structured numpy array...")
            converted_data = convert_matlab_struct_to_dict(var_data)
        else:
            # It's some other type
            print("Not a struct array, keeping as-is...")
            converted_data = var_data
            
    else:
        # Multiple variables - convert each one
        converted_data = {}
        for var_name, var_data in clean_data.items():
            print(f"Converting variable: {var_name}")
            
            # Check for mat_struct objects
            if (isinstance(var_data, np.ndarray) and 
                len(var_data) > 0 and 
                hasattr(var_data[0], '_fieldnames')):
                # Array of mat_struct objects
                converted_data[var_name] = convert_matlab_struct_to_dict(var_data)
            elif hasattr(var_data, 'dtype') and var_data.dtype.names:
                # Structured numpy array
                converted_data[var_name] = convert_matlab_struct_to_dict(var_data)
            else:
                # Other types
                converted_data[var_name] = var_data
    
    # Save as pickle file
    print(f"Saving Python pickle file: {pkl_filename}")
    
    try:
        with open(pkl_filename, 'wb') as f:
            pickle.dump(converted_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Error saving pickle file: {e}")
        raise
    
    print(f"Conversion completed successfully!")
    print(f"Output file: {pkl_filename}")
    
    return pkl_filename


def convert_sample_data():
    """
    Convert the sample_dat.mat file specifically.
    """
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mat_file = os.path.join(script_dir, 'sample_dat.mat')
    pkl_file = os.path.join(script_dir, 'sample_dat.pkl')
    
    if not os.path.exists(mat_file):
        print(f"Sample data file not found: {mat_file}")
        print("Please ensure sample_dat.mat is in the same directory as this script.")
        return None
    
    try:
        output_file = mat_to_pkl(mat_file, pkl_file)
        
        # Verify the conversion by loading and inspecting the data
        print("\nVerifying conversion...")
        with open(output_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Top-level structure type: {type(data)}")
        
        if isinstance(data, list):
            print(f"List with {len(data)} entries")
            if len(data) > 0:
                print(f"First entry type: {type(data[0])}")
                if isinstance(data[0], dict):
                    print(f"First entry fields: {list(data[0].keys())}")
                    # Show a sample of the data
                    for key, value in list(data[0].items())[:5]:
                        if isinstance(value, np.ndarray):
                            print(f"  {key}: array shape {value.shape}, dtype {value.dtype}")
                        else:
                            print(f"  {key}: {type(value)} = {value}")
                    
                    # Test the exact access pattern that GPFA uses
                    print("\nTesting GPFA access pattern:")
                    try:
                        n = 0  # First trial
                        y_dim = data[n]['spikes'].shape[0]
                        T_full = data[n]['spikes'].shape[1]
                        trial_id = data[n]['trialId']
                        print(f"  Trial {n}: trialId={trial_id}, spikes shape=({y_dim}, {T_full})")
                        print("  ✓ GPFA access pattern works!")
                    except Exception as e:
                        print(f"  ✗ GPFA access pattern failed: {e}")
                        
        elif isinstance(data, dict):
            print(f"Dictionary with keys: {list(data.keys())}")
            for key, value in data.items():
                if isinstance(value, list):
                    print(f"  {key}: List with {len(value)} entries")
                    if len(value) > 0 and isinstance(value[0], dict):
                        print(f"    First entry fields: {list(value[0].keys())}")
                elif isinstance(value, np.ndarray):
                    print(f"  {key}: Array shape {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
        else:
            print(f"Data type: {type(data)}")
        
        return output_file
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def inspect_mat_file(mat_filename: str):
    """
    Inspect the structure of a .mat file without converting it.
    """
    print(f"Inspecting MATLAB file: {mat_filename}")
    
    try:
        # Load with struct_as_record=False to see the true structure
        mat_data = scipy.io.loadmat(mat_filename, struct_as_record=False, squeeze_me=True)
        
        print("Raw loaded data keys:", list(mat_data.keys()))
        
        # Remove metadata
        clean_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}
        
        for var_name, var_data in clean_data.items():
            print(f"\nVariable: {var_name}")
            print(f"  Type: {type(var_data)}")
            
            if hasattr(var_data, 'dtype'):
                print(f"  Dtype: {var_data.dtype}")
                if hasattr(var_data.dtype, 'names') and var_data.dtype.names:
                    print(f"  Field names: {var_data.dtype.names}")
            
            if hasattr(var_data, 'shape'):
                print(f"  Shape: {var_data.shape}")
                
            # Try to peek at first element if it's an array
            if hasattr(var_data, '__len__') and len(var_data) > 0:
                first_elem = var_data[0]
                print(f"  First element type: {type(first_elem)}")
                
                # Check if it's a mat_struct
                if hasattr(first_elem, '_fieldnames'):
                    print(f"  mat_struct field names: {first_elem._fieldnames}")
                    # Try to access the fields
                    for field_name in first_elem._fieldnames:
                        field_value = getattr(first_elem, field_name)
                        if hasattr(field_value, 'shape'):
                            print(f"    {field_name}: shape {field_value.shape}")
                        else:
                            print(f"    {field_name}: {type(field_value)} = {field_value}")
                
    except Exception as e:
        print(f"Error inspecting file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # First inspect the structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mat_file = os.path.join(script_dir, 'sample_dat.mat')
    
    if os.path.exists(mat_file):
        print("=== INSPECTING MAT FILE STRUCTURE ===")
        inspect_mat_file(mat_file)
        print("\n" + "="*50)
        
        print("\n=== CONVERTING TO PKL ===")
        convert_sample_data()
    else:
        print(f"File not found: {mat_file}")