# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:53:39 2025

@author: timst
"""
import gc
import os
import logging
import sys
from datetime import datetime
import random
import shutil
import numpy as np
import h5py
import scipy.io
import pickle


def setup_logging(run_id: str):
    """Configure logging for the entire application.

    Log levels available:
        logging.DEBUG
        logging.INFO
        logging.WARNING
        logging.ERROR
        logging.CRITICAL
    """
    # Create logs directory if it doesn't exist
    import os
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging format with run_id
    formatter = logging.Formatter(
        f'%(asctime)s - {run_id} - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler(f'logs/pipeline_{run_id}.log')
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # root_logger.setLevel(logging.DEBUG)    # Uncomment for verbose debug output
    # root_logger.setLevel(logging.WARNING)  # Uncomment to show only warnings and above
    # root_logger.setLevel(logging.ERROR)    # Uncomment to show only errors and above
    # root_logger.setLevel(logging.CRITICAL) # Uncomment to show only critical errors
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return root_logger

def sanitize_path(path):
    """
    Ensure a path is safe and standardized for the current OS.
    
    Args:
        path (str): The input path string.
        
    Returns:
        str: A cleaned, OS-safe path.
    """
    if path is None or path == "":
        raise ValueError("Path is empty or None")
    
    path = os.path.normpath(path)  # normalize slashes, remove redundant slashes
    return path

def sanitize_and_create_dir(path):
    """
    Standardize path and ensure directory exists (for folders).
    
    Args:
        path (str): Directory path.
        
    Returns:
        str: Cleaned directory path.
    """
    clean_path = sanitize_path(path)
    os.makedirs(clean_path, exist_ok=True)
    return clean_path

def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def cleanup_memory(variables):
    for var in variables:
        del var
    gc.collect()
    
def get_figsize_from_pdf_spec(rowspan, colspan, pdf_spec):
    # pdf_spec = config['pdf_spec']
    grid_size = pdf_spec['grid_size']
    fig_size = pdf_spec['fig_size']    
    
    cell_width_in = fig_size[0]
    cell_height_in = fig_size[1] 
    
    nrows = grid_size[0]
    ncols = grid_size[1]
 
    cell_width_in = fig_size[0] / ncols
    cell_height_in = fig_size[1] / nrows
    
    width_in  = colspan * cell_width_in
    height_in = rowspan * cell_height_in
    
    return (width_in, height_in)


def save_plot(fig, subject_id, session_date, plot_name, output_root='plots', save_png=True, save_pdf=True):
    """
    Save a matplotlib figure to both PDF and PNG formats with standardized filenames.

    Args:
        fig          : Matplotlib figure object
        subject_id   : e.g., 'TS01'
        session_date : e.g., '20250405'
        plot_name    : e.g., 'isi_distribution', 'outcome_donut'
        output_root  : root directory for saving (default = 'plots/')
        save_png     : whether to save a .png version
        save_pdf     : whether to save a .pdf version

    Returns:
        dict with 'pdf' and 'png' paths (if saved)
    """
    output_dir = os.path.join(output_root, subject_id, session_date)
    os.makedirs(output_dir, exist_ok=True)

    base_path = os.path.join(output_dir, plot_name)
    saved_paths = {}

    if save_pdf:
        pdf_path = base_path + '.pdf'
        fig.savefig(pdf_path, bbox_inches='tight')
        saved_paths['pdf'] = pdf_path

    if save_png:
        png_path = base_path + '.png'
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        saved_paths['png'] = png_path

    return saved_paths

def create_memmap(data, dtype, mmap_path, logger=None):
    """
    Creates a memory-mapped array from input data and saves it to a specified file path.

    Args:
        data (numpy.ndarray): The input NumPy array whose data will be written to the
            memory-mapped file.
        dtype (numpy.dtype or str): The data type of the memory-mapped array.
        mmap_path (str): The file path where the memory-mapped array will be stored.
        logger (logging.Logger, optional): Logger instance for debug output.

    Returns:
        numpy.memmap: A memory-mapped array with the specified shape and data type,
            containing the data from the input array.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(mmap_path), exist_ok=True)
    
    if logger:
        logger.info(f"Creating memmap: {mmap_path} with shape {data.shape} and dtype {dtype}")
    
    memmap_arr = np.memmap(mmap_path, dtype=dtype, mode='w+', shape=data.shape)
    memmap_arr[:] = data[...]
    
    if logger:
        logger.info(f"Successfully created memmap: {mmap_path}")
    
    return memmap_arr

def get_memmap_path(base_path, identifier, logger=None):
    """
    Generates paths for memory-mapped data directory.

    Args:
        base_path (str): Base directory path (e.g., session output path)
        identifier (str): Identifier for the memmap subdirectory (e.g., 'sid_imaging')
        logger (logging.Logger, optional): Logger instance for debug output.

    Returns:
        str: Path to the memory-mapped data directory
    """
    memmap_dir = os.path.join(base_path, 'memmap', identifier)
    os.makedirs(memmap_dir, exist_ok=True)
    
    if logger:
        logger.info(f"Created memmap directory: {memmap_dir}")
    
    return memmap_dir

def clean_memmap_path(base_path, identifier=None, logger=None):
    """
    Clean memory mapping files for development/debugging.
    
    Args:
        base_path (str): Base directory path
        identifier (str, optional): Specific memmap subdirectory to clean. 
                                   If None, cleans entire memmap directory.
        logger (logging.Logger, optional): Logger instance for debug output.
    """
    try:
        if identifier is None:
            # Clean entire memmap directory
            memmap_base = os.path.join(base_path, 'memmap')
            if os.path.exists(memmap_base):
                shutil.rmtree(memmap_base)
                if logger:
                    logger.info(f"Cleaned entire memmap directory: {memmap_base}")
        else:
            # Clean specific identifier directory
            memmap_dir = os.path.join(base_path, 'memmap', identifier)
            if os.path.exists(memmap_dir):
                shutil.rmtree(memmap_dir)
                if logger:
                    logger.info(f"Cleaned memmap directory: {memmap_dir}")
    except Exception as e:
        # Log cleanup errors but don't raise (files might be in use)
        if logger:
            logger.warning(f"Failed to clean memmap directory: {e}")





# # Option 1: Convert and save to HDF5 (recommended)
# h5_path = utils.save_trial_data_h5(trial_data, 'output/data/session_001/spike_segments.h5', logger=self.logger)

# # Convert to MATLAB when needed for GPFA
# mat_path = utils.h5_to_mat(h5_path, logger=self.logger)

# # Option 2: Save directly to MATLAB format
# # mat_path = utils.save_trial_data_mat(trial_data, 'data/session_001/spike_segments.mat', logger)

# # Option 3: Manual conversion for more control
# trial_segments = utils.convert_trial_data_to_segments(trial_data, logger=self.logger)
# h5_path = utils.save_trial_segments_h5(trial_segments, 'output/data/session_001/spike_segments.h5', logger=self.logger)

# # Verify before using with GPFA
# is_valid = utils.verify_trial_segments_structure(trial_segments, logger=self.logger)






def save_trial_data_h5(trial_data, output_path, logger=None):
    """
    Save trial_data directly to HDF5 format with MATLAB-compatible structure.
    
    Args:
        trial_data (dict): Dictionary containing 'df_trials_with_segments' DataFrame
        output_path (str): Path for the output .h5 file
        logger (logging.Logger, optional): Logger instance
    
    Returns:
        str: Path to saved file
    """
    # Convert to GPFA format first
    trial_segments = convert_trial_data_to_segments(trial_data, logger)
    
    # Use existing function
    return save_trial_segments_h5(trial_segments, output_path, logger)


def save_trial_data_mat(trial_data, mat_path, logger=None):
    """
    Save trial_data directly to MATLAB .mat format.
    
    Args:
        trial_data (dict): Dictionary containing 'df_trials_with_segments' DataFrame
        mat_path (str): Path for output .mat file
        logger (logging.Logger, optional): Logger instance
    
    Returns:
        str: Path to saved file
    """
    # Convert to GPFA format first
    trial_segments = convert_trial_data_to_segments(trial_data, logger)
    
    # Use existing function
    return save_trial_segments_mat(trial_segments, mat_path, logger)


def save_trial_segments_h5(trial_segments, output_path, logger=None):
    """
    Save trial segmented spike data to HDF5 format with MATLAB-compatible structure.
    
    Args:
        trial_segments (list): List of dictionaries, each containing:
            - 'trialId': unique trial identifier
            - 'spikes': 2D numpy array (neurons x time)
        output_path (str): Path for the output .h5 file
        logger (logging.Logger, optional): Logger instance
    
    Returns:
        str: Path to saved file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if logger:
        logger.info(f"Saving {len(trial_segments)} trial segments to {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Create groups for organized storage
        dat_group = f.create_group('dat')
        
        # Store metadata
        dat_group.attrs['num_trials'] = len(trial_segments)
        dat_group.attrs['creation_time'] = datetime.now().isoformat()
        
        for i, trial in enumerate(trial_segments):
            trial_group = dat_group.create_group(f'trial_{i:06d}')
            
            # Store trialId
            trial_group.create_dataset('trialId', data=str(trial['trialId']))
            
            # Store spikes array
            spikes = trial['spikes']
            trial_group.create_dataset('spikes', data=spikes, 
                                     compression='gzip', compression_opts=6)
            
            # Store metadata
            trial_group.attrs['neurons'] = spikes.shape[0]
            trial_group.attrs['timepoints'] = spikes.shape[1]
    
    if logger:
        logger.info(f"Successfully saved trial segments to {output_path}")
    
    return output_path


def load_trial_segments_h5(h5_path, logger=None):
    """
    Load trial segmented spike data from HDF5 format.
    
    Args:
        h5_path (str): Path to the .h5 file
        logger (logging.Logger, optional): Logger instance
    
    Returns:
        list: List of dictionaries with 'trialId' and 'spikes' fields
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    
    if logger:
        logger.info(f"Loading trial segments from {h5_path}")
    
    trial_segments = []
    
    with h5py.File(h5_path, 'r') as f:
        dat_group = f['dat']
        num_trials = dat_group.attrs['num_trials']
        
        if logger:
            logger.info(f"Loading {num_trials} trials")
        
        for i in range(num_trials):
            trial_group = dat_group[f'trial_{i:06d}']
            
            # Handle string decoding for trialId
            trial_id = trial_group['trialId'][()]
            if isinstance(trial_id, bytes):
                trial_id = trial_id.decode('utf-8')
            
            trial_data = {
                'trialId': trial_id,
                'spikes': trial_group['spikes'][:]
            }
            trial_segments.append(trial_data)
    
    if logger:
        logger.info(f"Successfully loaded {len(trial_segments)} trial segments")
    
    return trial_segments


# Replace the save_trial_segments_mat and h5_to_mat functions with these corrected versions:

# Replace the save_trial_segments_mat and h5_to_mat functions with these corrected versions:

# Replace the save_trial_segments_mat and h5_to_mat functions with these corrected versions:

def save_trial_segments_mat(trial_segments, mat_path, logger=None):
    """
    Directly save trial segments to MATLAB .mat format.
    
    Args:
        trial_segments (list): List of dictionaries with 'trialId' and 'spikes'
        mat_path (str): Path for output .mat file
        logger (logging.Logger, optional): Logger instance
    
    Returns:
        str: Path to saved file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(mat_path), exist_ok=True)
    
    if logger:
        logger.info(f"Saving {len(trial_segments)} trial segments directly to {mat_path}")
    
    # Create MATLAB-compatible struct array
    # Method: Create numpy structured array with proper data types
    dtype = [('trialId', 'f8'), ('spikes', 'O')]  # trialId as double, spikes as object
    dat = np.empty(len(trial_segments), dtype=dtype)
    
    for i, trial in enumerate(trial_segments):
        # Convert trialId to numeric (float64/double)
        trial_id = trial['trialId']
        if isinstance(trial_id, str):
            # Try to extract numeric part if it's a string
            try:
                trial_id = float(trial_id.split('_')[0])  # Handle cases like "123_0"
            except (ValueError, AttributeError):
                trial_id = float(i + 1)  # Fallback to index + 1
        
        dat[i]['trialId'] = float(trial_id)
        
        # Ensure spikes are double precision
        spikes = trial['spikes'].astype(np.float64)
        dat[i]['spikes'] = spikes
    
    # Save as .mat file - this will create a proper MATLAB struct array
    scipy.io.savemat(mat_path, {'dat': dat}, oned_as='column')
    
    if logger:
        logger.info(f"Successfully saved to MATLAB format: {mat_path}")
        logger.info(f"MATLAB structure: dat(1x{len(trial_segments)} struct) with fields 'trialId' (double) and 'spikes' (double)")
    
    return mat_path


def h5_to_mat(h5_path, mat_path=None, logger=None):
    """
    Convert HDF5 trial segments to MATLAB .mat format with dat structure.
    
    Args:
        h5_path (str): Path to input .h5 file
        mat_path (str, optional): Path for output .mat file. If None, uses h5_path with .mat extension
        logger (logging.Logger, optional): Logger instance
    
    Returns:
        str: Path to created .mat file
    """
    if mat_path is None:
        mat_path = os.path.splitext(h5_path)[0] + '.mat'
    
    if logger:
        logger.info(f"Converting {h5_path} to MATLAB format: {mat_path}")
    
    # Load data from HDF5
    trial_segments = load_trial_segments_h5(h5_path, logger)
    
    # Create MATLAB-compatible struct array with proper data types
    dtype = [('trialId', 'f8'), ('spikes', 'O')]  # trialId as double, spikes as object
    dat = np.empty(len(trial_segments), dtype=dtype)
    
    for i, trial in enumerate(trial_segments):
        # Convert trialId to numeric (float64/double)
        trial_id = trial['trialId']
        if isinstance(trial_id, str):
            # Try to extract numeric part if it's a string
            try:
                trial_id = float(trial_id.split('_')[0])  # Handle cases like "123_0"
            except (ValueError, AttributeError):
                trial_id = float(i + 1)  # Fallback to index + 1
        
        dat[i]['trialId'] = float(trial_id)
        
        # Ensure spikes are double precision
        spikes = trial['spikes'].astype(np.float64)
        dat[i]['spikes'] = spikes
    
    # Save as .mat file - this will create a proper MATLAB struct array
    scipy.io.savemat(mat_path, {'dat': dat}, oned_as='column')
    
    if logger:
        logger.info(f"Successfully converted to MATLAB format: {mat_path}")
        logger.info(f"MATLAB structure: dat(1x{len(trial_segments)} struct) with fields 'trialId' (double) and 'spikes' (double)")
    
    return mat_path


def convert_trial_data_to_segments(trial_data, logger=None):
    """
    Convert trial_data structure from extract_trial_segments_simple to GPFA-compatible format.
    
    Args:
        trial_data (dict): Dictionary containing 'df_trials_with_segments' DataFrame
        logger (logging.Logger, optional): Logger instance
    
    Returns:
        list: List of dictionaries with 'trialId' and 'spikes' fields for GPFA
    """
    if logger:
        logger.info("Converting trial_data to GPFA-compatible segments")
    
    df_trials = trial_data['df_trials_with_segments']
    trial_segments = []
    
    for idx, row in df_trials.iterrows():
        trial_id = row['trial_index']
        spike_segment = row['spike_segment']
        
        # Ensure trial_id is numeric
        if isinstance(trial_id, str):
            try:
                trial_id = float(trial_id)
            except ValueError:
                trial_id = float(idx + 1)  # Fallback to index + 1
        
        # Handle case where spike_segment might be a list of segments per trial
        if isinstance(spike_segment, list):
            # Multiple segments per trial
            for seg_idx, spikes in enumerate(spike_segment):
                # Ensure spikes are double precision
                spikes = spikes.astype(np.float64)
                trial_segments.append({
                    'trialId': trial_id + seg_idx * 0.1,  # e.g., 1.0, 1.1, 1.2 for segments
                    'spikes': spikes
                })
        else:
            # Single segment per trial
            # Ensure spikes are double precision
            spike_segment = spike_segment.astype(np.float64)
            trial_segments.append({
                'trialId': trial_id,
                'spikes': spike_segment
            })
    
    if logger:
        logger.info(f"Converted {len(df_trials)} trials to {len(trial_segments)} segments")
    
    return trial_segments

def save_trial_segments_pkl(trial_segments, pkl_path, logger=None):
    """
    Save trial segments to pickle format (for Python-only workflows).
    
    Args:
        trial_segments (list): List of dictionaries with 'trialId' and 'spikes'
        pkl_path (str): Path for output .pkl file
        logger (logging.Logger, optional): Logger instance
    
    Returns:
        str: Path to saved file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    
    if logger:
        logger.info(f"Saving {len(trial_segments)} trial segments to {pkl_path}")
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(trial_segments, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    if logger:
        logger.info(f"Successfully saved to pickle format: {pkl_path}")
    
    return pkl_path


def load_trial_segments_pkl(pkl_path, logger=None):
    """
    Load trial segments from pickle format.
    
    Args:
        pkl_path (str): Path to .pkl file
        logger (logging.Logger, optional): Logger instance
    
    Returns:
        list: List of dictionaries with 'trialId' and 'spikes'
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
    
    if logger:
        logger.info(f"Loading trial segments from {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        trial_segments = pickle.load(f)
    
    if logger:
        logger.info(f"Successfully loaded {len(trial_segments)} trial segments")
    
    return trial_segments


def verify_trial_segments_structure(trial_segments, logger=None):
    """
    Verify that trial segments have the correct structure for GPFA.
    
    Args:
        trial_segments (list): List of trial dictionaries
        logger (logging.Logger, optional): Logger instance
    
    Returns:
        bool: True if structure is valid
    """
    if not isinstance(trial_segments, list):
        if logger:
            logger.error("trial_segments must be a list")
        return False
    
    if len(trial_segments) == 0:
        if logger:
            logger.warning("trial_segments is empty")
        return True
    
    # Check first trial structure
    first_trial = trial_segments[0]
    
    required_fields = ['trialId', 'spikes']
    for field in required_fields:
        if field not in first_trial:
            if logger:
                logger.error(f"Missing required field: {field}")
            return False
    
    # Check spikes array
    spikes = first_trial['spikes']
    if not isinstance(spikes, np.ndarray):
        if logger:
            logger.error("'spikes' must be a numpy array")
        return False
    
    if spikes.ndim != 2:
        if logger:
            logger.error(f"'spikes' must be 2D array, got {spikes.ndim}D")
        return False
    
    # Verify consistency across trials
    expected_neurons = spikes.shape[0]
    
    for i, trial in enumerate(trial_segments):
        if 'spikes' not in trial or 'trialId' not in trial:
            if logger:
                logger.error(f"Trial {i} missing required fields")
            return False
        
        if trial['spikes'].shape[0] != expected_neurons:
            if logger:
                logger.error(f"Trial {i} has {trial['spikes'].shape[0]} neurons, expected {expected_neurons}")
            return False
    
    if logger:
        logger.info(f"Verified {len(trial_segments)} trials with {expected_neurons} neurons")
    
    return True