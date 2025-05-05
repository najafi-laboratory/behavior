# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:53:39 2025

@author: timst
"""
import gc
import os

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
