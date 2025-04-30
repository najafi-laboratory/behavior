#!/usr/bin/env python3

import os
import sys
import fitz
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from datetime import datetime
import random
from scipy.stats import sem
import psytrack as psy
import pickle
import DataIO
import DataIO_all
import DataIOPsyTrack
from data_extraction import *
import pandas as pd
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from pdf2image import convert_from_path

from utils import config
from utils import directories
from utils.util import get_figsize_from_pdf_spec
from registry.update import update_figure_registry
from report_builder.build_pdf import build_pdf_from_registry


from figure_modules.outcomes import plot_outcomes
from figure_modules.left_right_percentage import plot_left_right_percentage
from figure_modules.decision_time import plot_decision_time
from figure_modules.decision_time_opto import plot_decision_time_opto
from figure_modules.decision_time_kernel_opto import plot_decision_time_kernel_opto
from figure_modules.decision_time_kernel_opto_diff import plot_decision_time_kernel_opto_diff
from figure_modules.decision_time_box_outcome import plot_decision_time_box_outcome
from figure_modules.decision_time_violin_outcome import plot_decision_time_violin_outcome
from figure_modules.decision_time_violin_side import plot_decision_time_violin_side
# from figure_modules.performance_opto import plot_performance_opto



from figure_modules.psychometric_opto_avg import plot_psychometric_opto_avg
from figure_modules.psychometric_opto_residual_avg import plot_psychometric_opto_residual_avg

from figure_modules.psychometric_opto_epoch import plot_psychometric_opto_epoch
from figure_modules.psychometric_opto_epoch_residual import plot_psychometric_opto_epoch_residual
from figure_modules.licking_opto import plot_licking_opto
from figure_modules.licking_opto_avg import plot_licking_opto_avg

from glm.glm_hmm import get_glm_hmm
from glm.glm_hmm_summary import summarize_glm_hmm_model
# from glm.glm_hmm_analysis import 

from glm.glm_plot_state_weights import plot_state_weights
from glm.glm_plot_state_radar import plot_combined_behavioral_radar
from glm.glm_plot_state_occupancy_accuracy import plot_state_occupancy_and_accuracy
from glm.glm_plot_metadata import plot_model_metadata_box
from glm.glm_plot_state_transition_matrix import plot_state_transition_matrix
from glm.glm_plot_state_transition_network import plot_state_transition_network

from neural.neural_report import neural_report

import warnings
warnings.filterwarnings('ignore')

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
    

    
    session_data_path = directories.SESSION_DATA_PATH
    # figure_dir_local = config.FIGURE_DIR_LOCAL
    # output_dir_onedrive = config.OUTPUT_DIR_ONEDRIVE
    # output_dir_local = config.OUTPUT_DIR_LOCAL
    
    


    # last_day = '20241215'
    #subject_list = ['YH7', 'YH10', 'LG03', 'VT01', 'FN14' , 'LG04' , 'VT02' , 'VT03']
    # subject_list = ['LCHR_TS01', 'LCHR_TS02']
    # subject_list = ['LCHR_TS01', 'LCHR_TS02', 'LG08_TS03', 'LG09_TS04', 'LG11_TS05']
    # subject_list = ['LCHR_TS01_update']
    # subject_list = ['LCHR_TS02_update']
    # subject_list = ['LCHR_TS01_update', 'LCHR_TS02_update']
    # subject_list = ['LCHR_TS02']
    # subject_list = ['LG09_TS04']
    # subject_list = ['LG09_TS04_update']
    # subject_list = ['LCHR_TS01', 'LCHR_TS02']
    # subject_list = ['LCHR_TS01', 'LCHR_TS02', 'SCHR_TS06', 'SCHR_TS07']
    # subject_list = ['LCHR_TS01', 'LCHR_TS02', 'SCHR_TS06', 'SCHR_TS07', 'SCHR_TS08', 'SCHR_TS09']
    # subject_list = ['LCHR_TS01']
    # subject_list = ['LCHR_TS01_opto', 'LCHR_TS02_opto']; opto = 1
    # subject_list = ['SCHR_TS06', 'SCHR_TS07', 'SCHR_TS08', 'SCHR_TS09']
    # subject_list = ['SCHR_TS06_reg', 'SCHR_TS07_reg', 'SCHR_TS08_reg', 'SCHR_TS09_reg']
    # subject_list = ['SCHR_TS06_reg', 'SCHR_TS08_reg']
    # subject_list = ['SCHR_TS07_reg', 'SCHR_TS09_reg']

    # subject_list = ['SCHR_TS06_reg']
    # subject_list = ['SCHR_TS07_reg']
    # subject_list = ['SCHR_TS08_reg']
    # subject_list = ['SCHR_TS09_reg']
    # subject_list = ['SCHR_TS06_reg','SCHR_TS07_reg','SCHR_TS08_reg','SCHR_TS09_reg']
    
    # subject_list = ['LCHR_TS01']
    # subject_list = ['LCHR_TS02_reg']
    
    # subject_list = ['LCHR_TS01_opto', 'LCHR_TS02_opto']; opto = 1
    
    # subject_list = ['LCHR_TS01_opto']; opto = 1
    # subject_list = ['LCHR_TS02_opto']; opto = 1
    # subject_list = ['SCHR_TS06_opto']; opto = 1
    # subject_list = ['SCHR_TS07_opto']; opto = 1
    # subject_list = ['SCHR_TS08_opto']; opto = 1
    # subject_list = ['SCHR_TS09_opto']; opto = 1
    # subject_list = ['SCHR_TS06_opto','SCHR_TS07_opto','SCHR_TS08_opto','SCHR_TS09_opto']; opto = 1

    # subject_list = ['LCHR_TS01_update', 'LCHR_TS02_update']; opto = 1
# 



    # subject_list = ['LCHR_TS01_update', 'LCHR_TS01', 'LCHR_TS02', 'SCHR_TS06', 'SCHR_TS07', 'SCHR_TS08', 'SCHR_TS09']
    # subject_list = ['LCHR_TS01', 'LCHR_TS02', 'SCHR_TS06', 'SCHR_TS07', 'SCHR_TS08', 'SCHR_TS09']
    # subject_list = ['LCHR_TS02', 'SCHR_TS06', 'SCHR_TS07', 'SCHR_TS08', 'SCHR_TS09']
    # subject_list = ['LCHR_TS01']
    # subject_list = ['LCHR_TS02']
    # subject_list = ['SCHR_TS06']
    # subject_list = ['SCHR_TS07']
    # subject_list = ['SCHR_TS08']
    # subject_list = ['SCHR_TS09']
    # subject_list = ['TS03', 'YH24']
    subject_list = ['TS03']
    # subject_list = ['YH24']
    
    # for subject in subject_list:
    #     update_cache_from_mat_files(subject, config.paths['session_data'], 'result.json')
    # extract_data(subject_list, config.paths['session_data'])

    # session_configs = session_config_list_2AFC

    M = load_json_to_dict('result.json')
    
    
    print("Data loaded from JSON. Proceeding with analysis...")
    M = filter_sessions(M, config.session_config_list_2AFC)
    
    
    # Filter 
    new_M = []
    for subjectIdx in range(len(subject_list)):
        for subjectDataIdx in range(len(M)):
            if M[subjectDataIdx]['name'] == subject_list[subjectIdx]:
                # SubjectIndxList.append(subjectDataIdx)
                new_M.append(M[subjectDataIdx])
    
    M = new_M
    
    for subjectIdx in range(len(M)):
        subject_report = fitz.open()
        
        print(M[subjectIdx]['name'])
        
        subject = M[subjectIdx]['name']        
        
        output_path = config.paths['output_dir_local']
        os.makedirs(output_path, exist_ok=True)
        pdf_filename = os.path.join(output_path, f"{subject}_report.pdf") 
        
        with PdfPages(pdf_filename) as pdf:
            
            page_offset = 0
            # # Generate and register figure
            # meta = plot_outcomes(M[subjectIdx], config.session_config_list_2AFC, subjectIdx)
            # update_figure_registry(meta, config.session_config_list_2AFC)
            
            
            def create_grid_spec(
                grid_size=(4, 8),
                figsize=(30, 15),
                constrained_layout=True,
            ):
                layout_mode = 'constrained' if constrained_layout else None
                fig = plt.figure(layout=layout_mode, figsize=figsize)
                gs = gridspec.GridSpec(*grid_size, figure=fig)       
                return fig, gs
        
            def create_gs_subplot(
                fig,
                gs,
                position=(0, 0),
                span=(2, 4),
                adjust_margins=True,
            ):
                """
                Create a figure and an Axes using GridSpec at a specified grid position.
            
                Parameters:
                    grid_size (tuple): (nrows, ncols) for GridSpec
                    figsize (tuple): size of the figure
                    position (tuple): (row, col) where the Axes starts
                    span (tuple): (rowspan, colspan) for the Axes
                    constrained_layout (bool): whether to use constrained layout
                    adjust_margins (bool): if True, set margins and spacing to zero
            
                Returns:
                    fig (Figure): The created figure
                    ax (Axes): The subplot inserted at the specified location
                    gs (GridSpec): The GridSpec object used
                """   
                row, col = position
                rowspan, colspan = span
                ax = fig.add_subplot(gs[row:row+rowspan, col:col+colspan])
            
                if adjust_margins:
                    fig.subplots_adjust(left=0, right=0, top=0, bottom=0, wspace=0, hspace=0)  
            
                return fig, ax, gs
       
            def rasterize_pdf_to_axes(pdf_path, ax, dpi=300):
                """
                Rasterize the first page of a PDF file and display it on the given matplotlib Axes.
            
                Parameters:
                    pdf_path (str): Path to the PDF file (assumes one-page).
                    ax (matplotlib.axes.Axes): Axes to display the image on.
                    dpi (int): Resolution for rasterization.
            
                Returns:
                    PIL.Image.Image: The rasterized image (for further use if needed).
                """
                # Normalize the file path
                pdf_path = pdf_path.replace('\\', '/').replace('//', '/')
            
                if not os.path.exists(pdf_path):
                    raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
                # Convert first page of PDF to image
                images = convert_from_path(pdf_path, dpi=dpi)
                img = images[0]  # Take the first page
            
                ax.imshow(img)
                ax.axis('off')
            
                return img    
            
            def add_pdf_page(fig, pdf, fname, plt):
                pdf.savefig(fig)
                plt.close(fig)               
                roi_fig = fitz.open(fname)
                subject_report.insert_pdf(roi_fig)
                roi_fig.close()
                os.remove(fname)                 
                print(f"PDF saved to: {pdf_filename}")             
       
            # opto = 0
            
            debug_print = 0
            
            upload = 0
            
            cover = 0
            # performance = 1
            glm = 0
            lick_plots = 0
            RT = 0
            RT_session = 0
            psychometric = 0
            
            neural = 1
            use_random_num = 0
       
            if neural:
                neural_report(M[subjectIdx], config.session_config_list_2AFC, subjectIdx)
        
            if cover:
                fig, gs = create_grid_spec()
                   
                # SESSION OUTCOMES
                fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 0), span=(2, 4), adjust_margins=True)            
                fname = plot_outcomes(M[subjectIdx], config.session_config_list_2AFC, subjectIdx)                             
                rasterize_pdf_to_axes(fname, ax)
                
                # LEFT RIGHT PERCENTAGE
                fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 0), span=(2, 4), adjust_margins=True)            
                fname = plot_left_right_percentage(M[subjectIdx], config.session_config_list_2AFC, subjectIdx)    
                rasterize_pdf_to_axes(fname, ax)
        
                # fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 4), span=(2, 2), adjust_margins=True)            
                # fname = plot_decision_time_box_outcome(M[subjectIdx], config.session_config_list_2AFC, subjectIdx)    
                # rasterize_pdf_to_axes(fname, ax)    
        
                # REACTION TIMES
                fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 4), span=(2, 2), adjust_margins=True)            
                fname = plot_decision_time_violin_side(M[subjectIdx], config.session_config_list_2AFC, subjectIdx)    
                rasterize_pdf_to_axes(fname, ax)      
        
                fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 6), span=(2, 2), adjust_margins=True)            
                fname = plot_decision_time_violin_outcome(M[subjectIdx], config.session_config_list_2AFC, subjectIdx)    
                rasterize_pdf_to_axes(fname, ax)    
    
           
                # PSYCHOMETRIC
                fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 4), span=(2, 2), adjust_margins=True)     
                fname = plot_psychometric_opto_avg(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, -1)
                rasterize_pdf_to_axes(fname, ax)
                
                fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 6), span=(2, 2), adjust_margins=True)     
                fname = plot_psychometric_opto_residual_avg(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, -1)
                rasterize_pdf_to_axes(fname, ax)            
    
                add_pdf_page(fig, pdf, fname, plt)  
    
            # # DECISION TIMES
            # fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 6), span=(2, 2), adjust_margins=True)            
            # fname = plot_decision_time(M[subjectIdx], config.session_config_list_2AFC, subjectIdx)    
            # rasterize_pdf_to_axes(fname, ax)
            
            # # DECISION TIMES OPTO
            # fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 6), span=(2, 2), adjust_margins=True)            
            # fname = plot_decision_time_opto(M[subjectIdx], config.session_config_list_2AFC, subjectIdx)    
            # rasterize_pdf_to_axes(fname, ax)            
               

            if glm:
                # fig, gs = create_grid_spec()   
                # fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 0), span=(2, 8), adjust_margins=True)            
                # fname = plot_performance_opto(M[subjectIdx], config.session_config_list_2AFC, subjectIdx)                             
                # rasterize_pdf_to_axes(fname, ax)
    
                # add_pdf_page(fig, pdf, fname, plt)
                
                # glm_hmm, model_results, session_data_by_date, all_sessions_df = get_glm_hmm(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, train=True)
                model_output = get_glm_hmm(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, train=False)
                summary = summarize_glm_hmm_model(model_output)
                plot_state_weights(model_output)
                plot_combined_behavioral_radar(summary)
                plot_state_occupancy_and_accuracy(summary)
                plot_model_metadata_box(model_output, summary)
                plot_state_transition_matrix(model_output, summary)
                plot_state_transition_network(model_output, summary)
                
                
                # glm_interpret(M, config, subjectIdx, sessionIdx=-1, glm_hmm=glm_hmm, model_results=model_results, session_data_by_date=session_data_by_date, all_sessions_df=all_sessions_df)
            
            # fig, gs = create_grid_spec()                        
            
            # # # DECISION TIMES OPTO
            # # fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 0), span=(4, 4), adjust_margins=True)            
            # # fname = plot_decision_time_opto(M[subjectIdx], config.session_config_list_2AFC, subjectIdx)    
            # # rasterize_pdf_to_axes(fname, ax)                 
                        
            # # add_pdf_page(fig, pdf, fname, plt)
            
            # # fig, gs = create_grid_spec()  
                        
            # # fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 0), span=(4, 4), adjust_margins=True)            
            # # fname = plot_decision_time_kernel_opto(M[subjectIdx], config.session_config_list_2AFC, subjectIdx)    
            # # rasterize_pdf_to_axes(fname, ax)               
   
   
    
            if RT:
                fig, gs = create_grid_spec() 
  
                # DECISION TIMES - CONTROL
                fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 0), span=(2, 2), adjust_margins=True)            
                fname = plot_decision_time_kernel_opto(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, opto=False, side=None)    
                rasterize_pdf_to_axes(fname, ax)      
                
                fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 2), span=(2, 2), adjust_margins=True)            
                fname = plot_decision_time_kernel_opto(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, opto=False, side='both')    
                rasterize_pdf_to_axes(fname, ax)              
    
                fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 4), span=(2, 2), adjust_margins=True)            
                fname = plot_decision_time_kernel_opto(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, opto=True, side='left')    
                rasterize_pdf_to_axes(fname, ax)              
    
                fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 6), span=(2, 2), adjust_margins=True)            
                fname = plot_decision_time_kernel_opto(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, opto=True, side='right')    
                rasterize_pdf_to_axes(fname, ax)              
    
                # DECISION TIMES - OPTO
                fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 0), span=(2, 2), adjust_margins=True)            
                fname = plot_decision_time_kernel_opto(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, opto=True, side=None)    
                rasterize_pdf_to_axes(fname, ax)      
                                
                fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 2), span=(2, 2), adjust_margins=True)            
                fname = plot_decision_time_kernel_opto_diff(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, opto=True, side='both')    
                rasterize_pdf_to_axes(fname, ax)        
                
                fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 4), span=(2, 2), adjust_margins=True)            
                fname = plot_decision_time_kernel_opto_diff(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, opto=True, side='left')    
                rasterize_pdf_to_axes(fname, ax)   

                fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 6), span=(2, 2), adjust_margins=True)            
                fname = plot_decision_time_kernel_opto_diff(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, opto=True, side='right')    
                rasterize_pdf_to_axes(fname, ax)                   
                
                # fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 2), span=(2, 2), adjust_margins=True)            
                # fname = plot_decision_time_kernel_opto(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, opto=True, side='both')    
                # rasterize_pdf_to_axes(fname, ax)              
    
                # fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 4), span=(2, 2), adjust_margins=True)            
                # fname = plot_decision_time_kernel_opto(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, opto=True, side='left')    
                # rasterize_pdf_to_axes(fname, ax)              
    
                # fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 6), span=(2, 2), adjust_margins=True)            
                # fname = plot_decision_time_kernel_opto(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, opto=True, side='right')    
                # rasterize_pdf_to_axes(fname, ax)            
                            
                add_pdf_page(fig, pdf, fname, plt)
                
            if RT_session:
                fig, gs = create_grid_spec() 
                
                for sessionIdx in range(len(M[subjectIdx]['dates'])):
                    # DECISION TIMES - CONTROL
                    fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 0), span=(2, 2), adjust_margins=True)            
                    fname = plot_decision_time_kernel_opto(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, sessionIdx=sessionIdx,opto=False, side=None)    
                    rasterize_pdf_to_axes(fname, ax)      
                    
                    fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 2), span=(2, 2), adjust_margins=True)            
                    fname = plot_decision_time_kernel_opto(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, sessionIdx=sessionIdx, opto=False, side='both')    
                    rasterize_pdf_to_axes(fname, ax)              
        
                    fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 4), span=(2, 2), adjust_margins=True)            
                    fname = plot_decision_time_kernel_opto(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, sessionIdx=sessionIdx, opto=True, side='left')    
                    rasterize_pdf_to_axes(fname, ax)              
        
                    fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 6), span=(2, 2), adjust_margins=True)            
                    fname = plot_decision_time_kernel_opto(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, sessionIdx=sessionIdx, opto=True, side='right')    
                    rasterize_pdf_to_axes(fname, ax)              
        
                    # DECISION TIMES - OPTO
                    fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 0), span=(2, 2), adjust_margins=True)            
                    fname = plot_decision_time_kernel_opto(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, sessionIdx=sessionIdx, opto=True, side=None)    
                    rasterize_pdf_to_axes(fname, ax)      
                                    
                    fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 2), span=(2, 2), adjust_margins=True)            
                    fname = plot_decision_time_kernel_opto_diff(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, sessionIdx=sessionIdx, opto=True, side='both')    
                    rasterize_pdf_to_axes(fname, ax)        
                    
                    fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 4), span=(2, 2), adjust_margins=True)            
                    fname = plot_decision_time_kernel_opto_diff(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, sessionIdx=sessionIdx, opto=True, side='left')    
                    rasterize_pdf_to_axes(fname, ax)   
    
                    fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 6), span=(2, 2), adjust_margins=True)            
                    fname = plot_decision_time_kernel_opto_diff(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, sessionIdx=sessionIdx, opto=True, side='right')    
                    rasterize_pdf_to_axes(fname, ax)                      
                
                add_pdf_page(fig, pdf, fname, plt)

            if lick_plots:
                # LICKS OPTO              
                
                fnames = plot_licking_opto(M[subjectIdx], config.session_config_list_2AFC, subjectIdx)
                dates = M[subjectIdx]['dates']
                if not isinstance(dates, list):
                    dates = [dates]
                for idx, fname in enumerate(fnames):
                    if idx < len(dates):
                        date_str = f"{dates[idx]}" 
                    else:
                        date_str = f"{dates[0]} - {dates[-1]}" 
                    fig, gs = create_grid_spec()    
                    fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 0), span=(4, 8), adjust_margins=True)           
                    rasterize_pdf_to_axes(fname, ax)              
                    # Add figure-level title
                    fig.suptitle(date_str, fontsize=8)
                    # fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
                    add_pdf_page(fig, pdf, fname, plt)
  
    
                fig, gs = create_grid_spec()
                fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 0), span=(4, 8), adjust_margins=True)            
                fname = plot_licking_opto_avg(M[subjectIdx], config.session_config_list_2AFC, subjectIdx)    
                rasterize_pdf_to_axes(fname, ax)  
                if len(dates) > 1:
                    date_str = f"{dates[0]} - {dates[-1]}"
                else:
                    date_str = f"{dates[0]}"  
                fig.suptitle(date_str, fontsize=8)
                add_pdf_page(fig, pdf, fname, plt)
   
    
            if psychometric:       
                # OPTO PSYCHOMETRIC SESSIONS
                # Determine number of required pages
                total_sessions = len(M[subjectIdx]["dates"])             
                n_pages_1, grid_size, block_size = generate_paged_pdf_spec(
                    config.session_config_list_2AFC,
                    total_items=total_sessions,
                    grid_size=(4, 8),
                    fig_size=(30, 15),
                    block_size=(2, 2),
                )            
                
                
                fig, gs = create_grid_spec()
                page_idx_prev = 0
                for sessionIdx in range(len(M[subjectIdx]['dates'])):
                   
                    page_idx, row, col = assign_grid_position(sessionIdx, grid_size, block_size) 
                    if page_idx != page_idx_prev:
                        add_pdf_page(fig, pdf, fname, plt)
                        fig, gs = create_grid_spec()
                        page_idx_prev = page_idx
                    
                    if debug_print:
                        print(f"[page {page_idx}] row: {row}, col: {col}")
                   
                    fig, ax, gs = create_gs_subplot(fig, gs, position=(row, col), span=block_size, adjust_margins=True)            
                    fname = plot_psychometric_opto_epoch(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, sessionIdx)
                    img = rasterize_pdf_to_axes(fname, ax)     
    
    
                add_pdf_page(fig, pdf, fname, plt)  
          
            
                # OPTO PSYCHOMETRIC RESIDUALS SESSIONS
                # Determine number of required pages
                total_sessions = len(M[subjectIdx]["dates"])             
                n_pages_1, grid_size, block_size = generate_paged_pdf_spec(
                    config.session_config_list_2AFC,
                    total_items=total_sessions,
                    grid_size=(4, 8),
                    fig_size=(30, 15),
                    block_size=(2, 2),
                )    
                
                fig, gs = create_grid_spec()
                page_idx_prev = 0
                for sessionIdx in range(len(M[subjectIdx]['dates'])):
                   
                    page_idx, row, col = assign_grid_position(sessionIdx, grid_size, block_size) 
                    if page_idx != page_idx_prev:
                        add_pdf_page(fig, pdf, fname, plt)
                        fig, gs = create_grid_spec()
                        page_idx_prev = page_idx
                    
                    print(f"[page {page_idx}] row: {row}, col: {col}")
                   
                    fig, ax, gs = create_gs_subplot(fig, gs, position=(row, col), span=block_size, adjust_margins=True)            
                    fname = plot_psychometric_opto_epoch_residual(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, sessionIdx)
                    img = rasterize_pdf_to_axes(fname, ax)     
    
    
                add_pdf_page(fig, pdf, fname, plt)        
                # save to STROPER files
      
        
      
        
#%%      
        
      
        meta = plot_left_right_percentage(M[subjectIdx], config.session_config_list_2AFC, subjectIdx)


        # Determine number of required pages
        total_sessions = len(M[subjectIdx]["dates"])        
        
        n_pages_1, grid_size, block_size, base_page_key = generate_paged_pdf_spec(
            config.session_config_list_2AFC,
            base_page_key="pdf_pg_opto_psychometric",
            total_items=total_sessions,
            grid_size=(4, 8),
            fig_size=(30, 15),
            block_size=(2, 2),
        )
        
        page_offset += 1

        for sessionIdx in range(len(M[subjectIdx]['dates'])):
            meta = plot_psychometric_opto_epoch(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, sessionIdx)
            update_figure_registry(meta, config.session_config_list_2AFC)

            page_key, row, col = assign_grid_position(sessionIdx, grid_size, block_size, base_page_key)            
        
            meta['layout'].update({              
                'page_key': page_key,
                'row': row,
                'col': col,
                'rowspan': block_size[0],
                'colspan': block_size[1],
            })
        
            update_figure_registry(meta, config.session_config_list_2AFC)

        
        n_pages_2, grid_size, block_size, base_page_key = generate_paged_pdf_spec(
            config.session_config_list_2AFC,
            base_page_key="pdf_pg_opto_psychometric_residual",
            total_items=total_sessions,
            grid_size=(4, 8),
            fig_size=(30, 15),
            block_size=(2, 2),
        )

        page_offset += n_pages_1

        for sessionIdx in range(len(M[subjectIdx]['dates'])):
            meta = plot_psychometric_opto_epoch_residual(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, sessionIdx)
            update_figure_registry(meta, config.session_config_list_2AFC)

            page_key, row, col = assign_grid_position(sessionIdx, grid_size, block_size, base_page_key)
        
            meta['layout'].update({
                'page': page_offset,                
                'page_key': page_key,
                'row': row,
                'col': col,
                'rowspan': block_size[0],
                'colspan': block_size[1],
            })
        
            update_figure_registry(meta, config.session_config_list_2AFC)


        # plot_right_left_percentage.run(plt.subplot(gs[2, 0:3]), M[i])        

        # plot_psychometric_epoch.run([plt.subplot(gs[j, 3]) for j in range(3)], M[i])
       
        # plot_psychometric_post.run(plt.subplot(gs[0, 4]), M[i], start_from='std')
        # plot_psychometric_post.run(plt.subplot(gs[1, 4]), M[i], start_from='start_date')
        # # plot_psychometric_post.run(plt.subplot(gs[2, 4]), M[i], start_from='non_naive')
       
        # plot_decision_time.run(plt.subplot(gs[0, 5]), M[i], start_from='std')    
        
    # n_pages_3, grid_size, block_size, base_page_key = generate_paged_pdf_spec(
    #     config.session_config_list_2AFC,
    #     base_page_key="pdf_pg_licking",
    #     total_items=total_sessions,
    #     grid_size=(4, 8),
    #     fig_size=(30, 15),
    #     block_size=(4, 8),
    # )        
        
    page_offset += n_pages_2        
        
    if lick_plots:
    
        # for sessionIdx in range(len(M[subjectIdx]['dates'])):        
            # page_offset += 1
                
        meta, page_offset = plot_licking_opto(M[subjectIdx], config.session_config_list_2AFC, subjectIdx, sessionIdx, page_offset)
        for entry in meta:
            n_pages_3, grid_size, block_size, base_page_key = generate_paged_pdf_spec(
                config.session_config_list_2AFC,
                base_page_key="pdf_pg_licking",
                total_items=1,
                grid_size=(4, 8),
                fig_size=(30, 15),
                block_size=(4, 8),
            )               
            # update_figure_registry(entry, config.session_config_list_2AFC)
            
            entry['layout'].update({
                # 'page': page_offset,                
                'page_key': base_page_key,
                'row': row,
                'col': col,
                'rowspan': block_size[0],
                'colspan': block_size[1],
            })
            
            update_figure_registry(entry, config.session_config_list_2AFC)
    
    # for subjectIdx in range(len(M)):
        # Build report
        # build_pdf_from_registry(config.session_config_list_2AFC, subjectIdx)

    
    # subject_report = fitz.open()
    # subject_session_data = M[0]
    # subject_session_data = M
    
    
 #%%   



# 🔸 1. Modular Figure Code
# Each figure script should:

# Take in data + config (subject name, task variant, etc.)

# Output a saved figure

# Optionally register itself to a figure registry

# python
# Copy
# Edit
# def plot_stim_accuracy(data, save_path):
#     fig, ax = plt.subplots()
#     # ... plotting code ...
#     fig.savefig(save_path)
#     plt.close(fig)
# 🔸 2. Figure Metadata Tracking
# Maintain a JSON or SQLite file to keep track of what was generated:

# python
# Copy
# Edit
# figure_registry = {
#     'stim_accuracy_TS02': {
#         'path': 'figures/TS02_stim_accuracy.png',
#         'subject': 'TS02',
#         'type': 'stimulus_curve',
#         'caption': 'Accuracy across stimulus duration',
#     },
#     ...
# }
# 🔸 3. Report Assembly Layer
# Load figures and plug them into templates or layouts:

# For PDFs: use matplotlib.backends.backend_pdf.PdfPages, or reportlab for layout control

# For HTML: use jinja2 templates and generate shareable reports

# For interactive reports: try Panel, Dash, or Streamlit if useful

# 🔹 Other Options You Could Consider
# Using matplotlib.figure.Figure objects in memory if you don’t want to write to disk immediately

# Saving as SVG for vector-based layouts, useful for post-processing in Illustrator or embedding in LaTeX

# Interactive plots (Plotly, Bokeh) for web or review tools, though not always PDF-friendly

# Notebook automation with nbconvert or papermill to generate notebooks into reports with embedded figures
#%%
# figure_registry = {
#     'stim_accuracy_TS02': {
#         'path': 'figures/TS02_stim_accuracy.png',
#         'subject': 'TS02',
#         'type': 'stimulus_curve',
#         'caption': 'Accuracy across stimulus duration',
#     },
#     ...
# }

    # {
    # 'fig_id': 'stim_curve_TS02',
    # 'path': 'figures/TS02_stim_curve.png',
    # 'subject': 'TS02',
    # 'tags': ['performance', 'stimulus', 'accuracy'],
    # 'caption': 'Stimulus accuracy curve for TS02',
    # }
    
    
    
    # Report Modules

    # Use a figure registry + templating to build:
    
    # PDFs (e.g. with ReportLab, LaTeX, matplotlib.backends.backend_pdf, Pillow)
    
    # HTML (e.g. with Jinja2)
    
    # PowerPoint (e.g. with python-pptx)
    
    
#%%
    
    # Chemo
    #########
    
    # if subject_list[0] == 'YH7':
    #     chemo_sessions = ['0627' , '0701' , '0625' , '0623' , '0613' , '0611' , '0704' , '0712' , '0716' , '0725' , '0806' , '0809' , '0814' , '0828' , '0821' ,] #YH7
    # elif subject_list[0] == 'VT01':
    #     chemo_sessions = ['0618' , '0701' , '0704' , '0710' , '0712' , '0715' , '0718'] #VT01
    # elif subject_list[0] == 'LG03':
    #     chemo_sessions = ['0701' , '0704' , '0709' , '0712' , '0726', '0731', '0806' , '0809' , '0814' , '0828' , '0821'] #LG03
    # elif subject_list[0] == 'LG04':
    #     chemo_sessions = ['0712' , '0717' , '0725', '0731', '0806' , '0814' , '0828' , '0821'] #LG04
    # else:
    #     chemo_sessions = []
    # dates = subject_session_data['dates']
    # for ch in chemo_sessions:
    #     if '2024' + ch in dates:
    #         Chemo[dates.index('2024' + ch)] = 1
    #     else:
    #         Chemo[dates.index('2024' + ch)] = 0
    Chemo = np.zeros(subject_session_data['total_sessions'])
    M[0]['Chemo'] = Chemo
    
    
    # start date of non-naive
    NonNaive = {'LCHR_TS01_update': '20250302',
                'LCHR_TS01_opto': '20250302',
                'LCHR_TS02_opto': '20250302',
                'LCHR_TS01_reg': '20250302',
                'LCHR_TS02_reg': '20250302',                
                'LCHR_TS01': '20250302',
                'LCHR_TS02': '20250302',
                'LCHR_TS02_update': '20250302',
                'SCHR_TS06_reg': '20250302',
                'SCHR_TS07_reg': '20250302',
                'SCHR_TS08_reg': '20250302',
                'SCHR_TS09_reg': '20250302',
                'SCHR_TS06_opto': '20250302',
                'SCHR_TS07_opto': '20250302',
                'SCHR_TS08_opto': '20250302',
                'SCHR_TS09_opto': '20250302'}
    
    # Start date for averaging
    # StartDate = {'LCHR_TS01_update': '20241222',
    #             'LCHR_TS01': '20241222',
    #             'LCHR_TS02': '20241222',
    #             'LCHR_TS02_update': '20241222',
    #             'LG08_TS03': '20241228',
    #             'LG09_TS04': '20241227',
    #             'LG09_TS04_update': '20241225',
    #             'LG11_TS05': '20241216'}

    # MoveCorrectSpout - First Session 
    MoveCorrectSpoutStart = {'LCHR_TS01_update': '20250213',
                             'LCHR_TS01_opto': '20250213',
                             'LCHR_TS02_opto': '20250213',
                             'LCHR_TS01_reg': '20250302',
                             'LCHR_TS02_reg': '20250302',                             
                             'LCHR_TS01': '20250213',
                             'LCHR_TS02': '20250213',
                             'LCHR_TS02_update': '20250213',
                             'SCHR_TS06_reg': '20250213',
                             'SCHR_TS07_reg': '20250213',
                             'SCHR_TS08_reg': '20250213',
                             'SCHR_TS09_reg': '20250213',
                             'SCHR_TS06_opto': '20250302',
                             'SCHR_TS07_opto': '20250302',
                             'SCHR_TS08_opto': '20250302',
                             'SCHR_TS09_opto': '20250302'}
    
    # # Start date for averaging
    # StartDate = {'LCHR_TS01_update': '20250216',
    #              'LCHR_TS01_opto': '20250216',
    #              'LCHR_TS02_opto': '20250216',
    #             'LCHR_TS01': '20250204',
    #             'LCHR_TS02': '20250205',
    #             'LCHR_TS02_update': '20241226',
    #             'LG08_TS03': '2025122',
    #             'LG09_TS04': '20241226',
    #             'LG09_TS04_update': '20241230',
    #             'LG11_TS05': '20241226',
    #             'SCHR_TS06_reg': '20250202',
    #             'SCHR_TS07_reg': '20250202',
    #             'SCHR_TS08_reg': '20250215',
    #             'SCHR_TS09_reg': '20250215'}
    
    # Start date for averaging
    StartDate = {'LCHR_TS01_update': '20250302',
                 'LCHR_TS01_opto': '20250224',
                 'LCHR_TS02_opto': '20250224',
                'LCHR_TS01_reg': '20250302',
                'LCHR_TS02_reg': '20250302',                 
                'LCHR_TS01': '20250302',
                'LCHR_TS02': '20250302',
                'LCHR_TS02_update': '20250302',
                'SCHR_TS06_reg': '20250302',
                'SCHR_TS07_reg': '20250302',
                'SCHR_TS08_reg': '20250302',
                'SCHR_TS09_reg': '20250302',
                'SCHR_TS06_opto': '20250302',
                'SCHR_TS07_opto': '20250302',
                'SCHR_TS08_opto': '20250302',
                'SCHR_TS09_opto': '20250302'}    
    
    Sessions_Eye_Data = {'LCHR_TS01': ['20250109']
                         }
    
    
    
    # add start dates to session data
    for i in range(len(M)):
        M[i]['non_naive'] = NonNaive[M[i]['name']]
        M[i]['start_date'] = StartDate[M[i]['name']]
        M[i]['move_correct_spout'] = MoveCorrectSpoutStart[M[i]['name']]
    

    
 
    ##########
    # PsyTrack
    size = len(M)
    weights = [{}] * size  # List of empty dictionaries
    K = [[]] * size  # List of zeros 
    hyper = [{}] * size  # List of empty dictionaries
    optList = [''] * size  # List of empty strings
    new_M = [{}] * size  # List of empty dictionaries
    hyp = [{}] * size  # List of empty dictionaries
    evd = [[]] * size  # List of zeros 
    wMode = [[]] * size  # List of zeros 
    hess_info = [{}] * size  # List of empty dictionaries
    
    
    eval_psy_model = 0
    if eval_psy_model:
        for i in range(len(M)):
            # MoveCorrectSpout - First Session           
            # MCSS_idx = M[i]['dates'].index(MoveCorrectSpoutStart[M[i]['name']])
            # M[i]['move_correct_spout'] = MoveCorrectSpoutStart[M[i]['name']]
            
            M[i]['inputs'] = {}
    
            weights[i] = {'bias': 1}  # a special key
                        # 's1': 1,    # use only the first column of s1 from inputs
                        # 's2': 1}    # use only the first column of s2 from inputs
                        
            # weights.append({'bias': 1})  # a special key
            #             # 's1': 1,    # use only the first column of s1 from inputs
            #             # 's2': 1}    # use only the first column of s2 from inputs                    
    
            
            # It is often useful to have the total number of weights K in your model
            weights_at_i = weights[i]
            # K = np.sum([weights[i] for i in weights.keys()])
            K[i] = np.sum([weights_at_i[j] for j in weights_at_i.keys()])
            # K.append(np.sum([weights_at_i[j] for j in weights_at_i.keys()]))
    
            hyper[i] = {'sigInit': 2**4.,      # Set to a single, large value for all weights. Will not be optimized further.
                    'sigma': [2**-4.]*K[i],   # Each weight will have it's own sigma optimized, but all are initialized the same
                    'sigDay': M[i]['dayLength']}        # Indicates that session boundaries will be ignored in the optimization
            # hyper.append({'sigInit': 2**4.,      # Set to a single, large value for all weights. Will not be optimized further.
            #         'sigma': [2**-4.]*K[i],   # Each weight will have it's own sigma optimized, but all are initialized the same
            #         'sigDay': M[i]['dayLength']})        # Indicates that session boundaries will be ignored in the optimization
                
    
            optList[i] = ['sigma']
            # optList[i] = ['sigma']
    
            # new_M[i] = psy.trim(M[i], END=100000)  # trim dataset to first 10,000 trials
    
            # hyp[i], evd[i], wMode[i], hess_info[i] = psy.hyperOpt(new_M[i], hyper[i], weights[i], optList[i])
            
    eye_data = 0
    
    for i in range(len(M)):
        M[i]['Chemo'] = np.zeros(M[i]['total_sessions'])  
        
        
        
        if eye_data:
            fig = plt.figure(layout='constrained', figsize=(30, 15))
            gs = GridSpec(4, 6, figure=fig)
            
            subject = M[i]['subject']
            if subject in Sessions_Eye_Data:
                for date in Sessions_Eye_Data[subject]:
                    pupil_area_search_str = "area_per_frame"                
                    print(date)
                    videos_dir = 'D:\\PHD\\Projects\\Single Interval Discrimination\\Single_Interval_Discrimination-Tim-2025-01-11\\videos\\'
                    # eye_labeled_csv_path = 'D:\\PHD\\Projects\\Single Interval Discrimination\\Single_Interval_Discrimination-Tim-2025-01-11\\videos\\LCHR_TS01_2afc_20250109_cam0_run007_20250106_184048DLC_resnet50_Single_Interval_DiscriminationJan11shuffle1_100000_area_per_frame.csv'
                    # eye_df = pd.read_csv(eye_labeled_csv_path)
                    # print(eye_df.info())
                    # print(eye_df.head())
                    # camlog_path = 'D:\\PHD\\Projects\\Interval Discrimination\\Single Interval Discrimination\\data\\videos\\LCHR_TS01\\20250109\\LCHR_TS01_2afc_20250109_cam0_run007_20250106_184048.camlog'
                    # camlog_df = pd.read_csv(camlog_path)
                    
                    plot_eye_trials.run(plt.subplot(gs[0, 0:3]), M[i])
                # if isinstance(subject_session_data_copy[key], list) and len(subject_session_data_copy[key]) == len(dates):
                #     subject_session_data_copy[key] = subject_session_data_copy[key][start_idx:]         
            
            plt.suptitle(M[i]['subject'])
            fname = os.path.join(str(i).zfill(4)+'.pdf')
            fig.set_size_inches(30, 15)
            fig.savefig(fname, dpi=300)
            plt.close()
            roi_fig = fitz.open(fname)
            subject_report.insert_pdf(roi_fig)
            roi_fig.close()
            os.remove(fname)        
        
        pg1 = 1
        pg2 = 0
        pg3 = 0
        pg4 = 0
        
        if opto:
            pg5 = 1
            pg6 = 1
            pg7 = 1
        else:
            pg5 = 0
            pg6 = 0
            pg7 = 0
        
        pg6 = 0
        pg7 = 1
        
        # pg1 = 0
        # pg2 = 0
        # pg3 = 0
        
        subject = remove_substrings(subject_list[0], ['_opto', '_reg'])
        subject = flip_underscore_parts(subject)
        subject = lowercase_h(subject)
        
        ################################# pg 1        
        if pg1:
            fig = plt.figure(layout='constrained', figsize=(30, 15))
            gs = GridSpec(4, 6, figure=fig)
    
            # plot_complete_trials.run(plt.subplot(gs[0, 0:3]), M[i])
            plot_side_outcome_percentage_nomcs.run(plt.subplot(gs[0, 0:3]), M[i])
            plot_side_outcome_percentage.run(plt.subplot(gs[1, 0:3]), M[i])
            plot_right_left_percentage.run(plt.subplot(gs[2, 0:3]), M[i])        
    
            plot_psychometric_epoch.run([plt.subplot(gs[j, 3]) for j in range(3)], M[i])
           
            plot_psychometric_post.run(plt.subplot(gs[0, 4]), M[i], start_from='std')
            plot_psychometric_post.run(plt.subplot(gs[1, 4]), M[i], start_from='start_date')
            # plot_psychometric_post.run(plt.subplot(gs[2, 4]), M[i], start_from='non_naive')
           
            plot_decision_time.run(plt.subplot(gs[0, 5]), M[i], start_from='std')                  
            plot_decision_time.run(plt.subplot(gs[1, 5]), M[i], start_from='start_date')
            # plot_decision_time.run(plt.subplot(gs[2, 5]), M[i], start_from='non_naive')        
    
            plot_psytrack_bias.run(plt.subplot(gs[3, 0:3]), M[i])  
            plot_psytrack_performance.run(plt.subplot(gs[3, 3:6]), M[i])        
                  
            
            plt.suptitle(subject)
            fname = os.path.join(str(i).zfill(4)+'.pdf')
            fig.set_size_inches(30, 15)
            fig.savefig(fname, dpi=300)
            plt.close()
            roi_fig = fitz.open(fname)
            subject_report.insert_pdf(roi_fig)
            roi_fig.close()
            os.remove(fname)

        ################################# pg 2
        if pg2:
            fig = plt.figure(layout='constrained', figsize=(30, 15))
            gs = GridSpec(4, 6, figure=fig)
            
            
            plot_decision_time_side.run(plt.subplot(gs[0, 1]), M[i], start_from='std')                  
            plot_decision_time_isi.run(plt.subplot(gs[0, 2]), M[i], start_from='start_date')
            plot_decision_time_sessions.run(plt.subplot(gs[1:2, 0:6]), M[i], max_rt=700, plot_type='std', start_from='std')
            plot_decision_time_sessions.run(plt.subplot(gs[2:3, 0:6]), M[i], max_rt=700, plot_type='lick-side', start_from='std')
            
            # plot_decision_time_sessions.run(plt.subplot(gs[1:2, 0:6]), M[i], max_rt=700, plot_type='std', start_from='start_date')
            # plot_decision_time_sessions.run(plt.subplot(gs[2:3, 0:6]), M[i], max_rt=700, plot_type='lick-side', start_from='start_date')
                         
              
            plt.suptitle(subject)
            fname = os.path.join(str(i).zfill(4)+'.pdf')
            fig.set_size_inches(30, 15)
            fig.savefig(fname, dpi=300)
            plt.close()
            roi_fig = fitz.open(fname)
            subject_report.insert_pdf(roi_fig)
            roi_fig.close()
            os.remove(fname)
        
  
        ################################# pg 3
        if pg3:
            fig = plt.figure(layout='constrained', figsize=(30, 15))
            gs = GridSpec(4, 6, figure=fig)        
            
            
            plot_isi_distribution_epoch.run([plt.subplot(gs[j, 3]) for j in range(3)], M[i])
            
            plot_isi_distribution.run(plt.subplot(gs[0, 4]), M[i], start_from='std')
            # plot_isi_distribution.run(plt.subplot(gs[1, 4]), M[i], start_from='start_date')
            # plot_isi_distribution.run(plt.subplot(gs[2, 4]), M[i], start_from='non_naive')        
            
            plt.suptitle(subject)
            fname = os.path.join(str(i).zfill(4)+'.pdf')
            fig.set_size_inches(30, 15)
            fig.savefig(fname, dpi=300)
            plt.close()
            roi_fig = fitz.open(fname)
            subject_report.insert_pdf(roi_fig)
            roi_fig.close()
            os.remove(fname)           
    
         
        ################################# pg 4    
        if pg4:
            fig = plt.figure(layout='constrained', figsize=(30, 15))
            gs = GridSpec(4, 6, figure=fig)        
            
            plot_sdt_d_prime.run(plt.subplot(gs[0, 0:3]), M[i], start_from='std')
            plot_sdt_criterion.run(plt.subplot(gs[1, 0:3]), M[i], start_from='std')           
            
            plt.suptitle(subject)
            fname = os.path.join(str(i).zfill(4)+'.pdf')
            fig.set_size_inches(30, 15)
            fig.savefig(fname, dpi=300)
            plt.close()
            roi_fig = fitz.open(fname)
            subject_report.insert_pdf(roi_fig)
            roi_fig.close()
            os.remove(fname)        
            
        ################################# pg 5 
        if pg5:
            fig = plt.figure(layout='constrained', figsize=(30, 15))
            gs = GridSpec(4, 6, figure=fig)        
            
            # plot_side_outcome_percentage.run(plt.subplot(gs[1, 0:3]), M[i])
            
            # plot_sdt_d_prime.run(plt.subplot(gs[0, 0:3]), M[i], start_from='std')
            # plot_sdt_criterion.run(plt.subplot(gs[1, 0:3]), M[i], start_from='std')  
            plot_side_outcome_percentage_nomcs_opto.run(plt.subplot(gs[0, 0:6]), M[i])
            
            row = 1
            colmax = 5
            col = 0
            for session_num in range(M[i]['total_sessions']):      
                # print(f"row: {row}, col: {col}")
                # plot_psychometric_post_opto_epoch.run(plt.subplot(gs[1, 0]), M[i], start_from='std')
                plot_psychometric_post_opto_epoch.run(plt.subplot(gs[row, col]), M[i], session_num)
                col = col + 1
                if col > colmax:
                    row = row + 1
                    col = 0
                    
            plot_psychometric_post_opto_epoch.run(plt.subplot(gs[row, col]), M[i], -1)
            # plot_psychometric_post_opto.run(plt.subplot(gs[row, col]), M[i], start_from='std')
            # plot_decision_time_side_opto.run(plt.subplot(gs[1:2, 1:2]), M[i], start_from='std')     
            # plot_psychometric_post.run(plt.subplot(gs[1, 4]), M[i], start_from='start_date')            
            
            plt.suptitle(subject)
            fname = os.path.join(str(i).zfill(4)+'.pdf')
            fig.set_size_inches(30, 15)
            fig.savefig(fname, dpi=300)
            plt.close()
            roi_fig = fitz.open(fname)
            subject_report.insert_pdf(roi_fig)
            roi_fig.close()
            os.remove(fname)                            
   
        ################################# pg 6
        if pg6:
            fig = plt.figure(layout='constrained', figsize=(30, 15))
            gs = GridSpec(4, 6, figure=fig)        
            
            # plot_side_outcome_percentage.run(plt.subplot(gs[1, 0:3]), M[i])
            
            # plot_sdt_d_prime.run(plt.subplot(gs[0, 0:3]), M[i], start_from='std')
            # plot_sdt_criterion.run(plt.subplot(gs[1, 0:3]), M[i], start_from='std')  
            plot_side_outcome_percentage_nomcs_opto.run(plt.subplot(gs[0, 0:6]), M[i])
            
            row = 1
            colmax = 5
            col = 0
            for session_num in range(M[i]['total_sessions']):      
                # print(f"row: {row}, col: {col}")
                # plot_psychometric_post_opto_epoch.run(plt.subplot(gs[1, 0]), M[i], start_from='std')
                plot_psychometric_post_opto_epoch_residual.run(plt.subplot(gs[row, col]), M[i], session_num)
                col = col + 1
                if col > colmax:
                    row = row + 1
                    col = 0
                    
            plot_psychometric_post_opto_epoch_residual.run(plt.subplot(gs[row, col]), M[i], session_num=-1)
            # plot_psychometric_post_opto.run(plt.subplot(gs[row, col]), M[i], start_from='std')
            # plot_decision_time_side_opto.run(plt.subplot(gs[1:2, 1:2]), M[i], start_from='std')     
            # plot_psychometric_post.run(plt.subplot(gs[1, 4]), M[i], start_from='start_date')            
            
            plt.suptitle(subject)
            fname = os.path.join(str(i).zfill(4)+'.pdf')
            fig.set_size_inches(30, 15)
            fig.savefig(fname, dpi=300)
            plt.close()
            roi_fig = fitz.open(fname)
            subject_report.insert_pdf(roi_fig)
            roi_fig.close()
            os.remove(fname)      
   
        ################################# pg 7
        if pg7:
            fig = plt.figure(layout='constrained', figsize=(30, 15))
            gs = GridSpec(4, 6, figure=fig)        
            
            # plot_side_outcome_percentage.run(plt.subplot(gs[1, 0:3]), M[i])
            
            # plot_sdt_d_prime.run(plt.subplot(gs[0, 0:3]), M[i], start_from='std')
            # plot_sdt_criterion.run(plt.subplot(gs[1, 0:3]), M[i], start_from='std')  
            # GLM.run(plt.subplot(gs[0, 0:6]), M[i])
            
            row = 1
            colmax = 5
            col = 0
            for session_num in range(M[i]['total_sessions']):      
                # print(f"row: {row}, col: {col}")
                # plot_psychometric_post_opto_epoch.run(plt.subplot(gs[1, 0]), M[i], start_from='std')
                GLM.run(plt.subplot(gs[row, col]), M[i], session_num)
                col = col + 1
                if col > colmax:
                    row = row + 1
                    col = 0
                    
            # plot_psychometric_post_opto_epoch_residual.run(plt.subplot(gs[row, col]), M[i], session_num=-1)
            # plot_psychometric_post_opto.run(plt.subplot(gs[row, col]), M[i], start_from='std')
            # plot_decision_time_side_opto.run(plt.subplot(gs[1:2, 1:2]), M[i], start_from='std')     
            # plot_psychometric_post.run(plt.subplot(gs[1, 4]), M[i], start_from='start_date')            
            
            plt.suptitle(subject)
            fname = os.path.join(str(i).zfill(4)+'.pdf')
            fig.set_size_inches(30, 15)
            fig.savefig(fname, dpi=300)
            plt.close()
            roi_fig = fitz.open(fname)
            subject_report.insert_pdf(roi_fig)
            roi_fig.close()
            os.remove(fname)     
    
    # subject_report.save(output_dir_onedrive+subject_list[0]+'\\'+subject_list[0]+'_'+last_day+'_result_clean.pdf')
    
    
    
    
    if not use_random_num:
        num_str = ''
    else:
        num_str = '_'+num_str
    
    subject_folder_onedrive = os.path.join(output_dir_onedrive, subject)
    subject_folder_local = os.path.join(output_dir_local, subject)
    os.makedirs(subject_folder_onedrive, exist_ok=True)
    os.makedirs(subject_folder_local, exist_ok=True)

    if opto:
        pdf_filename = f"{subject}_single_interval_report_opto_{formatted_date}{num_str}.pdf"
        if upload:
            subject_report.save(os.path.join(subject_folder_onedrive, pdf_filename))
        subject_report.save(os.path.join(subject_folder_local, pdf_filename))
    else:
        pdf_filename = f"{subject}_single_interval_report_{formatted_date}{num_str}.pdf"
        if upload:
            subject_report.save(os.path.join(subject_folder_onedrive, pdf_filename))
        subject_report.save(os.path.join(subject_folder_local, pdf_filename))

    subject_report.close()
    for i in range(len(M)):
        plot_trial_outcomes.run(M[i],output_dir_onedrive, output_dir_local,formatted_date)
        #plot_category_each_session.run(M[i],output_dir_onedrive, output_dir_local,last_day)
        
        
        # import os
        # import sys
        # os.makedirs(output_figs_dir, exist_ok = True)
        # os.makedirs(output_imgs_dir, exist_ok = True)

    if lick_plots:
        for i in range(len(M)):
            # plot_single_trial_licking.run(M[i],output_dir_onedrive, output_dir_local)
            # plot_average_licking.run(M[i],output_dir_onedrive, output_dir_local)   
            
            plot_licking_opto.run(M[i],output_dir_onedrive, output_dir_local, upload)
            plot_licking_opto_avg.run(M[i],output_dir_onedrive, output_dir_local, upload)
            
            
            # plot_average_licking_opto.run(M[i],output_dir_onedrive, output_dir_local)  
            # plot_pooled_licking_opto.run(M[i],output_dir_onedrive, output_dir_local)





#%%
import warnings
warnings.filterwarnings('ignore')
if 0:
# if __name__ == "__main__":
    # Get the current date
    current_date = datetime.now()
    # Format the date as 'yyyymmdd'
    formatted_date = current_date.strftime('%Y%m%d')

    session_data_path = 'C:\\behavior\\session_data'    
    # session_data_path = 'C:\\localscratch\\behavior\\session_data'
    # session_data_path = 'D:\\PHD\\Projects\\Interval Discrimination\\data\\mat_files'    

    output_dir_onedrive = 'C:\\Users\\timst\\OneDrive - Georgia Institute of Technology\\Najafi_Lab\\2__Data_Analysis\\Behavior\\Single_Interval_Discrimination\\'
    output_dir_local = 'C:\\Users\\timst\\OneDrive - Georgia Institute of Technology\\Desktop\\PHD\\SingleIntervalDiscrimination\\FIGS\\'
    
    subject_list = ['LCHR_TS01']
    
    M = DataIOPsyTrack.run(subject_list , session_data_path)

    subject_report = fitz.open()
    
    # start date of non-naive
    NonNaive = {'LCHR_TS01_update': '20241222',
                'LCHR_TS01': '20241203',
                'LCHR_TS02': '20241203',
                'LCHR_TS02_update': '20241203',
                'LG08_TS03': '20241228',
                'LG09_TS04': '20241227',
                'LG09_TS04_update': '20241222',
                'LG11_TS05': '20241230'}
    
    # Start date for averaging
    StartDate = {'LCHR_TS01_update': '20241226',
                'LCHR_TS01': '20241226',
                'LCHR_TS02': '20241226',
                'LCHR_TS02_update': '20241226',
                'LG08_TS03': '20241226',
                'LG09_TS04': '20241226',
                'LG09_TS04_update': '20241230',
                'LG11_TS05': '20241226'}    

    # MoveCorrectSpout - First Session 
    MoveCorrectSpoutStart = {'LCHR_TS01_update': '20241222',
                             'LCHR_TS01': '20241214',
                             'LCHR_TS02': '20241214',
                             'LCHR_TS02_update': '20241214',
                             'LG08_TS03': '20241215',
                             'LG09_TS04': '20241221',
                             'LG09_TS04_update': '20241225',
                             'LG11_TS05': '20241218'}
    
    # add start dates to session data
    for i in range(len(M)):
        M[i]['non_naive'] = NonNaive[M[i]['name']]
        M[i]['start_date'] = StartDate[M[i]['name']]
        M[i]['move_correct_spout'] = MoveCorrectSpoutStart[M[i]['name']]
        
    for i in range(len(M)):
        M[i]['Chemo'] = np.zeros(M[i]['total_sessions'])  
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 6, figure=fig)

        
        plot_sdt_d_prime.run(plt.subplot(gs[0, 0:3]), M[i], start_from='std')
        plot_sdt_criterion.run(plt.subplot(gs[1, 0:3]), M[i], start_from='std')
        # plot_complete_trials.run(plt.subplot(gs[0, 0:3]), M[i])
        # plot_side_outcome_percentage.run(plt.subplot(gs[1, 0:3]), M[i])
        # plot_right_left_percentage.run(plt.subplot(gs[2, 0:3]), M[i])        

        # plot_psychometric_epoch.run([plt.subplot(gs[j, 3]) for j in range(3)], M[i])
       
        # plot_psychometric_post.run(plt.subplot(gs[0, 4]), M[i], start_from='std')
        # plot_psychometric_post.run(plt.subplot(gs[1, 4]), M[i], start_from='start_date')
        # plot_psychometric_post.run(plt.subplot(gs[2, 4]), M[i], start_from='non_naive')
       
        # plot_decision_time.run(plt.subplot(gs[0, 5]), M[i], start_from='std')                  
        # plot_decision_time.run(plt.subplot(gs[1, 5]), M[i], start_from='start_date')
        # plot_decision_time.run(plt.subplot(gs[2, 5]), M[i], start_from='non_naive')        

        # plot_psytrack_bias.run(plt.subplot(gs[3, 0:3]), M[i])  
        # plot_psytrack_performance.run(plt.subplot(gs[3, 3:6]), M[i])        
              
        
        plt.suptitle(subject)
        fname = os.path.join(str(i).zfill(4)+'.pdf')
        fig.set_size_inches(30, 15)
        fig.savefig(fname, dpi=300)
        plt.close()
        roi_fig = fitz.open(fname)
        subject_report.insert_pdf(roi_fig)
        roi_fig.close()
        os.remove(fname)   

    subject_report.save(output_dir_onedrive+'single_interval_report'+'_'+formatted_date+'.pdf')
    subject_report.save(output_dir_local+'single_interval_report'+'_'+formatted_date+'.pdf')
    subject_report.close()





#%%

if 0:

    session_data_path = 'C:\\behavior\\session_data\\SampleRat\\'
    # subject_list = ['SampleRat']
    # path = session_data_path + '\\' + subject_list
    
    # Extract premade dataset from npz
    D = np.load(session_data_path + 'sampleRatData.npz', allow_pickle=True)['D'].item()
    
    print("The keys of the dict for this example animal:\\n   ", list(D.keys()))
    
    print("The shape of y:   ", D['y'].shape)
    print("The number of trials:   N =", D['y'].shape[0])
    print("The unique entries of y:   ", np.unique(D['y']))
    
    weights = {'bias': 1,  # a special key
                's1': 1,    # use only the first column of s1 from inputs
                's2': 1}    # use only the first column of s2 from inputs
    
    # It is often useful to have the total number of weights K in your model
    K = np.sum([weights[i] for i in weights.keys()])
    
    
    hyper= {'sigInit': 2**4.,      # Set to a single, large value for all weights. Will not be optimized further.
            'sigma': [2**-4.]*K,   # Each weight will have it's own sigma optimized, but all are initialized the same
            'sigDay': None}        # Indicates that session boundaries will be ignored in the optimization
    
    optList = ['sigma']
    
    new_D = psy.trim(D, END=10000)  # trim dataset to first 10,000 trials
    
    hyp, evd, wMode, hess_info = psy.hyperOpt(new_D, hyper, weights, optList)
    
    fig = psy.plot_weights(wMode, weights)
    
    fig = psy.plot_weights(wMode, weights, days=new_D["dayLength"], errorbar=hess_info["W_std"])
    
    fig_perf = psy.plot_performance(new_D)
    fig_bias = psy.plot_bias(new_D)



#%%
import DataIOPsyTrack

session_data_path = 'C:\\behavior\\session_data\\SampleRat\\'
# subject_list = ['SampleRat']
# path = session_data_path + '\\' + subject_list

# Extract premade dataset from npz
# D = np.load(session_data_path + 'sampleRatData.npz', allow_pickle=True)['D'].item()

session_data_path = 'C:\\behavior\\session_data'
subject_list = ['LCHR_TS01_update']

# M = DataIOPsyTrack.run(subject_list , session_data_path)

M[0]['inputs'] = {}

weights = {'bias': 1}  # a special key
            # 's1': 1,    # use only the first column of s1 from inputs
            # 's2': 1}    # use only the first column of s2 from inputs

# It is often useful to have the total number of weights K in your model
K = np.sum([weights[i] for i in weights.keys()])

hyper= {'sigInit': 2**4.,      # Set to a single, large value for all weights. Will not be optimized further.
        'sigma': [2**-4.]*K,   # Each weight will have it's own sigma optimized, but all are initialized the same
        'sigDay': M[0]['dayLength']}        # Indicates that session boundaries will be ignored in the optimization

optList = ['sigma']

new_M[0] = psy.trim(M[0], END=10000)  # trim dataset to first 10,000 trials

hyp, evd, wMode, hess_info = psy.hyperOpt(new_M[0], hyper, weights, optList)

#%%

fig = psy.plot_weights(wMode, weights)

fig = psy.plot_weights(wMode, weights, days=new_M[0]["dayLength"], errorbar=hess_info["W_std"])

fig_perf = psy.plot_performance(new_M[0])
# Access all axes in the figure using fig.get_axes()
axes = fig_perf.get_axes()
# Modify the properties of the first subplot (axs[0])
axes[0].set_ylabel("Choice Accuracy")



fig_bias = psy.plot_bias(new_M[0])

# Access all axes in the figure using fig.get_axes()
axes = fig_bias.get_axes()
# Modify the properties of the first subplot (axs[0])
axes[0].set_yticklabels(['Left', 0, 'Right'])

# axes.yticks([-0.5, 0.5], ['Left', 'Right'])


# Modify the properties of the first subplot (axs[0])
# axes[0].set_title("Top Plot")
# axes[0].set_ylabel("Y-axis for top plot")

# Modify the properties of the second subplot (axs[1])
# axes[1].set_title("Bottom Plot")
# axes[1].set_ylabel("Y-axis for bottom plot")

#%%
import warnings
warnings.filterwarnings('ignore')
if __name__ == "__main__":
    # Get the current date
    current_date = datetime.now()
    # Format the date as 'yyyymmdd'
    formatted_date = current_date.strftime('%Y%m%d')
    
    
    session_data_path = 'C:\\behavior\\session_data'
    # session_data_path = 'C:\\localscratch\\behavior\\session_data'
    # output_dir_onedrive = './figures/'
    output_dir_onedrive = 'C:\\Users\\timst\\OneDrive - Georgia Institute of Technology\\Najafi_Lab\\0_Data_analysis\\Behavior\\Single_Interval_Discrimination\\'
    # output_dir_local = './figures/'
    output_dir_local = 'C:\\Users\\timst\\OneDrive - Georgia Institute of Technology\\Desktop\\PHD\\SingleIntervalDiscrimination\\FIGS\\'
    # last_day = '20241215'
    #subject_list = ['YH7', 'YH10', 'LG03', 'VT01', 'FN14' , 'LG04' , 'VT02' , 'VT03']
    # subject_list = ['LCHR_TS01', 'LCHR_TS02']
    subject_list = ['LCHR_TS01', 'LCHR_TS02', 'LG08_TS03', 'LG11_TS05', 'LG09_TS04']

    session_data = DataIO.run(subject_list , session_data_path)
    

    subject_report = fitz.open()
    subject_session_data = session_data[0]
    # subject_session_data = session_data
    
    #########
    Chemo = np.zeros(subject_session_data['total_sessions'])
    if subject_list[0] == 'YH7':
        chemo_sessions = ['0627' , '0701' , '0625' , '0623' , '0613' , '0611' , '0704' , '0712' , '0716' , '0725' , '0806' , '0809' , '0814' , '0828' , '0821' ,] #YH7
    elif subject_list[0] == 'VT01':
        chemo_sessions = ['0618' , '0701' , '0704' , '0710' , '0712' , '0715' , '0718'] #VT01
    elif subject_list[0] == 'LG03':
        chemo_sessions = ['0701' , '0704' , '0709' , '0712' , '0726', '0731', '0806' , '0809' , '0814' , '0828' , '0821'] #LG03
    elif subject_list[0] == 'LG04':
        chemo_sessions = ['0712' , '0717' , '0725', '0731', '0806' , '0814' , '0828' , '0821'] #LG04
    else:
        chemo_sessions = []
    dates = subject_session_data['dates']
    for ch in chemo_sessions:
        if '2024' + ch in dates:
            Chemo[dates.index('2024' + ch)] = 1
        else:
            Chemo[dates.index('2024' + ch)] = 0
    
    session_data[0]['Chemo'] = Chemo
    ##########
    
    for i in range(len(session_data)):
        session_data[i]['Chemo'] = np.zeros(session_data[i]['total_sessions'])  
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 6, figure=fig)
        # plot_outcome.run(plt.subplot(gs[0, 0:3]), session_data[i])
        plot_complete_trials.run(plt.subplot(gs[0, 0:3]), session_data[i])
        # plot_early_lick_outcome.run(plt.subplot(gs[3, 3:5]), session_data[i])
        plot_psychometric_post.run(plt.subplot(gs[2, 2]), session_data[i])
        # plot_psychometric_percep.run(plt.subplot(gs[3, 2]), session_data[i])
        plot_psychometric_epoch.run([plt.subplot(gs[j, 3]) for j in range(3)], session_data[i])
        plot_reaction_time.run(plt.subplot(gs[0, 4]), session_data[i])
        # plot_reaction_outcome.run(plt.subplot(gs[0, 5]), session_data[i])
        
        # plot_reaction_time.run(plt.subplot(gs[1, 4]), session_data[i])
        # plot_decision_time.run(plt.subplot(gs[1, 4]), session_data[i])
        # plot_decision_outcome.run(plt.subplot(gs[1, 5]), session_data[i])
                    #plot_strategy.run(plt.subplot(gs[2, 5]), session_data[i])
        # plot_decision_time_isi.run(plt.subplot(gs[2, 4]), session_data[i])
        # plot_reaction_time_isi.run(plt.subplot(gs[2, 5]), session_data[i])
                    # plot_short_long_percentage.run(plt.subplot(gs[2, 0:2]), session_data[i])
        plot_side_outcome_percentage.run(plt.subplot(gs[1, 0:3]), session_data[i])
        plot_right_left_percentage.run(plt.subplot(gs[2, 0:2]), session_data[i])
        
        
        
        plt.suptitle(session_data[i]['subject'])
        fname = os.path.join(str(i).zfill(4)+'.pdf')
        fig.set_size_inches(30, 15)
        fig.savefig(fname, dpi=300)
        plt.close()
        roi_fig = fitz.open(fname)
        subject_report.insert_pdf(roi_fig)
        roi_fig.close()
        os.remove(fname)
    # subject_report.save(output_dir_onedrive+subject_list[0]+'\\'+subject_list[0]+'_'+last_day+'_result_clean.pdf')
    subject_report.save(output_dir_onedrive+'single_interval_report'+'_'+formatted_date+'.pdf')
    subject_report.save(output_dir_local+'single_interval_report'+'_'+formatted_date+'.pdf')
    subject_report.close()
    for i in range(len(session_data)):
        plot_trial_outcomes.run(session_data[i],output_dir_onedrive, output_dir_local,formatted_date)
        #plot_category_each_session.run(session_data[i],output_dir_onedrive, output_dir_local,last_day)
        
        
        # import os
        # import sys
        # os.makedirs(output_figs_dir, exist_ok = True)
        # os.makedirs(output_imgs_dir, exist_ok = True)



#%%
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 140

import psytrack as psy

seed = 31
num_weights = 4
num_trials = 5000
hyper = {'sigma'   : 2**np.array([-4.0,-5.0,-6.0,-7.0]),
         'sigInit' : 2**np.array([ 0.0, 0.0, 0.0, 0.0])}

# Simulate
simData = psy.generateSim(K=num_weights, N=num_trials, hyper=hyper,
                          boundary=6.0, iterations=1, seed=seed, savePath=None)

# Plot
psy.plot_weights(simData['W'].T);
plt.ylim(-3.6,3.6);

rec = psy.recoverSim(simData)

psy.plot_weights(rec['wMode'], errorbar=rec["hess_info"]["W_std"])
plt.plot(simData['W'], c="k", ls="-", alpha=0.5, lw=0.75, zorder=0)
plt.ylim(-3.6,3.6);

true_sigma = np.log2(rec['input']['sigma'])
avg_sigma = np.log2(rec['hyp']['sigma'])
err_sigma = rec['hess_info']['hyp_std']

plt.figure(figsize=(2,2))
colors = np.unique(list(psy.COLORS.values()))
for i in range(num_weights):
    plt.plot(i, true_sigma[i], color="black", marker="_", markersize=12, zorder=0)
    plt.errorbar([i], avg_sigma[i], yerr=2*err_sigma[i], color=colors[i], lw=1, marker='o', markersize=5)

plt.xticks([0,1,2,3]); plt.yticks(np.arange(-8,-2))
plt.gca().set_xticklabels([r"$\\sigma_1$", r"$\\sigma_2$", r"$\\sigma_3$", r"$\\sigma_4$"])
plt.xlim(-0.5,3.5); plt.ylim(-7.5,-3.5)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.ylabel(r"$\\log_2(\\sigma)$");



#%%

import warnings
warnings.filterwarnings('ignore')
if __name__ == "__main__":

    subject_report = fitz.open()

    
    for i in range(len(session_data)):
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 6, figure=fig)
        
        strategy.run(plt.subplot(gs[0, 0]), session_data[i])
        count_short_long.run(plt.subplot(gs[2, 0]), plt.subplot(gs[1, 0]), session_data[i])
        count_psychometric_curve.run(plt.subplot(gs[0, 1]), plt.subplot(gs[1, 1]),plt.subplot(gs[2, 1]),plt.subplot(gs[3, 1]),session_data[i])
        count_isi_flash.run(plt.subplot(gs[0, 2]), plt.subplot(gs[1, 2]),plt.subplot(gs[2, 2]),plt.subplot(gs[3, 2]), session_data[i])
        strategy_epoch.run([plt.subplot(gs[j, 3]) for j in range(4)], session_data[i])
        plt.suptitle(session_data[i]['subject']+' count strategy analysis')
        fname = os.path.join(str(i).zfill(4)+'.pdf')
        fig.set_size_inches(30, 15)
        fig.savefig(fname, dpi=300)
        plt.close()
        roi_fig = fitz.open(fname)
        subject_report.insert_pdf(roi_fig)
        roi_fig.close()
        os.remove(fname)
    subject_report.save(output_dir_onedrive+subject_list[0]+'\\'+subject_list[0]+'_'+last_day+'_count_strategy_clean.pdf')
    subject_report.close()
    
    
#%%

import warnings
warnings.filterwarnings('ignore')
if __name__ == "__main__":

    subject_report = fitz.open()

    
    for i in range(len(session_data)):
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 6, figure=fig)
        plot_decision_time.run(plt.subplot(gs[0, 0]), session_data[i])
        count_isi_decision_time.run(plt.subplot(gs[0, 2]), plt.subplot(gs[1, 2]),plt.subplot(gs[2, 2]),plt.subplot(gs[3, 2]), session_data[i])
        count_flash_decision_time.run(plt.subplot(gs[0, 1]), plt.subplot(gs[1, 1]),plt.subplot(gs[2, 1]),plt.subplot(gs[3, 1]), session_data[i])
        decision_time_dist.run([plt.subplot(gs[j, 3]) for j in range(4)], session_data[i])
        
        plt.suptitle(session_data[i]['subject']+' desicion time analysis')
        fname = os.path.join(str(i).zfill(4)+'.pdf')
        fig.set_size_inches(30, 15)
        fig.savefig(fname, dpi=300)
        plt.close()
        roi_fig = fitz.open(fname)
        subject_report.insert_pdf(roi_fig)
        roi_fig.close()
        os.remove(fname)
    subject_report.save(output_dir_onedrive+subject_list[0]+'\\'+subject_list[0]+'_'+last_day+'_count_decision_time_clean.pdf')
    subject_report.close()

#%%

import warnings
warnings.filterwarnings('ignore')
if __name__ == "__main__":
    session_data_path = 'C:\\Users\\Sana\\OneDrive\\Desktop\\PHD\\IntervalDiscrimination\\Data\\20240805\\'
    output_dir_onedrive = 'C:\\Users\\Sana\\OneDrive\\Desktop\\PHD\\IntervalDiscrimination\\FIGs\\20240805\\'
    output_dir_local = 'C:\\Users\\Sana\\OneDrive\\Desktop\\PHD\\IntervalDiscrimination\\FIGs\\20240805\\'
    last_day = '20240805'
    #subject_list = ['YH7', 'YH10', 'LG03', 'VT01', 'FN14' , 'LG04' , 'VT02' , 'VT03']
    subject_list = ['VT03']

    session_data = DataIO_all.run(subject_list , session_data_path)
    

    subject_report = fitz.open()
    subject_session_data = session_data[0]
    
    #########
    Chemo = np.zeros(subject_session_data['total_sessions'])
    if subject_list[0] == 'YH7':
        chemo_sessions = ['0627' , '0701' , '0625' , '0623' , '0613' , '0611' , '0704' , '0712' , '0716' , '0725', '0806' , '0809' , '0814'] #YH7
    elif subject_list[0] == 'VT01':
        chemo_sessions = ['0618' , '0701' , '0704' , '0710' , '0712' , '0715' , '0718'] #VT01
    elif subject_list[0] == 'LG03':
        chemo_sessions = ['0701' , '0704' , '0709' , '0712' , '0726', '0806' , '0809' , '0814'] #LG03
    elif subject_list[0] == 'LG04':
        chemo_sessions = ['0712' , '0717' , '0725', '0806' , '0814'] #LG04
    else:
        chemo_sessions = []
    dates = subject_session_data['dates']
    for ch in chemo_sessions:
        if '2024' + ch in dates:
            Chemo[dates.index('2024' + ch)] = 1
    session_data[0]['Chemo'] = Chemo
    ##########
    
    for i in range(len(session_data)):
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 6, figure=fig)
        plot_outcome.run(plt.subplot(gs[0, 0:3]), session_data[i])
        plot_complete_trials.run(plt.subplot(gs[1, 0:3]), session_data[i])
        plot_early_lick_outcome.run(plt.subplot(gs[3, 3:5]), session_data[i])
        plot_psychometric_post.run(plt.subplot(gs[2, 2]), session_data[i])
        plot_psychometric_percep.run(plt.subplot(gs[3, 2]), session_data[i])
        plot_psychometric_epoch.run([plt.subplot(gs[j, 3]) for j in range(3)], session_data[i])
        plot_reaction_time.run(plt.subplot(gs[0, 4]), session_data[i])
        plot_reaction_outcome.run(plt.subplot(gs[0, 5]), session_data[i])
        plot_decision_time.run(plt.subplot(gs[1, 4]), session_data[i])
        plot_decision_outcome.run(plt.subplot(gs[1, 5]), session_data[i])
        #plot_strategy.run(plt.subplot(gs[2, 5]), session_data[i])
        plot_decision_time_isi.run(plt.subplot(gs[2, 4]), session_data[i])
        plot_reaction_time_isi.run(plt.subplot(gs[2, 5]), session_data[i])
        plot_short_long_percentage.run(plt.subplot(gs[2, 0:2]), session_data[i])
        plot_right_left_percentage.run(plt.subplot(gs[3, 0:2]), session_data[i])
        plt.suptitle(session_data[i]['subject'])
        fname = os.path.join(str(i).zfill(4)+'.pdf')
        fig.set_size_inches(30, 15)
        fig.savefig(fname, dpi=300)
        plt.close()
        roi_fig = fitz.open(fname)
        subject_report.insert_pdf(roi_fig)
        roi_fig.close()
        os.remove(fname)
    subject_report.save(output_dir_onedrive+subject_list[0]+'\\'+subject_list[0]+'_'+last_day+'_result_all.pdf')
    subject_report.close()

#%%

import warnings
warnings.filterwarnings('ignore')
if __name__ == "__main__":

    subject_report = fitz.open()

    
    for i in range(len(session_data)):
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 6, figure=fig)
        
        strategy.run(plt.subplot(gs[0, 0]), session_data[i])
        count_short_long.run(plt.subplot(gs[2, 0]), plt.subplot(gs[1, 0]), session_data[i])
        count_psychometric_curve.run(plt.subplot(gs[0, 1]), plt.subplot(gs[1, 1]),plt.subplot(gs[2, 1]),plt.subplot(gs[3, 1]),session_data[i])
        count_isi_flash.run(plt.subplot(gs[0, 2]), plt.subplot(gs[1, 2]),plt.subplot(gs[2, 2]),plt.subplot(gs[3, 2]), session_data[i])
        count_isi_decision_time.run(plt.subplot(gs[0, 3]), plt.subplot(gs[1, 3]),plt.subplot(gs[2, 3]),plt.subplot(gs[3, 3]), session_data[i])
        count_flash_decision_time.run1(plt.subplot(gs[0, 4]), plt.subplot(gs[1, 4]),plt.subplot(gs[2, 4]),plt.subplot(gs[3, 4]), session_data[i])
        strategy_epoch.run([plt.subplot(gs[j, 5]) for j in range(4)], session_data[i])
        
        plt.suptitle(session_data[i]['subject']+' count strategy analysis')
        fname = os.path.join(str(i).zfill(4)+'.pdf')
        fig.set_size_inches(30, 15)
        fig.savefig(fname, dpi=300)
        plt.close()
        roi_fig = fitz.open(fname)
        subject_report.insert_pdf(roi_fig)
        roi_fig.close()
        os.remove(fname)
    subject_report.save(output_dir_onedrive+subject_list[0]+'\\'+subject_list[0]+'_'+last_day+'_count_strategy_all.pdf')
    subject_report.close()
    
#%%

import warnings
warnings.filterwarnings('ignore')
import os
import fitz
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import DataIO
if __name__ == "__main__":
    
    session_data_path = 'C:\\Users\\Sana\\OneDrive\\Desktop\\PHD\\IntervalDiscrimination\\Data\\20240819\\'
    subject_list = ['YH7', 'YH10', 'LG03', 'VT01', 'FN14' , 'LG04' , 'VT02' , 'VT03']
    subject_list = ['LG04']

    session_data = DataIO.run(subject_list , session_data_path)
    output_dir_onedrive = 'C:\\Users\\Sana\\OneDrive\\Desktop\\PHD\\IntervalDiscrimination\\FIGs\\20240819\\'
    output_dir_local = 'C:\\Users\\Sana\\OneDrive\\Desktop\\PHD\\IntervalDiscrimination\\FIGs\\20240819\\'

    for i in range(len(session_data)):
        subject_session_data = session_data[0]
        plot_single_trial_licking.run(subject_session_data,output_dir_onedrive, output_dir_local)
        plot_average_licking.run(subject_session_data,output_dir_onedrive, output_dir_local)    

#%%


if __name__ == "__main__":

    subject_list = ['LCHR_TS01', 'LCHR_TS02', 'LG08_TS03']

    session_data = DataIO.run(subject_list)
    subject_session_data = session_data[0]

    subject_report = fitz.open()
    for i in range(len(session_data)):
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(3, 6, figure=fig)
        plot_outcome.run(plt.subplot(gs[0, 0:3]), session_data[i])
        plot_complete_trials.run(plt.subplot(gs[1, 0:3]), session_data[i])
        plot_psychometric_post.run(plt.subplot(gs[2, 0]), session_data[i])
        plot_psychometric_percep.run(plt.subplot(gs[2, 1]), session_data[i])
        plot_psychometric_pre.run(plt.subplot(gs[2, 2]), session_data[i])
        plot_psychometric_epoch.run([plt.subplot(gs[i, 3]) for i in range(3)], session_data[i])
        plot_reaction_time.run(plt.subplot(gs[0, 4]), session_data[i])
        plot_reaction_outcome.run(plt.subplot(gs[0, 5]), session_data[i])
        plot_decision_time.run(plt.subplot(gs[1, 4]), session_data[i])
        plot_decision_outcome.run(plt.subplot(gs[1, 5]), session_data[i])
        plt.suptitle(session_data[i]['subject'])
        fname = os.path.join(str(i).zfill(4)+'.pdf')
        fig.set_size_inches(30, 15)
        fig.savefig(fname, dpi=300)
        plt.close()
        roi_fig = fitz.open(fname)
        subject_report.insert_pdf(roi_fig)
        roi_fig.close()
        os.remove(fname)
     
    # Get the current date
    current_date = datetime.now()
    # Format the date as 'yyyymmdd'
    formatted_date = current_date.strftime('%Y%m%d')
    # Random integer between 1 and 100
    random_integer = str(random.randint(1000, 9999))  
    subject_report.save('./figures/single_interval_subject_report_'+formatted_date+'_'+random_integer+'.pdf')
    subject_report.close()

