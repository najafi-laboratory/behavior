# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 08:46:06 2025

@author: timst
"""

import os
import random
import gc
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import fitz
from pdf2image import convert_from_path


import modules.utils as utils
import modules.config as config
import modules.data_extract as extract
import modules.data_load as load_data
import modules.data_preprocess as preprocess

from compute.psychometric import compute_psychometric, compute_psychometric_fit
from compute.bpod_session import prepare_session_overview_trial_type, prepare_session_overview_rt
from compute.distributions import compute_isi_pdf, compute_isi_pdf_grouped

from plot.plot_psychometric import plot_psychometric
from plot.plot_session_overview import plot_session_overview_trial_type, plot_session_overview_rt
from plot.plot_donuts import make_opto_control_donut, make_outcome_by_trial_side_non_naive_donut, make_trial_isi_donut, make_outcome_all_trials_donut
from plot.plot_isi_pdf import plot_isi_pdf
from plot.plot_response_time import plot_rt_histogram, plot_pcorrect_kde_by_group, plot_rt_density_by_group
from plot.plot_session_outcomes import plot_trial_outcomes_by_session
from plot.plot_rolling_performance_sessions import plot_rolling_performance_across_sessions

import warnings
warnings.filterwarnings("ignore", message="You passed a edgecolor/edgecolors .* for an unfilled marker .*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*layout engine that is incompatible.*")


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

def create_grid_spec(
    grid_size=(4, 8),
    figsize=(30, 15),
    constrained_layout=True,
):
    layout_mode = 'constrained' if constrained_layout else None
    fig = plt.figure(layout=layout_mode, figsize=figsize)
    gs = GridSpec(*grid_size, figure=fig)       
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
    pdf_path = pdf_path.replace('\\', '/').replace('//', '/')

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    images = convert_from_path(pdf_path, dpi=dpi)
    img = images[0]

    ax.imshow(img)
    ax.set_xlim(0, img.width)       # 🔒 Lock axes to image dimensions
    ax.set_ylim(img.height, 0)      # 🔒 Inverted y-axis for image display

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("")                # Clear any inherited title
    ax.set_xlabel("")               # Clear inherited labels
    ax.set_ylabel("")
    ax.set_frame_on(False)
    ax.axis('off') 
    
    del img
    del images    
    # gc.collect()  # Force cleanup
    # """
    # Rasterize the first page of a PDF file and display it on the given matplotlib Axes.

    # Parameters:
    #     pdf_path (str): Path to the PDF file (assumes one-page).
    #     ax (matplotlib.axes.Axes): Axes to display the image on.
    #     dpi (int): Resolution for rasterization.

    # Returns:
    #     PIL.Image.Image: The rasterized image (for further use if needed).
    # """
    # # Normalize the file path
    # pdf_path = pdf_path.replace('\\', '/').replace('//', '/')

    # if not os.path.exists(pdf_path):
    #     raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # # Convert first page of PDF to image
    # images = convert_from_path(pdf_path, dpi=dpi)
    # img = images[0]  # Take the first page

    # ax.imshow(img)
    # ax.axis('off')

    # return img    

def add_pdf_page(subject_report, pdf_filename, fig, pdf, fname, plt):
    pdf.savefig(fig)
    plt.close(fig)               
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    # os.remove(fname)                 
    print(f"PDF saved to: {pdf_filename}")       

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

def build_report_page(page_spec, plot_paths, dpi=300):
    """
    Create a single report page using a flexible grid layout and pre-rendered plot images.
    
    Returns a Matplotlib Figure object.
    """
    nrows = page_spec.get('nrows', 1)
    ncols = page_spec.get('ncols', 1)
    items = page_spec.get('items', [])

    fig = plt.figure(figsize=(8.5, 11))  # US letter size
    gs = GridSpec(nrows=nrows, ncols=ncols, figure=fig)

    for item in items:
        plot_name = item['plot']
        if plot_name not in plot_paths:
            print(f"⚠️ Plot '{plot_name}' not found in plot_paths. Skipping.")
            continue

        row = item.get('row', 0)
        col = item.get('col', 0)
        rowspan = item.get('rowspan', 1)
        colspan = item.get('colspan', 1)

        ax = fig.add_subplot(gs[row:row+rowspan, col:col+colspan])
        
        img_path = plot_paths[plot_name]
        ext = os.path.splitext(img_path)[-1].lower()

        try:
            if ext in ['.png', '.jpg', '.jpeg']:
                img = mpimg.imread(img_path)
                ax.imshow(img)
            elif ext == '.pdf':
                from matplotlib.backends.backend_pdf import PdfPages
                import fitz  # PyMuPDF
                doc = fitz.open(img_path)
                pix = doc[0].get_pixmap(dpi=dpi)
                img = mpimg.imread(pix.tobytes("ppm"), format='ppm')
                ax.imshow(img)
            else:
                raise ValueError(f"Unsupported format: {ext}")
        except Exception as e:
            print(f"⚠️ Failed to load {img_path}: {e}")
            ax.text(0.5, 0.5, f"Could not load: {plot_name}", ha='center', va='center')
        
        ax.axis('off')

    plt.tight_layout()
    return fig


import os
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

def get_saved_plot_paths(subject_id, session_date=None, plot_names=None, ext='pdf', root_dir='plots'):
    """
    Return a dict of plot_name → file path. Supports:
    - Session-specific plots: plots/{subject_id}/{session_date}/{plot_name}.{ext}
    - Subject-level plots: plots/{subject_id}/{plot_name}.{ext}
    """
    plot_names = plot_names or []
    paths = {}

    # Check both session and subject level
    session_dir = os.path.join(root_dir, subject_id, session_date) if session_date else None
    subject_dir = os.path.join(root_dir, subject_id)

    for name in plot_names:
        found = None

        # Try session-specific first
        if session_date:
            session_path = os.path.join(session_dir, f"{name}.{ext}")
            if os.path.exists(session_path):
                found = session_path

        # Fallback to subject-level
        if not found:
            subject_path = os.path.join(subject_dir, f"{name}.{ext}")
            if os.path.exists(subject_path):
                found = subject_path

        if found:
            paths[name] = found

    return paths

# def merge_report_sections(subject, section_dirnames, merged_filename='full_report.pdf'):
#     """
#     Merge multiple report section PDFs for a subject into one file.

#     Parameters:
#     - subject : str, subject name
#     - section_dirnames : list of str, subfolder names like ['session_cover', 'session_summaries', ...]
#     - merged_filename : str, final filename to save
#     """
#     merged = fitz.open()

#     for dirname in section_dirnames:
#         section_path = os.path.join('reports', subject, dirname, f"{subject}_{dirname}.pdf")
#         if os.path.exists(section_path):
#             with fitz.open(section_path) as section_doc:
#                 merged.insert_pdf(section_doc)
#         else:
#             print(f"⚠️ Missing section: {section_path}")

#     output_dir = os.path.join('reports', subject)
#     os.makedirs(output_dir, exist_ok=True)
#     out_path = os.path.join(output_dir, merged_filename)
#     merged.save(out_path)
#     merged.close()

#     print(f"✅ Merged report saved to: {out_path}")
#     return out_path

def merge_report_sections(subject, session_dates, section_dirnames, merged_filename='full_report.pdf'):
    """
    Merge report section PDFs for a subject, including per-session summaries.

    Parameters:
    - subject : str, subject name
    - session_dates : list of str (YYYY-MM-DD), for locating session summaries
    - section_dirnames : list of str, e.g., ['session_cover', 'session_summaries', 'psychometric']
    - merged_filename : str, output PDF filename
    """
    merged = fitz.open()

    for dirname in section_dirnames:
        if dirname == 'session_summaries':
            for date in session_dates:
                summary_path = os.path.join('reports', subject, 'session_summaries', f"{subject}_{date}_session_summaries.pdf")
                if os.path.exists(summary_path):
                    with fitz.open(summary_path) as doc:
                        merged.insert_pdf(doc)
                else:
                    print(f"⚠️ Missing session summary: {summary_path}")
        else:
            section_path = os.path.join('reports', subject, dirname, f"{subject}_{dirname}.pdf")
            if os.path.exists(section_path):
                with fitz.open(section_path) as doc:
                    merged.insert_pdf(doc)
            else:
                print(f"⚠️ Missing section: {section_path}")

    output_dir = os.path.join('reports', subject)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, merged_filename)
    merged.save(out_path)
    merged.close()

    print(f"✅ Merged report saved to: {out_path}")
    return out_path

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
    
    
    
    # subject_list = ['LCHR_TS01', 'LCHR_TS02', 'SCHR_TS06', 'SCHR_TS07', 'SCHR_TS08', 'SCHR_TS09']
    subject_list = ['LCHR_TS02', 'SCHR_TS06', 'SCHR_TS07', 'SCHR_TS08', 'SCHR_TS09']
    # subject_list = ["LCHR_TS01"]
    # subject_list = ["LCHR_TS02"]
    
    # subject_list = ["SCHR_TS06"]
    # subject_list = ["SCHR_TS07", 'SCHR_TS08', 'SCHR_TS09']
    
    subjects = ", ".join([s for s in subject_list])
    print(f"Extracting data for subjects {subjects}...")

    # Extract and store sessions
    # extract.batch_extract(subjects_root_dir, extracted_root_dir, subject_list)
    extract.batch_extract(subject_list, config, force=False)
    
  
    
    # subject_list = ['LCHR_TS01', 'LCHR_TS02', 'SCHR_TS06', 'SCHR_TS07', 'SCHR_TS08', 'SCHR_TS09']
    subject_list = ['LCHR_TS02', 'SCHR_TS06', 'SCHR_TS07', 'SCHR_TS08', 'SCHR_TS09']
    # subject_list = ["LCHR_TS01", "LCHR_TS02"]
    # subject_list = ["LCHR_TS01"]
    # subject_list = ["LCHR_TS02"]
    # subject_list = ["SCHR_TS06"]
    # subject_list = ["SCHR_TS07"]
    # subject_list = ["SCHR_TS08"]
    # subject_list = ["SCHR_TS09"]
    
    subjects = ", ".join([s for s in subject_list])
    print(f"Preprocessing data for subjects {subjects}...")    
    
    # datist - data extraction, data bridging, data cleaning, phantom data, data snoopingecho
    all_subjects_data = load_data.load_batch_sessions(subject_list, config)
    
    preprocessed_data = preprocess.batch_preprocess_sessions(subject_list, config, force=True)


    print(f"Running data analysis for subjects {subjects}...")
    
    
    cover = 0
    session_summary = 0
    psychometric = 0
    RT = 0
    
    
    
    
    
    merge_report = 1
    
    show_plot = False
    
    if cover:
        print(f"Compute summary cover for subjects {subjects}...")

        for subject in subject_list:
            
            print(f"Compute summary cover for subject {subject}...")    
        
            
            debug_print = False
        
            report = fitz.open()
            # output_path = config.paths['output_dir_local']
            output_path = os.path.join(f'reports\\{subject}\\session_cover')
            utils.sanitize_and_create_dir(output_path)
            pdf_fname = os.path.join(output_path, f"{subject}_session_cover.pdf")     
            
            pdf = PdfPages(pdf_fname)        
            
            print('')
            
            # OPTO PSYCHOMETRIC SESSIONS
            # Determine number of required pages
            # total_sessions = len(preprocessed_data[subject])             
            # n_pages_1, grid_size, block_size = generate_paged_pdf_spec(
            #     config.session_config_list_2AFC,
            #     total_items=total_sessions,
            #     grid_size=(4, 8),
            #     fig_size=(30, 15),
            #     block_size=(2, 2),
            # )                  
            
            fig, gs = create_grid_spec(grid_size=(4, 6),
                                       figsize=(30, 15),
                                       constrained_layout=True,
                                       )
            page_idx_prev = 0
            
            fname = ''  
                    
            combined_df = []
            
            for i in range(len(preprocessed_data[subject])): 
    
                df = preprocessed_data[subject][i]["df"]
                session_info = preprocessed_data[subject][i]["session_info"]        
                
                
                subject_name = session_info['subject_name']
                session_date = session_info['date']    
                
                # Optionally: tag each row with session metadata (useful for filtering later)
                df = df.copy()
                df["date"] = session_info["date"]
                df["SessionDate"] = session_info["SessionDate"]
                df["subject_name"] = session_info["subject_name"]
                
                combined_df.append(df)                
    
            
                
            # Session Summary Plots
            ##########################################################
            
            

            df_concat = pd.concat(combined_df, ignore_index=True)
            df_all_sessions = df_concat.copy()
            
            

            
            # plot_psychometric(df_all_sessions, ...)
            # plot_response_time_distribution(df_all_sessions, ...)
            # plot_pcorrect_kde_by_group(df_all_sessions, ...)            
            
            # split_opto = True
            # group_cols = ['session_date', 'trial_side']
            # if split_opto:
            #     group_cols.append('is_opto')
            # group_cols.append('mouse_correct')
            
            # summary = df.groupby(group_cols).size().unstack(fill_value=0)
            
            # filter out no lick
            df_all_sessions = df_all_sessions[df_all_sessions['lick'] != 0]
            # filter out naive
            df_all_sessions = df_all_sessions[df_all_sessions['naive'] == 0]
            # filter out move single spout
            df_all_sessions = df_all_sessions[df_all_sessions['MoveCorrectSpout'] == 0]   
           
            fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 0), span=(1, 6), adjust_margins=True)
            fname = plot_trial_outcomes_by_session(
                df_all_sessions,
                split_opto=True,
                normalize=True,
                show_counts=True,
                bar_spacing=0.5,
                show_plot=show_plot
            )
            rasterize_pdf_to_axes(fname, ax)  
            
            fig, ax, gs = create_gs_subplot(fig, gs, position=(1, 0), span=(1, 6), adjust_margins=True)
            df_concat = df_concat[df_concat['outcome'] != 'DidNotChoose'] 
            fname = plot_rolling_performance_across_sessions(df_concat, ax=None, show_plot=show_plot)
            rasterize_pdf_to_axes(fname, ax)            
                        
            fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 0), span=(1, 1), adjust_margins=True)
            fname = make_outcome_all_trials_donut(df_all_sessions, session_info, ax=None, show_plot=show_plot)   
            rasterize_pdf_to_axes(fname, ax)
            
            fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 1), span=(1, 1), adjust_margins=True)
            fname = make_outcome_by_trial_side_non_naive_donut(df_all_sessions, session_info, ax=None, show_plot=show_plot)
            rasterize_pdf_to_axes(fname, ax)
            
            fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 2), span=(1, 1), adjust_margins=True)
            fname = make_trial_isi_donut(df_all_sessions, session_info, ax=None, show_plot=show_plot)
            rasterize_pdf_to_axes(fname, ax)
            
            fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 3), span=(1, 1), adjust_margins=True)
            fname = make_opto_control_donut(df_all_sessions, session_info, ax=None, show_plot=show_plot)
            rasterize_pdf_to_axes(fname, ax)
            
            
            subject = df['subject_name'].unique()[0]            
            # If 'session_date' is not already datetime, convert it
            df['date'] = pd.to_datetime(df['date'])
            min_date = df['date'].min().strftime("%Y-%m-%d")
            max_date = df['date'].max().strftime("%Y-%m-%d")
            dates = min_date + '-' + max_date            
            title = f"{subject} | ISI Probability Density {min_date} to {max_date}"
            ax.set_title(title, y=1.05)
            
            fig, ax, gs = create_gs_subplot(fig, gs, position=(3, 0), span=(1, 1), adjust_margins=True)
            pdfs = {}
            pdfs = compute_isi_pdf(df_all_sessions)            
            fname = plot_isi_pdf(pdfs, session_info, title=title, show_plot=show_plot, is_cover=True)
            rasterize_pdf_to_axes(fname, ax)                
        
            title = f"{subject} | ISI Probability Density Opto {min_date} to {max_date}"
        
            fig, ax, gs = create_gs_subplot(fig, gs, position=(3, 1), span=(1, 1), adjust_margins=True)
            pdfs = {}
            pdfs = compute_isi_pdf_grouped(df_all_sessions)
            fname = plot_isi_pdf(pdfs, session_info, title=title, show_plot=show_plot, is_cover=True)
            rasterize_pdf_to_axes(fname, ax)                
            
            fig, ax, gs = create_gs_subplot(fig, gs, position=(3, 2), span=(1, 1), adjust_margins=True)
            fname = plot_rt_histogram(df_all_sessions, session_info, ax=None, show_plot=show_plot)
            rasterize_pdf_to_axes(fname, ax)
            # add_pdf_page(report, pdf_fname, fig, pdf, fname, plt)                
                 
            
            # AVG PSYCHOMETRIC ISI
            ##########################################################
            
            
            # if session_info['OptoSession']:
            #     condition_col='is_opto'
            # else:
            #     condition_col=None
            
            # # condition_col=None
            # df_psy = compute_psychometric(df_all_sessions, condition_col=condition_col)
            # df_psy['subject_name'] = session_info['subject_name']
            # df_psy['SessionDate'] = session_info['SessionDate']
            
            
            # df_psy['analysis_condition'] = 'opto'
            # # df_psy['opto_region'] = session_info.get('OptoRegionShortText', 'Control')
            
            # region = session_info.get('OptoRegionShortText', 'Unknown')
            # df_psy['opto_region'] = df_psy['condition'].apply(
            #     lambda x: region if x == 1 else 'Control'
            # )
            
            # side = session_info.get('OptoTargetSideText', None)
            # df_psy['opto_side'] = df_psy['condition'].apply(
            #     lambda x: side if x == 1 else None
            # )    
             
            # # Add readable labels
            # if 'condition' in df_psy.columns:
            #     region_abbrev = session_info.get('OptoRegionShortText', 'Opto')
            #     df_psy['condition_label'] = df_psy['condition'].map({
            #         0: 'Control',
            #         1: f"Opto {region_abbrev}"
            #     })
                
         
            # # get psychometric fits
            # fits = {}
            # for cond, group in df_psy.groupby('condition'):
            #     fit = compute_psychometric_fit(group)
            #     # print(f"📉 {cond} → β0: {fit['params'][0]:.3f}, β1: {fit['params'][1]:.3f}, threshold: {fit['threshold']:.3f}")
            #     fits[cond] = fit    
            
            
            # # PLOT Loop - it appears you have a video buffer overrun, why...yes..how did you know?
            # page_idx, row, col = assign_grid_position(i, grid_size, block_size) 
            # if page_idx != page_idx_prev:
            #     # add_pdf_page(fig, pdf, fname, plt)
            #     add_pdf_page(report, pdf_fname, fig, pdf, fname, plt)
            #     fig, gs = create_grid_spec()
            #     page_idx_prev = page_idx
            
            # if debug_print:
            #     print(f"[page {page_idx}] row: {row}, col: {col}")            
            
            # fig, ax, gs = create_gs_subplot(fig, gs, position=(3, 3), span=(1, 1), adjust_margins=True)
            # fname = plot_psychometric(df_psy, session_info, config.session_config_list_2AFC, ax=None, show_fit=True, fit_params=fits,show_plot=show_plot)
            # rasterize_pdf_to_axes(fname, ax)            
            
            
            
            
            
            
            
            
            # fig, ax, gs = create_gs_subplot(fig, gs, position=(3, 0), span=(1, 1), adjust_margins=True)
            # pdfs = {}
            # pdfs = compute_isi_pdf(df_all_sessions)
            # fname = plot_isi_pdf(pdfs, session_info, show_plot=show_plot)
            # rasterize_pdf_to_axes(fname, ax)    
            
            # if session_info['OptoSession']:
            #     condition_col='is_opto'
            # else:
            #     condition_col=None
            
            # # condition_col=None
            # df_psy = compute_psychometric(df, condition_col=condition_col)
            # df_psy['subject_name'] = session_info['subject_name']
            # df_psy['SessionDate'] = session_info['SessionDate']
            
            
            # df_psy['analysis_condition'] = 'opto'
            # # df_psy['opto_region'] = session_info.get('OptoRegionShortText', 'Control')
            
            # region = session_info.get('OptoRegionShortText', 'Unknown')
            # df_psy['opto_region'] = df_psy['condition'].apply(
            #     lambda x: region if x == 1 else 'Control'
            # )
            
            # side = session_info.get('OptoTargetSideText', None)
            # df_psy['opto_side'] = df_psy['condition'].apply(
            #     lambda x: side if x == 1 else None
            # )    
             
            # # Add readable labels
            # if 'condition' in df_psy.columns:
            #     region_abbrev = session_info.get('OptoRegionShortText', 'Opto')
            #     df_psy['condition_label'] = df_psy['condition'].map({
            #         0: 'Control',
            #         1: f"Opto {region_abbrev}"
            #     })
                
         
            # # get psychometric fits
            # fits = {}
            # for cond, group in df_psy.groupby('condition'):
            #     fit = compute_psychometric_fit(group)
            #     # print(f"📉 {cond} → β0: {fit['params'][0]:.3f}, β1: {fit['params'][1]:.3f}, threshold: {fit['threshold']:.3f}")
            #     fits[cond] = fit    
                     
            
            # fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 0), span=(2,2), adjust_margins=True)
            # fname = plot_psychometric(df_psy, session_info, config.session_config_list_2AFC, ax=None, show_fit=True, fit_params=fits,show_plot=show_plot)
            # rasterize_pdf_to_axes(fname, ax)
        
         
            add_pdf_page(report, pdf_fname, fig, pdf, fname, plt)            
            
        
            pdf.close() 
    if session_summary:
       
        print(f"Compute session plots for subjects {subjects}...")

        for subject in subject_list:
            
            print(f"Compute session plots for subject {subject}...")
            
            # show_plot = True
            debug_print = False
        
               
            
        # # for i in range(len(preprocessed_data["LCHR_TS01"])):
            for i in range(len(preprocessed_data[subject])): 
    
                df = preprocessed_data[subject][i]["df"]
                session_info = preprocessed_data[subject][i]["session_info"]        
                
                
                subject_name = session_info['subject_name']
                session_date = session_info['date']               
    
                report = fitz.open()
                # output_path = config.paths['output_dir_local']
                output_path = os.path.join(f'reports\\{subject}\\session_summaries')
                utils.sanitize_and_create_dir(output_path)
                pdf_fname = os.path.join(output_path, f"{subject}_{session_date}_session_summaries.pdf")     
                
                pdf = PdfPages(pdf_fname)        
                
                print('')
                
                # OPTO PSYCHOMETRIC SESSIONS
                # Determine number of required pages
                # total_sessions = len(preprocessed_data[subject])             
                # n_pages_1, grid_size, block_size = generate_paged_pdf_spec(
                #     config.session_config_list_2AFC,
                #     total_items=total_sessions,
                #     grid_size=(4, 8),
                #     fig_size=(30, 15),
                #     block_size=(2, 2),
                # )                  
                
                fig, gs = create_grid_spec(grid_size=(4, 4),
                                           figsize=(30, 30),
                                           constrained_layout=True,
                                           )
                page_idx_prev = 0
                
                fname = ''                 
    
    
                  
    
                # Session Summary Plots
                ##########################################################
                
                
                df_session_tt = prepare_session_overview_trial_type(df)
                df_session_tt['subject_name'] = session_info['subject_name']
                df_session_tt['SessionDate'] = session_info['SessionDate']
            
                fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 0), span=(1, 4), adjust_margins=True)
                fname = plot_session_overview_trial_type(df_session_tt, session_info, ax=None, title=None, show_plot=show_plot)
                rasterize_pdf_to_axes(fname, ax)
            
                df_session_rt = prepare_session_overview_rt(df)
                df_session_rt['subject_name'] = session_info['subject_name']
                df_session_rt['SessionDate'] = session_info['SessionDate']
            
                fig, ax, gs = create_gs_subplot(fig, gs, position=(1, 0), span=(1, 4), adjust_margins=True)
                fname = plot_session_overview_rt(df_session_rt, session_info, ax=None, title=None, show_plot=show_plot)
                rasterize_pdf_to_axes(fname, ax)
                
                

                
                
                # fig, ax = plt.subplots()
                fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 0), span=(1, 1), adjust_margins=True)
                fname = make_outcome_all_trials_donut(df_session_rt, session_info, ax=None, show_plot=show_plot)   
                rasterize_pdf_to_axes(fname, ax)
                
                fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 1), span=(1, 1), adjust_margins=True)
                fname = make_outcome_by_trial_side_non_naive_donut(df_session_rt, session_info, ax=None, show_plot=show_plot)
                rasterize_pdf_to_axes(fname, ax)
                
                fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 2), span=(1, 1), adjust_margins=True)
                fname = make_trial_isi_donut(df_session_rt, session_info, ax=None, show_plot=show_plot)
                rasterize_pdf_to_axes(fname, ax)
                
                fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 3), span=(1, 1), adjust_margins=True)
                fname = make_opto_control_donut(df_session_rt, session_info, ax=None, show_plot=show_plot)
                rasterize_pdf_to_axes(fname, ax)
                
                
                fig, ax, gs = create_gs_subplot(fig, gs, position=(3, 0), span=(1, 1), adjust_margins=True)
                pdfs = {}
                pdfs = compute_isi_pdf(df)
                fname = plot_isi_pdf(pdfs, session_info, show_plot=show_plot)
                rasterize_pdf_to_axes(fname, ax)                
            
                fig, ax, gs = create_gs_subplot(fig, gs, position=(3, 1), span=(1, 1), adjust_margins=True)
                pdfs = {}
                pdfs = compute_isi_pdf_grouped(df)
                fname = plot_isi_pdf(pdfs, session_info, title="ISI Probability Density - Opto", show_plot=show_plot)
                rasterize_pdf_to_axes(fname, ax)           
                
                # fig, ax, gs = create_gs_subplot(fig, gs, position=(row, col), span=block_size, adjust_margins=True)
                # fname = plot_rt_histogram(df, session_info, ax=None, show_plot=show_plot)
                # rasterize_pdf_to_axes(fname, ax)
                # # add_pdf_page(report, pdf_fname, fig, pdf, fname, plt)                
                
                
                
                
                
                add_pdf_page(report, pdf_fname, fig, pdf, fname, plt)
            
            # add_pdf_page(report, pdf_fname, fig, pdf, fname, plt)
                pdf.close()        
            
                # ISI Distributions-work work work work, workin on mus sheets, so ican get data rich, mho money mho family, 16 in middle of mi
                ##########################################################
                        
                # pdfs = compute_isi_pdf(df)
                # plot_isi_pdf(pdfs)
            
    if psychometric:
        
        print(f"Compute psychometrics for subjects {subjects}...")
        
        for subject in subject_list:
        # # for i in range(len(preprocessed_data["LCHR_TS01"])):
            
            
        # PSYCHOMETRIC SINGLE SESSION PLOTS
        ####################################################################################
            print(f"Compute psychometrics for subject {subject}...")
        
        
            # show_plot = True
            debug_print = False
        
            report = fitz.open()
            # output_path = config.paths['output_dir_local']
            output_path = os.path.join(f'reports\\{subject}\\psychometric')
            utils.sanitize_and_create_dir(output_path)
            pdf_fname = os.path.join(output_path, f"{subject}_psychometric.pdf")     
            
            pdf = PdfPages(pdf_fname)        
            
            print('')
            
            # OPTO PSYCHOMETRIC SESSIONS
            # Determine number of required pages
            total_sessions = len(preprocessed_data[subject])             
            n_pages_1, grid_size, block_size = generate_paged_pdf_spec(
                config.session_config_list_2AFC,
                total_items=total_sessions,
                grid_size=(4, 8),
                fig_size=(30, 15),
                block_size=(2, 2),
            )                  
            
            fig, gs = create_grid_spec()
            page_idx_prev = 0
            
            fname = ''
            
            for i in range(len(preprocessed_data[subject])):     
                
                # Access a session
                # df = preprocessed_data["LCHR_TS01"][0]["df"]
                # session_info = preprocessed_data["LCHR_TS01"][0]["session_info"]
                # df = preprocessed_data["LCHR_TS01"][i]["df"]
                # session_info = preprocessed_data["LCHR_TS01"][i]["session_info"]
                df = preprocessed_data[subject][i]["df"]
                session_info = preprocessed_data[subject][i]["session_info"]        
                
                
                subject_name = session_info['subject_name']
                session_date = session_info['date']
                
                
                # PSYCHOMETRIC ISI
                ##########################################################
                
                
                if session_info['OptoSession']:
                    condition_col='is_opto'
                else:
                    condition_col=None
                
                # condition_col=None
                df_psy = compute_psychometric(df, condition_col=condition_col)
                df_psy['subject_name'] = session_info['subject_name']
                df_psy['SessionDate'] = session_info['SessionDate']
                
                
                df_psy['analysis_condition'] = 'opto'
                # df_psy['opto_region'] = session_info.get('OptoRegionShortText', 'Control')
                
                region = session_info.get('OptoRegionShortText', 'Unknown')
                df_psy['opto_region'] = df_psy['condition'].apply(
                    lambda x: region if x == 1 else 'Control'
                )
                
                side = session_info.get('OptoTargetSideText', None)
                df_psy['opto_side'] = df_psy['condition'].apply(
                    lambda x: side if x == 1 else None
                )    
                 
                # Add readable labels
                if 'condition' in df_psy.columns:
                    region_abbrev = session_info.get('OptoRegionShortText', 'Opto')
                    df_psy['condition_label'] = df_psy['condition'].map({
                        0: 'Control',
                        1: f"Opto {region_abbrev}"
                    })
                    
             
                # get psychometric fits
                fits = {}
                for cond, group in df_psy.groupby('condition'):
                    fit = compute_psychometric_fit(group)
                    # print(f"📉 {cond} → β0: {fit['params'][0]:.3f}, β1: {fit['params'][1]:.3f}, threshold: {fit['threshold']:.3f}")
                    fits[cond] = fit    
                
                
                # PLOT Loop - it appears you have a video buffer overrun, why...yes..how did you know?
                page_idx, row, col = assign_grid_position(i, grid_size, block_size) 
                if page_idx != page_idx_prev:
                    # add_pdf_page(fig, pdf, fname, plt)
                    add_pdf_page(report, pdf_fname, fig, pdf, fname, plt)
                    fig, gs = create_grid_spec()
                    page_idx_prev = page_idx
                
                if debug_print:
                    print(f"[page {page_idx}] row: {row}, col: {col}")            
                
                fig, ax, gs = create_gs_subplot(fig, gs, position=(row, col), span=block_size, adjust_margins=True)
                fname = plot_psychometric(df_psy, session_info, config.session_config_list_2AFC, ax=None, show_fit=True, fit_params=fits,show_plot=show_plot)
                rasterize_pdf_to_axes(fname, ax)
                # add_pdf_page(report, pdf_fname, fig, pdf, fname, plt)
            
            add_pdf_page(report, pdf_fname, fig, pdf, fname, plt)
            pdf.close()        
  
    if RT:
         
         print(f"Compute response time for subjects {subjects}...")
         
         for subject in subject_list:
         # # for i in range(len(preprocessed_data["LCHR_TS01"])):
             
     
         # RESPONSE TIME SINGLE SESSION PLOTS
         ####################################################################################
             print(f"Compute response time for subject {subject}...")
         
         
             # show_plot = True
             debug_print = False
         
             report = fitz.open()
             # output_path = config.paths['output_dir_local']
             output_path = os.path.join(f'reports\\{subject}\\response_time')
             utils.sanitize_and_create_dir(output_path)
             pdf_fname = os.path.join(output_path, f"{subject}_response_time.pdf")     
             
             pdf = PdfPages(pdf_fname)        
             
             print('')
             
             # RESPONSE TIME SESSIONS
             # Determine number of required pages
             total_sessions = len(preprocessed_data[subject])             
             n_pages_1, grid_size, block_size = generate_paged_pdf_spec(
                 config.session_config_list_2AFC,
                 total_items=total_sessions,
                 grid_size=(4, 8),
                 fig_size=(30, 15),
                 block_size=(2, 2),
             )                  
             
             fig, gs = create_grid_spec()
             page_idx_prev = 0
             
             fname = ''
             
             for i in range(len(preprocessed_data[subject])):     
                 
                 # Access a session
                 # df = preprocessed_data["LCHR_TS01"][0]["df"]
                 # session_info = preprocessed_data["LCHR_TS01"][0]["session_info"]
                 # df = preprocessed_data["LCHR_TS01"][i]["df"]
                 # session_info = preprocessed_data["LCHR_TS01"][i]["session_info"]
                 df = preprocessed_data[subject][i]["df"]
                 session_info = preprocessed_data[subject][i]["session_info"]        
                 
                 
                 subject_name = session_info['subject_name']
                 session_date = session_info['date']
                 
                 
                 # RESPONSE TIME
                 ##########################################################
                 print(f"Compute response time for subject {subject} on {session_date}...") 
                 

       
                 # # PLOT Loop
                 # page_idx, row, col = assign_grid_position(i, grid_size, block_size) 
                 # if page_idx != page_idx_prev:
                 #     # add_pdf_page(fig, pdf, fname, plt)
                 #     add_pdf_page(report, pdf_fname, fig, pdf, fname, plt)
                 #     fig, gs = create_grid_spec()
                 #     page_idx_prev = page_idx
                 
                 # if debug_print:
                 #     print(f"[page {page_idx}] row: {row}, col: {col}")            
                 
                 # fig, ax, gs = create_gs_subplot(fig, gs, position=(row, col), span=block_size, adjust_margins=True)
                 # fname = plot_rt_histogram(df, session_info, ax=None, show_plot=show_plot)
                 # rasterize_pdf_to_axes(fname, ax)
                 # # add_pdf_page(report, pdf_fname, fig, pdf, fname, plt)
             

                 # PLOT Loop
                 page_idx, row, col = assign_grid_position(i, grid_size, block_size) 
                 if page_idx != page_idx_prev:
                     # add_pdf_page(fig, pdf, fname, plt)
                     add_pdf_page(report, pdf_fname, fig, pdf, fname, plt)
                     fig, gs = create_grid_spec()
                     page_idx_prev = page_idx
                 
                 if debug_print:
                     print(f"[page {page_idx}] row: {row}, col: {col}") 

                 block_size = (2, 2)

                 # fname = plot_pcorrect_kde_by_group(df, session_info, ax=None, show_plot=show_plot)

                 fade_factor = 1.5

                 fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 0), span=block_size, adjust_margins=True)
                 fname = plot_pcorrect_kde_by_group(df, session_info, ax=None, grouping='all', fade_factor=fade_factor, show_plot=show_plot)
                 rasterize_pdf_to_axes(fname, ax) 
                                  
                 fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 2), span=block_size, adjust_margins=True)
                 fname = plot_pcorrect_kde_by_group(df[df['trial_side'] == 'left'], session_info, ax=None, grouping=['trial_side'], fade_factor=fade_factor, show_plot=show_plot)
                 rasterize_pdf_to_axes(fname, ax)                  
                 
                 fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 4), span=block_size, adjust_margins=True)
                 fname = plot_pcorrect_kde_by_group(df[df['trial_side'] == 'right'], session_info, ax=None, grouping=['trial_side'], fade_factor=fade_factor, show_plot=show_plot)
                 rasterize_pdf_to_axes(fname, ax)                    
                 
                 fig, ax, gs = create_gs_subplot(fig, gs, position=(0, 6), span=block_size, adjust_margins=True)
                 fname = plot_pcorrect_kde_by_group(df, session_info, ax=None, grouping=['trial_side'], fade_factor=fade_factor,show_plot=show_plot)
                 rasterize_pdf_to_axes(fname, ax)                 
                 
                 fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 0), span=block_size, adjust_margins=True)
                 fname = plot_pcorrect_kde_by_group(df, session_info, ax=None, grouping=['is_opto'], fade_factor=fade_factor, show_plot=show_plot)
                 rasterize_pdf_to_axes(fname, ax)                  
                 
                 fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 2), span=block_size, adjust_margins=True)
                 fname = plot_pcorrect_kde_by_group(df[df['trial_side'] == 'left'], session_info, ax=None, grouping=['trial_side', 'is_opto'], fade_factor=fade_factor, show_plot=show_plot)
                 rasterize_pdf_to_axes(fname, ax)                   
                 
                 fig, ax, gs = create_gs_subplot(fig, gs, position=(2, 4), span=block_size, adjust_margins=True)
                 fname = plot_pcorrect_kde_by_group(df[df['trial_side'] == 'right'], session_info, ax=None, grouping=['trial_side', 'is_opto'], fade_factor=fade_factor, show_plot=show_plot)
                 rasterize_pdf_to_axes(fname, ax)                     
                 
                 # fname = plot_pcorrect_kde_by_group(df, session_info, ax=None, grouping=['trial_side', 'is_opto'], show_plot=show_plot)
                
                 # fname = plot_rt_density_by_group(df, session_info, ax=None, grouping=['trial_side', 'is_opto'], show_plot=show_plot)
                 
                 
                 # fname = plot_pcorrect_kde_by_group(df[df['trial_side'] == 'left'], session_info, ax=None, grouping=['trial_side', 'is_opto'], show_plot=show_plot)
                 # fname = plot_pcorrect_kde_by_group(df, session_info, ax=None, grouping=['trial_side'], show_plot=show_plot)
                 # fname = plot_pcorrect_kde_by_group(df, session_info, ax=None, grouping=None, show_plot=show_plot)
                             
                 add_pdf_page(report, pdf_fname, fig, pdf, fname, plt)
                
             
                
             
             # add_pdf_page(report, pdf_fname, fig, pdf, fname, plt)
             pdf.close()            
  
    
    print("")
  # for subject in subject_list:
    #     for i in range(len(preprocessed_data[subject])):
            
    # if merge_report:     
      
    #    # subject_list = ['LCHR_TS01', 'LCHR_TS02', 'SCHR_TS06', 'SCHR_TS07', 'SCHR_TS08', 'SCHR_TS09']
    #    # subject_list = ["LCHR_TS01", "LCHR_TS02"]   
    #    subject_list = ["LCHR_TS01"]
    #    # subject_list = ["LCHR_TS02"]
    #    # subject_list = ["SCHR_TS06"]
    #    # subject_list = ["SCHR_TS07"]
    #    # subject_list = ["SCHR_TS08"]
    #    # subject_list = ["SCHR_TS09"]

    #    for subject in subject_list:     
            
    #        for i in range(len(preprocessed_data[subject])):
    #           # get dates list
              
    #         merge_report_sections(
    #             subject=subject,
    #             section_dirnames=[
    #                 'session_cover',
    #                 'session_summaries',
    #                 'psychometric',
    #                 'response_time'
    #             ],
    #             merged_filename=f'{subject}_full_report.pdf'
    #         )
    
    if merge_report:
    
        for subject in subject_list:     
            session_dates = [
                preprocessed_data[subject][i]['session_info']['date']
                for i in range(len(preprocessed_data[subject]))
            ]
        
            merge_report_sections(
                subject=subject,
                session_dates=session_dates,
                section_dirnames=[
                    'session_cover',
                    'session_summaries',  # Will now loop by date
                    'psychometric',
                    'response_time'
                ],
                merged_filename=f'{subject}_full_report.pdf'
            )    
            
        print("")
    
        
        
    

    
    



