from scipy.stats import sem
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfFileReader, PdfFileWriter, PdfWriter, PdfReader
from scipy.interpolate import interp1d
from datetime import date
from statistics import mean 
import math
from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.patches as mpatches #ohoolahan doge, daq, dip switches, diva, and doogie hausarus cavedog?
from pathlib import Path
import re
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from utils.util import get_figsize_from_pdf_spec
from scipy.stats import norm
import statsmodels.api as sm
from scipy.stats import gaussian_kde
import seaborn as sns


print_debug = 0

def ensure_list(var):
    return var if isinstance(var, list) else [var]

def save_image(filename): 
    
    p = PdfPages(filename+'.pdf') 
    fig_nums = plt.get_fignums()   
    figs = [plt.figure(n) for n in fig_nums] 
      
    for fig in figs:  
        
        fig.savefig(p, format='pdf', dpi=300)
           
    p.close() 
    
def remove_substrings(s, substrings):
    for sub in substrings:
        s = s.replace(sub, "")
    return s

def flip_underscore_parts(s):
    parts = s.split("_", 1)  # Split into two parts at the first underscore
    if len(parts) < 2:
        return s  # Return original string if no underscore is found
    return f"{parts[1]}_{parts[0]}"

def lowercase_h(s):
    return s.replace('H', 'h')

def get_decision_df(session_data, session_idx):
    """
    Process licks for a single session.
    
    Args:
        session_data (dict): Contains left/right lick times and opto flags for trials.
    
    Returns:
        dict: Processed licks categorized into left/right and opto/non-opto.
    """
    processed_dec = []
    
    
    raw_data = session_data['raw']
    
    numTrials = raw_data[session_idx]['nTrials']
    
    outcomes_time = session_data['outcomes_time']
    outcome_time = outcomes_time[session_idx]
        
    trial_types = session_data['trial_type'][session_idx]
    opto_flags = session_data['opto_trial'][session_idx]
    
    opto_encode = np.nan
    
    for trial in range(numTrials):
           
        licks = {}
        valve_times = []
        rewarded = 0
        isNaive = 0
        no_lick = 0
        
        alignment = 0
        # alignment = outcome_time[trial]
        alignment = raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['WindowChoice'][0]        
        
        if not 'Port1In' in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events'].keys():
            licks['licks_left_start'] = [np.float64(np.nan)]
        elif isinstance(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1In'], (float, int)):
            licks['licks_left_start'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1In']]
        else:
            licks['licks_left_start'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1In']
            
        if not 'Port1Out' in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events'].keys():
            licks['licks_left_stop'] = [np.float64(np.nan)]
        elif isinstance(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1Out'], (float, int)):
            licks['licks_left_stop'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1Out']]
        else:
            licks['licks_left_stop'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port1Out']    
    
        if not 'Port3In' in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events'].keys():
            licks['licks_right_start'] = [np.float64(np.nan)]
        elif isinstance(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3In'], (float, int)):
            licks['licks_right_start'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3In']]
        else:
            licks['licks_right_start'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3In']       
        
        if not 'Port3Out' in raw_data[session_idx]['RawEvents']['Trial'][trial]['Events'].keys():
            licks['licks_right_stop'] = [np.float64(np.nan)]
        elif isinstance(raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3Out'], (float, int)):
            licks['licks_right_stop'] = [raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3Out']]
        else:
            licks['licks_right_stop'] = raw_data[session_idx]['RawEvents']['Trial'][trial]['Events']['Port3Out']

        ###
  
        # np.array([x - alignment for x in licks['licks_left_start']])
        licks['licks_left_start'] = [(x - alignment)*1000 for x in licks['licks_left_start']]
        licks['licks_left_stop'] = [(x - alignment)*1000 for x in licks['licks_left_stop']]
        licks['licks_right_start'] = [(x - alignment)*1000 for x in licks['licks_right_start']]
        licks['licks_right_stop'] = [(x - alignment)*1000 for x in licks['licks_right_stop']]
  
        # check for licks or spout touches before choice window
        # if licks['licks_left_start'][0] < -0.1:
        #     licks['licks_left_start'] = [np.float64(np.nan)]
        # if licks['licks_left_stop'][0] < -0.1:
        #     licks['licks_left_stop'] = [np.float64(np.nan)]
        # if licks['licks_right_start'][0] < -0.1:
        #     licks['licks_right_start'] = [np.float64(np.nan)]
        # if licks['licks_right_stop'][0] < -0.1:
        #     licks['licks_right_stop'] = [np.float64(np.nan)]            
    
        trial_type = "left" if trial_types[trial] == 1 else "right" 
        
        # Track valve open/close times for the trial (start/stop)
        if 'Reward' in raw_data[session_idx]['RawEvents']['Trial'][trial]['States']:
            is_naive = 0
            valve_times.append(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['Reward'][0] - alignment)
            valve_times.append(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['Reward'][1] - alignment)
            if not np.isnan(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['Reward'][0]):
                rewarded = 1
            else:
                rewarded = 0
        elif 'NaiveRewardDeliver' in raw_data[session_idx]['RawEvents']['Trial'][trial]['States']:  
            is_naive = 1
            valve_times.append(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['NaiveRewardDeliver'][0] - alignment)
            valve_times.append(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['NaiveRewardDeliver'][1] - alignment)
            if not np.isnan(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['NaiveRewardDeliver'][0]):
                rewarded = 1
            else:
                rewarded = 0
        else:
            print('what this????')
            valve_times.append(np.float64(np.nan))
            valve_times.append(np.float64(np.nan))
            
        if 'DidNotChoose' in raw_data[session_idx]['RawEvents']['Trial'][trial]['States']:  
            if not np.isnan(raw_data[session_idx]['RawEvents']['Trial'][trial]['States']['DidNotChoose'][0]):
                no_lick = 1
            
        is_opto = opto_flags[trial]   
        
        if is_opto and not is_naive:
            opto_encode = 0
            
        isi = session_data['isi_post_emp'][session_idx][trial]
        
        move_correct_spout = session_data['move_correct_spout_flag'][session_idx][trial]
            
        processed_dec.append({
            "trial": trial,
            "trial_side": trial_type,
            "isi": isi,
            "is_opto": is_opto,
            "is_naive": is_naive,
            "rewarded": rewarded,
            "no_lick": no_lick,
            "opto_encode": opto_encode,
            "move_correct_spout": move_correct_spout,
            "licks_left_start": licks['licks_left_start'],
            "licks_left_stop": licks['licks_left_stop'],
            "licks_right_start": licks['licks_right_start'],
            "licks_right_stop": licks['licks_right_stop'],
            "valve_start": valve_times[0],
            "valve_stop": valve_times[1]
        })
        
        opto_encode = opto_encode + 1
        
    return pd.DataFrame(processed_dec)

def get_earliest_lick(row):
    left = row['licks_left_start']
    right = row['licks_right_start']

    all_licks = []

    if isinstance(left, list):
        all_licks.extend([v for v in left if not np.isnan(v)])
    if isinstance(right, list):
        all_licks.extend([v for v in right if not np.isnan(v)])

    return min(all_licks) if all_licks else np.nan   

def filter_df(processed_dec):
    # filter tags
    filtered_df = processed_dec[(processed_dec['is_naive'] == False)]
    filtered_df = filtered_df[(filtered_df['no_lick'] == False)]
    filtered_df = filtered_df[(filtered_df['move_correct_spout'] == False)]   
         
    
    return filtered_df


def plot_decision_time_violin_outcome(M, config, subjectIdx, sessionIdx=-1, figure_id=None, show_plot=1, opto=False, side='both'):
    # figure meta
    rowspan, colspan = 2, 2
    fig_size = get_figsize_from_pdf_spec(rowspan, colspan, config['pdf_spec']['pdf_pg_opto_psychometric'])    
    fig, ax = plt.subplots(figsize=fig_size) 

    subject = config['list_config'][subjectIdx]['subject_name']

    is_avg = 0
    if sessionIdx != -1:
        decision_df = get_decision_df(M, sessionIdx)
        dates = M['dates'][sessionIdx]
    else:
        is_avg = 1
        dates = M['dates']
        combined_df = pd.DataFrame()
        for sessionIdx in range(0, len(dates)):
            decision_df = get_decision_df(M, sessionIdx)
            combined_df = pd.concat([combined_df, decision_df], ignore_index=True)
        decision_df = combined_df
    
    num_residuals = 5
    
    filtered_df = filter_df(decision_df)
    filtered_df['earliest_lick'] = filtered_df.apply(get_earliest_lick, axis=1) 
    filtered_df['reaction_time'] = filtered_df['earliest_lick']
    filtered_df['outcome'] = filtered_df['rewarded'].map({1: 'Correct', 0: 'Incorrect'})
    filtered_df['condition'] = filtered_df['is_opto'].map({0: 'Control', 1: 'Opto'})
    filtered_df['trial_type'] = filtered_df['trial_side'].map({'left': 'Left', 'right': 'Right'})
    
    
    ymin = -300
    ymax = 2000
   
      
    fig, ax = plt.subplots(figsize=fig_size)         
    sns.violinplot(x='outcome', y='reaction_time', hue='condition', data=filtered_df, order=['Correct', 'Incorrect'])  
    # sns.violinplot(
    #     x='outcome',
    #     y='reaction_time',
    #     hue='condition',
    #     data=filtered_df,
    #     order=['correct', 'incorrect'],           # Optional: enforce order
    #     hue_order=['control', 'opto'],            # Optional: enforce hue order
    #     inner='box',                              # Show box inside violins
    #     palette='Set2'                            # Optional: soft color palette
    # )   
    
    plt.title('Reaction Time by Outcome and Condition')
    plt.xlabel('Trial Outcome')
    plt.ylabel('Reaction Time (s)')
    plt.legend(title='Condition', loc='best')
    plt.ylim(ymin, ymax)  # Limit y-axis from yminx to ymax
    plt.tight_layout()
    plt.show()    
    
    plt.tight_layout() 
    
    if show_plot:
        plt.show()           
        
    subject = config['list_config'][subjectIdx]['subject_name']
    output_dir = os.path.join(config['paths']['figure_dir_local'] + subject)
    figure_id = f"{subject}_decision_time_violin_outcome"
    file_ext = '.pdf'
    filename = figure_id + file_ext
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)    
      
    return out_path    