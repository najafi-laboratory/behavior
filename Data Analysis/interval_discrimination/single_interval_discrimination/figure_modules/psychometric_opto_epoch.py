# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:54:44 2025

@author: timst
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import os
from utils.util import get_figsize_from_pdf_spec


# region_map = {
#     0: 'Control',
#     1: 'RLat',
#     2: 'LLat',
#     3: 'RIntA',
#     4: 'LIntA',
#     5: 'LPPC',
#     6: 'RPPC',
#     7: 'mPFC',
#     8: 'LPost',
#     9: 'RPost'
# }

region_map = {
    0: {'name': 'Control','location': 'center'},
    1: {'name': 'RLat',   'location': 'right'},
    2: {'name': 'LLat',   'location': 'left'},
    3: {'name': 'RIntA',  'location': 'right'},
    4: {'name': 'LIntA',  'location': 'left'},
    5: {'name': 'LPPC',   'location': 'left'},
    6: {'name': 'RPPC',   'location': 'right'},
    7: {'name': 'mPFC',   'location': 'center'},
    8: {'name': 'LPost',  'location': 'left'},
    9: {'name': 'RPost',  'location': 'right'},
}

left_regions = ['LIntA', 'LLat', 'LPPC', 'LPost']
right_regions = ['RIntA', 'RLat', 'RPPC', 'RPost']
center_regions = ['mPFC']

# location_colors = {
#     'left': '#1f77b4',     # blue
#     'right': '#d62728',    # red
#     'center': '#2ca02c'    # green
# }

# bin the data with timestamps.

def get_bin_stat(decision, session_settings, isi='post'):
    bin_size=100
    least_trials=1
    # set bins across isi range
    # short ISI: [50, 400, 750]ms.  associated with left lick
    # long ISI: [750, 1100, 1450]ms.  associated with right lick
    isi_long_mean = session_settings['ISILongMean_s'] * 1000
    bin_right = isi_long_mean + 400
    bins = np.arange(0, bin_right + bin_size, bin_size)
    bins = bins - bin_size / 2
    if isi=='pre':
        row = 4
    if isi=='post':
        row = 5
    bin_indices = np.digitize(decision[row,:], bins) - 1
    bin_mean = []
    bin_sem = []
    for i in range(len(bins)-1):
        direction = decision[1, bin_indices == i].copy()
        m = np.mean(direction) if len(direction) > least_trials else np.nan
        s = sem(direction) if len(direction) > least_trials else np.nan
        bin_mean.append(m)
        bin_sem.append(s)
    bin_mean = np.array(bin_mean)
    bin_sem  = np.array(bin_sem)
    bin_isi  = bins[:-1] + (bins[1]-bins[0]) / 2
    non_nan  = (1-np.isnan(bin_mean)).astype('bool')
    bin_mean = bin_mean[non_nan]
    bin_sem  = bin_sem[non_nan]
    bin_isi  = bin_isi[non_nan]
    return bin_mean, bin_sem, bin_isi


def separate_fix_jitter(decision):
    decision_fix = decision[:,decision[3,:]==0]
    decision_jitter = decision[:,decision[3,:]==1]
    decision_chemo = decision[:,decision[3,:]==2]
    decision_opto = decision[:,decision[3,:]==3]
    decision_opto_left = decision[:,decision[6,:]==1]
    decision_opto_right = decision[:,decision[6,:]==2]
    return decision_fix, decision_jitter, decision_chemo, decision_opto, decision_opto_left, decision_opto_right


def get_decision(subject_session_data, sessionIdx):
    decision = subject_session_data['decision'][sessionIdx]
    # decision = [np.concatenate(d, axis=1) for d in decision]
    decision = np.concatenate(decision, axis=1)
    jitter_flag = subject_session_data['jitter_flag'][sessionIdx]
    # jitter_flag = np.concatenate(jitter_flag).reshape(1,-1)
    jitter_flag = np.array(jitter_flag).reshape(1,-1)
    # opto_flag = subject_session_data['opto_flag']
    opto_flag = subject_session_data['opto_trial'][sessionIdx]
    opto_flag = np.array(opto_flag).reshape(1,-1)
    jitter_flag[0 , :] = jitter_flag[0 , :] + opto_flag[0 , :]*3
    # jitter_flag = jitter_flag + opto_flag*3
    # jitter_flag = [j + o * 3 for j, o in zip(jitter_flag, opto_flag)]
    opto_side = subject_session_data['opto_side'][sessionIdx]
    opto_side = np.array(opto_side).reshape(1,-1)
    outcomes = subject_session_data['outcomes'][sessionIdx]
    all_trials = 0
    # chemo_labels = subject_session_data['Chemo'][sessionIdx]
    # for j in range(len(chemo_labels)):
    #     if chemo_labels[j] == 1:
    #         jitter_flag[0 , all_trials:all_trials+len(outcomes[j])] = 2*np.ones(len(outcomes[j]))
    #     all_trials += len(outcomes[j])
    isi_pre_emp = subject_session_data['isi_pre_emp'][sessionIdx]
    # isi_pre_emp = np.concatenate(isi_pre_emp).reshape(1,-1)
    isi_pre_emp = np.array(isi_pre_emp).reshape(1,-1)
    
    isi_post_emp = subject_session_data['isi_post_emp'][sessionIdx]
    isi_post_emp = np.array(isi_post_emp).reshape(1,-1)
    # isi_post_emp = np.concatenate(isi_post_emp).reshape(1,-1)
    decision = np.concatenate([decision, jitter_flag, isi_pre_emp, isi_post_emp, opto_side], axis=0)
    non_nan = (1-np.isnan(np.sum(decision, axis=0))).astype('bool')
    decision = decision[:,non_nan]
    # row 0: time.
    # row 1: direction.
    # row 2: correctness.
    # row 3: jitter flag.
    # row 4: pre pert isi.
    # row 5: post pert isi.
    # row 6: opto side
    decision_fix, decision_jitter, decision_chemo, decision_opto, decision_opto_left, decision_opto_right = separate_fix_jitter(decision)
    return decision_fix, decision_jitter, decision_chemo, decision_opto, decision_opto_left, decision_opto_right


def plot_psychometric_opto_epoch(M, config, subjectIdx, sessionIdx, figure_id=None, show_plot=1):
   
    
 
    # figure meta
    rowspan, colspan = 2, 2
    fig_size = get_figsize_from_pdf_spec(rowspan, colspan, config['pdf_spec']['pdf_pg_opto_psychometric'])    
    fig, ax = plt.subplots(figsize=fig_size)    
    # fig, ax = plt.subplots(figsize=(4, 3))    
   
    
    OptoRegionIdx = M['raw'][sessionIdx]['TrialSettings'][0]['GUI']['OptoRegion']
    OptoRegion = region_map[OptoRegionIdx]['name']
    OptoSide = region_map[OptoRegionIdx]['location']
    
    # OptoPwrs = M['raw'][sessionIdx]['TrialSettings'][0]['GUI']['OptoRegion']
    
    # OptoSide = M['opto_side'][sessionIdx][0]    
    
    subject = config['list_config'][subjectIdx]['subject_name']
    
    session_settings = M['session_settings'][sessionIdx]
    isi_short_mean = session_settings['ISIShortMean_s'] * 1000
    isi_long_mean = session_settings['ISILongMean_s'] * 1000
    # isi_orig = session_settings['ISIOrig_s'] * 1000
    isi_orig = (isi_short_mean + isi_long_mean) / 2
    
    decision_fix, decision_jitter, decision_chemo, decision_opto, decision_opto_left, decision_opto_right = get_decision(M, sessionIdx)
    bin_mean_fix, bin_sem_fix, bin_isi_fix = get_bin_stat(decision_fix, session_settings)
    bin_mean_jitter, bin_sem_jitter, bin_isi_jitter = get_bin_stat(decision_jitter, session_settings)
    bin_mean_chemo, bin_sem_chemo, bin_isi_chemo = get_bin_stat(decision_chemo, session_settings)
    bin_mean_opto, bin_sem_opto, bin_isi_opto = get_bin_stat(decision_opto, session_settings)
    
    bin_mean_opto_left, bin_sem_opto_left, bin_isi_opto_left = get_bin_stat(decision_opto_left, session_settings)
    bin_mean_opto_right, bin_sem_opto_right, bin_isi_opto_right = get_bin_stat(decision_opto_right, session_settings)
    
    # fix
    ax.plot(
        bin_isi_fix,
        bin_mean_fix,
        color='black', marker='.', label='Control', markersize=4)
    ax.fill_between(
        bin_isi_fix,
        bin_mean_fix - bin_sem_fix,
        bin_mean_fix + bin_sem_fix,
        color='grey', alpha=0.2)
    
    # opto
    # all
    # ax.plot(
    #     bin_isi_opto,
    #     bin_mean_opto,
    #     color='indigo', marker='.', label='opto', markersize=4)
    # ax.fill_between(
    #     bin_isi_opto,
    #     bin_mean_opto - bin_sem_opto,
    #     bin_mean_opto + bin_sem_opto,
    #     color='violet', alpha=0.2)    
    
    
    # left_label = 'opto left'
    # right_label = 'opto right'
    
    # if subject not in ['LCHR_TS01_opto', 'LCHR_TS02_opto']:
    #     # left_label = subject
    #     # right_label = subject
    #     left_label = 'opto'
    #     right_label = 'opto'
    
    # left_label = 'Opto ' + OptoRegion
    # right_label = 'Opto ' + OptoRegion
    
    if OptoRegion in left_regions:
        mean_color = 'blue'
        sem_color = 'violet'
    elif OptoRegion in right_regions:
        mean_color = 'green'
        sem_color = 'lightgreen'
    else:
        mean_color = 'indigo'
        sem_color = 'violet'        
    
    label = 'Opto ' + OptoRegion
    
    if len(bin_isi_opto) > 0:
        ax.plot(
            bin_isi_opto,
            bin_mean_opto,
            color=mean_color, marker='.', label=label, markersize=4)
        ax.fill_between(
            bin_isi_opto,
            bin_mean_opto - bin_sem_opto,
            bin_mean_opto + bin_sem_opto,
            color=sem_color, alpha=0.2)       
    
    # if len(bin_isi_opto_left) > 0:
    #     # left
    #     ax.plot(
    #         bin_isi_opto_left,
    #         bin_mean_opto_left,
    #         color='blue', marker='.', label=left_label, markersize=4)
    #     ax.fill_between(
    #         bin_isi_opto_left,
    #         bin_mean_opto_left - bin_sem_opto_left,
    #         bin_mean_opto_left + bin_sem_opto_left,
    #         color='violet', alpha=0.2)   

    # if len(bin_isi_opto_right) > 0:
    #     # right
    #     ax.plot(
    #         bin_isi_opto_right,
    #         bin_mean_opto_right,
    #         color='green', marker='.', label=right_label, markersize=4)
    #     ax.fill_between(
    #         bin_isi_opto_right,
    #         bin_mean_opto_right - bin_sem_opto_right,
    #         bin_mean_opto_right + bin_sem_opto_right,
    #         color='lightgreen', alpha=0.2)   
    
    
     
    # ax.plot(
    #     bin_isi_jitter,
    #     bin_mean_jitter,
    #     color='limegreen', marker='.', label='jitter', markersize=4)
    # ax.fill_between(
    #     bin_isi_jitter,
    #     bin_mean_jitter - bin_sem_jitter,
    #     bin_mean_jitter + bin_sem_jitter,
    #     color='limegreen', alpha=0.2)
    # ax.plot(
    #     bin_isi_chemo,
    #     bin_mean_chemo,
    #     color='red', marker='.', label='chemo', markersize=4)
    # ax.fill_between(
    #     bin_isi_chemo,
    #     bin_mean_chemo - bin_sem_chemo,
    #     bin_mean_chemo + bin_sem_chemo,
    #     color='red', alpha=0.2)
    # ax.plot(
    #     bin_isi_opto,
    #     bin_mean_opto,
    #     color='dodgerblue', marker='.', label='opto', markersize=4)
    # ax.fill_between(
    #     bin_isi_opto,
    #     bin_mean_opto - bin_sem_opto,
    #     bin_mean_opto + bin_sem_opto,
    #     color='dodgerblue', alpha=0.2)
    
    x_left = isi_short_mean - 100
    x_right = isi_long_mean + 100
    cat = isi_orig
    x_left = 0
    x_right = 2*cat
    
    ax.vlines(
        cat, 0.0, 1.0,
        linestyle='--', color='mediumseagreen',
        label='Category Boundary')
    ax.hlines(0.5, x_left, x_right, linestyle='--', color='grey')
    # ax.vlines(500, 0.0, 1.0, linestyle=':', color='grey')
    ax.tick_params(tick1On=False)
    ax.tick_params(axis='x', rotation=45)    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([x_left,x_right])
    ax.set_ylim([-0.05,1.05])
    # ax.set_xticks(np.arange(6)*200)
    # ax.set_xticks(np.arange(11)*150)
    ax.set_xticks(np.arange(0,x_right,250))
    ax.set_yticks(np.arange(5)*0.25)
    ax.set_xlabel('ISI')
    ax.set_ylabel('Prob. of Choosing the Right Side (mean$\pm$sem)')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    start_from = []
    start_date = []
    date = M['dates'][sessionIdx]
    if start_from=='start_date':
        ax.set_title('average psychometric function from ' + start_date)
    elif start_from=='non_naive':
        ax.set_title('average psychometric function non-naive')
    else:
        ax.set_title('Psychometric Function ' + date)
        
        
    if show_plot:
        plt.show()
        
    subject = config['list_config'][subjectIdx]['subject_name']
    output_dir = os.path.join(config['paths']['figure_dir_local'] + subject)
    
    if figure_id is None:
        figure_id = f"{subject}_psychometric_opto_epoch_s{sessionIdx}"
    
    # figure_id = f"{subject}_psychometric_opto_epoch"
    
    file_ext = '.pdf'
    filename = figure_id + file_ext
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)    

    # return {
    #     'figure_id': figure_id,
    #     'path': out_path,
    #     'caption': f"Psychometric plot for {subject}",
    #     'subject': subject,
    #     'tags': ['performance', 'bias'],
    #     "layout": {
    #       "page": 1,
    #       "page_key": "pdf_pg_opto_psychometric", 
    #       "row": 0,
    #       "col": 0,
    #       "rowspan": rowspan,
    #       "colspan": colspan,
    #     }        
    # }     
    
    return out_path