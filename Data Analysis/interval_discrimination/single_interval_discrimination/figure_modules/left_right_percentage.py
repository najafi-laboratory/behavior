# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 15:08:54 2025

@author: timst
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import sem
from utils.util import get_figsize_from_pdf_spec

def plot_left_right_percentage(M, config, subjectIdx, show_plot=1):
    # figure meta
    rowspan, colspan = 2, 4
    fig_size = get_figsize_from_pdf_spec(rowspan, colspan, config['pdf_spec']['pdf_pg_cover'])    
    fig, ax = plt.subplots(figsize=fig_size) 
    
    # fig, ax = plt.subplots(figsize=(11, 6))
    max_sessions = 20
    # subject = M['subject']
    dates = M['dates']
    session_id = np.arange(len(dates)) + 1
    jitter_flag = M['jitter_flag']
    jitter_session = np.array([np.sum(j) for j in jitter_flag])
    jitter_session[jitter_session!=0] = 1
    # chemo_labels = M['Chemo']
    chemo_labels = []
    choice = []
    choice_right = []

    width = 0.5
    
    for i in range(len(dates)):
        
        decision = M['decision'][i]
        decision = np.concatenate(decision, axis=1)
        direction = decision[1 , :]
        left = sum(trial_type == 0 for trial_type in direction)
        right = sum(trial_type == 1 for trial_type in direction)
        choice.append(left/(left+right))
        choice_right.append(right/(left+right))
    
    ax.bar(
        session_id, choice_right,
        edgecolor='white',
        width=width,
        color='lightcoral',
        label='Right choice')
        # bottom = choice,    
    ax.bar(
        session_id, choice,
        edgecolor='white',
        width=width,
        bottom = choice_right,        
        color='lightblue',
        label='Left choice')

    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.hlines(0.5,0,len(dates)+1, linestyle='--' , color='silver' , lw = 0.5)
    #ax.yaxis.grid(True)
    ax.set_xlabel('training session')
    ax.set_ylabel('fraction of left choice')
    ax.set_ylim(0 , 1)
    ax.set_xticks(np.arange(len(dates))+1)
    ax.set_xticklabels(dates, rotation=45)
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    ax.set_title('left vs right choice fraction')
    ind = 0
    for xtick in ax.get_xticklabels():
        if jitter_session[ind] == 1:
            xtick.set_color('limegreen')
        # if chemo_labels[ind] == 1:
        #     xtick.set_color('red')
        ind = ind + 1


    
    if show_plot:
        plt.show()
        
    subject = config['list_config'][subjectIdx]['subject_name']
    output_dir = os.path.join(config['paths']['figure_dir_local'] + subject)
    figure_id = f"{subject}_left_right_percentage"
    file_ext = '.pdf'
    filename = figure_id + file_ext
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.close(fig)    

    # return {
    #     'figure_id': figure_id,
    #     'path': out_path,
    #     'caption': f"Left/Right percentage plot for {subject}",
    #     'subject': subject,
    #     'tags': ['performance', 'bias'],
    #     "layout": {
    #       "page": 0,
    #       "page_key": "pdf_pg_cover", 
    #       "row": 2,
    #       "col": 0,
    #       "rowspan": rowspan,
    #       "colspan": colspan,
    #     }        
    # }

    return out_path