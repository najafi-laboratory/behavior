# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 15:08:54 2025

@author: timst
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.util import get_figsize_from_pdf_spec

states = [
    'Reward',
    'Punish']
colors = [
    'limegreen',
    'coral']

# states = [
#     'Reward',
#     'RewardNaive',
#     'ChangingMindReward',
#     'Punish',
#     'PunishNaive']
# colors = [
#     'limegreen',
#     'springgreen',
#     'dodgerblue',
#     'coral',
#     'violet']

def get_side_outcomes(outcomes, states):
    num_session = len(outcomes)
    counts = np.zeros((num_session, len(states)))
    for i in range(num_session):
        for j in range(len(states)):
            counts[i,j] = np.sum(np.array(outcomes[i]) == states[j])
        counts[i,:] = counts[i,:] / (np.sum(counts[i,:])+1e-5)
    return counts

def plot_outcomes(M, config, subjectIdx, show_plot=1):
    
    # figure meta
    rowspan, colspan = 2, 4
    fig_size = get_figsize_from_pdf_spec(rowspan, colspan, config['pdf_spec']['pdf_pg_cover'])    
    fig, ax = plt.subplots(figsize=fig_size)    
    # fig, ax = plt.subplots(figsize=(4, 3))
    # fig, ax = plt.subplots(figsize=(11, 6))
    
    max_sessions=25                
    outcomes_left = M['outcomes_left']
    outcomes_right = M['outcomes_right']
    dates = M['dates']
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    dates = dates[start_idx:]
    outcomes_left = outcomes_left[start_idx:]
    outcomes_right = outcomes_right[start_idx:]
    session_id = np.arange(len(outcomes_left)) + 1    
    left_counts = get_side_outcomes(outcomes_left, states)
    right_counts = get_side_outcomes(outcomes_right, states)    
    left_bottom = np.cumsum(left_counts, axis=1)
    left_bottom[:,1:] = left_bottom[:,:-1]
    left_bottom[:,0] = 0        
    right_bottom = np.cumsum(right_counts, axis=1)
    right_bottom[:,1:] = right_bottom[:,:-1]
    right_bottom[:,0] = 0       
    width = 0.25
    for i in range(len(states)):
        # Plot the left bars
        ax.bar(
            session_id - width / 2, left_counts[:,i],  # Shift left by width/2
            bottom=left_bottom[:,i],
            edgecolor='white',
            width=width,
            color=colors[i],
            label=states[i])        
        # Plot the right bars
        ax.bar(
            session_id + width / 2, right_counts[:,i],  # Shift right by width/2
            bottom=right_bottom[:,i],
            edgecolor='white',
            width=width,
            color=colors[i])  # Optionally update label for right bars  

   
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.hlines(0.5,0,len(dates)+1, linestyle='--' , color='silver' , lw = 0.5)
    ax.hlines(0.75,0,len(dates)+1, linestyle='--' , color='silver' , lw = 0.5)    
    ax.yaxis.grid(False)
    ax.set_xlabel('training session')
    ax.set_ylabel('number of trials')
    top_labels = []
    for i in range(len(dates)):
        top_labels.append('L')  # Label for the first bar
        top_labels.append('R')  # Label for the second bar
    # Update x-ticks and x-tick labels
    tick_positions_bottom = np.arange(len(outcomes_left))+1
    tick_positions_top = np.repeat(tick_positions_bottom, 2)  # Repeat x positions for each set of bars (top labels)
    ax.set_xticks(tick_positions_bottom)
    # ax.set_xticks(tick_positions_top)  # Set the tick positions for the top labels
    # ax.set_xticklabels(top_labels, rotation=45, ha='right')  # Set the top labels and rotate    
    ax.set_yticks(np.arange(6)*0.2)
    ax.set_xticklabels(dates, rotation=45)
    

    
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    ax.set_title('reward/punish percentage for completed trials per side')
    
    if show_plot:
        plt.show()
    
    
    subject = config['list_config'][subjectIdx]['subject_name']
    output_dir = os.path.join(config['paths']['figure_dir_local'] + subject)
    figure_id = f"{subject}_outcomes"
    file_ext = '.pdf'
    filename = figure_id + file_ext
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=300)
    # fig.savefig(out_path, dpi=300)
    plt.close(fig)    

    # return {
    #     'figure_id': figure_id,
    #     'path': out_path,
    #     'caption': f"Outcome plot for {subject}",
    #     'subject': subject,
    #     'tags': ['performance', 'bias'],
    #     "layout": {
    #       "page": 0,
    #       "page_key": "pdf_pg_cover", 
    #       "row": 0,
    #       "col": 0,
    #       "rowspan": rowspan,
    #       "colspan": colspan,
    #     }        
    # }
    return out_path
