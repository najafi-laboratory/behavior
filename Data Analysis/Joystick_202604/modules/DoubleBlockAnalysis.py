# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 08:33:19 2025

@author: saminnaji3
"""

# %% importing_all_packages_and_modules
import tkinter as tk
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import fitz
import warnings
import numpy as np

from plot import block_realize
from plot import double_block_analysis
from plot import trajectories
from plot import trajectories_first_push
from plot import Number_of_trials
from plot import single_trial_adaptation
from plot import effective_probe_analysis
from plot import opto_single_trial_adaptation
from plot import single_trial_adaptation_hist
from plot import single_trial_adaptation_hist_new
from plot import block_behavior_opto_session
from plot import interval_behavior_opto_session
from plot import block_behavior
from plot import interval_behavior
from plot import aligned_trajectories 
from plot import plot_distribution
from plot import block_behavior_poster
from plot import adaptive_dist
from plot import block_interval_presentation
warnings.filterwarnings('ignore')

st_vg = ['_all' , '_ST' , '_VG']
st_vg_id = [[0] , [0, 1, 2]]



def save_temp_fig(fig, report):
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    temp_fig = fitz.open(fname)
    report.insert_pdf(temp_fig)
    temp_fig.close()
    os.remove(fname)
    
def run(session_data_1, block_data, interval_data, subject, output_dir_onedrive, last_day, seperate = 0, individual_trajectory = 0):
    report = fitz.open()
    
    # # performance
    # print('Plotting General Performance')
    # fig = plt.figure(layout='constrained', figsize=(30, 15))
    # gs = GridSpec(6, 7, figure=fig)
    # double_block_analysis.run_outcome(plt.subplot(gs[0:2, 0:6]) , session_data_1, 0)
    # Number_of_trials.run_trial_num(plt.subplot(gs[4:6, 0:2]) , session_data_1  , 0)
    # Number_of_trials.run_reward_num(plt.subplot(gs[2:4, 0:2]) , session_data_1 , 0)
    # Number_of_trials.run_delay(plt.subplot(gs[0, 6]) , session_data_1 , 0)
    
    # if seperate == 0:
    #     double_block_analysis.run_trajectory([plt.subplot(gs[j, 2:4]) for j in range(2 , 4)]  , session_data_1,block_data, 0)
    #     trajectories.run_trajectory(plt.subplot(gs[4, 2:4]) , session_data_1, 0 , ' Rewarded trials (grand average of all sessions)' )
    #     trajectories_first_push.run_trajectory(plt.subplot(gs[5, 2:4]), session_data_1, 0 , ' Rewarded trials (all pooled average)')
    #     Number_of_trials.run_dist([plt.subplot(gs[j, 4:6]) for j in range(2 , 4)] , session_data_1 , block_data , 0)
    #     Number_of_trials.run_dist_rewarded([plt.subplot(gs[j, 4:6]) for j in range(4 , 6)] , session_data_1 , block_data , 0)
    # else:
    #     double_block_analysis.run_trajectory([plt.subplot(gs[j, 2]) for j in range(2 , 4)]  , session_data_1,block_data, 1)
    #     trajectories.run_trajectory(plt.subplot(gs[4, 2]) , session_data_1, 1 , ' Rewarded trials (grand average of ST sessions)' )
    #     trajectories_first_push.run_trajectory(plt.subplot(gs[5, 2]), session_data_1, 1 , ' Rewarded trials (all ST pooled average)')
    #     Number_of_trials.run_dist([plt.subplot(gs[j, 3]) for j in range(2 , 4)] , session_data_1 , block_data , 1)
    #     Number_of_trials.run_dist_rewarded([plt.subplot(gs[j, 3]) for j in range(4 , 6)] , session_data_1 , block_data , 1)
        
    #     double_block_analysis.run_trajectory([plt.subplot(gs[j, 4]) for j in range(2 , 4)]  , session_data_1,block_data, 2)
    #     trajectories.run_trajectory(plt.subplot(gs[4, 4]) , session_data_1, 2 , ' Rewarded trials (grand average of VG)' )
    #     trajectories_first_push.run_trajectory(plt.subplot(gs[5, 4]), session_data_1, 2 , ' Rewarded trials (all VG pooled average)')
    #     Number_of_trials.run_dist([plt.subplot(gs[j, 5]) for j in range(2 , 4)] , session_data_1 , block_data , 2)
    #     Number_of_trials.run_dist_rewarded([plt.subplot(gs[j, 5]) for j in range(4 , 6)] , session_data_1 , block_data , 2)
    # plt.suptitle(subject)
    # save_temp_fig(fig, report)
    
    #Single Trial Adaptation
    print('Plotting Single Trial Adaptation')
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(4, 5, figure=fig)
    press_type = ['Short' , 'Long' , 'Reward' , 'EarlyPress2' , 'LatePress2']
    for st in st_vg_id[seperate]:
        single_trial_adaptation.run_dist(plt.subplot(gs[0, st]) ,session_data_1, block_data, st)
    for i in range(len(press_type)):
        single_trial_adaptation.run_count([plt.subplot(gs[1, i]), plt.subplot(gs[2, i])],plt.subplot(gs[0, 3]), session_data_1, block_data, press_type[i])
        single_trial_adaptation.run_delay(plt.subplot(gs[3, i]), session_data_1, block_data, press_type[i])
    save_temp_fig(fig, report)
    
    print('Plotting Single Trial Adaptation Hist')
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(4, 6, figure=fig)
    if seperate == 0:
        single_trial_adaptation_hist.run([plt.subplot(gs[0,0]),plt.subplot(gs[1,0])], session_data_1, block_data, [0.4, 0.1], 0, 'short', 0)
        single_trial_adaptation_hist.run([plt.subplot(gs[2,0]),plt.subplot(gs[3,0]), plt.subplot(gs[0,2]),plt.subplot(gs[1,2])], session_data_1, block_data, [3.5, 3.5], 0, 'short', 1, 1)
        single_trial_adaptation_hist.run([plt.subplot(gs[0,1]),plt.subplot(gs[1,1])], session_data_1, block_data, [0.4, 0.1], 0, 'long', 0)
        single_trial_adaptation_hist.run([plt.subplot(gs[2,1]),plt.subplot(gs[3,1]), plt.subplot(gs[1,2]),plt.subplot(gs[3,2])], session_data_1, block_data, [2, 2], 0, 'long', 1, 1)
    else:
        for st in [1, 2]:
            shift = 3*(st-1)
            single_trial_adaptation_hist.run([plt.subplot(gs[0,shift]),plt.subplot(gs[1,shift])], session_data_1, block_data, [0.4, 0.1], st, 'short', 0)
            single_trial_adaptation_hist.run([plt.subplot(gs[2,shift]),plt.subplot(gs[3,shift]), plt.subplot(gs[0,2 + shift]),plt.subplot(gs[1,2 + shift])], session_data_1, block_data, [3.5, 3.5], st, 'short', 1, 1)
            single_trial_adaptation_hist.run([plt.subplot(gs[0,1 + shift]),plt.subplot(gs[1,1 + shift])], session_data_1, block_data, [0.4, 0.1], st, 'long', 0)
            single_trial_adaptation_hist.run([plt.subplot(gs[2,1 + shift]),plt.subplot(gs[3,1 + shift]), plt.subplot(gs[2,2 + shift]),plt.subplot(gs[3,2 + shift])], session_data_1, block_data, [2, 2], st, 'long', 1, 1)
    plt.suptitle(subject)
    save_temp_fig(fig, report)
    
    # within_session_adaptation
    print('Plotting Within Session Adaptation')
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(6, 6, figure=fig)
    if seperate == 0:
        double_block_analysis.run_block_len([plt.subplot(gs[j, 0:3]) for j in range(4 , 6)]  , session_data_1,block_data, 0)
        double_block_analysis.run_delay([plt.subplot(gs[j, 0:3]) for j in range(0 , 2)]  ,plt.subplot(gs[0:2, 3]), session_data_1,block_data, 0)
        double_block_analysis.run_delay_rewarded([plt.subplot(gs[j, 0:3]) for j in range(2, 4)]  ,plt.subplot(gs[2:4, 3]), session_data_1,block_data, 0)
    else:
        double_block_analysis.run_block_len([plt.subplot(gs[j, 0:2]) for j in range(4 , 6)]  , session_data_1,block_data, 1)
        double_block_analysis.run_delay([plt.subplot(gs[j, 0:2]) for j in range(0 , 2)]  ,plt.subplot(gs[0:2, 2]), session_data_1,block_data, 1)
        double_block_analysis.run_delay_rewarded([plt.subplot(gs[j, 0:2]) for j in range(2, 4)]  ,plt.subplot(gs[2:4, 2]), session_data_1,block_data, 1)
        double_block_analysis.run_block_len([plt.subplot(gs[j, 3:5]) for j in range(4 , 6)]  , session_data_1,block_data, 2)
        double_block_analysis.run_delay([plt.subplot(gs[j, 3:5]) for j in range(0 , 2)]  ,plt.subplot(gs[0:2, 5]), session_data_1,block_data, 2)
        double_block_analysis.run_delay_rewarded([plt.subplot(gs[j, 3:5]) for j in range(2, 4)]  ,plt.subplot(gs[2:4, 5]), session_data_1,block_data, 2)
    plt.suptitle(subject)
    save_temp_fig(fig, report)
    
    # probe
    print('Plotting Probe Analysis')
    if seperate == 0:
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 4, figure=fig)
        effective_probe_analysis.run_grand_dist([plt.subplot(gs[0:2, 1]) ,plt.subplot(gs[0:2, 2])] , session_data_1,block_data, 0)
        effective_probe_analysis.run_grand_trajectory([plt.subplot(gs[j, 0]) for j in range(2 , 4)]  , session_data_1,block_data , 0 , 0)
        effective_probe_analysis.run_grand_trajectory([plt.subplot(gs[j, 0]) for j in range(2 , 4)]  , session_data_1,block_data , 1 , 0)
        effective_probe_analysis.run_grand_trajectory([plt.subplot(gs[j, 0]) for j in range(2 , 4)]  , session_data_1,block_data , 2 , 0)
        effective_probe_analysis.run_grand_trajectory_first_press_rewarded([plt.subplot(gs[j, 1]) for j in range(2 , 4)]  , session_data_1 , 0 , 0)
        effective_probe_analysis.run_grand_trajectory_first_press_rewarded([plt.subplot(gs[j, 1]) for j in range(2 , 4)]  , session_data_1 , 1 , 0)
        effective_probe_analysis.run_grand_trajectory_first_press_rewarded([plt.subplot(gs[j, 1]) for j in range(2 , 4)]  , session_data_1, 2 , 0)
        effective_probe_analysis.run_grand_trajectory_first_press([plt.subplot(gs[j, 2]) for j in range(2 , 4)]  , session_data_1 , 0 , 0)
        effective_probe_analysis.run_grand_trajectory_first_press([plt.subplot(gs[j, 2]) for j in range(2 , 4)]  , session_data_1 , 1 , 0)
        effective_probe_analysis.run_grand_trajectory_first_press([plt.subplot(gs[j, 2]) for j in range(2 , 4)]  , session_data_1, 2 , 0)
        effective_probe_analysis.run_grand_outcome(plt.subplot(gs[0, 0]) , session_data_1 , 'short', 0)
        effective_probe_analysis.run_grand_outcome(plt.subplot(gs[1, 0]) , session_data_1 , 'long', 0)
        plt.suptitle(subject + ' All Sessions')
        save_temp_fig(fig, report)
    else:
        label = [' Self Timed Sessions' , ' Visually Guided Sessions']
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 6, figure=fig)
        for st in [1, 2]:
            effective_probe_analysis.run_grand_dist([plt.subplot(gs[0:2, 1+ 3*(st-1)]) ,plt.subplot(gs[0:2, 2+ 3*(st-1)])] , session_data_1,block_data, st)
            effective_probe_analysis.run_grand_trajectory([plt.subplot(gs[j, 3*(st-1)]) for j in range(2 , 4)]  , session_data_1,block_data , 0 , st)
            effective_probe_analysis.run_grand_trajectory([plt.subplot(gs[j, 3*(st-1)]) for j in range(2 , 4)]  , session_data_1,block_data , 1 , st)
            effective_probe_analysis.run_grand_trajectory([plt.subplot(gs[j, 3*(st-1)]) for j in range(2 , 4)]  , session_data_1,block_data , 2 , st)
            effective_probe_analysis.run_grand_trajectory_first_press_rewarded([plt.subplot(gs[j, 1 + 3*(st-1)]) for j in range(2 , 4)]  , session_data_1 , 0 , st)
            effective_probe_analysis.run_grand_trajectory_first_press_rewarded([plt.subplot(gs[j, 1 + 3*(st-1)]) for j in range(2 , 4)]  , session_data_1 , 1 , st)
            effective_probe_analysis.run_grand_trajectory_first_press_rewarded([plt.subplot(gs[j, 1 + 3*(st-1)]) for j in range(2 , 4)]  , session_data_1, 2 , st)
            effective_probe_analysis.run_grand_trajectory_first_press([plt.subplot(gs[j, 2 + 3*(st-1)]) for j in range(2 , 4)]  , session_data_1 , 0 , st)
            effective_probe_analysis.run_grand_trajectory_first_press([plt.subplot(gs[j, 2 + 3*(st-1)]) for j in range(2 , 4)]  , session_data_1 , 1 , st)
            effective_probe_analysis.run_grand_trajectory_first_press([plt.subplot(gs[j, 2 + 3*(st-1)]) for j in range(2 , 4)]  , session_data_1, 2 , st)
            effective_probe_analysis.run_grand_outcome(plt.subplot(gs[0, 3*(st-1)]) , session_data_1 , 'short'+ label[st-1], st)
            effective_probe_analysis.run_grand_outcome(plt.subplot(gs[1, 3*(st-1)]) , session_data_1 , 'long'+ label[st-1], st)
        plt.suptitle(subject )
        save_temp_fig(fig, report)
        
    print('new')
        
    # within Block Delay/Outcome Adaptation
    print('Plotting within Block Delay/Outcome Adaptation')
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    if seperate == 1:
        gs = GridSpec(6, 3, figure=fig)
        block_behavior_poster.run_all_blocks([plt.subplot(gs[0, i]) for i in range(3)], session_data_1, block_data, st_seperate = seperate)
        block_behavior_poster.run_all_blocks([[plt.subplot(gs[j, i]) for i in range(3)] for j in range(1,6)], session_data_1, block_data ,outcome_ref='outcome', st_seperate = seperate)
    else:
        gs = GridSpec(6, 3, figure=fig)
        block_behavior_poster.run_all_blocks([plt.subplot(gs[0, i]) for i in [1]], session_data_1, block_data, st_seperate = seperate)
        block_behavior_poster.run_all_blocks([[plt.subplot(gs[j, i]) for i in [1]] for j in range(1,6)], session_data_1, block_data ,outcome_ref='outcome', st_seperate = seperate)
    plt.suptitle(subject)
    save_temp_fig(fig, report)
    
    # # within Block Delay/Outcome Adaptation
    # print('Plotting within Block Delay/Outcome Adaptation')
    # fig = plt.figure(layout='constrained', figsize=(30, 15))
    # if seperate == 1:
    #     gs = GridSpec(6, 3, figure=fig)
    #     block_interval_presentation.run_all_blocks([plt.subplot(gs[0, i]) for i in range(3)], session_data_1, block_data, st_seperate = seperate)
    #     #block_interval_presentation.run_all_blocks([[plt.subplot(gs[j, i]) for i in range(3)] for j in range(1,6)], session_data_1, block_data ,outcome_ref='outcome', st_seperate = seperate)
    # else:
    #     gs = GridSpec(6, 3, figure=fig)
    #     block_interval_presentation.run_all_blocks([plt.subplot(gs[0, i]) for i in [1]], session_data_1, block_data, st_seperate = seperate)
    #     #block_interval_presentation.run_all_blocks([[plt.subplot(gs[j, i]) for i in [1]] for j in range(1,6)], session_data_1, block_data ,outcome_ref='outcome', st_seperate = seperate)
    # plt.suptitle(subject)
    # save_temp_fig(fig, report)
    
    
    # # within Block dist adaptation
    # print('Plotting within Block dist Adaptation')
    # fig = plt.figure(layout='constrained', figsize=(30, 15))
    # if seperate == 1:
    #     gs = GridSpec(4, 3, figure=fig)
    #     adaptive_dist.run_all_blocks([[plt.subplot(gs[j, i]) for i in range(3)] for j in range(4)], session_data_1, block_data, st_seperate = seperate)
    #     #block_behavior_poster.run_all_blocks([[plt.subplot(gs[j, i]) for i in range(3)] for j in range(1,6)], session_data_1, block_data ,outcome_ref='outcome', st_seperate = seperate)
    # else:
    #     gs = GridSpec(4, 3, figure=fig)
    #     adaptive_dist.run_all_blocks([[plt.subplot(gs[i, 1]) for i in range(4)]], session_data_1, block_data, st_seperate = seperate)
    #     #block_behavior_poster.run_all_blocks([[plt.subplot(gs[j, i]) for i in [1]] for j in range(1,6)], session_data_1, block_data ,outcome_ref='outcome', st_seperate = seperate)
    # plt.suptitle(subject)
    # save_temp_fig(fig, report)
    
    # within Block Intervals Adaptation
    print('Plotting within Block Intervals Adaptation')
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    if seperate == 1:
        gs = GridSpec(7, 3, figure=fig)
        interval_behavior.run_all_blocks([[plt.subplot(gs[j, i]) for i in range(3)] for j in range(7)], session_data_1, block_data, interval_data, st_seperate = seperate)
    else:
        gs = GridSpec(7, 3, figure=fig)
        interval_behavior.run_all_blocks([[plt.subplot(gs[j, i]) for i in [1]] for j in range(7)], session_data_1, block_data, interval_data, st_seperate = seperate)
    plt.suptitle(subject)
    save_temp_fig(fig, report)
    
    if individual_trajectory:
        # within Block Intervals Adaptation
        print('Plotting Trajectories Alignments')
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 10, figure=fig)
        aligned_trajectories.aligned_data([plt.subplot(gs[0, i])  for i in range(0,10)],
                                          [plt.subplot(gs[1, i])  for i in range(0,10)],
                                          np.nan, np.nan,
                                          session_data_1, 1,np.nan, st = 0)
        save_temp_fig(fig, report)
    
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 10, figure=fig)
        aligned_trajectories.aligned_data_mega_session([plt.subplot(gs[1, i])  for i in range(0,10)],
                                          [plt.subplot(gs[2, i])  for i in range(0,10)],
                                          session_data_1, 1,np.nan, st = 0)
        save_temp_fig(fig, report)
    
    if seperate:
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(9, 4, figure=fig)
        plot_distribution.run_dist([plt.subplot(gs[i, 1])  for i in range(0,9)],
                                   np.nan, 
                                   np.nan, 
                                   np.nan,
                                   session_data_1, block_data, interval_data, plot_type = 'double', st = 1)
        
        plot_distribution.run_dist([plt.subplot(gs[i, 2])  for i in range(0,9)],
                                   np.nan, 
                                   np.nan, 
                                   np.nan,
                                   session_data_1, block_data, interval_data, plot_type = 'double', st = 2)
        plt.suptitle('self timed/visually guided')
        save_temp_fig(fig, report)
    else:
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(9, 4, figure=fig)
        plot_distribution.run_dist([plt.subplot(gs[i, 2])  for i in range(0,9)],
                                   np.nan, 
                                   np.nan, 
                                   np.nan,
                                   session_data_1, block_data, interval_data, plot_type = 'double', st = 0)
        plt.suptitle('All sessions distributions')
        save_temp_fig(fig, report)
    
    report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + '_double_block_analysis.pdf')
    report.close()
        
def run_opto(session_data_1, block_data,interval_data, subject, output_dir_onedrive, last_day, seperate = 0, individual_trajectory = 0):
    for st in st_vg_id[seperate]:
        report = fitz.open()
        st_label = 0
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 4, figure=fig)
        single_trial_adaptation_hist.run([plt.subplot(gs[0,0]),plt.subplot(gs[1,0])], session_data_1, block_data, [0.4, 0.1], st_label, 'short', 0)
        single_trial_adaptation_hist.run([plt.subplot(gs[0,1]),plt.subplot(gs[1,1]), plt.subplot(gs[2,1]),plt.subplot(gs[3,1])], session_data_1, block_data, [3.5, 3.5], st_label, 'short', 1, 1)
        single_trial_adaptation_hist.run([plt.subplot(gs[0,2]),plt.subplot(gs[1,2])], session_data_1, block_data, [0.4, 0.1], st_label, 'long', 0)
        single_trial_adaptation_hist.run([plt.subplot(gs[0,3]),plt.subplot(gs[1,3]), plt.subplot(gs[2,3]),plt.subplot(gs[3,3])], session_data_1, block_data, [3.5, 3.5], st_label, 'long', 1, 1)
        save_temp_fig(fig, report)
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 4, figure=fig)
        single_trial_adaptation_hist_new.run([plt.subplot(gs[0,1]),plt.subplot(gs[1,1]), plt.subplot(gs[2,1]),plt.subplot(gs[3,1])], session_data_1, block_data, [3.5, 3.5], st_label, 'short', 1, 1)
        single_trial_adaptation_hist_new.run([plt.subplot(gs[0,2]),plt.subplot(gs[1,2]), plt.subplot(gs[2,2]),plt.subplot(gs[3,2])], session_data_1, block_data, [3.5, 3.5], st_label, 'long', 1, 1)
        save_temp_fig(fig, report)
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        if seperate == 1:
            gs = GridSpec(6, 4, figure=fig)
            block_behavior_opto_session.run_all_blocks([plt.subplot(gs[0, i]) for i in range(3)],[plt.subplot(gs[0, i]) for i in range(1,3)],
                                                       [plt.subplot(gs[0, i]) for i in range(2,3)], [plt.subplot(gs[0, i]) for i in range(3, 4)],
                                                       session_data_1, block_data, st_seperate = seperate)
            block_behavior_opto_session.run_all_blocks([[plt.subplot(gs[j, i]) for i in range(3)] for j in range(1,6)],
                                                       [[plt.subplot(gs[j, i]) for i in range(1,3)] for j in range(1,6)],
                                                       [[plt.subplot(gs[j, i]) for i in range(2,3)] for j in range(1,6)],
                                                       [[plt.subplot(gs[j, i]) for i in range(3,4)] for j in range(1,6)],
                                                       session_data_1, block_data ,outcome_ref='outcome', st_seperate = seperate)
        else:
            gs = GridSpec(6, 4, figure=fig)
            block_behavior_opto_session.run_all_blocks([plt.subplot(gs[0, i]) for i in [0]],[plt.subplot(gs[0, i]) for i in [1]],
                                                       [plt.subplot(gs[0, i]) for i in [2]],[plt.subplot(gs[0, i]) for i in [3]],
                                                       session_data_1, block_data, st_seperate = seperate)
            block_behavior_opto_session.run_all_blocks([[plt.subplot(gs[j, i]) for i in [0]] for j in range(1,6)],
                                                       [[plt.subplot(gs[j, i]) for i in [1]] for j in range(1,6)],
                                                       [[plt.subplot(gs[j, i]) for i in [2]] for j in range(1,6)],
                                                       [[plt.subplot(gs[j, i]) for i in [3]] for j in range(1,6)],
                                                       session_data_1, block_data ,outcome_ref='outcome', st_seperate = seperate)
        save_temp_fig(fig, report)
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        if seperate == 1:
            gs = GridSpec(7, 4, figure=fig)
            interval_behavior_opto_session.run_all_blocks([[plt.subplot(gs[j, i]) for i in range(3)] for j in range(0,7)],
                                                       [[plt.subplot(gs[j, i]) for i in range(1,3)] for j in range(0,7)],
                                                       [[plt.subplot(gs[j, i]) for i in range(2,3)] for j in range(0,7)],
                                                       [[plt.subplot(gs[j, i]) for i in range(3,4)] for j in range(0,7)],
                                                       session_data_1, block_data ,interval_data, st_seperate = seperate)
        else:
            gs = GridSpec(7, 4, figure=fig)
            interval_behavior_opto_session.run_all_blocks([[plt.subplot(gs[j, i]) for i in [0]] for j in range(0,7)],
                                                       [[plt.subplot(gs[j, i]) for i in [1]] for j in range(0,7)],
                                                       [[plt.subplot(gs[j, i]) for i in [2]] for j in range(0,7)],
                                                       [[plt.subplot(gs[j, i]) for i in [3]] for j in range(0,7)],
                                                       session_data_1, block_data ,interval_data, st_seperate = seperate)
        save_temp_fig(fig, report)
        
        if individual_trajectory:
            fig = plt.figure(layout='constrained', figsize=(30, 15))
            gs = GridSpec(4, 11, figure=fig)
            aligned_trajectories.aligned_data([plt.subplot(gs[0, i])  for i in range(0,11)],
                                              [plt.subplot(gs[1, i])  for i in range(0,11)],
                                              [plt.subplot(gs[2, i])  for i in range(0,11)],
                                              [plt.subplot(gs[3, i])  for i in range(0,11)],
                                              session_data_1, 1,1,0)
            save_temp_fig(fig, report)
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 11, figure=fig)
        aligned_trajectories.aligned_data_mega_session([plt.subplot(gs[1, i])  for i in range(0,11)],
                                          [plt.subplot(gs[2, i])  for i in range(0,11)],
                                          session_data_1, 1,1,0)
        save_temp_fig(fig, report)
        
        if seperate:
            fig = plt.figure(layout='constrained', figsize=(30, 15))
            gs = GridSpec(9, 4, figure=fig)
            plot_distribution.run_dist([plt.subplot(gs[i, 0])  for i in range(0,9)],
                                       [plt.subplot(gs[i, 1])  for i in range(0,9)], 
                                       [plt.subplot(gs[i, 2])  for i in range(0,9)], 
                                       [plt.subplot(gs[i, 3])  for i in range(0,9)],
                                       session_data_1, block_data, interval_data, plot_type = 'double_opto_block', st = 1)
            plt.suptitle('self timed')
            save_temp_fig(fig, report)
            
            fig = plt.figure(layout='constrained', figsize=(30, 15))
            gs = GridSpec(9, 4, figure=fig)
            plot_distribution.run_dist([plt.subplot(gs[i, 0])  for i in range(0,9)],
                                       [plt.subplot(gs[i, 1])  for i in range(0,9)], 
                                       [plt.subplot(gs[i, 2])  for i in range(0,9)], 
                                       [plt.subplot(gs[i, 3])  for i in range(0,9)],
                                       session_data_1, block_data, interval_data, plot_type = 'double_opto_block', st = 2)
            plt.suptitle('visually guided')
            save_temp_fig(fig, report)
        else:
            fig = plt.figure(layout='constrained', figsize=(30, 15))
            gs = GridSpec(9, 4, figure=fig)
            plot_distribution.run_dist([plt.subplot(gs[i, 0])  for i in range(0,9)],
                                       [plt.subplot(gs[i, 1])  for i in range(0,9)], 
                                       [plt.subplot(gs[i, 2])  for i in range(0,9)], 
                                       [plt.subplot(gs[i, 3])  for i in range(0,9)],
                                       session_data_1, block_data, interval_data, plot_type = 'double_opto_block', st = 0)
            plt.suptitle('All sessions distributions')
            save_temp_fig(fig, report)
        
        report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_double_block_opto_analysis.pdf')
        report.close()
        
        
def run1(session_data_1, block_data, interval_data, subject, output_dir_onedrive, last_day, seperate = 0, individual_trajectory = 0):
    report = fitz.open()
    
    # performance
    print('Plotting General Performance')
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(6, 7, figure=fig)
    double_block_analysis.run_outcome(plt.subplot(gs[0:2, 0:6]) , session_data_1, 0)
    Number_of_trials.run_trial_num(plt.subplot(gs[4:6, 0:2]) , session_data_1  , 0)
    Number_of_trials.run_reward_num(plt.subplot(gs[2:4, 0:2]) , session_data_1 , 0)
    Number_of_trials.run_delay(plt.subplot(gs[0, 6]) , session_data_1 , 0)
    
    if seperate == 0:
        #double_block_analysis.run_trajectory([plt.subplot(gs[j, 2:4]) for j in range(2 , 4)]  , session_data_1,block_data, 0)
        #trajectories.run_trajectory(plt.subplot(gs[4, 2:4]) , session_data_1, 0 , ' Rewarded trials (grand average of all sessions)' )
        trajectories_first_push.run_trajectory(plt.subplot(gs[5, 2:4]), session_data_1, 0 , ' Rewarded trials (all pooled average)')
        Number_of_trials.run_dist([plt.subplot(gs[j, 4:6]) for j in range(2 , 4)] , session_data_1 , block_data , 0)
        Number_of_trials.run_dist_rewarded([plt.subplot(gs[j, 4:6]) for j in range(4 , 6)] , session_data_1 , block_data , 0)
    else:
        #double_block_analysis.run_trajectory([plt.subplot(gs[j, 2]) for j in range(2 , 4)]  , session_data_1,block_data, 1)
        trajectories.run_trajectory(plt.subplot(gs[4, 2]) , session_data_1, 1 , ' Rewarded trials (grand average of ST sessions)' )
        trajectories_first_push.run_trajectory(plt.subplot(gs[5, 2]), session_data_1, 1 , ' Rewarded trials (all ST pooled average)')
        Number_of_trials.run_dist([plt.subplot(gs[j, 3]) for j in range(2 , 4)] , session_data_1 , block_data , 1)
        Number_of_trials.run_dist_rewarded([plt.subplot(gs[j, 3]) for j in range(4 , 6)] , session_data_1 , block_data , 1)
        
        double_block_analysis.run_trajectory([plt.subplot(gs[j, 4]) for j in range(2 , 4)]  , session_data_1,block_data, 2)
        trajectories.run_trajectory(plt.subplot(gs[4, 4]) , session_data_1, 2 , ' Rewarded trials (grand average of VG)' )
        trajectories_first_push.run_trajectory(plt.subplot(gs[5, 4]), session_data_1, 2 , ' Rewarded trials (all VG pooled average)')
        Number_of_trials.run_dist([plt.subplot(gs[j, 5]) for j in range(2 , 4)] , session_data_1 , block_data , 2)
        Number_of_trials.run_dist_rewarded([plt.subplot(gs[j, 5]) for j in range(4 , 6)] , session_data_1 , block_data , 2)
    plt.suptitle(subject)
    save_temp_fig(fig, report)
    report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_double_block_opto_analysis.pdf')
    report.close()