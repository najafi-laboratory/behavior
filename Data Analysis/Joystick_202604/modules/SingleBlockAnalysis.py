# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 14:45:35 2025

@author: saminnaji3
"""

import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import fitz
import warnings
import numpy as np

from plot import single_block_analysis
from plot import single_trial_adaptation
from plot import opto_single_trial_adaptation
from plot import single_trial_adaptation_hist
from plot import aligned_trajectories 
from plot import single_trial_adaptation_hist_new
from plot import plot_distribution
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
    for st in st_vg_id[seperate]:
        report = fitz.open()
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 3, figure=fig)
        single_block_analysis.run_indvi(plt.subplot(gs[1, 2]) , session_data_1,block_data, st)
        single_block_analysis.run_trajectory([plt.subplot(gs[j, 1]) for j in range(1 , 4)]  , session_data_1,block_data, st)
        single_block_analysis.run_outcome(plt.subplot(gs[0, 0:2]) , session_data_1, st)
        single_block_analysis.run_trial_num(plt.subplot(gs[1, 0]) , session_data_1, st)
        single_block_analysis.run_delay(plt.subplot(gs[2:4, 0]) , session_data_1, st)
        single_block_analysis.run_probe(plt.subplot(gs[0, 2]) , session_data_1, st)
        plt.suptitle(subject)
        save_temp_fig(fig, report)
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 5, figure=fig)
        press_type = ['Short' , 'Long' , 'Reward' , 'EarlyPress2' , 'LatePress2']
        for st in st_vg_id[seperate]:
            single_trial_adaptation.run_dist(plt.subplot(gs[0, st]) ,session_data_1, block_data, st)
        single_trial_adaptation.run_cum_dist(plt.subplot(gs[0, 4]) ,session_data_1, block_data, st)
        for i in range(len(press_type)):
            single_trial_adaptation.run_count([plt.subplot(gs[1, i]), plt.subplot(gs[2, i])],plt.subplot(gs[0, 3]), session_data_1, block_data, press_type[i])
            single_trial_adaptation.run_delay(plt.subplot(gs[3, i]), session_data_1, block_data, press_type[i])
            
        save_temp_fig(fig, report)
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 5, figure=fig)
        single_trial_adaptation_hist.run([plt.subplot(gs[0,0]),plt.subplot(gs[1,0])], session_data_1, block_data, [0.4, 0.1], 2, 'short', 0)
        single_trial_adaptation_hist.run_filtered_outcome([plt.subplot(gs[0,2]),plt.subplot(gs[1,2])], session_data_1, block_data, st, 'short', label = 0)
        single_trial_adaptation_hist.run([plt.subplot(gs[0,1]),plt.subplot(gs[1,1]), plt.subplot(gs[2,1]),plt.subplot(gs[3,1])], session_data_1, block_data, [3.5, 3.5], 2, 'short', 1, 1)
        save_temp_fig(fig, report)
        
        if individual_trajectory:
            fig = plt.figure(layout='constrained', figsize=(30, 15))
            gs = GridSpec(4, 11, figure=fig)
            aligned_trajectories.aligned_data([plt.subplot(gs[0, i])  for i in range(0,11)],
                                              np.nan, np.nan, np.nan,
                                              session_data_1, np.nan,np.nan, st = 0)
            save_temp_fig(fig, report)
        
        
        # fig = plt.figure(layout='constrained', figsize=(30, 15))
        # gs = GridSpec(4, 11, figure=fig)
        # aligned_trajectories.aligned_data_mega_session([plt.subplot(gs[1, i])  for i in range(0,11)],
        #                                   np.nan,
        #                                   session_data_1, np.nan,np.nan, st = 0)
        # save_temp_fig(fig, report)
        
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(9, 4, figure=fig)
        plot_distribution.run_dist([plt.subplot(gs[i, 0])  for i in range(0,9)],
                                   np.nan,  
                                   np.nan, 
                                   np.nan,
                                   session_data_1, block_data, interval_data, plot_type = 'single_opto', st = 0)
        plt.suptitle('All sessions')
        save_temp_fig(fig, report)
        
        
        report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_single_block_analysis.pdf')
        report.close()
        
def run_opto(session_data_1, block_data, interval_data, subject, output_dir_onedrive, last_day, seperate = 0, individual_trajectory = 0):
    for st in st_vg_id[seperate]:
        report = fitz.open()
        
        st_label = 0
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 4, figure=fig)
        single_trial_adaptation_hist_new.run([plt.subplot(gs[0,1]),plt.subplot(gs[1,1]), plt.subplot(gs[2,1]),plt.subplot(gs[3,1])], session_data_1, block_data, [3.5, 3.5], st_label, 'short', 1, 1)
        save_temp_fig(fig, report)
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 5, figure=fig)
        single_trial_adaptation_hist.run([plt.subplot(gs[0,0]),plt.subplot(gs[1,0])], session_data_1, block_data, [0.4, 0.1], st_label, 'short', 0)
        single_trial_adaptation_hist.run([plt.subplot(gs[0,1]),plt.subplot(gs[1,1]), plt.subplot(gs[2,1]),plt.subplot(gs[3,1])], session_data_1, block_data, [3.5, 3.5], st_label, 'short', 1, 1)
        save_temp_fig(fig, report)
        
        if individual_trajectory:
            fig = plt.figure(layout='constrained', figsize=(30, 15))
            gs = GridSpec(4, 11, figure=fig)
            aligned_trajectories.aligned_data([plt.subplot(gs[0, i])  for i in range(0,11)],
                                              [plt.subplot(gs[1, i])  for i in range(0,11)], np.nan, np.nan,
                                              session_data_1, np.nan,1, st = 0)
            save_temp_fig(fig, report)
        
        fig = plt.figure(layout='constrained', figsize=(30, 15))
        gs = GridSpec(4, 11, figure=fig)
        aligned_trajectories.aligned_data_mega_session([plt.subplot(gs[1, i])  for i in range(0,11)],
                                          np.nan,
                                          session_data_1, np.nan,1, st = 0)
        save_temp_fig(fig, report)
        
        if seperate:
            fig = plt.figure(layout='constrained', figsize=(30, 15))
            gs = GridSpec(9, 4, figure=fig)
            plot_distribution.run_dist([plt.subplot(gs[i, 0])  for i in range(0,9)],
                                       np.nan, 
                                       np.nan, 
                                       np.nan,
                                       session_data_1, block_data, interval_data, plot_type = 'single_opto', st = 1)
            plt.suptitle('self timed')
            save_temp_fig(fig, report)
            
            fig = plt.figure(layout='constrained', figsize=(30, 15))
            gs = GridSpec(9, 4, figure=fig)
            plot_distribution.run_dist([plt.subplot(gs[i, 0])  for i in range(0,9)],
                                       np.nan, 
                                       np.nan, 
                                       np.nan,
                                       session_data_1, block_data, interval_data, plot_type = 'single_opto', st = 2)
            plt.suptitle('visually guided')
            save_temp_fig(fig, report)
        else:
            fig = plt.figure(layout='constrained', figsize=(30, 15))
            gs = GridSpec(9, 4, figure=fig)
            plot_distribution.run_dist([plt.subplot(gs[i, 0])  for i in range(0,9)],
                                       np.nan,  
                                       np.nan, 
                                       np.nan,
                                       session_data_1, block_data, interval_data, plot_type = 'single_opto', st = 0)
            plt.suptitle('All sessions')
            save_temp_fig(fig, report)
        
        report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_single_block_opto_analysis.pdf')
        report.close()