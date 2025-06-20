# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 08:59:59 2024

@author: saminnaji3
"""

import DataIO1
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np


from plot.fig1_outcome import plot_fig1
from plot.fig1_outcome_short_long_1 import plot_fig1_2
from plot.fig1_outcome_control_opto import plot_fig1_3
from plot.fig1_outcome_block import plot_fig1_4
from plot.fig1_outcome_OvsC_SvsL import plot_fig1_5
from plot.fig3_trajectory import plot_fig3_colored
from plot.fig2_trajectory_avg_sess import plot_fig2
# from plot.fig3_trajectories import plot_fig3
from plot.fig4_trajectory_avg_sess_superimpose import plot_fig4
# from plot.fig5_trajectory_avg_sess_superimpose_short_long1 import plot_fig5
from plot.fig6_trajectories_opto import plot_fig_opto_trajectories
# from fig6_trajectories import plot_fig_outcome_trajectories
from plot.fig3_trajectories_all_alignments_copy import plot_fig__trajectories1_all_aligned
from plot.fig6_trajectories_superimposed import plot_fig_outcome_trajectories_sup
from plot.fig6_trajectories_superimposed_all_outcomes import plot_fig_outcome_trajectories_sup_all
from plot.fig6_average_superimosed_short_long_all import plot_fig_outcome_trajectories_sup_SL
from plot.fig_delay_distribution import plot_delay_distribution
from plot.fig_superimposed_average_velocity_onset import plot_aligend_onset
from plot.fig_bpod import plot_bpod

# %%
window = tk.Tk()
window.wm_attributes('-topmost', 1)
window.withdraw()  
session_data_path = 'C:/Users/saminnaji3/Downloads/joystick/data/'
subject = 'LG06'

file_paths_1 = list(filedialog.askopenfilenames(parent=window, initialdir=session_data_path+subject, title='Select '+ subject + ' Sessions'))
session_data_1 = DataIO1.read_trials(session_data_path, subject, file_paths_1)

output_dir_onedrive = 'C:/Users/saminnaji3/Downloads/joystick/figure1/'
output_dir_local = 'C:/Users/saminnaji3/Downloads/joystick/figure1/'


# %%

import importlib
import plot.fig1_outcome

importlib.reload(plot.fig1_outcome)
from plot.fig1_outcome import plot_fig1
plot_fig1(session_data_1, output_dir_onedrive, output_dir_local)

import importlib
import plot.fig1_outcome_short_long_1

importlib.reload(plot.fig1_outcome_short_long_1)
from plot.fig1_outcome_short_long_1 import plot_fig1_2
plot_fig1_2(session_data_1, output_dir_onedrive, output_dir_local)


# %%

import importlib
import plot.fig6_average_superimosed_short_long_all

importlib.reload(plot.fig6_average_superimosed_short_long_all)
from plot.fig6_average_superimosed_short_long_all import plot_fig_outcome_trajectories_sup_SL

plot_fig_outcome_trajectories_sup_SL(session_data_1, output_dir_onedrive, output_dir_local)


from plot.fig_time_analysis_copy import plot_time_analysis
import importlib
import plot.fig_time_analysis_copy

importlib.reload(plot.fig_time_analysis_copy)
from plot.fig_time_analysis_copy import plot_time_analysis
plot_time_analysis(session_data_1, output_dir_onedrive, output_dir_local)



import importlib
import plot.fig_delay

importlib.reload(plot.fig_delay)
from plot.fig_delay_press import push_delay
push_delay(session_data_1, output_dir_onedrive, output_dir_local)

# all data 
import importlib
import plot.fig_delay

importlib.reload(plot.fig_delay)
from plot.fig_delay import push_delay
push_delay(session_data_1, output_dir_onedrive, output_dir_local)



# # all data 
import importlib
import plot.fig_bpod

importlib.reload(plot.fig_bpod)
from plot.fig_bpod import plot_bpod
plot_bpod(session_data_1, output_dir_onedrive, output_dir_local)



# all data 
import importlib
import plot.fig_event_interval

importlib.reload(plot.fig_event_interval)
from plot.fig_event_interval import event_interval
event_interval(session_data_1, output_dir_onedrive, output_dir_local)





