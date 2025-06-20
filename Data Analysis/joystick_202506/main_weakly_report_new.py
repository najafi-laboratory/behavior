# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:41:24 2025

@author: saminnaji3
"""

import tkinter as tk
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import fitz
import warnings
import pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')

# %%
from plot import opto_session
from plot import trial_delay
from plot import push_kinematic
from plot import probe_analysis
from plot import block_realize
from plot import outcome_analysis
from plot import double_block_analysis
from plot import trajectories
from plot import trajectories_first_push
from plot import trajectories_pooled
from plot import trajectories_1th_press_pooled
from plot import fig_superimposed_average_velocity_onset
from plot import plot_push_kinematic

from plot import opto_double_block
from plot import Number_of_trials
from plot import opto_blocks
from plot import single_block_analysis
from plot import opto_summary
from plot import adaptation_block_summary
from plot import single_trial_adaptation
from plot import block_behavior
from plot import meta_learning
# %%
session_data_all = []

session_data_path = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Data joystick/Opto Sessions'

# %% for my own computer
session_data_all = []

session_data_path = 'C:/Users/Sana/OneDrive - Georgia Institute of Technology/Data joystick/Double Block'
output_dir_onedrive = 'C:/Users/Sana/OneDrive - Georgia Institute of Technology/Figure Joystick/Weekly_report_20250324_20250406/'
output_dir_local = 'C:/Users/Sana/OneDrive - Georgia Institute of Technology/Figure Joystick/Weekly_report_20250324_20250406/'
last_day = '20250429'
st_vg = ['_all' , '_ST' , '_VG']
SUBJECTS = ['LChR_SA02' , 'LChR_SA04' , 'LChR_SA05' , 'LG_SA01' , 'LG_SA03', 'SM01_SChR' , 'SM02_SChR' , 'SM03_SChR' , 'SM04_SChR', 'SA_LG06', 'SA_LG07', 'SA_LG08']

# %%
import DataIO

window = tk.Tk()

window.wm_attributes('-topmost', 1)

window.withdraw()  
# SUBJECTS = ['LChR_SA02' , 'LChR_SA04' , 'LChR_SA05' , 'LG_SA01' , 'LG_SA03' , 'SM01_SChR' , 'SM02_SChR' , 'SM03_SChR' , 'SM04_SChR', 'SA_LG06', 'SA_LG07', 'SA_LG08']

subject = 'SM01_SChR'

file_paths_1 = list(filedialog.askopenfilenames(parent=window, initialdir=session_data_path+'/'+ subject, title='Select '+ subject + ' Sessions'))
session_data_1 = DataIO.read_trials(session_data_path, subject, file_paths_1)
#session_data_all.append(session_data_1)

# %%
output_dir_onedrive = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Figure Joystick/Temprary Figs/'
output_dir_local = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Figure Joystick/Temprary Figs/'
last_day = '20250429'
st_vg = ['_all' , '_ST' , '_VG']
SUBJECTS = ['LChR_SA02' , 'LChR_SA04' , 'LChR_SA05' , 'LG_SA01' , 'LG_SA03', 'SM01_SChR' , 'SM02_SChR' , 'SM03_SChR' , 'SM04_SChR', 'SA_LG06', 'SA_LG07', 'SA_LG08']

# %%
output_dir_onedrive = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Figure Joystick/Probe_Report_20241204_20250316/'
output_dir_local = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Figure Joystick/Probe_Report_20241204_20250316/'
last_day = '20250316'
st_vg = ['_all' , '_ST' , '_VG']
SUBJECTS = ['LChR_SA02' , 'LChR_SA04' , 'LChR_SA05' , 'LG_SA01' , 'LG_SA03']
# %%
#session_data = session_data_1.copy()
push_data = push_kinematic.read_kinematics(session_data_1)
# %%
block_data = trial_delay.read_block(session_data_1, push_data)

# %%
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(4,8, figure=fig)
plot_push_kinematic.run([plt.subplot(gs[0, j]) for j in range(8)],[plt.subplot(gs[1, j]) for j in range(8)],[plt.subplot(gs[2, j]) for j in range(8)], push_data)
plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_push_kinematic_opto_control.pdf')
subject_report.close()
# %%
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(4,8, figure=fig)
plot_push_kinematic.run_distict([plt.subplot(gs[0, j]) for j in range(8)],[plt.subplot(gs[1, j]) for j in range(8)],[plt.subplot(gs[2, j]) for j in range(8)], [plt.subplot(gs[3, j]) for j in range(8)],push_data)
plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_push_kinematic_CS_OS_CL_OL.pdf')
subject_report.close()

# %%
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(4,1, figure=fig)
plot_push_kinematic.run_distict_time(plt.subplot(gs[0, 0]),plt.subplot(gs[1 , 0]),plt.subplot(gs[2 , 0]), plt.subplot(gs[3 , 0]),push_data)
plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_push_kinematic_TIME.pdf')
subject_report.close()
# %% kinematic summary
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(2,3, figure=fig)
plot_push_kinematic.run_distict_time_all([plt.subplot(gs[0, 0]),plt.subplot(gs[0, 1]),plt.subplot(gs[0, 2]), plt.subplot(gs[1 , 0]), plt.subplot(gs[1 , 1])],push_data)
plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_push_kinematic_summary1.pdf')
subject_report.close()
# %% single trial adaptation
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(4, 5, figure=fig)
press_type = ['Short' , 'Long' , 'Reward' , 'EarlyPress2' , 'LatePress2']
#session_data_1 = session_data.copy()
for st in [0, 1 ,2]:
    
    single_trial_adaptation.run_dist(plt.subplot(gs[0, st]) ,session_data_1, block_data, st)
for i in range(len(press_type)):
    single_trial_adaptation.run_count([plt.subplot(gs[1, i]), plt.subplot(gs[2, i])],plt.subplot(gs[0, 3]), session_data_1, block_data, press_type[i])
    single_trial_adaptation.run_delay(plt.subplot(gs[3, i]), session_data_1, block_data, press_type[i])
    
plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_single_trial_adaptation_final.pdf')
subject_report.close()

# %% meta learning
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(3, 3, figure=fig)
meta_learning.run([plt.subplot(gs[0, i]) for i in range(3)], [plt.subplot(gs[1, i]) for i in range(3)], block_data, session_data_1)

plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_meta_learning_final3.pdf')
subject_report.close()
# %% block behavior
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(6, 6, figure=fig)

block_behavior.run([plt.subplot(gs[0, i]) for i in [0 , 2 , 4]], [plt.subplot(gs[0, i]) for i in [1 , 3 , 5]], session_data_1, block_data)
block_behavior.run([[plt.subplot(gs[j, i]) for i in [0 , 2 , 4]] for j in range(1,6)], [[plt.subplot(gs[j, i]) for i in [1 , 3 , 5]] for j in range(1,6)], session_data_1, block_data ,outcome_ref='outcome')

plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
#fig.set_size_inches(30, 15)
fig.savefig(fname)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_block_behavior_final.pdf')
subject_report.close()
# %% single block 


for st in [0, 1 ,2]:
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(4, 3, figure=fig)
    single_block_analysis.run_indvi(plt.subplot(gs[1, 2]) , session_data_1,block_data, st)
    single_block_analysis.run_trajectory([plt.subplot(gs[j, 1]) for j in range(1 , 4)]  , session_data_1,block_data, st)
    single_block_analysis.run_outcome(plt.subplot(gs[0, 0:2]) , session_data_1, st)
    single_block_analysis.run_trial_num(plt.subplot(gs[1, 0]) , session_data_1, st)
    single_block_analysis.run_delay(plt.subplot(gs[2:4, 0]) , session_data_1, st)
    plt.suptitle(subject)
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_single_block.pdf')
    subject_report.close()
# %% 2block_session


for st in [0 , 1 , 2]:
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(6, 6, figure=fig)
    
    double_block_analysis.run_trajectory([plt.subplot(gs[j, 3]) for j in range(2 , 4)]  , session_data_1,block_data, st)
    double_block_analysis.run_block_len([plt.subplot(gs[j, 4:6]) for j in range(4 , 6)]  , session_data_1,block_data, st)
    double_block_analysis.run_delay([plt.subplot(gs[j, 4:6]) for j in range(0 , 2)]  ,plt.subplot(gs[2:4, 1]), session_data_1,block_data, st)
    double_block_analysis.run_outcome(plt.subplot(gs[0:2, 0:4]) , session_data_1, st)
    double_block_analysis.run_epoch(plt.subplot(gs[2:4, 0]) , session_data_1,block_data, st)
    double_block_analysis.run_delay_rewarded([plt.subplot(gs[j, 4:6]) for j in range(2, 4)]  ,plt.subplot(gs[4:6, 1]), session_data_1,block_data, st)
    trajectories.run_trajectory(plt.subplot(gs[4, 3]) , session_data_1, 0 , ' Rewarded trials (grand average of all sessions)' )
    trajectories_first_push.run_trajectory(plt.subplot(gs[5, 3]), session_data_1, 0 , ' Rewarded trials (all pooled average)')
    plt.suptitle(subject)
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_double_block_report.pdf')
    subject_report.close()

# %% probe_analysis

for st in [0 , 1 , 2]:
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(4, 3, figure=fig)
    

    probe_analysis.run_dist(plt.subplot(gs[0:2, 2:4])  , session_data_1,block_data, st)
    probe_analysis.run_trajectory([plt.subplot(gs[j, 0]) for j in range(2 , 4)]  , session_data_1,block_data , 0 , st)
    probe_analysis.run_trajectory([plt.subplot(gs[j, 1]) for j in range(2 , 4)]  , session_data_1,block_data , 1 , st)
    probe_analysis.run_trajectory([plt.subplot(gs[j, 2]) for j in range(2 , 4)]  , session_data_1,block_data , 2 , st)
    probe_analysis.run_outcome(plt.subplot(gs[0:1, 0:2]) , session_data_1 , 'short', st)
    probe_analysis.run_outcome(plt.subplot(gs[1:2, 0:2]) , session_data_1 , 'long', st)
    
    
    plt.suptitle(subject)
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_probe_analysis_report.pdf')
    subject_report.close()
# %% grand_probe_analysis

for st in [0 , 1 , 2]:
    subject_report = fitz.open()
    
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(4, 4, figure=fig)
    

    probe_analysis.run_grand_dist([plt.subplot(gs[0:2, 1]) ,plt.subplot(gs[0:2, 2])] , session_data_1,block_data, st)
    probe_analysis.run_grand_trajectory([plt.subplot(gs[j, 0]) for j in range(2 , 4)]  , session_data_1,block_data , 0 , st)
    probe_analysis.run_grand_trajectory([plt.subplot(gs[j, 0]) for j in range(2 , 4)]  , session_data_1,block_data , 1 , st)
    probe_analysis.run_grand_trajectory([plt.subplot(gs[j, 0]) for j in range(2 , 4)]  , session_data_1,block_data , 2 , st)
    
    probe_analysis.run_grand_trajectory_first_press_rewarded([plt.subplot(gs[j, 1]) for j in range(2 , 4)]  , session_data_1 , 0 , st)
    probe_analysis.run_grand_trajectory_first_press_rewarded([plt.subplot(gs[j, 1]) for j in range(2 , 4)]  , session_data_1 , 1 , st)
    probe_analysis.run_grand_trajectory_first_press_rewarded([plt.subplot(gs[j, 1]) for j in range(2 , 4)]  , session_data_1, 2 , st)
    
    probe_analysis.run_grand_trajectory_first_press([plt.subplot(gs[j, 2]) for j in range(2 , 4)]  , session_data_1 , 0 , st)
    probe_analysis.run_grand_trajectory_first_press([plt.subplot(gs[j, 2]) for j in range(2 , 4)]  , session_data_1 , 1 , st)
    probe_analysis.run_grand_trajectory_first_press([plt.subplot(gs[j, 2]) for j in range(2 , 4)]  , session_data_1, 2 , st)
    
    probe_analysis.run_grand_outcome(plt.subplot(gs[0:1, 0]) , session_data_1 , 'short', st)
    probe_analysis.run_grand_outcome(plt.subplot(gs[1:2, 0]) , session_data_1 , 'long', st)
    
    
    plt.suptitle(subject)
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_grand_probe_analysis_report.pdf')
    subject_report.close()

# %% block_realize

for st in [0]:
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(5, 4, figure=fig)
    
    block_realize.run_block_realize([plt.subplot(gs[4, j]) for j in range(0 , 2)]  , session_data_1,block_data, st)
    block_realize.run_initial_adaptation([plt.subplot(gs[j, 0]) for j in range(0 , 3)]   , session_data_1,block_data, st)
    block_realize.run_partial_adaptation([plt.subplot(gs[j, 1]) for j in range(0 , 3)]   , session_data_1,block_data, st)
    block_realize.run_partial_adaptation_new([plt.subplot(gs[j, 2]) for j in range(0 , 3)]   , session_data_1,block_data, st)
    block_realize.run_epoch(plt.subplot(gs[0:2, 3]) , session_data_1,block_data, st)
    
    plt.suptitle(subject)
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_block_realize_analysis_report.pdf')
    subject_report.close()
 
# %% performance 


for st in [0 , 1 , 2]:
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(6, 4, figure=fig)
    Number_of_trials.run_trial_num(plt.subplot(gs[4:6, 1]) , session_data_1  , st)
    Number_of_trials.run_reward_num(plt.subplot(gs[2:4, 1]) , session_data_1 , st)
    double_block_analysis.run_trajectory([plt.subplot(gs[j, 0]) for j in range(2 , 4)]  , session_data_1,block_data, st)
    double_block_analysis.run_outcome(plt.subplot(gs[0:2, 0:3]) , session_data_1, st)
    trajectories.run_trajectory(plt.subplot(gs[4, 0]) , session_data_1, 0 , ' Rewarded trials (grand average of all sessions)' )
    trajectories_first_push.run_trajectory(plt.subplot(gs[5, 0]), session_data_1, 0 , ' Rewarded trials (all pooled average)')
    Number_of_trials.run_delay(plt.subplot(gs[0, 3]) , session_data_1 , st)
    Number_of_trials.run_dist([plt.subplot(gs[j, 2]) for j in range(2 , 4)] , session_data_1 , block_data , st)
    Number_of_trials.run_dist_rewarded([plt.subplot(gs[j, 2]) for j in range(4 , 6)] , session_data_1 , block_data , st)
    plt.suptitle(subject)
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_performance.pdf')
    subject_report.close()

# %% within session adaptation


for st in [0 , 1 , 2]:
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(6, 5, figure=fig)
    
    double_block_analysis.run_block_len([plt.subplot(gs[j, 0:3]) for j in range(4 , 6)]  , session_data_1,block_data, st)
    double_block_analysis.run_delay([plt.subplot(gs[j, 0:3]) for j in range(0 , 2)]  ,plt.subplot(gs[0:2, 3]), session_data_1,block_data, st)
    double_block_analysis.run_delay_rewarded([plt.subplot(gs[j, 0:3]) for j in range(2, 4)]  ,plt.subplot(gs[2:4, 3]), session_data_1,block_data, st)
    plt.suptitle(subject)
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_within_session_adaptation.pdf')
    subject_report.close()

# %% within block adapaion


for st in [0 , 1,  2]:
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(6, 5, figure=fig)
    
    block_realize.run_block_realize([plt.subplot(gs[j, 0:3]) for j in range(0 , 2)]  , session_data_1,block_data, st)
    #block_realize.run_block_realize_new([plt.subplot(gs[j, 0:3]) for j in range(2 , 4)]  , session_data_1,block_data, st)
    block_realize.run_initial_adaptation(plt.subplot(gs[4:6, 3])   , session_data_1,block_data, st )
    block_realize.run_partial_adaptation(plt.subplot(gs[0:2, 3])   , session_data_1,block_data, st )
    #block_realize.run_partial_adaptation_new(plt.subplot(gs[2:4, 3])   , session_data_1,block_data, st )
    block_realize.run_epoch(plt.subplot(gs[0:2, 4]) , session_data_1,block_data, st)
    block_realize.run_epoch_new(plt.subplot(gs[2:4, 4]) , session_data_1,block_data, st)
    double_block_analysis.run_epoch(plt.subplot(gs[4:6, 4]) , session_data_1,block_data, st)
    
    plt.suptitle(subject)
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_within_block_adaptation.pdf')
    subject_report.close()
# %% opto block analysis

for st in [2]:
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(6, 5, figure=fig)
    #opto_blocks.run_epoch(plt.subplot(gs[0:2, 4]) , session_data_1,block_data, st)
    opto_double_block.outcome(plt.subplot(gs[0:2, 0:2])  , session_data_1,block_data, 0, st)
    opto_blocks.outcome_all(plt.subplot(gs[2:4,0:2])  , session_data_1,st)
    opto_double_block.outcome(plt.subplot(gs[2:4, 2:4])  , session_data_1,block_data, 1, st)
    #opto_blocks.run_epoch_fix(plt.subplot(gs[0:2, 2]) , session_data_1,block_data, st)
    opto_blocks.run_initial_adaptation(plt.subplot(gs[0:2, 2:4])   , session_data_1,block_data, st )
    #opto_blocks.run_aligned_adaptation(plt.subplot(gs[0:2, 2:4])   , session_data_1,block_data, st , 1)
    #opto_blocks.run_aligned_adaptation(plt.subplot(gs[2:4, 2:4])   , session_data_1,block_data, st , 0)
    plt.suptitle(subject)
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_opto.pdf')
    subject_report.close()
# %% opto summary block analysis

for st in [0 , 1 , 2]:
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(6, 5, figure=fig)
    #opto_blocks.run_epoch(plt.subplot(gs[0:2, 4]) , session_data_1,block_data, st)
    opto_summary.outcome(plt.subplot(gs[2:4, 0:2])  , session_data_1,block_data, 0, st)
    opto_summary.outcome_all(plt.subplot(gs[0:2,0])  , session_data_1,0,st)
    opto_summary.outcome_all(plt.subplot(gs[0:2,1])  , session_data_1,1,st)
    opto_summary.outcome(plt.subplot(gs[2:4, 2:4])  , session_data_1,block_data, 1, st)
    opto_summary.run_epoch_fix(plt.subplot(gs[0:3, 4]) , session_data_1,block_data, st)
    opto_summary.run_initial_adaptation(plt.subplot(gs[4:6, 0:2])   , session_data_1,block_data, st )
    opto_summary.run_final_adaptation(plt.subplot(gs[4:6, 2:4])   , session_data_1,block_data, st )
    
    plt.suptitle(subject)
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_opto_all.pdf')
    subject_report.close()
# %% addaptation summary test

for st in [0 , 1 , 2]:
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(6, 5, figure=fig)
    #opto_blocks.run_epoch(plt.subplot(gs[0:2, 4]) , session_data_1,block_data, st)
    adaptation_block_summary.run_initial_adaptation_added(plt.subplot(gs[4:6, 0:2]) , plt.subplot(gs[4:6, 2:4])  , session_data_1,block_data, st )
    adaptation_block_summary.run_final_adaptation_added(plt.subplot(gs[4:6, 2:4]) , plt.subplot(gs[4:6, 0:2])  , session_data_1,block_data, st )
    
    plt.suptitle(subject)
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_adaptation_test.pdf')
    subject_report.close()
# %% test 


for st in [0 , 1 , 2]:
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(6, 4, figure=fig)
    
    opto_blocks.run_delay(plt.subplot(gs[0, 3]) , session_data_1 , 'onset1', st , 'visdetect1-onset1', onset = 1)
    opto_blocks.run_delay(plt.subplot(gs[1, 3]) , session_data_1 , 'amp1', st , 'amp1', onset = 0)
    opto_blocks.run_delay(plt.subplot(gs[2, 3]) , session_data_1 , 'peak1', st , 'visdetect1-peak1', onset = 1)
    opto_blocks.run_delay(plt.subplot(gs[3, 3]) , session_data_1 , 'velocity1', st , 'velocity1', onset = 0)
    
    opto_blocks.run_delay_opto(plt.subplot(gs[0, 2]) , session_data_1 , 'onset1', st , 'visdetect1-onset1', onset = 1)
    opto_blocks.run_delay_opto(plt.subplot(gs[1, 2]) , session_data_1 , 'amp1', st , 'amp1', onset = 0)
    opto_blocks.run_delay_opto(plt.subplot(gs[2, 2]) , session_data_1 , 'peak1', st , 'visdetect1-peak1', onset = 1)
    opto_blocks.run_delay_opto(plt.subplot(gs[3, 2]) , session_data_1 , 'velocity1', st , 'velocity1', onset = 0)
    
    plt.suptitle(subject)
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_push.pdf')
    subject_report.close()  
# %% opto block performance

for st in [0 , 1 , 2]:
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(6, 5, figure=fig)
    opto_double_block.outcome(plt.subplot(gs[0:2, 0:2])  , session_data_1, st)
    opto_blocks.outcome_all(plt.subplot(gs[2:4,0])  , session_data_1, st)
    
    plt.suptitle(subject)
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_opto.pdf')
    subject_report.close()
# %% outcome_analysis

for st in [0 , 1 , 2]:
    
    
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(4, 3, figure=fig)
    
    outcome_analysis.run_dist(plt.subplot(gs[0:2, 0])  , session_data_1,block_data , 'Reward', st)
    outcome_analysis.run_dist(plt.subplot(gs[0:2, 1])  , session_data_1,block_data , 'LatePress2', st)
    outcome_analysis.run_dist(plt.subplot(gs[0:2, 2])  , session_data_1,block_data , 'EarlyPress2', st)
    outcome_analysis.run_max_min(plt.subplot(gs[2, 0])  , session_data_1,block_data , 'Reward', st)
    
    plt.suptitle(subject)
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_outcome_analysis_report.pdf')
    subject_report.close()

# %% trajectories
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(5, len(session_data_all), figure=fig)
num = 0


for subject in SUBJECTS:
    session_data_1 = session_data_all[num]
    trajectories.run_trajectory(plt.subplot(gs[0, num]) , session_data_1, 0 , subject+ ' Rewarded trials (all sessions averaged)' )
    trajectories.run_trajectory(plt.subplot(gs[1, num]) , session_data_1, 1 , 'Rewarded trials (Self-Timed sessions averaged)')
    trajectories.run_trajectory(plt.subplot(gs[2, num]) , session_data_1, 2 , 'Rewarded trials (Visually-Giuded sessions averaged)')
    trajectories.run_trajectory_probe(plt.subplot(gs[3, num])  , session_data_1, 0 , 1 , 'Rewarded probe trials (all sessions averaged)')
    trajectories.run_trajectory_probe(plt.subplot(gs[4, num])  , session_data_1, 1 , 2 , 'Rewarded probe+1 trials (all sessions averaged)')
    num = num + 1
    
plt.suptitle('All Trajectories')
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ last_day  + '_all_trajectories_grand_average.pdf')
subject_report.close()

# %% trajectories 1st press
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(5, len(session_data_all), figure=fig)
num = 0


for subject in SUBJECTS:
    
    session_data_1 = session_data_all[num]
    trajectories_first_push.run_trajectory(plt.subplot(gs[0, num]) ,  session_data_1, 0 ,  subject+ ' Rewarded trials (all sessions averaged)')
    trajectories_first_push.run_trajectory(plt.subplot(gs[1, num]) , session_data_1, 1 , 'Rewarded trials (Self-Timed sessions averaged)')
    trajectories_first_push.run_trajectory(plt.subplot(gs[2, num]) , session_data_1, 2 , 'Rewarded trials (Visually-Giuded sessions averaged)')
    trajectories_first_push.run_trajectory_probe(plt.subplot(gs[3, num])  , session_data_1, 0 , 1 , 'Rewarded probe trials (all sessions averaged)')
    trajectories_first_push.run_trajectory_probe(plt.subplot(gs[4, num])  , session_data_1, 1 , 2 , 'Rewarded probe+1 trials (all sessions averaged)')
    
    num = num + 1
    
plt.suptitle('All Trajectories')
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ last_day  + '_all_trajectories_first_press_grand_average.pdf')
subject_report.close()

# %% trajectories pooled
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(5, len(session_data_all), figure=fig)
num = 0


for subject in SUBJECTS:
    
    session_data_1 = session_data_all[num]
    trajectories_pooled.run_trajectory(plt.subplot(gs[0, num]) , session_data_1, 0 ,  subject+' Rewarded trials (all sessions averaged)')
    trajectories_pooled.run_trajectory(plt.subplot(gs[1, num]) , session_data_1, 1 , 'Rewarded trials (Self-Timed sessions averaged)')
    trajectories_pooled.run_trajectory(plt.subplot(gs[2, num]) , session_data_1, 2 , 'Rewarded trials (Visually-Giuded sessions averaged)')
    trajectories_pooled.run_trajectory_probe(plt.subplot(gs[3, num])  , session_data_1, 0 , 1 , 'Rewarded probe trials (all sessions averaged)')
    trajectories_pooled.run_trajectory_probe(plt.subplot(gs[4, num])  , session_data_1, 1 , 2 , 'Rewarded probe+1 trials (all sessions averaged)')
    
    num = num + 1
    
plt.suptitle('All Trajectories')
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ last_day  + '_all_trajectories_pooled.pdf')
subject_report.close()

# %% trajectories pooled 1st press
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(5, 5, figure=fig)
num = 0


for subject in SUBJECTS:
    
    session_data_1 = session_data_all[num]
    trajectories_1th_press_pooled.run_trajectory(plt.subplot(gs[0, num]) , session_data_1, 0 ,  subject+' Rewarded trials (all sessions averaged)')
    trajectories_1th_press_pooled.run_trajectory(plt.subplot(gs[1, num]) , session_data_1, 1 , 'Rewarded trials (Self-Timed sessions averaged)')
    trajectories_1th_press_pooled.run_trajectory(plt.subplot(gs[2, num]) , session_data_1, 2 , 'Rewarded trials (Visually-Giuded sessions averaged)')
    trajectories_1th_press_pooled.run_trajectory_probe(plt.subplot(gs[3, num])  , session_data_1, 0 , 1 , 'Rewarded probe trials (all sessions averaged)')
    trajectories_1th_press_pooled.run_trajectory_probe(plt.subplot(gs[4, num])  , session_data_1, 1 , 2 , 'Rewarded probe+1 trials (all sessions averaged)')
    
    num = num + 1
    
plt.suptitle('All Trajectories')
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ last_day  + '_all_trajectories_first_press_pooled.pdf')
subject_report.close()

# %% opto session

for st in [0]:
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(4,5, figure=fig)
    

    opto_session.outcome(plt.subplot(gs[0:2, 4:6])  , session_data_1, st)
    opto_session.outcome_mod(plt.subplot(gs[2:4, 4:6])  , session_data_1, st)
    opto_session.run_trajectory([plt.subplot(gs[j, 0]) for j in range(1 , 4)]  , session_data_1 , st)
    opto_session.run_trajectory_outcome([plt.subplot(gs[j, 1]) for j in range(1 , 4)]  , session_data_1 , 'Reward', st)
    plt.subplot(gs[1, 0]).set_ylim([0.05 , 1.2])
    plt.subplot(gs[2, 0]).set_ylim([0.05 , 1.2])
    plt.subplot(gs[3, 0]).set_ylim([0.05 , 1.2])
    plt.subplot(gs[1, 1]).set_ylim([0.05 , 3])
    plt.subplot(gs[2, 1]).set_ylim([0.05 , 3])
    plt.subplot(gs[3, 1]).set_ylim([0.05 , 3])
    plt.subplot(gs[1, 3]).set_ylim([0.05 , 3.5])
    plt.subplot(gs[2, 3]).set_ylim([0.05 , 3.5])
    plt.subplot(gs[3, 3]).set_ylim([0.05 , 3.5])
    plt.subplot(gs[1, 2]).set_ylim([0.05 , 2])
    plt.subplot(gs[2, 2]).set_ylim([0.05 , 2])
    plt.subplot(gs[3, 2]).set_ylim([0.05 , 2])
    opto_session.run_trajectory_outcome([plt.subplot(gs[j, 2]) for j in range(1 , 4)]  , session_data_1 , 'LatePress1', st)
    opto_session.run_trajectory_outcome([plt.subplot(gs[j, 3]) for j in range(1 , 4)]  , session_data_1 , 'EarlyPress2', st)
    
    
    plt.suptitle(subject)
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_opto_analysis_report2.pdf')
    subject_report.close()
    
# %% delay 
from plot.fig_delay_press_all import push_delay

 
push_delay(session_data_1, output_dir_onedrive, output_dir_local)

# %% opto 2block
for st in [0]:
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(4,5, figure=fig)
    

    #opto_session.outcome(plt.subplot(gs[0:2, 4:6])  , session_data_1, st)
    #opto_double_block.outcome(plt.subplot(gs[2:4, 4:6])  , session_data_1, st)
    #opto_double_block.run_epoch(plt.subplot(gs[0:2, 3]) , session_data_1 , block_data, st)
    #opto_double_block.run_opto_delay(plt.subplot(gs[2:4, 3]) , session_data_1 , block_data, st)
    opto_double_block.run_epoch_compare(plt.subplot(gs[0:2, 3]) , session_data_1 , block_data, st)
    
    
    plt.suptitle(subject)
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_opto_double_block_report_test1.pdf')
    subject_report.close()
    
# %%
from plot.fig_superimposed_average_velocity_onset import plot_aligend_onset
plot_aligend_onset (
        session_data_1,
        output_dir_onedrive, 
        output_dir_local
        )