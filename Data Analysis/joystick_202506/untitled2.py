# -*- coding: utf-8 -*-
"""
Created on Tue May  6 08:06:18 2025

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

import DataIO
from plot import trial_delay
from plot import push_kinematic
from plot import interval_calculater
from plot import block_realize
from plot import double_block_analysis
from plot import trajectories
from plot import trajectories_first_push
from plot import Number_of_trials
from plot import single_block_analysis
from plot import opto_summary
from plot import single_trial_adaptation
from plot import block_behavior
from plot import meta_learning
from plot import effective_probe_analysis
from plot import interval_behavior
from plot import opto_single_trial_adaptation
from plot import all_outcome_opto_single_trial_adaptation

warnings.filterwarnings('ignore')

# %% no_need_to_be_run
session_data_all = []
folder_options = ['All Data' , 'Single Block' , 'Double Block' , 'Opto Sessions' , 'Imaging Sessions' , 'Chemo Sessions']

# %% for_laptop
session_data_path = 'C:/Users/Sana/OneDrive - Georgia Institute of Technology/Data joystick/Double Block'
output_dir_onedrive = 'C:/Users/Sana/OneDrive - Georgia Institute of Technology/Figure Joystick/Weekly_report_20250324_20250406/'
output_dir_local = 'C:/Users/Sana/OneDrive - Georgia Institute of Technology/Figure Joystick/Weekly_report_20250324_20250406/'
last_day = '20250429'
st_vg = ['_all' , '_ST' , '_VG']
SUBJECTS = ['LChR_SA02' , 'LChR_SA04' , 'LChR_SA05' , 'LG_SA01' , 'LG_SA03', 'SM01_SChR' , 'SM02_SChR' , 'SM03_SChR' , 'SM04_SChR', 'SA_LG06', 'SA_LG07', 'SA_LG08']

# %% for_lab_pc
session_data_path = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Data joystick/Opto Sessions'

# %% temprary_figures_folder
output_dir_onedrive = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Figure Joystick/Temprary Figs/'
output_dir_local = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Figure Joystick/Temprary Figs/'
last_day = '20250526'
st_vg = ['_all' , '_ST' , '_VG']
seperate = 0
st_vg_id = [[0] , [0, 1, 2]]
SUBJECTS = ['LChR_SA02' , 'LChR_SA04' , 'LChR_SA05' , 'LG_SA01' , 'LG_SA03', 'SM01_SChR' , 'SM02_SChR' , 'SM03_SChR' , 'SM04_SChR', 'SA_LG06', 'SA_LG07', 'SA_LG08']

# %% weekly_report_folder
output_dir_onedrive = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Figure Joystick/Summary_20250502/'
output_dir_local = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Figure Joystick/Summary_20250502/'
last_day = '20250506'
st_vg = ['_all' , '_ST' , '_VG']
seperate = 0
st_vg_id = [[0] , [0, 1, 2]]
SUBJECTS = ['LChR_SA02' , 'LChR_SA04' , 'LChR_SA05' , 'LG_SA01' , 'LG_SA03', 'SM01_SChR' , 'SM02_SChR' , 'SM03_SChR' , 'SM04_SChR', 'SA_LG06', 'SA_LG07', 'SA_LG08']
# %% reading_data

window = tk.Tk()
window.wm_attributes('-topmost', 1)
window.withdraw()
subject = 'SM01_SChR'

file_paths_1 = list(filedialog.askopenfilenames(parent=window, initialdir=session_data_path+'/'+ subject, title='Select '+ subject + ' Sessions'))
session_data_1 = DataIO.read_trials(session_data_path, subject, file_paths_1)

#session_data_all.append(session_data_1)
push_data = push_kinematic.read_kinematics(session_data_1)
block_data = trial_delay.read_block(session_data_1, push_data)
interval_data = interval_calculater.calculate_interal(push_data, block_data, session_data_1)

# %% opto_summary_block_analysis
for st in st_vg_id[seperate]:
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(6, 4, figure=fig)
    #opto_blocks.run_epoch(plt.subplot(gs[0:2, 4]) , session_data_1,block_data, st)
    #opto_summary.outcome(plt.subplot(gs[2:4, 0:2])  , session_data_1,block_data, 0, st)
    #opto_summary.outcome_all(plt.subplot(gs[0:2,0])  , session_data_1,0,st)
    #opto_summary.outcome_all(plt.subplot(gs[0:2,1])  , session_data_1,1,st)
    #opto_summary.outcome(plt.subplot(gs[2:4, 2:4])  , session_data_1,block_data, 1, st)
    #opto_summary.run_first_epoch(plt.subplot(gs[0:3, 4]) , session_data_1,block_data, st)
    opto_summary.run_first_epoch(plt.subplot(gs[0:3, 0]) , session_data_1,block_data, st, shaded = 0, sep_short = 1, sep_long = 1)
    opto_summary.run_first_epoch(plt.subplot(gs[0:3, 1]) , session_data_1,block_data, st, shaded = 0, sep_short = 2, sep_long = 2)
    opto_summary.run_first_epoch(plt.subplot(gs[0:3, 2]) , session_data_1,block_data, st, shaded = 0, sep_short = 3, sep_long = 3)
    opto_summary.run_first_epoch(plt.subplot(gs[0:3, 3]) , session_data_1,block_data, st, shaded = 0, sep_short = 4, sep_long = 6)
    #opto_summary.run_initial_adaptation(plt.subplot(gs[4:6, 0:2])   , session_data_1,block_data, st )
    #opto_summary.run_final_adaptation(plt.subplot(gs[4:6, 2:4])   , session_data_1,block_data, st )
    
    plt.suptitle(subject)
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_opto_test11.pdf')
    subject_report.close()
    
    
# %% opto_single_trial_adaptation
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(4, 7, figure=fig)
press_type = ['Opto_Reward' , 'Control_Reward' , 'Control']
axs_id = [0 , 0 , 2]
color = ['cyan' , 'k' , 'k']
# for st in st_vg_id[seperate]:
#     single_trial_adaptation.run_dist(plt.subplot(gs[0, st]) ,session_data_1, block_data, st)
# for i in range(len(press_type)):
#     opto_single_trial_adaptation.run_opto_count([plt.subplot(gs[1, i]), plt.subplot(gs[1, i])],plt.subplot(gs[0, 3]), session_data_1, block_data, press_type[i])
#     opto_single_trial_adaptation.run_opto_num([plt.subplot(gs[3, i]), plt.subplot(gs[3, i])], session_data_1, block_data, press_type[i])
#     opto_single_trial_adaptation.run_opto_delay(plt.subplot(gs[2, axs_id[i]]), session_data_1, block_data, press_type[i], color[i])
opto_single_trial_adaptation.run_opto_outcome([plt.subplot(gs[1, 0])], session_data_1, block_data)
plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_opto_single_trial_adaptation_test11.pdf')
subject_report.close()


# %% single_block
for st in st_vg_id[seperate]:
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(4, 3, figure=fig)
    single_block_analysis.run_indvi(plt.subplot(gs[2, 2]) , session_data_1,block_data, st)
    single_block_analysis.run_dist(plt.subplot(gs[1, 2]) , session_data_1, block_data, st)
    single_block_analysis.run_trajectory([plt.subplot(gs[j, 1]) for j in range(1 , 4)]  , session_data_1,block_data, st)
    single_block_analysis.run_outcome(plt.subplot(gs[0, 0:2]) , session_data_1, st, label = 1)
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
    subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[st] + '_single_block_test18.pdf')
    subject_report.close()
    
    
# %% single_trial adaptation
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(4, 5, figure=fig)
press_type = ['Short' , 'Long' , 'Reward' , 'EarlyPress2' , 'LatePress2']
for st in st_vg_id[seperate]:
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

# %% single_trial adaptation_cancat
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(4, 5, figure=fig)
press_type = ['Short' , 'Long' , 'Reward' , 'EarlyPress2' , 'LatePress2']
# for st in st_vg_id[seperate]:
#     single_trial_adaptation.run_dist(plt.subplot(gs[0, st]) ,session_data_1, block_data, st)
single_trial_adaptation.run_delay_concat([plt.subplot(gs[3, i]) for i in range(3)], session_data_1, block_data)
single_trial_adaptation.run_delay_concat_s_l([plt.subplot(gs[2, i]) for i in range(3)], session_data_1, block_data)
single_trial_adaptation.run_delay_concat_init_end([plt.subplot(gs[1, i]) for i in range(5)], session_data_1, block_data)
#for i in range(len(press_type)):
    #single_trial_adaptation.run_count([plt.subplot(gs[1, i]), plt.subplot(gs[2, i])],plt.subplot(gs[0, 3]), session_data_1, block_data, press_type[i])
    #single_trial_adaptation.run_delay(plt.subplot(gs[3, i]), session_data_1, block_data, press_type[i])
single_trial_adaptation.run_delay_concat_meta_learning([plt.subplot(gs[0, i]) for i in range(5)], session_data_1, block_data)    
    
plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_single_trial_adaptation_test22.pdf')
subject_report.close()
# %% block_behavior
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))

if seperate == 1:
    gs = GridSpec(6, 6, figure=fig)
    block_behavior.run([plt.subplot(gs[0, i]) for i in [0 , 2 , 4]], [plt.subplot(gs[0, i]) for i in [1 , 3 , 5]], session_data_1, block_data, st_seperate = seperate)
    block_behavior.run([[plt.subplot(gs[j, i]) for i in [0 , 2 , 4]] for j in range(1,6)], [[plt.subplot(gs[j, i]) for i in [1 , 3 , 5]] for j in range(1,6)], session_data_1, block_data ,outcome_ref='outcome', st_seperate = seperate)
else:
    gs = GridSpec(6, 4, figure=fig)
    block_behavior.run([plt.subplot(gs[0, i]) for i in [1]], [plt.subplot(gs[0, i]) for i in [2]], session_data_1, block_data, st_seperate = seperate)
    block_behavior.run([[plt.subplot(gs[j, i]) for i in [1]] for j in range(1,6)], [[plt.subplot(gs[j, i]) for i in [2]] for j in range(1,6)], session_data_1, block_data ,outcome_ref='outcome', st_seperate = seperate)

plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
#fig.set_size_inches(30, 15)
fig.savefig(fname)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_block_behavior_3.pdf')
subject_report.close()
##################################
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))

if seperate == 1:
    gs = GridSpec(6, 5, figure=fig)
    block_behavior.run_all_blocks([plt.subplot(gs[0, i]) for i in range(3)], session_data_1, block_data, st_seperate = seperate)
    block_behavior.run_all_blocks([[plt.subplot(gs[j, i]) for i in range(3)] for j in range(1,6)], session_data_1, block_data ,outcome_ref='outcome', st_seperate = seperate)
else:
    gs = GridSpec(6, 5, figure=fig)
    block_behavior.run_all_blocks([plt.subplot(gs[0, i]) for i in [1]], session_data_1, block_data, st_seperate = seperate)
    block_behavior.run_all_blocks([[plt.subplot(gs[j, i]) for i in [1]] for j in range(1,6)], session_data_1, block_data ,outcome_ref='outcome', st_seperate = seperate)

plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
#fig.set_size_inches(30, 15)
fig.savefig(fname)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_all_blocks_behavior_3.pdf')
subject_report.close()

# %% interval_behavior
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
if seperate == 1:
    gs = GridSpec(7, 6, figure=fig)
    interval_behavior.run([[plt.subplot(gs[j, i]) for i in [0 , 2 , 4]] for j in range(7)], [[plt.subplot(gs[j, i]) for i in [1 , 3 , 5]] for j in range(7)], session_data_1, block_data, interval_data, st_seperate = seperate)
else:
    gs = GridSpec(7, 4, figure=fig)
    interval_behavior.run([[plt.subplot(gs[j, i]) for i in [1]] for j in range(7)], [[plt.subplot(gs[j, i]) for i in [2]] for j in range(7)], session_data_1, block_data, interval_data, st_seperate = seperate)
    
plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
#fig.set_size_inches(30, 15)
fig.savefig(fname)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_block_interval1.pdf')
subject_report.close()


subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
if seperate == 1:
    gs = GridSpec(7, 3, figure=fig)
    interval_behavior.run_all_blocks([[plt.subplot(gs[j, i]) for i in range(3)] for j in range(7)], session_data_1, block_data, interval_data, st_seperate = seperate)
else:
    gs = GridSpec(7, 3, figure=fig)
    interval_behavior.run_all_blocks([[plt.subplot(gs[j, i]) for i in [1]] for j in range(7)], session_data_1, block_data, interval_data, st_seperate = seperate)
    
plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
#fig.set_size_inches(30, 15)
fig.savefig(fname)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_all_blocks_interval1.pdf')
subject_report.close()