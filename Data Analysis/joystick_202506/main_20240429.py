# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 14:20:35 2025

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
from plot import opto_trajectory
from plot import block_behavior_opto_session

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
SUBJECTS = ['LChR_SA02' , 'LChR_SA04' , 'LChR_SA05' , 'LG_SA01' , 'LG_SA03', 'SM01_SChR' , 'SM02_SChR' , 'SM03_SChR' , 'SM04_SChR', 'SA_LG06', 'SA_LG07', 'SA_LG08', 'SM05_Vgat', 'SM06_Vgat']

# %% for_lab_pc
session_data_path = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Data joystick/Single Block'

# %% temprary_figures_folder
output_dir_onedrive = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Figure Joystick/Temprary Figs/'
output_dir_local = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Figure Joystick/Temprary Figs/'
last_day = '20250602'
st_vg = ['_all' , '_ST' , '_VG']
seperate = 0
st_vg_id = [[0] , [0, 1, 2]]
SUBJECTS = ['LChR_SA02' , 'LChR_SA04' , 'LChR_SA05' , 'LG_SA01' , 'LG_SA03', 'SM01_SChR' , 'SM02_SChR' , 'SM03_SChR' , 'SM04_SChR', 'SA_LG06', 'SA_LG07', 'SA_LG08']

# %% weekly_report_folder
output_dir_onedrive = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Figure Joystick/Summary_20250610/'
output_dir_local = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Figure Joystick/Summary_20250610/'
last_day = '20250610'
st_vg = ['_all' , '_ST' , '_VG']
seperate = 0
st_vg_id = [[0] , [0, 1, 2]]
SUBJECTS = ['LChR_SA02' , 'LChR_SA04' , 'LChR_SA05' , 'LG_SA01' , 'LG_SA03', 'SM01_SChR' , 'SM02_SChR' , 'SM03_SChR' , 'SM04_SChR', 'SA_LG06', 'SA_LG07', 'SA_LG08']
# %% reading_data

window = tk.Tk()
window.wm_attributes('-topmost', 1)
window.withdraw()
subject = 'SM06_Vgat'

file_paths_1 = list(filedialog.askopenfilenames(parent=window, initialdir=session_data_path+'/'+ subject, title='Select '+ subject + ' Sessions'))
session_data_1 = DataIO.read_trials(session_data_path, subject, file_paths_1)

#session_data_all.append(session_data_1)
push_data = push_kinematic.read_kinematics(session_data_1)
block_data = trial_delay.read_block(session_data_1, push_data)
interval_data = interval_calculater.calculate_interal(push_data, block_data, session_data_1)
# %% block_behavior_opto
subject_report = fitz.open()
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

plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
#fig.set_size_inches(30, 15)
fig.savefig(fname)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_all_blocks_behavior_opto_not_seperateed.pdf')
subject_report.close()
# %% testtttt
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(2,2, figure=fig)
opto_trajectory.run_trajectory([plt.subplot(gs[0,0]), plt.subplot(gs[1,0]), plt.subplot(gs[0,1]), plt.subplot(gs[1,1])], session_data_1, block_data, 0, 'test')
plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_test_please2.pdf')
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

# %% opto_single_trial_adaptation
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(4, 5, figure=fig)
press_type = ['Opto_Reward' , 'Control_Reward' , 'Control']
axs_id = [0 , 0 , 2]
color = ['cyan' , 'k' , 'k']
for st in st_vg_id[seperate]:
    single_trial_adaptation.run_dist(plt.subplot(gs[0, st]) ,session_data_1, block_data, st)
for i in range(len(press_type)):
    opto_single_trial_adaptation.run_opto_count([plt.subplot(gs[1, i]), plt.subplot(gs[1, i])],plt.subplot(gs[0, 3]), session_data_1, block_data, press_type[i])
    opto_single_trial_adaptation.run_opto_num([plt.subplot(gs[3, i]), plt.subplot(gs[3, i])], session_data_1, block_data, press_type[i])
    opto_single_trial_adaptation.run_opto_delay(plt.subplot(gs[2, axs_id[i]]), session_data_1, block_data, press_type[i], color[i])
    
plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_opto_single_trial_adaptation_reward.pdf')
subject_report.close()
# %% all_outcome_opto_single_trial_adaptation
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(4, 6, figure=fig)
press_type = ['Opto_Reward' , 'Control_Reward' ,'Opto_EarlyPress2' , 'Control_EarlyPress2','Opto_LatePress2' , 'Control_LatePress2']
axs_id = [0 , 0 , 2,2, 4, 4]
color = ['cyan' , 'k','cyan' , 'k','cyan' , 'k']
for st in st_vg_id[seperate]:
    single_trial_adaptation.run_dist(plt.subplot(gs[0, st]) ,session_data_1, block_data, st)
for i in range(len(press_type)):
    all_outcome_opto_single_trial_adaptation.run_opto_count([plt.subplot(gs[1, i]), plt.subplot(gs[1, i])],plt.subplot(gs[0, 3]), session_data_1, block_data, press_type[i])
    all_outcome_opto_single_trial_adaptation.run_opto_num([plt.subplot(gs[3, i]), plt.subplot(gs[3, i])], session_data_1, block_data, press_type[i])
    all_outcome_opto_single_trial_adaptation.run_opto_delay(plt.subplot(gs[2, axs_id[i]]), session_data_1, block_data, press_type[i], color[i])
    
plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_all_outcome_opto_single_trial_adaptation.pdf')
subject_report.close()
# %% meta_learning
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(3, 3, figure=fig)
if seperate == 1:
    meta_learning.run([plt.subplot(gs[0, i]) for i in range(3)], [plt.subplot(gs[1, i]) for i in range(3)],[plt.subplot(gs[2, i]) for i in range(3)], block_data, session_data_1, st_seperate = seperate)
else:
    meta_learning.run([plt.subplot(gs[0, i]) for i in [1]], [plt.subplot(gs[1, i]) for i in [1]],[plt.subplot(gs[2, i]) for i in [1]], block_data, session_data_1, st_seperate = seperate)

plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_meta_learning_final.pdf')
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
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_block_behavior_not_Seperated.pdf')
subject_report.close()
##################################
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))

if seperate == 1:
    gs = GridSpec(6, 3, figure=fig)
    block_behavior.run_all_blocks([plt.subplot(gs[0, i]) for i in range(3)], session_data_1, block_data, st_seperate = seperate)
    block_behavior.run_all_blocks([[plt.subplot(gs[j, i]) for i in range(3)] for j in range(1,6)], session_data_1, block_data ,outcome_ref='outcome', st_seperate = seperate)
else:
    gs = GridSpec(6, 3, figure=fig)
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
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_all_blocks_behavior_not_seperateed.pdf')
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
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_block_interval_final.pdf')
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
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day  + '_all_blocks_interval_final.pdf')
subject_report.close()
# %% grand_probe_analysis
for st in st_vg_id[seperate]:
    subject_report = fitz.open()
    
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(4, 4, figure=fig)
    

    effective_probe_analysis.run_grand_dist([plt.subplot(gs[0:2, 1]) ,plt.subplot(gs[0:2, 2])] , session_data_1,block_data, st)
    effective_probe_analysis.run_grand_trajectory([plt.subplot(gs[j, 0]) for j in range(2 , 4)]  , session_data_1,block_data , 0 , st)
    effective_probe_analysis.run_grand_trajectory([plt.subplot(gs[j, 0]) for j in range(2 , 4)]  , session_data_1,block_data , 1 , st)
    effective_probe_analysis.run_grand_trajectory([plt.subplot(gs[j, 0]) for j in range(2 , 4)]  , session_data_1,block_data , 2 , st)
    
    effective_probe_analysis.run_grand_trajectory_first_press_rewarded([plt.subplot(gs[j, 1]) for j in range(2 , 4)]  , session_data_1 , 0 , st)
    effective_probe_analysis.run_grand_trajectory_first_press_rewarded([plt.subplot(gs[j, 1]) for j in range(2 , 4)]  , session_data_1 , 1 , st)
    effective_probe_analysis.run_grand_trajectory_first_press_rewarded([plt.subplot(gs[j, 1]) for j in range(2 , 4)]  , session_data_1, 2 , st)
    
    effective_probe_analysis.run_grand_trajectory_first_press([plt.subplot(gs[j, 2]) for j in range(2 , 4)]  , session_data_1 , 0 , st)
    effective_probe_analysis.run_grand_trajectory_first_press([plt.subplot(gs[j, 2]) for j in range(2 , 4)]  , session_data_1 , 1 , st)
    effective_probe_analysis.run_grand_trajectory_first_press([plt.subplot(gs[j, 2]) for j in range(2 , 4)]  , session_data_1, 2 , st)
    
    effective_probe_analysis.run_grand_outcome(plt.subplot(gs[0, 0]) , session_data_1 , 'short', st)
    effective_probe_analysis.run_grand_outcome(plt.subplot(gs[1, 0]) , session_data_1 , 'long', st)
    
    
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

# %% performance
for st in st_vg_id[seperate]:
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(6, 4, figure=fig)
    Number_of_trials.run_trial_num(plt.subplot(gs[4:6, 1]) , session_data_1  , st)
    Number_of_trials.run_reward_num(plt.subplot(gs[2:4, 1]) , session_data_1 , st)
    double_block_analysis.run_trajectory([plt.subplot(gs[j, 0]) for j in range(2 , 4)]  , session_data_1,block_data, st)
    double_block_analysis.run_outcome(plt.subplot(gs[0:2, 0:3]) , session_data_1, st)
    trajectories.run_trajectory(plt.subplot(gs[4, 0]) , session_data_1, st , ' Rewarded trials (grand average of all sessions)' )
    trajectories_first_push.run_trajectory(plt.subplot(gs[5, 0]), session_data_1, st , ' Rewarded trials (all pooled average)')
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
    
# %% within_session_adaptation
for st in st_vg_id[seperate]:
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
    
# %% within_block_adapaion
for st in st_vg_id[seperate]:
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

# %% opto_summary_block_analysis
for st in st_vg_id[seperate]:
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

# %% single_block
for st in st_vg_id[seperate]:
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


















