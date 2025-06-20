# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:26:03 2024

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
import DataIO

window = tk.Tk()
window.wm_attributes('-topmost', 1)

window.withdraw()  
session_data_path = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Data joystick/Double Block'
# SUBJECTS = ['LChR_SA02' , 'LChR_SA04' , 'LChR_SA05' , 'LG_SA01' , 'LG_SA03']
# SUBJECTS = ['LG01' , 'LG10' , 'VG01']
subject = 'LG01'

file_paths_1 = list(filedialog.askopenfilenames(parent=window, initialdir=session_data_path+'/'+ subject, title='Select '+ subject + ' Sessions'))
session_data_1 = DataIO.read_trials(session_data_path, subject, file_paths_1)

# %%
output_dir_onedrive = 'C:/Users/saminnaji3/Downloads/joystick/20250113/'
output_dir_local = 'C:/Users/saminnaji3/Downloads/joystick/20250113/'
last_day = '20250113'

# %%
import DataIO

window = tk.Tk()
window.wm_attributes('-topmost', 1)

window.withdraw()  
session_data_path = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Data joystick/Single Block'
SUBJECTS = ['LChR_SA02' , 'LChR_SA04' , 'LChR_SA05' , 'LG_SA01' , 'LG_SA03']
count = 0
session_data_1 = defaultdict(list)
for subject in SUBJECTS:


    file_paths_1 = list(filedialog.askopenfilenames(parent=window, initialdir=session_data_path+'/'+ subject, title='Select '+ subject + ' Sessions'))
    session_data_2 = DataIO.read_trials(session_data_path, subject, file_paths_1)
    #session_data_2 = pd.DataFrame(session_data_2)
   
    for key, value in session_data_2.items():
        session_data_1[key].append(value)
   
# %%
session_data_3 = defaultdict(list)
for key, value in session_data_1.items():
    print(key)
    if isinstance(session_data_1[key][0], list):
        session_data_3[key] = sum(session_data_1[key], [])
    else:
        session_data_3[key] = session_data_1[key]

# %% block outcome 
from plot.fig_outcome_block_new import plot_fig1_3


plot_fig1_3(session_data_1, output_dir_onedrive, output_dir_local)

# %% short/long outcome
from plot.fig1_outcome_short_long_1 import plot_fig1_2

plot_fig1_2(session_data_1, output_dir_onedrive, output_dir_local)
# %% trial delay
from plot.trial_delay import plot_fig7

plot_fig7(session_data_1, output_dir_onedrive, output_dir_local)
# %% all trial outcome
from plot.fig1_outcome import plot_fig1
plot_fig1(session_data_1, output_dir_onedrive, output_dir_local)

# %% delay 
from plot.fig_delay_press_all import push_delay
push_delay(session_data_1, output_dir_onedrive, output_dir_local)
# %% probe short long outcome
import importlib
import plot.plot_short_long_outcome


from plot.plot_short_long_outcome import plot_fig1_probe
plot_fig1_probe(session_data_1, output_dir_onedrive, output_dir_local)

# %% trjectories
import importlib
import plot.fig6_average_superimosed_short_long_all

importlib.reload(plot.fig6_average_superimosed_short_long_all)
from plot.fig6_average_superimosed_short_long_all import plot_fig_outcome_trajectories_sup_SL

plot_fig_outcome_trajectories_sup_SL(session_data_1, output_dir_onedrive, output_dir_local)



# %%
from plot import trial_delay

block_data = trial_delay.read_block(session_data_1)


# %% delay 
from plot.fig_delay_press_all import push_delay

 
push_delay(session_data_1, output_dir_onedrive, output_dir_local)


# %% statistical delay analysis
from plot import statistical_analysis_delay



subject_report = fitz.open()
#last_day = '20241208'


fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(4, 7, figure=fig)

statistical_analysis_delay.run(plt.subplot(gs[0, 0:3]),plt.subplot(gs[0, 3:5]),session_data_1,block_data)
statistical_analysis_delay.run_indvi([plt.subplot(gs[j, 0:3]) for j in range(1 , 4)],[plt.subplot(gs[k, 3:5]) for k in range(1 , 4)],session_data_1,block_data)
statistical_analysis_delay.run_QC([plt.subplot(gs[j, 5:7]) for j in range(1 , 4)],session_data_1,block_data)



plt.suptitle(subject)

fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + '_delay_stat.pdf')
subject_report.close()


# %% adaptation delay analysis
from plot import block_adaptation_delay


subject_report = fitz.open()
#last_day = '20241208'


fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(4, 6, figure=fig)


block_adaptation_delay.run(plt.subplot(gs[0, 0:3]),session_data_1,block_data)
block_adaptation_delay.run_exclude_first_block(plt.subplot(gs[0, 3:6]),session_data_1,block_data)
block_adaptation_delay.run_indivi([[plt.subplot(gs[1, j]) for j in range(0 , 6)] , [plt.subplot(gs[2, j]) for j in range(0 , 6)] , [plt.subplot(gs[3, j]) for j in range(0 , 6)]],session_data_1,block_data)

plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + '_delay_adaptation.pdf')
subject_report.close()

# %% normalized block analysis
from plot import normalized_block


subject_report = fitz.open()
#last_day = '20241212'


fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(4, 6, figure=fig)



normalized_block.run_indvi([[plt.subplot(gs[0, j]) for j in range(0 , 6)] , [plt.subplot(gs[1, j]) for j in range(0 , 6)]] ,[plt.subplot(gs[2, j]) for j in range(0 , 6)],[plt.subplot(gs[3, j]) for j in range(0 , 6)],session_data_1,block_data)

plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + '_normalized_block.pdf')
subject_report.close()

# %% non normalized block analysis
from plot import non_normalized_block


subject_report = fitz.open()
#last_day = '20241212'


fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(4, 6, figure=fig)



non_normalized_block.run_indvi([[plt.subplot(gs[0, j]) for j in range(0 , 6)] , [plt.subplot(gs[1, j]) for j in range(0 , 6)]] ,[plt.subplot(gs[2, j]) for j in range(0 , 6)],[plt.subplot(gs[3, j]) for j in range(0 , 6)],session_data_1,block_data)

plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + '_non_normalized_block.pdf')
subject_report.close()

# %% single block_session
from plot import single_block_analysis


subject_report = fitz.open()
#last_day = '20241204'


fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(4, 3, figure=fig)

single_block_analysis.run_indvi(plt.subplot(gs[1, 2]) , session_data_1,block_data)
single_block_analysis.run_trajectory([plt.subplot(gs[j, 1]) for j in range(1 , 4)]  , session_data_1,block_data)
single_block_analysis.run_outcome(plt.subplot(gs[0, 0:2]) , session_data_1)
single_block_analysis.run_trial_num(plt.subplot(gs[1, 0]) , session_data_1)
single_block_analysis.run_delay(plt.subplot(gs[2:4, 0]) , session_data_1)

plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + '_single_block_ST_report.pdf')
subject_report.close()
# %% mega single block_session
from plot import mega_single_session


subject_report = fitz.open()
#last_day = '20241204'



fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(4, 3, figure=fig)

# single_block_analysis.run_indvi(plt.subplot(gs[1, 2]) , session_data_1,block_data)
# single_block_analysis.run_trajectory([plt.subplot(gs[j, 1]) for j in range(1 , 4)]  , session_data_1,block_data)
mega_single_session.run_outcome(plt.subplot(gs[0, 0:2]) , session_data_1)
# single_block_analysis.run_trial_num(plt.subplot(gs[1, 0]) , session_data_1)
# single_block_analysis.run_delay(plt.subplot(gs[2:4, 0]) , session_data_1)

plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ last_day + '_mega_single_block_all_report.pdf')
subject_report.close()

# %% 2block_session
from plot import double_block_analysis

st_vg = ['_all' , '_ST' , '_VG']
for st in [0 , 1 , 2]:
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(6, 3, figure=fig)
    
    double_block_analysis.run_trajectory([plt.subplot(gs[j, 1]) for j in range(2 , 4)]  , session_data_1,block_data, st)
    double_block_analysis.run_block_len([plt.subplot(gs[j, 1]) for j in range(4 , 6)]  , session_data_1,block_data, st)
    double_block_analysis.run_delay([plt.subplot(gs[j, 2]) for j in range(2 , 4)]  ,plt.subplot(gs[2:4, 0]), session_data_1,block_data, st)
    double_block_analysis.run_outcome(plt.subplot(gs[0:2, 0:2]) , session_data_1, st)
    #double_block_analysis.run_delay_adapt( [plt.subplot(gs[j, 0]) for j in range(2 , 4)] , session_data_1,block_data, st)
    double_block_analysis.run_epoch(plt.subplot(gs[0:2, 2]) , session_data_1,block_data, st)
    double_block_analysis.run_delay_rewarded([plt.subplot(gs[j, 2]) for j in range(4, 6)]  ,plt.subplot(gs[4:6, 0]), session_data_1,block_data, st)
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
# %% moving_epoch
from plot import moving_epoch


subject_report = fitz.open()
#last_day = '20241204'


fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(5, 3, figure=fig)
st = 0


moving_epoch.run_delay_adapt( [plt.subplot(gs[j, 0]) for j in range(0 , 4)] , session_data_1,block_data, st)

st = 1


moving_epoch.run_delay_adapt( [plt.subplot(gs[j, 1]) for j in range(0 , 4)] , session_data_1,block_data, st)
st = 2


moving_epoch.run_delay_adapt( [plt.subplot(gs[j, 2]) for j in range(0 , 4)] , session_data_1,block_data, st)
plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + '_adaptation_all_report.pdf')
subject_report.close()
# %% probe_analysis
from plot import probe_analysis


st_vg = ['_all' , '_ST' , '_VG']
for st in [0 , 1 , 2]:
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(4, 3, figure=fig)
    

    probe_analysis.run_dist(plt.subplot(gs[0:2, 2:4])  , session_data_1,block_data, st)
    probe_analysis.run_trajectory([plt.subplot(gs[j, 0]) for j in range(2 , 4)]  , session_data_1,block_data , 0 , st)
    probe_analysis.run_trajectory([plt.subplot(gs[j, 1]) for j in range(2 , 4)]  , session_data_1,block_data , 1 , st)
    probe_analysis.run_trajectory([plt.subplot(gs[j, 2]) for j in range(2 , 4)]  , session_data_1,block_data , 2 , st)
    probe_analysis.run_outcome(plt.subplot(gs[0:2, 0:2]) , session_data_1 , st)
    
    
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
# %% outcome_analysis
from plot import outcome_analysis


st_vg = ['_all' , '_ST' , '_VG']
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
from plot import trajectories
import DataIO
from plot import trial_delay


session_data_path = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Data joystick/Double Block'
SUBJECTS = ['LChR_SA02' , 'LChR_SA04' , 'LChR_SA05' , 'LG_SA01' , 'LG_SA03']
#SUBJECTS = ['LChR_SA02']
output_dir_onedrive = 'C:/Users/saminnaji3/Downloads/joystick/'
output_dir_local = 'C:/Users/saminnaji3/Downloads/joystick/'
last_day = '20250109'
#sessio_data_all = []
subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(5, 5, figure=fig)
num = 0


for subject in SUBJECTS:
    if num >0 :
        window = tk.Tk()
        window.wm_attributes('-topmost', 1)
        window.withdraw()  
        file_paths_1 = list(filedialog.askopenfilenames(parent=window, initialdir=session_data_path+'/'+ subject, title='Select '+ subject + ' Sessions'))
        session_data_1 = DataIO.read_trials(session_data_path, subject, file_paths_1)
        sessio_data_all.append(session_data_1)
    else:
        session_data_1 = sessio_data_all[0]
    trajectories.run_trajectory(plt.subplot(gs[0, num]) , session_data_1, 0 , 'Rewarded trials (all sessions averaged)')
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
subject_report.save(output_dir_onedrive+ last_day  + '_all_trajectories.pdf')
subject_report.close()

# %% trajectories 1st press
from plot import trajectories_first_push
import DataIO
from plot import trial_delay


session_data_path = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Data joystick/Double Block'
SUBJECTS = ['LChR_SA02' , 'LChR_SA04' , 'LChR_SA05' , 'LG_SA01' , 'LG_SA03']
#SUBJECTS = ['LChR_SA02']
output_dir_onedrive = 'C:/Users/saminnaji3/Downloads/joystick/'
output_dir_local = 'C:/Users/saminnaji3/Downloads/joystick/'
last_day = '20250109'

subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(5, 5, figure=fig)
num = 0


for subject in SUBJECTS:
    
    session_data_1 = sessio_data_all[num]
    trajectories_first_push.run_trajectory(plt.subplot(gs[0, num]) , session_data_1, 0 , 'Rewarded trials (all sessions averaged)')
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
subject_report.save(output_dir_onedrive+ last_day  + '_all_trajectories_first_press.pdf')
subject_report.close()

# %% trajectories pooled
from plot import trajectories_pooled
import DataIO
from plot import trial_delay


session_data_path = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Data joystick/Double Block'
SUBJECTS = ['LChR_SA02' , 'LChR_SA04' , 'LChR_SA05' , 'LG_SA01' , 'LG_SA03']
#SUBJECTS = ['LChR_SA02']
output_dir_onedrive = 'C:/Users/saminnaji3/Downloads/joystick/'
output_dir_local = 'C:/Users/saminnaji3/Downloads/joystick/'
last_day = '20250109'

subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(5, 5, figure=fig)
num = 0


for subject in SUBJECTS:
    
    session_data_1 = sessio_data_all[num]
    trajectories_pooled.run_trajectory(plt.subplot(gs[0, num]) , session_data_1, 0 , 'Rewarded trials (all sessions averaged)')
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
from plot import trajectories_1th_press_pooled
import DataIO
from plot import trial_delay


session_data_path = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Data joystick/Double Block'
SUBJECTS = ['LChR_SA02' , 'LChR_SA04' , 'LChR_SA05' , 'LG_SA01' , 'LG_SA03']
#SUBJECTS = ['LChR_SA02']
output_dir_onedrive = 'C:/Users/saminnaji3/Downloads/joystick/'
output_dir_local = 'C:/Users/saminnaji3/Downloads/joystick/'
last_day = '20250109'

subject_report = fitz.open()
fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(5, 5, figure=fig)
num = 0


for subject in SUBJECTS:
    
    session_data_1 = sessio_data_all[num]
    trajectories_1th_press_pooled.run_trajectory(plt.subplot(gs[0, num]) , session_data_1, 0 , 'Rewarded trials (all sessions averaged)')
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
# %% block_realize
from plot import block_realize


st_vg = ['_all' , '_ST' , '_VG']
num = 0
for st in [0 , 1 , 2]:
    subject_report = fitz.open()
    fig = plt.figure(layout='constrained', figsize=(30, 15))
    gs = GridSpec(5, 3, figure=fig)
    
    block_realize.run_block_realize([plt.subplot(gs[4, j]) for j in range(0 , 2)]  , session_data_1,block_data, st)
    block_realize.run_initial_adaptation([plt.subplot(gs[j, 0]) for j in range(0 , 3)]   , session_data_1,block_data, st)
    block_realize.run_partial_adaptation([plt.subplot(gs[j, 1]) for j in range(0 , 3)]   , session_data_1,block_data, st)
    
    plt.suptitle(subject)
    fname = os.path.join(str(0).zfill(4)+'.pdf')
    fig.set_size_inches(30, 15)
    fig.savefig(fname, dpi=300)
    plt.close()
    roi_fig = fitz.open(fname)
    subject_report.insert_pdf(roi_fig)
    roi_fig.close()
    os.remove(fname)
    subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + st_vg[num] + '_block_realize_analysis_report.pdf')
    subject_report.close()
    num = num + 1
# %% adaptation epoch analysis
from plot import epoch_adaptation


subject_report = fitz.open()
last_day = '20241208'


fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(4, 6, figure=fig)


epoch_adaptation.run(plt.subplot(gs[0, 0:3]),session_data_1,block_data)

epoch_adaptation.run_indivi([[plt.subplot(gs[1, j]) for j in range(0 , 6)] , [plt.subplot(gs[2, j]) for j in range(0 , 6)] , [plt.subplot(gs[3, j]) for j in range(0 , 6)]],session_data_1,block_data)

plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + '_epoch_adaptation.pdf')
subject_report.close()

# %% all delay analysis
from plot import statistical_analysis_delay
from plot import block_adaptation_delay


subject_report = fitz.open()
last_day = '20241126'


fig = plt.figure(layout='constrained', figsize=(30, 15))
gs = GridSpec(4, 6, figure=fig)

statistical_analysis_delay.run(plt.subplot(gs[0, 0:3]),plt.subplot(gs[0, 3:5]),session_data_1,block_data)
statistical_analysis_delay.run_indvi([plt.subplot(gs[j, 0:3]) for j in range(1 , 2)],[plt.subplot(gs[j, 3:5]) for j in range(1 , 2)],session_data_1,block_data)
block_adaptation_delay.run(plt.subplot(gs[2, 0:3]),session_data_1,block_data)



plt.suptitle(subject)
fname = os.path.join(str(0).zfill(4)+'.pdf')
fig.set_size_inches(30, 15)
fig.savefig(fname, dpi=300)
plt.close()
roi_fig = fitz.open(fname)
subject_report.insert_pdf(roi_fig)
roi_fig.close()
os.remove(fname)
subject_report.save(output_dir_onedrive+ subject +'/' + subject +'_' + last_day + '_outcome_stat.pdf')
subject_report.close()