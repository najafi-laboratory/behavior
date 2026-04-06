# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 09:12:43 2025

@author: saminnaji3
"""
import tkinter as tk
from tkinter import filedialog
import numpy as np

from ReadData import DataIO
from ReadData import trial_delay
from ReadData import interval_calculater
from modeling_input import behavior_reader
from model_poly import run_many_sessions
from model_poly import ModelConfig
import ploting_result 
import summary_plot
import model_single_percep
from model_single_percep import OnlineQuadPerceptron
from model_single_percep import run_model
from model_single_percep import find_best_lr
from model_single_percep import find_best_lambda_noise
# %% setting input and output path
session_data_all = []
'''
    folder_options = ['All Data' , 'Single Block' , 'Double Block' , 'Opto Sessions' , 'Imaging Sessions' , 'Chemo Sessions', 'Opto Random Reward', 
                      'Opto Random EarlyPress', 'Opto All Press']
    SUBJECTS = ['LChR_SA02' , 'LChR_SA04' , 'LChR_SA05' , 'LG_SA01' , 'LG_SA03', 'SM01_SChR' , 'SM02_SChR' , 'SM03_SChR' , 'SM04_SChR', 'SA_LG06', 'SA_LG07',
                'SA_LG08', 'SM05_Vgat', 'SM06_Vgat', 'SA_LG09', 'SA_LG11']
'''
# fill this part out
last_day = '20251105'
out_name = 'modeling'
data_name = 'Double Block'
subject = 'SA_LG11'


session_data_path = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Data joystick/'+ data_name 
output_dir_onedrive = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Figure Joystick/' + out_name + '/'
output_dir_local = 'C:/Users/saminnaji3/OneDrive - Georgia Institute of Technology/Figure Joystick/' + out_name + '/'
st_vg = ['_all' , '_ST' , '_VG']
seperate = 0
st_vg_id = [[0] , [0, 1, 2]]
# %% reading_data

window = tk.Tk()
window.wm_attributes('-topmost', 1)
window.withdraw()

file_paths_1 = list(filedialog.askopenfilenames(parent=window, initialdir=session_data_path+'/'+ subject, title='Select '+ subject + ' Sessions'))
session_data_1 = DataIO.read_trials(session_data_path, subject, file_paths_1)
#session_data_all.append(session_data_1) 

block_data = trial_delay.read_block(session_data_1)
interval_data = interval_calculater.calculate_interal(block_data, session_data_1)
windows = {
     0: (0.40, 0.93),  # short-valid
     1: (0.93, 3.13),  # long-valid
 }
behavioral_data , sessions_input = behavior_reader(session_data_1, st = 0, exlude_nan = 0)
# %% single perceptron learning noise
[delay_short, delay_long, label_short, label_long] = model_single_percep.define_input(behavioral_data)
# short_model = OnlineQuadPerceptron(lambda_noise=0.10, lr0=2e-2, lr_decay=5e-4, seed=0)
# long_model  = OnlineQuadPerceptron(lambda_noise=0.10, lr0=2e-2, lr_decay=5e-4, seed=1)


short_model = OnlineQuadPerceptron(
    lambda_noise=0.10,
    lr0=0.05,
    lr_decay=0,
    lr_lambda=1e-4,
    seed=0
)
long_model  = OnlineQuadPerceptron(
    lambda_noise=0.10,
    lr0=0.05,
    lr_decay=0,
    lr_lambda=1e-4,
    seed=1
)

short_info = model_single_percep.online_train_stream_with_omissions(short_model,
                                                np.concatenate(delay_short),
                                                np.concatenate(label_short))

long_info = model_single_percep.online_train_stream_with_omissions(long_model,
                                               np.concatenate(delay_long),
                                               np.concatenate(label_long))

samples = np.linspace(0, 5, num=2500)
p_short = short_model.predict(samples)
p_long = long_model.predict(samples)
trial_nums_short = [len(s) for s in delay_short]
trial_nums_long = [len(s) for s in delay_long]
short_info['session_len'] = trial_nums_short
long_info['session_len'] = trial_nums_long
ploting_result.plot_single_percep(short_info, long_info, p_short, p_long,samples, output_dir_onedrive, subject)

# %% only best lr
[delay_short, delay_long, label_short, label_long] = model_single_percep.define_input(behavioral_data)

find_best_lr(delay_short, label_short, seed_indx = 0)
find_best_lr(delay_long, label_long, seed_indx = 1)

# %% best lr and best noise

[delay_short, delay_long, label_short, label_long] = model_single_percep.define_input(behavioral_data)

find_best_lambda_noise(delay_short, label_short, seed_indx = 0)
find_best_lambda_noise(delay_long, label_long, seed_indx = 1)

# %%

import model_single_percep_2026
from model_single_percep_2026 import OnlineQuadPerceptron
from model_single_percep_2026 import run_model_fixed_lambda
from model_single_percep_2026 import find_best_lambda_noise_mc


[delay_short, delay_long, label_short, label_long] = model_single_percep_2026.define_input(behavioral_data)

(best_lam_sh, best_lr0_sh, best_decay_sh), best_loss_sh = find_best_lambda_noise_mc(delay_short, label_short, seed_indx=0, n_mc_train=100)
(best_lam_lo, best_lr0_lo, best_decay_lo), best_loss_sh =find_best_lambda_noise_mc(delay_long, label_long, seed_indx=1, n_mc_train=100)

# %%
[delay_short, delay_long, label_short, label_long] = model_single_percep_2026.define_input(behavioral_data)


short_model = OnlineQuadPerceptron(
    lambda_noise=best_lam_sh,
    lr0=best_lr0_sh,
    lr_decay=best_decay_sh,
    seed=0,
    n_mc_train=100
)
long_model  = OnlineQuadPerceptron(
    lambda_noise=best_lam_lo,
    lr0=best_lr0_lo,
    lr_decay=best_decay_lo,
    seed=1,
    n_mc_train=100
)

short_info = model_single_percep_2026.online_train_stream_with_omissions(short_model,
                                                np.concatenate(delay_short),
                                                np.concatenate(label_short))

long_info = model_single_percep_2026.online_train_stream_with_omissions(long_model,
                                               np.concatenate(delay_long),
                                               np.concatenate(label_long))

samples = np.linspace(0, 5, num=2500)
p_short = short_model.predict(samples)
p_long = long_model.predict(samples)
trial_nums_short = [len(s) for s in delay_short]
trial_nums_long = [len(s) for s in delay_long]
short_info['session_len'] = trial_nums_short
long_info['session_len'] = trial_nums_long
ploting_result.plot_single_percep(short_info, long_info, p_short, p_long,samples, output_dir_onedrive, subject)
# %% single perceptron learning noise
[delay_short, delay_long, label_short, label_long] = model_single_percep.define_input(behavioral_data)
short_model = OnlineQuadPerceptron(lambda_noise=0.02, lr0=0.05, lr_decay=0.0, seed=0)
long_model  = OnlineQuadPerceptron(lambda_noise=0.02, lr0=0.05, lr_decay=0.0, seed=1)

short_info = model_single_percep.online_train_stream_with_omissions(short_model,
                                                np.concatenate(delay_short),
                                                np.concatenate(label_short))

long_info = model_single_percep.online_train_stream_with_omissions(long_model,
                                               np.concatenate(delay_long),
                                               np.concatenate(label_long))

samples = np.linspace(0, 5, num=2500)
p_short = short_model.predict(samples)
p_long = long_model.predict(samples)
trial_nums_short = [len(s) for s in delay_short]
trial_nums_long = [len(s) for s in delay_long]
short_info['session_len'] = trial_nums_short
long_info['session_len'] = trial_nums_long
ploting_result.plot_single_percep(short_info, long_info, p_short, p_long,samples, output_dir_onedrive, subject)
# %%
cfg = ModelConfig(action_L=0.4, action_U=2.2, graded_reward=False)
results = run_many_sessions(sessions_input, windows, cfg, mode="off_policy")

ploting_result.run_all(sessions_input, windows, results, output_dir_onedrive)
ploting_result.run(sessions_input, windows, results, output_dir_onedrive)
# %%
ploting_result.run_summary(sessions_input, windows, results, output_dir_onedrive)
# %%
import plot_summary
plot_summary.run_summary(sessions_input, windows, results, output_dir_onedrive)

