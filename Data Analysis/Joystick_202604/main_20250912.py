# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 10:06:27 2025

@author: saminnaji3
"""

import tkinter as tk
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import fitz
import warnings

from ReadData import DataIO
from ReadData import trial_delay
from ReadData import push_kinematic
from ReadData import interval_calculater
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
from plot import modeling_input
from plot.modeling import cost_function
from plot.modeling import simulate_model
from plot import modeling
from plot import opto_single_trial_adaptation_earlypress
from scipy.optimize import minimize
from plot import single_trial_adaptation_hist
from plot import fig_delay_press_all
warnings.filterwarnings('ignore')
# %% setting input and output path
session_data_all = []
'''
    folder_options = ['All Data' , 'Single Block' , 'Double Block' , 'Opto Sessions' , 'Imaging Sessions' , 'Chemo Sessions', 'Opto Random Reward', 
                      'Opto Random EarlyPress', 'Opto All Press']
    SUBJECTS = ['LChR_SA02' , 'LChR_SA04' , 'LChR_SA05' , 'LG_SA01' , 'LG_SA03', 'SM01_SChR' , 'SM02_SChR' , 'SM03_SChR' , 'SM04_SChR', 'SA_LG06', 'SA_LG07',
                'SA_LG08', 'SM05_Vgat', 'SM06_Vgat', 'SA_LG09', 'SA_LG11']
'''
# fill this part out
last_day = '20260320'
out_name = 'Weekly_report_20260301'
data_name = 'Double Block'
subject = 'SA17_LG'


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
# %% single block sessions analysis if there is no opto sessions you can comment third line
from modules import SingleBlockAnalysis
SingleBlockAnalysis.run(session_data_1, block_data, interval_data, subject, output_dir_onedrive, last_day, seperate = 0)

#SingleBlockAnalysis.run_opto(session_data_1, block_data, interval_data, subject, output_dir_onedrive, last_day, seperate = 0)

# %% double block sessions analysis if there is no opto sessions you can comment third line
from modules import DoubleBlockAnalysis
DoubleBlockAnalysis.run(session_data_1, block_data, interval_data, subject, output_dir_onedrive, last_day, seperate =1)

#DoubleBlockAnalysis.run_opto(session_data_1, block_data, interval_data, subject, output_dir_onedrive, last_day, seperate = 0)