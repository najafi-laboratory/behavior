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


window = tk.Tk()
window.wm_attributes('-topmost', 1)
window.withdraw()  
session_data_path = 'C:/Users/saminnaji3/Downloads/joystick/data'
subject = 'LG11'

file_paths_1 = list(filedialog.askopenfilenames(parent=window, initialdir=session_data_path+'\\'+ subject, title='Select '+ subject + ' Sessions'))
session_data_1 = DataIO1.read_trials(session_data_path, subject, file_paths_1)

output_dir_onedrive = 'C:/Users/saminnaji3/Downloads/joystick/fiure/'
output_dir_local = 'C:/Users/saminnaji3/Downloads/joystick/fiure/'



print(session_data_1['total_sessions'])
chemo = np.zeros(session_data_1['total_sessions'])
# FN13: '0730', '0726', '0724', '0722', '0718', '0716', '0714', '0712', '0710', '0618', '0613','0611', '0610', '0530', '0528'
# FN10; '0717','0719','0722'
# LG02:'0726', '0724', '0722', '0716', '0714', '0712', '0710', '0703', '0701', '0627', '0625', '0622', '0620', '0617', '0613', '0611', '0606', '0604', '0531', '0529', '0524'
# VG01: '0730', '0723', '0716', '0711', '0709', '0705', '0702', '0626', '0620', '0613', '0611', '0530', '0528', '0524'
# LG01_IO: '0910', '0905','0830' ,'0820', '0818', '0815', '0813', '0806', '0802', '0730', '0726', '0724', '0722', '0718', '0716', '0714', '0712', '0710', '0703'
# LG01_PPC: '0701', '0627', '0625', '0622', '0617', '0613', '0611', '0606', '0604', '0531', '0529', '0523', '0521', '0514', '0509' 
# YH4: '0703', '0701', '0627', '0625', '0622', '0620', '0617', '0613', '0611', '0606', '0604', '0531', '0529', '0429', '0424', '0417', '0410', '0408', '0405', '0403', '0401', '0329' 
# YH5: '0703','0627','0625','0622','0620','0617','0613','0611','0606','0604','0531','0529','0429','0424','0417','0410','0408','0405','0403','0401','0329'
# LG06:'0910', '0905','0830','0823','0813','0806','0802','0730','0712','0710'
# LG05: '1003','0911', '0905','0823','0813','0802','0730','0711','0709'
# LG07: '0827'0726, 0724, 0722, 0718
# LG08: '1002','0911','0906'
# LG09: '1002','0920','0906' NOTE: '1002' should be omitted 
# LG10: '1002','0911','0906'
# LG11: '1002','0913','0906'
chemo_sessions = [] # import dates as '0701' month+day
dates = session_data_1['dates']
for ch in chemo_sessions:
    if '2024' + ch in dates:
        chemo[dates.index('2024' + ch)] = 1
session_data_1['chemo'] = chemo
session_data_1['chemo_sess_num'] = len(chemo_sessions)


import importlib
import plot.fig1_outcome

importlib.reload(plot.fig1_outcome)
from plot.fig1_outcome import plot_fig1
plot_fig1(session_data_1, output_dir_onedrive, output_dir_local)

# for all (SUMMARY)
import importlib
import plot.fig1_outcome_short_long_1

importlib.reload(plot.fig1_outcome_short_long_1)
from plot.fig1_outcome_short_long_1 import plot_fig1_2
plot_fig1_2(session_data_1, output_dir_onedrive, output_dir_local)


# all data (short long summary)
import importlib
import plot.fig6_average_superimosed_short_long_all

importlib.reload(plot.fig6_average_superimosed_short_long_all)
from plot.fig6_average_superimosed_short_long_all import plot_fig_outcome_trajectories_sup_SL

plot_fig_outcome_trajectories_sup_SL(session_data_1, output_dir_onedrive, output_dir_local)



# Epoch outcome
from plot.fig_Outcome_Epoch import plot_outcome_epoh
import importlib
import plot.fig_Outcome_Epoch

importlib.reload(plot.fig_Outcome_Epoch)
from plot.fig_Outcome_Epoch import plot_outcome_epoh
plot_outcome_epoh(session_data_1, output_dir_onedrive, output_dir_local)


# Epoch trajectory
from plot.fig_average_trajectories_epoch_add import average_superimposed_epoch_add
import importlib
import plot.fig_average_trajectories_epoch_add

importlib.reload(plot.fig_average_trajectories_epoch_add)
from plot.fig_average_trajectories_epoch_add import average_superimposed_epoch_add
average_superimposed_epoch_add(session_data_1, output_dir_onedrive, output_dir_local)


# all data 
import importlib
import plot.fig_event_interval

importlib.reload(plot.fig_event_interval)
from plot.fig_event_interval import event_interval
event_interval(session_data_1, output_dir_onedrive, output_dir_local)


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