#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import random
import math
from matplotlib.lines import Line2D

def save_image(filename): 
     
    p = PdfPages(filename+'.pdf')
    fig_nums = plt.get_fignums()   
    figs = [plt.figure(n) for n in fig_nums] 
    for fig in figs:  
        fig.savefig(p, format='pdf', dpi=300)
          
    p.close() 

def plot_fig_outcome_trajectories_sup_all(
        session_data,
        output_dir_onedrive, 
        output_dir_local
        ):
    savefiles = 1
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_id = np.arange(len(outcomes)) + 1
    sess_vis1_rew_control = []
    sess_vis1_DNP1_control = []
    sess_vis1_DNP2_control = []
    sess_vis1_EP2_control = []
    sess_vis2_rew_control = []
    sess_vis2_DNP2_control = []
    sess_rew_control = []
    sess_num = np.zeros((4 , len(session_id)))

    if len(session_id)>1:
        output = np.zeros((2 , len(session_id)))
    else:
        output = np.zeros(2)
    for i in range(0 , len(session_id)):
        print('session id:' , session_id[i])
        vis1_data = session_data['encoder_positions_aligned_vis1'][i]
        vis2_data = session_data['encoder_positions_aligned_vis2'][i]
        rew_data = session_data['encoder_positions_aligned_rew'][i]
        outcome = outcomes[i]
        delay = session_data['session_press_delay'][i]
        vis1_rew_control = []
        vis1_DNP1_control = []
        vis1_DNP2_control = []
        vis1_EP2_control = []
        vis2_rew_control = []
        vis2_DNP2_control = []
        rew_control = []
        a = 0
        for j in range(0 , len(outcome)):
            if (outcome[j] == 'Reward'):
                vis1_rew_control.append(vis1_data[j])
                vis2_rew_control.append(vis2_data[j])
                rew_control.append(rew_data[a])
                sess_num[0 , i] += 1
                a = a+1
            elif (outcome[j] == 'DidNotPress1'):
                vis1_DNP1_control.append(vis1_data[j])
                sess_num[1 , i] += 1
            elif (outcome[j] == 'DidNotPress2'):
                vis1_DNP2_control.append(vis1_data[j])
                vis2_DNP2_control.append(vis2_data[j])
                sess_num[2 , i] += 1
            elif (outcome[j] == 'EarlyPress2'):
                vis1_EP2_control.append(vis1_data[j])
                sess_num[3 , i] += 1
                
        sess_vis1_rew_control.append(np.mean(np.array(vis1_rew_control) , axis = 0))
        sess_vis1_DNP1_control.append(np.mean(np.array(vis1_DNP1_control) , axis = 0))
        sess_vis1_DNP2_control.append(np.mean(np.array(vis1_DNP2_control) , axis = 0))
        sess_vis1_EP2_control.append(np.mean(np.array(vis1_EP2_control) , axis = 0))
        sess_vis2_rew_control.append(np.mean(np.array(vis2_rew_control) , axis = 0))
        sess_vis2_DNP2_control.append(np.mean(np.array(vis2_DNP2_control) , axis = 0))
        sess_rew_control.append(np.mean(np.array(rew_control) , axis = 0))
    

    
    press_window = session_data['session_press_window']
    vis_stim_2_enable = session_data['vis_stim_2_enable']
    encoder_times_vis1 = session_data['encoder_times_aligned_VisStim1']
    encoder_times_vis2 = session_data['encoder_times_aligned_VisStim2']
    encoder_times_rew = session_data['encoder_times_aligned_Reward'] 
    time_left_VisStim1 = session_data['time_left_VisStim1']
    time_right_VisStim1 = session_data['time_right_VisStim1']

    time_left_VisStim2 = session_data['time_left_VisStim2']
    time_right_VisStim2 = session_data['time_right_VisStim2']

    time_left_rew = session_data['time_left_rew']
    time_right_rew = session_data['time_right_rew']
    
    chemo_labels = session_data['chemo']
    control_sess = session_data['total_sessions'] - session_data['chemo_sess_num']
    chemo_sess = session_data['chemo_sess_num']
    control_count = 0
    chem_count = 0
    
    start_idx = 0
    stop_idx = session_data['total_sessions'] - 1

    target_thresh = session_data['session_target_thresh']
    t = []
    for i in range(0 , stop_idx+1):
        t.append(np.mean(target_thresh[i]))
    target_thresh = np.mean(t)

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.7)
    fig.suptitle(subject + ' Average Trajectories superimposed\n if any short exists is in dashed\n'+'target thresholds='+str(t))
    
    for i in range(0 , stop_idx+1):
        isSelfTimedMode = session_data['isSelfTimedMode'][i][0]
        isShortDelay = session_data['isShortDelay'][i]
        press_reps = session_data['press_reps'][i]
        press_window = session_data['press_window'][i]  


        y_top = 3.5
        if chemo_labels[i] == 1:
            c1 = (chemo_sess-chem_count)/(chemo_sess+1)
            chem_count = chem_count + 1
        else:
            c2 = (control_sess-control_count)/(control_sess+1)
            control_count = control_count + 1

        axs[0,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
        axs[0,0].set_title('VisStim1 Aligned.\n') 
        axs[0,0].axvline(x = 0, color = 'r', linestyle='--')
        axs[0,0].set_xlim(time_left_VisStim1, 4.0)
        axs[0,0].set_ylim(-0.2, target_thresh+1.25)
        axs[0,0].spines['right'].set_visible(False)
        axs[0,0].spines['top'].set_visible(False)
        axs[0,0].set_xlabel('Time from VisStim1 (s)')
        axs[0,0].set_ylabel('Joystick deflection (deg) Rewarded trials')
        if sess_num[0 , i] > 0:
            if chemo_labels[i] == 1:
                axs[0,0].plot(encoder_times_vis1 , sess_vis1_rew_control[i] , color = [c1 , 0 , 0], label=dates[i][4:]+'('+str(int(sess_num[0 , i]))+')'+ '(chemo)')
            else:
                axs[0,0].plot(encoder_times_vis1 , sess_vis1_rew_control[i] , color = [c2 , c2 , c2], label=dates[i][4:]+'('+str(int(sess_num[0 , i]))+')')
        axs[0,0].legend()


        axs[0,1].axvline(x = 0, color = 'r', linestyle='--')
        axs[0,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
        axs[0,1].set_title('WaitForPress2 Aligned.\n') 
        axs[0,1].set_xlim(-1, 2.0)
        axs[0,1].set_ylim(-0.2, y_top)
        axs[0,1].spines['right'].set_visible(False)
        axs[0,1].spines['top'].set_visible(False)
        axs[0,1].set_ylabel('Joystick deflection (deg)')
        if sess_num[0 , i] > 0:
            if chemo_labels[i] == 1:
                axs[0,1].plot(encoder_times_vis2 , sess_vis2_rew_control[i] , color = [c1 , 0 , 0], label=dates[i][4:]+'('+str(int(sess_num[0 , i]))+')'+ '(chemo)')
            else:
                axs[0,1].plot(encoder_times_vis2 , sess_vis2_rew_control[i] , color = [c2 , c2 , c2], label=dates[i][4:]+'('+str(int(sess_num[0 , i]))+')')


        axs[0,2].axvline(x = 0, color = 'r', linestyle='--')
        axs[0,2].axhline(y = target_thresh, color = '0.6', linestyle='--')
        axs[0,2].set_title('Reward Aligned.\n')                   
        axs[0,2].set_xlim(-1.0, 1.5)
        axs[0,2].set_ylim(-0.2, y_top)
        axs[0,2].spines['right'].set_visible(False)
        axs[0,2].spines['top'].set_visible(False)
        axs[0,2].set_xlabel('Time from Reward (s)')
        axs[0,2].set_ylabel('Joystick deflection (deg)')
        if sess_num[0 , i] > 0:
            if chemo_labels[i] == 1:
                axs[0,2].plot(encoder_times_rew , sess_rew_control[i] , color = [c1 , 0 , 0], label=dates[i][4:]+'('+str(int(sess_num[0 , i]))+')'+ '(chemo)')
            else:
                axs[0,2].plot(encoder_times_rew , sess_rew_control[i] , color = [c2 , c2 , c2], label=dates[i][4:]+'('+str(int(sess_num[0 , i]))+')')
        

        axs[1,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
        axs[1,0].set_title('VisStim1 Aligned.\n') 
        axs[1,0].axvline(x = 0, color = 'r', linestyle='--')
        axs[1,0].set_xlim(time_left_VisStim1, 4.0)
        axs[1,0].set_ylim(-0.2, target_thresh+1.25)
        axs[1,0].spines['right'].set_visible(False)
        axs[1,0].spines['top'].set_visible(False)
        axs[1,0].set_xlabel('Time from VisStim1 (s)')
        axs[1,0].set_ylabel('Joystick deflection (deg) DidNotPress1 Trials')
        if sess_num[1 , i] > 0:
            if chemo_labels[i] == 1:
                axs[1,0].plot(encoder_times_vis1 , sess_vis1_DNP1_control[i] , color = [c1 , 0 , 0], label=dates[i][4:]+'('+str(int(sess_num[1 , i]))+')'+'(chemo)')
            else: 
                axs[1,0].plot(encoder_times_vis1 , sess_vis1_DNP1_control[i] , color = [c2 , c2 , c2], label=dates[i][4:]+'('+str(int(sess_num[1 , i]))+')')
        axs[1,0].legend()

        axs[2,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
        axs[2,0].set_title('VisStim1 Aligned.\n') 
        axs[2,0].axvline(x = 0, color = 'r', linestyle='--')
        axs[2,0].set_xlim(time_left_VisStim1, 4.0)
        axs[2,0].set_ylim(-0.2, target_thresh+1.25)
        axs[2,0].spines['right'].set_visible(False)
        axs[2,0].spines['top'].set_visible(False)
        axs[2,0].set_xlabel('Time from VisStim1 (s)')
        axs[2,0].set_ylabel('Joystick deflection (deg) DidNotPress2 Trials')
        if sess_num[2 , i] > 0:
            if chemo_labels[i] == 1:
                axs[2,0].plot(encoder_times_vis1 , sess_vis1_DNP2_control[i] , color = [c1 , 0 , 0], label=dates[i][4:]+'('+str(int(sess_num[2 , i]))+')'+'(chemo)')
            else:
                axs[2,0].plot(encoder_times_vis1 , sess_vis1_DNP2_control[i] , color = [c2 , c2 , c2], label=dates[i][4:]+'('+str(int(sess_num[2 , i]))+')')
        axs[2,0].legend()


        axs[2,1].axvline(x = 0, color = 'r', linestyle='--')
        axs[2,1].axhline(y = target_thresh, color = '0.6', linestyle='--')
        axs[2,1].set_title('WaitForPress2 Aligned.\n') 
        axs[2,1].set_xlim(-1, 2.0)
        axs[2,1].set_ylim(-0.2, y_top)
        axs[2,1].spines['right'].set_visible(False)
        axs[2,1].spines['top'].set_visible(False)
        axs[2,1].set_ylabel('Joystick deflection (deg) DidNotPress2 Trials')
        if sess_num[2 , i] > 0:
            if chemo_labels[i] == 1:
                axs[2,1].plot(encoder_times_vis2 , sess_vis2_DNP2_control[i] , color = [c1 , 0 , 0])
            else:
                axs[2,1].plot(encoder_times_vis2 , sess_vis2_DNP2_control[i] , color = [c2 , c2 , c2])

        axs[3,0].axhline(y = target_thresh, color = '0.6', linestyle='--')
        axs[3,0].set_title('VisStim1 Aligned.\n') 
        axs[3,0].axvline(x = 0, color = 'r', linestyle='--')
        axs[3,0].set_xlim(time_left_VisStim1, 4.0)
        axs[3,0].set_ylim(-0.2, target_thresh+1.25)
        axs[3,0].spines['right'].set_visible(False)
        axs[3,0].spines['top'].set_visible(False)
        axs[3,0].set_xlabel('Time from VisStim1 (s)')
        axs[3,0].set_ylabel('Joystick deflection (deg) EarlyPress2 trials')
        if sess_num[3 , i] > 0:
            if chemo_labels[i] == 1:
                axs[3,0].plot(encoder_times_vis1 , sess_vis1_EP2_control[i] , color = [c1 , 0 , 0], label=dates[i][4:]+'('+str(int(sess_num[3 , i]))+')'+'(chemo)')
            else:
                axs[3,0].plot(encoder_times_vis1 , sess_vis1_EP2_control[i] , color = [c2 , c2 , c2], label=dates[i][4:]+'('+str(int(sess_num[3 , i]))+')')
        axs[3,0].legend()


    if savefiles:
        output_figs_dir = output_dir_onedrive + subject + '/'    
        output_imgs_dir = output_dir_local + subject + '/outcome_imgs/'    
        os.makedirs(output_figs_dir, exist_ok = True)
        os.makedirs(output_imgs_dir, exist_ok = True)
        fig.savefig(output_figs_dir + subject + '_average_trajectories_outcomevise_superimpose.pdf', dpi=300)
        fig.savefig(output_imgs_dir + subject  + '_average_trajectories_outcomevise_superimpose.png', dpi=300)

