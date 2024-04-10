import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import time
import tkinter as tk
from tkinter import filedialog


# updated to get session data from file open dialog box in main.py
# session_data_path = 'C:\\behavior\\joystick\\session_data_joystick_figs' 
# session_data_path = '.\\session_data'  # code test dir
session_data_path = 'C:/behavior/session_data' 


# read .mat to dict
def load_mat(fname):
    def _check_keys(d):
        for key in d:
            if isinstance(d[key], sio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, sio.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
    data = _check_keys(data)
    data = data['SessionData']
    return data


# extract all session data for a subject

def read_trials(subject, file_names):
    
    

    # file_names = os.listdir(os.path.join(session_data_path, subject))
    file_names.sort(key=lambda x: x[-19:])
    session_raw_data = []
    session_encoder_data = []
    encoder_time_max = 200
    ms_per_s = 1000
    
    time_left_VisStim1 = -0.1
    time_right_VisStim1 = 7
    time_VisStim1 = -time_left_VisStim1+time_right_VisStim1    
    
    time_left_VisStim2 = -3
    time_right_VisStim2 = 7
    time_VisStim2 = -time_left_VisStim2+time_right_VisStim2
    
    time_left_rew = -3
    time_right_rew = 7
    time_rew = -time_left_rew+time_right_rew
    
    session_encoder_times_aligned = np.linspace(0, encoder_time_max, num=encoder_time_max*ms_per_s)
    session_encoder_times_aligned_VisStim1 = np.linspace(time_left_VisStim1, time_right_VisStim1, num=int(time_VisStim1*ms_per_s))
    # session_encoder_times_aligned_VisStim2 = np.linspace(0, time_VisStim2, num=time_VisStim2*ms_per_s)
    session_encoder_times_aligned_VisStim2 = np.linspace(time_left_VisStim2, time_right_VisStim2, num=int(time_VisStim2*ms_per_s))
    session_encoder_times_aligned_Reward = np.linspace(time_left_rew, time_right_rew, num=int(time_rew*ms_per_s))
    session_encoder_positions_aligned = []
    session_encoder_positions_aligned_vis1 = []
    session_encoder_positions_aligned_vis2 = []
    session_encoder_positions_aligned_rew = []
    session_rewarded_trials = []
        
    session_encoder_positions_avg_vis1_short_rew = []
    session_encoder_positions_avg_vis1_long_rew = []    
    session_encoder_positions_avg_vis2_short_rew = []
    session_encoder_positions_avg_vis2_long_rew = []    
    session_encoder_positions_avg_rew_short = []
    session_encoder_positions_avg_rew_long = []
        
    # deprecated, these were static when we were using same settings for all sessions/trials
    # session_press_reps = 0
    # session_press_window = 0
        
    session_press_reps = []
    session_press_window = []
        
    
    session_target_thresh = []
    VisStim2Enable = 1
    session_InterruptedVisStimTrials = []
    
    # session modes and delays
    session_isSelfTimedMode = []
    session_isShortDelay = []
    
    session_press_delay = []
    
    session_press_delay_avg = []
    session_press_delay_short_avg = []
    session_press_delay_long_avg = []    
    
    session_short_delay_hit_rate = []
    session_long_delay_hit_rate = []
    
    session_short_delay_pun_rate = []
    session_long_delay_pun_rate = []
    
    session_num_short = []
    session_num_long = []
    
    session_short_num_rew = []      
    session_short_num_pun = []
    session_long_num_rew = []
    session_long_num_pun = []
    
    session_short_num= []
    session_long_num = []
    
    # session_encoder_positions_avg = []
    session_encoder_positions_avg_vis1 = []
    session_encoder_positions_avg_vis2 = []
    session_encoder_positions_avg_rew = []
    session_post_lick = []
    session_choice = []
    session_com = []
    session_dates = []
    session_outcomes = []
    session_licking = []
    session_reaction = []
    session_iti = []
    session_isi = []
    session_avsync = []
    LR12_start = 0

    
    for f in range(len(file_names)):
        # the first session with LR_12
        fname = file_names[f]
        print(fname)
        if LR12_start==0 and fname[14:16]=='12':
            LR12_start = f
        # one session data
        raw_data = load_mat(os.path.join(session_data_path, subject, fname))
        session_raw_data.append(raw_data)
        # number of trials
        nTrials = raw_data['nTrials']
        # trial target
        TrialTypes = raw_data['TrialTypes']
        # session date
        session_dates.append(fname[-19:-11])
        # loop over one session for extracting data
        trial_encoder_data = []
        trial_encoder_positions_aligned = []
        trial_encoder_positions_aligned_vis1 = []
        trial_encoder_positions_aligned_vis1_rew = []
        
        trial_encoder_positions_aligned_vis1_short = []
        trial_encoder_positions_aligned_vis1_rew_short = []
        trial_encoder_positions_aligned_vis1_long = []
        trial_encoder_positions_aligned_vis1_rew_long = []
        
        trial_encoder_positions_aligned_vis2_short = []
        trial_encoder_positions_aligned_vis2_long = []      
        trial_encoder_positions_aligned_vis2_rew_short = []
        trial_encoder_positions_aligned_vis2_rew_long = []
                
        trial_encoder_positions_aligned_rew_short = []
        trial_encoder_positions_aligned_rew_long = []                
        
        
        
        
        trial_encoder_times_aligned_vis1 = []
        trial_encoder_positions_aligned_vis2 = []
        trial_encoder_positions_aligned_vis2_rew = []
        trial_encoder_times_aligned_vis2 = []
        trial_encoder_positions_aligned_rew = []
        trial_num_rewarded = []
        
        trial_num_short_pun = []
        trial_num_long_pun = []

        trial_short_num = []
        trial_long_num = []
        
        trial_reps = 0
        trial_InterruptedVisStimTrials = []
        
        
        # mode and delay vars
        trial_isSelfTimedMode = []
        trial_isShortDelay = []
        
        trial_press_delay = []
        trial_press_delay_short = []
        trial_press_delay_long = []
        
        trial_press_reps = []
        trial_press_window = []
        
        trial_target_thresh = []
        
        # lick vars
        trial_lick_data = []
        
        trial_post_lick = []
        trial_outcomes = []
        trial_licking = []
        trial_reaction = []
        trial_iti = []
        trial_isi = []
        trial_avsync = []
        trial_choice = []
        trial_com = []
        # for i in range(nTrials):
        
        # for i in range(nTrials):
        #     trial_states = raw_data['RawEvents']['Trial'][i]['States']    
        
        for i in range(nTrials):
        # for i in range(0, 100):
            # handle key error if only one trial in a session
            if nTrials > 1:
                trial_states = raw_data['RawEvents']['Trial'][i]['States']
                trial_events = raw_data['RawEvents']['Trial'][i]['Events']                
                trial_reps = raw_data['TrialSettings'][i]['GUI']['Reps']
                # session_press_window = raw_data['TrialSettings'][i]['GUI']['PressWindow_s']  # deprecated, press window changes during session, needs update
            else:
                continue
                # trial_states = raw_data['RawEvents']['Trial']['States']
                # trial_events = raw_data['RawEvents']['Trial']['Events']            
            
            # trial GUI params
            trial_GUI_Params = raw_data['TrialSettings'][i]['GUI']
            
            # outcome
            outcome = states_labeling(trial_states, trial_reps)
            trial_outcomes.append(outcome)
       
            # mode: vis-guided or self timed
            isSelfTimedMode = 0    # assume vis-guided (all sessions were vi)
            if 'SelfTimedMode' in trial_GUI_Params:
                isSelfTimedMode = trial_GUI_Params['SelfTimedMode']
            
            trial_isSelfTimedMode.append(isSelfTimedMode)            
            
            # if vis-guided, short/long delay trial?
            # if vis-guided, vis stim 2 enabled? 
            isShortDelay = 0
            if not isSelfTimedMode:                
                trial_type = raw_data['TrialTypes'][i]
                if trial_type == 1:
                    isShortDelay = 1
                                  
            trial_isShortDelay.append(isShortDelay)
                
            if isSelfTimedMode:
                press_delay = trial_GUI_Params['PrePress2Delay_s']
            else:
                if isShortDelay:
                    press_delay = trial_GUI_Params['PressVisDelayShort_s']
                    trial_press_delay_short.append(press_delay)
                    trial_short_num.append(i)
                else:
                    press_delay = trial_GUI_Params['PressVisDelayLong_s']
                    trial_press_delay_long.append(press_delay)
                    trial_long_num.append(i)
                    
            trial_press_delay.append(press_delay)
                        
        
            # press reps 
            press_reps = trial_GUI_Params['Reps']
            trial_press_reps.append(press_reps)
            
            # press window
            press_window = trial_GUI_Params['PressWindow_s']
            press_window_extend = trial_GUI_Params['PressWindowExtend_s']            
            
            # update to add extended window for warmup            
            trial_press_window.append(press_window)
        
       
            # encoder data                
            encoder_data = raw_data['EncoderData'][i]
            trial_encoder_data.append(encoder_data)
            encoder_data_aligned = {}
          
            # interpolate encoder data to align times for averaging
            # maybe only use one array of linspace for efficiency
            # trial_reps = trial_GUI_Params['Reps']
            trial_target_thresh.append(trial_GUI_Params['Threshold'])
            # if (fname == 'YH5_Joystick_visual_7_20240209_150957.mat'):
            #     print()
                        
            
            if encoder_data['nPositions']:                
                times = encoder_data['Times']
                positions = encoder_data['Positions']
            else:
                times = [0.0, trial_states['ITI'][1]]
                positions = [0.0, 0.0]
                
            # if outcome == 'Other':
                # print('Other')
            
            # all trials encoder positions
            # encoder_data_aligned = np.interp(session_encoder_times_aligned, times, positions)
            # all trials encoder data
            # encoder_data_aligned = {'Positions': encoder_data_aligned}
            # process encoder data for rewarded trials
            # if outcome == 'Reward':                    
            encoder_data_aligned = np.interp(session_encoder_times_aligned, times, positions)
            
            trial_encoder_positions_aligned.append(encoder_data_aligned)
            
            # find times and pos aligned to vis stim 1
            if 'VisualStimulus1' in trial_states.keys() and not np.isnan(trial_states['VisualStimulus1'][0]):
                VisStim1Start = trial_states['VisualStimulus1'][0]
            elif 'VisStimInterrupt' in trial_states.keys() and not np.isnan(trial_states['VisStimInterrupt'][0]):                
                trial_InterruptedVisStimTrials.append(i)
                VisStim1Start = 0
            else:
                print('Should be either vis 1 or vis interrupt, check')
            
            if VisStim1Start > 12:
                continue # vis detect missed stim, go to next trial
                
            vis_diff = np.abs(VisStim1Start - session_encoder_times_aligned)
            min_vis_diff = np.min(np.abs(VisStim1Start - session_encoder_times_aligned))
            # print(i)
            closest_aligned_time_vis1_idx = [ind for ind, ele in enumerate(vis_diff) if ele == min_vis_diff][0]
                              
            left_idx_VisStim1 = int(closest_aligned_time_vis1_idx+time_left_VisStim1*ms_per_s)
            right_idx_VisStim1 = int(closest_aligned_time_vis1_idx+(time_right_VisStim1*ms_per_s))
            # pad with nan if left idx < 0
            if left_idx_VisStim1 < 0:
                nan_pad = np.zeros(-left_idx_VisStim1)
                nan_pad[:] = np.nan
                trial_encoder_positions_aligned_VisStim1 = np.append(nan_pad, encoder_data_aligned[0:right_idx_VisStim1])
            else:                        
                trial_encoder_positions_aligned_VisStim1 = encoder_data_aligned[left_idx_VisStim1:right_idx_VisStim1]
            
            trial_encoder_positions_aligned_vis1.append(trial_encoder_positions_aligned_VisStim1)
            
            # get separate short and long delay avgs
            if not isSelfTimedMode:
                if isShortDelay:
                    trial_encoder_positions_aligned_vis1_short.append(trial_encoder_positions_aligned_VisStim1)
                else:               
                    # print("long delay trial", i)
                    trial_encoder_positions_aligned_vis1_long.append(trial_encoder_positions_aligned_VisStim1)
            
            if outcome == 'Reward':
                trial_encoder_positions_aligned_vis1_rew.append(trial_encoder_positions_aligned_VisStim1)
                # get separate short and long delay avgs - rewarded
                if not isSelfTimedMode:
                    if isShortDelay:
                        trial_encoder_positions_aligned_vis1_rew_short.append(trial_encoder_positions_aligned_VisStim1)
                    else:                    
                        trial_encoder_positions_aligned_vis1_rew_long.append(trial_encoder_positions_aligned_VisStim1)            
            
            
            # plt.plot(trial_encoder_times_aligned_VisStim1, trial_encoder_positions_aligned_VisStim1)
            # plt.plot(session_encoder_times_aligned_VisStim1, trial_encoder_positions_aligned_VisStim1)
            
            
            HasVis2 = 0
            VisStim2Start = 0
            if not np.isnan(trial_states['VisualStimulus2'][0]):
                VisStim2Start = trial_states['VisualStimulus2'][0]
                HasVis2 = 1
            elif not np.isnan(trial_states['WaitForPress2'][0]):
                VisStim2Start = trial_states['WaitForPress2'][0]
                HasVis2 = 1
            else:
                HasVis2 = 0
            
            # find times and pos aligned to vis stim 2
            if "VisStim2Enable" in trial_GUI_Params:
                VisStim2Enable = trial_GUI_Params['VisStim2Enable']
            
            # !!update this later to account more mixed trials in a given session with two array averages for align
            # OR align at WaitForPress 
            # if VisStim2Enable and not np.isnan(trial_states['VisualStimulus2'][0]):
            #     VisStim2Start = trial_states['VisualStimulus2'][0]                        
            # elif  not np.isnan(trial_states['WaitForPress2'][0]):
            #     VisStim2Start = trial_states['WaitForPress2'][0]
            # else:
                

            if trial_reps > 1 and HasVis2:
                vis_diff = np.abs(VisStim2Start - session_encoder_times_aligned)
                min_vis_diff = np.min(np.abs(VisStim2Start - session_encoder_times_aligned))
                # print(i)
                
                closest_aligned_time_vis2_idx = [ind for ind, ele in enumerate(vis_diff) if ele == min_vis_diff][0]
                left_idx_VisStim2 = int(closest_aligned_time_vis2_idx+time_left_VisStim2*ms_per_s)
                right_idx_VisStim2 = int(closest_aligned_time_vis2_idx+(time_right_VisStim2*ms_per_s))
                # pad with nan if left idx < 0
                if left_idx_VisStim2 < 0:
                    nan_pad = np.zeros(-left_idx_VisStim2)
                    nan_pad[:] = np.nan
                    trial_encoder_positions_aligned_VisStim2 = np.append(nan_pad, encoder_data_aligned[0:right_idx_VisStim2])
                else:                       
                    # trial_encoder_times_aligned_VisStim2 = session_encoder_times_aligned[left_idx_VisStim2:right_idx_VisStim2]
                    trial_encoder_positions_aligned_VisStim2 = encoder_data_aligned[left_idx_VisStim2:right_idx_VisStim2]
                
                trial_encoder_positions_aligned_vis2.append(trial_encoder_positions_aligned_VisStim2)
                
                if not isSelfTimedMode:
                    if isShortDelay:
                        trial_encoder_positions_aligned_vis2_short.append(trial_encoder_positions_aligned_VisStim2)
                    else:
                        trial_encoder_positions_aligned_vis2_long.append(trial_encoder_positions_aligned_VisStim2)
                
                if outcome == 'Reward':
                    trial_encoder_positions_aligned_vis2_rew.append(trial_encoder_positions_aligned_VisStim2)
                    
                    if not isSelfTimedMode:
                        if isShortDelay:
                            trial_encoder_positions_aligned_vis2_rew_short.append(trial_encoder_positions_aligned_VisStim2)
                        else:
                            trial_encoder_positions_aligned_vis2_rew_long.append(trial_encoder_positions_aligned_VisStim2)
            else:
                trial_encoder_positions_aligned_VisStim2 = np.zeros(session_encoder_times_aligned_VisStim2.size)
                trial_encoder_positions_aligned_VisStim2[:] = np.nan
                trial_encoder_positions_aligned_vis2.append(trial_encoder_positions_aligned_VisStim2)
               
                if not isSelfTimedMode:
                    if isShortDelay:
                        trial_encoder_positions_aligned_vis2_short.append(trial_encoder_positions_aligned_VisStim2)
                    else:
                        trial_encoder_positions_aligned_vis2_long.append(trial_encoder_positions_aligned_VisStim2)
                
                if outcome == 'Reward':
                    trial_encoder_positions_aligned_vis2_rew.append(trial_encoder_positions_aligned_VisStim2)  
                   
                    if not isSelfTimedMode:
                        if isShortDelay:
                            trial_encoder_positions_aligned_vis2_rew_short.append(trial_encoder_positions_aligned_VisStim2)
                        else:
                            trial_encoder_positions_aligned_vis2_rew_long.append(trial_encoder_positions_aligned_VisStim2)
                


                # print(i)
                # plt.plot(session_encoder_times_aligned[0:5000], encoder_data_aligned[0:5000])
                #plt.plot(session_encoder_times_aligned_VisStim2, trial_encoder_positions_aligned_VisStim2)
            
            if outcome == 'Reward':
                # find times and pos aligned to reward
                if trial_reps == 3:
                    RewardStart = trial_states['Reward3'][0]
                elif trial_reps == 2:
                    if 'Reward2' in trial_states.keys() and not np.isnan(trial_states['Reward2'][0]):
                        RewardStart = trial_states['Reward2'][0]
                    elif 'Reward' in trial_states.keys() and not np.isnan(trial_states['Reward'][0]):
                            RewardStart = trial_states['Reward'][0]
                elif trial_reps == 1:
                    if 'Reward' in trial_states.keys() and not np.isnan(trial_states['Reward'][0]):
                        RewardStart = trial_states['Reward'][0]
                    elif 'Reward1' in trial_states.keys() and not np.isnan(trial_states['Reward1'][0]):
                        RewardStart = trial_states['Reward1'][0]
                    
                rew_diff = np.abs(RewardStart - session_encoder_times_aligned)
                min_rew_diff = np.min(np.abs(RewardStart - session_encoder_times_aligned))
                closest_aligned_time_rew_idx = [ind for ind, ele in enumerate(rew_diff) if ele == min_rew_diff][0]
                left_idx_rew = int(closest_aligned_time_rew_idx+time_left_rew*ms_per_s)
                right_idx_rew = int(closest_aligned_time_rew_idx+(time_right_rew*ms_per_s))
                # pad with nan if left idx < 0
                if left_idx_rew < 0:
                    nan_pad = np.zeros(-left_idx_rew)
                    nan_pad[:] = np.nan
                    trial_encoder_positions_aligned_Reward = np.append(nan_pad, encoder_data_aligned[0:right_idx_rew])
                else:                       
                    trial_encoder_positions_aligned_Reward = encoder_data_aligned[left_idx_rew:right_idx_rew]
                
                # if i == 121:
                #     print()
                trial_encoder_positions_aligned_rew.append(trial_encoder_positions_aligned_Reward)
                
                if not isSelfTimedMode:
                    if isShortDelay:
                        trial_encoder_positions_aligned_rew_short.append(trial_encoder_positions_aligned_Reward)
                    else:
                        trial_encoder_positions_aligned_rew_long.append(trial_encoder_positions_aligned_Reward)
                # index of rewarded trials
                trial_num_rewarded.append(i)
           
            elif outcome == 'Punish':
                if not isSelfTimedMode:
                    if isShortDelay:
                        # trial_encoder_positions_aligned_rew_short.append(trial_encoder_positions_aligned_Reward)                        
                        # index of punished short delay trials
                        trial_num_short_pun.append(i)
                    else:
                        # trial_encoder_positions_aligned_rew_long.append(trial_encoder_positions_aligned_Reward)                
                        # index of punished long delay trials
                        trial_num_long_pun.append(i)
            # elif outcome == 'Punish':
            
                
            # e
            
            
            # plt.plot(session_encoder_times_aligned[0:5000], encoder_data_aligned[0:5000])
            #plt.plot(session_encoder_times_aligned_Reward, trial_encoder_positions_aligned_Reward)
              
                
        # encoder trajectory average across session for rewarded trials, vis stim 1 aligned        
        try:
            pos_vis1 = np.sum(trial_encoder_positions_aligned_vis1_rew[0:], axis=0)
        except:
            print(fname)
            time.sleep(15)
           
        # vis 1 session average - rewarded
        pos_vis1 = np.sum(trial_encoder_positions_aligned_vis1_rew[0:], axis=0)        
        sess_enc_avg_vis1 = pos_vis1/len(trial_num_rewarded)
        
        
        numTrials = len(trial_outcomes)
        numRewPunTrials = numTrials - trial_outcomes.count('Other')    
        
        sess_enc_avg_vis1_short_rew = 0
        sess_enc_avg_vis1_long_rew = 0
        sess_short_hit_rate = 0
        sess_long_hit_rate = 0
        if not isSelfTimedMode:
            # if isShortDelay:
                # vis 1 session average - short - rewarded
                if len(trial_encoder_positions_aligned_vis1_rew_short[0:]) > 0:
                    pos_vis1_short_rew = np.sum(trial_encoder_positions_aligned_vis1_rew_short[0:], axis=0)       
                    sess_enc_avg_vis1_short_rew = pos_vis1_short_rew/len(trial_encoder_positions_aligned_vis1_rew_short[0:])
                
                # short delay hit rate
                if len(trial_encoder_positions_aligned_vis1_short[0:]) > 0:
                    sess_short_num_rew = len(trial_encoder_positions_aligned_vis1_rew_short[0:])
                    
                    # sess_short_num_pun = len(trial_encoder_positions_aligned_vis1_short[0:]) - sess_short_num_rew
                    sess_short_num_pun = len(trial_num_short_pun)
                    
                    # sess_short_hit_rate = round(sess_short_num_rew/len(trial_encoder_positions_aligned_vis1_short[0:]), 2)
                    # sess_short_hit_rate = round(sess_short_num_rew/len(trial_short_num), 2)
                    sess_short_hit_rate = round(sess_short_num_rew/(sess_short_num_rew + sess_short_num_pun), 2)
                   
                    # sess_short_pun_rate = round(1 - sess_short_hit_rate, 2)
                    # sess_short_pun_rate = round(sess_short_num_pun/len(trial_short_num), 2)
                    sess_short_pun_rate = round(sess_short_num_pun/(sess_short_num_rew + sess_short_num_pun), 2)
                   
                    print('sess_short_num_rew', sess_short_num_rew)
                    print('sess_short_num_pun', sess_short_num_pun)                     
                    
                    # print('trial_num_short_pun', len(trial_num_short_pun))
                    
                    print('sess_short_hit_rate', sess_short_hit_rate)
                    print('sess_short_pun_rate', sess_short_pun_rate)
                    print('sess_short_hit_rate + sess_short_pun_rate', round(sess_short_hit_rate + sess_short_pun_rate, 2))
            # else:        
                # vis 1 session average - long - rewarded
                # print('len(trial_encoder_positions_aligned_vis1_rew_long)', len(trial_encoder_positions_aligned_vis1_rew_long))
                if len(trial_encoder_positions_aligned_vis1_rew_long[0:]) > 0:
                    pos_vis1_long_rew = np.sum(trial_encoder_positions_aligned_vis1_rew_long[0:], axis=0)        
                    sess_enc_avg_vis1_long_rew = pos_vis1_long_rew/len(trial_encoder_positions_aligned_vis1_rew_long[0:])
                                
                # long delay hit rate
                if len(trial_encoder_positions_aligned_vis1_long[0:]) > 0:
                    sess_long_num_rew = len(trial_encoder_positions_aligned_vis1_rew_long[0:])
                    # sess_long_num_pun = len(trial_encoder_positions_aligned_vis1_long[0:]) - sess_long_num_rew                    
                    sess_long_num_pun = len(trial_num_long_pun)   
                    
                    # sess_long_hit_rate = round(sess_long_num_rew/len(trial_encoder_positions_aligned_vis1_long[0:]), 2)
                    # sess_long_hit_rate = round(sess_long_num_rew/len(trial_long_num), 2)
                    sess_long_hit_rate = round(sess_long_num_rew/(sess_long_num_rew + sess_long_num_pun), 2)
                    
                    # sess_long_pun_rate = round(1 - sess_long_hit_rate, 2)
                    # sess_long_pun_rate = round(sess_long_num_pun/len(trial_long_num), 2)
                    sess_long_pun_rate = round(sess_long_num_pun/(sess_long_num_rew + sess_long_num_pun), 2)
                    
                    print('sess_long_num_rew', sess_long_num_rew)
                    print('sess_long_num_pun', sess_long_num_pun)  
                    
                    
                    # print('trial_num_long_pun', len(trial_num_long_pun))
                    
                    print('sess_long_hit_rate', sess_long_hit_rate)                    
                    print('sess_long_pun_rate', sess_long_pun_rate)        
                    print('sess_long_hit_rate + sess_long_pun_rate', round(sess_long_hit_rate + sess_long_pun_rate, 2))   
        
        if 0:        
            for i in range(len(trial_encoder_positions_aligned[0:4])):
                plt.plot(session_encoder_times_aligned_VisStim1,trial_encoder_positions_aligned_vis1_rew[i], label=i)
                plt.legend(loc='upper right')
                # plt.show()
        
                plt.plot(session_encoder_times_aligned_VisStim1,sess_enc_avg_vis1)
        
        # encoder trajectory average across session for rewarded trials, vis stim 2 aligned
        pos_vis2 = np.sum(trial_encoder_positions_aligned_vis2_rew, axis=0)        
        sess_enc_avg_vis2 = pos_vis2/len(trial_num_rewarded)
        
        
        
        sess_enc_avg_vis2_short_rew = 0
        sess_enc_avg_vis2_long_rew = 0
        if not isSelfTimedMode:
            # if isShortDelay:
                # vis 2 session average - short - rewarded
                pos_vis2_short_rew = np.sum(trial_encoder_positions_aligned_vis2_rew_short[0:], axis=0)        
                sess_enc_avg_vis2_short_rew = pos_vis2_short_rew/len(trial_encoder_positions_aligned_vis2_rew_short[0:]) 
                
            # else:
                # vis 2 session average - long - rewarded
                pos_vis2_long_rew = np.sum(trial_encoder_positions_aligned_vis2_rew_long[0:], axis=0)        
                sess_enc_avg_vis2_long_rew = pos_vis2_long_rew/len(trial_encoder_positions_aligned_vis2_rew_long[0:])
                        
        if 0:
            plt.plot(session_encoder_times_aligned_VisStim2, sess_enc_avg_vis2)
                
            for i in range(10):
                plt.plot(session_encoder_times_aligned_VisStim2,trial_encoder_positions_aligned_vis2_rew[i], label=i)
                plt.legend(loc='upper right')
                # plt.show()
                     
            for i in range(len(trial_encoder_positions_aligned[0:4])):
                plt.plot(session_encoder_times_aligned,trial_encoder_positions_aligned[i], label=i)
                plt.legend(loc='upper right')
                # plt.show()
               
            for i in range(len(trial_encoder_positions_aligned[0:4])):
                plt.plot(session_encoder_times_aligned[0:7000],trial_encoder_positions_aligned[i][0:7000], label=i)
                plt.legend(loc='upper right')
        

        pos_rew = np.sum(trial_encoder_positions_aligned_rew, axis=0)
        sess_enc_avg_rew = pos_rew/len(trial_num_rewarded)
        
        
        sess_enc_avg_rew_short = 0
        sess_enc_avg_rew_long = 0
        if not isSelfTimedMode:
            # if isShortDelay:
                # rew session average - short
                pos_rew_short = np.sum(trial_encoder_positions_aligned_rew_short[0:], axis=0)        
                sess_enc_avg_rew_short = pos_rew_short/len(trial_encoder_positions_aligned_rew_short[0:]) 
            # else:
                # rew session average - long
                pos_rew_long = np.sum(trial_encoder_positions_aligned_rew_long[0:], axis=0)        
                sess_enc_avg_rew_long = pos_vis2_long_rew/len(trial_encoder_positions_aligned_rew_long[0:])         
        
        if 0:
            for i in range(len(trial_encoder_positions_aligned[0:4])):
                plt.plot(session_encoder_times_aligned_Reward,trial_encoder_positions_aligned_rew[i], label=i)
                plt.legend(loc='upper right')
            
            
            plt.plot(session_encoder_times_aligned_Reward, sess_enc_avg_rew)
        
        
        # press delay averages
        sess_press_delay_avg = np.sum(trial_press_delay, axis=0)/len(trial_press_delay)
        
        
        if len(trial_press_delay_short) > 0:
            sess_press_delay_short_avg = np.sum(trial_press_delay_short , axis=0)/len(trial_press_delay_short)
        else:
            sess_press_delay_short_avg = 0
        
        if len(trial_press_delay_long) > 0:
            sess_press_delay_long_avg = np.sum(trial_press_delay_long , axis=0)/len(trial_press_delay_long)
        else:
            sess_press_delay_long_avg = 0
        
        
        
        
        session_target_thresh.append(trial_target_thresh)
        # session_press_reps = trial_reps
        
        # save one session data
        session_encoder_data.append(trial_encoder_data)
        session_encoder_positions_aligned.append(trial_encoder_positions_aligned)
        session_encoder_positions_aligned_vis1.append(trial_encoder_positions_aligned_vis1)
        session_encoder_positions_aligned_vis2.append(trial_encoder_positions_aligned_vis2)
        session_encoder_positions_aligned_rew.append(trial_encoder_positions_aligned_rew)
        session_rewarded_trials.append(trial_num_rewarded)
        session_InterruptedVisStimTrials.append(trial_InterruptedVisStimTrials)
        session_encoder_positions_avg_vis1.append(sess_enc_avg_vis1)
        session_encoder_positions_avg_vis2.append(sess_enc_avg_vis2)
        session_encoder_positions_avg_rew.append(sess_enc_avg_rew)
        
        session_encoder_positions_avg_vis1_short_rew.append(sess_enc_avg_vis1_short_rew)
        session_encoder_positions_avg_vis1_long_rew.append(sess_enc_avg_vis1_long_rew)
        
        session_encoder_positions_avg_vis2_short_rew.append(sess_enc_avg_vis2_short_rew)
        session_encoder_positions_avg_vis2_long_rew.append(sess_enc_avg_vis2_long_rew)
        
        session_encoder_positions_avg_rew_short.append(sess_enc_avg_rew_short)
        session_encoder_positions_avg_rew_long.append(sess_enc_avg_rew_long)
        
        session_press_delay.append(trial_press_delay)
        
        session_press_delay_avg.append(sess_press_delay_avg)
        session_press_delay_short_avg.append(sess_press_delay_short_avg)
        session_press_delay_long_avg.append(sess_press_delay_long_avg)   
        
        session_short_delay_hit_rate.append(sess_short_hit_rate)
        session_short_delay_pun_rate.append(sess_short_pun_rate)
        session_long_delay_hit_rate.append(sess_long_hit_rate)
        session_long_delay_pun_rate.append(sess_long_pun_rate)
        
        session_short_num_rew.append(sess_short_num_rew)        
        session_short_num_pun.append(sess_short_num_pun)
        session_long_num_rew.append(sess_long_num_rew)
        session_long_num_pun.append(sess_long_num_pun)
                        
        # session_short_num_pun.append(trial_num_short_pun)
        # session_long_num_pun.append(trial_num_long_pun)        

                    
        session_short_num.append(trial_short_num)
        session_long_num.append(trial_long_num)

        # sess_enc_avg_vis1_short_rew = 0
        # sess_enc_avg_vis1_long_rew = 0
        
        # sess_enc_avg_vis2_short_rew = 0
        # sess_enc_avg_vis2_long_rew = 0
        
        # sess_enc_avg_rew_short = 0
        # sess_enc_avg_rew_long = 0
        
        # session_encoder_positions_avg.append(sess_enc_avg)
        session_choice.append(trial_choice)
        session_com.append(trial_com)
        session_post_lick.append(trial_post_lick)
        session_outcomes.append(trial_outcomes)
        session_reaction.append(trial_reaction)
        session_iti.append(trial_iti)
        session_licking.append(trial_licking)
        session_isi.append(trial_isi)
        session_avsync.append(trial_avsync)
        
        session_isSelfTimedMode.append(trial_isSelfTimedMode)
        session_isShortDelay.append(trial_isShortDelay)
        
        session_press_reps.append(trial_press_reps)
        session_press_window.append(trial_press_window)
    
    # merge all session data
    data = {
        'total_sessions' : len(session_outcomes),
        'subject' : subject,
        'filename' : file_names,
        'LR12_start' : LR12_start,
        'raw' : session_raw_data,
        'dates' : session_dates,
    	'outcomes' : session_outcomes,
        'iti' : session_iti,
        'reaction' : session_reaction,
        'licking' : session_licking,
        'choice' : session_choice,
        'encoder' : session_encoder_data,
        'encoder_time_aligned' : session_encoder_times_aligned,
        'encoder_pos_aligned' : session_encoder_positions_aligned,
        'time_left_VisStim1' : time_left_VisStim1,
        'time_right_VisStim1' : time_right_VisStim1,        
        'time_left_VisStim2' : time_left_VisStim2,
        'time_right_VisStim2' : time_right_VisStim2,        
        'time_left_rew' : time_left_rew,
        'time_right_rew' : time_right_rew,        
        'encoder_times_aligned_VisStim1': session_encoder_times_aligned_VisStim1,
        'encoder_times_aligned_VisStim2': session_encoder_times_aligned_VisStim2,
        'encoder_times_aligned_Reward': session_encoder_times_aligned_Reward,
        'encoder_positions_aligned_vis1': session_encoder_positions_aligned_vis1,
        'encoder_positions_aligned_vis2': session_encoder_positions_aligned_vis2,
        'encoder_positions_aligned_rew': session_encoder_positions_aligned_rew,
        'rewarded_trials' : session_rewarded_trials,
        'session_InterruptedVisStimTrials' : session_InterruptedVisStimTrials,
        'session_target_thresh' : session_target_thresh,
        'session_press_reps' : session_press_reps,
        'session_press_window' : session_press_window,
        'vis_stim_2_enable' : VisStim2Enable,
        'encoder_pos_avg_vis1' : session_encoder_positions_avg_vis1,        
        'encoder_pos_avg_vis2' : session_encoder_positions_avg_vis2,        
        'encoder_pos_avg_rew' : session_encoder_positions_avg_rew,         
        'encoder_pos_avg_vis1_short' : session_encoder_positions_avg_vis1_short_rew,
        'encoder_pos_avg_vis1_long' : session_encoder_positions_avg_vis1_long_rew,        
        'encoder_pos_avg_vis2_short' : session_encoder_positions_avg_vis2_short_rew,
        'encoder_pos_avg_vis2_long' : session_encoder_positions_avg_vis2_long_rew,        
        'encoder_pos_avg_rew_short' : session_encoder_positions_avg_rew_short,
        'encoder_pos_avg_rew_long' : session_encoder_positions_avg_rew_long,
        'session_press_delay' : session_press_delay,
        'session_press_delay_avg' : session_press_delay_avg,
        'session_press_delay_short_avg' : session_press_delay_short_avg,
        'session_press_delay_long_avg' : session_press_delay_long_avg,
        'session_short_delay_hit_rate' : session_short_delay_hit_rate,
        'session_long_delay_hit_rate' : session_long_delay_hit_rate,           
        'session_short_delay_pun_rate' : session_short_delay_pun_rate,
        'session_long_delay_pun_rate' : session_long_delay_pun_rate,        
        'session_short_num_rew' : session_short_num_rew,
        'session_short_num_pun' : session_short_num_pun,
        'session_long_num_rew' : session_long_num_rew,
        'session_long_num_pun' : session_long_num_pun,   
        
        # 'session_short_num_pun' : session_short_num_pun,
        # 'session_long_num_pun' : session_num_short_pun,  
        
        'session_short_num' : session_short_num,
        'session_long_num' : session_long_num,        
        'com' : session_com,
        'post_lick' : session_post_lick,
        'isi' : session_isi,
        'avsync' : session_avsync,
        'isSelfTimedMode' : session_isSelfTimedMode,
        'isShortDelay' : session_isShortDelay,
        'press_reps' : session_press_reps,
        'press_window' : session_press_window
    }
    return data


# labeling every trials for a subject
def states_labeling(trial_states, reps):
    # if 'DidNotPress1' in trial_states.keys() and not np.isnan(trial_states['DidNotPress1'][0]):
    #     outcome = 'DidNotPress1'
    # elif 'DidNotPress2' in trial_states.keys() and not np.isnan(trial_states['DidNotPress2'][0]):
    #     outcome = 'DidNotPress2'
    # elif 'DidNotPress3' in trial_states.keys() and not np.isnan(trial_states['DidNotPress3'][0]):
    #     outcome = 'DidNotPress3'
    # if 'Punish' in trial_states.keys() and not np.isnan(trial_states['Punish'][0]):
    #     outcome = 'Punish'
    # elif reps == 1 and 'Reward1' in trial_states.keys() and not np.isnan(trial_states['Reward1'][0]):
    #     outcome = 'Reward'
    # elif reps == 2 and 'Reward2' in trial_states.keys() and not np.isnan(trial_states['Reward2'][0]):
    #     outcome = 'Reward'
    # elif reps == 3 and 'Reward3' in trial_states.keys() and not np.isnan(trial_states['Reward3'][0]):        
    #     outcome = 'Reward'
    # elif 'Reward' in trial_states.keys() and not np.isnan(trial_states['Reward'][0]):
    #     outcome = 'Reward'
    # else:
    #     outcome = 'Other'        
    # elif 'Reward1' in trial_states.keys() and not np.isnan(trial_states['Reward1'][0]):
    #     outcome = 'Reward' 
    # elif 'Reward2' in trial_states.keys() and not np.isnan(trial_states['Reward2'][0]):
    #     outcome = 'Reward'  
    # elif 'Reward3' in trial_states.keys() and not np.isnan(trial_states['Reward3'][0]):
    #     outcome = 'Reward'          
    # else:
    #     outcome = 'Other'
    # keep below code for compatibilty with pre-version 8
    if 'Punish' in trial_states.keys() and not np.isnan(trial_states['Punish'][0]):
        outcome = 'Punish'
    elif reps == 1 and 'Reward1' in trial_states.keys() and not np.isnan(trial_states['Reward1'][0]):
        outcome = 'Reward'
    elif reps == 2 and 'Reward2' in trial_states.keys() and not np.isnan(trial_states['Reward2'][0]):
        outcome = 'Reward'
    elif reps == 3 and 'Reward3' in trial_states.keys() and not np.isnan(trial_states['Reward3'][0]):        
        outcome = 'Reward'
    elif 'Reward' in trial_states.keys() and not np.isnan(trial_states['Reward'][0]):
        outcome = 'Reward'
    else:
        outcome = 'Other' 
        

        
    return outcome        


# get choice from outcomes and trial types
def compute_choice():
    0