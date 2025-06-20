import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import time
import tkinter as tk
from tkinter import filedialog
from scipy.signal import savgol_filter


# read .mat to dict
def load_mat(fname):
    def _check_keys(d):
        for key in d:
            if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        elem_list = []
        
        if ndarray.ndim == 0:
            return ndarray
        
        for sub_elem in ndarray:
            if isinstance(sub_elem, sio.matlab.mio5_params.mat_struct):
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
def read_trials(session_data_path, subject, file_names):
    
    

    # file_names = os.listdir(os.path.join(session_data_path, subject))
    file_names.sort(key=lambda x: x[-19:])
    session_raw_data = []
    session_encoder_data = []
    encoder_time_max = 200
    ms_per_s = 1000
    
    time_left_VisStim1 = -0.5
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
    session_opto_tag = []
    session_probe = []
    session_probe_raw = []
    session_press_reps = []
    session_press_window = []
    
    session_onset1 = []
    session_onset2 = []
    session_velocity1 = []
    session_velocity2 = []
    session_peak1 = []
    session_peak2 = []
    session_amp1 = []
    session_amp2 = []
        
    
    session_target_thresh = []
    VisStim2Enable = 1
    session_InterruptedVisStimTrials = []
    
    # session modes and delays
    session_isSelfTimedMode = []
    session_isChemo = []
    session_isShortDelay = []
    
    session_press_delay = []
    
    session_press_delay_avg = []
    session_press_delay_short_avg = []
    session_press_delay_long_avg = []
    
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
    session_comp_delay = []
    LR12_start = 0
    
    # number of trials in each session
    session_num_total_trial = []
    session_num_effective_trial = []
    session_ids = []
    id_curr = 0

    
    for f in range(len(file_names)):
        # the first session with LR_12
        fname = file_names[f]
        id_curr = id_curr + 1
        print('Processing session data: ' , fname[-11-8:-11])
        if LR12_start==0 and fname[14:16]=='12':
            LR12_start = f
        # one session data
        raw_data = load_mat(os.path.join(session_data_path, subject, fname))
        session_raw_data.append(raw_data)
        
        # number of trials
        nTrials = raw_data['nTrials']
        print('Number of trials = ' , nTrials)
        
        # excluding the exluded data
        Exluded = raw_data['Excluded']
        if Exluded> 0:
            print('Number of excluded trials from the end = ' , Exluded)
          
        # assisted trials label
        Assisted = raw_data['Assisted'][:nTrials-Exluded]
        if Assisted.count(1)> 0:
            print('Number of assisted trials = ' , Assisted.count(1))
        
        # warm up trials
        Warmup = raw_data['IsWarmupTrial'][:nTrials-Exluded]
        if Warmup.count(1)> 0:
            print('Number of warmup trials = ' , Warmup.count(1))
        
        
        session_num_total_trial.append(nTrials)
        session_ids.append(id_curr)
        session_num_effective_trial.append(nTrials - Exluded - Assisted.count(1) - Warmup.count(1))
        
        
        if 'OptoTag' in raw_data.keys():
            opto_tag = raw_data['OptoTag'][:nTrials-Exluded]
        else:
            opto_tag = np.zeros(nTrials-Exluded)
        
        if 'ProbeTrial' in raw_data.keys():
            probe = raw_data['ProbeTrial'][:nTrials-Exluded]
            trial_effective_probe = np.zeros(nTrials-Exluded)
        else:
            probe = np.zeros(nTrials-Exluded)
            trial_effective_probe = np.zeros(nTrials-Exluded)
        # trial target
        TrialTypes = raw_data['TrialTypes'][:nTrials-Exluded]
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
        trial_reps = 0
        trial_InterruptedVisStimTrials = []
        
        # mode and delay vars
        trial_isSelfTimedMode = []
        trial_isShortDelay = []
        trial_isChemo = 0
        if raw_data['TrialSettings'][0]['GUI']['ChemogeneticSession'] == 1:
            trial_isChemo = 1
        
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
        trial_comp_delay = []
        
        #defining an array to find which trial is short, long, reward
        short_trials_vis1 = [] 
        long_trials_vis1 = []
        short_trials_vis2 = []
        long_trials_vis2 = []
        short_trial_rew = []
        long_trial_rew = []
        
        trial_onset1 = []
        trial_onset2 = []
        trial_velocity1 = []
        trial_velocity2 = []
        trial_peak1 = []
        trial_peak2 = []
        trial_amp1 = []
        trial_amp2 = []
       
        for i in range(nTrials-Exluded):
        
            # handle key error if only one trial in a session
            if nTrials-Exluded > 1:
                trial_states = raw_data['RawEvents']['Trial'][i]['States']
                trial_events = raw_data['RawEvents']['Trial'][i]['Events']  
                if 'Reps' in raw_data['TrialSettings'][i]['GUI']:              
                    trial_reps = raw_data['TrialSettings'][i]['GUI']['Reps']
                else:
                    trial_reps = 2
            else:
                print('Attenstion: this session will be excluded')
                continue
                
            
            # trial GUI params
            trial_GUI_Params = raw_data['TrialSettings'][i]['GUI']
            
       
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
            
            if isSelfTimedMode == 1:
                trial_type = raw_data['TrialTypes'][i]
                if trial_type == 1:
                    isShortDelay = 1
                                      
            trial_isShortDelay.append(isShortDelay)
                
            if isSelfTimedMode:
                if isShortDelay:
                    if 'PrePress2Delay_s' in trial_GUI_Params:
                        press_delay = trial_GUI_Params['PrePress2Delay_s']
                        trial_press_delay_short.append(press_delay)
                    elif 'PressVisDelayShort_s' in trial_GUI_Params:
                        press_delay = trial_GUI_Params['PressVisDelayShort_s']
                        trial_press_delay_short.append(press_delay)
                    elif ('PreVis2DelayShort_s' in trial_GUI_Params) and ('PrePress2DelayShort_s' in trial_GUI_Params):
                        press_delay = trial_GUI_Params['PreVis2DelayShort_s']
                        trial_press_delay_short.append(press_delay)
                    elif ('PrePress2DelayShort_s' in trial_GUI_Params) and not('PreVis2DelayShort_s' in trial_GUI_Params):
                        press_delay = trial_GUI_Params['PrePress2DelayShort_s']
                        trial_press_delay_short.append(press_delay)
                else:
                    if 'PressVisDelayLong_s' in trial_GUI_Params:
                        press_delay = trial_GUI_Params['PressVisDelayLong_s']
                        trial_press_delay_long.append(press_delay)
                    elif ('PreVis2DelayLong_s' in trial_GUI_Params) and ('PrePress2DelayLong_s' in trial_GUI_Params):
                        press_delay = trial_GUI_Params['PreVis2DelayLong_s']
                        trial_press_delay_short.append(press_delay)
                    elif ('PrePress2DelayLong_s' in trial_GUI_Params) and not('PreVis2DelayLong_s' in trial_GUI_Params):
                        press_delay = trial_GUI_Params['PrePress2DelayLong_s']
                        trial_press_delay_short.append(press_delay)
            else:
                if isShortDelay:
                    if 'PressVisDelayShort_s' in trial_GUI_Params:
                        press_delay = trial_GUI_Params['PressVisDelayShort_s']
                        trial_press_delay_short.append(press_delay)
                    elif ('PreVis2DelayShort_s' in trial_GUI_Params) and ('PrePress2DelayShort_s' in trial_GUI_Params):
                        press_delay = trial_GUI_Params['PreVis2DelayShort_s']
                        trial_press_delay_short.append(press_delay)
                    elif ('PrePress2DelayShort_s' in trial_GUI_Params) and not('PreVis2DelayShort_s' in trial_GUI_Params):
                        press_delay = trial_GUI_Params['PrePress2DelayShort_s']
                        trial_press_delay_short.append(press_delay)
                else:
                    if 'PressVisDelayLong_s' in trial_GUI_Params:
                        press_delay = trial_GUI_Params['PressVisDelayLong_s']
                        trial_press_delay_long.append(press_delay)
                    elif ('PreVis2DelayLong_s' in trial_GUI_Params) and ('PrePress2DelayLong_s' in trial_GUI_Params):
                        press_delay = trial_GUI_Params['PreVis2DelayLong_s']
                        trial_press_delay_short.append(press_delay)
                    elif ('PrePress2DelayLong_s' in trial_GUI_Params) and not('PreVis2DelayLong_s' in trial_GUI_Params):
                        press_delay = trial_GUI_Params['PrePress2DelayLong_s']
                        trial_press_delay_short.append(press_delay)
                    
            trial_press_delay.append(press_delay)
                        
        
            # press reps 
            if 'Reps' in trial_GUI_Params:
                press_reps = trial_GUI_Params['Reps']
            else:
                press_reps = 2
            trial_press_reps.append(press_reps)
            
            # press window
            if 'PressWindow_s' in trial_GUI_Params:
                press_window = trial_GUI_Params['PressWindow_s']
            else:
                press_window = trial_GUI_Params['Press1Window_s']
                      
            
            # update to add extended window for warmup            
            trial_press_window.append(press_window)
        
       
            # encoder data                
            encoder_data = raw_data['EncoderData'][i]
            trial_encoder_data.append(encoder_data)
            encoder_data_aligned = {}
            
            
            
            # interpolate encoder data to align times for averaging
            # maybe only use one array of linspace for efficiency
            # NOTICE: In the previous recordings we dont have the GUI param WarmupThreshold
            if 'IsWarmupTrial' in raw_data:
                if raw_data['IsWarmupTrial'][i] == 1:
                    trial_target_thresh.append(trial_GUI_Params['WarmupThreshold'])
                else:
                    trial_target_thresh.append(trial_GUI_Params['Threshold'])
            else:
                trial_target_thresh.append(trial_GUI_Params['Threshold'])
                        
            
            if encoder_data['nPositions']:                
                times = encoder_data['Times']
                positions = encoder_data['Positions']
                
            else:
                times = [0.0, trial_states['ITI'][1]]
                positions = [0.0, 0.0]
            
            
            # outcome
            outcome = states_labeling(trial_states, trial_reps)
            if Assisted[i] == 1:
                outcome = 'Assisted'
            if Warmup[i] == 1:
                outcome = 'Warmup'
                
            ################## late press
            
            latetime = 0
            if outcome == 'DidNotPress1':
                late_index = np.where(times > trial_states['DidNotPress1'][1])[0][0]
                late_index_stop = np.where(times > times[late_index] + 1.5)[0]
                if len(late_index_stop) > 0:
                    late_index_stop = late_index_stop[0]
                else:
                    late_index_stop = -1
                if len(positions[late_index:late_index_stop]) > 0 :
                    if max(positions[late_index:late_index_stop]) > trial_target_thresh[-1]:
                        outcome = 'LatePress1'
                        pos = positions[late_index:late_index_stop]
                        latetime = times[np.where(np.array(pos)> trial_target_thresh[-1])[0][0]+late_index]
            elif outcome == 'DidNotPress2':
                late_index = np.where(times > trial_states['DidNotPress2'][1])[0][0]
                late_index_stop = np.where(times > times[late_index] + 1.5)[0]
                if len(late_index_stop) > 0:
                    late_index_stop = late_index_stop[0]
                else:
                    late_index_stop = -1
                if len(positions[late_index:late_index_stop]) > 0 :
                    if max(positions[late_index:late_index_stop]) > trial_target_thresh[-1]:
                        outcome = 'LatePress2'
                        pos = positions[late_index:late_index_stop]
                        latetime = times[np.where(np.array(pos)> trial_target_thresh[-1])[0][0]+late_index]
            trial_outcomes.append(outcome)
            ###########################
            
            Delay = delay_computer(trial_states, outcome , latetime)
            trial_comp_delay.append(Delay)
            ##########################
            
            #################### effective probe trial
            if (probe[i] == 1) and (outcome in ['Reward' , 'DidNotPress2' , 'LatePress2']):
                trial_effective_probe[i] = 1
            
            ########################## push properties start
            # time_new = session_encoder_times_aligned
            # positions_new = np.interp(session_encoder_times_aligned, times, positions)
            # positons_filtered = savgol_filter(positions_new, window_length=40, polyorder=3)
            # velocity = np.diff(positons_filtered, prepend=positons_filtered[0])
            # velocity_filtered = savgol_filter(velocity, window_length=40, polyorder=1)
            
            # if outcome == 'LatePress1':
            #     trial_onset2.append(np.nan)
            #     trial_peak2.append(np.nan)
            #     trial_amp2.append(np.nan)
            #     trial_velocity2.append(np.nan)
                
            #     press_ind = np.where(time_new > latetime)[0][0]
            #     trial_amp1.append(np.max(positons_filtered[press_ind:press_ind+400]))
            #     if len(np.where(velocity_filtered[press_ind-100:press_ind]>0)[0]) > 0 :
            #         v_ind = np.where(velocity_filtered[press_ind-100:press_ind]>0)[0][0]
            #     else:
            #         v_ind = 0
            #     trial_onset1.append(time_new[v_ind+press_ind-100])
            #     trial_peak1.append(time_new[np.argmax(positons_filtered[press_ind:press_ind+400])+press_ind])
            #     trial_velocity1.append(trial_peak1[-1] - trial_onset1[-1])
            # elif outcome == 'EarlyPress1':
            #     trial_onset2.append(np.nan)
            #     trial_peak2.append(np.nan)
            #     trial_amp2.append(np.nan)
            #     trial_velocity2.append(np.nan)
                
            #     press_ind = np.where(time_new > trial_states['EarlyPress1'][0])[0][0]
            #     trial_amp1.append(np.max(positons_filtered[press_ind:press_ind+400]))
            #     if len(np.where(velocity_filtered[press_ind-100:press_ind]>0)[0]) > 0 :
            #         v_ind = np.where(velocity_filtered[press_ind-100:press_ind]>0)[0][0]
            #     else:
            #         v_ind = 0
            #     trial_onset1.append(time_new[v_ind+press_ind-100])
            #     trial_peak1.append(time_new[np.argmax(positons_filtered[press_ind:press_ind+400])+press_ind])
            #     trial_velocity1.append(trial_peak1[-1] - trial_onset1[-1])
            # elif outcome == 'DidNotPress2':
            #     trial_onset2.append(np.nan)
            #     trial_peak2.append(np.nan)
            #     trial_amp2.append(np.nan)
            #     trial_velocity2.append(np.nan)
                
            #     press_ind = np.where(time_new > trial_states['Press1'][0])[0][0]
            #     trial_amp1.append(np.max(positons_filtered[press_ind:press_ind+400]))
            #     if len(np.where(velocity_filtered[press_ind-100:press_ind]>0)[0]) > 0 :
            #         v_ind = np.where(velocity_filtered[press_ind-100:press_ind]>0)[0][0]
            #     else:
            #         v_ind = 0
            #     trial_onset1.append(time_new[v_ind+press_ind-100])
            #     trial_peak1.append(time_new[np.argmax(positons_filtered[press_ind:press_ind+400])+press_ind])
            #     trial_velocity1.append(trial_peak1[-1] - trial_onset1[-1])
            # elif outcome == 'Reward':
            #     press_ind = np.where(time_new > trial_states['Press1'][0])[0][0]
            #     trial_amp1.append(np.max(positons_filtered[press_ind:press_ind+400]))
            #     if len(np.where(velocity_filtered[press_ind-100:press_ind]>0)[0]) > 0 :
            #         v_ind = np.where(velocity_filtered[press_ind-100:press_ind]>0)[0][0]
            #     else:
            #         v_ind = 0
            #     trial_onset1.append(time_new[v_ind+press_ind-100])
            #     trial_peak1.append(time_new[np.argmax(positons_filtered[press_ind:press_ind+400])+press_ind])
            #     trial_velocity1.append(trial_peak1[-1] - trial_onset1[-1])
                
            #     if len(np.where(time_new > trial_states['Press2'][0])[0]) > 0:
            #         press_ind = np.where(time_new > trial_states['Press2'][0])[0][0]
            #     else:
            #         #print(time_new)
            #         #print( time_new[-1])
            #         press_ind = int(time_new[-1])
            #     trial_amp2.append(np.max(positons_filtered[press_ind:press_ind+400]))
            #     if len(np.where(velocity_filtered[press_ind-100:press_ind]>0)[0]) > 0 :
            #         v_ind = np.where(velocity_filtered[press_ind-100:press_ind]>0)[0][0]
            #     else:
            #         v_ind = 0
            #     trial_onset2.append(time_new[v_ind+press_ind-100])
            #     trial_peak2.append(time_new[np.argmax(positons_filtered[press_ind:press_ind+400])+press_ind])
            #     trial_velocity2.append(trial_peak1[-1] - trial_onset1[-1])
            # elif outcome == 'LatePress2':
            #     press_ind = np.where(time_new > trial_states['Press1'][0])[0][0]
            #     trial_amp1.append(np.max(positons_filtered[press_ind:press_ind+400]))
            #     if len(np.where(velocity_filtered[press_ind-100:press_ind]>0)[0]) > 0 :
            #         v_ind = np.where(velocity_filtered[press_ind-100:press_ind]>0)[0][0]
            #     else:
            #         v_ind = 0
            #     trial_onset1.append(time_new[v_ind+press_ind-100])
            #     trial_peak1.append(time_new[np.argmax(positons_filtered[press_ind:press_ind+400])+press_ind])
            #     trial_velocity1.append(trial_peak1[-1] - trial_onset1[-1])
                
            #     press_ind = np.where(time_new > latetime)[0][0]
            #     trial_amp2.append(np.max(positons_filtered[press_ind:press_ind+400]))
            #     if len(np.where(velocity_filtered[press_ind-100:press_ind]>0)[0]) > 0 :
            #         v_ind = np.where(velocity_filtered[press_ind-100:press_ind]>0)[0][0]
            #     else:
            #         v_ind = 0
            #     trial_onset2.append(time_new[v_ind+press_ind-100])
            #     trial_peak2.append(time_new[np.argmax(positons_filtered[press_ind:press_ind+400])+press_ind])
            #     trial_velocity2.append(trial_peak1[-1] - trial_onset1[-1])
            # elif outcome == 'EarlyPress2':
            #     press_ind = np.where(time_new > trial_states['Press1'][0])[0][0]
            #     trial_amp1.append(np.max(positons_filtered[press_ind:press_ind+400]))
            #     if len(np.where(velocity_filtered[press_ind-100:press_ind]>0)[0]) > 0 :
            #         v_ind = np.where(velocity_filtered[press_ind-100:press_ind]>0)[0][0]
            #     else:
            #         v_ind = 0
            #     trial_onset1.append(time_new[v_ind+press_ind-100])
            #     trial_peak1.append(time_new[np.argmax(positons_filtered[press_ind:press_ind+400])+press_ind])
            #     trial_velocity1.append(trial_peak1[-1] - trial_onset1[-1])
                
            #     press_ind = np.where(time_new > trial_states['EarlyPress2'][0])[0][0]
            #     trial_amp2.append(np.max(positons_filtered[press_ind:press_ind+400]))
            #     if len(np.where(velocity_filtered[press_ind-100:press_ind]>0)[0]) > 0 :
            #         v_ind = np.where(velocity_filtered[press_ind-100:press_ind]>0)[0][0]
            #     else:
            #         v_ind = 0
            #     trial_onset2.append(time_new[v_ind+press_ind-100])
            #     trial_peak2.append(time_new[np.argmax(positons_filtered[press_ind:press_ind+400])+press_ind])
            #     trial_velocity2.append(trial_peak1[-1] - trial_onset1[-1])
            # else:
            #     trial_onset2.append(np.nan)
            #     trial_peak2.append(np.nan)
            #     trial_amp2.append(np.nan)
            #     trial_velocity2.append(np.nan)
            #     trial_onset1.append(np.nan)
            #     trial_peak1.append(np.nan)
            #     trial_amp1.append(np.nan)
            #     trial_velocity1.append(np.nan)
                
            
            
            
            #########################push properties end
            
            # all trials encoder positions
            encoder_data_aligned = np.interp(session_encoder_times_aligned, times, positions)
            
            trial_encoder_positions_aligned.append(encoder_data_aligned)
            
            # find times and pos aligned to vis stim 1
            if 'VisualStimulus1' in trial_states.keys() and not np.isnan(trial_states['VisualStimulus1'][0]):
                VisStim1Start = trial_states['VisualStimulus1'][0]
            elif 'VisStimInterrupt' in trial_states.keys() and not np.isnan(trial_states['VisStimInterrupt'][0]):                
                trial_InterruptedVisStimTrials.append(i)
                VisStim1Start = 0
            else:
                # to exclude this trial
                # when press happens before visual stimulus 1 
                VisStim1Start = 0
                #print('Should be either vis 1 or vis interrupt, check')
            
            if VisStim1Start > 12:
                print('Attention: this trial will be exluded')
                continue # vis detect missed stim, go to next trial
            
            
            vis_diff = np.abs(VisStim1Start - session_encoder_times_aligned)
            min_vis_diff = np.min(np.abs(VisStim1Start - session_encoder_times_aligned))
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
                        short_trials_vis1.append(i)
                    else:                    
                        trial_encoder_positions_aligned_vis1_rew_long.append(trial_encoder_positions_aligned_VisStim1)
                        long_trials_vis1.append(i)            
            
            
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
                
                if len(trial_encoder_positions_aligned_VisStim2) == 10000:
                    trial_encoder_positions_aligned_vis2.append(trial_encoder_positions_aligned_VisStim2)
                
                if not isSelfTimedMode:
                    if isShortDelay:
                        if len(trial_encoder_positions_aligned_VisStim2) == 10000:
                            trial_encoder_positions_aligned_vis2_short.append(trial_encoder_positions_aligned_VisStim2)
                    else:
                        if len(trial_encoder_positions_aligned_VisStim2) == 10000:
                            trial_encoder_positions_aligned_vis2_long.append(trial_encoder_positions_aligned_VisStim2)
                
                if outcome == 'Reward':
                    if len(trial_encoder_positions_aligned_VisStim2) == 10000:
                        trial_encoder_positions_aligned_vis2_rew.append(trial_encoder_positions_aligned_VisStim2)
                    
                    if not isSelfTimedMode:
                        if isShortDelay:
                            if len(trial_encoder_positions_aligned_VisStim2) == 10000:
                                trial_encoder_positions_aligned_vis2_rew_short.append(trial_encoder_positions_aligned_VisStim2)
                            short_trials_vis2.append(i)
                        else:
                            if len(trial_encoder_positions_aligned_VisStim2) == 10000:
                                trial_encoder_positions_aligned_vis2_rew_long.append(trial_encoder_positions_aligned_VisStim2)
                            long_trials_vis2.append(i)
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
                if len(trial_encoder_positions_aligned_Reward) == 10000:
                    trial_encoder_positions_aligned_rew.append(trial_encoder_positions_aligned_Reward)
                
                if not isSelfTimedMode:
                    if isShortDelay:
                        if len(trial_encoder_positions_aligned_Reward) == 10000:
                            trial_encoder_positions_aligned_rew_short.append(trial_encoder_positions_aligned_Reward)
                        short_trial_rew.append(i)
                    else:
                        if len(trial_encoder_positions_aligned_Reward) == 10000:
                            trial_encoder_positions_aligned_rew_long.append(trial_encoder_positions_aligned_Reward)
                        long_trial_rew.append(i)
                
                
                # index of rewarded trials
                trial_num_rewarded.append(i)
            
                
        # encoder trajectory average across session for rewarded trials, vis stim 1 aligned        
        try:
            pos_vis1 = np.sum(trial_encoder_positions_aligned_vis1_rew[0:], axis=0)
        except:
            print(fname)
            time.sleep(15)
           
        # vis 1 session average - rewarded
        pos_vis1 = np.sum(trial_encoder_positions_aligned_vis1_rew[0:], axis=0)        
        sess_enc_avg_vis1 = pos_vis1/len(trial_num_rewarded)
        
        sess_enc_avg_vis1_short_rew = 0
        sess_enc_avg_vis1_long_rew = 0
        if not isSelfTimedMode:
            # if isShortDelay:
                # vis 1 session average - short - rewarded
                pos_vis1_short_rew = np.sum(trial_encoder_positions_aligned_vis1_rew_short[0:], axis=0)        
                sess_enc_avg_vis1_short_rew = pos_vis1_short_rew/len(trial_encoder_positions_aligned_vis1_rew_short[0:])
            # else:        
                # vis 1 session average - long - rewarded
                #print('len(trial_encoder_positions_aligned_vis1_rew_long)', len(trial_encoder_positions_aligned_vis1_rew_long))
                pos_vis1_long_rew = np.sum(trial_encoder_positions_aligned_vis1_rew_long[0:], axis=0)        
                sess_enc_avg_vis1_long_rew = pos_vis1_long_rew/len(trial_encoder_positions_aligned_vis1_rew_long[0:])
        
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
        session_opto_tag.append(opto_tag)
        session_probe_raw.append(probe)
        session_probe.append(trial_effective_probe)
        
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
        session_comp_delay.append(trial_comp_delay)
        
        session_press_delay_avg.append(sess_press_delay_avg)
        session_press_delay_short_avg.append(sess_press_delay_short_avg)
        session_press_delay_long_avg.append(sess_press_delay_long_avg)        
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
        session_isChemo.append(trial_isChemo)
        
        
        session_onset2.append(trial_onset2)
        session_peak2.append(trial_peak2)
        session_amp2.append(trial_amp2)
        session_velocity2.append(trial_velocity2)
        session_onset1.append(trial_onset1)
        session_peak1.append(trial_peak1)
        session_amp1.append(trial_amp1)
        session_velocity1.append(trial_velocity1)
        
        session_press_reps.append(trial_press_reps)
        session_press_window.append(trial_press_window)
    
    # merge all session data
    session_numChemo = session_isChemo.count(1)
    data = {
        'num_total_trial' : session_num_total_trial,
        'IDs' : session_ids,
        'num_effective_trial' : session_num_effective_trial, 
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
        'session_opto_tag' : session_opto_tag,
        'session_probe_raw' : session_probe_raw,
        'session_probe' : session_probe,
        'session_comp_delay': session_comp_delay,
        'com' : session_com,
        'post_lick' : session_post_lick,
        'isi' : session_isi,
        'avsync' : session_avsync,
        'isSelfTimedMode' : session_isSelfTimedMode,
        'isShortDelay' : session_isShortDelay,
        'chemo' : session_isChemo, 
        'chemo_sess_num' : session_numChemo,
        'press_reps' : session_press_reps,
        'press_window' : session_press_window,
        'trial_encoder_pos_vis1_short' : trial_encoder_positions_aligned_vis1_rew_short,
        'trial_encoder_pos_vis1_long' : trial_encoder_positions_aligned_vis1_rew_long,
        'trial_encoder_pos_vis2_short' : trial_encoder_positions_aligned_vis2_rew_short,
        'trial_encoder_pos_vis2_long' : trial_encoder_positions_aligned_vis2_rew_long,
        'trial_encoder_pos_rew_short' : trial_encoder_positions_aligned_rew_short,
        'trial_encoder_pos_rew_long' : trial_encoder_positions_aligned_rew_long,
        'short_trials_vis1' : short_trials_vis1,
        'long_trials_vis1' :long_trials_vis1,
        'short_trials_vis2' : short_trials_vis2,
        'long_trials_vis2' : long_trials_vis2,
        'short_trial_rew' : short_trial_rew,
        'long_trial_rew' : long_trial_rew,
        'onset1' : session_onset1,
        'onset2' : session_onset2,
        'peak1' : session_peak1,
        'velocity1' : session_velocity1,
        'peak2' : session_peak2,
        'velocity2' : session_velocity2,
        'amp1' : session_amp1,
        'amp2' : session_amp2
    }
    return data


# labeling every trials for a subject
def states_labeling(trial_states, reps):
    if ('Punish' in trial_states.keys() and not np.isnan(trial_states['Punish'][0])) or ('EarlyPressPunish' in trial_states.keys() and not np.isnan(trial_states['EarlyPressPunish'][0])) or ('EarlyPress1Punish' in trial_states.keys() and not np.isnan(trial_states['EarlyPress1Punish'][0])) or ('EarlyPress2Punish' in trial_states.keys() and not np.isnan(trial_states['EarlyPress2Punish'][0]))or ('LatePress1' in trial_states.keys() and not np.isnan(trial_states['LatePress1'][0]))or ('LatePress2' in trial_states.keys() and not np.isnan(trial_states['LatePress2'][0])):
        if 'DidNotPress1' in trial_states.keys() and not np.isnan(trial_states['DidNotPress1'][0]):
            outcome = 'DidNotPress1'
        elif 'DidNotPress2' in trial_states.keys() and not np.isnan(trial_states['DidNotPress2'][0]):
            outcome = 'DidNotPress2'
        elif 'DidNotPress3' in trial_states.keys() and not np.isnan(trial_states['DidNotPress3'][0]):
            outcome = 'DidNotPress3'
        elif 'EarlyPress' in trial_states.keys() and not np.isnan(trial_states['EarlyPress'][0]):
            outcome = 'EarlyPress'
        elif 'EarlyPress1' in trial_states.keys() and not np.isnan(trial_states['EarlyPress1'][0]):
            outcome = 'EarlyPress1'
        elif 'EarlyPress2' in trial_states.keys() and not np.isnan(trial_states['EarlyPress2'][0]):
            outcome = 'EarlyPress2'
        elif 'LatePress1' in trial_states.keys() and not np.isnan(trial_states['LatePress1'][0]):
            outcome = 'LatePress1'
        elif 'LatePress2' in trial_states.keys() and not np.isnan(trial_states['LatePress2'][0]):
            outcome = 'LatePress2'
        else:
            outcome = 'Punish'
    elif reps == 1 and 'Reward1' in trial_states.keys() and not np.isnan(trial_states['Reward1'][0]):
        outcome = 'Reward'
    elif reps == 2 and 'Reward2' in trial_states.keys() and not np.isnan(trial_states['Reward2'][0]):
        outcome = 'Reward'
    elif reps == 3 and 'Reward3' in trial_states.keys() and not np.isnan(trial_states['Reward3'][0]):        
        outcome = 'Reward'
    elif 'Reward' in trial_states.keys() and not np.isnan(trial_states['Reward'][0]):
        outcome = 'Reward'
    elif 'VisStimInterruptDetect1' in trial_states.keys() and not np.isnan(trial_states['VisStimInterruptDetect1'][0]):
        outcome = 'VisStimInterruptDetect1'
    elif 'VisStimInterruptDetect2' in trial_states.keys() and not np.isnan(trial_states['VisStimInterruptDetect2'][0]):
        outcome = 'VisStimInterruptDetect2'
    elif 'VisStimInterruptGray1' in trial_states.keys() and not np.isnan(trial_states['VisStimInterruptGray1'][0]):
        outcome = 'VisStimInterruptGray1'
    elif 'VisStimInterruptGray2' in trial_states.keys() and not np.isnan(trial_states['VisStimInterruptGray2'][0]):
        outcome = 'VisStimInterruptGray2'
    else:
        outcome = 'Other' # VisInterrupt
    
    return outcome        


# get choice from outcomes and trial types
# def compute_choice():
def delay_computer(trial_states, outcome , latetime):
    if outcome == 'EarlyPress1':
        delay = np.nan
    elif outcome == 'EarlyPress':
        delay = np.nan
    elif outcome == 'EarlyPress2':
        delay = trial_states['EarlyPress2'][0] - trial_states['LeverRetract1'][1]
    elif outcome == 'DidNotPess1':
        delay = np.nan
    elif outcome == 'LatePress1':
        delay = np.nan
    elif outcome == 'DidNotPess2':
        delay = np.nan
    elif outcome == 'LatePress2':
        delay = latetime - trial_states['LeverRetract1'][1]
    elif outcome == 'Reward':
        if not np.isnan(trial_states['Press2'][0]):
            delay = trial_states['Press2'][0] - trial_states['LeverRetract1'][1]
        else:
            delay = trial_states['PreRewardDelay'][0] - trial_states['LeverRetract1'][1]
    else:
        delay = np.nan
    return delay        