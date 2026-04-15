# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 07:26:55 2025

@author: saminnaji3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from datetime import date
from matplotlib.lines import Line2D
import re
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import normalize
import scipy.io as sio
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

states = ['trial_vis1' , 'trial_push1' , 'trial_retract1_init' , 'trial_retract1', 'trial_vis2'
          , 'trial_wait2', 'trial_push2', 'trial_reward', 'trial_punish' ,'trial_retract2']

def IsSelfTimed(session_data):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    chemo_labels = session_data['chemo']
    isSelfTime = session_data['isSelfTimedMode']
    
    VG = []
    ST = []
    for i in range(0 , len(isSelfTime)):
        if isSelfTime[i][5] == 1 or isSelfTime[i][5] == np.nan:
            ST.append(i)
        else:
            VG.append(i)
    return ST , VG

def read_bpod_mat_data(raw_data, Exluded):
    def block_start_end(raw, Exluded):
        trial_type = raw['TrialTypes']
        warmup_num = len(np.where(np.array(raw['IsWarmupTrial']) == 1))
        #print(warmup_num)
        warmup_num = 0
        diff = np.diff(trial_type[:len(trial_type)-Exluded])
        
        short_start = np.where(diff == 255)[0] + 1
        long_start = np.where(diff == 1)[0] + 1
        short_end = np.where(diff == 1)[0] + 1
        long_end = np.where(diff == 255)[0] + 1
        
        if trial_type[0] == 1:
            short_start = np.insert(short_start,0,warmup_num)
        else:
            long_start = np.insert(long_start,0,warmup_num)
            
            
        if len(short_start) > len(short_end):
            short_end = np.insert(short_end,0,len(trial_type))
            
        elif len(long_start) > len(long_end) and len(long_end) > 0:
            long_end = np.insert(long_end,-1,len(trial_type))
        elif len(long_start) > len(long_end):
            long_end = np.insert(long_end,0,len(trial_type))
            
            
        return short_start , short_end , long_start , long_end
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
        if ndarray.ndim > 0:
            for sub_elem in ndarray:
                if isinstance(sub_elem, sio.matlab.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_tolist(sub_elem))
                else:
                    elem_list.append(sub_elem)
        return elem_list

    def get_push_onset(js_pos, js_time, start_time, end_time):
        def find_half_peak_point_before(velocity, peak_idx):
            half_peak_value = 2.5
            for i in range(peak_idx - 1, -1, -1):
                if velocity[i] <= half_peak_value:
                    return i
            return 0

        def find_onethird_peak_point_before(velocity, peak_idx):
            peak_value = velocity[peak_idx]
            onethird_peak_value = peak_value * 0.4
            for i in range(peak_idx - 1, -1, -1):
                if velocity[i] <= onethird_peak_value:
                    return i
            return 0

        def velocity_onset(js_pos, start, end):
            start = max(0,start)
            end = min(len(js_pos), end)
            peaks,_ = find_peaks(js_pos[start:end],distance=65, height=5)

            onset4velocity = []
            if len(peaks) >= 1:
                peaks = peaks + start
            if len(peaks) == 0:
                peaks = end
                onset4velocity.append(find_onethird_peak_point_before(js_pos,peaks))
                return onset4velocity
  
            if len(peaks) >= 1:
                peaks = np.hstack((peaks,end))
                for i in range(0, len(peaks)):
                    onset4velocity.append(find_onethird_peak_point_before(js_pos, peaks[i]))
                return onset4velocity
              
        
        #print(interpolator)
        new_time = np.arange(0, 60000, 1)
        # interpolator = interp1d(js_time, js_pos, bounds_error=False)
        # new_pos = interpolator(new_time)
        new_pos = np.interp(new_time , js_time, js_pos)
        idx_start = np.argmin(np.abs(new_time - start_time))
        idx_end = np.argmin(np.abs(new_time - end_time))
        # print(new_pos)
        # print(interpolator)
        new_pos = savgol_filter(new_pos, window_length=40, polyorder=3)
        vel = np.gradient(new_pos, new_time)
        vel = savgol_filter(vel, window_length=40, polyorder=1)
        onset4velocity = velocity_onset(vel, idx_start, idx_end)
        if onset4velocity[0] == 0:
            push = np.array([np.nan])
        else:
            push = np.array([new_time[onset4velocity[0]]])
        return push
    
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
    
    def main():
        raw = raw_data
        Exluded = raw['Excluded']
        short_start , short_end , long_start , long_end = block_start_end(raw, Exluded)
        block_start = np.array(sorted(list(short_start)+list(long_start)))
        print(block_start)
        block_end = np.array(sorted(list(short_end)+list(long_end)))
        #print(block_start)
        trial_block = []
        trial_pos_in_block = []
        trial_block_len = []
        trial_vis1 = []
        trial_push1 = []
        trial_retract1 = []
        trial_vis2 = []
        trial_wait2 = []
        trial_push2 = []
        trial_retract2 = []
        trial_reward = []
        trial_punish = []
        trial_no1stpush = []
        trial_no2ndpush = []
        trial_early2ndpush = []
        trial_late2ndpush = []
        trial_iti = []
        trial_lick = []
        trial_delay = []
        trial_ST = []
        trial_js_pos = []
        trial_js_time = []
        trial_outcome = []
        trial_retract1_init = []
        trial_probe = []
        trial_opto = []
        if 'OptoTag' in raw.keys():
            trial_opto = raw['OptoTag'][:raw['nTrials']-Exluded]
        else:
            trial_opto = np.zeros(raw['nTrials']-Exluded)
        num_trials = 0
        for i in range(raw['nTrials']-Exluded):
            num_trials = num_trials + 1
            trial_states = raw['RawEvents']['Trial'][i]['States']
            trial_events = raw['RawEvents']['Trial'][i]['Events']
            trial_probe.append(raw['ProbeTrial'][i])
            #print(i)
            temp1 = np.where(block_start > i)[0]
            if len(temp1) == 0:
                temp = len(block_start)
            else:
                temp = np.where(block_start > i)[0][0]
            trial_block.append(temp-1)
            #print(i, trial_block[-1], block_start[trial_block[-1]])
            trial_pos_in_block.append(i-block_start[trial_block[-1]])
            trial_block_len.append(block_end[trial_block[-1]]-block_start[trial_block[-1]]+1)
            trial_outcome.append(states_labeling(trial_states, 2))
            #print(trial_pos_in_block)
            
            # push1 onset.
            start = 1000*np.array(trial_states['VisDetect1'][0]).reshape(-1)-500
            end = 1000*np.array(trial_states['LeverRetract1'][0]).reshape(-1)-100
            push1 = get_push_onset(
                np.array(raw['EncoderData'][i]['Positions']).reshape(-1),
                1000*np.array(raw['EncoderData'][i]['Times']).reshape(-1),
                start, end)
            
            # push2 onset.
            start = 1000*np.array(trial_states['WaitForPress2'][0]).reshape(-1)
            if np.isnan(start):
                if not np.isnan(push1):
                    start = 1000*np.array(trial_states['LeverRetract1'][1]).reshape(-1)
            if ('RotaryEncoder1_1' in trial_events.keys() and np.size(trial_events['RotaryEncoder1_1'])>1):
                end = 1000*np.array(trial_events['RotaryEncoder1_1'][-1]).reshape(-1)
                a = 1
                #end = 1000*np.array(trial_states['LeverRetractFinal'][1]).reshape(-1)
            else:
                end = 1000*np.array(trial_states['LeverRetractFinal'][1]).reshape(-1)
                a = 2
            push2 = get_push_onset(
                np.array(raw['EncoderData'][i]['Positions']).reshape(-1),
                1000*np.array(raw['EncoderData'][i]['Times']).reshape(-1),
                start, end)
            
            if np.isnan(push1):
                push2 = np.array([np.nan])
            
            trial_push1.append(push1)
            trial_push2.append(push2)
            if push2 < push1:
                print(start, end, a)

            trial_vis1.append(1000*np.array(trial_states['VisualStimulus1']).reshape(-1))
            # 1st retract.
            trial_retract1.append(1000*np.array(trial_states['LeverRetract1'][1]).reshape(-1))
            # 1st retract.
            trial_retract1_init.append(1000*np.array(trial_states['LeverRetract1'][0]).reshape(-1))
            # 2nd stim.
            trial_vis2.append(1000*np.array(trial_states['VisualStimulus2']).reshape(-1))
            # wait for 2nd push.
            trial_wait2.append(1000*np.array(trial_states['WaitForPress2'][0]).reshape(-1))
            # 2nd retract.
            if ('LeverRetractFinal' in trial_states.keys()):
                trial_retract2.append(1000*np.array(trial_states['LeverRetractFinal'][0]).reshape(-1))
            else:
                trial_retract2.append(1000*np.array([np.nan, np.nan]).reshape(-1))
            # reward.
            trial_reward.append(1000*np.array(trial_states['Reward']).reshape(-1))
            #print(1000*np.array(trial_states['Reward']))
            # punish.
            if not np.isnan(trial_states['Punish'][0]):
                trial_punish.append(1000*np.array(trial_states['Punish']).reshape(-1))
            else:
                trial_punish.append(1000*np.array(trial_states['EarlyPress2Punish']).reshape(-1))
            # did not push 1.
            trial_no1stpush.append(1000*np.array(trial_states['DidNotPress1']).reshape(-1))
            # did not push 2.
            trial_no2ndpush.append(1000*np.array(trial_states['DidNotPress2']).reshape(-1))
            # early push 2.
            trial_early2ndpush.append(1000*np.array(trial_states['EarlyPress2']).reshape(-1))
            # late push2
            if 'LatePress2' in trial_states.keys():
                trial_late2ndpush.append(1000*np.array(trial_states['LatePress2']).reshape(-1))
            else:
                trial_late2ndpush.append([np.nan,np.nan])
            # licking events.
            if 'Port2In' in trial_events.keys():
                lick_all = 1000*np.array(trial_events['Port2In']).reshape(1,-1)
                lick_label = np.zeros_like(lick_all).reshape(1,-1)
                lick_label[lick_all>1000*np.array(trial_states['Reward'][0])] = 1
                trial_lick.append(np.concatenate((lick_all, lick_label), axis=0))
            else:
                trial_lick.append(np.array([[np.nan],[np.nan]]))
            # self timed / visually guide
            trial_GUI_Params = raw['TrialSettings'][i]['GUI']
            # mode: vis-guided or self timed
            isSelfTimedMode = 0    # assume vis-guided (all sessions were vi)
            if 'SelfTimedMode' in trial_GUI_Params:
                isSelfTimedMode = trial_GUI_Params['SelfTimedMode']
                
            trial_ST.append(isSelfTimedMode)
            # iti.
            trial_iti.append(1000*np.array(trial_states['ITI']).reshape(-1))
            # delay
            
            if np.min(raw['TrialTypes']) == np.max(raw['TrialTypes']):
                #print('single bock session')
                #print(np.min(raw['TrialTypes']), np.max(raw['TrialTypes']))
                #print(len(raw['TrialTypes']))
                #trial_delay.append(1000*raw['TrialSettings'][i]['GUI']['PressVisDelayLong_s'])
                ## added temprary
                if 'PrePress2DelayShort_s' in raw['TrialSettings'][i]['GUI'].keys():
                    if raw['TrialTypes'][i] == 1:
                        trial_delay.append(1000*raw['TrialSettings'][i]['GUI']['PrePress2DelayShort_s'])
                    if raw['TrialTypes'][i] == 2:
                        trial_delay.append(1000*raw['TrialSettings'][i]['GUI']['PrePress2DelayLong_s'])
                else:
                    if raw['TrialTypes'][i] == 1:
                        trial_delay.append(1000*raw['TrialSettings'][i]['GUI']['PressVisDelayShort_s'])
                    if raw['TrialTypes'][i] == 2:
                        trial_delay.append(1000*raw['TrialSettings'][i]['GUI']['PressVisDelayLong_s'])
                ## end adding
            else:
                if 'PrePress2DelayShort_s' in raw['TrialSettings'][i]['GUI'].keys():
                    if raw['TrialTypes'][i] == 1:
                        trial_delay.append(1000*raw['TrialSettings'][i]['GUI']['PrePress2DelayShort_s'])
                    if raw['TrialTypes'][i] == 2:
                        trial_delay.append(1000*raw['TrialSettings'][i]['GUI']['PrePress2DelayLong_s'])
                else:
                    if raw['TrialTypes'][i] == 1:
                        trial_delay.append(1000*raw['TrialSettings'][i]['GUI']['PressVisDelayShort_s'])
                    if raw['TrialTypes'][i] == 2:
                        trial_delay.append(1000*raw['TrialSettings'][i]['GUI']['PressVisDelayLong_s'])
            # joystick trajectory.
            js_pos = np.array(raw['EncoderData'][i]['Positions'])
            js_time = 1000*np.array(raw['EncoderData'][i]['Times'])
            if np.abs(js_pos[0]) > 0.9 or np.abs(js_time[0]) > 1e-5:
                trial_js_pos.append(np.array([0,0,0,0,0]))
                trial_js_time.append(np.array([0,1,2,3,4]))
            else:
                trial_js_pos.append(js_pos)
                trial_js_time.append(js_time)
                
                
        print('num trials: ', num_trials)
        bpod_sess_data = {

            'trial_types'        : np.array(raw['TrialTypes'][:raw['nTrials']-Exluded]),
            'trial_vis1'         : trial_vis1,
            'trial_probe'        : trial_probe,
            'trial_opto'         : trial_opto,
            'trial_push1'        : trial_push1,
            'trial_retract1'     : trial_retract1,
            'trial_retract1_init': trial_retract1_init,
            'trial_vis2'         : trial_vis2,
            'trial_wait2'        : trial_wait2,
            'trial_push2'        : trial_push2,
            'trial_retract2'     : trial_retract2,
            'trial_reward'       : trial_reward,
            'trial_punish'       : trial_punish,
            'trial_no1stpush'    : trial_no1stpush,
            'trial_no2ndpush'    : trial_no2ndpush,
            'trial_early2ndpush' : trial_early2ndpush,
            'trial_late2ndpush'  : trial_late2ndpush,
            'trial_iti'          : trial_iti,
            'trial_lick'         : trial_lick,
            'trial_delay'        : trial_delay,
            'trial_ST'           : trial_ST,
            'trial_js_pos'       : trial_js_pos,
            'trial_js_time'      : trial_js_time,
            'trial_outcome'      : trial_outcome,
            'trial_block'        : trial_block,
            'trial_block_len'    : trial_block_len,
            'trial_pos_in_block' : trial_pos_in_block,
            }

        return bpod_sess_data
    bpod_sess_data = main()
    
    return bpod_sess_data

def pad_seq(align_data, align_time):
    #print(np.min([np.nanmin(t) for t in align_time]))
    #print(np.max([np.nanmax(t) for t in align_time]))
    pad_time = np.arange(
        np.min([np.nanmin(t) for t in align_time]),
        np.max([np.nanmax(t) for t in align_time]) + 1)
    pad_data = []
    pad_time_all = []
    for data, time in zip(align_data, align_time):
        
        aligned_seq = np.full_like(pad_time, np.nan, dtype=float)
        aligned_seq_time = np.full_like(pad_time, np.nan, dtype=float)
        idx = np.searchsorted(pad_time, time)
        aligned_seq[idx] = data
        aligned_seq_time[idx] = time
        pad_data.append(aligned_seq)
        pad_time_all.append(pad_time)
    return pad_data, pad_time, pad_time_all

def selected_trials_js_pos(align_data, refrence, condition, refrence2 = np.nan, condition2 = np.nan):
    
    if np.isnan(condition2):
        selected_trial_id = np.where(refrence == condition)[0]
        #print(selected_trial_id)
        selected_align_data = align_data[selected_trial_id , :]
        selected_labels = refrence[selected_trial_id]
    else:
        selected_trial_id = np.where(np.logical_and(refrence == condition,refrence2== condition2))[0]
        selected_align_data = align_data[selected_trial_id , :]
        selected_labels = refrence[selected_trial_id]
    return selected_align_data, selected_labels

def get_trial_outcome(event_timing, trials):
    """
    takes neural_trials andthe trial string.
    returns outcome label for that trial:
        Reward = 0
        DidNotPress1 = 1
        DidNotPress2 = 2
        LatePress2 = 3
        EarlyPress2 = 4
        probe = 5
        other outcomes = -1
    """
    if event_timing['trial_probe'][trials] == 1:
        trial_outcome = 5
    elif not np.isnan(event_timing['trial_reward'][trials][0]):
        trial_outcome = 0
    elif not np.isnan(event_timing['trial_no1stpush'][trials][0]):
        trial_outcome = 1
    elif not np.isnan(event_timing['trial_no2ndpush'][trials][0])and (np.isnan(event_timing['trial_push2'][trials][0])):
        trial_outcome = 2
    elif (not np.isnan(event_timing['trial_no2ndpush'][trials][0])) and (not np.isnan(event_timing['trial_push2'][trials][0])):
        #print('late_press1')
        trial_outcome = 3
    elif (not np.isnan(event_timing['trial_late2ndpush'][trials][0])):
        #print('late_press2')
        trial_outcome = 3
    elif not np.isnan(event_timing['trial_early2ndpush'][trials][0]):
        trial_outcome = 4
    else:
        trial_outcome = -1
    return trial_outcome

def get_js_pos(event_timing, state):
    interval = 1
    js_time = event_timing['trial_js_time']
    js_pos = event_timing['trial_js_pos']
    
    
    trial_type = event_timing['trial_types']
    trial_opto = event_timing['trial_opto']
    epoch = []
    # epoch = np.array([neural_trials[t]['block_epoch']
    #           for t in neural_trials.keys()])
    trial_outcome = np.array([get_trial_outcome(event_timing, t)
              for t in range(len(trial_type))])
    #trial_outcome = trial_type
    inter_time = []
    inter_pos = []
    for (pos, time) in zip(js_pos, js_time):
        interpolator = interp1d(time, pos, bounds_error=False)
        new_time = np.arange(np.min(time), np.max(time), interval)
        new_pos = interpolator(new_time)
        inter_time.append(new_time)
        inter_pos.append(new_pos)
    if np.size(event_timing[state][0]) == 1:
        time_state = [
            event_timing[state][t] for t in range(len(event_timing[state]))]
    if np.size(event_timing[state][0]) == 2:
        time_state = [
            event_timing[state][t][0] for t in range(len(event_timing[state]))]
    print(state + ': ' + str(np.count_nonzero(~np.isnan(time_state))))
    trial_type = np.array([trial_type[i]
                   for i in range(len(inter_time))
                   if not np.isnan(time_state)[i]])
    opto = np.array([trial_opto[i]
                   for i in range(len(inter_time))
                   if not np.isnan(time_state)[i]])
    outcome = np.array([trial_outcome[i]
                   for i in range(len(inter_time))
                   if not np.isnan(time_state)[i]])

    zero_state = [np.argmin(np.abs(inter_time[i] - time_state[i]))
                  for i in range(len(inter_time))
                  if not np.isnan(time_state)[i]]
    align_data = [inter_pos[i] - inter_pos[i][0]
                  for i in range(len(inter_pos))
                  if not np.isnan(time_state)[i]]
    align_time = [inter_time[i]
                  for i in range(len(inter_time))
                  if not np.isnan(time_state)[i]]
    align_time = [align_time[i] - align_time[i][zero_state[i]]
                  for i in range(len(align_time))]
    if len(align_data) > 0:
        align_data, align_time, time_all = pad_seq(align_data, align_time)
        align_data = np.array(align_data)
    else:
        align_data = np.array([[np.nan]])
        align_time = np.array([np.nan])
        trial_type = np.array([np.nan])
        epoch = np.array([np.nan])
        outcome = np.array([np.nan])
        opto = np.array([np.nan])
    return align_data, align_time, time_all, trial_type, epoch, outcome, opto


def lay_out(axs, x_label = '', y_label = '', title = '', x_lim = [], y_lim = [], legend = 0):
    if not len(x_label) == 0:
        axs.set_xlabel(x_label)
    if not len(y_label) == 0:
        axs.set_ylabel(y_label)
    if not len(title) == 0:
        axs.set_title(title)
    if not len(x_lim) == 0:
        axs.set_xlim(x_lim)
    if not len(y_lim) == 0:
        axs.set_ylim(y_lim)
    if not legend == 0:
        axs.legend()
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
def plot_aligned_data(axs, align_data, align_time, state, color_tag = 'b'):
    
    trajectory_mean = np.nanmean(align_data , axis = 0)
    #print(trajectory_mean.shape, align_data.shape)
    #print(trajectory_mean)
    trajectory_sem = np.nanstd(align_data , axis = 0)/np.sqrt(len(align_data))
    axs.fill_between(align_time/1000 ,trajectory_mean-trajectory_sem , trajectory_mean+trajectory_sem, color = color_tag , alpha = 0.2)
    axs.plot(align_time/1000 , trajectory_mean , color = color_tag, linewidth = 0.1, label = 'n = ' + str(len(align_data)))
    
    if state == 'trial_vis1' or state == 'trial_vis2':
        axs.axvspan(0, 0.1, alpha=0.2, color='gray')
    else:
        axs.axvline(x = 0, color = 'gray', linestyle='--', linewidth = 1)
    if state == 'trial_retract1':
        axs.axvline(x = 0.3, color = 'y', linestyle='--', linewidth = 1, alpha = 0.3)
        axs.axvline(x = 0.7, color = 'b', linestyle='--', linewidth = 1, alpha = 0.3)
        
def run(axs1, axs2, session_data, refrence = 'opto', st = 0):
    dates = session_data['dates']
    
    num_session = len(dates)
    ST , VG = IsSelfTimed(session_data)
    if st == 1:
        plotting_sessions = np.array(ST)
        num_session = len(ST)
    elif st == 2:
        plotting_sessions = np.array(VG)
        num_session = len(VG)
    else:
        plotting_sessions = np.arange(num_session)
    num_session = len(plotting_sessions)
    
    cmap1 = plt.cm.Grays
    norm1 = plt.Normalize(vmin=0, vmax=num_session+1)   
    cmap2 = plt.cm.Blues
    norm2 = plt.Normalize(vmin=0, vmax=num_session+1) 
    
    for i in plotting_sessions:
        for state in range(len(states)):
            align_data, align_time, time_all, trial_type, epoch, outcome, opto = get_js_pos(session_data['events_timing'][i], states[state])
            if refrence == 'opto':
                selected_align_data1, selected_labels1 = selected_trials_js_pos(align_data, opto, 0, refrence2 = np.nan, condition2 = np.nan)
                selected_align_data2, selected_labels2 = selected_trials_js_pos(align_data, opto, 1, refrence2 = np.nan, condition2 = np.nan)
            elif refrence == 'short':
                selected_align_data1, selected_labels1 = selected_trials_js_pos(align_data, trial_type, 1, refrence2 = opto, condition2 = 0)
                selected_align_data2, selected_labels2 = selected_trials_js_pos(align_data, trial_type, 1, refrence2 = opto, condition2 = 1)
            elif refrence == 'long':
                selected_align_data1, selected_labels1 = selected_trials_js_pos(align_data, trial_type, 2, refrence2 = opto, condition2 = 0)
                selected_align_data2, selected_labels2 = selected_trials_js_pos(align_data, trial_type, 2, refrence2 = opto, condition2 = 1)
            plot_aligned_data(axs1[state], selected_align_data1, align_time, states[state], color_tag = cmap1(norm1(i+1)))
            plot_aligned_data(axs2[state], selected_align_data2, align_time, states[state], color_tag = cmap2(norm2(i+1)))
            lay_out(axs1[state], x_label = 'Time (s)', y_label = 'Joystick pos (deg)', title = states[state][6:], x_lim = [-2 , 4], y_lim = [-0.05, 4.5], legend = 0)
            lay_out(axs2[state], x_label = 'Time (s)', y_label = 'Joystick pos (deg)', title = states[state][6:], x_lim = [-2 , 4], y_lim = [-0.05, 4.5], legend = 0)
        
def run_mega_session(axs1, session_data, axs2 = np.nan, refrence = 'opto', st = 0):
    dates = session_data['dates']
    num_session = len(dates)
    ST , VG = IsSelfTimed(session_data)
    if st == 1:
        plotting_sessions = np.array(ST)
        num_session = len(ST)
    elif st == 2:
        plotting_sessions = np.array(VG)
        num_session = len(VG)
    else:
        plotting_sessions = np.arange(num_session)
    num_session = len(plotting_sessions)
    
    cmap1 = plt.cm.Grays
    norm1 = plt.Normalize(vmin=0, vmax=num_session+1)   
    cmap2 = plt.cm.Blues
    norm2 = plt.Normalize(vmin=0, vmax=num_session+1) 
    
    for state in range(len(states)):
        align_data = []
        align_time = []
        time_all = []
        trial_type = []
        epoch = []
        outcome = []
        opto = []
        for i in plotting_sessions:
            align_data_temp, align_time_temp, time_all_temp, trial_type_temp, epoch_temp, outcome_temp, opto_temp = get_js_pos(session_data['events_timing'][i], states[state])
            align_data.append(align_data_temp)
            align_time.append(align_time_temp)
            time_all.append(time_all_temp)
            trial_type.append(trial_type_temp)
            epoch.append(epoch_temp)
            outcome.append(outcome_temp)
            opto.append(opto_temp)
            #print(align_data_temp.shape)
        align_data = [sub for group in align_data for sub in group]
        time_all = [sub for group in time_all for sub in group]
        if len(align_data) > 0:
            align_data, align_time, time_all = pad_seq(align_data, time_all)
            align_data = np.array(align_data)
        trial_type = np.array([sub for group in trial_type for sub in group])
        epoch = np.array([sub for group in epoch for sub in group])
        outcome = np.array([sub for group in outcome for sub in group])
        opto = np.array([sub for group in opto for sub in group])
        if refrence == 'opto':
            selected_align_data1, selected_labels1 = selected_trials_js_pos(align_data, opto, 0, refrence2 = np.nan, condition2 = np.nan)
            selected_align_data2, selected_labels2 = selected_trials_js_pos(align_data, opto, 1, refrence2 = np.nan, condition2 = np.nan)
        elif refrence == 'opto_db':
            selected_align_data1, selected_labels1 = selected_trials_js_pos(align_data, trial_type, 1, refrence2 = opto, condition2 = 0)
            selected_align_data2, selected_labels2 = selected_trials_js_pos(align_data, trial_type, 1, refrence2 = opto, condition2 = 1)
            selected_align_data3, selected_labels1 = selected_trials_js_pos(align_data, trial_type, 2, refrence2 = opto, condition2 = 0)
            selected_align_data4, selected_labels2 = selected_trials_js_pos(align_data, trial_type, 2, refrence2 = opto, condition2 = 1)
            plot_aligned_data(axs2[state], selected_align_data3, align_time, states[state], color_tag = 'k')
            plot_aligned_data(axs2[state], selected_align_data4, align_time, states[state], color_tag = 'b')
            lay_out(axs2[state], x_label = 'Time (s)', y_label = 'Joystick pos (deg)', title = states[state][6:], x_lim = [-2 , 4], y_lim = [-0.05, 4.5], legend = 1)
        elif refrence == 'short':
            selected_align_data1, selected_labels1 = selected_trials_js_pos(align_data, trial_type, 1, refrence2 = opto, condition2 = 0)
            selected_align_data2, selected_labels2 = selected_trials_js_pos(align_data, trial_type, 1, refrence2 = opto, condition2 = 1)
        elif refrence == 'long':
            selected_align_data1, selected_labels1 = selected_trials_js_pos(align_data, trial_type, 2, refrence2 = opto, condition2 = 0)
            selected_align_data2, selected_labels2 = selected_trials_js_pos(align_data, trial_type, 2, refrence2 = opto, condition2 = 1)
        plot_aligned_data(axs1[state], selected_align_data1, align_time, states[state], color_tag = 'k')
        plot_aligned_data(axs1[state], selected_align_data2, align_time, states[state], color_tag = 'b')
        lay_out(axs1[state], x_label = 'Time (s)', y_label = 'Joystick pos (deg)', title = states[state][6:], x_lim = [-2 , 4], y_lim = [-0.05, 4.5], legend = 1)
        #lay_out(axs2[state], x_label = 'Time (s)', y_label = 'Joystick pos (deg)', title = states[state][6:], x_lim = [-2 , 4], y_lim = [-0.05, 4.5], legend = 0)
        

        