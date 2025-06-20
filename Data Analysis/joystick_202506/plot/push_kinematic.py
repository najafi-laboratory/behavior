# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 13:20:29 2025

@author: saminnaji3
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

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
    
def read_kinematics(
        session_data
        ):
    
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_id = np.arange(len(outcomes)) + 1
    session_raw = session_data['raw']
    
    session_vis1 = []
    session_push1 = []
    session_retract1 = []
    session_retract1_init = []
    session_vis2 = []
    session_wait2 = []
    session_push2 = []
    session_retract2 = []
    session_reward = []
    session_punish = []
    session_no1stpush = []
    session_no2ndpush = []
    session_early2ndpush = []
    session_iti = []
    session_lick = []
    session_delay = []
    session_js_pos = []
    session_js_time = []
    session_outcome = []
    session_trial_types = []
    session_opto_tag = []
    
    for j in range(0 , len(session_id)):
        outcome = outcomes[j]
        raw = session_raw[j]
        
        trial_vis1 = []
        trial_push1 = []
        trial_retract1 = []
        trial_retract1_init = []
        trial_vis2 = []
        trial_wait2 = []
        trial_push2 = []
        trial_retract2 = []
        trial_reward = []
        trial_punish = []
        trial_no1stpush = []
        trial_no2ndpush = []
        trial_early2ndpush = []
        trial_iti = []
        trial_lick = []
        trial_delay = []
        trial_js_pos = []
        trial_js_time = []
        trial_outcome = []
        trial_opto_tag = session_data['session_opto_tag'][j]
        
        
        print('processing push kinematic for session: ', session_data['dates'][j])
        for i in range(0 , len(outcome)):
            
            trial_states = raw['RawEvents']['Trial'][i]['States']
            trial_events = raw['RawEvents']['Trial'][i]['Events']
            
            trial_outcome.append(outcome[i])
            
            # push1 onset.
            start = 1000*np.array(trial_states['VisDetect1'][0]).reshape(-1)-500
            end = 1000*np.array(trial_states['LeverRetract1'][0]).reshape(-1)-100
            push1 = get_push_onset(
                np.array(raw['EncoderData'][i]['Positions']).reshape(-1),
                1000*np.array(raw['EncoderData'][i]['Times']).reshape(-1),
                start, end)
            
            # late push1 onsset
            if outcome[i] == 'LatePress1':
                start = 1000*np.array(trial_states['DidNotPress1'][0]).reshape(-1)
                end = start+ 1500
                push1 = get_push_onset(
                    np.array(raw['EncoderData'][i]['Positions']).reshape(-1),
                    1000*np.array(raw['EncoderData'][i]['Times']).reshape(-1),
                    start, end)
                #print(push1, x_push1)
                
            # push2 onset.
            start = 1000*np.array(trial_states['WaitForPress2'][0]).reshape(-1)
            #start = 1000*np.array(trial_states['LeverRetract1'][1]).reshape(-1)
            if ('RotaryEncoder1_1' in trial_events.keys() and np.size(trial_events['RotaryEncoder1_1'])>1):

                end = 1000*np.array(trial_events['RotaryEncoder1_1'][1]).reshape(-1)

            else:
                end = np.array([np.nan])
            push2 = get_push_onset(
                np.array(raw['EncoderData'][i]['Positions']).reshape(-1),
                1000*np.array(raw['EncoderData'][i]['Times']).reshape(-1),
                start, end)
            if outcome[i] == 'EarlyPress2':
                start = 1000*np.array(trial_states['LeverRetract1'][1]).reshape(-1)
                if ('RotaryEncoder1_1' in trial_events.keys() and np.size(trial_events['RotaryEncoder1_1'])>1):

                    end = 1000*np.array(trial_events['RotaryEncoder1_1'][1]).reshape(-1)

                else:
                    end = np.array([np.nan])
                push2 = get_push_onset(
                    np.array(raw['EncoderData'][i]['Positions']).reshape(-1),
                    1000*np.array(raw['EncoderData'][i]['Times']).reshape(-1),
                    start, end)
                
            if np.isnan(push1):
                push2 = np.array([np.nan])
            if not np.isnan(push2):
                if push2 < 1000*np.array(trial_states['LeverRetract1'][1]).reshape(-1):
                    push2 = np.array([np.nan])
            trial_push1.append(push1)
            trial_push2.append(push2)
            

            trial_vis1.append(1000*np.array(trial_states['VisualStimulus1']).reshape(-1))
            # 1st retract.
            trial_retract1.append(1000*np.array(trial_states['LeverRetract1'][1]).reshape(-1))
            trial_retract1_init.append(1000*np.array(trial_states['LeverRetract1'][0]).reshape(-1))
            # 2nd stim.
            trial_vis2.append(1000*np.array(trial_states['VisualStimulus2']).reshape(-1))
            # wait for 2nd push.
            trial_wait2.append(1000*np.array(trial_states['WaitForPress2'][0]).reshape(-1))
            # 2nd retract.
            if ('LeverRetract2' in trial_states.keys()):
                trial_retract2.append(1000*np.array(trial_states['LeverRetract2'][0]).reshape(-1))
            else:
                trial_retract2.append(1000*np.array([np.nan, np.nan]).reshape(-1))
            # reward.
            trial_reward.append(1000*np.array(trial_states['Reward']).reshape(-1))
            # punish.
            trial_punish.append(1000*np.array(trial_states['Punish']).reshape(-1))
            # did not push 1.
            trial_no1stpush.append(1000*np.array(trial_states['DidNotPress1']).reshape(-1))
            # did not push 2.
            trial_no2ndpush.append(1000*np.array(trial_states['DidNotPress2']).reshape(-1))
            # early push 2.
            trial_early2ndpush.append(1000*np.array(trial_states['EarlyPress2']).reshape(-1))
            # licking events.
            if 'Port2In' in trial_events.keys():
                lick_all = 1000*np.array(trial_events['Port2In']).reshape(1,-1)
                lick_label = np.zeros_like(lick_all).reshape(1,-1)
                lick_label[lick_all>1000*np.array(trial_states['Reward'][0])] = 1
                trial_lick.append(np.concatenate((lick_all, lick_label), axis=0))
            else:
                trial_lick.append(np.array([[np.nan],[np.nan]]))
                
            # iti.
            trial_iti.append(1000*np.array(trial_states['ITI']).reshape(-1))
            # delay
            #print(np.min(raw['TrialTypes']),np.max(raw['TrialTypes']))
            if np.min(raw['TrialTypes']) == np.max(raw['TrialTypes']):
                if raw['TrialTypes'][i] == 1:
                    trial_delay.append(1000*raw['TrialSettings'][i]['GUI']['PrePress2DelayShort_s'])
                if raw['TrialTypes'][i] == 2:
                    trial_delay.append(1000*raw['TrialSettings'][i]['GUI']['PrePress2DelayLong_s'])
                #trial_delay.append(1000*raw['TrialSettings'][i]['GUI']['PressVisDelayLong_s'])
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
        
        session_vis1.append(trial_vis1)
        session_push1.append(trial_push1)
        session_retract1.append(trial_retract1)
        session_retract1_init.append(trial_retract1_init)
        session_vis2.append(trial_vis2)
        session_wait2.append(trial_wait2)
        session_push2.append(trial_push2)
        session_retract2.append(trial_retract2)
        session_reward.append(trial_reward)
        session_punish.append(trial_punish)
        session_no1stpush.append(trial_no1stpush)
        session_no2ndpush.append(trial_no2ndpush)
        session_early2ndpush.append(trial_early2ndpush)
        session_iti.append(trial_iti)
        session_lick.append(trial_lick)
        session_delay.append(trial_delay)
        session_js_pos.append(trial_js_pos)
        session_js_time.append(trial_js_time)
        session_outcome.append(trial_outcome)
        session_trial_types.append(np.array(raw['TrialTypes']))
        session_opto_tag.append(trial_opto_tag)
    
    push_data = {
            'trial_types'  : session_trial_types,
            'vis1'         : session_vis1,
            'push1'        : session_push1,
            'retract1'     : session_retract1,
            'retract1_init': session_retract1_init,
            'vis2'         : session_vis2,
            'wait2'        : session_wait2,
            'push2'        : session_push2,
            'retract2'     : session_retract2,
            'reward'       : session_reward,
            'punish'       : session_punish,
            'no1stpush'    : session_no1stpush,
            'no2ndpush'    : session_no2ndpush,
            'early2ndpush' : session_early2ndpush,
            'iti'          : session_iti,
            'lick'         : session_lick,
            'delay'        : session_delay,
            'js_pos'       : session_js_pos,
            'js_time'      : session_js_time,
            'outcome'      : session_outcome,
            'opto'         : session_opto_tag,
            'date'         : dates ,
            'subject'      : subject
            }
    return push_data 