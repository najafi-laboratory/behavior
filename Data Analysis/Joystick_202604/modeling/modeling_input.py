# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 18:31:09 2025

@author: saminnaji3
"""
import numpy as np

def IsSelfTimed(session_data):
    isSelfTime = session_data['isSelfTimedMode']
    
    VG = []
    ST = []
    all_sess = np.arange(0, len(isSelfTime))
    for i in range(0 , len(isSelfTime)):
        if isSelfTime[i][5] == 1 or isSelfTime[i][5] == np.nan:
            ST.append(i)
        else:
            VG.append(i)
    return ST , VG , all_sess

def behavior_reader(session_data, st, exlude_nan = 0):
    
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_raw = session_data['raw']
    ST , VG , all_sess = IsSelfTimed(session_data)
    if st == 1:
        sessions = ST
    elif st == 2:
        sessions = VG
    else:
        sessions = all_sess
    
    delay_vector = []
    target_low_vector = []
    target_high_vrctor = []
    type_vector = []
    type_vector_bin = []
    sessions_input = []
    session_outcome = []
    print(sessions)
    for i in sessions:
        outcome = outcomes[i]
        delay = np.array(session_data['session_comp_delay'][i])
        print('building behavioral vectors for  ' + dates[i])
        trial_types = session_raw[i]['TrialTypes'][0:len(outcome)]
        predelay2 = session_raw[i]['PrePress2Delay']
        trial_setting = session_raw[i]['TrialSettings']
        
        trial_lowerband = []
        trial_upperband = []
        trial_type = []
        trial_delay = []
        trial_outcome = []
        
        for trial in range(len(outcome)):
            if np.isnan(delay[trial]):
                if exlude_nan:
                    break
                else:
                    if outcome[trial] == 'Reward':
                        trial_outcome.append(1)
                    else:
                        trial_outcome.append(0)
                    trial_delay.append(delay[trial])
                    trial_type.append(trial_types[trial])
                    if trial_types[trial] == 1:
                        if 'Press2WindowShort_s' in trial_setting[trial]['GUI'].keys():
                            window = trial_setting[trial]['GUI']['Press2WindowShort_s'] 
                    else:
                        if 'Press2WindowLong_s' in trial_setting[trial]['GUI'].keys():
                            window = trial_setting[trial]['GUI']['Press2WindowLong_s'] 
                            
                    if 'Press2Window_s' in trial_setting[trial]['GUI'].keys():
                        window =trial_setting[trial]['GUI']['Press2Window_s'] 
                    
                    if session_data['isSelfTimedMode'][i][0] == 1:
                        prepress2delay = predelay2[trial]
                        trial_lowerband.append(prepress2delay)
                        trial_upperband.append(prepress2delay + window)
                    else:
                        prepress2delay = predelay2[trial] + 0.15
                        trial_lowerband.append(prepress2delay)
                        trial_upperband.append(prepress2delay + window + 0.15)
            else:
                if outcome[trial] == 'Reward':
                    trial_outcome.append(1)
                else:
                    trial_outcome.append(0)
                trial_delay.append(delay[trial])
                trial_type.append(trial_types[trial])
                if trial_types[trial] == 1:
                    if 'Press2WindowShort_s' in trial_setting[trial]['GUI'].keys():
                        window = trial_setting[trial]['GUI']['Press2WindowShort_s'] 
                else:
                    if 'Press2WindowLong_s' in trial_setting[trial]['GUI'].keys():
                        window = trial_setting[trial]['GUI']['Press2WindowLong_s'] 
                        
                if 'Press2Window_s' in trial_setting[trial]['GUI'].keys():
                    window =trial_setting[trial]['GUI']['Press2Window_s'] 
                
                if session_data['isSelfTimedMode'][i][0] == 1:
                    prepress2delay = predelay2[trial]
                    trial_lowerband.append(prepress2delay)
                    trial_upperband.append(prepress2delay + window)
                else:
                    prepress2delay = predelay2[trial] + 0.15
                    trial_lowerband.append(prepress2delay)
                    trial_upperband.append(prepress2delay + window + 0.15)
            
        session_outcome.append(trial_outcome)
        delay_vector.append(trial_delay)
        target_low_vector.append(trial_lowerband)
        target_high_vrctor.append(trial_upperband)
        type_vector.append(trial_type)
        type_vector_bin.append(np.array(trial_type)-1)
        sessions_input_temp = {
                'mouse_intervals': trial_delay ,
                'blocks': np.array(trial_type)-1 ,
                'cues': trial_lowerband,
                'upper_band': trial_upperband 
                }
        sessions_input.append(sessions_input_temp)
    
    behavioral_data = {
            'behavior': delay_vector ,
            'type': type_vector ,
            'target_high': target_high_vrctor , 
            'target_low': target_low_vector ,
            'label': session_outcome
            }
    
    return behavioral_data , sessions_input