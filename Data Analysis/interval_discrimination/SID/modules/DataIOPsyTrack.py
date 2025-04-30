# -*- coding: utf-8 -*-

import os
import scipy.io as sio
import scipy.stats as stats
import numpy as np
from tqdm import tqdm

session_data_path = 'C:\\behavior\\session_data'

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


def calculate_d_prime_and_criterion(hit_rate, false_alarm_rate):
    """
    Calculate d' (sensitivity) and c (criterion) for SDT based on hit and false alarm rates.
    
    Args:
        hit_rate (float): Proportion of correct identifications of long stimulus (Hits / (Hits + Misses)).
        false_alarm_rate (float): Proportion of false alarms (False Alarms / (False Alarms + Correct Rejections)).
    
    Returns:
        tuple: d' (sensitivity) and c (criterion)
    """
    # Clip values to avoid extremes (0 or 1) for z-score calculation
    hit_rate = np.clip(hit_rate, 1e-10, 1 - 1e-10)
    false_alarm_rate = np.clip(false_alarm_rate, 1e-10, 1 - 1e-10)
    
    # Compute z-scores for hit rate and false alarm rate
    z_hr = stats.norm.ppf(hit_rate)
    z_far = stats.norm.ppf(false_alarm_rate)
    
    # Calculate d' (sensitivity)
    d_prime = z_hr - z_far
    
    # Calculate c (criterion)
    criterion = -0.5 * (z_hr + z_far)
    
    return d_prime, criterion

# extract all session data for a subject

def read_trials(subject, session_data_path, fname):

    file_names = []
    if fname is None:
        file_names = os.listdir(os.path.join(session_data_path, subject))        
    else:
        file_names = [fname]     
        
    file_names.sort(key=lambda x: x[-19:])
    session_raw_data = []
    session_dates = []
    
    session_states = []
    session_ProcessedSessionData = []
    session_lick_eye = []    
    
    session_outcomes = []
    session_outcomes_left = []
    session_outcomes_right = []    
    session_outcomes_clean = []
    session_outcomes_time = []
    session_choice_start = []    
    session_lick = []
    session_reaction = []
    session_decision = []
    session_stim_start = []
    session_stim_seq = []
    session_pre_isi = []
    session_post_isi = []
    session_number_flash = []
    session_post_isi_early_included = []
    session_post_isi_mean = []
    session_jitter_flag = []
    session_opto_flag = []
    session_opto_trial = []
    session_opto_side = []
    session_pre_isi_emp = []
    session_post_isi_type = []
    
    session_MoveCorrectSpout = []
    session_TrialTypes = []
    
    session_outcomes_left_opto_on = []
    session_outcomes_right_opto_on = []
    session_outcomes_left_opto_off = []
    session_outcomes_right_opto_off = []                
    
    session_left_hit_rate = 0
    session_left_false_alarm_rate = 0
    session_right_hit_rate = 0
    session_right_false_alarm_rate = 0  
    

    session_left_d_prime = 0
    session_left_criterion = 0
    session_right_d_prime = 0
    session_right_criterion = 0
    
    session_d_prime = []
    session_criterion = []
    
    session_settings = []
    
    #PsyTrack
    session_name = subject
    session_y = []
    session_answer = []
    session_correct = []
    session_dayLength = []
    session_inputs = []
    
    session_filenames = []
    
    
    # session_id = os.path.splitext(fname)[0]
    
    for f in tqdm(range(len(file_names))):
        #PsyTrack        
        trial_y = []
        trial_answer = []
        trial_correct = []
        trial_dayLength = 0
        trial_inputs = []        
        
        
        fname = file_names[f]
        session_filenames.append(fname.replace('.mat', ''))
        # one session data.
        raw_data = load_mat(os.path.join(session_data_path, subject, fname))
        
        #PsyTrack         
        # session_name.append(subject)
        
        
        # get settings
        session_settings.append(raw_data['TrialSettings'][0]['GUI'])
        
        # jitter flag.
        trial_jitter_flag = [raw_data['TrialSettings'][i]['GUI']['ActRandomISI']
                      for i in range(raw_data['nTrials'])]
        session_raw_data.append(raw_data)
        # opto_tag
        if 'OptoTag' in raw_data.keys():
            optotag = raw_data['OptoTag']
        else:
            optotag = [0]*raw_data['nTrials']
            
        # set optoside according to gui param from update 3-6-25
        if 'OptoSide' in raw_data['TrialSettings'][0]['GUI'].keys():
            if raw_data['TrialSettings'][0]['GUI']['OptoSide'] == 1:
                # left
                raw_data['OptoSide'] = 0
            elif raw_data['TrialSettings'][0]['GUI']['OptoSide'] == 2:
                # right
                raw_data['OptoSide'] = 1
        # if not manually labeled and not in gui params, use other value
        elif 'OptoSide' not in raw_data.keys():
            raw_data['OptoSide'] = -2
            
        if raw_data['TrialSettings'][0]['GUI']['OptoSession']:
            optotrial = [int(x) for x in raw_data['OptoType']]
            optoside = [(raw_data['OptoSide'] + 1) * x for x in optotrial]  # 1 - left, 2 - right
        else:
            optotrial = [0]*raw_data['nTrials']
            # optoside = np.nan
            # raw_data['OptoSide'] = 0
            optoside = [(raw_data['OptoSide'] + 1) * x for x in optotrial]  # 1 - left, 2 - right
            
        # number of trials.
        nTrials = raw_data['nTrials']
        # trial target
        trial_types = np.array(raw_data['TrialTypes'])
        # session date
        session_dates.append(fname[-19:-11])
        # loop over one session for extracting data
        
        trial_states_list = []
        trial_ProcessedSessionData = []
        trial_lick_eye = []
        
            
        trial_outcomes = []
        trial_outcomes_left = []
        trial_outcomes_right = []        
        trial_outcomes_clean = []
        trial_outcomes_time = []
        trial_choice_start = []        
        trial_lick = []
        trial_reaction = []
        trial_decision = []
        trial_stim_start = []
        trial_stim_seq = []
        trial_pre_isi = []
        trial_post_isi = []
        trial_number_flash = []
        trial_post_isi_early_included = []
        trial_post_isi_mean = []
        trial_pre_isi_emp = []
        trial_post_isi_type = []
        trial_MoveCorrectSpout = []
        
        trial_outcomes_left_opto_on = []
        trial_outcomes_right_opto_on = []
        trial_outcomes_left_opto_off = []
        trial_outcomes_right_opto_off = []        
        
        trial_left_hits = 0
        trial_left_false_alarms = 0
        trial_left_misses = 0
        trial_left_correct_rejections = 0
        trial_right_hits = 0
        trial_right_false_alarms = 0
        trial_right_misses = 0
        trial_right_correct_rejections = 0
        
        for i in range(nTrials):
            trial_states = raw_data['RawEvents']['Trial'][i]['States']
            trial_events = raw_data['RawEvents']['Trial'][i]['Events']
            # outcome
            outcome , outcome_time = states_labeling(trial_states)
            outcome_clean = outcome
            
            

            # eye tracking data
            ################################################################
            trial_states_list.append(trial_states)
            trial_ProcessedSessionData.append(raw_data['ProcessedSessionData'][i])
            
            licking_events = []
            direction = []
            correctness = []
            trial_type = []
            if 'Port1In' in trial_events.keys():
                # get lick time
                lick_left = np.array(trial_events['Port1In']).reshape(-1)
                licking_events.append(lick_left)
                direction.append(np.zeros_like(lick_left))
                if trial_types[i] == 1:
                    correctness.append(np.ones_like(lick_left))
                else:
                    correctness.append(np.zeros_like(lick_left))
            if 'Port3In' in trial_events.keys():               
                # get lick time
                lick_right = np.array(trial_events['Port3In']).reshape(-1)                    
                licking_events.append(lick_right)
                direction.append(np.ones_like(lick_right))
                if trial_types[i] == 2:
                    correctness.append(np.ones_like(lick_right))
                else:
                    correctness.append(np.zeros_like(lick_right))
            if len(licking_events) > 0:                    
                licking_events = np.concatenate(licking_events).reshape(1,-1)   
                correctness = np.concatenate(correctness).reshape(1,-1)
                direction = np.concatenate(direction).reshape(1,-1)
                # lick array
                # row 1 time of lick event
                # row 2 lick direction - 0 left, 1 right
                # row 3 correctness - 0 incorrect, 1 correct                    
                lick = np.concatenate([1000*licking_events, direction, correctness])
                lick = lick[: , lick[0, :].argsort()]
                trial_lick_eye.append(lick)
            else:
                trial_lick_eye.append(np.array([[np.nan], [np.nan], [np.nan]]))            
            
            ####################################################################################################
            
            

            
            # trial_outcomes.append(outcome)
           
            trial_outcomes_time.append(outcome_time)
            # stimulus start.
            if ('VisStimTrigger' in trial_states.keys() and
                not np.isnan(trial_states['VisStimTrigger'][0])):
                trial_stim_start.append(np.array([1000*np.array(trial_states['VisStimTrigger'][1])]))
            else:
                trial_stim_start.append(np.array([np.nan]))
            # stimulus sequence.
            if ('BNC1High' in trial_events.keys() and
                'BNC1Low' in trial_events.keys() and
                np.array(trial_events['BNC1High']).reshape(-1).shape[0]==np.array(trial_events['BNC1Low']).reshape(-1).shape[0] and
                np.array(trial_events['BNC1High']).reshape(-1).shape[0] >= 2
                ):
                stim_seq = 1000*np.array([trial_events['BNC1High'], trial_events['BNC1Low']])
            else:
                stim_seq = np.array([[np.nan], [np.nan]])
            trial_stim_seq.append(stim_seq)
            # pre prerturbation isi.
            stim_pre_isi = np.median(stim_seq[0,1:3] - stim_seq[1,0:2])
            
            ########### finding early licks
            if not 'Port1In' in trial_events.keys():
                port1 = [np.nan]
            elif type(trial_events['Port1In']) == float:
                port1 = [trial_events['Port1In']]
            else:
                port1 = trial_events['Port1In']
            if not 'Port3In' in trial_events.keys():
                port3 = [np.nan]
            elif type(trial_events['Port3In']) == float:
                port3 = [trial_events['Port3In']]
            else:
                port3 = trial_events['Port3In']
                
            early_lick = False
            early_lick_limited = False
            if len(stim_seq[1,:]) > 2:
                t1 = stim_seq[1,2]/1000
                if not (isinstance(port1 , (int , float)) or isinstance(port3 , (int , float))):
                    early_lick_limited = any(ele >= t1 and ele < t1+0.1 for ele in port1) or any(ele >= t1 and ele < t1+0.1 for ele in port3)
                    early_lick = any(ele < t1+0.1 for ele in port1) or any( ele < t1+0.1 for ele in port3)
            if early_lick: 
                outcome_clean = 'EarlyLick'
            elif early_lick_limited:
                outcome_clean = 'EarlyLickLimited'
                
            early_lick = False
            early_lick_limited = False                
            ###########
            
            ########## late choice
            late_choice = False
            
            if outcome == 'Reward':
                if trial_states['Reward'][0] - trial_states['VisStimTrigger'][1] > 4.5:
                    late_choice = True
            elif outcome == 'Punish':
                if trial_states['Punish'][0] - trial_states['VisStimTrigger'][1] > 4.5:
                    late_choice = True
            if late_choice: 
                outcome_clean = 'LateChoice'
            
            late_choice = False            
            ##########
            
            
            ########## switching trials
            switching = False
            if outcome == 'Reward':
                if (type(trial_states['WindowChoice'][0]) == float) or (type(trial_states['WindowChoice'][0]) == np.float64) or all(np.isnan(trial_states[states[k]][0])):
                    choice_window = trial_states['WindowChoice'][0]
                else:
                    choice_window = trial_states['WindowChoice'][-1][0]
                if len(stim_seq[1,:]) > 2:
                    t1 = stim_seq[1,2]/1000
                    if trial_types[i] == 1:
                        switching = any(ele >= t1 and ele < choice_window for ele in port3)
                    else: 
                        switching = any(ele >= t1 and ele < choice_window for ele in port1)
            elif outcome == 'Punish':
                if (type(trial_states['WindowChoice'][0]) == float) or (type(trial_states['WindowChoice'][0]) == np.float64) or all(np.isnan(trial_states[states[k]][0])):
                    choice_window = trial_states['WindowChoice'][0]
                else:
                    choice_window = trial_states['WindowChoice'][-1][0]
                if len(stim_seq[1,:]) > 2:
                    t1 = stim_seq[1,2]/1000
                    if not (isinstance(port1 , (int , float)) or isinstance(port3 , (int , float))):
                        if trial_types[i] == 1:
                            switching = any(ele >= t1 and ele < choice_window for ele in port1)
                        else: 
                            switching = any(ele >= t1 and ele < choice_window for ele in port3)
                
            if switching:
                outcome_clean = 'Switching'
            
            ##########
            
            # revert to prev version
            outcome_clean = outcome
            
            # filter MoveCorrectSpout AntiBias
            if ('MoveCorrectSpout' in raw_data.keys()):
                if (raw_data['MoveCorrectSpout'][i]):
                    outcome = 'MoveCorrectSpout'                    
                    outcome_clean = 'MoveCorrectSpout'
                    trial_MoveCorrectSpout.append(1)
                else:
                    trial_MoveCorrectSpout.append(0)
            else:
                trial_MoveCorrectSpout.append(0)                    
            
                
            # filter warmup trials
            
            
            trial_outcomes.append(outcome)                    
            trial_outcomes_clean.append(outcome_clean)        
            
            # left and right outcomes
            if trial_types[i] == 1:
                trial_outcomes_left.append(outcome)
            else:
                trial_outcomes_right.append(outcome)   
              
            # opto, left and right outcomes 
            if optotrial[i] == 1:
            # left and right outcomes
                if trial_types[i] == 1:
                    trial_outcomes_left_opto_on.append(outcome)
                else:
                    trial_outcomes_right_opto_on.append(outcome)                
            else:
                if trial_types[i] == 1:
                    trial_outcomes_left_opto_off.append(outcome)
                else:
                    trial_outcomes_right_opto_off.append(outcome)                    
                                                     
            # if (not outcome_clean == 'EarlyLick' and not outcome_clean == 'earlyLickLimited' and not outcome_clean == 'Switching' and not outcome_clean == 'LateChoice'):
                # trial_dayLength += 1
                
                # trial_y.append()
            
            # pre_isi_emp = 1000*np.mean(raw_data['ProcessedSessionData'][i]['trial_isi']['PreISI'])
            pre_isi_emp = np.float64(0)  # no pre isi for single interval              
            trial_pre_isi.append(stim_pre_isi)
            trial_pre_isi_emp.append(pre_isi_emp)
            
            # if i == 20:
            #     print(i)

                        
            # post perturbation isi.
            if (not outcome_clean == 'EarlyLick' and not outcome_clean == 'earlyLickLimited' and not outcome_clean == 'Switching' and not outcome_clean == 'LateChoice' and not outcome_clean == 'MoveCorrectSpout'):
                # stim_post_isi_mean = 1000*np.mean(raw_data['ProcessedSessionData'][i]['trial_isi']['PostISI'])
                stim_post_isi_mean = 1000*np.mean(raw_data['ProcessedSessionData'][i]['trial_isi']['PostISI'][0])
                if stim_seq.shape[1]<2:
                    stim_post_isi = np.nan
                    number_flash = np.nan
                    if outcome == 'Reward':
                        stim_post_isi_early_included = 900
                        number_flash = 0
                    else: 
                        stim_post_isi_early_included = np.nan
                else:
                    interupt = np.count_nonzero(stim_seq[0 , :] < 1000*outcome_time)
                    stim_post_isi = np.median(stim_seq[0,3:interupt] - stim_seq[1,2:interupt-1])
                    stim_post_isi_early_included = np.median(stim_seq[0,3:interupt] - stim_seq[1,2:interupt-1])
                    if (interupt == 3):
                        stim_post_isi = np.median(stim_seq[0,3:4] - stim_seq[1,2:4-1])
                        number_flash = 0
                        if outcome == 'Reward':
                            stim_post_isi_early_included = 900
                    else:
                        number_flash = interupt - 3
            else:
                stim_post_isi = np.nan
                number_flash = np.nan 
                stim_post_isi_early_included = np.nan 
                stim_post_isi_mean = np.nan
            trial_post_isi.append(stim_post_isi)
            trial_number_flash.append(number_flash)
            trial_post_isi_early_included.append(stim_post_isi_early_included)
            trial_post_isi_mean.append(stim_post_isi_mean)
            trial_post_isi_type.append(int(1000*np.mean(raw_data['ProcessedSessionData'][i]['trial_isi']['PostISI']) > 500))
            
            # choice window
            if ('WindowChoice' in trial_states.keys() and
                not np.isnan(trial_states['WindowChoice'][0])):
                trial_choice_start.append(trial_states['WindowChoice'][0])
            else:
                trial_choice_start.append(np.nan)     
                
            # lick events.
            if ('VisStimTrigger' in trial_states.keys() and
                i > 14 and not np.isnan(trial_states['VisStimTrigger'][1]) and not outcome_clean == 'EarlyLick' and 
                not outcome_clean == 'Switching' and not outcome_clean == 'earlyLickLimited' and not outcome_clean == 'LateChoice' and 
                not outcome_clean == 'MoveCorrectSpout'):
                licking_events = []
                direction = []
                correctness = []
                trial_type = []
                num_left = 0            
                
                if 'Port1In' in trial_events.keys():
                    # get lick time relative to window
                    # lick_left = np.array(trial_events['Port1In'] - trial_states['WindowChoice'][0]).reshape(-1)
                    # get lick time
                    lick_left = np.array(trial_events['Port1In']).reshape(-1)
                    licking_events.append(lick_left)
                    direction.append(np.zeros_like(lick_left))
                    num_left = len(lick_left)
                    if trial_types[i] == 1:
                        correctness.append(np.ones_like(lick_left))
                    else:
                        correctness.append(np.zeros_like(lick_left))
                if 'Port3In' in trial_events.keys():
                    # get lick time relative to window
                    # lick_right = np.array(trial_events['Port3In'] - trial_states['WindowChoice'][0]).reshape(-1)                  
                    # get lick time
                    lick_right = np.array(trial_events['Port3In']).reshape(-1)                    
                    trial_states['WindowChoice'][1]
                    licking_events.append(lick_right)
                    direction.append(np.ones_like(lick_right))
                    if trial_types[i] == 2:
                        correctness.append(np.ones_like(lick_right))
                    else:
                        correctness.append(np.zeros_like(lick_right))
                if len(licking_events) > 0:                    
                    licking_events = np.concatenate(licking_events).reshape(1,-1)
                    correctness = np.concatenate(correctness).reshape(1,-1)
                    direction = np.concatenate(direction).reshape(1,-1)
                    # lick array
                    # row 1 time of lick event
                    # row 2 lick direction - 0 left, 1 right
                    # row 3 correctness - 0 incorrect, 1 correct                    
                    lick = np.concatenate([1000*licking_events, direction, correctness])
                    lick = lick[: , lick[0, :].argsort()]
                    # all licking events.
                    trial_lick.append(lick)
                    # reaction licking. ie. the lick after start of vis stim
                    # check that licks are after 'VisStimTrigger'
                    reaction_idx = np.where(lick[0]>1000*trial_states['VisStimTrigger'][1])[0]
                    if len(reaction_idx)>0:                     
                        # lick_reaction = lick.copy()[:, reaction_idx[0]].reshape(3,1)
                        # get earliest lick as 'reaction'
                        # find minimum reaction time as lick
                        # if mouse licked both spouts at same time (somehow?), check if trial rewarded to select lick column
                        lick_min_idx = np.where(lick[0] == np.min(lick[0]))
                        lick_copy_min = lick.copy()[:, lick_min_idx]
                        if lick_copy_min.size > 3:
                            print(i)
                            print('double')   
                            lick_copy_min = lick_copy_min.reshape(3, 2)
                            if outcome == 'Reward':                                
                                lick_min_idx = np.where(lick_copy_min[2] == 1)
                            else:
                                lick_min_idx = np.where(lick_copy_min[2] == 0)                            
                        lick_reaction = lick.copy()[:, lick_min_idx].reshape(3,1)                        
                        trial_reaction.append(lick_reaction)
                    else:
                        trial_reaction.append(np.array([[np.nan], [np.nan], [np.nan]]))
                   
                    
                   # effective licking to outcome. ie. licks after start of window choice
                    # decision_idx = np.where(lick[0]>1000*trial_states['WindowChoice'][0])[0]
                    # lick = lick[:,np.where(lick[0]>1000*trial_states['WindowChoice'][0])].reshape(3,1)
                    lick = lick[:,lick[0]>1000*trial_states['WindowChoice'][0]]
                    
                    # licks always after window
                    # decision_idx = [0]
                    decision_idx = np.where(lick[0]>1000*trial_states['WindowChoice'][0])[0]
                    
                    if (len(decision_idx)>0 and
                        stim_seq.shape[1]>=2
                        ):
                        # lick_decision = lick.copy()[:, decision_idx[0]].reshape(3,1)
                        # get earliest lick as 'decision'
                        # find minimum reaction time as lick
                        # if mouse licked both spouts at same time (somehow?, happens trial 57 TS01 1-16-25), check if trial rewarded to select lick column
                        lick_min_idx = np.where(lick[0] == np.min(lick[0]))                        
                        lick_copy_min = lick.copy()[:, np.where(lick[0] == np.min(lick[0]))]
                        if lick_copy_min.size > 3:
                            print(i)
                            print('double')   
                            lick_copy_min = lick_copy_min.reshape(3, 2)
                            if outcome == 'Reward':                                
                                lick_min_idx = np.where(lick_copy_min[2] == 1)
                            else:
                                lick_min_idx = np.where(lick_copy_min[2] == 0)               
                      
                        lick_decision = lick.copy()[:, lick_min_idx].reshape(3,1)
                        trial_decision.append(lick_decision)
                                                                  
                                                
                        # calculate hits as correct identification of short/long duration isi
                        if trial_types[i] == 1:
                            if lick_decision[2] == 1:
                                trial_left_hits += 1
                                trial_right_correct_rejections += 1
                            else:
                                trial_left_misses += 1
                                trial_right_false_alarms += 1
                        else:
                            if lick_decision[2] == 1:
                                trial_right_hits += 1
                                trial_left_correct_rejections += 1
                            else:
                                trial_right_misses += 1
                                trial_left_false_alarms += 1
                                             
                        #PsyTrack
                        # lick array
                        trial_dayLength += 1                          

                        # row 2 lick direction - 0 left, 1 right                        
                        trial_y.append(int(lick_decision[1]))
 
                        if trial_types[i] == 1:
                            trial_answer.append(1)
                        else:
                            trial_answer.append(2)                            
                        
                        # row 3 correctness - 0 incorrect, 1 correct
                        if lick_decision[2] == 0:
                            trial_correct.append(0)
                        else:
                            trial_correct.append(1)
                        
                    else:
                        trial_decision.append(np.array([[np.nan], [np.nan], [np.nan]]))

                else:
                    trial_lick.append(np.array([[np.nan], [np.nan], [np.nan]]))
                    trial_reaction.append(np.array([[np.nan], [np.nan], [np.nan]]))
                    trial_decision.append(np.array([[np.nan], [np.nan], [np.nan]]))

            else:
                trial_lick.append(np.array([[np.nan], [np.nan], [np.nan]]))
                trial_reaction.append(np.array([[np.nan], [np.nan], [np.nan]]))
                trial_decision.append(np.array([[np.nan], [np.nan], [np.nan]]))




        # signal detection theory
        session_left_hit_rate = trial_left_hits / (trial_left_hits + trial_left_misses)
        session_left_false_alarm_rate = trial_left_false_alarms / (trial_left_false_alarms + trial_left_correct_rejections)
        
        session_right_hit_rate = trial_right_hits / (trial_right_hits + trial_right_misses)
        session_right_false_alarm_rate = trial_right_false_alarms / (trial_right_false_alarms + trial_right_correct_rejections)        
        
        # Calculate d' and criterion
        session_left_d_prime, session_left_criterion = calculate_d_prime_and_criterion(session_left_hit_rate, session_left_false_alarm_rate)
        session_right_d_prime, session_right_criterion = calculate_d_prime_and_criterion(session_right_hit_rate, session_right_false_alarm_rate)

      
        session_d_prime.append(session_left_d_prime)
        session_criterion.append(session_left_criterion)

        # save one session data
        #PsyTrack
        session_y.append(trial_y)
        session_answer.append(trial_answer)
        session_correct.append(trial_correct)
        session_dayLength.append(trial_dayLength)
        
    
        session_states.append(trial_states_list)
        session_ProcessedSessionData.append(trial_ProcessedSessionData)
        session_lick_eye.append(trial_lick_eye)
    
        session_outcomes.append(trial_outcomes)
        session_outcomes_left.append(trial_outcomes_left)
        session_outcomes_right.append(trial_outcomes_right)        
        session_outcomes_clean.append(trial_outcomes_clean)
        session_outcomes_time.append(trial_outcomes_time)
        session_choice_start.append(trial_choice_start)             
        session_lick.append(trial_lick)
        session_reaction.append(trial_reaction)
        session_decision.append(trial_decision)
        session_stim_start.append(trial_stim_start)
        session_stim_seq.append(trial_stim_seq)
        session_pre_isi.append(trial_pre_isi)
        session_pre_isi_emp.append(trial_pre_isi_emp)
        session_post_isi_early_included.append(trial_post_isi_early_included)
        session_post_isi.append(trial_post_isi)
        session_post_isi_type.append(trial_post_isi_type)
        session_number_flash.append(trial_number_flash)
        session_post_isi_mean.append(trial_post_isi_mean)
        session_jitter_flag.append(trial_jitter_flag)
        session_opto_flag.append(optotag)
        session_opto_trial.append(optotrial)
        session_opto_side.append(optoside)
        session_MoveCorrectSpout.append(trial_MoveCorrectSpout)
        session_TrialTypes.append(trial_types)                
        
        session_outcomes_left_opto_on.append(trial_outcomes_left_opto_on)
        session_outcomes_right_opto_on.append(trial_outcomes_right_opto_on)
        session_outcomes_left_opto_off.append(trial_outcomes_left_opto_off)
        session_outcomes_right_opto_off.append(trial_outcomes_right_opto_off)        
        
    #PsyTrack
    y = []
    answer = []
    correct = []
    dayLength = []
    
    for sublist in session_y:
        y += sublist
        
    y = np.asarray(y, dtype=np.float64)        
        
    for sublist in session_answer:
        answer += sublist        
      
    answer = np.asarray(answer, dtype=np.float64)   
      
    for sublist in session_correct:
        correct += sublist

    correct = np.asarray(correct, dtype=np.float64)  

    # for sublist in session_dayLength:
    dayLength = np.asarray(session_dayLength)     
        
    # merge all session data
    data = {
        'name' : session_name,
        'y' : y,
        'answer' : answer,
        'correct' : correct,
        'dayLength' : dayLength,
        'total_sessions' : len(session_outcomes),
        'subject' : subject,
        'filename' : file_names,
        'raw' : session_raw_data,
        'dates' : session_dates,
        'outcomes' : session_outcomes,
        'outcomes_left' : session_outcomes_left,
        'outcomes_right' : session_outcomes_right,        
        'outcomes_clean' : session_outcomes_clean,
        'outcomes_time' : session_outcomes_time,
        'choice_start' : session_choice_start,        
        'lick' : session_lick,
        'reaction' : session_reaction,
        'decision' : session_decision,
        'stim_start' : session_stim_start,
        'stim_seq' : session_stim_seq,
        'pre_isi' : session_pre_isi,
        'post_isi' : session_post_isi,
        'isi_pre_emp' : session_pre_isi_emp,
        'post_isi_early_included': session_post_isi_early_included,
        'number_flash': session_number_flash,
        'isi_post_emp' : session_post_isi_mean,
        'jitter_flag' : session_jitter_flag,
        'post_isi_type' : session_post_isi_type,
        'opto_flag' : session_opto_flag,
        'opto_trial' : session_opto_trial,
        'opto_side' : session_opto_side,
        'move_correct_spout_flag' : session_MoveCorrectSpout,
        'trial_type' : session_TrialTypes,
        'd_prime' : session_d_prime,
        'criterion' : session_criterion,
        'states' : session_states,
        'ProcessedSessionData' : session_ProcessedSessionData,
        'lick_eye' : session_lick_eye,
        'outcomes_left_opto_on' : session_outcomes_left_opto_on,
        'outcomes_right_opto_on' : session_outcomes_right_opto_on,
        'outcomes_left_opto_off' : session_outcomes_left_opto_off,
        'outcomes_right_opto_off' : session_outcomes_right_opto_off,
        'session_settings' : session_settings,
        'session_filenames' : session_filenames,
    }
    return data


# labeling every trials for a subject
def states_labeling(trial_states):
    if 'ChangingMindReward' in trial_states.keys() and not np.isnan(trial_states['ChangingMindReward'][0]):
        outcome = 'ChangingMindReward'
        outcome_time = trial_states['ChangingMindReward'][0]
    elif 'WrongInitiation' in trial_states.keys() and not np.isnan(trial_states['WrongInitiation'][0]):
        outcome = 'WrongInitiation'
        outcome_time = trial_states['WrongInitiation'][0]
    elif 'Punish' in trial_states.keys() and not np.isnan(trial_states['Punish'][0]):
        outcome = 'Punish'
        outcome_time = trial_states['Punish'][0]
    elif 'Reward' in trial_states.keys() and not np.isnan(trial_states['Reward'][0]):
        outcome = 'Reward'
        outcome_time = trial_states['Reward'][0]
    elif 'PunishNaive' in trial_states.keys() and not np.isnan(trial_states['PunishNaive'][0]):
        outcome = 'PunishNaive'
        outcome_time = trial_states['PunishNaive'][0]
    elif 'RewardNaive' in trial_states.keys() and not np.isnan(trial_states['RewardNaive'][0]):
        outcome = 'RewardNaive'
        outcome_time = trial_states['RewardNaive'][0]
    elif 'EarlyChoice' in trial_states.keys() and not np.isnan(trial_states['EarlyChoice'][0]):
        outcome = 'EarlyChoice'
        outcome_time = trial_states['EarlyChoice'][0]
    elif 'DidNotChoose' in trial_states.keys() and not np.isnan(trial_states['DidNotChoose'][0]):
        outcome = 'DidNotChoose'
        outcome_time = trial_states['DidNotChoose'][0]
    else:
        outcome = 'Other'
        outcome_time = np.nan
    return outcome , outcome_time


# read session data given subjects
def run(subject_list , session_data_path, file_path=None):
    session_data = []
    if not isinstance(subject_list, list):
        subject_list = [subject_list]
    for sub in subject_list:
        print('reading data for ' + sub)
        session_data.append(read_trials(sub, session_data_path, file_path))
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    return session_data

