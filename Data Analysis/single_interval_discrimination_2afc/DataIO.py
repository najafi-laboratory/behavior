import os
import scipy.io as sio
import numpy as np
from tqdm import tqdm

session_data_path = 'C:\\behavior\\session_data'

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

def read_trials(subject):

    file_names = os.listdir(os.path.join(session_data_path, subject))
    file_names.sort(key=lambda x: x[-19:])
    session_raw_data = []
    session_dates = []
    session_outcomes = []
    session_outcomes_left = []
    session_outcomes_right = []
    session_gocue_start = []
    session_choice_start = []
    session_lick = []
    session_reaction = []
    session_decision = []
    session_stim_start = []
    session_stim_seq = []
    session_isi_pre_perc = []
    session_isi_pre_emp = []
    session_isi_post_perc = []
    session_isi_post_emp = []
    session_jitter_flag = []
    states = [
        'Reward',
        'RewardNaive',
        'ChangingMindReward',
        'Punish',
        'PunishNaive']
    for f in tqdm(range(len(file_names))):
        fname = file_names[f]
        # one session data.
        raw_data = load_mat(os.path.join(session_data_path, subject, fname))
        # jitter flag.
        trial_jitter_flag = [raw_data['TrialSettings'][i]['GUI']['ActRandomISI']
                      for i in range(raw_data['nTrials'])]
        session_raw_data.append(raw_data)
        # number of trials.
        nTrials = raw_data['nTrials']
        # trial target
        trial_types = np.array(raw_data['TrialTypes'])
        # session date
        session_dates.append(fname[-19:-11])
        # loop over one session for extracting data
        trial_outcomes = []
        trial_outcomes_left = []
        trial_outcomes_right = []
        trial_gocue_start = []
        trial_choice_start = []
        trial_lick = []
        trial_reaction = []
        trial_decision = []
        trial_stim_start = []
        trial_stim_seq = []
        trial_isi_pre_perc = []
        trial_isi_pre_emp = []
        trial_isi_post_perc = []
        trial_isi_post_emp = []
        for i in range(nTrials):
            trial_states = raw_data['RawEvents']['Trial'][i]['States']
            trial_events = raw_data['RawEvents']['Trial'][i]['Events']
            # outcome
            outcome = states_labeling(trial_states)
            trial_outcomes.append(outcome)
            # left and right outcomes
            if trial_types[i] == 1:
                trial_outcomes_left.append(outcome)
            else:
                trial_outcomes_right.append(outcome)
            # stimulus start.
            if ('VisStimTrigger' in trial_states.keys() and
                not np.isnan(trial_states['VisStimTrigger'][0])):
                trial_stim_start.append(np.array([1000*np.array(trial_states['VisStimTrigger'][1])]))
            else:
                trial_stim_start.append(np.array([np.nan]))
            # stimulus sequence.
            # if photodiode high/low toggle and same number of high/low and at least 2 highs/lows
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
            pre_isi_perc = np.mean(stim_seq[0,1:3] - stim_seq[1,0:2])
            # pre_isi_emp = 1000*np.mean(raw_data['ProcessedSessionData'][i]['trial_isi']['PreISI'])
            pre_isi_emp = np.float64(0)  # no pre isi for single interval            
            trial_isi_pre_perc.append(pre_isi_perc)
            trial_isi_pre_emp.append(pre_isi_emp)
            # post perturbation isi.
            if stim_seq.shape[1]<4:
                isi_post_perc = np.nan
            else:
                isi_post_perc = np.mean(stim_seq[0,3:] - stim_seq[1,2:-1])
            isi_post_emp = 1000*np.mean(raw_data['ProcessedSessionData'][i]['trial_isi']['PostISI'])
            trial_isi_post_perc.append(isi_post_perc)
            trial_isi_post_emp.append(isi_post_emp)
            # go cue
            if ('GoCue' in trial_states.keys() and
                not np.isnan(trial_states['GoCue'][0])):
                trial_gocue_start.append(trial_states['GoCue'][0])
            else:
                trial_gocue_start.append(np.nan)
            # choice window
            if ('WindowChoice' in trial_states.keys() and
                not np.isnan(trial_states['WindowChoice'][0])):
                trial_choice_start.append(trial_states['WindowChoice'][0])
            else:
                trial_choice_start.append(np.nan)                                
            # lick events.
            if ('VisStimTrigger' in trial_states.keys() and
                not np.isnan(trial_states['VisStimTrigger'][1])):
                licking_events = []
                direction = []
                correctness = []
                if 'Port1In' in trial_events.keys():
                    lick_left = np.array(trial_events['Port1In']).reshape(-1)
                    licking_events.append(lick_left)
                    direction.append(np.zeros_like(lick_left))
                    if trial_types[i] == 1:
                        correctness.append(np.ones_like(lick_left))
                    else:
                        correctness.append(np.zeros_like(lick_left))
                if 'Port3In' in trial_events.keys():
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
                    # all licking events.
                    trial_lick.append(lick)
                    # reaction licking. ie. licks after start of vis stim
                    reaction_idx = np.where(lick[0]>1000*trial_states['VisStimTrigger'][1])[0]
                    if len(reaction_idx)>0:
                        lick_reaction = lick.copy()[:, reaction_idx[0]].reshape(3,1)
                        trial_reaction.append(lick_reaction)
                    else:
                        trial_reaction.append(np.array([[np.nan], [np.nan], [np.nan]]))
                    # effective licking to outcome. ie. licks after start of window choice
                    decision_idx = np.where(lick[0]>1000*trial_states['WindowChoice'][1])[0]
                    if (len(decision_idx)>0 and
                        stim_seq.shape[1]>=2
                        ):
                        lick_decision = lick.copy()[:, decision_idx[0]].reshape(3,1)
                        trial_decision.append(lick_decision)
                    else:
                        trial_decision.append(np.array([[np.nan], [np.nan], [np.nan]]))
                    # left and right outcomes
                    
                    
                    # if outcome == 'RewardNaive' or 
                    # outcome == 'Reward' or
                    # outcome == 'ChangingMindReward':
                    #     # left side 
                    #     if trial_types[i] == 1:
                            
                    #     # right side
                    #     else:
                    #     trial_side_outcomes
                else:
                    trial_lick.append(np.array([[np.nan], [np.nan], [np.nan]]))
                    trial_reaction.append(np.array([[np.nan], [np.nan], [np.nan]]))
                    trial_decision.append(np.array([[np.nan], [np.nan], [np.nan]]))
            else:
                trial_lick.append(np.array([[np.nan], [np.nan], [np.nan]]))
                trial_reaction.append(np.array([[np.nan], [np.nan], [np.nan]]))
                trial_decision.append(np.array([[np.nan], [np.nan], [np.nan]]))
        # save one session data
        session_outcomes.append(trial_outcomes)
        session_outcomes_left.append(trial_outcomes_left)
        session_outcomes_right.append(trial_outcomes_right)
        session_gocue_start.append(trial_gocue_start)
        session_choice_start.append(trial_choice_start)        
        session_lick.append(trial_lick)
        session_reaction.append(trial_reaction)
        session_decision.append(trial_decision)
        session_stim_start.append(trial_stim_start)
        session_stim_seq.append(trial_stim_seq)
        session_isi_pre_perc.append(trial_isi_pre_perc)
        session_isi_pre_emp.append(trial_isi_pre_emp)
        session_isi_post_perc.append(trial_isi_post_perc)
        session_isi_post_emp.append(trial_isi_post_emp)
        session_jitter_flag.append(trial_jitter_flag)
    # merge all session data
    data = {
        'total_sessions' : len(session_outcomes),
        'subject' : subject,
        'filename' : file_names,
        'raw' : session_raw_data,
        'dates' : session_dates,
        'outcomes' : session_outcomes,
        'outcomes_left' : session_outcomes_left,
        'outcomes_right' : session_outcomes_right,
        'gocue_start' : session_gocue_start,
        'choice_start' : session_choice_start,
        'lick' : session_lick,
        'reaction' : session_reaction,
        'decision' : session_decision,
        'stim_start' : session_stim_start,
        'stim_seq' : session_stim_seq,
        'isi_pre_perc' : session_isi_pre_perc,
        'isi_pre_emp' : session_isi_pre_emp,
        'isi_post_perc' : session_isi_post_perc,
        'isi_post_emp' : session_isi_post_emp,
        'jitter_flag' : session_jitter_flag,
    }
    return data


# labeling every trials for a subject
def states_labeling(trial_states):
    if 'ChangingMindReward' in trial_states.keys() and not np.isnan(trial_states['ChangingMindReward'][0]):
        outcome = 'ChangingMindReward'
    elif 'WrongInitiation' in trial_states.keys() and not np.isnan(trial_states['WrongInitiation'][0]):
        outcome = 'WrongInitiation'
    elif 'Punish' in trial_states.keys() and not np.isnan(trial_states['Punish'][0]):
        outcome = 'Punish'
    elif 'Reward' in trial_states.keys() and not np.isnan(trial_states['Reward'][0]):
        outcome = 'Reward'
    elif 'PunishNaive' in trial_states.keys() and not np.isnan(trial_states['PunishNaive'][0]):
        outcome = 'PunishNaive'
    elif 'RewardNaive' in trial_states.keys() and not np.isnan(trial_states['RewardNaive'][0]):
        outcome = 'RewardNaive'
    elif 'EarlyChoice' in trial_states.keys() and not np.isnan(trial_states['EarlyChoice'][0]):
        outcome = 'EarlyChoice'
    elif 'DidNotChoose' in trial_states.keys() and not np.isnan(trial_states['DidNotChoose'][0]):
        outcome = 'DidNotChoose'
    else:
        outcome = 'Other'
    return outcome


# read session data given subjects
def run(subject_list):
    session_data = []
    for sub in subject_list:
        print('reading data for ' + sub)
        session_data.append(read_trials(sub))
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    return session_data

