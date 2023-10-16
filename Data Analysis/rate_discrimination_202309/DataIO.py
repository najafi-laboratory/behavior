import os
import scipy.io as sio
import numpy as np

session_data_path = '.\\session_data'

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
    session_post_lick = []
    session_choice = []
    session_dates = []
    session_outcomes = []
    session_reaction = []
    session_iti = []
    session_isi = []
    session_avsync = []
    LR12_start = 0
    for f in range(len(file_names)):
        # the first session with LR_12
        fname = file_names[f]
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
        trial_post_lick = []
        trial_outcomes = []
        trial_reaction = []
        trial_iti = []
        trial_isi = []
        trial_avsync = []
        trial_choice = []
        for i in range(nTrials):
            # handle key error if only one trial in a session
            if nTrials > 1:
                trial_states = raw_data['RawEvents']['Trial'][i]['States']
                trial_events = raw_data['RawEvents']['Trial'][i]['Events']
            else:
                trial_states = raw_data['RawEvents']['Trial']['States']
                trial_events = raw_data['RawEvents']['Trial']['Events']
            # outcome
            outcome = states_labeling(trial_states)
            trial_outcomes.append(outcome)
            # iti duration
            iti = trial_states['ITI']
            trial_iti.append(iti[1]-iti[0])
            # licking events after stim onset
            if 'VisStimTrigger' in trial_states.keys() and not np.isnan(trial_states['VisStimTrigger'][1]):
                stim_start = trial_states['VisStimTrigger'][1]
                licking_events = []
                direction = []
                correctness = []
                if 'Port1In' in trial_events.keys():
                    lick_left = np.array(trial_events['Port1In']).reshape(-1)
                    licking_events.append(lick_left)
                    direction.append(np.zeros_like(lick_left))
                    if TrialTypes[i] == 1:
                        correctness.append(np.ones_like(lick_left))
                    else:
                        correctness.append(np.zeros_like(lick_left))
                if 'Port3In' in trial_events.keys():
                    lick_right = np.array(trial_events['Port3In']).reshape(-1)
                    licking_events.append(lick_right)
                    direction.append(np.ones_like(lick_right))
                    if TrialTypes[i] == 2:
                        correctness.append(np.ones_like(lick_right))
                    else:
                        correctness.append(np.zeros_like(lick_right))
                if len(licking_events) > 0:
                    licking_events = np.concatenate(licking_events).reshape(1,-1) - stim_start
                    correctness = np.concatenate(correctness).reshape(1,-1)
                    direction = np.concatenate(direction).reshape(1,-1)
                    trial_reaction.append(
                        np.concatenate([licking_events, correctness, direction]))
            # stim isi
            if ('BNC1High' in trial_events.keys()) and ('BNC1Low' in trial_events.keys()):
                BNC1High = trial_events['BNC1High']
                BNC1High = np.array(BNC1High).reshape(-1)
                BNC1Low = trial_events['BNC1Low']
                BNC1Low = np.array(BNC1Low).reshape(-1)
                if len(BNC1High) > 1 and len(BNC1Low) > 1:
                    BNC1High = BNC1High[1:]
                    BNC1Low = BNC1Low[0:-1]
                    if np.size(BNC1High) == np.size(BNC1Low):
                        BNC1HighLow = BNC1High - BNC1Low
                        trial_isi.append(BNC1HighLow)
            # trial choice
                        if outcome in [
                            'Reward',
                            'RewardNaive',
                            'Punish',
                            'PunishNaive']:
                            if outcome in [
                                'Reward',
                                'RewardNaive']:
                                choice = TrialTypes[i]-1
                            if outcome in [
                                'Punish',
                                'PunishNaive']:
                                choice = 2-TrialTypes[i]
                            # 0-left / 1-right
                            trial_choice.append(np.array([BNC1HighLow[-1], choice]))
            # post-perturbation licking events
                        post_start_idx = np.where(
                            np.abs(BNC1HighLow - BNC1HighLow[-1]) <= 0.05)[0]
                        if (len(post_start_idx) > 0 and
                            BNC1HighLow[-1] < 1.1 and
                            BNC1HighLow[-1] > -0.1
                            ):
                            # early choice included
                            post_start_idx = post_start_idx[0]
                            lick_left = np.array([])
                            lick_right = np.array([])
                            if 'Port1In' in trial_events.keys():
                                lick_left = np.array(trial_events['Port1In']).reshape(-1)
                                if len(lick_left) > 0:
                                    lick_left = lick_left[post_start_idx:]
                            if 'Port3In' in trial_events.keys():
                                lick_right = np.array(trial_events['Port3In']).reshape(-1)
                                if len(lick_left) > 0:
                                    lick_right = lick_right[post_start_idx:]
                            right_pc = len(lick_right)/(len(lick_left)+len(lick_right)+1e-5)
                            trial_post_lick.append(np.array([BNC1HighLow[-1], right_pc]))
            # av sync
            if ('BNC1High' in trial_events.keys()) and ('BNC2High' in trial_events.keys()):
                BNC1High = trial_events['BNC1High']
                BNC1High = np.array(BNC1High).reshape(-1)
                BNC2High = trial_events['BNC2High']
                BNC2High = np.array(BNC2High).reshape(-1)
                FirstGrating = BNC1High[0]
                FirstGratingAudioDiff = FirstGrating - BNC2High
                FirstGratingAudioDiff_abs = abs(FirstGratingAudioDiff)
                avsync_abs = min(FirstGratingAudioDiff_abs).tolist()
                AudioStimStartIndex = FirstGratingAudioDiff_abs.tolist().index(avsync_abs)
                avsync = (FirstGrating - BNC2High)[AudioStimStartIndex]
                trial_avsync.append(avsync.tolist())
        # save one session data
        session_choice.append(trial_choice)
        session_post_lick.append(trial_post_lick)
        session_outcomes.append(trial_outcomes)
        session_iti.append(trial_iti)
        session_reaction.append(trial_reaction)
        session_isi.append(trial_isi)
        session_avsync.append(trial_avsync)
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
        'choice' : session_choice,
        'post_lick' : session_post_lick,
        'isi' : session_isi,
        'avsync' : session_avsync,
    }
    return data


# labeling every trials for a subject
def states_labeling(trial_states):
    if 'Reward' in trial_states.keys() and not np.isnan(trial_states['Reward'][0]):
        outcome = 'Reward'
    elif 'Punish' in trial_states.keys() and not np.isnan(trial_states['Punish'][0]):
        outcome = 'Punish'
    elif 'ExtraStimDurPostRew_Naive' in trial_states.keys() and not np.isnan(trial_states['ExtraStimDurPostRew_Naive'][0]):
        outcome = 'RewardNaive'
    elif 'PunishNaive' in trial_states.keys() and not np.isnan(trial_states['PunishNaive'][0]):
        outcome = 'PunishNaive'
    elif 'WrongInitiation' in trial_states.keys() and not np.isnan(trial_states['WrongInitiation'][0]):
        outcome = 'WrongInitiation'
    elif 'EarlyChoice' in trial_states.keys() and not np.isnan(trial_states['EarlyChoice'][0]):
        outcome = 'EarlyChoice'
    elif 'EarlyChoice_1' in trial_states.keys() and not np.isnan(trial_states['EarlyChoice_1'][0]):
        outcome = 'EarlyChoice'
    elif 'EarlyChoiceDurCenterLick' in trial_states.keys() and not np.isnan(trial_states['EarlyChoiceDurCenterLick'][0]):
        outcome = 'EarlyChoice'
    elif 'DidNotChoose' in trial_states.keys() and not np.isnan(trial_states['DidNotChoose'][0]):
        outcome = 'DidNotChoose'
    elif 'DidNotConfirm' in trial_states.keys() and not np.isnan(trial_states['DidNotConfirm'][0]):
        outcome = 'DidNotConfirm'
    elif 'DidNotLickCenter' in trial_states.keys() and not np.isnan(trial_states['DidNotLickCenter'][0]):
        outcome = 'DidNotLickCenter'
    else:
        outcome = 'Other'
    return outcome


# get choice from outcomes and trial types
def compute_choice():
    0