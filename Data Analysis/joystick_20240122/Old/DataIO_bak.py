import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import time

session_data_path = 'C:\\behavior\\session_data' 
# session_data_path = '.\\session_data'  # code test dir



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
    session_encoder_data = []
    encoder_time_max = 30
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
    session_press_reps = 0
    session_press_window = 0
    session_target_thresh = []
    VisStim2Enable = 1
    
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
        trial_encoder_times_aligned_vis1 = []
        trial_encoder_positions_aligned_vis2 = []
        trial_encoder_times_aligned_vis2 = []
        trial_encoder_positions_aligned_rew = []
        trial_num_rewarded = []
        trial_reps = 0
        
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
        
        for i in range(nTrials):
            trial_states = raw_data['RawEvents']['Trial'][i]['States']
            
        for i in range(nTrials):
        # for i in range(119, 123):
            # handle key error if only one trial in a session
            if nTrials > 1:
                trial_states = raw_data['RawEvents']['Trial'][i]['States']
                trial_events = raw_data['RawEvents']['Trial'][i]['Events']                
                trial_reps = raw_data['TrialSettings'][i]['GUI']['Reps']
                session_press_window = raw_data['TrialSettings'][i]['GUI']['PressWindow_s']
            else:
                continue
                # trial_states = raw_data['RawEvents']['Trial']['States']
                # trial_events = raw_data['RawEvents']['Trial']['Events']            
            # outcome
            outcome = states_labeling(trial_states, trial_reps)
            trial_outcomes.append(outcome)
            # iti duration
            iti = trial_states['ITI']
            trial_iti.append(iti[1]-iti[0])
            # licking events after stim onset
            if ('VisStimTrigger' in trial_states.keys() and
                not np.isnan(trial_states['VisStimTrigger'][1])):
                stim_start = trial_states['VisStimTrigger'][1]
                licking_events = []
                direction = []
                correctness = []
                if 'Port1In' in trial_events.keys():
                    lick_left = np.array(trial_events['Port1In']).reshape(-1) - stim_start
                    licking_events.append(lick_left)
                    direction.append(np.zeros_like(lick_left))
                    if TrialTypes[i] == 1:
                        correctness.append(np.ones_like(lick_left))
                    else:
                        correctness.append(np.zeros_like(lick_left))
                if 'Port3In' in trial_events.keys():
                    lick_right = np.array(trial_events['Port3In']).reshape(-1) - stim_start
                    licking_events.append(lick_right)
                    direction.append(np.ones_like(lick_right))
                    if TrialTypes[i] == 2:
                        correctness.append(np.ones_like(lick_right))
                    else:
                        correctness.append(np.zeros_like(lick_right))
                if len(licking_events) > 0:
                    licking_events = np.concatenate(licking_events).reshape(1,-1)
                    correctness = np.concatenate(correctness).reshape(1,-1)
                    direction = np.concatenate(direction).reshape(1,-1)
                    lick = np.concatenate([licking_events, correctness, direction])
                    trial_licking.append(lick)
            # reaction time
                    lick = lick[:,lick[0,:]>0]
                    if lick.shape[1] > 0:
                        reaction_idx = np.argmin(lick[0,:])
                        trial_reaction.append(lick[:,reaction_idx].reshape(1,-1))
            # stim isi
            if ('BNC1High' in trial_events.keys() and
                'BNC1Low' in trial_events.keys()):
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
                            'Punish',
                            'ChangingMindReward']:
                            if outcome in [
                                'Reward', 
                                ]:
                                choice = TrialTypes[i]-1
                            if outcome in [
                                'Punish',
                                'ChangingMindReward'
                                ]:
                                choice = 2-TrialTypes[i]
                            # 0-left / 1-right
                            trial_choice.append(np.array([BNC1HighLow[-1], choice]))
            # change of mind choice 0:right2left 1:left2right
                        if outcome in [
                            'ChangingMindReward'
                            ]:
                            trial_com.append(TrialTypes[i]-1)
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
           
            # encoder data                
            encoder_data = raw_data['EncoderData'][i]
            trial_encoder_data.append(encoder_data)
            encoder_data_aligned = {}
          
            # interpolate encoder data to align times for averaging
            # maybe only use one array of linspace for efficiency
            # trial_reps = raw_data['TrialSettings'][i]['GUI']['Reps']
            trial_target_thresh = raw_data['TrialSettings'][i]['GUI']['Threshold']
            # if (fname == 'YH5_Joystick_visual_7_20240209_150957.mat'):
            #     print()
            if encoder_data['nPositions']:                
                times = encoder_data['Times']
                positions = encoder_data['Positions']
                
                # all trials encoder positions
                # encoder_data_aligned = np.interp(session_encoder_times_aligned, times, positions)
                # all trials encoder data
                # encoder_data_aligned = {'Positions': encoder_data_aligned}
                # process encoder data for rewarded trials
                if outcome == 'Reward':
                    encoder_data_aligned = np.interp(session_encoder_times_aligned, times, positions)
                    
                    trial_encoder_positions_aligned.append(encoder_data_aligned)
                    
                    # find times and pos aligned to vis stim 1
                    VisStim1Start = trial_states['VisualStimulus1'][0]  
                    
                    if VisStim1Start > 12:
                        continue # vis detect missed stim, go to next trial
                        
                    vis_diff = np.abs(VisStim1Start - session_encoder_times_aligned)
                    min_vis_diff = np.min(np.abs(VisStim1Start - session_encoder_times_aligned))
                    closest_aligned_time_vis1_idx = [ind for ind, ele in enumerate(vis_diff) if ele == min_vis_diff][0]
                    
                    # LeverRetract1Stop = trial_states['LeverRetract1'][1]
                    # retract1_diff = np.abs(LeverRetract1Stop - session_encoder_times_aligned)
                    # min_retract1_diff = np.min(np.abs(LeverRetract1Stop - session_encoder_times_aligned))
                    # closest_aligned_time_retract1_idx = [ind for ind, ele in enumerate(retract1_diff) if ele == min_retract1_diff][0]
                    
                    left_idx_VisStim1 = int(closest_aligned_time_vis1_idx+time_left_VisStim1*ms_per_s)
                    right_idx_VisStim1 = int(closest_aligned_time_vis1_idx+(time_right_VisStim1*ms_per_s))
                    # pad with nan if left idx < 0
                    if left_idx_VisStim1 < 0:
                        nan_pad = np.zeros(-left_idx_VisStim1)
                        nan_pad[:] = np.nan
                        trial_encoder_positions_aligned_VisStim1 = np.append(nan_pad, encoder_data_aligned[0:right_idx_VisStim1])
                    else:                        
                        # trial_encoder_times_aligned_VisStim1 = session_encoder_times_aligned[left_idx_VisStim1:right_idx_VisStim1]                    
                        # trial_encoder_positions_aligned_VisStim1 = encoder_data_aligned[left_idx_VisStim1:closest_aligned_time_retract1_idx]
                        # zeros_needed = len(trial_encoder_times_aligned_VisStim1) - len(trial_encoder_positions_aligned_VisStim1)
                        # trial_encoder_positions_aligned_VisStim1 = np.append(trial_encoder_positions_aligned_VisStim1, np.zeros(zeros_needed))
                        trial_encoder_positions_aligned_VisStim1 = encoder_data_aligned[left_idx_VisStim1:right_idx_VisStim1]
                    
                    trial_encoder_positions_aligned_vis1.append(trial_encoder_positions_aligned_VisStim1)
                    
                    # plt.plot(trial_encoder_times_aligned_VisStim1, trial_encoder_positions_aligned_VisStim1)
                    # plt.plot(session_encoder_times_aligned_VisStim1, trial_encoder_positions_aligned_VisStim1)
                    
                    
                    # find times and pos aligned to vis stim 2
                    if "VisStim2Enable" in raw_data['TrialSettings'][i]['GUI']:
                        VisStim2Enable = raw_data['TrialSettings'][i]['GUI']['VisStim2Enable']
                    
                    # !!update this later to account more mixed trials in a given session with two array averages for align
                    # OR align at WaitForPress 
                    if VisStim2Enable:
                        VisStim2Start = trial_states['VisualStimulus2'][0]                        
                    else:
                        VisStim2Start = trial_states['WaitForPress2'][0]
                        
                    if trial_reps > 1:
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
                    else:
                        trial_encoder_positions_aligned_VisStim2 = np.zeros(session_encoder_times_aligned_VisStim2.size)
                        trial_encoder_positions_aligned_VisStim2[:] = np.nan
                        trial_encoder_positions_aligned_vis2.append(trial_encoder_positions_aligned_VisStim2)
                            
                    
                    # plt.plot(session_encoder_times_aligned[0:5000], encoder_data_aligned[0:5000])
                    #plt.plot(session_encoder_times_aligned_VisStim2, trial_encoder_positions_aligned_VisStim2)
                    
                    # find times and pos aligned to reward
                    if trial_reps == 3:
                        RewardStart = trial_states['Reward3'][0]
                    elif trial_reps == 2:
                        RewardStart = trial_states['Reward2'][0]
                    elif trial_reps == 1:
                        if 'Reward' in trial_states.keys() and not np.isnan(trial_states['Reward'][0]):
                            RewardStart = trial_states['Reward'][0]
                        else:
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
                        # trial_encoder_times_aligned_VisStim2 = session_encoder_times_aligned[left_idx_VisStim2:right_idx_VisStim2]
                        trial_encoder_positions_aligned_Reward = encoder_data_aligned[left_idx_rew:right_idx_rew]
                    
                    # if i == 121:
                    #     print()
                    trial_encoder_positions_aligned_rew.append(trial_encoder_positions_aligned_Reward)
                    
                    
                    # index of rewarded trials
                    trial_num_rewarded.append(i)
                                        
                    # plt.plot(session_encoder_times_aligned[0:5000], encoder_data_aligned[0:5000])
                    #plt.plot(session_encoder_times_aligned_Reward, trial_encoder_positions_aligned_Reward)
                    
                    #indices = [ind for ind, ele in enumerate(session_encoder_times_aligned) if ele == 1]
           # trial_encoder_positions_aligned.append(encoder_data_aligned)
            
            # plt.plot(trial_encoder_times_aligned_VisStim1,trial_encoder_positions_aligned_VisStim1, 'o')
            # plt.plot(trial_encoder_times_aligned_VisStim2,trial_encoder_positions_aligned_VisStim2, 'o')
            # plt.plot(times,positions, 'o')
            # plt.plot(session_encoder_times_aligned,encoder_data_aligned, 'o')  
            # plt.plot(session_encoder_times_aligned,sess_enc_avg, 'o') 
        
        # encoder trajectory average across session for rewarded trials
        # if i == 251:
        #     print()
        # pos = np.sum(trial_encoder_positions_aligned[0:], axis=0)        
        # sess_enc_avg = pos/len(trial_encoder_positions_aligned)
        
        # update trial_encoder_positions_aligned_VisStim1 and 2 above for common time vector
        # session_encoder_times_aligned_VisStim1
        # session_encoder_times_aligned_VisStim2
        
        # plt.plot(session_encoder_times_aligned_VisStim2, trial_encoder_positions_aligned_vis2[0], label='trial 1')
        # plt.plot(session_encoder_times_aligned_VisStim2, trial_encoder_positions_aligned_vis2[1], label='trial 2')
        
        # encoder trajectory average across session for rewarded trials, vis stim 1 aligned
        
        try:
            pos_vis1 = np.sum(trial_encoder_positions_aligned_vis1[0:], axis=0)
        except:
            print(fname)
            time.sleep(15)
        
        
        pos_vis1 = np.sum(trial_encoder_positions_aligned_vis1[0:], axis=0)        
        sess_enc_avg_vis1 = pos_vis1/len(trial_num_rewarded)
        
        
        if 0:
        
            for i in range(len(trial_encoder_positions_aligned[0:4])):
                plt.plot(session_encoder_times_aligned_VisStim1,trial_encoder_positions_aligned_vis1[i], label=i)
                plt.legend(loc='upper right')
                # plt.show()
        
                plt.plot(session_encoder_times_aligned_VisStim1,sess_enc_avg_vis1)
        
        # encoder trajectory average across session for rewarded trials, vis stim 2 aligned
        pos_vis2 = np.sum(trial_encoder_positions_aligned_vis2, axis=0)        
        sess_enc_avg_vis2 = pos_vis2/len(trial_num_rewarded)
        
        if 0:
            plt.plot(session_encoder_times_aligned_VisStim2, sess_enc_avg_vis2)
        
        
            for i in range(10):
                plt.plot(session_encoder_times_aligned_VisStim2,trial_encoder_positions_aligned_vis2[i], label=i)
                plt.legend(loc='upper right')
                # plt.show()
        
        # [0:5000]
              
            for i in range(len(trial_encoder_positions_aligned[0:4])):
                plt.plot(session_encoder_times_aligned,trial_encoder_positions_aligned[i], label=i)
                plt.legend(loc='upper right')
                # plt.show()
        
       
            for i in range(len(trial_encoder_positions_aligned[0:4])):
                plt.plot(session_encoder_times_aligned[0:7000],trial_encoder_positions_aligned[i][0:7000], label=i)
                plt.legend(loc='upper right')
        
        # try:
        #     pos_rew = np.sum(trial_encoder_positions_aligned_rew, axis=0)  
        # except:
        #     print(fname)
        #     time.sleep(15)
        pos_rew = np.sum(trial_encoder_positions_aligned_rew, axis=0)
        sess_enc_avg_rew = pos_rew/len(trial_num_rewarded)
        
        if 0:
            for i in range(len(trial_encoder_positions_aligned[0:4])):
                plt.plot(session_encoder_times_aligned_Reward,trial_encoder_positions_aligned_rew[i], label=i)
                plt.legend(loc='upper right')
            
            
            plt.plot(session_encoder_times_aligned_Reward, sess_enc_avg_rew)
        
        session_target_thresh = trial_target_thresh
        session_press_reps = trial_reps
        # session_encoder_times_aligned_VisStim1
        # session_encoder_times_aligned_VisStim2
        
        # save one session data
        session_encoder_data.append(trial_encoder_data)
        session_encoder_positions_aligned.append(trial_encoder_positions_aligned)
        session_encoder_positions_aligned_vis1.append(trial_encoder_positions_aligned_vis1)
        session_encoder_positions_aligned_vis2.append(trial_encoder_positions_aligned_vis2)
        session_encoder_positions_aligned_rew.append(trial_encoder_positions_aligned_rew)
        session_rewarded_trials.append(trial_num_rewarded)
        session_encoder_positions_avg_vis1.append(sess_enc_avg_vis1)
        session_encoder_positions_avg_vis2.append(sess_enc_avg_vis2)
        session_encoder_positions_avg_rew.append(sess_enc_avg_rew)
        
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
        'session_target_thresh' : session_target_thresh,
        'session_press_reps' : session_press_reps,
        'session_press_window' : session_press_window,
        'vis_stim_2_enable' : VisStim2Enable,
        'encoder_pos_avg_vis1' : session_encoder_positions_avg_vis1,        
        'encoder_pos_avg_vis2' : session_encoder_positions_avg_vis2,        
        'encoder_pos_avg_rew' : session_encoder_positions_avg_rew, 
        'com' : session_com,
        'post_lick' : session_post_lick,
        'isi' : session_isi,
        'avsync' : session_avsync,
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
    # elif 'Reward1' in trial_states.keys() and not np.isnan(trial_states['Reward1'][0]):
    #     outcome = 'Reward' 
    # elif 'Reward2' in trial_states.keys() and not np.isnan(trial_states['Reward2'][0]):
    #     outcome = 'Reward'  
    # elif 'Reward3' in trial_states.keys() and not np.isnan(trial_states['Reward3'][0]):
    #     outcome = 'Reward'          
    # else:
    #     outcome = 'Other'
        

        
    return outcome        


# get choice from outcomes and trial types
def compute_choice():
    0