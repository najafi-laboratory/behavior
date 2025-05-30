import numpy as np
from labcams import parse_cam_log, unpackbits
import matplotlib.pyplot as plt
# import re
import pandas as pd

def parse_log_comments(comm,msg = 'trial_start:',strkeyname = 'i_trial',strformat=int):
    # this will get the start frames for each trial
    # comm = map(lambda x: x.strip('#').replace(' ',''),comm)
    # commmsg = list(filter(lambda x:msg in x ,comm))
    table = []
    pulse_frames = comm
    for i,lg in enumerate(pulse_frames):
        lg = re.split(',|-|'+msg,lg)
        table.append({strkeyname:strformat(lg[-1]),
                         'iframe':int(lg[0]),
                         'timestamp':float(lg[1])})
    return pd.DataFrame(table)

def run(ax, subject_session_data):

    # files = ['/home/joao/Desktop/_cam0_run004_20231209_193033.camlog',
    #         '/home/joao/Desktop/_cam1_run004_20231209_193032.camlog']
    
    fname = 'D:\\PHD\\Projects\\Interval Discrimination\\Single Interval Discrimination\\data\\videos\\LCHR_TS01\\20250109\\LCHR_TS01_2afc_20250109_cam0_run007_20250106_184048.camlog'
    # fname = 'D:\\PHD\\Projects\\Interval Discrimination\\Single Interval Discrimination\\data\\videos\\LCHR_TS01\\20250114\\LCHR_TS01_2afc_20250114_cam0_run000_20250114_174750.camlog'
    # fname = 'D:\\PHD\\Projects\\Interval Discrimination\\Single Interval Discrimination\\data\\videos\\LCHR_TS01\\20250112\\LCHR_TS01_2afc_20250112_cam0_run001_20250112_194911.camlog'
    
    logdata,comments = parse_cam_log(fname)
    
    session_start_time = logdata['timestamp'][0]
    logdata['timestamp'] = logdata['timestamp'] - session_start_time
    
    # unpacked_binary = unpackbits(logdata['var2'].values,output_binary=True) # this is just for plotting
    
    # %matplotlib notebook
    # import pylab as plt
    # import numpy as np
    
    # plt.plot(unpacked_binary.T+np.arange(unpacked_binary.shape[0]))
    
    
    onsets,offsets = unpackbits(logdata['var2'].values)
    
    # [(k,len(onsets[k])) for k in onsets.keys()]
    # for k in onsets.keys():
    #     plt.plot(onsets[k],np.ones_like(onsets[k])*int(k)+1,'ko',markerfacecolor = 'none')
    
    # get the start frame id for each trial (assuming trials on gpio port 2)
    frames_where_the_pulse_was_first_seen = logdata.frame_id[onsets[2]].values   
    frames_before_next_trial_start = frames_where_the_pulse_was_first_seen[1:].copy()
    # get frames where sync signal first registers on gpio port
    frames_where_the_pulse_was_first_seen +=1
    # convert to list for enumeration
    frames_where_the_pulse_was_first_seen = frames_where_the_pulse_was_first_seen.tolist()
    frames_before_next_trial_start = frames_before_next_trial_start.tolist()
    
    # append last frame of session
    frames_before_next_trial_start.append(int(logdata.shape[0]-1))
        
    # log,comm = parse_cam_log(fname)
    # comm[7] = 'trial_start:'.join(map(str, frames_where_the_pulse_was_first_seen))
    # comm.append('trial_start:'.join(map(str, frames_where_the_pulse_was_first_seen)))
    # start_trials = parse_log_comments(comm)
    # start_trials = parse_log_comments(frames_where_the_pulse_was_first_seen)
    # end_trials = parse_log_comments(comm,'trial_end:')
    # merge start and end 
    # start_trials = start_trials.rename(columns={'iframe':'start_frame','timestamp':'grating_start_timestamp'})
    # end_trials = end_trials.rename(columns={'iframe':'end_frame','timestamp':'end_timestamp'})
    # trialinfo = pd.merge(start_trials,end_trials)
    # trialinfo   
    
    # build table of start/stop frames for each trial
    table = []
    strkeyname = 'i_trial'
    strformat=int
    for i,log_idx in enumerate(frames_where_the_pulse_was_first_seen):    
        table.append({strkeyname:strformat(i+1),
                         'iframe':int(logdata.loc[log_idx][0]),
                         'timestamp':float(logdata.loc[log_idx][1])})    
    start_trials = pd.DataFrame(table)
    start_trials = start_trials.rename(columns={'iframe':'start_frame','timestamp':'grating_start_timestamp'})
    
    table = []
    strkeyname = 'i_trial'
    strformat=int
    for i,log_idx in enumerate(frames_before_next_trial_start):     
        table.append({strkeyname:strformat(i+1),
                         'iframe':int(logdata.loc[log_idx][0]),
                         'timestamp':float(logdata.loc[log_idx][1])})    
    end_trials = pd.DataFrame(table)    
    end_trials = end_trials.rename(columns={'iframe':'end_frame','timestamp':'end_timestamp'})  
    session_video_info = pd.merge(start_trials,end_trials)
    session_video_info
    
    eye_labeled_csv_path = 'D:\\PHD\\Projects\\Single Interval Discrimination\\Single_Interval_Discrimination-Tim-2025-01-11\\videos\\LCHR_TS01_2afc_20250109_cam0_run007_20250106_184048DLC_resnet50_Single_Interval_DiscriminationJan11shuffle1_100000_area_per_frame.csv'
    # eye_labeled_csv_path = 'D:\\PHD\\Projects\\Single Interval Discrimination\\Single_Interval_Discrimination-Tim-2025-01-11\\videos\\LCHR_TS01_2afc_20250114_cam0_run000_20250114_174750croppedDLC_resnet50_Single_Interval_DiscriminationJan11shuffle1_100000_area_per_frame.csv'
    # eye_labeled_csv_path = 'D:\\PHD\\Projects\\Single Interval Discrimination\\Single_Interval_Discrimination-Tim-2025-01-11\\videos\\LCHR_TS01_2afc_20250112_cam0_run001_20250112_194911croppedDLC_resnet50_Single_Interval_DiscriminationJan11shuffle1_100000_area_per_frame.csv'
    
    # for now, assume starting indices of pupil area aligns with camlog
    eye_df = pd.read_csv(eye_labeled_csv_path)    
    
    nTrials = 309  # 1-09-25
    # nTrials = 325  # 1-14-25
    # nTrials = 301 # 1-12-25
    img_time = 0.0166
    
    trial_time_baseline = []
    left_trials = []
    right_trials = []
    
    # all trials
    session_pupil_area = []
    session_trial_time_baseline = []
        
    session_trial_time_stim_start_aligned_baseline = []
    session_trial_time_grating_2_start_aligned_baseline = []
    
    session_pupil_area_stim_start_aligned = []
    session_grating_start_times_grating_1_aligned = []
    session_grating_stop_times_grating_1_aligned = []
         
    session_pupil_area_grating_2_start_aligned = []
    session_grating_start_times_grating_2_aligned = []
    session_grating_stop_times_grating_2_aligned = []    

    # left trials
    session_pupil_area_left = []
    
    session_pupil_area_stim_start_aligned_left = []
    session_grating_start_times_grating_1_aligned_left = []
    session_grating_stop_times_grating_1_aligned_left = []
         
    session_pupil_area_grating_2_start_aligned_left = []
    session_grating_start_times_grating_2_aligned_left = []
    session_grating_stop_times_grating_2_aligned_left = []  
    
    # left trials rewarded
    session_pupil_area_left_rewarded = []
    
    session_pupil_area_stim_start_aligned_left_rewarded = []
    session_grating_start_times_grating_1_aligned_left_rewarded = []
    session_grating_stop_times_grating_1_aligned_left_rewarded = []
         
    session_pupil_area_grating_2_start_aligned_left_rewarded = []
    session_grating_start_times_grating_2_aligned_left_rewarded = []
    session_grating_stop_times_grating_2_aligned_left_rewarded = []
    
    # left trials punished
    session_pupil_area_left_punished = []
    
    session_pupil_area_stim_start_aligned_left_punished = []
    session_grating_start_times_grating_1_aligned_left_punished = []
    session_grating_stop_times_grating_1_aligned_left_punished = []
         
    session_pupil_area_grating_2_start_aligned_left_punished = []
    session_grating_start_times_grating_2_aligned_left_punished = []
    session_grating_stop_times_grating_2_aligned_left_punished = []    
    
    # right trials
    session_pupil_area_right = []    
    
    session_pupil_area_stim_start_aligned_right = []
    session_grating_start_times_grating_1_aligned_right = []
    session_grating_stop_times_grating_1_aligned_right = []
         
    session_pupil_area_grating_2_start_aligned_right = []
    session_grating_start_times_grating_2_aligned_right = []
    session_grating_stop_times_grating_2_aligned_right = []       
    
    # right trials rewarded
    session_pupil_area_right_rewarded = []    
    
    session_pupil_area_stim_start_aligned_right_rewarded = []
    session_grating_start_times_grating_1_aligned_right_rewarded = []
    session_grating_stop_times_grating_1_aligned_right_rewarded = []
         
    session_pupil_area_grating_2_start_aligned_right_rewarded = []
    session_grating_start_times_grating_2_aligned_right_rewarded = []
    session_grating_stop_times_grating_2_aligned_right_rewarded = []   
    
    # right trials punished
    session_pupil_area_right_punished = []    
    
    session_pupil_area_stim_start_aligned_right_punished = []
    session_grating_start_times_grating_1_aligned_right_punished = []
    session_grating_stop_times_grating_1_aligned_right_punished = []
         
    session_pupil_area_grating_2_start_aligned_right_punished = []
    session_grating_start_times_grating_2_aligned_right_punished = []
    session_grating_stop_times_grating_2_aligned_right_punished = []   
    
    
    # all trials
    trial_pupil_area = []
    trial_grating_start_times = []
    trial_grating_stop_times = []
            
    trial_pupil_area_stim_start_aligned = []
    trial_grating_start_times_grating_1_aligned = []
    trial_grating_stop_times_grating_1_aligned = []
         
    trial_grating_start_times_grating_2_aligned = []
    trial_pupil_area_grating_2_start_aligned = []
    trial_grating_start_times_grating_2_aligned = []
    trial_grating_stop_times_grating_2_aligned = [] 
    
    # left trials
    trial_pupil_area_left = []
    
    trial_pupil_area_stim_start_aligned_left = []
    trial_grating_start_times_grating_1_aligned_left = []
    trial_grating_stop_times_grating_1_aligned_left = []
         
    trial_pupil_area_grating_2_start_aligned_left = []
    trial_grating_start_times_grating_2_aligned_left = []
    trial_grating_stop_times_grating_2_aligned_left = []  
    
    # left trials rewarded
    trial_pupil_area_left_rewarded = []
    
    trial_pupil_area_stim_start_aligned_left_rewarded = []
    trial_grating_start_times_grating_1_aligned_left_rewarded = []
    trial_grating_stop_times_grating_1_aligned_left_rewarded = []
         
    trial_pupil_area_grating_2_start_aligned_left_rewarded = []
    trial_grating_start_times_grating_2_aligned_left_rewarded = []
    trial_grating_stop_times_grating_2_aligned_left_rewarded = []
    
    # left trials punished
    trial_pupil_area_left_punished = []
    
    trial_pupil_area_stim_start_aligned_left_punished = []
    trial_grating_start_times_grating_1_aligned_left_punished = []
    trial_grating_stop_times_grating_1_aligned_left_punished = []
         
    trial_pupil_area_grating_2_start_aligned_left_punished = []
    trial_grating_start_times_grating_2_aligned_left_punished = []
    trial_grating_stop_times_grating_2_aligned_left_punished = []          
    
    # right trials
    trial_pupil_area_right = []
    
    trial_pupil_area_stim_start_aligned_right = []
    trial_grating_start_times_grating_1_aligned_right = []
    trial_grating_stop_times_grating_1_aligned_right = []
         
    trial_pupil_area_grating_2_start_aligned_right = []
    trial_grating_start_times_grating_2_aligned_right = []
    trial_grating_stop_times_grating_2_aligned_right = []    
    
    # right trials rewarded
    trial_pupil_area_right_rewarded = []    
    
    trial_pupil_area_stim_start_aligned_right_rewarded = []
    trial_grating_start_times_grating_1_aligned_right_rewarded = []
    trial_grating_stop_times_grating_1_aligned_right_rewarded = []
         
    trial_pupil_area_grating_2_start_aligned_right_rewarded = []
    trial_grating_start_times_grating_2_aligned_right_rewarded = []
    trial_grating_stop_times_grating_2_aligned_right_rewarded = []   
    
    # right trials punished
    trial_pupil_area_right_punished = []    
    
    trial_pupil_area_stim_start_aligned_right_punished = []
    trial_grating_start_times_grating_1_aligned_right_punished = []
    trial_grating_stop_times_grating_1_aligned_right_punished = []
         
    trial_pupil_area_grating_2_start_aligned_right_punished = []
    trial_grating_start_times_grating_2_aligned_right_punished = []
    trial_grating_stop_times_grating_2_aligned_right_punished = []        
    
    for i in range(0,nTrials):
    # for i in range(0,4):
        print(i)       
        
        rewarded = False
        
        trial_info = session_video_info[session_video_info['i_trial'] == i+1]
        trial_start_frame = int(trial_info['start_frame'].iloc[0])
        trial_end_frame = int(trial_info['end_frame'].iloc[0])
        trial_logdata = logdata.iloc[trial_start_frame:trial_end_frame+1]
        trial_start_time = trial_logdata['timestamp'][trial_start_frame]
        trial_logdata['timestamp'] = trial_logdata['timestamp'] - trial_start_time
        # print(i)
        trial_states = subject_session_data['states'][0][i]
        trial_time = trial_logdata['timestamp'].values
        
        trial_eye_df = eye_df.iloc[trial_start_frame:trial_end_frame+1]
        pupil_area = trial_eye_df['area'].tolist()

        if i == 95:
            print(i)        

        # clip to ~8 seconds
        trace_index = 452  # 1-9-25
        # trace_index = 390  # 1-14-25
        # trace_index = 393  # 1-12-25
        
        trial_time = trial_time[0:trace_index]
        pupil_area = pupil_area[0:trace_index]
        
        # impute nan values for pupil areas outside of expected range, update to use dlc coordinate calculations
        upper_area_lim = 3500 # 1-09-25
        lower_area_lim = 750 
        
        # upper_area_lim = 700 # 1-14-25
        # lower_area_lim = 200
        
        # upper_area_lim = 600 # 1-12-25
        # lower_area_lim = 200
        
        pupil_area = [x if x <= upper_area_lim else float('nan') for x in pupil_area]
        pupil_area = [x if x >= lower_area_lim else float('nan') for x in pupil_area]
        
        trial_pupil_area.append(pupil_area)        
        
        if i == 0:
            trial_time_baseline = trial_time
            
            time_step = np.mean(np.diff(trial_time_baseline))
            
            trial_time_stim_start_aligned_baseline = np.arange(start=-1, stop=9, step=time_step)
            # trial_time_stim_start_aligned_baseline_zero = np.argmin(np.abs(trial_time_stim_start_aligned_baseline))
            
            trial_time_grating_2_start_aligned_baseline = np.arange(start=-2, stop=8, step=time_step)
            # trial_time_grating_2_start_aligned_baseline_zero = np.argmin(np.abs(trial_time_grating_2_start_aligned_baseline))
            
        
        # pupil_area_stim_start_aligned = np.full(len(trial_time_stim_start_aligned_baseline), np.nan)
        # trial_pupil_area_grating_2_start_aligned = np.full(len(trial_time_stim_start_aligned_baseline), np.nan)
        
        move_correct_spout = subject_session_data['move_correct_spout_flag'][0][i]        
        trial_type = int(subject_session_data['trial_type'][0][i])
        
        
        # if trial_type == 1:
            
        # else:
        
        plot_each_trial = 0   
        
        if i == 281:
            print('debug')
        
        if plot_each_trial:
            plt.plot(trial_time, pupil_area)
        
     
        # licks
        # lick array
        # row 1 time of lick event
        # row 2 lick direction - 0 left, 1 right
        # row 3 correctness - 0 incorrect, 1 correct
        if 'lick_eye' in subject_session_data.keys():           
            trial_lick_eye = subject_session_data['lick_eye'][0][i]
            # left_licks = trial_lick_eye[:,np.where(trial_lick_eye[1,:] == 0)[0]]
            left_licks = trial_lick_eye[:,np.where(trial_lick_eye[1,:] == 0)]
            left_licks = left_licks[0][0] / 1000
            for left_lick in left_licks:
                if plot_each_trial:
                    plt.axvline(x=left_lick, color='blue', linestyle='--', label='left')
            right_licks = trial_lick_eye[:,np.where(trial_lick_eye[1,:] == 1)]
            right_licks = right_licks[0][0] / 1000       
            # Plot multiple vertical lines
            for right_lick in right_licks:
                if plot_each_trial:
                    plt.axvline(x=right_lick, color='red', linestyle='--', label='right')
        
        # Add vertical lines at specific x positions
        if 'AudStimTrigger' in trial_states.keys():
            VisStimStart = trial_states['AudStimTrigger'][0]
            VisStimStop = trial_states['AudStimTrigger'][1]
            if plot_each_trial:
                plt.axvline(VisStimStart, color='k', linestyle='--')
                plt.axvline(VisStimStop, color='k', linestyle='--')
        
        if 'RewardNaive' in trial_states.keys():
            RewardStart = trial_states['RewardNaive'][0]
            RewardStop = trial_states['RewardNaive'][1]
            if plot_each_trial:
                plt.axvline(RewardStart, color='g', linestyle='--')
                plt.axvline(RewardStop, color='g', linestyle='--')  
            if not np.isnan(trial_states['RewardNaive'][0]):
                rewarded = True
        
        if 'Reward' in trial_states.keys():
            RewardStart = trial_states['Reward'][0]
            RewardStop = trial_states['Reward'][1]
            if plot_each_trial:
                plt.axvline(RewardStart, color='g', linestyle='--')
                plt.axvline(RewardStop, color='g', linestyle='--') 
            if not np.isnan(trial_states['Reward'][0]):
                rewarded = True            
        
        if 'PostRewardDelay' in trial_states.keys():
            PostRewardDelayStart = trial_states['PostRewardDelay'][0]
            PostRewardDelayStop = trial_states['PostRewardDelay'][1]
            if plot_each_trial:
                plt.axvline(PostRewardDelayStart, color='g', linestyle='--')
                plt.axvline(PostRewardDelayStop, color='g', linestyle='--')  
                plt.fill_between(trial_time, pupil_area, where=((trial_time >= PostRewardDelayStart) & (trial_time <= PostRewardDelayStop)), color='lightgreen', alpha=0.5)            

        if 'TimeOutPunish' in trial_states.keys():
            TimeOutPunishStart = trial_states['TimeOutPunish'][0]
            TimeOutPunishStop = trial_states['TimeOutPunish'][1]
            if plot_each_trial:
                plt.axvline(TimeOutPunishStart, color='r', linestyle='--')
                plt.axvline(TimeOutPunishStop, color='r', linestyle='--') 
            if not np.isnan(trial_states['TimeOutPunish'][0]):
                rewarded = False

        # stim sequence
        trial_ProcessedSessionData = subject_session_data['ProcessedSessionData'][0][i]
        
        trial_seq = trial_ProcessedSessionData['trial_seq']
        trial_seq = np.array(trial_seq, dtype=np.float32)
        grating_bounds = trial_seq - np.roll(trial_seq, 1)
        
        grating_starts = np.array(np.where(grating_bounds == 2), dtype=np.float32) * img_time
        grating_stops = np.array(np.where(grating_bounds == -2), dtype=np.float32) * img_time
        
        grating_start_times = (grating_starts + VisStimStart)[0]
        grating_stop_times = (grating_stops + VisStimStart)[0]
        
        trial_grating_start_times.append(grating_start_times)
        trial_grating_stop_times.append(grating_stop_times)
        

        
        if plot_each_trial:
            # Shade the area between the vertical lines
            plt.fill_between(trial_time, pupil_area, where=((trial_time >= grating_start_times[0]) & (trial_time <= grating_stop_times[0])), color='lightblue', alpha=0.5)
            plt.fill_between(trial_time, pupil_area, where=((trial_time >= grating_start_times[1]) & (trial_time <= grating_stop_times[1])), color='lightblue', alpha=0.5)
    
            plt.show()   
        
        # align traces to start of stim        
        trial_time_stim_start_aligned = trial_time - VisStimStart
        
        
        stim_start_aligned = 0
        stim_stop_aligned = VisStimStop - VisStimStart
        grating_start_times_grating_1_aligned = grating_start_times - VisStimStart
        grating_stop_times_grating_1_aligned = grating_stop_times - VisStimStart
        
        if plot_each_trial:
            plt.plot(trial_time_stim_start_aligned, pupil_area)
            plt.axvline(stim_start_aligned, color='k', linestyle='--')
            plt.axvline(stim_stop_aligned, color='k', linestyle='--')
            plt.fill_between(trial_time_stim_start_aligned, pupil_area, where=((trial_time_stim_start_aligned >= grating_start_times_grating_1_aligned[0]) & (trial_time_stim_start_aligned <= grating_stop_times_grating_1_aligned[0])), color='lightblue', alpha=0.5)
            plt.fill_between(trial_time_stim_start_aligned, pupil_area, where=((trial_time_stim_start_aligned >= grating_start_times_grating_1_aligned[1]) & (trial_time_stim_start_aligned <= grating_stop_times_grating_1_aligned[1])), color='lightblue', alpha=0.5)
            plt.show()   
        
        
        pupil_area_stim_start_aligned = np.full(len(trial_time_stim_start_aligned_baseline), np.nan)
        sig_start_idx = np.argmin(np.abs(trial_time_stim_start_aligned_baseline + grating_start_times[0]))
        pupil_area_stim_start_aligned[sig_start_idx:len(pupil_area)+sig_start_idx] = pupil_area.copy()        
        
        trial_pupil_area_stim_start_aligned.append(pupil_area_stim_start_aligned)
        trial_grating_start_times_grating_1_aligned.append(grating_start_times_grating_1_aligned)
        trial_grating_stop_times_grating_1_aligned.append(grating_stop_times_grating_1_aligned)   
        
        if plot_each_trial:    
            plt.plot(trial_time_stim_start_aligned_baseline, pupil_area_stim_start_aligned)
            plt.axvline(stim_start_aligned, color='k', linestyle='--')
            plt.axvline(stim_stop_aligned, color='k', linestyle='--')
            plt.fill_between(trial_time_stim_start_aligned_baseline, pupil_area_stim_start_aligned, where=((trial_time_stim_start_aligned_baseline >= grating_start_times_grating_1_aligned[0]) & (trial_time_stim_start_aligned_baseline <= grating_stop_times_grating_1_aligned[0])), color='lightblue', alpha=0.5)
            plt.fill_between(trial_time_stim_start_aligned_baseline, pupil_area_stim_start_aligned, where=((trial_time_stim_start_aligned_baseline >= grating_start_times_grating_1_aligned[1]) & (trial_time_stim_start_aligned_baseline <= grating_stop_times_grating_1_aligned[1])), color='lightblue', alpha=0.5)        
            plt.show()   
        # if subject_session_data['states'][0][i]:
            
        # else:
        
        # align traces to start of grating 2
        shift_value = grating_start_times[1]
        trial_time_grating_2_start_aligned = trial_time - shift_value

        stim_start_grating_2_aligned = VisStimStart - shift_value
        stim_stop_grating_2_aligned = VisStimStop - shift_value
        grating_start_times_grating_2_aligned = grating_start_times - shift_value
        grating_stop_times_grating_2_aligned = grating_stop_times - shift_value
            
        
        if plot_each_trial:
            plt.plot(trial_time_grating_2_start_aligned, pupil_area)
            plt.axvline(stim_start_grating_2_aligned, color='k', linestyle='--')
            plt.axvline(stim_stop_grating_2_aligned, color='k', linestyle='--')
            plt.fill_between(trial_time_grating_2_start_aligned, pupil_area, where=((trial_time_grating_2_start_aligned >= grating_start_times_grating_2_aligned[0]) & (trial_time_grating_2_start_aligned <= grating_stop_times_grating_2_aligned[0])), color='lightblue', alpha=0.5)
            plt.fill_between(trial_time_grating_2_start_aligned, pupil_area, where=((trial_time_grating_2_start_aligned >= grating_start_times_grating_2_aligned[1]) & (trial_time_grating_2_start_aligned <= grating_stop_times_grating_2_aligned[1])), color='lightblue', alpha=0.5)                
            plt.show()
        
        
        pupil_area_grating_2_start_aligned = np.full(len(trial_time_grating_2_start_aligned_baseline), np.nan)
        sig_start_idx = np.argmin(np.abs(trial_time_grating_2_start_aligned_baseline + grating_start_times[1]))
        pupil_area_grating_2_start_aligned[sig_start_idx:len(pupil_area)+sig_start_idx] = pupil_area.copy()
        

        trial_pupil_area_grating_2_start_aligned.append(pupil_area_grating_2_start_aligned)
        trial_grating_start_times_grating_2_aligned.append(grating_start_times_grating_2_aligned)
        trial_grating_stop_times_grating_2_aligned.append(grating_stop_times_grating_2_aligned)         
        
        if plot_each_trial:
            plt.plot(trial_time_grating_2_start_aligned_baseline, pupil_area_grating_2_start_aligned)
            plt.axvline(stim_start_grating_2_aligned, color='k', linestyle='--')
            plt.axvline(stim_stop_grating_2_aligned, color='k', linestyle='--')
            plt.fill_between(trial_time_grating_2_start_aligned_baseline, pupil_area_grating_2_start_aligned, where=((trial_time_grating_2_start_aligned_baseline >= grating_start_times_grating_2_aligned[0]) & (trial_time_grating_2_start_aligned_baseline <= grating_stop_times_grating_2_aligned[0])), color='lightblue', alpha=0.5)
            plt.fill_between(trial_time_grating_2_start_aligned_baseline, pupil_area_grating_2_start_aligned, where=((trial_time_grating_2_start_aligned_baseline >= grating_start_times_grating_2_aligned[1]) & (trial_time_grating_2_start_aligned_baseline <= grating_stop_times_grating_2_aligned[1])), color='lightblue', alpha=0.5)                
            plt.show()        
        
        if not move_correct_spout:
            if trial_type == 1:
                trial_pupil_area_left.append(pupil_area)
                
                trial_pupil_area_stim_start_aligned_left.append(pupil_area_stim_start_aligned)
                trial_grating_start_times_grating_1_aligned_left.append(grating_start_times_grating_1_aligned)
                trial_grating_stop_times_grating_1_aligned_left.append(grating_stop_times_grating_1_aligned)
                     
                trial_pupil_area_grating_2_start_aligned_left.append(pupil_area_grating_2_start_aligned)
                trial_grating_start_times_grating_2_aligned_left.append(grating_start_times_grating_2_aligned)
                trial_grating_stop_times_grating_2_aligned_left.append(grating_stop_times_grating_2_aligned)  
                
                if rewarded:
                    # left trials rewarded
                    trial_pupil_area_left_rewarded.append(pupil_area)
                    
                    trial_pupil_area_stim_start_aligned_left_rewarded.append(pupil_area_stim_start_aligned)
                    trial_grating_start_times_grating_1_aligned_left_rewarded.append(grating_start_times_grating_1_aligned)
                    trial_grating_stop_times_grating_1_aligned_left_rewarded.append(grating_stop_times_grating_1_aligned)
                         
                    trial_pupil_area_grating_2_start_aligned_left_rewarded.append(pupil_area_grating_2_start_aligned)
                    trial_grating_start_times_grating_2_aligned_left_rewarded.append(grating_start_times_grating_2_aligned)
                    trial_grating_stop_times_grating_2_aligned_left_rewarded.append(grating_stop_times_grating_2_aligned)                      
                else:
                    # left trials punished
                    trial_pupil_area_left_punished.append(pupil_area)
                    
                    trial_pupil_area_stim_start_aligned_left_punished.append(pupil_area_stim_start_aligned)
                    trial_grating_start_times_grating_1_aligned_left_punished.append(grating_start_times_grating_1_aligned)
                    trial_grating_stop_times_grating_1_aligned_left_punished.append(grating_stop_times_grating_1_aligned)
                         
                    trial_pupil_area_grating_2_start_aligned_left_punished.append(pupil_area_grating_2_start_aligned)
                    trial_grating_start_times_grating_2_aligned_left_punished.append(grating_start_times_grating_2_aligned)
                    trial_grating_stop_times_grating_2_aligned_left_punished.append(grating_stop_times_grating_2_aligned)                       
                        
            else:    
                trial_pupil_area_right.append(pupil_area)
                
                trial_pupil_area_stim_start_aligned_right.append(pupil_area_stim_start_aligned)
                trial_grating_start_times_grating_1_aligned_right.append(grating_start_times_grating_1_aligned)
                trial_grating_stop_times_grating_1_aligned_right.append(grating_stop_times_grating_1_aligned)
                     
                trial_pupil_area_grating_2_start_aligned_right.append(pupil_area_grating_2_start_aligned)
                trial_grating_start_times_grating_2_aligned_right.append(grating_start_times_grating_2_aligned)
                trial_grating_stop_times_grating_2_aligned_right.append(grating_stop_times_grating_2_aligned)        
        
                if rewarded:
                    # right trials rewarded
                    trial_pupil_area_right_rewarded.append(pupil_area)
                    
                    trial_pupil_area_stim_start_aligned_right_rewarded.append(pupil_area_stim_start_aligned)
                    trial_grating_start_times_grating_1_aligned_right_rewarded.append(grating_start_times_grating_1_aligned)
                    trial_grating_stop_times_grating_1_aligned_right_rewarded.append(grating_stop_times_grating_1_aligned)
                         
                    trial_pupil_area_grating_2_start_aligned_right_rewarded.append(pupil_area_grating_2_start_aligned)
                    trial_grating_start_times_grating_2_aligned_right_rewarded.append(grating_start_times_grating_2_aligned)
                    trial_grating_stop_times_grating_2_aligned_right_rewarded.append(grating_stop_times_grating_2_aligned)                      
                else:
                    # right trials punished
                    trial_pupil_area_right_punished.append(pupil_area)
                    
                    trial_pupil_area_stim_start_aligned_right_punished.append(pupil_area_stim_start_aligned)
                    trial_grating_start_times_grating_1_aligned_right_punished.append(grating_start_times_grating_1_aligned)
                    trial_grating_stop_times_grating_1_aligned_right_punished.append(grating_stop_times_grating_1_aligned)
                         
                    trial_pupil_area_grating_2_start_aligned_right_punished.append(pupil_area_grating_2_start_aligned)
                    trial_grating_start_times_grating_2_aligned_right_punished.append(grating_start_times_grating_2_aligned)
                    trial_grating_stop_times_grating_2_aligned_right_punished.append(grating_stop_times_grating_2_aligned)          
        
        # align traces to start of response window
        trial_time_grating_response_window_start_aligned = trial_time - grating_start_times[1]        
        
        # print(i)          
    
    # all trials
    session_trial_time_baseline.append(trial_time_baseline)
    session_pupil_area.append(trial_pupil_area)
    
    session_trial_time_stim_start_aligned_baseline.append(trial_time_stim_start_aligned_baseline)
    session_trial_time_grating_2_start_aligned_baseline.append(trial_time_grating_2_start_aligned_baseline)             
    
    session_pupil_area_stim_start_aligned.append(trial_pupil_area_stim_start_aligned)
    session_grating_start_times_grating_1_aligned.append(trial_grating_start_times_grating_1_aligned)
    session_grating_stop_times_grating_1_aligned.append(trial_grating_stop_times_grating_1_aligned)
         
    session_pupil_area_grating_2_start_aligned.append(trial_pupil_area_grating_2_start_aligned)
    session_grating_start_times_grating_2_aligned.append(trial_grating_start_times_grating_2_aligned)
    session_grating_stop_times_grating_2_aligned.append(trial_grating_stop_times_grating_2_aligned)
    
    # left trials
    session_pupil_area_left.append(trial_pupil_area_left)
    
    session_pupil_area_stim_start_aligned_left.append(trial_pupil_area_stim_start_aligned_left)
    session_grating_start_times_grating_1_aligned_left.append(trial_grating_start_times_grating_1_aligned_left)
    session_grating_stop_times_grating_1_aligned_left.append(trial_grating_stop_times_grating_1_aligned_left)
         
    session_pupil_area_grating_2_start_aligned_left.append(trial_pupil_area_grating_2_start_aligned_left)
    session_grating_start_times_grating_2_aligned_left.append(trial_grating_start_times_grating_2_aligned_left)
    session_grating_stop_times_grating_2_aligned_left.append(trial_grating_stop_times_grating_2_aligned_left)  
    
    # left trials rewarded
    session_pupil_area_left_rewarded.append(trial_pupil_area_left_rewarded)
    
    session_pupil_area_stim_start_aligned_left_rewarded.append(trial_pupil_area_stim_start_aligned_left_rewarded)
    session_grating_start_times_grating_1_aligned_left_rewarded.append(trial_grating_start_times_grating_1_aligned_left_rewarded)
    session_grating_stop_times_grating_1_aligned_left_rewarded.append(trial_grating_stop_times_grating_1_aligned_left_rewarded)
         
    session_pupil_area_grating_2_start_aligned_left_rewarded.append(trial_pupil_area_grating_2_start_aligned_left_rewarded)
    session_grating_start_times_grating_2_aligned_left_rewarded.append(trial_grating_start_times_grating_2_aligned_left_rewarded)
    session_grating_stop_times_grating_2_aligned_left_rewarded.append(trial_grating_stop_times_grating_2_aligned_left_rewarded)
       
    trial_pupil_area_left_rewarded_mean = np.nanmean(trial_pupil_area_left_rewarded, axis=0)
    
    plt.plot(trial_time_baseline, trial_pupil_area_left_rewarded_mean)
    plt.show()
    
    trial_pupil_area_stim_start_aligned_left_rewarded_mean = np.nanmean(trial_pupil_area_stim_start_aligned_left_rewarded, axis=0)
   
    plt.plot(trial_time_stim_start_aligned_baseline, trial_pupil_area_stim_start_aligned_left_rewarded_mean)
    plt.show()
    
    trial_pupil_area_grating_2_start_aligned_left_rewarded_mean = np.nanmean(trial_pupil_area_grating_2_start_aligned_left_rewarded, axis=0)
    
    plt.plot(trial_time_grating_2_start_aligned_baseline, trial_pupil_area_grating_2_start_aligned_left_rewarded_mean)
    plt.show()    

    # left trials punished
    session_pupil_area_left_punished.append(trial_pupil_area_left_punished)
    
    session_pupil_area_stim_start_aligned_left_punished.append(trial_pupil_area_stim_start_aligned_left_punished)
    session_grating_start_times_grating_1_aligned_left_punished.append(trial_grating_start_times_grating_1_aligned_left_punished)
    session_grating_stop_times_grating_1_aligned_left_punished.append(trial_grating_stop_times_grating_1_aligned_left_punished)
         
    session_pupil_area_grating_2_start_aligned_left_punished.append(trial_pupil_area_grating_2_start_aligned_left_punished)
    session_grating_start_times_grating_2_aligned_left_punished.append(trial_grating_start_times_grating_2_aligned_left_punished)
    session_grating_stop_times_grating_2_aligned_left_punished.append(trial_grating_stop_times_grating_2_aligned_left_punished)     
    
    trial_pupil_area_left_punished_mean = np.nanmean(trial_pupil_area_left_punished, axis=0)
    
    plt.plot(trial_time_baseline, trial_pupil_area_left_punished_mean)
    plt.show()
    
    trial_pupil_area_stim_start_aligned_left_punished_mean = np.nanmean(trial_pupil_area_stim_start_aligned_left_punished, axis=0)
   
    plt.plot(trial_time_stim_start_aligned_baseline, trial_pupil_area_stim_start_aligned_left_punished_mean)
    plt.show()
    
    trial_pupil_area_grating_2_start_aligned_left_punished_mean = np.nanmean(trial_pupil_area_grating_2_start_aligned_left_punished, axis=0)
    
    plt.plot(trial_time_grating_2_start_aligned_baseline, trial_pupil_area_grating_2_start_aligned_left_punished_mean)
    plt.show()     
    
    # right trials
    session_pupil_area_right.append(trial_pupil_area_right)    
    
    session_pupil_area_stim_start_aligned_right.append(trial_pupil_area_stim_start_aligned_right)
    session_grating_start_times_grating_1_aligned_right.append(trial_grating_start_times_grating_1_aligned_right)
    session_grating_stop_times_grating_1_aligned_right.append(trial_grating_stop_times_grating_1_aligned_right)
         
    session_pupil_area_grating_2_start_aligned_right.append(trial_pupil_area_grating_2_start_aligned_right)
    session_grating_start_times_grating_2_aligned_right.append(trial_grating_start_times_grating_2_aligned_right)
    session_grating_stop_times_grating_2_aligned_right.append(trial_grating_stop_times_grating_2_aligned_right)      



    # right trials rewarded
    session_pupil_area_right_rewarded.append(trial_pupil_area_right_rewarded)
    
    session_pupil_area_stim_start_aligned_right_rewarded.append(trial_pupil_area_stim_start_aligned_right_rewarded)
    session_grating_start_times_grating_1_aligned_right_rewarded.append(trial_grating_start_times_grating_1_aligned_right_rewarded)
    session_grating_stop_times_grating_1_aligned_right_rewarded.append(trial_grating_stop_times_grating_1_aligned_right_rewarded)
         
    session_pupil_area_grating_2_start_aligned_right_rewarded.append(trial_pupil_area_grating_2_start_aligned_right_rewarded)
    session_grating_start_times_grating_2_aligned_right_rewarded.append(trial_grating_start_times_grating_2_aligned_right_rewarded)
    session_grating_stop_times_grating_2_aligned_right_rewarded.append(trial_grating_stop_times_grating_2_aligned_right_rewarded)
       
    trial_pupil_area_right_rewarded_mean = np.nanmean(trial_pupil_area_right_rewarded, axis=0)
    
    plt.plot(trial_time_baseline, trial_pupil_area_right_rewarded_mean)
    plt.show()
    
    trial_pupil_area_stim_start_aligned_right_rewarded_mean = np.nanmean(trial_pupil_area_stim_start_aligned_right_rewarded, axis=0)
   
    plt.plot(trial_time_stim_start_aligned_baseline, trial_pupil_area_stim_start_aligned_right_rewarded_mean)
    plt.show()
    
    trial_pupil_area_grating_2_start_aligned_right_rewarded_mean = np.nanmean(trial_pupil_area_grating_2_start_aligned_right_rewarded, axis=0)
    
    plt.plot(trial_time_grating_2_start_aligned_baseline, trial_pupil_area_grating_2_start_aligned_right_rewarded_mean)
    plt.show()    

    # right trials punished
    session_pupil_area_right_punished.append(trial_pupil_area_right_punished)
    
    session_pupil_area_stim_start_aligned_right_punished.append(trial_pupil_area_stim_start_aligned_right_punished)
    session_grating_start_times_grating_1_aligned_right_punished.append(trial_grating_start_times_grating_1_aligned_right_punished)
    session_grating_stop_times_grating_1_aligned_right_punished.append(trial_grating_stop_times_grating_1_aligned_right_punished)
         
    session_pupil_area_grating_2_start_aligned_right_punished.append(trial_pupil_area_grating_2_start_aligned_right_punished)
    session_grating_start_times_grating_2_aligned_right_punished.append(trial_grating_start_times_grating_2_aligned_right_punished)
    session_grating_stop_times_grating_2_aligned_right_punished.append(trial_grating_stop_times_grating_2_aligned_right_punished)  

    trial_pupil_area_right_punished_mean = np.nanmean(trial_pupil_area_right_punished, axis=0)
    
    plt.plot(trial_time_baseline, trial_pupil_area_right_punished_mean)
    plt.show()
    
    trial_pupil_area_stim_start_aligned_right_punished_mean = np.nanmean(trial_pupil_area_stim_start_aligned_right_punished, axis=0)
   
    plt.plot(trial_time_stim_start_aligned_baseline, trial_pupil_area_stim_start_aligned_right_punished_mean)
    plt.show()
    
    trial_pupil_area_grating_2_start_aligned_right_punished_mean = np.nanmean(trial_pupil_area_grating_2_start_aligned_right_punished, axis=0)
    
    plt.plot(trial_time_grating_2_start_aligned_baseline, trial_pupil_area_grating_2_start_aligned_right_punished_mean)
    plt.show() 


    ########################
    
    
    # left grating 1 aligned
    plt.plot(trial_time_stim_start_aligned_baseline, trial_pupil_area_stim_start_aligned_left_rewarded_mean)
    plt.plot(trial_time_stim_start_aligned_baseline, trial_pupil_area_stim_start_aligned_left_punished_mean)
    plt.show()      
    
    
    # right grating 1 aligned
    plt.plot(trial_time_stim_start_aligned_baseline, trial_pupil_area_stim_start_aligned_right_rewarded_mean)
    plt.plot(trial_time_stim_start_aligned_baseline, trial_pupil_area_stim_start_aligned_right_punished_mean)
    plt.show()    
    
    # left grating 2 aligned
    plt.plot(trial_time_grating_2_start_aligned_baseline, trial_pupil_area_grating_2_start_aligned_left_rewarded_mean)
    plt.plot(trial_time_grating_2_start_aligned_baseline, trial_pupil_area_grating_2_start_aligned_left_punished_mean)
    plt.show()  

    # right grating 2 aligned
    plt.plot(trial_time_grating_2_start_aligned_baseline, trial_pupil_area_grating_2_start_aligned_right_rewarded_mean)
    plt.plot(trial_time_grating_2_start_aligned_baseline, trial_pupil_area_grating_2_start_aligned_right_punished_mean)
    plt.show() 
    
    
    print(i)


    # max_sessions=25
    # outcomes = subject_session_data['outcomes']
    # dates = subject_session_data['dates']
    # chemo_labels = subject_session_data['Chemo']
    # jitter_flag = subject_session_data['jitter_flag']
    # jitter_session = np.array([np.sum(j) for j in jitter_flag])
    # jitter_session[jitter_session!=0] = 1
    # start_idx = 0
    # if max_sessions != -1 and len(dates) > max_sessions:
    #     start_idx = len(dates) - max_sessions
    # outcomes = outcomes[start_idx:]
    # dates = dates[start_idx:]
    # jitter_session = jitter_session[start_idx:]
    # chemo_labels = chemo_labels[start_idx:]
    # counts = count_label(outcomes, states)
    # session_id = np.arange(len(outcomes)) + 1
    # bottom = np.cumsum(counts, axis=1)
    # bottom[:,1:] = bottom[:,:-1]
    # bottom[:,0] = 0
    # width = 0.5
    # for i in range(len(states)):
    #     ax.bar(
    #         session_id, counts[:,i],
    #         bottom=bottom[:,i],
    #         edgecolor='white',
    #         width=width,
    #         color=colors[i],
    #         label=states[i])
    # ax.tick_params(tick1On=False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.hlines(0.5,0,len(dates)+1, linestyle='--' , color='silver' , lw = 0.5) 
    # ax.hlines(0.75,0,len(dates)+1, linestyle='--' , color='silver' , lw = 0.5)    
    # #ax.yaxis.grid(True)
    # ax.set_xlabel('training session')
    # ax.set_ylabel('number of trials')
    # ax.set_xticks(np.arange(len(outcomes))+1)
    # ax.set_yticks(np.arange(6)*0.2)
    # # # Create the second Y axis
    # # ax2 = ax.twinx()
    # # # logdatahronize the tick marks between both Y axes
    # # ax2.set_yticks(ax.get_yticks())  # Set ax2's ticks to be the same as ax1's
    # #ax.set_xticklabels(dates, rotation='vertical')
    # ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    # ax.set_title('reward/punish percentage for completed trials')
    # dates_label = dates
    # for i in range(0 , len(chemo_labels)):
    #     if chemo_labels[i] == 1:
    #         dates_label[i] = dates[i] + '(chemo)'
    #     if jitter_session[i] == 1:
    #         dates_label[i] =  dates_label[i] + '(jittered)'
    # ax.set_xticklabels(dates_label, rotation=45)
    # ind = 0
    # for xtick in ax.get_xticklabels():
    #     if jitter_session[ind] == 1:
    #         xtick.set_color('limegreen')
    #     if chemo_labels[ind] == 1:
    #         xtick.set_color('red')
    #     ind = ind + 1
