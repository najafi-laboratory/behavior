import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import date

def save_image(filename): 
    
    # PdfPages is a wrapper around pdf  
    # file so there is no clash and 
    # create files with no error. 
    p = PdfPages(filename+'.pdf')
      
    # get_fignums Return list of existing 
    # figure numbers 
    fig_nums = plt.get_fignums()   
    figs = [plt.figure(n) for n in fig_nums] 
      
    # iterating over the numbers in list 
    for fig in figs:  
        
        # and saving the files 
        fig.savefig(p, format='pdf', dpi=300)
          
    # close the object 
    p.close() 

def plot_fig2(
        session_data,
        output_dir_onedrive, 
        output_dir_local
        ):

    savefiles = 1
    
    max_sessions=10
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    outcomes = outcomes[start_idx:]
    # dates = dates[start_idx:]
    # session_data = session_data[start_idx:]
    session_id = np.arange(len(outcomes)) + 1      
    
    today = date.today()
    today_formatted = str(today)[2:]
    year = today_formatted[0:2]
    month = today_formatted[3:5]
    day = today_formatted[6:]
    today_string = year + month + day
    
    # print()
    
    # isSelfTimedMode = session_data['isSelfTimedMode'][0]
    
   
    for i in range(start_idx, session_data['total_sessions']):
        # print(i)
        isSelfTimedMode = session_data['isSelfTimedMode'][i][0]
        isShortDelay = session_data['isShortDelay'][i]
        press_reps = session_data['press_reps'][i]
        press_window = session_data['press_window'][i]  
        press_delay_avg = session_data['session_press_delay_avg'][i]
        press_delay_short_avg = session_data['session_press_delay_short_avg'][i]
        press_delay_long_avg = session_data['session_press_delay_long_avg'][i]
                        
        # press_delay =  session_data['session_press_delay'][i]
        
        print('session', i)
        print('isSelfTimedMode', isSelfTimedMode)
        print('isShortDelay', isShortDelay)
        print('press_reps', press_reps)
        print('press_window', press_window)
        
        print('press_delay_avg', press_delay_avg)
        print('press_delay_short_avg', press_delay_short_avg)
        print('press_delay_long_avg', press_delay_long_avg)
        
        numTrials = len(session_data['outcomes'][i])
        numRewardedTrials = len(session_data['rewarded_trials'][i])
                
        session_date_formatted = dates[i][2:]       
        
        # temp, update with press window averaging
        # press_window = session_data['session_press_window']
        press_window = press_window[1]
        
        vis_stim_2_enable = session_data['vis_stim_2_enable']
        
        # vis_stim_2_enable deprecated as an indicator of vis2 or wait2, use session_wait_2_aligned
        isWaitForPress2 = (len(session_data['session_wait_2_aligned'][i]) > 0)
        if isWaitForPress2:
            align2Title = 'WaitForPress2 Aligned\n'
        else:
            align2Title = 'VisStim2 Aligned.\n'
        
        encoder_times_vis1 = session_data['encoder_times_aligned_VisStim1']
        encoder_pos_avg_vis1 = session_data['encoder_pos_avg_vis1'][i]
        encoder_times_vis2 = session_data['encoder_times_aligned_VisStim2']
        encoder_pos_avg_vis2 = session_data['encoder_pos_avg_vis2'][i]
        encoder_times_rew = session_data['encoder_times_aligned_Reward']
        encoder_pos_avg_rew = session_data['encoder_pos_avg_rew'][i]
        
        time_left_VisStim1 = session_data['time_left_VisStim1']
        time_right_VisStim1 = session_data['time_right_VisStim1']
        
        time_left_VisStim2 = session_data['time_left_VisStim2']
        time_right_VisStim2 = session_data['time_right_VisStim2']
        
        time_left_rew = session_data['time_left_rew']
        time_right_rew = session_data['time_right_rew']
        
        target_thresh = session_data['session_target_thresh']
        
        # encoder_times_vis1 = session_data['encoder_times_aligned_VisStim1']            
        # encoder_times_vis2 = session_data['encoder_times_aligned_VisStim2']            
        # encoder_times_rew = session_data['encoder_times_aligned_Reward']            
        
        
        encoder_pos_avg_vis1_short = session_data['encoder_pos_avg_vis1_short'][i]
        encoder_pos_avg_vis1_long = session_data['encoder_pos_avg_vis1_long'][i]

        encoder_pos_avg_vis2_short = session_data['encoder_pos_avg_vis2_short'][i]
        encoder_pos_avg_vis2_long = session_data['encoder_pos_avg_vis2_long'][i]
        
        encoder_pos_avg_rew_short = session_data['encoder_pos_avg_rew_short'][i]
        encoder_pos_avg_rew_long = session_data['encoder_pos_avg_rew_long'][i]
                
        singleRow = 0        
        isOnlyShort = 0
        isOnlyLong = 0
        # type(encoder_pos_avg_vis1_short) is np.ndarray
        if not isSelfTimedMode:
            # if len(encoder_pos_avg_vis1_short) < 2:
            if not (type(encoder_pos_avg_vis1_short) is np.ndarray):
                isOnlyLong = 1
                singleRow = 1
            # elif len(encoder_pos_avg_vis1_long) < 2:
            elif not (type(encoder_pos_avg_vis1_long) is np.ndarray):
                isOnlyShort = 1
                singleRow = 1
        else:
            singleRow = 1
            
        # if isSelfTimedMode: 
        if singleRow: 
        
        
            # numTrials = len(session_data['outcomes'][i])
            # numRewardedTrials = len(session_data['rewarded_trials'][i])
                    
            # session_date_formatted = dates[i][2:]       
            
            # # temp, update with press window averaging
            # # press_window = session_data['session_press_window']
            # press_window = press_window[1]
            
            # vis_stim_2_enable = session_data['vis_stim_2_enable']
            
            # encoder_times_vis1 = session_data['encoder_times_aligned_VisStim1']
            # encoder_pos_avg_vis1 = session_data['encoder_pos_avg_vis1'][i]
            # encoder_times_vis2 = session_data['encoder_times_aligned_VisStim2']
            # encoder_pos_avg_vis2 = session_data['encoder_pos_avg_vis2'][i]
            # encoder_times_rew = session_data['encoder_times_aligned_Reward']
            # encoder_pos_avg_rew = session_data['encoder_pos_avg_rew'][i]
            
            # time_left_VisStim1 = session_data['time_left_VisStim1']
            # time_right_VisStim1 = session_data['time_right_VisStim1']
            
            # time_left_VisStim2 = session_data['time_left_VisStim2']
            # time_right_VisStim2 = session_data['time_right_VisStim2']
            
            # time_left_rew = session_data['time_left_rew']
            # time_right_rew = session_data['time_right_rew']
            
            # target_thresh = session_data['session_target_thresh']
            
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
            fig.subplots_adjust(hspace=0.7)
            if isSelfTimedMode:
                fig.suptitle(subject + ' Self Timed - ' + session_date_formatted + '  ' + str(numRewardedTrials) + '/' + str(numTrials) + ' Trials Rewarded.\nPress Window:' + ' ' + str(press_window) + 's')
            else:
                if isOnlyShort:
                    fig.suptitle(subject + ' Visually Guided - ' + session_date_formatted + '  ' + str(numRewardedTrials) + '/' + str(numTrials) + ' Trials Rewarded.\nPress Window:' + ' ' + str(press_window) + 's' + '\nShort Delay Only ' + str(round(press_delay_short_avg, 3)) + 'ms')    
                elif isOnlyLong:
                    fig.suptitle(subject + ' Visually Guided - ' + session_date_formatted + '  ' + str(numRewardedTrials) + '/' + str(numTrials) + ' Trials Rewarded.\nPress Window:' + ' ' + str(press_window) + 's' + '\nLong Delay Only ' + str(round(press_delay_long_avg, 3)) + 'ms')        
                else:
                    fig.suptitle(subject + ' Visually Guided - ' + session_date_formatted + '  ' + str(numRewardedTrials) + '/' + str(numTrials) + ' Trials Rewarded.\nPress Window:' + ' ' + str(press_window) + 's')
                                
                
            y_top = 3.5
            
            # vis 1 aligned
            
            if isSelfTimedMode:
                axs[0].plot(encoder_times_vis1, encoder_pos_avg_vis1,'-', label='Average Trajectory')
            elif isOnlyShort:
                axs[0].plot(encoder_times_vis1, encoder_pos_avg_vis1_short,'-', label='Average Trajectory')
            else:
                axs[0].plot(encoder_times_vis1, encoder_pos_avg_vis1_long,'-', label='Average Trajectory')    
            
            axs[0].axvline(x = 0, color = 'r', label = 'VisStim1', linestyle='--')
            
            target_thresh = 2.0
            axs[0].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')    
            
            
            axs[0].set_title('VisStim1 Aligned.\n')        
            axs[0].legend(loc='upper right')
            axs[0].set_xlim(time_left_VisStim1, 4.0)
            # axs[0].set_ylim(-0.2, target_thresh+1)
            axs[0].set_ylim(-0.2, y_top)
            axs[0].spines['right'].set_visible(False)
            axs[0].spines['top'].set_visible(False)
            axs[0].set_xlabel('Time from VisStim1 (s)')
            axs[0].set_ylabel('Joystick deflection (deg)')
                
            # vis 2 or waitforpress aligned
            # axs[1].plot(encoder_times_vis2, encoder_pos_avg_vis2,'-', label='Average Trajectory')
            
            
            if isSelfTimedMode:
                axs[1].plot(encoder_times_vis2, encoder_pos_avg_vis2,'-', label='Average Trajectory')
            elif isOnlyShort:
                axs[1].plot(encoder_times_vis2, encoder_pos_avg_vis2_short,'-', label='Average Trajectory')
            else:
                axs[1].plot(encoder_times_vis2, encoder_pos_avg_vis2_long,'-', label='Average Trajectory')   
            
            
            if vis_stim_2_enable:
                axs[1].axvline(x = 0, color = 'r', label = 'VisStim2', linestyle='--')
                # axs[1].set_title('VisStim2 Aligned.\n')
                axs[1].set_title(align2Title)                
                axs[1].set_xlabel('Time from VisStim2 (s)')
            else:
                axs[1].axvline(x = 0, color = 'r', label = 'WaitForPress2', linestyle='--')
                axs[1].set_title('WaitForPress2 Aligned.\n')
                axs[1].set_xlabel('Time from WaitForPress2 (s)')
                
                
            # update to get average threshold across range of trials
            # temp thresh
            target_thresh = 2.0
            axs[1].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
            
            
            axs[1].legend(loc='upper right')
            axs[1].set_xlim(-1, 2.0)
            # axs[1].set_ylim(-0.2, target_thresh+1)
            axs[1].set_ylim(-0.2, y_top)
            axs[1].spines['right'].set_visible(False)
            axs[1].spines['top'].set_visible(False)
            axs[1].set_ylabel('Joystick deflection (deg)')
            
            # reward aligned
            # fig3, axs3 = plt.subplots(1, figsize=(10, 4))
            # axs[2].subplots_adjust(hspace=0.7)
            # axs[2].plot(encoder_times_rew, encoder_pos_avg_rew,'-', label='Average Trajectory')
            
            
            if isSelfTimedMode:
                axs[2].plot(encoder_times_rew, encoder_pos_avg_rew,'-', label='Average Trajectory')
            elif isOnlyShort:
                axs[2].plot(encoder_times_rew, encoder_pos_avg_rew_short,'-', label='Average Trajectory')
            else:
                axs[2].plot(encoder_times_rew, encoder_pos_avg_rew_long,'-', label='Average Trajectory')              
            
            
            axs[2].axvline(x = 0, color = 'r', label = 'Reward', linestyle='--')
            axs[2].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
            # axs[2].set_title(subject + ' - ' + session_date_formatted)
            axs[2].set_title('Reward Aligned.\n')    
            axs[2].legend(loc='upper right')               
            axs[2].set_xlim(-1.0, 1.5)
            # axs[2].set_ylim(-0.2, target_thresh+1)
            axs[2].set_ylim(-0.2, y_top)
            axs[2].spines['right'].set_visible(False)
            axs[2].spines['top'].set_visible(False)
            axs[2].set_xlabel('Time from Reward (s)')
            axs[2].set_ylabel('Joystick deflection (deg)')
            
            
            fig.tight_layout()
            
            # output_imgs_dir = 'C:\\data analysis\\behavior\\joystick\\figures\\'+subject+'\\' + 'avg_trajectory_imgs\\'        
            
            
            
            # output_figs_dir = 'C:/Users/timst/OneDrive - Georgia Institute of Technology/Najafi_Lab/0_Data_analysis/Behavior/Joystick/' +subject+'\\'
            # output_imgs_dir = 'C:/Users/timst/OneDrive - Georgia Institute of Technology/Najafi_Lab/0_Data_analysis/Behavior/Joystick/'+subject+'\\' + 'avg_trajectory_imgs\\'        
            
            if savefiles:
                # these for saving files
                output_figs_dir = output_dir_onedrive + subject + '/'    
                # output_imgs_dir = 'C:/Users/timst/OneDrive - Georgia Institute of Technology/Najafi_Lab/0_Data_analysis/Behavior/Joystick/'+subject+'\\' + 'avg_trajectory_imgs\\'        
                output_imgs_dir = output_dir_local + subject + '/avg_trajectory_imgs/'
                os.makedirs(output_figs_dir, exist_ok = True)
                os.makedirs(output_imgs_dir, exist_ok = True)
                output_pdf_filename = output_figs_dir + subject + '_all_sess_trajectory_ave_trs'
                save_image(output_pdf_filename)
                fig.savefig(output_imgs_dir + subject +'_'+ session_date_formatted + '_trajectory_ave_trs' + '.png', dpi=300)
            
    
            # filename = 'C:\\data analysis\\behavior\\joystick\\figures\\'+subject+'\\'+'fig2_'+subject+'_avg_trajectory'  
            # os.makedirs('C:\\behavior\\joystick\\figures\\'+subject+'\\'+dates[i], exist_ok = True)
            # save_image(filename)
            # fig.savefig('C:\\behavior\\joystick\\figures\\'+subject+'\\'+dates[i]+'\\fig2_'+subject+'_avg_trajectory.png', dpi=300)
            # plt.close(fig)
        else:

            # isSelfTimedMode = session_data['isSelfTimedMode'][i]
            # isShortDelay = session_data['isShortDelay'][i]
            # press_reps = session_data['press_reps'][i]
            # press_window = session_data['press_window'][i]  
            
            # print('session', i)
            # print('isSelfTimedMode', isSelfTimedMode)
            # print('isShortDelay', isShortDelay)
            # print('press_reps', press_reps)
            # print('press_window', press_window)
                        
            # numTrials = len(session_data['outcomes'][i])
            # numRewardedTrials = len(session_data['rewarded_trials'][i])
                    
            # session_date_formatted = dates[i][2:]       
            
            # # temp, update with press window averaging
            # # press_window = session_data['session_press_window']
            
            # press_window = press_window[1]
            
            # vis_stim_2_enable = session_data['vis_stim_2_enable']
            
            # encoder_times_vis1 = session_data['encoder_times_aligned_VisStim1']            
            # encoder_times_vis2 = session_data['encoder_times_aligned_VisStim2']            
            # encoder_times_rew = session_data['encoder_times_aligned_Reward']            
            
            
            # encoder_pos_avg_vis1_short = session_data['encoder_pos_avg_vis1_short'][i]
            # encoder_pos_avg_vis1_long = session_data['encoder_pos_avg_vis1_long'][i]

            # encoder_pos_avg_vis2_short = session_data['encoder_pos_avg_vis2_short'][i]
            # encoder_pos_avg_vis2_long = session_data['encoder_pos_avg_vis2_long'][i]
            
            # encoder_pos_avg_rew_short = session_data['encoder_pos_avg_rew_short'][i]
            # encoder_pos_avg_rew_long = session_data['encoder_pos_avg_rew_long'][i]
            
            # time_left_VisStim1 = session_data['time_left_VisStim1']
            # time_right_VisStim1 = session_data['time_right_VisStim1']
            
            # time_left_VisStim2 = session_data['time_left_VisStim2']
            # time_right_VisStim2 = session_data['time_right_VisStim2']
            
            # time_left_rew = session_data['time_left_rew']
            # time_right_rew = session_data['time_right_rew']
            
            # target_thresh = session_data['session_target_thresh']
            
            # 2 rows, top for short delay, bottom for long delay
            
            # for short/long vis-guided, first plot regular averages
            # ----------------------------------------------------------------------------------------------------------------
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
            fig.subplots_adjust(hspace=0.7)
            if isSelfTimedMode:
                fig.suptitle(subject + ' Self Timed - ' + session_date_formatted + '  ' + str(numRewardedTrials) + '/' + str(numTrials) + ' Trials Rewarded.\nPress Window:' + ' ' + str(press_window) + 's')
            else:
                if isOnlyShort:
                    fig.suptitle(subject + ' Visually Guided - ' + session_date_formatted + '  ' + str(numRewardedTrials) + '/' + str(numTrials) + ' Trials Rewarded.\nPress Window:' + ' ' + str(press_window) + 's' + '\nShort Delay Only ' + str(round(press_delay_short_avg, 3)) + 'ms')    
                elif isOnlyLong:
                    fig.suptitle(subject + ' Visually Guided - ' + session_date_formatted + '  ' + str(numRewardedTrials) + '/' + str(numTrials) + ' Trials Rewarded.\nPress Window:' + ' ' + str(press_window) + 's' + '\nLong Delay Only ' + str(round(press_delay_long_avg, 3)) + 'ms')        
                else:
                    fig.suptitle(subject + ' Visually Guided - ' + session_date_formatted + '  ' + str(numRewardedTrials) + '/' + str(numTrials) + ' Trials Rewarded.\nPress Window:' + ' ' + str(press_window) + 's')
                
            y_top = 3.5
            
            # vis 1 aligned
            
            if isSelfTimedMode:
                axs[0].plot(encoder_times_vis1, encoder_pos_avg_vis1,'-', label='Average Trajectory')
            elif isOnlyShort:
                axs[0].plot(encoder_times_vis1, encoder_pos_avg_vis1_short,'-', label='Average Trajectory')
            else:
                axs[0].plot(encoder_times_vis1, encoder_pos_avg_vis1_long,'-', label='Average Trajectory')    
            
            axs[0].axvline(x = 0, color = 'r', label = 'VisStim1', linestyle='--')
            
            target_thresh = 2.0
            axs[0].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')    
            
            
            axs[0].set_title('VisStim1 Aligned.\n')        
            axs[0].legend(loc='upper right')
            axs[0].set_xlim(time_left_VisStim1, 4.0)
            # axs[0].set_ylim(-0.2, target_thresh+1)
            axs[0].set_ylim(-0.2, y_top)
            axs[0].spines['right'].set_visible(False)
            axs[0].spines['top'].set_visible(False)
            axs[0].set_xlabel('Time from VisStim1 (s)')
            axs[0].set_ylabel('Joystick deflection (deg)')
                
            # vis 2 or waitforpress aligned
            # axs[1].plot(encoder_times_vis2, encoder_pos_avg_vis2,'-', label='Average Trajectory')
            
            
            if isSelfTimedMode:
                axs[1].plot(encoder_times_vis2, encoder_pos_avg_vis2,'-', label='Average Trajectory')
            elif isOnlyShort:
                axs[1].plot(encoder_times_vis2, encoder_pos_avg_vis2_short,'-', label='Average Trajectory')
            else:
                axs[1].plot(encoder_times_vis2, encoder_pos_avg_vis2_long,'-', label='Average Trajectory')   
            
            
            if vis_stim_2_enable:
                axs[1].axvline(x = 0, color = 'r', label = 'VisStim2', linestyle='--')
                # axs[1].set_title('VisStim2 Aligned.\n')
                axs[1].set_title(align2Title)
                axs[1].set_xlabel('Time from VisStim2 (s)')
            else:
                axs[1].axvline(x = 0, color = 'r', label = 'WaitForPress2', linestyle='--')
                axs[1].set_title('WaitForPress2 Aligned.\n')
                axs[1].set_xlabel('Time from WaitForPress2 (s)')
                
                
            # update to get average threshold across range of trials
            # temp thresh
            target_thresh = 2.0
            axs[1].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
            
            
            axs[1].legend(loc='upper right')
            axs[1].set_xlim(-1, 2.0)
            # axs[1].set_ylim(-0.2, target_thresh+1)
            axs[1].set_ylim(-0.2, y_top)
            axs[1].spines['right'].set_visible(False)
            axs[1].spines['top'].set_visible(False)
            axs[1].set_ylabel('Joystick deflection (deg)')
            
            # reward aligned
            # fig3, axs3 = plt.subplots(1, figsize=(10, 4))
            # axs[2].subplots_adjust(hspace=0.7)
            # axs[2].plot(encoder_times_rew, encoder_pos_avg_rew,'-', label='Average Trajectory')
            
            
            if isSelfTimedMode:
                axs[2].plot(encoder_times_rew, encoder_pos_avg_rew,'-', label='Average Trajectory')
            elif isOnlyShort:
                axs[2].plot(encoder_times_rew, encoder_pos_avg_rew_short,'-', label='Average Trajectory')
            else:
                axs[2].plot(encoder_times_rew, encoder_pos_avg_rew_long,'-', label='Average Trajectory')              
            
            
            axs[2].axvline(x = 0, color = 'r', label = 'Reward', linestyle='--')
            axs[2].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
            # axs[2].set_title(subject + ' - ' + session_date_formatted)
            axs[2].set_title('Reward Aligned.\n')    
            axs[2].legend(loc='upper right')               
            axs[2].set_xlim(-1.0, 1.5)
            # axs[2].set_ylim(-0.2, target_thresh+1)
            axs[2].set_ylim(-0.2, y_top)
            axs[2].spines['right'].set_visible(False)
            axs[2].spines['top'].set_visible(False)
            axs[2].set_xlabel('Time from Reward (s)')
            axs[2].set_ylabel('Joystick deflection (deg)')
            
            
            fig.tight_layout()
            # start short/long avgs
            # -----------------------------------------------------------------------------------------------------------------
            
            fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 8))
            fig.subplots_adjust(hspace=0.7)
            fig.suptitle(subject + ' Visually Guided - ' + session_date_formatted + '  ' + str(numRewardedTrials) + '/' + str(numTrials) + ' Trials Rewarded.\nPress Window:' + ' ' + str(press_window) + 's' + '\nTop - Short Delay ' + str(round(press_delay_short_avg, 3)) + 'ms\nBottom - Long Delay ' + str(round(press_delay_long_avg, 3)) + 'ms')
            
            y_top = 3.5
            
            
            # plot short delay
            
            # vis 1 aligned
            axs[0,0].plot(encoder_times_vis1, encoder_pos_avg_vis1_short,'-', label='Average Trajectory', color='darkblue')
            axs[0,0].axvline(x = 0, color = 'r', label = 'VisStim1', linestyle='--')
            
            target_thresh = 2.0
            axs[0,0].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')    
            
            
            axs[0,0].set_title('VisStim1 Aligned.\n')        
            axs[0,0].legend(loc='upper right')
            axs[0,0].set_xlim(time_left_VisStim1, 4.0)
            # axs[0,0].set_ylim(-0.2, target_thresh+1)
            axs[0,0].set_ylim(-0.2, y_top)
            axs[0,0].spines['right'].set_visible(False)
            axs[0,0].spines['top'].set_visible(False)
            axs[0,0].set_xlabel('Time from VisStim1 (s)')
            axs[0,0].set_ylabel('Joystick deflection (deg)')
                
            # vis 2 or waitforpress aligned
            axs[0,1].plot(encoder_times_vis2, encoder_pos_avg_vis2_short,'-', label='Average Trajectory', color='darkblue')
            if vis_stim_2_enable:
                axs[0,1].axvline(x = 0, color = 'r', label = 'VisStim2', linestyle='--')
                # axs[0,1].set_title('VisStim2 Aligned.\n')
                axs[0,1].set_title(align2Title)
                axs[0,1].set_xlabel('Time from VisStim2 (s)')
            else:
                axs[0,1].axvline(x = 0, color = 'r', label = 'WaitForPress2', linestyle='--')
                axs[0,1].set_title('WaitForPress2 Aligned.\n')
                axs[0,1].set_xlabel('Time from WaitForPress2 (s)')
                
                
            # update to get average threshold across range of trials
            # temp thresh
            target_thresh = 2.0
            axs[0,1].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
            
            
            axs[0,1].legend(loc='upper right')
            axs[0,1].set_xlim(-1, 2.0)
            # axs[0,1].set_ylim(-0.2, target_thresh+1)
            axs[0,1].set_ylim(-0.2, y_top)
            axs[0,1].spines['right'].set_visible(False)
            axs[0,1].spines['top'].set_visible(False)
            axs[0,1].set_ylabel('Joystick deflection (deg)')
            
            # reward aligned
            # fig3, axs3 = plt.subplots(1, figsize=(10, 4))
            # axs[2].subplots_adjust(hspace=0.7)
            axs[0,2].plot(encoder_times_rew, encoder_pos_avg_rew_short,'-', label='Average Trajectory', color='darkblue')
            axs[0,2].axvline(x = 0, color = 'r', label = 'Reward', linestyle='--')
            axs[0,2].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
            # axs[0,2].set_title(subject + ' - ' + session_date_formatted)
            axs[0,2].set_title('Reward Aligned.\n')    
            axs[0,2].legend(loc='upper right')               
            axs[0,2].set_xlim(-1.0, 1.5)
            # axs[0,2].set_ylim(-0.2, target_thresh+1)
            axs[0,2].set_ylim(-0.2, y_top)
            axs[0,2].spines['right'].set_visible(False)
            axs[0,2].spines['top'].set_visible(False)
            axs[0,2].set_xlabel('Time from Reward (s)')
            axs[0,2].set_ylabel('Joystick deflection (deg)')
            
            
            # ------------------------------------------------------------------------------------------------------------------
            # plot long delay
            
            # vis 1 aligned
            axs[1,0].plot(encoder_times_vis1, encoder_pos_avg_vis1_long,'-', label='Average Trajectory')
            axs[1,0].axvline(x = 0, color = 'r', label = 'VisStim1', linestyle='--')
            
            target_thresh = 2.0
            axs[1,0].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')    
            
            
            axs[1,0].set_title('VisStim1 Aligned.\n')        
            axs[1,0].legend(loc='upper right')
            axs[1,0].set_xlim(time_left_VisStim1, 4.0)
            # axs[1,0].set_ylim(-0.2, target_thresh+1)
            axs[1,0].set_ylim(-0.2, y_top)
            axs[1,0].spines['right'].set_visible(False)
            axs[1,0].spines['top'].set_visible(False)
            axs[1,0].set_xlabel('Time from VisStim1 (s)')
            axs[1,0].set_ylabel('Joystick deflection (deg)')
                
            # vis 2 or waitforpress aligned
            axs[1,1].plot(encoder_times_vis2, encoder_pos_avg_vis2_long,'-', label='Average Trajectory')
            if vis_stim_2_enable:
                axs[1,1].axvline(x = 0, color = 'r', label = 'VisStim2', linestyle='--')
                # axs[1,1].set_title('VisStim2 Aligned.\n')
                axs[1,1].set_title(align2Title)
                axs[1,1].set_xlabel('Time from VisStim2 (s)')
            else:
                axs[1,1].axvline(x = 0, color = 'r', label = 'WaitForPress2', linestyle='--')
                axs[1,1].set_title('WaitForPress2 Aligned.\n')
                axs[1,1].set_xlabel('Time from WaitForPress2 (s)')
                
                
            # update to get average threshold across range of trials
            # temp thresh
            target_thresh = 2.0
            axs[1,1].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
            
            
            axs[1,1].legend(loc='upper right')
            axs[1,1].set_xlim(-1, 2.0)
            # axs[1,1].set_ylim(-0.2, target_thresh+1)
            axs[1,1].set_ylim(-0.2, y_top)
            axs[1,1].spines['right'].set_visible(False)
            axs[1,1].spines['top'].set_visible(False)
            axs[1,1].set_ylabel('Joystick deflection (deg)')
            
            # reward aligned
            # fig3, axs3 = plt.subplots(1, figsize=(10, 4))
            # axs[2].subplots_adjust(hspace=0.7)
            axs[1,2].plot(encoder_times_rew, encoder_pos_avg_rew_long,'-', label='Average Trajectory')
            axs[1,2].axvline(x = 0, color = 'r', label = 'Reward', linestyle='--')
            axs[1,2].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
            # axs[1,2].set_title(subject + ' - ' + session_date_formatted)
            axs[1,2].set_title('Reward Aligned.\n')    
            axs[1,2].legend(loc='upper right')               
            axs[1,2].set_xlim(-1.0, 1.5)
            # axs[1,2].set_ylim(-0.2, target_thresh+1)
            axs[1,2].set_ylim(-0.2, y_top)
            axs[1,2].spines['right'].set_visible(False)
            axs[1,2].spines['top'].set_visible(False)
            axs[1,2].set_xlabel('Time from Reward (s)')
            axs[1,2].set_ylabel('Joystick deflection (deg)')            
            
            
            
            
            
            fig.tight_layout()
            
            # output_imgs_dir = 'C:\\data analysis\\behavior\\joystick\\figures\\'+subject+'\\' + 'avg_trajectory_imgs\\'        
            
            
            
            # output_figs_dir = 'C:/Users/timst/OneDrive - Georgia Institute of Technology/Najafi_Lab/0_Data_analysis/Behavior/Joystick/' +subject+'\\'
            # output_imgs_dir = 'C:/Users/timst/OneDrive - Georgia Institute of Technology/Najafi_Lab/0_Data_analysis/Behavior/Joystick/'+subject+'\\' + 'avg_trajectory_imgs\\'        
            
            if savefiles:
                # these for saving files
                output_figs_dir = output_dir_onedrive + subject + '/'    
                # output_imgs_dir = 'C:/Users/timst/OneDrive - Georgia Institute of Technology/Najafi_Lab/0_Data_analysis/Behavior/Joystick/'+subject+'\\' + 'avg_trajectory_imgs\\'        
                output_imgs_dir = output_dir_local + subject + '/avg_trajectory_imgs/'
                os.makedirs(output_figs_dir, exist_ok = True)
                os.makedirs(output_imgs_dir, exist_ok = True)
                output_pdf_filename = output_figs_dir + subject + '_all_sess_trajectory_ave_trs'
                save_image(output_pdf_filename)
                fig.savefig(output_imgs_dir + subject +'_'+ session_date_formatted + '_trajectory_ave_trs' + '.png', dpi=300)
            
    
            # filename = 'C:\\data analysis\\behavior\\joystick\\figures\\'+subject+'\\'+'fig2_'+subject+'_avg_trajectory'  
            # os.makedirs('C:\\behavior\\joystick\\figures\\'+subject+'\\'+dates[i], exist_ok = True)
            # save_image(filename)
            # fig.savefig('C:\\behavior\\joystick\\figures\\'+subject+'\\'+dates[i]+'\\fig2_'+subject+'_avg_trajectory.png', dpi=300)
            # plt.close(fig)        
            
            
        
    print('Completed fig2 trajectories for ' + subject)
    print()
    if savefiles:
        plt.close("all")


# debugging

session_data = session_data_1
plot_fig2(session_data,         
          output_dir_onedrive,
          output_dir_local)

# session_data = session_data_2
# plot_fig2(session_data,         
#           output_dir_onedrive,
#           output_dir_local)

session_data = session_data_3
plot_fig2(session_data,         
          output_dir_onedrive,
          output_dir_local)

# session_data = session_data_4
# plot_fig2(session_data,         
#           output_dir_onedrive,
#           output_dir_local)

# session_data = session_data_5
# plot_fig2(session_data,         
#           output_dir_onedrive,
#           output_dir_local)

# session_data = session_data_6
# plot_fig2(session_data,         
#           output_dir_onedrive,
#           output_dir_local)