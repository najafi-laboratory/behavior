import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import date
from bokeh.plotting import figure, output_file, show  
from bokeh.palettes import Magma, Inferno, Plasma, Viridis, Cividis, Greys, Blues, Reds, Greens
# import bokeh.palettes as bp
# import bokeh as bp
# https://docs.bokeh.org/en/latest/docs/reference/palettes.html


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

# def plot_fig5(
#         session_data,
#         max_sessions=10,
#         max_superimpose=3
#         ):
    
def plot_fig5(
        session_data,
        output_dir_onedrive, 
        output_dir_local
        ):    
    
    savefiles = 0
    # print(date_range_start)
    # print(date_range_stop)
    
    subject = session_data['subject']
    
    print('plotting superimposed trajectories for ' + subject)  # add dates selected in update
    
    # return if < 2 total sessions
    if session_data['total_sessions'] < 2:
        print('less than 2 sessions loaded for ' + subject)
        return
    
    
    
    # print('plotting superimposed trajectories for ' + subject + ' session ', session_date)  # add dates selected in update
    
    
    
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    

    # print('Subject ' + subject)
    # print('Date List')
    # for date_i in dates:
    #     print(date_i)
        
    
    start_idx = 0
    stop_idx = session_data['total_sessions'] - 1
    # start_idx = dates.index(date_range_start)
    # stop_idx = dates.index(date_range_stop)
    
    # dates = dates[start_idx:stop_idx+1]
    
    
    # if max_sessions != -1 and len(dates) > max_sessions:
    #     start_idx = len(dates) - max_sessions
    # super_start_idx = 0
    # if max_superimpose != -1 and len(dates) > max_superimpose:
    #     super_start_idx = len(dates) - max_superimpose
    # super_dates = session_data['dates'][super_start_idx:]
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
    
    # numTrials = len(session_data['outcomes'][i])
    # numRewardedTrials = len(session_data['rewarded_trials'][i])
    
    press_window = session_data['session_press_window']
    
    vis_stim_2_enable = session_data['vis_stim_2_enable']
    
    encoder_times_vis1 = session_data['encoder_times_aligned_VisStim1']
    # encoder_pos_avg_vis1 = session_data['encoder_pos_avg_vis1'][i]
    encoder_times_vis2 = session_data['encoder_times_aligned_VisStim2']
    # encoder_pos_avg_vis2 = session_data['encoder_pos_avg_vis2'][i]
    encoder_times_rew = session_data['encoder_times_aligned_Reward']
    # encoder_pos_avg_rew = session_data['encoder_pos_avg_rew'][i]
    
    time_left_VisStim1 = session_data['time_left_VisStim1']
    time_right_VisStim1 = session_data['time_right_VisStim1']
    
    time_left_VisStim2 = session_data['time_left_VisStim2']
    time_right_VisStim2 = session_data['time_right_VisStim2']
    
    time_left_rew = session_data['time_left_rew']
    time_right_rew = session_data['time_right_rew']
    
    target_thresh = session_data['session_target_thresh']
    
    encoder_pos_avg_vis1 = []
    encoder_pos_avg_vis2 = []
    encoder_pos_avg_rew = []
    
    encoder_pos_avg_vis1_short = []
    encoder_pos_avg_vis1_long = []

    encoder_pos_avg_vis2_short = []
    encoder_pos_avg_vis2_long = []
    
    encoder_pos_avg_rew_short = []
    encoder_pos_avg_rew_long = []
    
    numTrials = []
    numRewardedTrials = []
    
    
    # palette = Plasma[256][palette_start:palette_start + session_data['total_sessions']]
    # palette = Plasma[256][palette_start:palette_start + max_superimpose]
    #Magma, Inferno, Plasma, Viridis, Cividis, Greys256
    #palette = bp.palettes.Plasma[256]
    # import bokeh as bp
    # palette = Greys[256]
    palette = Blues[256]
    palette = palette[0:180]
    palette_idx = 0
    palette_indices = []
    
    # palette_luminosity_increment = int(np.floor(len(palette) / len(dates)))
    palette_luminosity_increment = int(np.floor(len(palette) / (len(range(start_idx, stop_idx + 1))-1)))-1
        
    for i in range(0, session_data['total_sessions']):
    # for i in range(start_idx, stop_idx + 1):
        # if i > 0:
        #     palette_indices.append(palette_idx + palette_luminosity_increment)
        # else:
        #     palette_indices.append(palette_idx)
        
        palette_indices.append(palette_idx)
        palette_idx = palette_idx + palette_luminosity_increment
    
    palette = [palette[i] for i in palette_indices]
    palette.reverse()
    
    singleRow = []     
    isOnlyShort = []
    isOnlyLong = []   
    sessionIdxsToPlot = []
    
    # for i in range(0, session_data['total_sessions']):
    for i in range(start_idx, stop_idx + 1):   
        # print(i)
        
        isSelfTimedMode = session_data['isSelfTimedMode'][i][0]
        isShortDelay = session_data['isShortDelay'][i]
        press_reps = session_data['press_reps'][i]
        press_window = session_data['press_window'][i]  
        
        print('session', i)
        print('isSelfTimedMode', isSelfTimedMode)
        print('isShortDelay', isShortDelay)
        print('press_reps', press_reps)
        print('press_window', press_window)
        
        
        numTrials.append(session_data['outcomes'][i])
        numRewardedTrials.append(len(session_data['rewarded_trials'][i]))
        
        session_date = dates[i][2:] 

        encoder_pos_avg_vis1.append(session_data['encoder_pos_avg_vis1'][i])

        encoder_pos_avg_vis2.append(session_data['encoder_pos_avg_vis2'][i])

        encoder_pos_avg_rew.append(session_data['encoder_pos_avg_rew'][i])
        
        
        encoder_pos_avg_vis1_short.append(session_data['encoder_pos_avg_vis1_short'][i])
        encoder_pos_avg_vis1_long.append(session_data['encoder_pos_avg_vis1_long'][i])

        encoder_pos_avg_vis2_short.append(session_data['encoder_pos_avg_vis2_short'][i])
        encoder_pos_avg_vis2_long.append(session_data['encoder_pos_avg_vis2_long'][i])
        
        encoder_pos_avg_rew_short.append(session_data['encoder_pos_avg_rew_short'][i])
        encoder_pos_avg_rew_long.append(session_data['encoder_pos_avg_rew_long'][i])
        
        # encoder_pos_avg_vis1_short = session_data['encoder_pos_avg_vis1_short'][i]
        # encoder_pos_avg_vis1_long = session_data['encoder_pos_avg_vis1_long'][i]

        # encoder_pos_avg_vis2_short = session_data['encoder_pos_avg_vis2_short'][i]
        # encoder_pos_avg_vis2_long = session_data['encoder_pos_avg_vis2_long'][i]
        
        # encoder_pos_avg_rew_short = session_data['encoder_pos_avg_rew_short'][i]
        # encoder_pos_avg_rew_long = session_data['encoder_pos_avg_rew_long'][i]
                
        # singleRow = 0        
        # isOnlyShort = 0
        # isOnlyLong = 0
        

        # type(encoder_pos_avg_vis1_short) is np.ndarray
        if not isSelfTimedMode:
            # if len(encoder_pos_avg_vis1_short) < 2:
            if not (type(encoder_pos_avg_vis1_short) is np.ndarray):
                # isOnlyLong = 1
                # singleRow = 1
                isOnlyLong.append(1)
                isOnlyShort.append(0)
                singleRow.append(1)
            # elif len(encoder_pos_avg_vis1_long) < 2:
            elif not (type(encoder_pos_avg_vis1_long) is np.ndarray):
                # isOnlyShort = 1
                # singleRow = 1
                isOnlyShort.append(1)
                isOnlyLong.append(0)
                singleRow.append(1)
            else:
                isOnlyShort.append(0)
                isOnlyLong.append(0)
                singleRow.append(0)
                sessionIdxsToPlot.append(i)
        else:
            # singleRow = 1  
            isOnlyShort.append(0)
            isOnlyLong.append(0)
            singleRow.append(1)
            
        print(i)
        # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
        # fig.subplots_adjust(hspace=0.7)
        # fig.suptitle(subject + ' - ' + dates[i] + '  ' + str(numRewardedTrials) + '/' + str(numTrials) + ' Trials Rewarded.\nPress Window:' + ' ' + str(press_window) + 's')
        
        # # vis 1 aligned
        # axs[0].plot(encoder_times_vis1, encoder_pos_avg_vis1,'-', label='Average Trajectory')
        # axs[0].axvline(x = 0, color = 'r', label = 'VisStim1', linestyle='--')
        # axs[0].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')    
        # axs[0].set_title('VisStim1 Aligned.\n')        
        # axs[0].legend(loc='upper right')
        # axs[0].set_xlim(time_left_VisStim1, 4.0)
        # axs[0].set_ylim(-0.2, target_thresh+1.25)
        # axs[0].spines['right'].set_visible(False)
        # axs[0].spines['top'].set_visible(False)
        # axs[0].set_xlabel('Time from VisStim1 (s)')
        # axs[0].set_ylabel('Joystick deflection (deg)')
            
        # # vis 2 or waitforpress aligned
        # axs[1].plot(encoder_times_vis2, encoder_pos_avg_vis2,'-', label='Average Trajectory')
        # if vis_stim_2_enable:
        #     axs[1].axvline(x = 0, color = 'r', label = 'VisStim2', linestyle='--')
        #     axs[1].set_title('VisStim2 Aligned.\n')
        #     axs[1].set_xlabel('Time from VisStim2 (s)')
        # else:
        #     axs[1].axvline(x = 0, color = 'r', label = 'WaitForPress2', linestyle='--')
        #     axs[1].set_title('WaitForPress2 Aligned.\n')
        #     axs[1].set_xlabel('Time from WaitForPress2 (s)')
            
        # axs[1].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
        # axs[1].legend(loc='upper right')
        # axs[1].set_xlim(-1, 2.0)
        # axs[1].set_ylim(-0.2, target_thresh+1.25)
        # axs[1].spines['right'].set_visible(False)
        # axs[1].spines['top'].set_visible(False)
        # axs[1].set_ylabel('Joystick deflection (deg)')
        
        # # reward aligned
        # # fig3, axs3 = plt.subplots(1, figsize=(10, 4))
        # # axs[2].subplots_adjust(hspace=0.7)
        # axs[2].plot(encoder_times_rew, encoder_pos_avg_rew,'-', label='Average Trajectory')
        # axs[2].axvline(x = 0, color = 'r', label = 'Reward', linestyle='--')
        # axs[2].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
        # # axs[2].set_title(subject + ' - ' + dates[i])
        # axs[2].set_title('Reward Aligned.\n')    
        # axs[2].legend(loc='upper right')               
        # axs[2].set_xlim(-1.0, 1.5)
        # axs[2].set_ylim(-0.2, target_thresh+1.25)
        # axs[2].spines['right'].set_visible(False)
        # axs[2].spines['top'].set_visible(False)
        # axs[2].set_xlabel('Time from Reward (s)')
        # axs[2].set_ylabel('Joystick deflection (deg)')
        
        
        # fig.tight_layout()
        # os.makedirs('./figures/'+subject+'/'+dates[i], exist_ok = True)
        # save_image(filename)
        # fig.savefig('./figures/'+subject+'/'+dates[i]+'/fig4_'+subject+'_avg_trajectory.png', dpi=300)
        # # plt.close(fig)
    
    # if singleRow:    
    
        
    
    
    #     fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
    #     fig.subplots_adjust(hspace=0.7)
    #     # fig.suptitle(subject + ' - ' + dates[i] + '  ' + str(numRewardedTrials) + '/' + str(numTrials) + ' Trials Rewarded.\nPress Window:' + ' ' + str(press_window) + 's')
    #     fig.suptitle(subject + ' Average Trajectories Superimposed')
        
    #     # sessions_idxs = range(0, session_data['total_sessions'])
    #     sessions_idxs = range(0, len(dates))
    #     # for j in sessions_idxs:
    #     #     print(j)
        
        
        
    #     # vis 1 aligned
    #     for i in sessions_idxs:
    #         # isSelfTimedMode = session_data['isSelfTimedMode'][i]
    #         # isShortDelay = session_data['isShortDelay'][i]
    #         # press_reps = session_data['press_reps'][i]
    #         # press_window = session_data['press_window'][i]  
            
    #         # print('session', i)
    #         # print('isSelfTimedMode', isSelfTimedMode)
    #         # print('isShortDelay', isShortDelay)
    #         # print('press_reps', press_reps)
    #         # print('press_window', press_window)
            
    #         # if (i == 0) or (i == (session_data['total_sessions'] - 1)):            
    #         if (i == 0) or (i == stop_idx):            
    #             axs[0].plot(encoder_times_vis1, encoder_pos_avg_vis1[i],'-', color=palette[i], label=dates[i][2:])
    #         else:
    #             axs[0].plot(encoder_times_vis1, encoder_pos_avg_vis1[i],'-', color=palette[i])
            
    #     y_top = 3.5
        
        
        
    #     # print(subject)
    #     axs[0].axvline(x = 0, color = 'r', label = 'VisStim1', linestyle='--')
        
    #     # update to get average threshold across range of sessions superimposed
    #     # temp thresh
    #     target_thresh = 2.0
    #     # axs[0].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')    
    #     axs[0].axhline(y = 'add avg thresh', color = '0.6', label = 'Target Threshold', linestyle='--')
        
        
    #     axs[0].set_title('VisStim1 Aligned.\n')        
    #     axs[0].legend(loc='upper right')
    #     axs[0].set_xlim(time_left_VisStim1, 4.0)
    #     # axs[0].set_ylim(-0.2, target_thresh+1.25)
    #     axs[0].set_ylim(-0.2, target_thresh+1.25)
    #     axs[0].spines['right'].set_visible(False)
    #     axs[0].spines['top'].set_visible(False)
    #     axs[0].set_xlabel('Time from VisStim1 (s)')
    #     axs[0].set_ylabel('Joystick deflection (deg)')
            
    #     # vis 2 or waitforpress aligned
    #     for i in sessions_idxs:        
    #         # if (i == 0) or (i == (session_data['total_sessions'] - 1)):
    #         if (i == 0) or (i == (len(list(sessions_idxs))-1)):                
    #             axs[1].plot(encoder_times_vis2, encoder_pos_avg_vis2[i],'-', color=palette[i], label=dates[i][2:])
    #         else:
    #             axs[1].plot(encoder_times_vis2, encoder_pos_avg_vis2[i],'-', color=palette[i])
                
    #     if vis_stim_2_enable:
    #         axs[1].axvline(x = 0, color = 'r', label = 'VisStim2', linestyle='--')
    #         axs[1].set_title('VisStim2 Aligned.\n')
    #         axs[1].set_xlabel('Time from VisStim2 (s)')
    #     else:
    #         axs[1].axvline(x = 0, color = 'r', label = 'WaitForPress2', linestyle='--')
    #         axs[1].set_title('WaitForPress2 Aligned.\n')
    #         axs[1].set_xlabel('Time from WaitForPress2 (s)')
            
    #     axs[1].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
    #     axs[1].legend(loc='upper right')
    #     axs[1].set_xlim(-1, 2.0)
    #     # axs[1].set_ylim(-0.2, target_thresh+1.25)
    #     axs[1].set_ylim(-0.2, y_top)
    #     axs[1].spines['right'].set_visible(False)
    #     axs[1].spines['top'].set_visible(False)
    #     axs[1].set_ylabel('Joystick deflection (deg)')
        
    #     # reward aligned
    #     # fig3, axs3 = plt.subplots(1, figsize=(10, 4))
    #     # axs[2].subplots_adjust(hspace=0.7)
    #     for i in sessions_idxs:        
    #         # if (i == 0) or (i == (session_data['total_sessions'] - 1)): 
    #         if (i == 0) or (i == (len(list(sessions_idxs))-1)):    
    #             axs[2].plot(encoder_times_rew, encoder_pos_avg_rew[i],'-', color=palette[i], label=dates[i][2:])
    #         else:
    #             axs[2].plot(encoder_times_rew, encoder_pos_avg_rew[i],'-', color=palette[i])       
            
    #     axs[2].axvline(x = 0, color = 'r', label = 'Reward', linestyle='--')
    #     axs[2].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
    #     # axs[2].set_title(subject + ' - ' + session_date)
    #     axs[2].set_title('Reward Aligned.\n')    
    #     axs[2].legend(loc='upper right')               
    #     axs[2].set_xlim(-1.0, 1.5)
    #     # axs[2].set_ylim(-0.2, target_thresh+1.25)
    #     axs[2].set_ylim(-0.2, y_top)
    #     axs[2].spines['right'].set_visible(False)
    #     axs[2].spines['top'].set_visible(False)
    #     axs[2].set_xlabel('Time from Reward (s)')
    #     axs[2].set_ylabel('Joystick deflection (deg)')    
        
    #     fig.tight_layout()
    #     # filename = 'C:\\data analysis\\behavior\\joystick\\figures\\'+subject+'\\'+'fig4_'+subject+'_avg_trajectory_superimpose'    
        
    #     # img_dir = 'C:\\data analysis\\behavior\\joystick\\figures\\'+subject+'\\fig4_' + today_string
    #     # os.makedirs(img_dir, exist_ok = True)
    #     # save_image(filename)
    #     # fig.savefig(img_dir + '\\fig4_'+subject+'_avg_trajectory_superimpose.png', dpi=300)
    #     # plt.close(fig)
        
        
    #     # output_imgs_dir = 'C:\\data analysis\\behavior\\joystick\\figures\\'+subject+'\\' + 'avg_trajectory_superimpose_imgs\\'        
        
        
    #     # output_figs_dir = 'C:/Users/timst/OneDrive - Georgia Institute of Technology/Najafi_Lab/0_Data_analysis/Behavior/Joystick/' +subject+'\\'
    #     # output_imgs_dir = 'C:/Users/timst/OneDrive - Georgia Institute of Technology/Najafi_Lab/0_Data_analysis/Behavior/Joystick/'+subject+'\\' + 'avg_trajectory_superimpose_imgs\\'        
        
    #     if savefiles:
    #         output_figs_dir = output_dir_onedrive + subject + '/'            
    #         output_imgs_dir = output_dir_local + subject + '/avg_trajectory_superimpose_imgs/'
    #         os.makedirs(output_figs_dir, exist_ok = True)
    #         os.makedirs(output_imgs_dir, exist_ok = True)
    #         output_pdf_filename = output_figs_dir + subject+'_Trajectory_sup_short_long'
    #         save_image(output_pdf_filename)
    #         fig.savefig(output_imgs_dir + subject + '_Trajectory_sup_short_long' + '.png', dpi=300)
        
    #     # output_figs_dir = output_dir_onedrive + subject + '/'            
    #     # output_imgs_dir = output_dir_local + subject + '/avg_trajectory_imgs/'
    # else:
        
        
        # check if any sessions have short/long, then plot only those
        
        
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 8))
    fig.subplots_adjust(hspace=0.7)
    # fig.suptitle(subject + ' - ' + dates[i] + '  ' + str(numRewardedTrials) + '/' + str(numTrials) + ' Trials Rewarded.\nPress Window:' + ' ' + str(press_window) + 's')
    fig.suptitle(subject + ' Average Trajectories Superimposed')
    
    # sessions_idxs = range(0, session_data['total_sessions'])
    sessions_idxs = range(0, len(dates))
    # for j in sessions_idxs:
    #     print(j)
    
    
    
    # vis 1 aligned short delay
        
    stop_idx = sessionIdxsToPlot[-1]
    # for i in sessions_idxs:
    for i in sessionIdxsToPlot:
        # isSelfTimedMode = session_data['isSelfTimedMode'][i]
        # isShortDelay = session_data['isShortDelay'][i]
        # press_reps = session_data['press_reps'][i]
        # press_window = session_data['press_window'][i]  
        
        # print('session', i)
        # print('isSelfTimedMode', isSelfTimedMode)
        # print('isShortDelay', isShortDelay)
        # print('press_reps', press_reps)
        # print('press_window', press_window)
        
        # if (i == 0) or (i == (session_data['total_sessions'] - 1)):

        # needs enough sessions so that vis1_short > 1 length
        if (i == 0) or (i == stop_idx):            
            axs[0, 0].plot(encoder_times_vis1, encoder_pos_avg_vis1_short[i],'-', color=palette[i], label=dates[i][2:])
        else:
            axs[0, 0].plot(encoder_times_vis1, encoder_pos_avg_vis1_short[i],'-', color=palette[i])
        
    y_top = 3.5
    
    
    
    # print(subject)
    axs[0, 0].axvline(x = 0, color = 'r', label = 'VisStim1', linestyle='--')
    
    # update to get average threshold across range of sessions superimposed
    # temp thresh
    target_thresh = 2.0
    # axs[0, 0].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')    
    axs[0, 0].axhline(y = 'add avg thresh', color = '0.6', label = 'Target Threshold', linestyle='--')
    
    
    axs[0, 0].set_title('VisStim1 Aligned.\n')        
    axs[0, 0].legend(loc='upper right')
    axs[0, 0].set_xlim(time_left_VisStim1, 4.0)
    # axs[0, 0].set_ylim(-0.2, target_thresh+1.25)
    axs[0, 0].set_ylim(-0.2, target_thresh+1.25)
    axs[0, 0].spines['right'].set_visible(False)
    axs[0, 0].spines['top'].set_visible(False)
    axs[0, 0].set_xlabel('Time from VisStim1 (s)')
    axs[0, 0].set_ylabel('Joystick deflection (deg)')
        
    
    # vis 2 or waitforpress aligned - short delay
    
    
    # for i in sessions_idxs:
    for i in sessionIdxsToPlot:      
        # if (i == 0) or (i == (session_data['total_sessions'] - 1)):
        if (i == 0) or (i == (len(list(sessions_idxs))-1)):                
            axs[0, 1].plot(encoder_times_vis2, encoder_pos_avg_vis2_short[i],'-', color=palette[i], label=dates[i][2:])
        else:
            axs[0, 1].plot(encoder_times_vis2, encoder_pos_avg_vis2_short[i],'-', color=palette[i])
            
    if vis_stim_2_enable:
        axs[0, 1].axvline(x = 0, color = 'r', label = 'VisStim2', linestyle='--')
        axs[0, 1].set_title('VisStim2 Aligned.\n')
        axs[0, 1].set_xlabel('Time from VisStim2 (s)')
    else:
        axs[0, 1].axvline(x = 0, color = 'r', label = 'WaitForPress2', linestyle='--')
        axs[0, 1].set_title('WaitForPress2 Aligned.\n')
        axs[0, 1].set_xlabel('Time from WaitForPress2 (s)')
        
    axs[0, 1].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
    axs[0, 1].legend(loc='upper right')
    axs[0, 1].set_xlim(-1, 2.0)
    # axs[0, 1].set_ylim(-0.2, target_thresh+1.25)
    axs[0, 1].set_ylim(-0.2, y_top)
    axs[0, 1].spines['right'].set_visible(False)
    axs[0, 1].spines['top'].set_visible(False)
    axs[0, 1].set_ylabel('Joystick deflection (deg)')
    
    # reward aligned - short delay
    # fig3, axs3 = plt.subplots(1, figsize=(10, 4))
    # axs[0, 2].subplots_adjust(hspace=0.7)
    
    
    
    # for i in sessions_idxs:
    for i in sessionIdxsToPlot:       
        # if (i == 0) or (i == (session_data['total_sessions'] - 1)): 
        if (i == 0) or (i == (len(list(sessions_idxs))-1)):    
            axs[0, 2].plot(encoder_times_rew, encoder_pos_avg_rew[i],'-', color=palette[i], label=dates[i][2:])
        else:
            axs[0, 2].plot(encoder_times_rew, encoder_pos_avg_rew[i],'-', color=palette[i])       
        
    axs[0, 2].axvline(x = 0, color = 'r', label = 'Reward', linestyle='--')
    axs[0, 2].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
    # axs[0, 2].set_title(subject + ' - ' + session_date)
    axs[0, 2].set_title('Reward Aligned.\n')    
    axs[0, 2].legend(loc='upper right')               
    axs[0, 2].set_xlim(-1.0, 1.5)
    # axs[0, 2].set_ylim(-0.2, target_thresh+1.25)
    axs[0, 2].set_ylim(-0.2, y_top)
    axs[0, 2].spines['right'].set_visible(False)
    axs[0, 2].spines['top'].set_visible(False)
    axs[0, 2].set_xlabel('Time from Reward (s)')
    axs[0, 2].set_ylabel('Joystick deflection (deg)')    
    
    fig.tight_layout()
    # filename = 'C:\\data analysis\\behavior\\joystick\\figures\\'+subject+'\\'+'fig4_'+subject+'_avg_trajectory_superimpose'    
    
    # img_dir = 'C:\\data analysis\\behavior\\joystick\\figures\\'+subject+'\\fig4_' + today_string
    # os.makedirs(img_dir, exist_ok = True)
    # save_image(filename)
    # fig.savefig(img_dir + '\\fig4_'+subject+'_avg_trajectory_superimpose.png', dpi=300)
    # plt.close(fig)
    
    
    # output_imgs_dir = 'C:\\data analysis\\behavior\\joystick\\figures\\'+subject+'\\' + 'avg_trajectory_superimpose_imgs\\'        
    
    
    # output_figs_dir = 'C:/Users/timst/OneDrive - Georgia Institute of Technology/Najafi_Lab/0_Data_analysis/Behavior/Joystick/' +subject+'\\'
    # output_imgs_dir = 'C:/Users/timst/OneDrive - Georgia Institute of Technology/Najafi_Lab/0_Data_analysis/Behavior/Joystick/'+subject+'\\' + 'avg_trajectory_superimpose_imgs\\'        
    
    if savefiles:
        output_figs_dir = output_dir_onedrive + subject + '/'            
        output_imgs_dir = output_dir_local + subject + '/avg_trajectory_superimpose_imgs/'
        os.makedirs(output_figs_dir, exist_ok = True)
        os.makedirs(output_imgs_dir, exist_ok = True)
        output_pdf_filename = output_figs_dir + subject+'_Trajectory_sup_short_long'
        save_image(output_pdf_filename)
        fig.savefig(output_imgs_dir + subject + '_Trajectory_sup_short_long' + '.png', dpi=300)
    
    # output_figs_dir = output_dir_onedrive + subject + '/'            
    # output_imgs_dir = output_dir_local + subject + '/avg_trajectory_imgs/'        
    
    
    print('Completed fig5 trajectories superimposed short/long for ' + subject)
    print()
    
    if savefiles:
        plt.close("all")


# debugging

session_data = session_data_1
plot_fig5(session_data,         
          output_dir_onedrive,
          output_dir_local)

# session_data = session_data_2
# plot_fig5(session_data,         
          # output_dir_onedrive,
          # output_dir_local)
    
# session_data = session_data_3
# plot_fig5(session_data,         
          # output_dir_onedrive,
          # output_dir_local)

# session_data = session_data_4
# plot_fig5(session_data,         
          # output_dir_onedrive,
          # output_dir_local)
