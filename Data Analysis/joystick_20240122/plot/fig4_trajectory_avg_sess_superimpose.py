import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import datetime
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

# def plot_fig4(
#         session_data,
#         max_sessions=10,
#         max_superimpose=3
#         ):
    
def plot_fig4(
        session_data,
        date_range_start,
        date_range_stop
        ):    
    
    print('plotting fig4')
    print(date_range_start)
    print(date_range_stop)
    subject = session_data['subject']
    # return if 0 total sessions
    if session_data['total_sessions'] == 0:
        print('no session data in folder for ' + subject)
        return
    
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    print('Subject ' + subject)
    print('Date List')
    for date in dates:
        print(date)
        
    
    # start_idx = 0
    start_idx = dates.index(date_range_start)
    stop_idx = dates.index(date_range_stop)
    
    dates = dates[start_idx:stop_idx+1]
    
    
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
    filename = 'C:\\behavior\\joystick\\figures\\'+subject+'\\'+'fig4_'+subject+'_avg_trajectory_superimpose'    
    
    # using now() to get current time
    current_time = datetime.datetime.now()
    year = current_time.year
    month = current_time.month
    day = current_time.day
    time_string = str(year) + str(month) + str(day)
    
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
    
    numTrials = []
    numRewardedTrials = []
    
    
    # palette = Plasma[256][palette_start:palette_start + session_data['total_sessions']]
    # palette = Plasma[256][palette_start:palette_start + max_superimpose]
    #Magma, Inferno, Plasma, Viridis, Cividis, Greys256
    #palette = bp.palettes.Plasma[256]
    # import bokeh as bp
    # palette = Greys[256]
    palette = Blues[256]
    palette = palette
    palette_idx = 1
    palette_indices = []
    
    # palette_luminosity_increment = int(np.floor(len(palette) / len(dates)))
    palette_luminosity_increment = int(np.floor(len(palette) / len(range(start_idx, stop_idx + 1))))
        
    # for i in range(0, session_data['total_sessions']):
    for i in range(start_idx, stop_idx + 1):
        if i > 0:
            palette_indices.append(palette_idx + palette_luminosity_increment)
        else:
            palette_indices.append(palette_idx)
        palette_idx = palette_idx + palette_luminosity_increment
    
    palette = [palette[i] for i in palette_indices]
    palette.reverse()
    
    # for i in range(0, session_data['total_sessions']):
    for i in range(start_idx, stop_idx + 1):   
        # print(i)
        
        numTrials.append(session_data['outcomes'][i])
        numRewardedTrials.append(len(session_data['rewarded_trials'][i]))

        encoder_pos_avg_vis1.append(session_data['encoder_pos_avg_vis1'][i])

        encoder_pos_avg_vis2.append(session_data['encoder_pos_avg_vis2'][i])

        encoder_pos_avg_rew.append(session_data['encoder_pos_avg_rew'][i])
        
        
        
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
        # axs[0].set_xlabel('trial time from VisStim1 [s]')
        # axs[0].set_ylabel('joystick deflection [deg]')
            
        # # vis 2 or waitforpress aligned
        # axs[1].plot(encoder_times_vis2, encoder_pos_avg_vis2,'-', label='Average Trajectory')
        # if vis_stim_2_enable:
        #     axs[1].axvline(x = 0, color = 'r', label = 'VisStim2', linestyle='--')
        #     axs[1].set_title('VisStim2 Aligned.\n')
        #     axs[1].set_xlabel('trial time from VisStim2 [s]')
        # else:
        #     axs[1].axvline(x = 0, color = 'r', label = 'WaitForPress2', linestyle='--')
        #     axs[1].set_title('WaitForPress2 Aligned.\n')
        #     axs[1].set_xlabel('trial time from WaitForPress2 [s]')
            
        # axs[1].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
        # axs[1].legend(loc='upper right')
        # axs[1].set_xlim(-1, 2.0)
        # axs[1].set_ylim(-0.2, target_thresh+1.25)
        # axs[1].spines['right'].set_visible(False)
        # axs[1].spines['top'].set_visible(False)
        # axs[1].set_ylabel('joystick deflection [deg]')
        
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
        # axs[2].set_xlabel('trial time from Reward [s]')
        # axs[2].set_ylabel('joystick deflection [deg]')
        
        
        # fig.tight_layout()
        # os.makedirs('./figures/'+subject+'/'+dates[i], exist_ok = True)
        # save_image(filename)
        # fig.savefig('./figures/'+subject+'/'+dates[i]+'/fig4_'+subject+'_avg_trajectory.png', dpi=300)
        # # plt.close(fig)
    
        
    
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
    fig.subplots_adjust(hspace=0.7)
    # fig.suptitle(subject + ' - ' + dates[i] + '  ' + str(numRewardedTrials) + '/' + str(numTrials) + ' Trials Rewarded.\nPress Window:' + ' ' + str(press_window) + 's')
    fig.suptitle(subject + ' Average Trajectories')
    
    # sessions_idxs = range(0, session_data['total_sessions'])
    sessions_idxs = range(0, len(dates))
    for j in sessions_idxs:
        print(j)
    # vis 1 aligned
    for i in sessions_idxs:        
        # if (i == 0) or (i == (session_data['total_sessions'] - 1)):            
        if (i == 0) or (i == stop_idx):            
            axs[0].plot(encoder_times_vis1, encoder_pos_avg_vis1[i],'-', color=palette[i], label=dates[i])
        else:
            axs[0].plot(encoder_times_vis1, encoder_pos_avg_vis1[i],'-', color=palette[i])
        
    # print(subject)
    axs[0].axvline(x = 0, color = 'r', label = 'VisStim1', linestyle='--')
    axs[0].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')    
    axs[0].set_title('VisStim1 Aligned.\n')        
    axs[0].legend(loc='upper right')
    axs[0].set_xlim(time_left_VisStim1, 4.0)
    axs[0].set_ylim(-0.2, target_thresh+1.25)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].set_xlabel('trial time from VisStim1 [s]')
    axs[0].set_ylabel('joystick deflection [deg]')
        
    # vis 2 or waitforpress aligned
    for i in sessions_idxs:        
        # if (i == 0) or (i == (session_data['total_sessions'] - 1)):
        if (i == 0) or (i == (len(list(sessions_idxs))-1)):                
            axs[1].plot(encoder_times_vis2, encoder_pos_avg_vis2[i],'-', color=palette[i], label=dates[i])
        else:
            axs[1].plot(encoder_times_vis2, encoder_pos_avg_vis2[i],'-', color=palette[i])
            
    if vis_stim_2_enable:
        axs[1].axvline(x = 0, color = 'r', label = 'VisStim2', linestyle='--')
        axs[1].set_title('VisStim2 Aligned.\n')
        axs[1].set_xlabel('trial time from VisStim2 [s]')
    else:
        axs[1].axvline(x = 0, color = 'r', label = 'WaitForPress2', linestyle='--')
        axs[1].set_title('WaitForPress2 Aligned.\n')
        axs[1].set_xlabel('trial time from WaitForPress2 [s]')
        
    axs[1].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
    axs[1].legend(loc='upper right')
    axs[1].set_xlim(-1, 2.0)
    axs[1].set_ylim(-0.2, target_thresh+1.25)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].set_ylabel('joystick deflection [deg]')
    
    # reward aligned
    # fig3, axs3 = plt.subplots(1, figsize=(10, 4))
    # axs[2].subplots_adjust(hspace=0.7)
    for i in sessions_idxs:        
        # if (i == 0) or (i == (session_data['total_sessions'] - 1)): 
        if (i == 0) or (i == (len(list(sessions_idxs))-1)):    
            axs[2].plot(encoder_times_rew, encoder_pos_avg_rew[i],'-', color=palette[i], label=dates[i])
        else:
            axs[2].plot(encoder_times_rew, encoder_pos_avg_rew[i],'-', color=palette[i])       
        
    axs[2].axvline(x = 0, color = 'r', label = 'Reward', linestyle='--')
    axs[2].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')
    # axs[2].set_title(subject + ' - ' + dates[i])
    axs[2].set_title('Reward Aligned.\n')    
    axs[2].legend(loc='upper right')               
    axs[2].set_xlim(-1.0, 1.5)
    axs[2].set_ylim(-0.2, target_thresh+1.25)
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['top'].set_visible(False)
    axs[2].set_xlabel('trial time from Reward [s]')
    axs[2].set_ylabel('joystick deflection [deg]')    
    
    fig.tight_layout()
    img_dir = 'C:\\behavior\\joystick\\figures\\'+subject+'\\fig4_' + time_string
    os.makedirs(img_dir, exist_ok = True)
    save_image(filename)
    fig.savefig(img_dir + '\\fig4_'+subject+'_avg_trajectory_superimpose.png', dpi=300)
    # plt.close(fig)
    
    print('Completed fig4 trajectories superimposed for ' + subject)
    print()
    plt.close("all")


# debugging

# session_data = session_data_1
# plot_fig4(session_data)

# session_data = session_data_2
# plot_fig4(session_data)
    
# session_data = session_data_3
# plot_fig4(session_data)

# session_data = session_data_4
# plot_fig4(session_data)
