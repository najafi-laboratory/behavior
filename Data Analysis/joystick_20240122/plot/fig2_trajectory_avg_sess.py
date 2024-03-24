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
        max_sessions=10
        ):
    

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
    
    for i in range(start_idx, session_data['total_sessions']):
        # print(i)
        
        numTrials = len(session_data['outcomes'][i])
        numRewardedTrials = len(session_data['rewarded_trials'][i])
                
        session_date_formatted = dates[i][2:]       
        
        press_window = session_data['session_press_window']
        
        vis_stim_2_enable = session_data['vis_stim_2_enable']
        
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
        
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
        fig.subplots_adjust(hspace=0.7)
        fig.suptitle(subject + ' - ' + session_date_formatted + '  ' + str(numRewardedTrials) + '/' + str(numTrials) + ' Trials Rewarded.\nPress Window:' + ' ' + str(press_window) + 's')
        
        y_top = 3.5
        
        # vis 1 aligned
        axs[0].plot(encoder_times_vis1, encoder_pos_avg_vis1,'-', label='Average Trajectory')
        axs[0].axvline(x = 0, color = 'r', label = 'VisStim1', linestyle='--')
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
        axs[1].plot(encoder_times_vis2, encoder_pos_avg_vis2,'-', label='Average Trajectory')
        if vis_stim_2_enable:
            axs[1].axvline(x = 0, color = 'r', label = 'VisStim2', linestyle='--')
            axs[1].set_title('VisStim2 Aligned.\n')
            axs[1].set_xlabel('Time from VisStim2 (s)')
        else:
            axs[1].axvline(x = 0, color = 'r', label = 'WaitForPress2', linestyle='--')
            axs[1].set_title('WaitForPress2 Aligned.\n')
            axs[1].set_xlabel('Time from WaitForPress2 (s)')
            
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
        axs[2].plot(encoder_times_rew, encoder_pos_avg_rew,'-', label='Average Trajectory')
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
        output_figs_dir = 'C:/Users/gtg424h/OneDrive - Georgia Institute of Technology/Najafi_Lab/0_Data_analysis/Joystick/' +subject+'\\'
        # output_imgs_dir = 'C:\\data analysis\\behavior\\joystick\\figures\\'+subject+'\\' + 'avg_trajectory_imgs\\'        
        output_imgs_dir = 'C:/Users/gtg424h/OneDrive - Georgia Institute of Technology/Najafi_Lab/0_Data_analysis/Joystick/'+subject+'\\' + 'avg_trajectory_imgs\\'        
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
    plt.close("all")


# debugging

# session_data = session_data_1
# plot_fig2(session_data)

# session_data = session_data_2
# plot_fig2(session_data)
    
# session_data = session_data_3
# plot_fig2(session_data)

# session_data = session_data_4
# plot_fig2(session_data)
