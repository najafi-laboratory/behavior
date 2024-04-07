import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfFileReader, PdfFileWriter, PdfWriter, PdfReader
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

def plot_fig3(
        session_data,
        online_dir,
        local_dir
        ):
    
    max_sessions=10,
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
    
    numSessions = session_data['total_sessions']
    
    for i in range(start_idx, numSessions):
    # for i in range(start_idx, 1):
        
        session_date = dates[i][2:]
                
        print('plotting trajectories for ' + subject + ' session ', session_date)
        
        TrialOutcomes = session_data['outcomes'][i]

        numTrials = len(session_data['outcomes'][i])
        RewardedTrials = session_data['rewarded_trials'][i]
        numRewardedTrials = len(RewardedTrials)
                
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
        
        num_rows = 10
        num_columns = 3
        
        plots_per_page = num_rows * num_columns
                
        # num_pages = int(np.ceil(numRewardedTrials/plots_per_page))
        num_pages = int(np.ceil(numTrials/plots_per_page))
        
        # num_plots_bottom_page = int(numRewardedTrials - (plots_per_page * (num_pages - 1)))
        num_plots_bottom_page = int(numTrials - (plots_per_page * (num_pages - 1)))
        num_rows_bottom_page = int(np.ceil(num_plots_bottom_page / num_columns))
        
        current_page = 1
        
        top_left_trial = 0
        bottom_right_trial = top_left_trial + plots_per_page
        
        pdf_streams = []
        pdf_paths = []
        
        # num_pages = 2
        for page in range(0, num_pages):   
            # print()
            # print('current page', current_page)
            
            if current_page == num_pages:
                # print('bottom page ', current_page)
                bottom_right_trial = top_left_trial + num_plots_bottom_page
                # num_rows = num_rows_bottom_page
            
            fig, axs = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(20, 30))
            
            # fig.suptitle(subject + ' - ' + session_date + '  ' + str(numRewardedTrials) + '/' + str(numTrials) + ' Trials Rewarded.\nPress Window:' + ' ' + str(press_window) + 's\nVisStim1 Aligned.')
            fig.suptitle(subject + ' - ' + session_date + '  ' + str(numRewardedTrials) + '/' + str(numTrials) + ' Trials Rewarded.\nPress Window:' + ' ' + str(press_window) + 's\nVisStim1 Aligned.')            
            
            # fig.tight_layout(rect=[0, 0.03, 1, 0.98]) # [left, bottom, right, top]
            fig.tight_layout(rect=[0.01, 0.03, 1, 0.98]) # [left, bottom, right, top]
            fig.subplots_adjust(hspace=0.4)
            # fig.subplots_adjust(wspace=0.4, hspace=0.4) # if need more space horiz between subplots
            
            row = 0
            col = 0
            row_counter = 0
            col_counter = 0
            
            
            for trial in range(top_left_trial, bottom_right_trial):
            # for trial in range(0, 30):
                # print('trial idx', trial)
                # if trial == top_left_trial:
                #     print('top_left_trial ', top_left_trial)
                # elif trial == bottom_right_trial - 1:
                #     print('bottom_right_trial ', bottom_right_trial - 1)
                    
                if row == 10:
                    row = 0
                
                if col == 3:
                    row = row + 1
                    col = 0
                
                # if trial == numRewardedTrials:
                #     break
                
                # print(trial)
                encoder_positions_aligned_vis1 = session_data['encoder_positions_aligned_vis1'][i][trial]
                # encoder_positions_aligned_vis2 = session_data['encoder_positions_aligned_vis2'][i][trial]
                # encoder_positions_aligned_rew = session_data['encoder_positions_aligned_rew'][i][trial]
                
                y_top = 3.5
                
                if TrialOutcomes[trial] == 'Reward':
                    axs[row, col].plot(encoder_times_vis1, encoder_positions_aligned_vis1,'-', color = '#1f77b4', label='Trajectory')
                elif TrialOutcomes[trial] == 'Punish':
                    axs[row, col].plot(encoder_times_vis1, encoder_positions_aligned_vis1,'-', color = '#d62728', label='Trajectory')
                else:
                    axs[row, col].plot(encoder_times_vis1, encoder_positions_aligned_vis1,'-', color = '#e377c2', label='Trajectory')
                    
                axs[row, col].axvline(x = 0, color = 'r', label = 'VisStim1', linestyle='--')
                axs[row, col].axhline(y = target_thresh, color = '0.6', label = 'Target Threshold', linestyle='--')    
                axs[row, col].set_title('Trial ' + str(trial + 1))        
                axs[row, col].legend(loc='upper right')
                axs[row, col].set_xlim(time_left_VisStim1, 7.0)
                # axs[row, col].set_ylim(-0.2, target_thresh+1.5)
                axs[row, col].set_ylim(-0.2, y_top)
                axs[row, col].spines['right'].set_visible(False)
                axs[row, col].spines['top'].set_visible(False)
                axs[row, col].set_xlabel('Time from VisStim1 (s)')
                axs[row, col].set_ylabel('Joystick deflection (deg)')
                        
                col = col + 1
                    
                # trial_counter = trial_counter + 1
                                
            top_left_trial = bottom_right_trial
            bottom_right_trial = top_left_trial + plots_per_page                                                            
            current_page = current_page + 1
            
            output_pdf_dir = 'C:/Users/gtg424h/OneDrive - Georgia Institute of Technology/Najafi_Lab/0_Data_analysis/Joystick/' +subject+'\\'
            output_pdf_pages_dir = 'C:\\data analysis\\behavior\\joystick\\figures\\'+subject+'\\trajectories\\' + 'trajectories_' + session_date + '\\'
            os.makedirs(output_pdf_dir, exist_ok = True)
            os.makedirs(output_pdf_pages_dir, exist_ok = True)
            output_pdf_filename = output_pdf_pages_dir + subject +  session_date + '_trajectory_all_trs' + str(page)
            pdf_paths.append(output_pdf_filename + '.pdf')
            save_image(output_pdf_filename)        
            plt.close(fig)
            
            # output_imgs_dir = output_figs_dir + 'avg_trajectory_imgs\\'        
            # os.makedirs(output_imgs_dir, exist_ok = True)
            # output_pdf_filename = output_figs_dir + today_string + '_' + subject+'_avg_trajectory'
            # save_image(output_pdf_filename)
            # fig.savefig(output_imgs_dir + today_string + '_' + subject + '_avg_trajectory_' + session_date_formatted + '.png', dpi=300)
            
            
            
            
            
            
            # os.makedirs('C:\\behavior\\joystick\\figures\\'+subject+'\\trajectories_' + session_date + '\\', exist_ok = True)
            # filename = 'C:\\behavior\\joystick\\figures\\'+subject+'\\trajectories_' + session_date + '\\'+'fig3_'+subject+'_trajectory_' + str(page)
            # pdf_paths.append(filename + '.pdf')
            # save_image(filename)        
            # plt.close(fig)
        
        output = PdfWriter()
        # pdf_streams = []
        # pdf_paths = []
        # pdf_paths.append('C:\\behavior\\joystick\\figures\\YH4\\trajectories\\fig3_YH4_trajectory_0.pdf')
        # pdf_paths.append('C:\\behavior\\joystick\\figures\\YH4\\trajectories\\fig3_YH4_trajectory_1.pdf')
        pdf_files = []
        for pdf_path in pdf_paths:
            f = open(pdf_path, "rb")
            pdf_streams.append(PdfReader(f))
            pdf_files.append(f)
    
        for pdf_file_stream in pdf_streams:
            output.add_page(pdf_file_stream.pages[0])
        
        for pdf_file in pdf_files:
            pdf_file.close()
    
    
        # outputStream = open(r'C:\\data analysis\\behavior\\joystick\\figures\\'+subject+'\\' + today_string + '_'+subject+'_trajectory_combined_'+session_date+'.pdf', "wb")
        outputStream = open(r'' + output_pdf_dir + subject + '_' + session_date + '_trajectory_all_trs' + '.pdf', "wb")
        output.write(outputStream)
        outputStream.close()

    print('Completed fig3 trajectories for ' + subject)
    print()
    plt.close("all")


# debugging

# session_data = session_data_1
# plot_fig3(session_data)

# session_data = session_data_2
# plot_fig3(session_data)
    
# session_data = session_data_3
# plot_fig3(session_data)

# session_data = session_data_4
# plot_fig3(session_data)
