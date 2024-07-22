import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfFileReader, PdfFileWriter, PdfWriter, PdfReader
from datetime import date
from statistics import mean 

def save_image(filename): 
    
    p = PdfPages(filename+'.pdf') 
    fig_nums = plt.get_fignums()   
    figs = [plt.figure(n) for n in fig_nums] 
      
    for fig in figs:  
        
        fig.savefig(p, format='pdf', dpi=300) # type: ignore
           
    p.close() 
def plot_fig__trajectories1_all_aligned(
        session_data,
        output_dir_onedrive, 
        output_dir_local
        ):
    colors = [
        'limegreen',
        'coral',
        'lightcoral',
        'dodgerblue',
        'orange',
        'deeppink',
        'violet',
        'mediumorchid',
        'darkgreen' ,
        'purple',
        'deeppink',
        'grey']
    states = ['VisDetect1' , 
              'VisualStimulus1' , 
              'LeverRetract1' , 
              'PreVis2Delay' , 
              'VisDetect2' , 
              'VisualStimulus2' , 
              'WaitForPress2' , 
              'LeverRetract2' , 
              'Reward']
    max_sessions=10
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    outcomes = outcomes[start_idx:]
    session_id = np.arange(len(outcomes)) + 1

    today = date.today()
    today_formatted = str(today)[2:]
    year = today_formatted[0:2]
    month = today_formatted[3:5]
    day = today_formatted[6:]
    today_string = year + month + day
    numSessions = session_data['total_sessions']
    
    encoder_time_max = 5
    ms_per_s = 1000
    savefiles = 1
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_id = np.arange(len(outcomes)) + 1
    #a = 170
    for i in range(0 , len(session_id)):
        print('session id:' , session_id[i])
        raw_data = session_data['raw'][i]
        numTrials = raw_data['nTrials']
        time = np.zeros((numTrials , len(states)))
                
        session_date = dates[i][2:]
                
        print('plotting trajectories with all alignments for ' + subject + ' session ', session_date)
        
        TrialOutcomes = session_data['outcomes'][i]
        press_window = session_data['session_press_window']
        
        target_thresh = session_data['session_target_thresh']
        #target_thresh1 = mean(target_thresh[i])
        
        num_rows = 10
        num_columns = 3
        
        plots_per_page = num_rows * num_columns
                
        num_pages = int(np.ceil(numTrials/plots_per_page))
        
        num_plots_bottom_page = int(numTrials - (plots_per_page * (num_pages - 1)))
        num_rows_bottom_page = int(np.ceil(num_plots_bottom_page / num_columns))
        
        current_page = 1
        
        top_left_trial = 0
        bottom_right_trial = top_left_trial + plots_per_page
        
        pdf_streams = []
        pdf_paths = []
        
        for page in range(0, num_pages):   
            
            if current_page == num_pages:
                bottom_right_trial = top_left_trial + num_plots_bottom_page
            
            fig, axs = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(20, 30))
            
            fig.suptitle(subject + ' - ' + session_date + '  ' + str(numTrials) + ' Trials.\nPress Window:' + ' ' + str(press_window[0][0]) + 's.')            
            
            fig.tight_layout(rect=[0.01, 0.03, 1, 0.98]) # type: ignore # [left, bottom, right, top]
            fig.subplots_adjust(hspace=0.4)
           
            row = 0
            col = 0
            row_counter = 0
            col_counter = 0
            
            
            for trial in range(top_left_trial, bottom_right_trial):
            
                if row == 10:
                    row = 0
                
                if col == 3:
                    row = row + 1
                    col = 0
                
                
                ############
                trial_states = raw_data['RawEvents']['Trial'][trial]['States']
                encoder_data = raw_data['EncoderData'][trial]              
                times = encoder_data['Times']
                positions = encoder_data['Positions']
                encoder_times_vis1 = np.linspace(0, encoder_time_max, num=encoder_time_max*ms_per_s)  
                encoder_positions_aligned_vis1 = np.interp(encoder_times_vis1, times, positions)
                target_thresh1 = target_thresh[i][trial]
                ############
                
                #encoder_positions_aligned_vis1 = session_data['encoder_positions_aligned_vis1'][i][trial]
                
                y_top = 3.5
                if TrialOutcomes[trial] == 'Reward':
                    axs[row, col].plot(encoder_times_vis1, encoder_positions_aligned_vis1,'-', color = '#1f77b4', label='Reward')
                elif TrialOutcomes[trial] == 'DidNotPress1':
                    axs[row, col].plot(encoder_times_vis1, encoder_positions_aligned_vis1,'-', color = colors[1], label='DidNotPress1')
                elif TrialOutcomes[trial] == 'DidNotPress2':
                    axs[row, col].plot(encoder_times_vis1, encoder_positions_aligned_vis1,'-', color = colors[2], label='DidNotPress2')
                elif TrialOutcomes[trial] == 'EarlyPress1':
                    axs[row, col].plot(encoder_times_vis1, encoder_positions_aligned_vis1,'-', color = colors[3], label='EarlyPress1')
                elif TrialOutcomes[trial] == 'EarlyPress2':
                    axs[row, col].plot(encoder_times_vis1, encoder_positions_aligned_vis1,'-', color = colors[4], label='EarlyPress2')
                elif TrialOutcomes[trial] == 'EarlyPress':
                    axs[row, col].plot(encoder_times_vis1, encoder_positions_aligned_vis1,'-', color = colors[5], label='EarlyPress')
                else:
                    axs[row, col].plot(encoder_times_vis1, encoder_positions_aligned_vis1,'-', color = colors[6], label='Others')
                    
                axs[row, col].axhline(y = target_thresh1, color = '0.6', label = 'Target Threshold', linestyle='--')
                
                axs[row, col].set_title('Trial ' + str(trial + 1))        
                axs[row, col].legend(loc='upper right')
                #axs[row, col].set_xlim(time_left_VisStim1, 7.0)
                axs[row, col].set_ylim(-0.2, y_top)
                axs[row, col].spines['right'].set_visible(False)
                axs[row, col].spines['top'].set_visible(False)
                axs[row, col].set_xlabel('Time from VisStim1 (s)')
                axs[row, col].set_ylabel('Joystick deflection (deg)')
                for k in range(len(states)):
                    time[trial , k] = trial_states[states[k]][0]
                    if not np.isnan(time[trial , k]): 
                        axs[row, col].axvline(x = time[trial , k], color = colors[k], linestyle='--' , label = states[k])
                axs[row, col].legend()

                
                
                col = col + 1
                    
                                
            top_left_trial = bottom_right_trial
            bottom_right_trial = top_left_trial + plots_per_page                                                            
            current_page = current_page + 1
            
            
            # output_dir_onedrive, 
            # output_dir_local
            
            output_pdf_dir =  output_dir_onedrive + subject + '/'
            output_pdf_pages_dir = output_dir_local + subject + '/trajectories/trajectories_' + session_date + '/'
            os.makedirs(output_pdf_dir, exist_ok = True)
            os.makedirs(output_pdf_pages_dir, exist_ok = True)
            output_pdf_filename = output_pdf_pages_dir + subject +  session_date + '_trajectory_all_trs' + str(page)
            pdf_paths.append(output_pdf_filename + '.pdf')
            save_image(output_pdf_filename)        
            plt.close(fig)
            
        
        output = PdfWriter()
        pdf_files = []
        for pdf_path in pdf_paths:
            f = open(pdf_path, "rb")
            pdf_streams.append(PdfReader(f))
            pdf_files.append(f)
    
        for pdf_file_stream in pdf_streams:
            output.add_page(pdf_file_stream.pages[0])
        
        for pdf_file in pdf_files:
            pdf_file.close()
    
    
        outputStream = open(r'' + output_pdf_dir + subject + '_' + session_date + '_trajectory_all_trs_all_aligned' + '.pdf', "wb")
        output.write(outputStream)
        outputStream.close()

    print('Completed fig3_1 trajectories for ' + subject)
    print()
    plt.close("all")



        