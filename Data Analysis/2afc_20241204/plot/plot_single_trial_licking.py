#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfFileReader, PdfFileWriter, PdfWriter, PdfReader
from datetime import date
from statistics import mean 
import math

def save_image(filename): 
    
    p = PdfPages(filename+'.pdf') 
    fig_nums = plt.get_fignums()   
    figs = [plt.figure(n) for n in fig_nums] 
      
    for fig in figs:  
        
        fig.savefig(p, format='pdf', dpi=300)
           
    p.close() 
    
    
def run(
        session_data,
        output_dir_onedrive, 
        output_dir_local
        ):
       
    states = [
        'Reward',
        'RewardNaive',
        'ChangingMindReward',
        'Punish',
        'PunishNaive',
        'WrongInitiation',
        'DidNotChoose' ,
        'InitCue' , 
        'InitCueAgain' ,
        'GoCue' ,
        'VisStimTrigger' ,
        'WindowChoice'
        ]
    colors = [
        'springgreen',
        'dodgerblue',
        'coral',
        'violet',
        'orange',
        'grey' ,
        'mediumorchid',
        'darkgreen' ,
        'purple',
        'cyan' ,
        'gold' , 
        'hotpink'
        ]
    raw_data = session_data['raw']
    max_sessions= 8
    subject = session_data['subject']
    dates = session_data['dates']
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    
    today = date.today()
    today_formatted = str(today)[2:]
    year = today_formatted[0:2]
    month = today_formatted[3:5]
    day = today_formatted[6:]
    today_string = year + month + day
    numSessions = session_data['total_sessions']
    session_id = np.arange(session_data['total_sessions']) + 1

    for i in range(0 , len(session_id)):
        print('session id:' , session_id[i])
        session_date = dates[i][2:]
        numTrials = raw_data[i]['nTrials'] 
        outcomes = session_data['outcomes']
        outcomes_clean = session_data['outcomes_clean']
        number_flash = session_data['number_flash']
                
        print('plotting licking for ' + subject + ' session ', session_date)
        
        
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
            
            fig.suptitle(subject + ' - ' + session_date + ' Number of Trials: ' + str(numTrials))            
            
            fig.tight_layout(rect=[0.01, 0.03, 1, 0.98]) # [left, bottom, right, top]
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
                if not 'Port1In' in raw_data[i]['RawEvents']['Trial'][trial]['Events'].keys():
                    port1 = [np.nan]
                elif type(raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port1In']) == float:
                    port1 = [raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port1In']]
                else:
                    port1 = raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port1In']

                if not 'Port2In' in raw_data[i]['RawEvents']['Trial'][trial]['Events'].keys():
                    port2= [np.nan]
                elif type(raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port2In']) == float:
                    port2 = [raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port2In']]
                else:
                    port2 = raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port2In']

                if not 'Port3In' in raw_data[i]['RawEvents']['Trial'][trial]['Events'].keys():
                    port3= [np.nan]
                elif type(raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port3In']) == float:
                    port3 = [raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port3In']]
                else:
                    port3 = raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port3In']

                trial_states =  raw_data[i]['RawEvents']['Trial'][trial]['States']
                trial_events =  raw_data[i]['RawEvents']['Trial'][trial]['Events']
                step = 10000
                start = 0
                maximum = math.ceil(np.nanmax([np.nanmax(port1),np.nanmax(port2),np.nanmax(port3)]))
                if not np.isnan(maximum):
                    stop = int(maximum)
                else:
                    stop = 10
                time = np.linspace(start , stop , step*(stop-start)+1)
                time = np.round(time , 4)
                time1 = np.zeros((raw_data[i]['nTrials'] , len(states)))
                lick1 = np.zeros(step*(stop-start)+1)
                lick2 = np.zeros(step*(stop-start)+1)
                lick3 = np.zeros(step*(stop-start)+1)
                for t in range(len(port1)):
                    lick1[np.where(time == round(port1[t] , 4))] = 1
                for t in range(len(port2)):
                    lick2[np.where(time == round(port2[t] , 4))] = 1
                for t in range(len(port3)):
                    lick3[np.where(time == round(port3[t] , 4))] = 1
                ############
                linew = 1.0
                for k in range(len(states)):
                    if states[k] in trial_states.keys():
                        if (type(trial_states[states[k]][0]) == float) or (type(trial_states[states[k]][0]) == np.float64) or all(np.isnan(trial_states[states[k]][0])):
                            time1[trial , k] = trial_states[states[k]][0]
                        else:
                            time1[trial , k] = trial_states[states[k]][-1][0]
                    else :
                        time1[trial , k] = np.nan
                    if not np.isnan(time1[trial , k]): 
                        axs[row, col].axvline(x = time1[trial , k], ymin=-0.5, ymax=0.25, linewidth=linew, color = colors[k], linestyle='--' , label = states[k])
                
                stim_seq = np.divide(session_data['stim_seq'][i][trial],1000)
                if not np.isnan(stim_seq[0 , 0]):
                    for j in range(len(stim_seq[0 , :])-1):
                        axs[row, col].fill_betweenx(y = [0 , 1.2], x1 = stim_seq[1 , j], x2 = stim_seq[0 , j], color = 'yellow' , alpha=0.2)
                        if j <2:
                            axs[row, col].fill_betweenx(y = [0 , 1.2], x1 = stim_seq[1 , j], x2 = stim_seq[0 , j+1], color = 'gray' , alpha=0.2)
                        else:
                            axs[row, col].fill_betweenx(y = [0 , 1.2], x1 = stim_seq[1 , j], x2 = stim_seq[0 , j+1], color = 'lavender')


                if session_data['isi_post_emp'][i][trial] > 500:
                    trial_type = 'long'
                else:
                    trial_type = 'short'
                        
                axs[row, col].plot(time , lick1, color = 'red',linewidth=linew,label='Left')
                axs[row, col].plot(time , lick2,color = 'black',linewidth=linew,label='center')
                axs[row, col].plot(time , lick3, color = 'limegreen',linewidth=linew,label='Right')
                
                axs[row, col].legend()
                title_color = 'black'
                if outcomes_clean[i][trial] == 'EarlyLick':
                    title_color = 'yellow'
                elif outcomes_clean[i][trial] == 'Switching':
                    title_color = 'pink'
                elif outcomes_clean[i][trial] == 'EarlyLickLimited':
                    title_color = 'blue'
                elif outcomes_clean[i][trial] == 'LateChoice':
                    title_color = 'red'
                if np.isnan(number_flash[i][trial]):
                    FC = 'nan'
                else:
                    FC = str(number_flash[i][trial])
                axs[row, col].set_title('Trial ' + str(trial + 1) + ', '+ trial_type + ' ISI' + ' (FC=' + FC + ')', color = title_color)        
                axs[row, col].legend(loc='upper right')
                axs[row, col].set_ylim(-0.5, 1.5)
                axs[row, col].spines['right'].set_visible(False)
                axs[row, col].spines['top'].set_visible(False)
                axs[row, col].set_xlabel('Time(s)')
                axs[row, col].set_ylabel('Licks')
                
                ###############

                
                
                col = col + 1
                    
                                
            top_left_trial = bottom_right_trial
            bottom_right_trial = top_left_trial + plots_per_page                                                            
            current_page = current_page + 1
            
            
            output_dir_onedrive, 
            output_dir_local
            
            output_pdf_dir =  output_dir_onedrive + subject + '/'
            output_pdf_pages_dir = output_dir_local + subject + '/licking/licking_' + session_date + '/'
            os.makedirs(output_pdf_dir, exist_ok = True)
            os.makedirs(output_pdf_pages_dir, exist_ok = True)
            output_pdf_filename = output_pdf_pages_dir + subject +  session_date + '_licking' + str(page)
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
    
    
        outputStream = open(r'' + output_pdf_dir + subject + '_' + session_date + '_licking' + '.pdf', "wb")
        output.write(outputStream)
        outputStream.close()

    print('Completed fig3_1 lickings for ' + subject)
    print()
    plt.close("all")

