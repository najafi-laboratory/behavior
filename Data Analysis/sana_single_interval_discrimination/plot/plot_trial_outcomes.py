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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
        output_dir_local,
        last_date
        ):
       
    states = [
        'Reward',
        'RewardNaive',
        'Punish',
        'WrongInitiation',
        'DidNotChoose' ,
        'EarlyLick' ,
        'EarlyLickLimited' ,
        'Switching'
        ]
    colors = [
        'limegreen',
        'springgreen' ,
        'r',
        'white',
        'gray' ,
        'yellow' ,
        'orange' ,
        'pink' ,
        'white' 
        ]
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    raw_data = session_data['raw']
    subject = session_data['subject']
    # chemo_labels = session_data['Chemo']
    chemo_labels = []
    jitter_flag = session_data['jitter_flag']
    jitter_session = np.array([np.sum(j) for j in jitter_flag])
    jitter_session[jitter_session!=0] = 1
    numsess = len(outcomes)
    
    num_rows = 10
    num_columns = 1
        
    plots_per_page = num_rows * num_columns
    num_pages = int(np.ceil(numsess/plots_per_page))
        
    num_plots_bottom_page = int(numsess - (plots_per_page * (num_pages - 1)))
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
        legend_elements = [Line2D([0], [0], marker='o',color='white',
                                 label='Reward' , markerfacecolor='limegreen'),
                           Line2D([0], [0], marker='o',color='white',
                                 label='Switching' , markerfacecolor='pink'),
                           Line2D([0], [0], marker='o',color='white',
                                 label='EarlyLick' , markerfacecolor='yellow'),
                           Line2D([0], [0], marker='o',color='white',
                                 label='EarlyLickLimited' , markerfacecolor='orange'),
                           Line2D([0], [0], marker='o',color='white',
                                 label='RewardNaive', markerfacecolor='springgreen'),
                           Line2D([0], [0], marker='o',color='white',
                                 label='Punish', markerfacecolor='r'),
                           Line2D([0], [0], marker='o',color='white',
                                 label='WrongInitiation', markerfacecolor='white', markeredgecolor='b'),
                           Line2D([0], [0], marker='o',color='white', 
                                 label='DidNotChoose', markerfacecolor='gray')]

        fig.legend(handles=legend_elements, loc="upper right") 

        fig.suptitle(subject + ' - '  + ' Number of Sessions: ' + str(numsess))            

        fig.tight_layout(rect=[0.01, 0.03, 1, 0.98]) # [left, bottom, right, top]
        fig.subplots_adjust(hspace=0.4)

        row = 0
        col = 0
        row_counter = 0
        col_counter = 0


        for sess in range(top_left_trial, bottom_right_trial):
            
            session_date = dates[sess]
            if row == num_rows:
                row = 0

            if col == num_columns:
                row = row + 1
                col = 0
            
            
            outcome = outcomes[sess]
            raw_data = session_data['raw'][sess]
            trial_types = np.array(raw_data['TrialTypes'])
            x = np.arange(len(trial_types))+1
            
            color_code = []
            edge = []
            for i in range(len(outcome)):
                a = 0
                for j in range(len(states)):
                    if outcome[i] == states[j]:
                        a = 1
                        color_code.append(colors[j])
                if a == 0:
                    color_code.append(colors[-1])
                if outcome[i] == states[3]:
                    edge.append('blue')
                #elif outcome[i] == states[4]:
                 #   edge.append('black')
                else:
                    edge.append(color_code[-1])
            
            axs[row].scatter(x , 3-trial_types , color = color_code , edgecolor =edge)
            # if chemo_labels[sess] == 1:
            #     if jitter_session[sess] == 0:
            #         axs[row].set_title(session_date, color = 'red')
            #     else:
            #         axs[row].set_title(session_date + ' (jittered)', color = 'red')
            # else:
            #     if jitter_session[sess] == 0:
            #         axs[row].set_title(session_date , color = 'black')
            #     else: 
            #         axs[row].set_title(session_date + ' (jittered)', color = 'black')
            axs[row].set_ylim(0.5 , 2.5)
            axs[row].set_xticks(np.arange(len(outcome)//20)*20)
            
            col = col + 1
                    
                                
        top_left_trial = bottom_right_trial
        bottom_right_trial = top_left_trial + plots_per_page                                                            
        current_page = current_page + 1
        
        output_dir_onedrive, 
        output_dir_local

        output_pdf_dir =  output_dir_onedrive + subject + '/'
        output_pdf_pages_dir = output_dir_local + subject + '/bpod/bpod_' + session_date + '/'
        os.makedirs(output_pdf_dir, exist_ok = True)
        os.makedirs(output_pdf_pages_dir, exist_ok = True)
        output_pdf_filename = output_pdf_pages_dir + subject +  session_date + '_outcome' + str(page)
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


    outputStream = open(r'' + output_pdf_dir + subject + '_' + last_date + '_Bpod_outcome' + '.pdf', "wb")
    output.write(outputStream)
    outputStream.close()


        
            
            
        
        

