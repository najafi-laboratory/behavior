#!/usr/bin/env python
# coding: utf-8

# In[1]:



from scipy.stats import sem
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfFileReader, PdfFileWriter, PdfWriter, PdfReader
from scipy.interpolate import interp1d
from datetime import date
from statistics import mean 
import math

def ensure_list(var):
    return var if isinstance(var, list) else [var]

def save_image(filename): 
    
    p = PdfPages(filename+'.pdf') 
    fig_nums = plt.get_fignums()   
    figs = [plt.figure(n) for n in fig_nums] 
      
    for fig in figs:  
        
        fig.savefig(p, format='pdf', dpi=300)
           
    p.close() 

def remove_substrings(s, substrings):
    for sub in substrings:
        s = s.replace(sub, "")
    return s

def flip_underscore_parts(s):
    parts = s.split("_", 1)  # Split into two parts at the first underscore
    if len(parts) < 2:
        return s  # Return original string if no underscore is found
    return f"{parts[1]}_{parts[0]}"

def lowercase_h(s):
    return s.replace('H', 'h')

def run(subject_session_data,output_dir_onedrive, output_dir_local):
    max_sessions = 20
    subject = subject_session_data['subject']
    dates = subject_session_data['dates']
    session_id = np.arange(len(dates)) + 1
    jitter_flag = subject_session_data['jitter_flag']
    raw_data = subject_session_data['raw']
    outcomes = subject_session_data['outcomes']
    outcomes_time = subject_session_data['outcomes_time']
    #categories = subject_session_data['isi_post_emp']
    
    subject = remove_substrings(subject, ['_opto', '_reg'])
    subject = flip_underscore_parts(subject)
    subject = lowercase_h(subject)
    
    
    trial_type = subject_session_data['trial_type']
    opto_flag = subject_session_data['opto_trial']
    
    row = 4 
    col = 5
    pre_delay = 300
    post_delay = 3000
    # alignments = ['1st flash' , '3th flash' , '4th flash' , 'choice window' , 'outcome']
    alignments = ['Choice Window' , 'Outcome']
    row_names = ['rewarded short' , 'rewarded long' , 'punished short' , 'punished long']
    n_bins = 500
    
    # Choice Variables
    choice_series_right_rs_opto = []
    choice_series_left_rs_opto = []
    choice_series_right_rs = []
    choice_series_left_rs = []
    choice_series_right_rl_opto = []
    choice_series_left_rl_opto = []
    choice_series_right_rl = []
    choice_series_left_rl = []
    choice_series_right_ps_opto = []
    choice_series_left_ps_opto = []
    choice_series_right_ps = []
    choice_series_left_ps = []
    choice_series_right_pl_opto = []
    choice_series_left_pl_opto = []
    choice_series_right_pl = []
    choice_series_left_pl = []
    
    # Outcome Variables
    outcome_series_right_rs_opto = []
    outcome_series_left_rs_opto = []
    outcome_series_right_rs = []
    outcome_series_left_rs = []
    outcome_series_right_rl_opto = []
    outcome_series_left_rl_opto = []
    outcome_series_right_rl = []
    outcome_series_left_rl = []
    outcome_series_right_ps_opto = []
    outcome_series_left_ps_opto = []
    outcome_series_right_ps = []
    outcome_series_left_ps = []
    outcome_series_right_pl_opto = []
    outcome_series_left_pl_opto = []
    outcome_series_right_pl = []
    outcome_series_left_pl = []
    
    
    for i in range(len(dates)):
        print(i)
        fig, axs = plt.subplots(nrows=4, ncols=len(alignments), figsize=(40, 30))
        #fig2, axs1 = plt.subplots(nrows=4, ncols=len(alignments), figsize=(40, 30))
        pdf_streams = []
        pdf_paths = []
        numTrials = raw_data[i]['nTrials']
        outcome = outcomes[i]
        outcome_time = outcomes_time[i]
        session_date = dates[i]
        
        row = 0
        for j in range(len(alignments)):
            series_right_rl = []
            series_right_rs = []
            series_right_ps = []
            series_right_pl = []
            
            series_right_rl_num = []
            series_right_rs_num = []
            series_right_ps_num = []
            series_right_pl_num = []            
            
            series_right_rl_opto = []
            series_right_rs_opto = []
            series_right_ps_opto = []
            series_right_pl_opto = []            

            series_right_rl_opto_num = []
            series_right_rs_opto_num = []
            series_right_ps_opto_num = []
            series_right_pl_opto_num = []                     
            
            series_left_rl = []
            series_left_rs = []
            series_left_ps = []
            series_left_pl = []
            
            series_left_rl_num = []
            series_left_rs_num = []
            series_left_ps_num = []
            series_left_pl_num = []
                               
            series_left_rl_opto = []
            series_left_rs_opto = []
            series_left_ps_opto = []
            series_left_pl_opto = []              
            
            series_left_rl_opto_num = []
            series_left_rs_opto_num = []
            series_left_ps_opto_num = []
            series_left_pl_opto_num = []
            
            colors = []
            
            print(i)
            for trial in range(numTrials):
                print(trial)
                
                stim_seq = np.divide(subject_session_data['stim_seq'][i][trial],1000)
                step = 10000
                start = 0
                # category = 1000*np.mean(raw_data[i]['ProcessedSessionData'][trial]['trial_isi']['PostISI'])
                # category = 1000*np.mean(raw_data[i]['TrialSettings'][0]['GUI']['ISIOrig_s'])
                

                if not 'Port1In' in raw_data[i]['RawEvents']['Trial'][trial]['Events'].keys():
                        port1 = []
                elif type(raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port1In']) == float:
                    port1 = [raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port1In']]
                else:
                    port1 = raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port1In']

                if not 'Port2In' in raw_data[i]['RawEvents']['Trial'][trial]['Events'].keys():
                    port2= []
                elif type(raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port2In']) == float:
                    port2 = [raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port2In']]
                else:
                    port2 = raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port2In']

                if not 'Port3In' in raw_data[i]['RawEvents']['Trial'][trial]['Events'].keys():
                    port3= []
                elif type(raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port3In']) == float:
                    port3 = [raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port3In']]
                else:
                    port3 = raw_data[i]['RawEvents']['Trial'][trial]['Events']['Port3In']
                    
                # make sure port vars are list type
                port1 = ensure_list(port1)
                port2 = ensure_list(port2)
                port3 = ensure_list(port3)                
                
                
                if j == 1:
                    alignment = outcome_time[trial]                    
                elif j == 0:
                    alignment = raw_data[i]['RawEvents']['Trial'][trial]['States']['WindowChoice'][0]                    
                        
                if opto_flag[i][trial]:
                    print(trial)
                    
                if not alignment == 'nan':
                    if outcome[trial] == 'Reward':
                        if trial_type[i][trial] == 1:
                            if opto_flag[i][trial]:
                                series_right_rs_opto.append([x - alignment for x in port3])
                                series_left_rs_opto.append([x - alignment for x in port1])   
                                
                                series_right_rs_opto_num.append(trial)
                                series_left_rs_opto_num.append(trial)
                            else:
                                series_right_rs.append([x - alignment for x in port3])
                                series_left_rs.append([x - alignment for x in port1])
                                
                                series_right_rs_num.append(trial)
                                series_left_rs_num.append(trial)                                
                        else:
                            if opto_flag[i][trial]:       
                                series_right_rl_opto.append([x - alignment for x in port3])
                                series_left_rl_opto.append([x - alignment for x in port1])   
                                
                                series_right_rl_opto_num.append(trial)
                                series_left_rl_opto_num.append(trial)
                            else:
                                series_right_rl.append([x - alignment for x in port3])
                                series_left_rl.append([x - alignment for x in port1])
                                
                                series_right_rl_num.append(trial)
                                series_left_rl_num.append(trial)                                
                             
                                

                    if outcome[trial] == 'Punish':
                        if trial_type[i][trial] == 1:
                            if opto_flag[i][trial]:
                                series_right_ps_opto.append([x - alignment for x in port3])
                                series_left_ps_opto.append([x - alignment for x in port1])   
                                
                                
                            else:
                                series_right_ps.append([x - alignment for x in port3])
                                series_left_ps.append([x - alignment for x in port1])
                                
                                
                        else:
                            if opto_flag[i][trial]:
                                series_right_pl_opto.append([x - alignment for x in port3])
                                series_left_pl_opto.append([x - alignment for x in port1])  
                                
                                
                            else:                        
                                series_right_pl.append([x - alignment for x in port3])
                                series_left_pl.append([x - alignment for x in port1])    
                                
                if j == 1:
                    alignment = outcome_time[trial]     
                    
                    outcome_series_right_rs_opto.append(series_right_rs_opto)
                    outcome_series_left_rs_opto.append(series_left_rs_opto)
                    outcome_series_right_rs.append(series_right_rs)
                    outcome_series_left_rs.append(series_left_rs)
                    
                    outcome_series_right_rl_opto.append(series_right_rl_opto)
                    outcome_series_left_rl_opto.append(series_left_rl_opto)
                    outcome_series_right_rl.append(series_right_rl)
                    outcome_series_left_rl.append(series_left_rl)
                    
                    outcome_series_right_ps_opto.append(series_right_ps_opto)
                    outcome_series_left_ps_opto.append(series_left_ps_opto)
                    outcome_series_right_ps.append(series_right_ps)
                    outcome_series_left_ps.append(series_left_ps)
                    
                    outcome_series_right_pl_opto.append(series_right_pl_opto)
                    outcome_series_left_pl_opto.append(series_left_pl_opto)
                    outcome_series_right_pl.append(series_right_pl)
                    outcome_series_left_pl.append(series_left_pl)
                    
                elif j == 0:
                    alignment = raw_data[i]['RawEvents']['Trial'][trial]['States']['WindowChoice'][0]   
                    
                    # Append base variables to choice_ and outcome_ lists
                    choice_series_right_rs_opto.append(series_right_rs_opto)
                    choice_series_left_rs_opto.append(series_left_rs_opto)
                    choice_series_right_rs.append(series_right_rs)
                    choice_series_left_rs.append(series_left_rs)
                    
                    choice_series_right_rl_opto.append(series_right_rl_opto)
                    choice_series_left_rl_opto.append(series_left_rl_opto)
                    choice_series_right_rl.append(series_right_rl)
                    choice_series_left_rl.append(series_left_rl)
                    
                    choice_series_right_ps_opto.append(series_right_ps_opto)
                    choice_series_left_ps_opto.append(series_left_ps_opto)
                    choice_series_right_ps.append(series_right_ps)
                    choice_series_left_ps.append(series_left_ps)
                    
                    choice_series_right_pl_opto.append(series_right_pl_opto)
                    choice_series_left_pl_opto.append(series_left_pl_opto)
                    choice_series_right_pl.append(series_right_pl)
                    choice_series_left_pl.append(series_left_pl)                 
                    
                                        
            
            '''
            For blue and light blue, common color names in string format are:

            Blue → "blue"
            Light Blue → "lightblue"
            Other variations include:
            
            Sky Blue → "skyblue"
            Powder Blue → "powderblue"
            Deep Sky Blue → "deepskyblue"
            Dodger Blue → "dodgerblue"
            Steel Blue → "steelblue"
            
            
            For red and light red, the common color names in string format are:
            
            Red → "red"
            Light Red → No standard "lightred", but similar options include:
            Salmon → "salmon"
            Light Coral → "lightcoral"
            Indian Red → "indianred"
            Tomato → "tomato"   
            
            
            For green, dark green, and light green in Matplotlib, you can use the following color names as strings:
            
            Green → "green" or "#008000"
            Dark Green → "darkgreen" or "#006400"
            Light Green → "lightgreen" or "#90EE90"
            Lime Green (a bright alternative) → "limegreen" or "#32CD32"
            Pale Green (a very light alternative) → "palegreen" or "#98FB98"            
            
            For black, dark black, and light black (grayish tones) in Matplotlib, use these color names:
            
            Black → "black" or "#000000"
            Dark Gray (near black) → "dimgray" or "#696969"
            Light Gray (faded black) → "lightgray" or "#D3D3D3"
            Very Dark Gray → "gray" or "#808080"
            Charcoal (deep dark gray) → "slategray" or "#708090"            
            
            '''
            
            xlim_left = -0.1
            xlim_right = 4
            ylim_bot = 0
            

            
#             axs1[0 , j].vlines(0 ,len(series_left_rs), 0, linestyle='--', color='grey')
#             axs1[0 , j].set_xlim([-2,6])
#             axs1[0 , j].set_title('reward, short, ' + alignments[j])
#             if len(series_center_rs) > 0:
#                 axs1[0 , j].hist(np.concatenate(series_center_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'black')
#                 axs1[0 , j].hist(np.concatenate(series_right_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'red')
#                 axs1[0 , j].hist(np.concatenate(series_left_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'limegreen')

#             axs1[1 , j].vlines(0 ,len(series_left_rl), 0, linestyle='--', color='grey')
#             axs1[1 , j].set_xlim([-2,6])
#             axs1[1 , j].set_title('reward, long, ' + alignments[j])
#             if len(series_center_rl) > 0:
#                 axs1[1 , j].hist(np.concatenate(series_center_rl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'black')
#                 axs1[1 , j].hist(np.concatenate(series_right_rl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'red')
#                 axs1[1 , j].hist(np.concatenate(series_left_rl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'limegreen')

#             axs1[2 , j].vlines(0 ,len(series_left_ps), 0, linestyle='--', color='grey')
#             axs1[2 , j].set_xlim([-2,6])
#             axs1[2 , j].set_title('punish, short, ' + alignments[j])
#             if len(series_center_ps) > 0:
#                 axs1[2 , j].hist(np.concatenate(series_center_ps), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'black')
#                 axs1[2 , j].hist(np.concatenate(series_right_ps), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'red')
#                 axs1[2 , j].hist(np.concatenate(series_left_ps), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'limegreen')


#             axs1[3 , j].vlines(0 ,len(series_left_pl), 0, linestyle='--', color='grey')
#             axs1[3 , j].set_xlim([-2,6])
#             axs1[3 , j].set_title('punish, long, ' + alignments[j])
#             if len(series_center_pl) > 0:
#                 axs1[3 , j].hist(np.concatenate(series_center_pl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'black')
#                 axs1[3 , j].hist(np.concatenate(series_right_pl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'red')
#                 axs1[3 , j].hist(np.concatenate(series_left_pl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'limegreen')








        output_dir_onedrive, 
        output_dir_local
        
        output_pdf_dir =  output_dir_onedrive + subject + '/'
        output_pdf_pages_dir = output_dir_local + subject + '/lick_traces/pooled_lick_traces/'
        os.makedirs(output_pdf_dir, exist_ok = True)
        os.makedirs(output_pdf_pages_dir, exist_ok = True)
        output_pdf_filename = output_pdf_pages_dir + subject +  session_date + '_lick_traces' + str(i)
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

        directory = output_pdf_dir + '/lick_traces/'
        # directory = output_pdf_dir
        os.makedirs(directory, exist_ok=True)  # Creates directory if it doesn't exist
        outputStream = open(r'' + directory + subject + '_' + session_date + '_lick_traces' + '.pdf', "wb")
        output.write(outputStream)
        outputStream.close()
        




