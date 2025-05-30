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
    
    trial_type = subject_session_data['trial_type']
    
    row = 4 
    col = 5
    pre_delay = 300
    post_delay = 3000
    # alignments = ['1st flash' , '3th flash' , '4th flash' , 'choice window' , 'outcome']
    alignments = ['choice window' , 'outcome']
    row_names = ['rewarded short' , 'rewarded long' , 'punished short' , 'punished long']
    n_bins = 500
    
    
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
        for j in range(len(alignments)):
            series_right_rl = []
            series_right_rs = []
            series_right_ps = []
            series_right_pl = []
            series_center_rl = []
            series_center_rs = []
            series_center_ps = []
            series_center_pl = []
            series_left_rl = []
            series_left_rs = []
            series_left_ps = []
            series_left_pl = []
            colors = []

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
                
                # if j == 4:
                #     alignment = outcome_time[trial]
                # elif j == 3:
                #     alignment = raw_data[i]['RawEvents']['Trial'][trial]['States']['WindowChoice'][0]
                # elif j == 2:
                #     if len(stim_seq[1 , :]) > 2:
                #         alignment = stim_seq[1 , 2]
                #     else:
                #         alignment = 'nan'
                # elif j == 1:
                #     if len(stim_seq[1 , :]) > 3:
                #         alignment = stim_seq[1 , 3]
                #     else:
                #         alignment = 'nan'
                # elif j == 0:
                #     if len(stim_seq[1 , :]) > 0:
                #         alignment = stim_seq[1 , 0]
                #     else:
                #         alignment = 'nan'
                
                if j == 1:
                    alignment = outcome_time[trial]                    
                elif j == 0:
                    alignment = raw_data[i]['RawEvents']['Trial'][trial]['States']['WindowChoice'][0]                    
                        
                if not alignment == 'nan':
                    if outcome[trial] == 'Reward':
                        if trial_type[i][trial] == 1:
                            series_right_rs.append([x - alignment for x in port3])
                            series_center_rs.append([x - alignment for x in port2])
                            series_left_rs.append([x - alignment for x in port1])
                        else:
                            colors.append('red')
                            series_right_rl.append([x - alignment for x in port3])
                            series_center_rl.append([x - alignment for x in port2])
                            series_left_rl.append([x - alignment for x in port1])
                                
                        # if category < 500:
                        #     if len(stim_seq[1 , :]) > 3:
                        #         series_right_rs.append([x - alignment for x in port3])
                        #         series_center_rs.append([x - alignment for x in port2])
                        #         series_left_rs.append([x - alignment for x in port1])
                        # if category > 500:
                        #     if len(stim_seq[1 , :]) > 3:
                        #         colors.append('red')
                        #         series_right_rl.append([x - alignment for x in port3])
                        #         series_center_rl.append([x - alignment for x in port2])
                        #         series_left_rl.append([x - alignment for x in port1])

                    if outcome[trial] == 'Punish':
                        if trial_type[i][trial] == 1:
                            colors.append('yellow')
                            series_right_ps.append([x - alignment for x in port3])
                            series_center_ps.append([x - alignment for x in port2])
                            series_left_ps.append([x - alignment for x in port1])
                        else:
                            colors.append('green')
                            series_right_pl.append([x - alignment for x in port3])
                            series_center_pl.append([x - alignment for x in port2])
                            series_left_pl.append([x - alignment for x in port1])                       
                                        
                        
                        # if category < 500:
                        #     if len(stim_seq[1 , :]) > 3:
                        #         colors.append('yellow')
                        #         series_right_ps.append([x - alignment for x in port3])
                        #         series_center_ps.append([x - alignment for x in port2])
                        #         series_left_ps.append([x - alignment for x in port1])

                        # if category > 500:
                        #     if len(stim_seq[1 , :]) > 3:
                        #         colors.append('green')
                        #         series_right_pl.append([x - alignment for x in port3])
                        #         series_center_pl.append([x - alignment for x in port2])
                        #         series_left_pl.append([x - alignment for x in port1])
            xlim_left = 0.1
            xlim_right = 4
                        
            axs[0 , j].vlines(0 ,len(series_left_rs), 0, linestyle='--', color='grey')
            axs[0 , j].eventplot(series_center_rs, color='black', linelengths = 0.3)
            axs[0 , j].eventplot(series_right_rs, color='red', linelengths = 0.3)
            axs[0 , j].eventplot(series_left_rs, color='limegreen', linelengths = 0.3)
            axs[0 , j].set_xlim([xlim_left,xlim_right])
            axs[0 , j].set_title('reward, short, ' + alignments[j])
            if len(series_center_rs) > 0:
                axs[0 , j].hist(np.concatenate(series_center_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'black', alpha = 0.2)
                axs[0 , j].hist(np.concatenate(series_right_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'red', alpha = 0.2)
                axs[0 , j].hist(np.concatenate(series_left_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'limegreen', alpha = 0.2)

            axs[1 , j].vlines(0 ,len(series_left_rl), 0, linestyle='--', color='grey')
            axs[1 , j].eventplot(series_center_rl, color='black', linelengths = 0.3)
            axs[1 , j].eventplot(series_right_rl, color='red', linelengths = 0.3)
            axs[1 , j].eventplot(series_left_rl, color='limegreen', linelengths = 0.3)
            axs[1 , j].set_xlim([xlim_left,xlim_right])
            axs[1 , j].set_title('reward, long, ' + alignments[j])
            if len(series_center_rl) > 0:
                axs[1 , j].hist(np.concatenate(series_center_rl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'black', alpha = 0.2)
                axs[1 , j].hist(np.concatenate(series_right_rl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'red', alpha = 0.2)
                axs[1 , j].hist(np.concatenate(series_left_rl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'limegreen', alpha = 0.2)

            axs[2 , j].vlines(0 ,len(series_left_ps), 0, linestyle='--', color='grey')
            axs[2 , j].eventplot(series_center_ps, color='black', linelengths = 0.3)
            axs[2 , j].eventplot(series_right_ps, color='red', linelengths = 0.3)
            axs[2 , j].eventplot(series_left_ps, color='limegreen', linelengths = 0.3)
            axs[2 , j].set_xlim([xlim_left,xlim_right])
            axs[2 , j].set_title('punish, short, ' + alignments[j])
            if len(series_center_ps) > 0:
                axs[2 , j].hist(np.concatenate(series_center_ps), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'black', alpha = 0.4)
                axs[2 , j].hist(np.concatenate(series_right_ps), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'red', alpha = 0.4)
                axs[2 , j].hist(np.concatenate(series_left_ps), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'green', alpha = 0.4)


            axs[3 , j].vlines(0 ,len(series_left_pl), 0, linestyle='--', color='grey')
            axs[3 , j].eventplot(series_center_pl, color='black', linelengths = 0.3)
            axs[3 , j].eventplot(series_right_pl, color='red', linelengths = 0.3)
            axs[3 , j].eventplot(series_left_pl, color='limegreen', linelengths = 0.3)
            axs[3 , j].set_xlim([xlim_left,xlim_right])
            axs[3 , j].set_title('punish, long, ' + alignments[j])
            if len(series_center_pl) > 0:
                axs[3 , j].hist(np.concatenate(series_center_pl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'black' , alpha = 0.2)
                axs[3 , j].hist(np.concatenate(series_right_pl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'red' , alpha = 0.2)
                axs[3 , j].hist(np.concatenate(series_left_pl), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'limegreen' , alpha = 0.2)
                
            
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
        output_pdf_pages_dir = output_dir_local + subject + '/_alingment/alingment_' + session_date + '/'
        os.makedirs(output_pdf_dir, exist_ok = True)
        os.makedirs(output_pdf_pages_dir, exist_ok = True)
        output_pdf_filename = output_pdf_pages_dir + subject +  session_date + '_alingment' + str(i)
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


        outputStream = open(r'' + output_pdf_dir + subject + '_' + session_date + '_alingment' + '.pdf', "wb")
        output.write(outputStream)
        outputStream.close()
        




