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
            
            series_center_rl = []
            series_center_rs = []
            series_center_ps = []
            series_center_pl = []
     
            series_center_rl_opto = []
            series_center_rs_opto = []
            series_center_ps_opto = []
            series_center_pl_opto = []            
            
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
                        
                if opto_flag[i][trial]:
                    print(trial)
                    
                if not alignment == 'nan':
                    if outcome[trial] == 'Reward':
                        if trial_type[i][trial] == 1:
                            if opto_flag[i][trial]:
                                series_right_rs_opto.append([x - alignment for x in port3])
                                series_center_rs_opto.append([x - alignment for x in port2])
                                series_left_rs_opto.append([x - alignment for x in port1])   
                                
                                series_right_rs_opto_num.append(trial)
                                series_left_rs_opto_num.append(trial)
                            else:
                                series_right_rs.append([x - alignment for x in port3])
                                series_center_rs.append([x - alignment for x in port2])
                                series_left_rs.append([x - alignment for x in port1])
                                
                                series_right_rs_num.append(trial)
                                series_left_rs_num.append(trial)                                
                        else:
                            if opto_flag[i][trial]:       
                                series_right_rl_opto.append([x - alignment for x in port3])
                                series_center_rl_opto.append([x - alignment for x in port2])
                                series_left_rl_opto.append([x - alignment for x in port1])   
                                
                                series_right_rl_opto_num.append(trial)
                                series_left_rl_opto_num.append(trial)
                            else:
                                series_right_rl.append([x - alignment for x in port3])
                                series_center_rl.append([x - alignment for x in port2])
                                series_left_rl.append([x - alignment for x in port1])
                                
                                series_right_rl_num.append(trial)
                                series_left_rl_num.append(trial)                                
                             
                                
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
                            if opto_flag[i][trial]:
                                series_right_ps_opto.append([x - alignment for x in port3])
                                series_center_ps_opto.append([x - alignment for x in port2])
                                series_left_ps_opto.append([x - alignment for x in port1])   
                                
                                
                            else:
                                series_right_ps.append([x - alignment for x in port3])
                                series_center_ps.append([x - alignment for x in port2])
                                series_left_ps.append([x - alignment for x in port1])
                                
                                
                        else:
                            if opto_flag[i][trial]:
                                series_right_pl_opto.append([x - alignment for x in port3])
                                series_center_pl_opto.append([x - alignment for x in port2])
                                series_left_pl_opto.append([x - alignment for x in port1])  
                                
                                
                            else:                        
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
            
            # SHORT DUR
            #  averages
            ################################################################################################################  
            
            # Define colors: one color per group
            colorlist = ['green', 'lightgreen', 'dimgray', 'lightgray']
            
            row = 0
            
            # Combine all events
            all_events = series_left_rs + series_left_rs_opto + series_right_ps + series_right_ps_opto # Concatenate both groups        
            
            # Step 1: Find the minimum number of licks across all trials
            min_length_series_left_rs = min((len(x) for x in series_left_rs), default=0)
            min_length_series_left_rs_opto = min((len(x) for x in series_left_rs_opto), default=0)
            min_length_series_right_ps = min((len(x) for x in series_right_ps), default=0)
            min_length_series_right_ps_opto = min((len(x) for x in series_right_ps_opto), default=0)
            

            # Step 2: Limit each trial to the minimum number of licks
            limited_licks_series_left_rs = [x[:min_length_series_left_rs] for x in series_left_rs]
            
            limited_licks_series_left_rs_opto = [x[:min_length_series_left_rs_opto] for x in series_left_rs_opto]
            limited_licks_series_right_ps = [x[:min_length_series_right_ps] for x in series_right_ps]
            limited_licks_series_right_ps_opto = [x[:min_length_series_right_ps_opto] for x in series_right_ps_opto]
            
            # Step 3: Transpose to group licks by index (1st, 2nd, 3rd lick)
            licks_transposed_series_left_rs = np.array(limited_licks_series_left_rs).T            
            licks_transposed_series_left_rs_opto = np.array(limited_licks_series_left_rs_opto).T
            licks_transposed_series_right_ps = np.array(limited_licks_series_right_ps).T
            licks_transposed_series_right_ps_opto = np.array(limited_licks_series_right_ps_opto).T
            
            # Step 4: Calculate the element-wise average of the licks across trials
            avg_lick_trace_series_left_rs = np.mean(licks_transposed_series_left_rs, axis=1) if licks_transposed_series_left_rs.ndim > 1 else np.array([0], dtype=float)                       
            avg_lick_trace_series_left_rs_opto = np.mean(licks_transposed_series_left_rs_opto, axis=1) if licks_transposed_series_left_rs_opto.ndim > 1 else np.array([0], dtype=float)             
            avg_lick_trace_series_right_ps = np.mean(licks_transposed_series_right_ps, axis=1) if licks_transposed_series_right_ps.ndim > 1 else np.array([0], dtype=float)            
            avg_lick_trace_series_right_ps_opto = np.mean(licks_transposed_series_right_ps_opto, axis=1) if licks_transposed_series_right_ps_opto.ndim > 1 else np.array([0])

            
            avg_lick_traces = [
                avg_lick_trace_series_left_rs,
                avg_lick_trace_series_left_rs_opto,
                avg_lick_trace_series_right_ps,
                avg_lick_trace_series_right_ps_opto
            ]
            
            # Step 1: Find the maximum length of the traces
            max_length = max(len(trace) for trace in avg_lick_traces)
            
            
            padded_lick_traces = [np.pad(trace, (0, max_length - len(trace)), mode='constant', constant_values=np.nan).tolist() for trace in avg_lick_traces]
            y_positions = np.arange(len(padded_lick_traces))
            num_traces = len(padded_lick_traces)
               
            # axs[row , j].eventplot(padded_lick_traces, lineoffsets=y_positions, color=colorlist, linelengths = 0.3)            
            
            # axs[row , j].set_xlim([xlim_left,xlim_right])
            # axs[row,  j].set_ylim(-1, num_traces)  # Ensure all rows fit within view            
            # axs[row , j].set_title('Type: Short ISI, Sorted: Trial Order, Aligned: ' + alignments[j])

            # # Define y-ticks and labels for each sublist
            
            # yticklabels = ['Left Avg', 'Left Opto Avg', 'Right Avg', 'Right Opto Avg']
            # yticks = np.arange(len(yticklabels))
            
            # Set yticks and labels
            # axs[row , j].set_yticks(yticks)
            # axs[row , j].set_yticklabels(yticklabels)

            # # # Add one empty plot for each color-label pair

            # # Add legend
            # axs[row , j].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)            
            
            # # Get the figure handle from the axes
            # figure_handle = axs[row , j].figure
            # figure_handle.show()
            
            ################################################################################################################  
            
            labels = ['Left Lick', \
                      'Left Lick - Opto', \
                      'Right Lick', \
                      'Right Lick - Opto',]            
            
            # Flattened y-positions: each sublist gets a unique row
            num_sublists_1 = len(series_left_rs)
            num_sublists_2 = len(series_left_rs_opto)
            num_sublists_3 = len(series_right_ps)
            num_sublists_4 = len(series_right_ps_opto)
            
            y_positions = list(range(num_sublists_1)) + \
                list(range(num_sublists_1, num_sublists_1 + num_sublists_2)) + \
                list(range(num_sublists_1 + num_sublists_2, num_sublists_1 + num_sublists_2 + num_sublists_3)) + \
                list(range(num_sublists_1 + num_sublists_2 + num_sublists_3, num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4))            
            
            # Updated y_positions with space for averages
            y_positions = []
            modified_traces = []  # Will contain avg traces + original traces
            
            sublists = [series_left_rs, series_left_rs_opto, series_right_ps, series_right_ps_opto]
            
            y_offset = 0  # Keeps track of row index
            for idx, sublist in enumerate(sublists):
                # Add the original traces with updated positions
                for trace in sublist:
                    modified_traces.append(trace)
                    y_positions.append(y_offset)
                    y_offset += 1  # Shift for next row 
                    
                # Insert the average trace above the sublist
                modified_traces.append(padded_lick_traces[idx])
                y_positions.append(y_offset)  # Position for average
                y_offset += 1  # Shift position
            
           
            
            
            
            # Define colors: one color per group
            colorlist = ['green', 'lightgreen', 'dimgray', 'lightgray']
            colors = ['green'] * num_sublists_1 + ['lightgreen'] * num_sublists_2  + \
                ['dimgray'] * num_sublists_3 + ['lightgray'] * num_sublists_4 
            
            colors = ['green'] * num_sublists_1 + \
                ['red'] * 1 + \
                ['lightgreen'] * num_sublists_2  + \
                ['red'] * 1 + \
                ['dimgray'] * num_sublists_3 + \
                ['red'] * 1 + \
                ['lightgray'] * num_sublists_4 + \
                ['red'] * 1
          
            # # Step 1: Concatenate all the lists
            # concatenated_data = np.concatenate(series_left_rs)
            
            # # Step 2: Compute the histogram
            # # `bins` defines the range of values (e.g., from min to max of the data)
            # bins = 100  # Number of bins you want to divide the range into
            # hist, bin_edges = np.histogram(concatenated_data, bins=bins, density=True)
            
            # hist = hist * 100
            
            # # Step 3: Normalize the histogram to get the probability density
            # # This is already handled by setting `density=True` in np.histogram
            
            # # Step 4: Plot the probability density curve
            # # Compute the midpoints of the bins
            # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # if j == 1:
            #     axs[row , j].plot(bin_centers, hist, label="Probability Density")
            
            
            # 'reward, short, ' + alignments[j]
            # trial ordered
            ################################################################################################################
            
            axs[row , j].vlines(0 ,len(y_positions), 0, linestyle='--', color='grey')
              
            axs[row,  j].eventplot(modified_traces, color=colors, linelengths=0.6, lineoffsets=y_positions)
            
            ylim_top = len(y_positions)
            axs[row , j].set_xlim([xlim_left,xlim_right])
            axs[row , j].set_ylim([ylim_bot,ylim_top])
            axs[row , j].set_title('Type: Short ISI, Sorted: Trial Order, Aligned: ' + alignments[j])
            # if len(series_right_rs) > 0:
                
            # axs[row , j].hist(np.concatenate(series_center_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'black', alpha = 0.2)
            # axs[row , j].hist(np.concatenate(series_right_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'red', alpha = 0.2)            
            # axs[row , j].hist(np.concatenate(series_left_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'limegreen', alpha = 0.2)

            # Add one empty plot for each color-label pair
            for k in range(len(colorlist)):
                axs[row , j].plot([], [], color=colorlist[k], label=labels[k])


           
            # Define y-ticks and labels for each sublist
            num_sublists_1_len = num_sublists_1 + 1
            num_sublists_2_len = num_sublists_2 + 1
            num_sublists_3_len = num_sublists_3 + 1
            num_sublists_4_len = num_sublists_4 + 1

            # yticks = [num_sublists_1 / 2, num_sublists_1 + num_sublists_2 / 2, 
            #           num_sublists_1 + num_sublists_2 + num_sublists_3 / 2, 
            #           num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4 / 2]
            
            yticks = [num_sublists_1_len / 2, num_sublists_1_len + num_sublists_2_len / 2, 
                      num_sublists_1_len + num_sublists_2_len + num_sublists_3_len / 2, 
                      num_sublists_1_len + num_sublists_2_len + num_sublists_3_len + num_sublists_4_len / 2]            
            
            yticklabels = ['Left', 'Left Opto', 'Right', 'Right Opto']
            
            
            # y_trials = list(range(0,len(y_positions),10))
            # y_ticks = list(range(len(modified_traces)))
            
            # Set yticks every 10 trials            
            # Set yticks and labels
            # axs[row , j].set_yticks(yticks)
            # axs[row , j].set_yticklabels(yticklabels)
            trial_yticks = np.arange(0, len(modified_traces) + 1, 10)
            axs[row , j].set_yticks(trial_yticks)
            axs[row , j].set_yticklabels([str(y) for y in trial_yticks])
            axs[row , j].set_ylabel("Trials", color='black')
            
            line_thickness = 0.5  # Adjust line thickness here
            
            # Draw partition lines to divide sublists
            axs[row , j].axhline(y=num_sublists_1+0.5, color='grey', linestyle='--', linewidth=line_thickness)
            axs[row , j].axhline(y=num_sublists_1 + num_sublists_2+1.5, color='grey', linestyle='--', linewidth=line_thickness)
            axs[row , j].axhline(y=num_sublists_1 + num_sublists_2 + num_sublists_3+2.5, color='grey', linestyle='--', linewidth=line_thickness)          
           
            # Add legend
            # axs[row , j].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
            axs[row , j].legend(loc='best', bbox_to_anchor=(1,1), ncol=1)
           
            # Create a twin y-axis
            axs2 = axs[row , j].twinx() 
            
            # Set different tick marks (e.g., transformed scale)
            axs2.set_ylim(axs[row , j].get_ylim())  # Keep same limits
            # axs2.set_yticks(y_positions)  # tick positions
            # axs2.set_yticklabels(y_ticks)  # tick labels
            axs2.set_yticks(yticks)  # tick positions
            axs2.set_yticklabels(yticklabels)  # tick labels            
            axs2.set_ylabel("Lick Type", color='black')
            # axs2.tick_params(axis='y', labelcolor='black',  size=2, labelsize=3)        
            axs2.tick_params(axis='y', labelcolor='black')  
            
            figure_handle = axs[row , j].figure
            figure_handle.show()
                        
            
            # SHORT DUR
            # 
            ################################################################################################################            
            if 0:
                row = row + 1
                
                labels = ['Left Lick', \
                          'Left Lick - Opto', \
                          'Right Lick', \
                          'Right Lick - Opto',]            
                
                # Flattened y-positions: each sublist gets a unique row
                num_sublists_1 = len(series_left_rs)
                num_sublists_2 = len(series_left_rs_opto)
                num_sublists_3 = len(series_right_ps)
                num_sublists_4 = len(series_right_ps_opto)
                
                y_positions = list(range(num_sublists_1)) + list(range(num_sublists_1, num_sublists_1 + num_sublists_2)) + \
                    list(range(num_sublists_1 + num_sublists_2, num_sublists_1 + num_sublists_2 + num_sublists_3)) + \
                    list(range(num_sublists_1 + num_sublists_2 + num_sublists_3, num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4))
                
                y_ticks = list(range(num_sublists_1)) + list(range(num_sublists_2)) + list(range(num_sublists_3)) + list(range(num_sublists_4))
                
                y_ticks = [x + 1 for x in y_ticks]
                
                
                # Define colors: one color per group
                colorlist = ['green', 'lightgreen', 'dimgray', 'lightgray']
                colors = ['green'] * num_sublists_1 + ['lightgreen'] * num_sublists_2  + \
                    ['dimgray'] * num_sublists_3 + ['lightgray'] * num_sublists_4 
                
              
                # # Step 1: Concatenate all the lists
                # concatenated_data = np.concatenate(series_left_rs)
                
                # # Step 2: Compute the histogram
                # # `bins` defines the range of values (e.g., from min to max of the data)
                # bins = 100  # Number of bins you want to divide the range into
                # hist, bin_edges = np.histogram(concatenated_data, bins=bins, density=True)
                
                # hist = hist * 100
                
                # # Step 3: Normalize the histogram to get the probability density
                # # This is already handled by setting `density=True` in np.histogram
                
                # # Step 4: Plot the probability density curve
                # # Compute the midpoints of the bins
                # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # if j == 1:
                #     axs[row , j].plot(bin_centers, hist, label="Probability Density")
                
                
                # 'reward, short, ' + alignments[j]
                # trial ordered
                ################################################################################################################
                
                axs[row , j].vlines(0 ,len(y_positions), 0, linestyle='--', color='grey')
                # axs[row , j].eventplot(series_center_rs, color='black', linelengths = 0.3)
                # axs[row , j].eventplot(series_right_rs, color='red', linelengths = 0.3)
                # axs[row , j].eventplot(series_left_rs, color='limegreen', linelengths = 0.3)
                
                # axs[row , j].eventplot(series_right_rs_opto, color='red', linelengths = 0.3)
                # axs[row , j].eventplot(series_left_rs_opto, color='indigo', linelengths = 0.3)            
                
                axs[row , j].eventplot(all_events, color=colors, lineoffsets=y_positions, linelengths = 0.6)            
                
                axs[row , j].set_xlim([xlim_left,xlim_right])
                axs[row , j].set_title('Type: Short ISI, Sorted: Trial Order, Aligned: ' + alignments[j])
                # if len(series_right_rs) > 0:
                    
                # axs[row , j].hist(np.concatenate(series_center_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'black', alpha = 0.2)
                # axs[row , j].hist(np.concatenate(series_right_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'red', alpha = 0.2)
                
                # axs[row , j].hist(np.concatenate(series_left_rs), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'limegreen', alpha = 0.2)
    
                # Add one empty plot for each color-label pair
                for k in range(len(colorlist)):
                    axs[row , j].plot([], [], color=colorlist[k], label=labels[k])
    
                # Add legend
                axs[row , j].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
               
                # Define y-ticks and labels for each sublist
                num_sublists_1_len = num_sublists_1 + 1
                num_sublists_2_len = num_sublists_2 + 1
                num_sublists_3_len = num_sublists_3 + 1
                num_sublists_4_len = num_sublists_4 + 1
    
                # yticks = [num_sublists_1 / 2, num_sublists_1 + num_sublists_2 / 2, 
                #           num_sublists_1 + num_sublists_2 + num_sublists_3 / 2, 
                #           num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4 / 2]
                
                yticks = [num_sublists_1_len / 2, num_sublists_1_len + num_sublists_2_len / 2, 
                          num_sublists_1_len + num_sublists_2_len + num_sublists_3_len / 2, 
                          num_sublists_1_len + num_sublists_2_len + num_sublists_3_len + num_sublists_4_len / 2]  
                
                yticklabels = ['Left', 'Left Opto', 'Right', 'Right Opto']
                
                # Set yticks and labels
                axs[row , j].set_yticks(yticks)
                axs[row , j].set_yticklabels(yticklabels)
                
                line_thickness = 0.5  # Adjust line thickness here
                
                # Draw partition lines to divide sublists
                axs[row , j].axhline(y=num_sublists_1-0.5, color='grey', linestyle='--', linewidth=line_thickness)
                axs[row , j].axhline(y=num_sublists_1 + num_sublists_2-0.5, color='grey', linestyle='--', linewidth=line_thickness)
                axs[row , j].axhline(y=num_sublists_1 + num_sublists_2 + num_sublists_3-0.5, color='grey', linestyle='--', linewidth=line_thickness)          
               
                # Create a twin y-axis
                axs2 = axs[row , j].twinx() 
                
                # Set different tick marks (e.g., transformed scale)
                axs2.set_ylim(axs[row , j].get_ylim())  # Keep same limits
                axs2.set_yticks(y_positions)  # tick positions
                axs2.set_yticklabels(y_ticks)  # tick labels
                axs2.set_ylabel("Trials", color='black')
                axs2.tick_params(axis='y', labelcolor='black',  size=2, labelsize=3)            
           
           

            # 'reward, long, ' + alignments[j]
            # trial ordered
            ################################################################################################################
            
            row = row + 1
            
            # Combine all events
            # all_events = series_left_rs + series_left_rs_opto + series_right_rl + series_right_rl_opto # Concatenate both groups        
            
            # Step 1: Find the minimum number of licks across all trials
            min_length_series_left_pl = min((len(x) for x in series_left_pl), default=0)
            min_length_series_left_pl_opto = min((len(x) for x in series_left_pl_opto), default=0)
            min_length_series_right_rl = min((len(x) for x in series_right_rl), default=0)
            min_length_series_right_rl_opto = min((len(x) for x in series_right_rl_opto), default=0)
            

            # Step 2: Limit each trial to the minimum number of licks
            limited_licks_series_left_pl = [x[:min_length_series_left_pl] for x in series_left_pl]            
            limited_licks_series_left_pl_opto = [x[:min_length_series_left_pl_opto] for x in series_left_pl_opto]
            limited_licks_series_right_rl = [x[:min_length_series_right_rl] for x in series_right_rl]
            limited_licks_series_right_rl_opto = [x[:min_length_series_right_rl_opto] for x in series_right_rl_opto]
            
            # Step 3: Transpose to group licks by index (1st, 2nd, 3rd lick)
            licks_transposed_series_left_pl = np.array(limited_licks_series_left_pl).T            
            licks_transposed_series_left_pl_opto = np.array(limited_licks_series_left_pl_opto).T
            licks_transposed_series_right_rl = np.array(limited_licks_series_right_rl).T
            licks_transposed_series_right_rl_opto = np.array(limited_licks_series_right_rl_opto).T
            
            # Step 4: Calculate the element-wise average of the licks across trials
            avg_lick_trace_series_left_pl = np.mean(licks_transposed_series_left_pl, axis=1) if licks_transposed_series_left_pl.ndim > 1 else np.array([0], dtype=float)   
            avg_lick_trace_series_left_pl_opto = np.mean(licks_transposed_series_left_pl_opto, axis=1) if licks_transposed_series_left_pl_opto.ndim > 1 else np.array([0], dtype=float) 
            avg_lick_trace_series_right_rl = np.mean(licks_transposed_series_right_rl, axis=1) if licks_transposed_series_right_rl.ndim > 1 else np.array([0], dtype=float)               
            avg_lick_trace_series_right_rl_opto = np.mean(licks_transposed_series_right_rl_opto, axis=1) if licks_transposed_series_right_rl_opto.ndim > 1 else np.array([0], dtype=float) 

# else np.full(licks_transposed_series_left_rs_opto.shape[:1], np.nan)

            # first_elements = [sublist[0] for sublist in series_right_rl if sublist]
            
            avg_lick_traces = [
                avg_lick_trace_series_left_pl,
                avg_lick_trace_series_left_pl_opto,
                avg_lick_trace_series_right_rl,
                avg_lick_trace_series_right_rl_opto
            ]
            
            # Step 1: Find the maximum length of the traces
            max_length = max(len(trace) for trace in avg_lick_traces)
            
            
            padded_lick_traces = [np.pad(trace, (0, max_length - len(trace)), mode='constant', constant_values=np.nan).tolist() for trace in avg_lick_traces]
            y_positions = np.arange(len(padded_lick_traces))
            num_traces = len(padded_lick_traces)            
            
            

            labels = ['Left Lick', \
                      'Left Lick - Opto', \
                      'Right Lick', \
                      'Right Lick - Opto',]
       
            num_sublists_1 = len(series_left_pl)
            num_sublists_2 = len(series_left_pl_opto)
            num_sublists_3 = len(series_right_rl)
            num_sublists_4 = len(series_right_rl_opto)
        
            y_positions = list(range(num_sublists_1)) + \
                list(range(num_sublists_1, num_sublists_1 + num_sublists_2)) + \
                list(range(num_sublists_1 + num_sublists_2, num_sublists_1 + num_sublists_2 + num_sublists_3)) + \
                list(range(num_sublists_1 + num_sublists_2 + num_sublists_3, num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4))            
            
            # Updated y_positions with space for averages
            y_positions = []
            modified_traces = []  # Will contain avg traces + original traces
            
            sublists = [series_left_pl, series_left_pl_opto, series_right_rl, series_right_rl_opto]
            
            y_offset = 0  # Keeps track of row index
            for idx, sublist in enumerate(sublists):
                # Add the original traces with updated positions
                for trace in sublist:
                    modified_traces.append(trace)
                    y_positions.append(y_offset)
                    y_offset += 1  # Shift for next row  
                    
                # Insert the average trace above the sublist
                modified_traces.append(padded_lick_traces[idx])
                y_positions.append(y_offset)  # Position for average
                y_offset += 1  # Shift position
            
          
            
            y_ticks = list(range(len(modified_traces)))
            
            # Define colors: one color per group
            colorlist = ['green', 'lightgreen', 'dimgray', 'lightgray']
            colors = ['green'] * num_sublists_1 + ['lightgreen'] * num_sublists_2  + \
                ['dimgray'] * num_sublists_3 + ['lightgray'] * num_sublists_4 
            
            colors = ['green'] * num_sublists_1 + \
                ['red'] * 1 + \
                ['lightgreen'] * num_sublists_2  + \
                ['red'] * 1 + \
                ['dimgray'] * num_sublists_3 + \
                ['red'] * 1 + \
                ['lightgray'] * num_sublists_4 + \
                ['red'] * 1
          
            
            # Combine all events
            all_events = series_left_pl + series_left_pl_opto + series_right_rl + series_right_rl_opto # Concatenate both groups   

            axs[row , j].vlines(0 ,len(y_positions), 0, linestyle='--', color='grey')
            
            axs[row,  j].eventplot(modified_traces, color=colors, linelengths=0.6, lineoffsets=y_positions)
                      
            # axs[row , j].eventplot(all_events, color=colors, lineoffsets=y_positions, linelengths = 0.6)
            
            axs[row , j].set_xlim([xlim_left,xlim_right])
            axs[row , j].set_title('Type: Long ISI, Sorted: Trial Order, Aligned: ' + alignments[j])

            # Add one empty plot for each color-label pair
            for k in range(len(colorlist)):
                axs[row , j].plot([], [], color=colorlist[k], label=labels[k])



            # Define y-ticks and labels for each sublist
            num_sublists_1_len = num_sublists_1 + 1
            num_sublists_2_len = num_sublists_2 + 1
            num_sublists_3_len = num_sublists_3 + 1
            num_sublists_4_len = num_sublists_4 + 1

            # yticks = [num_sublists_1 / 2, num_sublists_1 + num_sublists_2 / 2, 
            #           num_sublists_1 + num_sublists_2 + num_sublists_3 / 2, 
            #           num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4 / 2]
            
            yticks = [num_sublists_1_len / 2, num_sublists_1_len + num_sublists_2_len / 2, 
                      num_sublists_1_len + num_sublists_2_len + num_sublists_3_len / 2, 
                      num_sublists_1_len + num_sublists_2_len + num_sublists_3_len + num_sublists_4_len / 2]  
            
            yticklabels = ['Left', 'Left Opto', 'Right', 'Right Opto']
            
            # # Set yticks and labels
            # axs[row , j].set_yticks(yticks)
            # axs[row , j].set_yticklabels(yticklabels)
            
            trial_yticks = np.arange(0, len(modified_traces) + 1, 10)
            axs[row , j].set_yticks(trial_yticks)
            axs[row , j].set_yticklabels([str(y) for y in trial_yticks])
            axs[row , j].set_ylabel("Trials", color='black')            
            
            line_thickness = 0.5  # Adjust line thickness here
            
            # Draw partition lines to divide sublists
            axs[row , j].axhline(y=num_sublists_1 + 0.5, color='grey', linestyle='--', linewidth=line_thickness)
            axs[row , j].axhline(y=num_sublists_1 + num_sublists_2 + 1.5, color='grey', linestyle='--', linewidth=line_thickness)
            axs[row , j].axhline(y=num_sublists_1 + num_sublists_2 + num_sublists_3 + 2.5, color='grey', linestyle='--', linewidth=line_thickness) 


            # Add legend
            axs[row , j].legend(loc='best', bbox_to_anchor=(1,1), ncol=1)


            # Create a twin y-axis
            axs2 = axs[row , j].twinx() 
            
            # Set different tick marks (e.g., transformed scale)
            axs2.set_ylim(axs[row , j].get_ylim())  # Keep same limits
            # axs2.set_yticks(y_positions)  # tick positions
            # axs2.set_yticklabels(y_ticks)  # tick labels
            # axs2.set_ylabel("Trials", color='black')
            axs2.set_yticks(yticks)  # tick positions
            axs2.set_yticklabels(yticklabels)  # tick labels            
            axs2.set_ylabel("Lick Type", color='black')            
            axs2.tick_params(axis='y', labelcolor='black')      

            # 'reward, short, ' + alignments[j]
            # time-to-first-lick ordered
            ################################################################################################################
            if 0:
                row = row + 1
                   
                labels = ['Left Lick', \
                          'Left Lick - Opto', \
                          'Right Lick', \
                          'Right Lick - Opto',]
                   
                series_left_rs_sorted = sorted(series_left_rs, key=lambda x: x[0])
                series_left_rs_opto_sorted = sorted(series_left_rs_opto, key=lambda x: x[0])            
                series_right_ps_sorted = sorted(series_right_ps, key=lambda x: x[0])
                series_right_ps_opto_sorted = sorted(series_right_ps_opto, key=lambda x: x[0])
                
                
                ##############            
                # Flattened y-positions: each sublist gets a unique row
                num_sublists_1 = len(series_left_rs_sorted)
                num_sublists_2 = len(series_left_rs_opto_sorted)
                num_sublists_3 = len(series_right_ps_sorted)
                num_sublists_4 = len(series_right_ps_opto_sorted)
                            
                y_positions = list(range(num_sublists_1)) + list(range(num_sublists_1, num_sublists_1 + num_sublists_2)) + \
                    list(range(num_sublists_1 + num_sublists_2, num_sublists_1 + num_sublists_2 + num_sublists_3)) + \
                    list(range(num_sublists_1 + num_sublists_2 + num_sublists_3, num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4))
                
                y_ticks = list(range(num_sublists_1)) + list(range(num_sublists_2)) + list(range(num_sublists_3)) + list(range(num_sublists_4))
                
                y_ticks = [x + 1 for x in y_ticks]            
                
                # Define colors: one color per group
                colors = ['green'] * num_sublists_1 + ['lightgreen'] * num_sublists_2  + \
                    ['dimgray'] * num_sublists_3 + ['lightgray'] * num_sublists_4 
                
                # Combine all events
                all_events = series_left_rs_sorted + series_left_rs_opto_sorted + series_right_ps_sorted + series_right_ps_opto_sorted # Concatenate both groups                                        
                
                axs[row , j].eventplot(all_events, color=colors, lineoffsets=y_positions, linelengths = 0.6)
                   
                axs[row , j].vlines(0 ,len(y_positions), 0, linestyle='--', color='grey')
                                    
                axs[row , j].set_xlim([xlim_left,xlim_right])
                axs[row , j].set_title('Type: Short ISI, Sorted: First Lick, Aligned: ' + alignments[j])                
                   
                # Add one empty plot for each color-label pair
                for k in range(len(colorlist)):
                    axs[row , j].plot([], [], color=colorlist[k], label=labels[k])
                   
                # Add legend
                axs[row , j].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
                   
                # Define y-ticks and labels for each sublist
                yticks = [num_sublists_1 / 2, num_sublists_1 + num_sublists_2 / 2, 
                          num_sublists_1 + num_sublists_2 + num_sublists_3 / 2, 
                          num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4 / 2]
                
                yticklabels = ['Left', 'Left Opto', 'Right', 'Right Opto']
                
                # Set yticks and labels
                axs[row , j].set_yticks(yticks)
                axs[row , j].set_yticklabels(yticklabels)
                
                line_thickness = 0.5  # Adjust line thickness here
                
                # Draw partition lines to divide sublists
                axs[row , j].axhline(y=num_sublists_1, color='grey', linestyle='--', linewidth=line_thickness)
                axs[row , j].axhline(y=num_sublists_1 + num_sublists_2, color='grey', linestyle='--', linewidth=line_thickness)
                axs[row , j].axhline(y=num_sublists_1 + num_sublists_2 + num_sublists_3, color='grey', linestyle='--', linewidth=line_thickness) 
                   
                # Create a twin y-axis
                axs2 = axs[row , j].twinx() 
                
                # Set different tick marks (e.g., transformed scale)
                axs2.set_ylim(axs[row , j].get_ylim())  # Keep same limits
                axs2.set_yticks(y_positions)  # tick positions
                axs2.set_yticklabels(y_ticks)  # tick labels
                axs2.set_ylabel("Trials", color='black')
                axs2.tick_params(axis='y', labelcolor='black',  size=2, labelsize=3)      

            # 'reward, short, ' + alignments[j]
            # time-to-first-lick ordered
            ################################################################################################################

            row= row + 1
            
            series_left_rs_sorted = sorted(series_left_rs, key=lambda x: x[0])
            series_left_rs_opto_sorted = sorted(series_left_rs_opto, key=lambda x: x[0])            
            series_right_ps_sorted = sorted(series_right_ps, key=lambda x: x[0])
            series_right_ps_opto_sorted = sorted(series_right_ps_opto, key=lambda x: x[0])

            # Step 1: Find the minimum number of licks across all trials
            min_length_series_left_rs = min((len(x) for x in series_left_rs), default=0)
            min_length_series_left_rs_opto = min((len(x) for x in series_left_rs_opto), default=0)
            min_length_series_right_ps = min((len(x) for x in series_right_ps), default=0)
            min_length_series_right_ps_opto = min((len(x) for x in series_right_ps_opto), default=0)
            
            
            # Step 2: Limit each trial to the minimum number of licks
            limited_licks_series_left_rs = [x[:min_length_series_left_rs] for x in series_left_rs]
            
            limited_licks_series_left_rs_opto = [x[:min_length_series_left_rs_opto] for x in series_left_rs_opto]
            limited_licks_series_right_ps = [x[:min_length_series_right_ps] for x in series_right_ps]
            limited_licks_series_right_ps_opto = [x[:min_length_series_right_ps_opto] for x in series_right_ps_opto]
            
            # Step 3: Transpose to group licks by index (1st, 2nd, 3rd lick)
            licks_transposed_series_left_rs = np.array(limited_licks_series_left_rs).T            
            licks_transposed_series_left_rs_opto = np.array(limited_licks_series_left_rs_opto).T
            licks_transposed_series_right_ps = np.array(limited_licks_series_right_ps).T
            licks_transposed_series_right_ps_opto = np.array(limited_licks_series_right_ps_opto).T
            
            # Step 4: Calculate the element-wise average of the licks across trials
            # avg_lick_trace_series_left_rs = np.mean(licks_transposed_series_left_rs, axis=1)                          
            # avg_lick_trace_series_left_rs_opto = np.mean(licks_transposed_series_left_rs_opto, axis=1)              
            # avg_lick_trace_series_right_ps = np.mean(licks_transposed_series_right_ps, axis=1)              
            # avg_lick_trace_series_right_ps_opto = np.mean(licks_transposed_series_right_ps_opto, axis=1)     
            
            
            avg_lick_trace_series_left_rs = np.mean(licks_transposed_series_left_rs, axis=1) if licks_transposed_series_left_rs.ndim > 1 else np.array([0], dtype=float)                       
            avg_lick_trace_series_left_rs_opto = np.mean(licks_transposed_series_left_rs_opto, axis=1) if licks_transposed_series_left_rs_opto.ndim > 1 else np.array([0], dtype=float)             
            avg_lick_trace_series_right_ps = np.mean(licks_transposed_series_right_ps, axis=1) if licks_transposed_series_right_ps.ndim > 1 else np.array([0], dtype=float)            
            avg_lick_trace_series_right_ps_opto = np.mean(licks_transposed_series_right_ps_opto, axis=1) if licks_transposed_series_right_ps_opto.ndim > 1 else np.array([0])
            
            avg_lick_traces = [
                avg_lick_trace_series_left_rs,
                avg_lick_trace_series_left_rs_opto,
                avg_lick_trace_series_right_ps,
                avg_lick_trace_series_right_ps_opto
            ]
            
            # Step 1: Find the maximum length of the traces
            max_length = max(len(trace) for trace in avg_lick_traces)
            
            
            padded_lick_traces = [np.pad(trace, (0, max_length - len(trace)), mode='constant', constant_values=np.nan).tolist() for trace in avg_lick_traces]
            y_positions = np.arange(len(padded_lick_traces))
            num_traces = len(padded_lick_traces)                        

            labels = ['Left Lick', \
                      'Left Lick - Opto', \
                      'Right Lick', \
                      'Right Lick - Opto',]
       
            num_sublists_1 = len(series_left_rs_sorted)
            num_sublists_2 = len(series_left_rs_opto_sorted)
            num_sublists_3 = len(series_right_ps_sorted)
            num_sublists_4 = len(series_right_ps_opto_sorted)
        
            y_positions = list(range(num_sublists_1)) + \
                list(range(num_sublists_1, num_sublists_1 + num_sublists_2)) + \
                list(range(num_sublists_1 + num_sublists_2, num_sublists_1 + num_sublists_2 + num_sublists_3)) + \
                list(range(num_sublists_1 + num_sublists_2 + num_sublists_3, num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4))            
            
            # Updated y_positions with space for averages
            y_positions = []
            modified_traces = []  # Will contain avg traces + original traces
            
      
            # sublists = [series_left_pl, series_left_pl_opto, series_right_rl, series_right_rl_opto]
            # Combine all events
            sublists = [series_left_rs_sorted, series_left_rs_opto_sorted, series_right_ps_sorted, series_right_ps_opto_sorted] # Concatenate both groups        
            
            y_offset = 0  # Keeps track of row index
            for idx, sublist in enumerate(sublists):
                # Add the original traces with updated positions
                for trace in sublist:
                    modified_traces.append(trace)
                    y_positions.append(y_offset)
                    y_offset += 1  # Shift for next row  
                    
                # Insert the average trace above the sublist
                modified_traces.append(padded_lick_traces[idx])
                y_positions.append(y_offset)  # Position for average
                y_offset += 1  # Shift position
            
          
            
            y_ticks = list(range(len(modified_traces)))
            
            # Define colors: one color per group
            colorlist = ['green', 'lightgreen', 'dimgray', 'lightgray']
            colors = ['green'] * num_sublists_1 + ['lightgreen'] * num_sublists_2  + \
                ['dimgray'] * num_sublists_3 + ['lightgray'] * num_sublists_4 
            
            colors = ['green'] * num_sublists_1 + \
                ['red'] * 1 + \
                ['lightgreen'] * num_sublists_2  + \
                ['red'] * 1 + \
                ['dimgray'] * num_sublists_3 + \
                ['red'] * 1 + \
                ['lightgray'] * num_sublists_4 + \
                ['red'] * 1
          
            
            # Combine all events
            all_events = series_left_pl + series_left_pl_opto + series_right_rl + series_right_rl_opto # Concatenate both groups   

            axs[row , j].vlines(0 ,len(y_positions), 0, linestyle='--', color='grey')
            
            axs[row,  j].eventplot(modified_traces, color=colors, linelengths=0.6, lineoffsets=y_positions)
                      
            # axs[row , j].eventplot(all_events, color=colors, lineoffsets=y_positions, linelengths = 0.6)
            
            axs[row , j].set_xlim([xlim_left,xlim_right])
            axs[row , j].set_title('Type: Short ISI, Sorted: First Lick, Aligned: ' + alignments[j])

            # Add one empty plot for each color-label pair
            for k in range(len(colorlist)):
                axs[row , j].plot([], [], color=colorlist[k], label=labels[k])



            # Define y-ticks and labels for each sublist
            num_sublists_1_len = num_sublists_1 + 1
            num_sublists_2_len = num_sublists_2 + 1
            num_sublists_3_len = num_sublists_3 + 1
            num_sublists_4_len = num_sublists_4 + 1

            # yticks = [num_sublists_1 / 2, num_sublists_1 + num_sublists_2 / 2, 
            #           num_sublists_1 + num_sublists_2 + num_sublists_3 / 2, 
            #           num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4 / 2]
            
            yticks = [num_sublists_1_len / 2, num_sublists_1_len + num_sublists_2_len / 2, 
                      num_sublists_1_len + num_sublists_2_len + num_sublists_3_len / 2, 
                      num_sublists_1_len + num_sublists_2_len + num_sublists_3_len + num_sublists_4_len / 2]  
            
            yticklabels = ['Left', 'Left Opto', 'Right', 'Right Opto']
            
            # # Set yticks and labels
            # axs[row , j].set_yticks(yticks)
            # axs[row , j].set_yticklabels(yticklabels)
            
            trial_yticks = np.arange(0, len(modified_traces) + 1, 10)
            axs[row , j].set_yticks(trial_yticks)
            axs[row , j].set_yticklabels([str(y) for y in trial_yticks])
            axs[row , j].set_ylabel("Trials", color='black')            
            
            line_thickness = 0.5  # Adjust line thickness here
            
            # Draw partition lines to divide sublists
            axs[row , j].axhline(y=num_sublists_1 + 0.5, color='grey', linestyle='--', linewidth=line_thickness)
            axs[row , j].axhline(y=num_sublists_1 + num_sublists_2 + 1.5, color='grey', linestyle='--', linewidth=line_thickness)
            axs[row , j].axhline(y=num_sublists_1 + num_sublists_2 + num_sublists_3 + 2.5, color='grey', linestyle='--', linewidth=line_thickness) 


            # Add legend
            axs[row , j].legend(loc='best', bbox_to_anchor=(1,1), ncol=1)


            # Create a twin y-axis
            axs2 = axs[row , j].twinx() 
            
            # Set different tick marks (e.g., transformed scale)
            axs2.set_ylim(axs[row , j].get_ylim())  # Keep same limits
            # axs2.set_yticks(y_positions)  # tick positions
            # axs2.set_yticklabels(y_ticks)  # tick labels
            # axs2.set_ylabel("Trials", color='black')
            axs2.set_yticks(yticks)  # tick positions
            axs2.set_yticklabels(yticklabels)  # tick labels            
            axs2.set_ylabel("Lick Type", color='black')            
            axs2.tick_params(axis='y', labelcolor='black')    


            # 'reward, long, ' + alignments[j]
            # time-to-first-lick ordered
            ################################################################################################################
                    
  
            row = row + 1
            
            # Combine all events
            # all_events = series_left_rs + series_left_rs_opto + series_right_rl + series_right_rl_opto # Concatenate both groups        
            
            series_left_pl_sorted = sorted(series_left_pl, key=lambda x: x[0])
            series_left_pl_opto_sorted = sorted(series_left_pl_opto, key=lambda x: x[0])            
            series_right_rl_sorted = sorted(series_right_rl, key=lambda x: x[0])
            series_right_rl_opto_sorted = sorted(series_right_rl_opto, key=lambda x: x[0])   
            
            
            # Step 1: Find the minimum number of licks across all trials
            min_length_series_left_pl = min((len(x) for x in series_left_pl), default=0)
            min_length_series_left_pl_opto = min((len(x) for x in series_left_pl_opto), default=0)
            min_length_series_right_rl = min((len(x) for x in series_right_rl), default=0)
            min_length_series_right_rl_opto = min((len(x) for x in series_right_rl_opto), default=0)
            

            # Step 2: Limit each trial to the minimum number of licks
            limited_licks_series_left_pl = [x[:min_length_series_left_pl] for x in series_left_pl]            
            limited_licks_series_left_pl_opto = [x[:min_length_series_left_pl_opto] for x in series_left_pl_opto]
            limited_licks_series_right_rl = [x[:min_length_series_right_rl] for x in series_right_rl]
            limited_licks_series_right_rl_opto = [x[:min_length_series_right_rl_opto] for x in series_right_rl_opto]
            
            # Step 3: Transpose to group licks by index (1st, 2nd, 3rd lick)
            licks_transposed_series_left_pl = np.array(limited_licks_series_left_pl).T            
            licks_transposed_series_left_pl_opto = np.array(limited_licks_series_left_pl_opto).T
            licks_transposed_series_right_rl = np.array(limited_licks_series_right_rl).T
            licks_transposed_series_right_rl_opto = np.array(limited_licks_series_right_rl_opto).T
            
            # Step 4: Calculate the element-wise average of the licks across trials
            avg_lick_trace_series_left_pl = np.mean(licks_transposed_series_left_pl, axis=1) if licks_transposed_series_left_pl.ndim > 1 else np.array([0], dtype=float)                          
            avg_lick_trace_series_left_pl_opto = np.mean(licks_transposed_series_left_pl_opto, axis=1) if licks_transposed_series_left_pl_opto.ndim > 1 else np.array([0], dtype=float)              
            avg_lick_trace_series_right_rl = np.mean(licks_transposed_series_right_rl, axis=1) if licks_transposed_series_right_rl.ndim > 1 else np.array([0], dtype=float)              
            avg_lick_trace_series_right_rl_opto = np.mean(licks_transposed_series_right_rl_opto, axis=1) if licks_transposed_series_right_rl_opto.ndim > 1 else np.array([0], dtype=float)  

            # first_elements = [sublist[0] for sublist in series_right_rl if sublist]
            
            avg_lick_traces = [
                avg_lick_trace_series_left_pl,
                avg_lick_trace_series_left_pl_opto,
                avg_lick_trace_series_right_rl,
                avg_lick_trace_series_right_rl_opto
            ]
            
            # Step 1: Find the maximum length of the traces
            max_length = max(len(trace) for trace in avg_lick_traces)
            
            
            padded_lick_traces = [np.pad(trace, (0, max_length - len(trace)), mode='constant', constant_values=np.nan).tolist() for trace in avg_lick_traces]
            y_positions = np.arange(len(padded_lick_traces))
            num_traces = len(padded_lick_traces)            
            
            

            labels = ['Left Lick', \
                      'Left Lick - Opto', \
                      'Right Lick', \
                      'Right Lick - Opto',]
       
            num_sublists_1 = len(series_left_pl)
            num_sublists_2 = len(series_left_pl_opto)
            num_sublists_3 = len(series_right_rl)
            num_sublists_4 = len(series_right_rl_opto)
        
            y_positions = list(range(num_sublists_1)) + \
                list(range(num_sublists_1, num_sublists_1 + num_sublists_2)) + \
                list(range(num_sublists_1 + num_sublists_2, num_sublists_1 + num_sublists_2 + num_sublists_3)) + \
                list(range(num_sublists_1 + num_sublists_2 + num_sublists_3, num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4))            
            
            # Updated y_positions with space for averages
            y_positions = []
            modified_traces = []  # Will contain avg traces + original traces            
            
            # sublists = [series_left_pl, series_left_pl_opto, series_right_rl, series_right_rl_opto]
            sublists = [series_left_pl_sorted, series_left_pl_opto_sorted, series_right_rl_sorted, series_right_rl_opto_sorted]
            
            y_offset = 0  # Keeps track of row index
            for idx, sublist in enumerate(sublists):
                # Add the original traces with updated positions
                for trace in sublist:
                    modified_traces.append(trace)
                    y_positions.append(y_offset)
                    y_offset += 1  # Shift for next row  
                    
                # Insert the average trace above the sublist
                modified_traces.append(padded_lick_traces[idx])
                y_positions.append(y_offset)  # Position for average
                y_offset += 1  # Shift position
            
          
            
            y_ticks = list(range(len(modified_traces)))
            
            # Define colors: one color per group
            colorlist = ['green', 'lightgreen', 'dimgray', 'lightgray']
            colors = ['green'] * num_sublists_1 + ['lightgreen'] * num_sublists_2  + \
                ['dimgray'] * num_sublists_3 + ['lightgray'] * num_sublists_4 
            
            colors = ['green'] * num_sublists_1 + \
                ['red'] * 1 + \
                ['lightgreen'] * num_sublists_2  + \
                ['red'] * 1 + \
                ['dimgray'] * num_sublists_3 + \
                ['red'] * 1 + \
                ['lightgray'] * num_sublists_4 + \
                ['red'] * 1
          
            
            # Combine all events
            all_events = series_left_pl + series_left_pl_opto + series_right_rl + series_right_rl_opto # Concatenate both groups   

            axs[row , j].vlines(0 ,len(y_positions), 0, linestyle='--', color='grey')
            
            axs[row,  j].eventplot(modified_traces, color=colors, linelengths=0.6, lineoffsets=y_positions)
                      
            # axs[row , j].eventplot(all_events, color=colors, lineoffsets=y_positions, linelengths = 0.6)
            
            axs[row , j].set_xlim([xlim_left,xlim_right])
            axs[row , j].set_title('Type: Long ISI, Sorted: First Lick, Aligned: ' + alignments[j])

            # Add one empty plot for each color-label pair
            for k in range(len(colorlist)):
                axs[row , j].plot([], [], color=colorlist[k], label=labels[k])



            # Define y-ticks and labels for each sublist
            num_sublists_1_len = num_sublists_1 + 1
            num_sublists_2_len = num_sublists_2 + 1
            num_sublists_3_len = num_sublists_3 + 1
            num_sublists_4_len = num_sublists_4 + 1

            # yticks = [num_sublists_1 / 2, num_sublists_1 + num_sublists_2 / 2, 
            #           num_sublists_1 + num_sublists_2 + num_sublists_3 / 2, 
            #           num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4 / 2]
            
            yticks = [num_sublists_1_len / 2, num_sublists_1_len + num_sublists_2_len / 2, 
                      num_sublists_1_len + num_sublists_2_len + num_sublists_3_len / 2, 
                      num_sublists_1_len + num_sublists_2_len + num_sublists_3_len + num_sublists_4_len / 2]  
            
            yticklabels = ['Left', 'Left Opto', 'Right', 'Right Opto']
            
            # # Set yticks and labels
            # axs[row , j].set_yticks(yticks)
            # axs[row , j].set_yticklabels(yticklabels)
            
            trial_yticks = np.arange(0, len(modified_traces) + 1, 10)
            axs[row , j].set_yticks(trial_yticks)
            axs[row , j].set_yticklabels([str(y) for y in trial_yticks])
            axs[row , j].set_ylabel("Trials", color='black')            
            
            line_thickness = 0.5  # Adjust line thickness here
            
            # Draw partition lines to divide sublists
            axs[row , j].axhline(y=num_sublists_1 + 0.5, color='grey', linestyle='--', linewidth=line_thickness)
            axs[row , j].axhline(y=num_sublists_1 + num_sublists_2 + 1.5, color='grey', linestyle='--', linewidth=line_thickness)
            axs[row , j].axhline(y=num_sublists_1 + num_sublists_2 + num_sublists_3 + 2.5, color='grey', linestyle='--', linewidth=line_thickness) 


            # Add legend
            axs[row , j].legend(loc='best', bbox_to_anchor=(1,1), ncol=1)


            # Create a twin y-axis
            axs2 = axs[row , j].twinx() 
            
            # Set different tick marks (e.g., transformed scale)
            axs2.set_ylim(axs[row , j].get_ylim())  # Keep same limits
            # axs2.set_yticks(y_positions)  # tick positions
            # axs2.set_yticklabels(y_ticks)  # tick labels
            # axs2.set_ylabel("Trials", color='black')
            axs2.set_yticks(yticks)  # tick positions
            axs2.set_yticklabels(yticklabels)  # tick labels            
            axs2.set_ylabel("Lick Type", color='black')            
            axs2.tick_params(axis='y', labelcolor='black')              
    
            
            
            
            
            
            
            
            
            
            
            
            # 'reward, long, ' + alignments[j]
            # time-to-first-lick ordered
            ################################################################################################################
            if 0:
                row = row + 1
    
                labels = ['Left Lick', \
                          'Left Lick - Opto', \
                          'Right Lick', \
                          'Right Lick - Opto',]
    
                series_left_pl_sorted = sorted(series_left_pl, key=lambda x: x[0])
                series_left_pl_opto_sorted = sorted(series_left_pl_opto, key=lambda x: x[0])            
                series_right_rl_sorted = sorted(series_right_rl, key=lambda x: x[0])
                series_right_rl_opto_sorted = sorted(series_right_rl_opto, key=lambda x: x[0])
                
                
                ##############            
                # Flattened y-positions: each sublist gets a unique row
                num_sublists_1 = len(series_left_pl_sorted)
                num_sublists_2 = len(series_left_pl_opto_sorted)
                num_sublists_3 = len(series_right_rl_sorted)
                num_sublists_4 = len(series_right_rl_opto_sorted)
              
                y_positions = list(range(num_sublists_1)) + list(range(num_sublists_1, num_sublists_1 + num_sublists_2)) + \
                    list(range(num_sublists_1 + num_sublists_2, num_sublists_1 + num_sublists_2 + num_sublists_3)) + \
                    list(range(num_sublists_1 + num_sublists_2 + num_sublists_3, num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4))
                
                y_ticks = list(range(num_sublists_1)) + list(range(num_sublists_2)) + list(range(num_sublists_3)) + list(range(num_sublists_4))
                
                y_ticks = [x + 1 for x in y_ticks]               
                
                # Define colors: one color per group
                colors = ['green'] * num_sublists_1 + ['lightgreen'] * num_sublists_2  + \
                    ['dimgray'] * num_sublists_3 + ['lightgray'] * num_sublists_4 
                
                # Combine all events
                all_events = series_left_pl_sorted + series_left_pl_opto_sorted + series_right_rl_sorted + series_right_rl_opto_sorted # Concatenate both groups        
    
                axs[row , j].vlines(0 ,len(y_positions), 0, linestyle='--', color='grey')
                          
                axs[row , j].eventplot(all_events, color=colors, lineoffsets=y_positions, linelengths = 0.6)
                
                axs[row , j].set_xlim([xlim_left,xlim_right])
                axs[row , j].set_title('Type: Long ISI, Sorted: First Lick, Aligned: ' + alignments[j])
    
                # Add one empty plot for each color-label pair
                for k in range(len(colorlist)):
                    axs[row , j].plot([], [], color=colorlist[k], label=labels[k])
    
                # Add legend
                axs[row , j].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    
                # Define y-ticks and labels for each sublist
                yticks = [num_sublists_1 / 2, num_sublists_1 + num_sublists_2 / 2, 
                          num_sublists_1 + num_sublists_2 + num_sublists_3 / 2, 
                          num_sublists_1 + num_sublists_2 + num_sublists_3 + num_sublists_4 / 2]
                
                yticklabels = ['Left', 'Left Opto', 'Right', 'Right Opto']
                
                # Set yticks and labels
                axs[row , j].set_yticks(yticks)
                axs[row , j].set_yticklabels(yticklabels)
                
                line_thickness = 0.5  # Adjust line thickness here
                
                # Draw partition lines to divide sublists
                axs[row , j].axhline(y=num_sublists_1, color='grey', linestyle='--', linewidth=line_thickness)
                axs[row , j].axhline(y=num_sublists_1 + num_sublists_2, color='grey', linestyle='--', linewidth=line_thickness)
                axs[row , j].axhline(y=num_sublists_1 + num_sublists_2 + num_sublists_3, color='grey', linestyle='--', linewidth=line_thickness) 
    
                # Create a twin y-axis
                axs2 = axs[row , j].twinx() 
                
                # Set different tick marks (e.g., transformed scale)
                axs2.set_ylim(axs[row , j].get_ylim())  # Keep same limits
                axs2.set_yticks(y_positions)  # tick positions
                axs2.set_yticklabels(y_ticks)  # tick labels
                axs2.set_ylabel("Trials", color='black')
                axs2.tick_params(axis='y', labelcolor='black',  size=2, labelsize=3)      
            # 'punish, short, ' + alignments[j]
            # trial ordered
            ################################################################################################################


            # 'punish, short, ' + alignments[j]
            # time-to-first-lick ordered
            ################################################################################################################



            if 0:
                axs[2 , j].vlines(0 ,len(series_left_ps), 0, linestyle='--', color='grey')
                axs[2 , j].eventplot(series_center_ps, color='black', linelengths = 0.3)
                axs[2 , j].eventplot(series_right_ps, color='red', linelengths = 0.3)
                axs[2 , j].eventplot(series_left_ps, color='limegreen', linelengths = 0.3)
                axs[2 , j].set_xlim([xlim_left,xlim_right])
                axs[2 , j].set_title('punish, short, ' + alignments[j])
                # if len(series_center_ps) > 0:
                    
                axs[2 , j].hist(np.concatenate(series_center_ps), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'black', alpha = 0.4)
                axs[2 , j].hist(np.concatenate(series_right_ps), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'red', alpha = 0.4)
                axs[2 , j].hist(np.concatenate(series_left_ps), bins=n_bins, histtype='step', stacked=True, fill=False , color = 'green', alpha = 0.4)







                axs[3 , j].vlines(0 ,len(series_left_pl), 0, linestyle='--', color='grey')
                axs[3 , j].eventplot(series_center_pl, color='black', linelengths = 0.3)
                axs[3 , j].eventplot(series_right_pl, color='red', linelengths = 0.3)
                axs[3 , j].eventplot(series_left_pl, color='limegreen', linelengths = 0.3)
                axs[3 , j].set_xlim([xlim_left,xlim_right])
                axs[3 , j].set_title('punish, long, ' + alignments[j])
                # if len(series_center_pl) > 0:
                    
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
        output_pdf_pages_dir = output_dir_local + subject + '/lick_traces/lick_traces_' + session_date + '/'
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
        




