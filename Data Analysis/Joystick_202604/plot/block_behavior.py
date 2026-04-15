# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 11:48:54 2025

@author: saminnaji3
"""


import numpy as np
import matplotlib.pyplot as plt
start_trial = 10

def IsSelfTimed(session_data):
    isSelfTime = session_data['isSelfTimedMode']
    
    VG = []
    ST = []
    for i in range(0 , len(isSelfTime)):
        if isSelfTime[i][start_trial] == 1 or isSelfTime[i][start_trial] == np.nan:
            ST.append(i)
        else:
            VG.append(i)
    return ST , VG

def plot_brokenaxis(axs, array1, array2, array3, array4, title, pre, post, ylabel, sem1 = np.nan, sem2 = np.nan, sem3 = np.nan, sem4 = np.nan, sem_plot = 0):
    #print(array1)
    x1 = np.arange(len(array1))
    x2 = np.arange(len(array2)) + x1[-1] + 6
    #bax = brokenaxes(xlims=((x1[0] , x1[-1]), (x2[0], x2[-1])), subplot_spec=axs)
    #bax.plot(x1, array1, label=label, color=color_tag)
    #bax.plot(x2, array2, color=color_tag)
    axs.plot(x1[10:], array1[10:], label='short', color='y')
    axs.plot(x2[:-10], array2[:-10], color='y')
    axs.plot(x1[10:], array3[10:], label='long', color='b')
    axs.plot(x2[:-10], array4[:-10], color='b')
    
    axs.plot(x1[:11], array1[:11], label='short', color='b')
    axs.plot(x2[-11:], array2[-11:], color='b')
    axs.plot(x1[:11], array3[:11], label='long', color='y')
    axs.plot(x2[-11:], array4[-11:], color='y')
    
    if sem_plot:
        axs.fill_between(x1[10:], array1[10:]-sem1[10:], array1[10:]+sem1[10:], color='y', alpha=0.3)
        axs.fill_between(x2[:-10], array2[:-10]-sem2[:-10], array2[:-10]+sem2[:-10], color='y', alpha=0.3)
        axs.fill_between(x1[10:], array3[10:]-sem3[10:], array3[10:]+sem3[10:], color='b', alpha=0.3)
        axs.fill_between(x2[:-10], array4[:-10]-sem4[:-10], array4[:-10]+sem4[:-10], color='b', alpha=0.3)
        
        axs.fill_between(x1[:11], array1[:11]-sem1[:11], array1[:11]+sem1[:11], color='b', alpha=0.3)
        axs.fill_between(x2[-11:], array2[-11:]-sem2[-11:], array2[-11:]+sem2[-11:], color='b', alpha=0.3)
        axs.fill_between(x1[:11], array3[:11]-sem3[:11], array3[:11]+sem3[:11], color='y', alpha=0.3)
        axs.fill_between(x2[-11:], array4[-11:]-sem4[-11:], array4[-11:]+sem4[-11:], color='y', alpha=0.3)
    axs.axvline(x = pre, color = 'grey', linewidth = 1, linestyle='--')
    axs.axvline(x = x2[-1]-post, color = 'grey', linewidth = 1, linestyle='--')
    
    axs.set_xticks([0 , 5 , 10 , 15 , 20 , 25, 30, 34, 39, 44, 49, 54, 59, 64, 69, 74])
    x_labels = [-10, -5, 'start' , 5, 10, 15, 20, 24, -25, -20, -15, -10, -5, 'end', 5, 10]
    axs.set_xticklabels(x_labels)
    axs.set_ylabel(ylabel)
    axs.set_title(title)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    
def first_block_array(block_data, refrence, st, pre, post):
    short_initial = []
    long_initial = []
    short_end = []
    long_end = []
    width = 25
    
    for i in st:
        curr_data = block_data[refrence][i]
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        
        if len(long_id) > 0:
            if short_id[0] == 0:
                short_id = short_id[1:]
        if len(long_id) > 0:
            if long_id[0] ==0:
                long_id = long_id[1:]
        
        block_len = len(block_data[refrence][i][short_id[0]])
        if block_len< width:
            nan_pad = np.zeros(width-block_len)
            nan_pad[:] = np.nan
            temp = np.append(curr_data[short_id[0]][0:block_len] , nan_pad)
            short_initial.append(np.append(curr_data[short_id[0]-1][-pre:], temp))
        else:
            short_initial.append(np.append(curr_data[short_id[0]-1][-pre:],curr_data[short_id[0]][0:width]))
            
        if block_len< width:
            nan_pad = np.zeros(width-block_len)
            nan_pad[:] = np.nan
            temp = np.append(nan_pad, curr_data[short_id[0]][0:block_len])
            if len(curr_data)>short_id[0]+1:
                if len(curr_data[short_id[0]+1]) > post:
                    short_end.append(np.append(temp, curr_data[short_id[0]+1][:post]))
                else:
                    nan_pad1 = np.zeros(post-len(curr_data[short_id[0]+1]))
                    nan_pad1[:] = np.nan
                    temp1 = np.append(temp, curr_data[short_id[0]+1])
                    short_end.append(np.append(temp1, nan_pad1))
            else:
                nan_pad1 = np.zeros(post)
                nan_pad1[:] = np.nan
                short_end.append(np.append(temp, nan_pad1))
        else:
            if len(curr_data)>short_id[0]+1:
                if len(curr_data[short_id[0]+1]) > post:
                    short_end.append(np.append(curr_data[short_id[0]][-width:], curr_data[short_id[0]+1][:post]))
                else:
                    nan_pad1 = np.zeros(post-len(curr_data[short_id[0]+1]))
                    nan_pad1[:] = np.nan
                    temp1 = np.append(curr_data[short_id[0]][-width:], curr_data[short_id[0]+1])
                    short_end.append(np.append(temp1, nan_pad1))
            else:
                nan_pad1 = np.zeros(post)
                nan_pad1[:] = np.nan
                short_end.append(np.append(curr_data[short_id[0]][-width:], nan_pad1))
                
        block_len = len(block_data[refrence][i][long_id[0]])
        if block_len< width:
            nan_pad = np.zeros(width-block_len)
            nan_pad[:] = np.nan
            temp = np.append(curr_data[long_id[0]][0:block_len] , nan_pad)
            long_initial.append(np.append(curr_data[long_id[0]-1][-pre:], temp))
        else:
            long_initial.append(np.append(curr_data[long_id[0]-1][-pre:],curr_data[long_id[0]][0:width]))
            
        if block_len< width:
            nan_pad = np.zeros(width-block_len)
            nan_pad[:] = np.nan
            temp = np.append(nan_pad, curr_data[long_id[0]][0:block_len])
            if len(curr_data)>long_id[0]+1:
                if len(curr_data[long_id[0]+1]) > post:
                    long_end.append(np.append(temp, curr_data[long_id[0]+1][:post]))
                else:
                    nan_pad1 = np.zeros(post-len(curr_data[long_id[0]+1]))
                    nan_pad1[:] = np.nan
                    temp1 = np.append(temp, curr_data[long_id[0]+1])
                    long_end.append(np.append(temp1, nan_pad1))
            else:
                nan_pad1 = np.zeros(post)
                nan_pad1[:] = np.nan
                long_end.append(np.append(temp, nan_pad1))
        else:
            if len(curr_data)>long_id[0]+1:
                if len(curr_data[long_id[0]+1]) > post:
                    long_end.append(np.append(curr_data[long_id[0]][-width:], curr_data[long_id[0]+1][:post]))
                else:
                    nan_pad1 = np.zeros(post-len(curr_data[long_id[0]+1]))
                    nan_pad1[:] = np.nan
                    temp1 = np.append(curr_data[long_id[0]][-width:], curr_data[long_id[0]+1])
                    long_end.append(np.append(temp1, nan_pad1))
            else:
                nan_pad1 = np.zeros(post)
                nan_pad1[:] = np.nan
                long_end.append(np.append(curr_data[long_id[0]][-width:], nan_pad1))
                
    return  short_initial, short_end, long_initial, long_end
        
def rest_block_array(block_data, refrence, st, pre, post, first_exclude = 1):
    short_initial = []
    long_initial = []
    short_end = []
    long_end = []
    width = 25
    
    for i in st:
        curr_data = block_data[refrence][i]
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        
        if len(long_id) > 0:
            if short_id[0] == 0:
                short_id = short_id[1:]
        if len(long_id) > 0:
            if long_id[0] ==0:
                long_id = long_id[1:]
        if first_exclude == 1:
            if len(short_id) > 0:
                short_id = short_id[1:]
            else:
                break
            if len(long_id) > 0:
                long_id = long_id[1:]
            else:
                break
        
        for j in range(len(short_id)):
            block_len = len(block_data[refrence][i][short_id[j]])
            if block_len< width:
                nan_pad = np.zeros(width-block_len)
                nan_pad[:] = np.nan
                temp = np.append(curr_data[short_id[j]][0:block_len] , nan_pad)
                short_initial.append(np.append(curr_data[short_id[j]-1][-pre:], temp))
            else:
                short_initial.append(np.append(curr_data[short_id[j]-1][-pre:],curr_data[short_id[j]][0:width]))
                
            if block_len< width:
                nan_pad = np.zeros(width-block_len)
                nan_pad[:] = np.nan
                temp = np.append(nan_pad, curr_data[short_id[j]][0:block_len])
                if len(curr_data)>short_id[j]+1:
                    if len(curr_data[short_id[j]+1]) > post:
                        short_end.append(np.append(temp, curr_data[short_id[j]+1][:post]))
                    else:
                        nan_pad1 = np.zeros(post-len(curr_data[short_id[j]+1]))
                        nan_pad1[:] = np.nan
                        temp1 = np.append(temp, curr_data[short_id[j]+1])
                        short_end.append(np.append(temp1, nan_pad1))
                else:
                    nan_pad1 = np.zeros(post)
                    nan_pad1[:] = np.nan
                    short_end.append(np.append(temp, nan_pad1))
            else:
                if len(curr_data)>short_id[j]+1:
                    if len(curr_data[short_id[j]+1]) > post:
                        short_end.append(np.append(curr_data[short_id[j]][-width:], curr_data[short_id[j]+1][:post]))
                    else:
                        nan_pad1 = np.zeros(post-len(curr_data[short_id[j]+1]))
                        nan_pad1[:] = np.nan
                        temp1 = np.append(curr_data[short_id[j]][-width:], curr_data[short_id[j]+1])
                        short_end.append(np.append(temp1, nan_pad1))
                else:
                    nan_pad1 = np.zeros(post)
                    nan_pad1[:] = np.nan
                    short_end.append(np.append(curr_data[short_id[j]][-width:], nan_pad1))
        for j in range(len(long_id)):     
            block_len = len(block_data[refrence][i][long_id[j]])
            if block_len< width:
                nan_pad = np.zeros(width-block_len)
                nan_pad[:] = np.nan
                temp = np.append(curr_data[long_id[j]][0:block_len] , nan_pad)
                long_initial.append(np.append(curr_data[long_id[j]-1][-pre:], temp))
            else:
                long_initial.append(np.append(curr_data[long_id[j]-1][-pre:],curr_data[long_id[j]][0:width]))
                
            if block_len< width:
                nan_pad = np.zeros(width-block_len)
                nan_pad[:] = np.nan
                temp = np.append(nan_pad, curr_data[long_id[j]][0:block_len])
                if len(curr_data)>long_id[j]+1:
                    if len(curr_data[long_id[j]+1]) > post:
                        long_end.append(np.append(temp, curr_data[long_id[j]+1][:post]))
                    else:
                        nan_pad1 = np.zeros(post-len(curr_data[long_id[j]+1]))
                        nan_pad1[:] = np.nan
                        temp1 = np.append(temp, curr_data[long_id[j]+1])
                        long_end.append(np.append(temp1, nan_pad1))
                else:
                    nan_pad1 = np.zeros(post)
                    nan_pad1[:] = np.nan
                    long_end.append(np.append(temp, nan_pad1))
            else:
                if len(curr_data)>long_id[j]+1:
                    if len(curr_data[long_id[j]+1]) > post:
                        long_end.append(np.append(curr_data[long_id[j]][-width:], curr_data[long_id[j]+1][:post]))
                    else:
                        nan_pad1 = np.zeros(post-len(curr_data[long_id[j]+1]))
                        nan_pad1[:] = np.nan
                        temp1 = np.append(curr_data[long_id[j]][-width:], curr_data[long_id[j]+1])
                        long_end.append(np.append(temp1, nan_pad1))
                else:
                    nan_pad1 = np.zeros(post)
                    nan_pad1[:] = np.nan
                    long_end.append(np.append(curr_data[long_id[j]][-width:], nan_pad1))
                
    return  short_initial, short_end, long_initial, long_end

def mean_cal(arrays):
    output1 = np.nanmean(np.array(arrays[0]), axis=0)
    sem_output1 = np.nanstd(np.array(arrays[0]) , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(arrays[0]))))
    
    output2 = np.nanmean(np.array(arrays[1]), axis=0)
    sem_output2 = np.nanstd(np.array(arrays[1]) , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(arrays[1]))))
    
    # for i in range(len(arrays[2])):
    #     print(len(arrays[2][i]))
    output3 = np.nanmean(np.array(arrays[2]), axis=0)
    sem_output3 = np.nanstd(np.array(arrays[2]) , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(arrays[2]))))
    
    output4 = np.nanmean(np.array(arrays[3]), axis=0)
    sem_output4 = np.nanstd(np.array(arrays[3]) , axis = 0)/np.sqrt(np.count_nonzero(~np.isnan(np.array(arrays[3]))))
    
    return output1,output2, output3, output4, sem_output1, sem_output2, sem_output3, sem_output4

def count(arrays , outcome):
    output = []
    for i in range(len(arrays)):
        temp = np.zeros(len(arrays[i][0]))
        for j in range(len(arrays[i][0])):
            curr = []
            for t in range(len(arrays[i])):
                curr.append(arrays[i][t][j])
            
            temp[j] = curr.count(outcome)/len(arrays[i])
        output.append(temp)
    return output[0] , output[1] , output[2] , output[3]

def run(axs1, axs2, session_data , block_data, outcome_ref = 'None', st_seperate = 1):
    delay_vector = block_data['delay_all']
    possible_outcomes = ['Reward' , 'EarlyPress2' , 'LatePress2' , 'DidNotPress1' , 'DidNotPress2']
    ST , VG = IsSelfTimed(session_data)
    print(ST)
    print(VG)
    all_sessions = np.arange(len(delay_vector))
    if st_seperate == 1:
        all_st = [all_sessions , np.array(ST) , np.array(VG)]
    else:
        all_st = [all_sessions]
    pre = 10
    post = 10
    st_label = [' (All)' , ' (ST)' , ' (VG)']
    if outcome_ref == 'None':
        refrence = 'delay'
    else:
        refrence = 'outcome'
    #print(len(all_st))
    for st in range(len(all_st)):
        short_initial, short_end, long_initial, long_end = first_block_array(block_data, refrence, all_st[st], pre, post)
        if outcome_ref == 'None':
            
            output1, output2, output3, output4, sem_output1, sem_output2, sem_output3, sem_output4 = mean_cal([short_initial, short_end, long_initial, long_end])
            title = 'First transition' + st_label[st]
            #print(st_label[st], all_st[st])
            plot_brokenaxis(axs1[st], output1, output2, output3, output4, title, pre, post, 'Delay (s)', sem1 = sem_output1, sem2 = sem_output2, sem3 = sem_output3, sem4 = sem_output4,sem_plot = 1)
        else:
            num = 0
            for outcome_curr in possible_outcomes:
                output1, output2, output3, output4 = count([short_initial, short_end, long_initial, long_end] , outcome_curr)
                y_label = 'Fraction of ' + outcome_curr
                title = 'First transition' + st_label[st]
                
                plot_brokenaxis(axs1[num][st], output1, output2, output3, output4, title, pre, post, y_label)
                num = num + 1
            
        short_initial, short_end, long_initial, long_end = rest_block_array(block_data, refrence, all_st[st], pre, post)
        if outcome_ref == 'None':
            title = 'Rest of transitions' + st_label[st]
            output1, output2, output3, output4, sem_output1, sem_output2, sem_output3, sem_output4 = mean_cal([short_initial, short_end, long_initial, long_end])
            
            plot_brokenaxis(axs2[st], output1, output2, output3, output4, title, pre, post, 'Delay (s)', sem1 = sem_output1, sem2 = sem_output2, sem3 = sem_output3, sem4 = sem_output4,sem_plot = 1)
        else:
            num = 0
            for outcome_curr in possible_outcomes:
                output1, output2, output3, output4 = count([short_initial, short_end, long_initial, long_end] , outcome_curr)
                y_label = 'Fraction of ' + outcome_curr
                title = 'Rest of transitions' + st_label[st]
                
                plot_brokenaxis(axs2[num][st], output1, output2, output3, output4, title, pre, post, y_label)
                num = num + 1

def run_all_blocks(axs1, session_data , block_data, outcome_ref = 'None', st_seperate = 1):
    delay_vector = block_data['delay_all']
    possible_outcomes = ['Reward' , 'EarlyPress2' , 'LatePress2' , 'DidNotPress1' , 'DidNotPress2']
    ST , VG = IsSelfTimed(session_data)
    all_sessions = np.arange(len(delay_vector))
    if st_seperate == 1:
        all_st = [all_sessions , np.array(ST) , np.array(VG)]
    else:
        all_st = [all_sessions]
    pre = 10
    post = 10
    st_label = [' (All)' , ' (ST)' , ' (VG)']
    if outcome_ref == 'None':
        refrence = 'delay'
    else:
        refrence = 'outcome'
    #print(len(all_st))
    for st in range(len(all_st)):
        short_initial, short_end, long_initial, long_end = rest_block_array(block_data, refrence, all_st[st], pre, post, first_exclude = 0)
        if outcome_ref == 'None':
            title = 'All transitions' + st_label[st]
            output1, output2, output3, output4, sem_output1, sem_output2, sem_output3, sem_output4 = mean_cal([short_initial, short_end, long_initial, long_end])
            
            plot_brokenaxis(axs1[st], output1, output2, output3, output4, title, pre, post, 'Delay (s)', sem1 = sem_output1, sem2 = sem_output2, sem3 = sem_output3, sem4 = sem_output4,sem_plot = 1)
        else:
            num = 0
            for outcome_curr in possible_outcomes:
                output1, output2, output3, output4 = count([short_initial, short_end, long_initial, long_end] , outcome_curr)
                y_label = 'Fraction of ' + outcome_curr
                title = 'All transitions' + st_label[st]
                
                plot_brokenaxis(axs1[num][st], output1, output2, output3, output4, title, pre, post, y_label)
                num = num + 1
    
    
    
    
    
    