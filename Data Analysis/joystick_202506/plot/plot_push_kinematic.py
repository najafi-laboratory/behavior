# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 15:16:09 2025

@author: saminnaji3
"""

import os
import fitz
import csv
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d

from tqdm import tqdm
from matplotlib.gridspec import GridSpec

def pad_seq(align_data, align_time):
    pad_time = np.arange(
        np.min([np.min(t) for t in align_time]),
        np.max([np.max(t) for t in align_time]) + 1)
    pad_data = []
    for data, time in zip(align_data, align_time):
        aligned_seq = np.full_like(pad_time, np.nan, dtype=float)
        idx = np.searchsorted(pad_time, time)
        aligned_seq[idx] = data
        pad_data.append(aligned_seq)
    return pad_data, pad_time
        
        
def get_js_pos(neural_trials, state):
    interval = 1
    session_align_data = []
    session_align_time = []
    session_trial_type = []
    session_outcome = []
    session_opto = []
    
    for j in range(len(neural_trials['trial_types'])):
        js_time = neural_trials['js_time'][j]
        js_pos = neural_trials['js_pos'][j]
        
        trial_type = neural_trials['trial_types'][j]
        opto = neural_trials['opto'][j]
        epoch = []
        # epoch = np.array([neural_trials[t]['block_epoch']
        #           for t in neural_trials.keys()])
        outcome = neural_trials['outcome'][j]
        inter_time = []
        inter_pos = []
        for (pos, time) in zip(js_pos, js_time):
            interpolator = interp1d(time, pos, bounds_error=False)
            new_time = np.arange(np.min(time), np.max(time), interval)
            new_pos = interpolator(new_time)
            inter_time.append(new_time)
            inter_pos.append(new_pos)
        if np.size(neural_trials[state][j][next(iter(range(len(neural_trials[state][j]))))]) == 1:
            time_state = neural_trials[state][j]
                
        if np.size(neural_trials[state][j][next(iter(range(len(neural_trials[state][j]))))]) == 2:
            time_state = [
                neural_trials[state][j][t][0] 
                for t in range(len(neural_trials[state][j]))]
            
        trial_type = np.array([trial_type[i]
                   for i in range(len(inter_time))
                   if not np.isnan(time_state)[i]])
        opto = np.array([opto[i-1]
                   for i in range(len(inter_time))
                   if not np.isnan(time_state)[i]])
        
        outcome = np.array([outcome[i]
                       for i in range(len(inter_time))
                       if not np.isnan(time_state)[i]])
        zero_state = [np.argmin(np.abs(inter_time[i] - time_state[i]))
                      for i in range(len(inter_time))
                      if not np.isnan(time_state)[i]]
        align_data = [inter_pos[i] - inter_pos[i][0]
                      for i in range(len(inter_pos))
                      if not np.isnan(time_state)[i]]
        align_time = [inter_time[i]
                      for i in range(len(inter_time))
                      if not np.isnan(time_state)[i]]
        align_time = [align_time[i] - align_time[i][zero_state[i]]
                      for i in range(len(align_time))]
        if len(align_data) > 0:
            align_data, align_time = pad_seq(align_data, align_time)
            align_data = np.array(align_data)
        else:
            align_data = np.array([[np.nan]])
            align_time = np.array([np.nan])
            trial_type = np.array([np.nan])
            outcome = np.array([np.nan])
            opto = np.array([np.nan])
    
        session_align_data.append(align_data)
        session_align_time.append(align_time)
        session_trial_type.append(trial_type)
        session_outcome.append(outcome)
        session_opto.append(opto)
        
    return [session_align_data, session_align_time, session_trial_type, session_outcome, session_opto]


def get_js_average_time(neural_trials, state, ref):
    interval = 1
    session_align_data = []
    session_align_time = []
    session_trial_type = []
    session_comp_time = []
    session_outcome = []
    session_opto = []
    
    for j in range(len(neural_trials['trial_types'])):
        js_time = neural_trials['js_time'][j]
        js_pos = neural_trials['js_pos'][j]
        
        trial_type = neural_trials['trial_types'][j]
        opto = neural_trials['opto'][j]
        epoch = []
        # epoch = np.array([neural_trials[t]['block_epoch']
        #           for t in neural_trials.keys()])
        outcome = neural_trials['outcome'][j]
        inter_time = []
        inter_pos = []
        for (pos, time) in zip(js_pos, js_time):
            interpolator = interp1d(time, pos, bounds_error=False)
            new_time = np.arange(np.min(time), np.max(time), interval)
            new_pos = interpolator(new_time)
            inter_time.append(new_time)
            inter_pos.append(new_pos)
        if np.size(neural_trials[state][j][next(iter(range(len(neural_trials[state][j]))))]) == 1:
            time_state = neural_trials[state][j]
                
        if np.size(neural_trials[state][j][next(iter(range(len(neural_trials[state][j]))))]) == 2:
            time_state = [
                neural_trials[state][j][t][0] 
                for t in range(len(neural_trials[state][j]))]
            
            
        if np.size(neural_trials[ref][j][next(iter(range(len(neural_trials[ref][j]))))]) == 1:
            time_ref = neural_trials[ref][j]
                
        if np.size(neural_trials[ref][j][next(iter(range(len(neural_trials[ref][j]))))]) == 2:
            time_ref = [
                neural_trials[ref][j][t][0] 
                for t in range(len(neural_trials[ref][j]))]
            
        trial_type = np.array([trial_type[i]
                   for i in range(len(inter_time))
                   if not np.isnan(time_state)[i] and not np.isnan(time_ref)[i]])
        opto = np.array([opto[i-1]
                   for i in range(len(inter_time))
                   if not np.isnan(time_state)[i] and not np.isnan(time_ref)[i]])
        
        outcome = np.array([outcome[i]
                       for i in range(len(inter_time))
                       if not np.isnan(time_state)[i] and not np.isnan(time_ref)[i]])
        zero_state = [np.argmin(np.abs(inter_time[i] - time_state[i]))
                      for i in range(len(inter_time))
                      if not np.isnan(time_state)[i]]
        
        comp_time = [np.abs(time_state[i] - time_ref[i])
                      for i in range(len(inter_time))
                      if not np.isnan(time_state)[i] and not np.isnan(time_ref)[i]]
        
        align_data = [inter_pos[i] - inter_pos[i][0]
                      for i in range(len(inter_pos))
                      if not np.isnan(time_state)[i]]
        align_time = [inter_time[i]
                      for i in range(len(inter_time))
                      if not np.isnan(time_state)[i]]
        align_time = [align_time[i] - align_time[i][zero_state[i]]
                      for i in range(len(align_time))]
        if len(align_data) > 0:
            align_data, align_time = pad_seq(align_data, align_time)
            align_data = np.array(align_data)
        else:
            align_data = np.array([[np.nan]])
            align_time = np.array([np.nan])
            trial_type = np.array([np.nan])
            outcome = np.array([np.nan])
            opto = np.array([np.nan])
            comp_time = np.array([np.nan])
    
        session_align_data.append(align_data)
        session_align_time.append(align_time)
        session_trial_type.append(trial_type)
        session_outcome.append(outcome)
        session_opto.append(opto)
        session_comp_time.append(comp_time)
        
    return [session_align_data, session_align_time, session_trial_type, session_outcome, session_opto, session_comp_time]

def plot_beh(axs, align_data, align_time, trial_type, outcome, state, color_tag):
    
    trajectory_mean = np.mean(align_data , axis = 0)
    trajectory_sem = np.std(align_data , axis = 0)/np.sqrt(len(align_data))
    axs.fill_between(align_time/1000 ,trajectory_mean-trajectory_sem , trajectory_mean+trajectory_sem, color = color_tag , alpha = 0.1)
   
    axs.plot(align_time/1000 , trajectory_mean , color = color_tag)
    axs.axvline(x = 0, color = 'gray', linestyle='--')
    #axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('joystick deflection (deg)')
    #axs.set_xlim([-2 , 1.5])
    axs.set_title('aligned on '+ state)
def plot_beh_dev(axs, align_data, align_time, trial_type, outcome, state, condition, tag, color_tag):
    indx = np.where(condition == tag)[0]
    trajectory_mean = np.mean(align_data[indx][:] , axis = 0)
    trajectory_sem = np.std(align_data[indx][:] , axis = 0)/np.sqrt(len(align_data[indx]))
    axs.fill_between(align_time/1000 ,trajectory_mean-trajectory_sem , trajectory_mean+trajectory_sem, color = color_tag , alpha = 0.1)
   
    axs.plot(align_time/1000 , trajectory_mean , color = color_tag)
    axs.axvline(x = 0, color = 'gray', linestyle='--')
    #axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('joystick deflection (deg)')
    #axs.set_xlim([-2 , 1.5])
    axs.set_title('aligned on '+ state)
    
def plot_time_dev(axs, comp_time, trial_type, outcome, state, condition, tag, color_tag , num, i, title):
    indx = np.where(condition == tag)[0]
    #print(comp_time[[1 ,2]])
    comp_time_temp = [comp_time[i] for i in indx]
    trajectory_mean = np.mean(comp_time_temp)
    trajectory_sem = np.std(comp_time_temp)/np.sqrt(len(comp_time_temp))
    #axs.fill_between(align_time/1000 ,trajectory_mean-trajectory_sem , trajectory_mean+trajectory_sem, color = color_tag , alpha = 0.1)
    axs.scatter(num+i/50 , trajectory_mean/1000, color = color_tag, s = 5)
    #axs.plot(align_time/1000 , trajectory_mean , color = color_tag)
    axs.errorbar(num+i/50, trajectory_mean/1000, yerr=trajectory_sem/1000 , color = color_tag)
    #axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    #axs.set_xlabel('Time (s)')
    axs.set_ylabel('Time (s)')
    #axs.set_xlim([-2 , 1.5])
    axs.set_title(title)
    
def plot_time_dev_all(axs, comp_time, trial_type, outcome, state, condition, tag, color_tag , num, i, title):
    indx = np.where(condition == tag)[0]
    #print(comp_time[[1 ,2]])
    comp_time_temp = [comp_time[i] for i in indx]
    trajectory_mean = np.mean(comp_time_temp)
    trajectory_sem = np.std(comp_time_temp)/np.sqrt(len(comp_time_temp))
    #axs.fill_between(align_time/1000 ,trajectory_mean-trajectory_sem , trajectory_mean+trajectory_sem, color = color_tag , alpha = 0.1)
    axs.scatter(num+i/60 , trajectory_mean/1000, color = color_tag, s = 5)
    #axs.plot(align_time/1000 , trajectory_mean , color = color_tag)
    axs.errorbar(num+i/60, trajectory_mean/1000, yerr=trajectory_sem/1000 , color = color_tag)
    #axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    #axs.set_xlabel('Time (s)')
    axs.set_ylabel('Time (s)')
    #axs.set_xlim([-2 , 1.5])
    axs.set_title(title)
    return num+i/60, trajectory_mean/1000, color_tag
    
def run(axs,axs1, axs2, push_data):
    all_states = ['vis1' , 'vis2' , 'push1' , 'push2' , 'retract1' , 'wait2' , 'reward' , 'early2ndpush']
    num = 0
    
    for state in all_states:
        print(state)
        [session_align_data, session_align_time, session_trial_type, session_outcome, session_opto] = get_js_pos(push_data, state)
        cmap = plt.cm.inferno
        norm = plt.Normalize(vmin=0, vmax=len(push_data['outcome'])+1)
        
        for i in range(len(session_align_data)):
            color_tag = cmap(norm(len(push_data['outcome'])+1-i))
            plot_beh(axs[num], session_align_data[i], session_align_time[i], session_trial_type[i], session_outcome[i], state, color_tag)
            condition = session_trial_type[i]
            plot_beh_dev(axs1[num], session_align_data[i], session_align_time[i], session_trial_type[i], session_outcome[i], state, condition, 1, color_tag)
            plot_beh_dev(axs2[num], session_align_data[i], session_align_time[i], session_trial_type[i], session_outcome[i], state, condition, 2, color_tag)
            
        
        num = num + 1
def run_distict(axs,axs1, axs2,axs3, push_data):
    all_states = ['vis1' , 'vis2' , 'push1' , 'push2' , 'retract1' , 'wait2' , 'reward' , 'early2ndpush']
    num = 0
    
    for state in all_states:
        print(state)
        [session_align_data, session_align_time, session_trial_type, session_outcome, session_opto] = get_js_pos(push_data, state)
        cmap = plt.cm.inferno
        norm = plt.Normalize(vmin=0, vmax=len(push_data['outcome'])+1)
        
        for i in range(len(session_align_data)):
            color_tag = cmap(norm(len(push_data['outcome'])+1-i))
            #plot_beh(axs[num], session_align_data[i], session_align_time[i], session_trial_type[i], session_outcome[i], state, color_tag)
            condition1 = session_trial_type[i]
            condition2 = session_opto[i]
            condition = condition1
            for j in range(len(condition)):
                if condition1[j] == 1 and condition2[j] == 0:
                    condition[j] =0 
                elif condition1[j] == 1 and condition2[j] == 1:
                    condition[j] =1 
                elif condition1[j] == 2 and condition2[j] == 0:
                    condition[j] =2 
                elif condition1[j] == 2 and condition2[j] == 1:
                    condition[j] =3 
            plot_beh_dev(axs[num], session_align_data[i], session_align_time[i], session_trial_type[i], session_outcome[i], state, condition, 0, color_tag)
            plot_beh_dev(axs1[num], session_align_data[i], session_align_time[i], session_trial_type[i], session_outcome[i], state, condition, 1, color_tag)
            plot_beh_dev(axs2[num], session_align_data[i], session_align_time[i], session_trial_type[i], session_outcome[i], state, condition, 2, color_tag)
            plot_beh_dev(axs3[num], session_align_data[i], session_align_time[i], session_trial_type[i], session_outcome[i], state, condition, 3, color_tag)
            
        
        num = num + 1
        
def run_distict_time(axs,axs1, axs2,axs3, push_data):
    all_states = ['push1'  , 'push2' , 'push2' , 'push2' , 'push2']
    all_ref = ['vis1'  , 'push1' , 'retract1', 'vis2', 'wait2']
    num = 0
    
    for (state, ref) in zip(all_states, all_ref):
        print(ref , ' to ', state)
        [session_align_data, session_align_time, session_trial_type, session_outcome, session_opto, session_comp_time] = get_js_average_time(push_data, state, ref)
        cmap = plt.cm.inferno
        norm = plt.Normalize(vmin=0, vmax=len(push_data['outcome'])+1)
        
        for i in range(len(session_align_data)):
            color_tag = cmap(norm(len(push_data['outcome'])+1-i))
            #plot_beh(axs[num], session_align_data[i], session_align_time[i], session_trial_type[i], session_outcome[i], state, color_tag)
            condition1 = session_trial_type[i]
            condition2 = session_opto[i]
            condition = condition1
            for j in range(len(condition)):
                if condition1[j] == 1 and condition2[j] == 0:
                    condition[j] =0 
                elif condition1[j] == 1 and condition2[j] == 1:
                    condition[j] =1 
                elif condition1[j] == 2 and condition2[j] == 0:
                    condition[j] =2 
                elif condition1[j] == 2 and condition2[j] == 1:
                    condition[j] =3 
            plot_time_dev(axs, session_comp_time[i],  session_trial_type[i], session_outcome[i], state, condition, 0, color_tag, num, i, 'control short')
            plot_time_dev(axs1, session_comp_time[i], session_trial_type[i], session_outcome[i], state, condition, 1, color_tag, num, i, 'opto short')
            plot_time_dev(axs2, session_comp_time[i],  session_trial_type[i], session_outcome[i], state, condition, 2, color_tag, num, i,'control long')
            plot_time_dev(axs3, session_comp_time[i],  session_trial_type[i], session_outcome[i], state, condition, 3, color_tag, num, i, 'opto long')
            
        num = num + 1
    tick_index = [0.1 , 1.1 , 2.1, 3.1, 4.1]
    dates_label = ['vis1-push1', 'push1-push2' , 'retract1-push2' , 'vis2-push2' , 'wait2-push2']
    
    axs.set_xticks(np.sort(tick_index))
    axs.set_xticklabels(dates_label, rotation='vertical')
    axs1.set_xticks(np.sort(tick_index))
    axs1.set_xticklabels(dates_label, rotation='vertical')
    axs2.set_xticks(np.sort(tick_index))
    axs2.set_xticklabels(dates_label, rotation='vertical')
    axs3.set_xticks(np.sort(tick_index))
    axs3.set_xticklabels(dates_label, rotation='vertical')
    axs.axhline(y = 1, color = 'gray', linestyle='--')
    axs.axhline(y = 0.25, color = 'gray', linestyle='--')
    axs1.axhline(y = 1, color = 'gray', linestyle='--')
    axs1.axhline(y = 0.25, color = 'gray', linestyle='--')
    axs2.axhline(y = 1, color = 'gray', linestyle='--')
    axs2.axhline(y = 0.25, color = 'gray', linestyle='--')
    axs3.axhline(y = 1, color = 'gray', linestyle='--')
    axs3.axhline(y = 0.25, color = 'gray', linestyle='--')

def run_distict_time_all(axs, push_data):
    all_states = ['push1'  , 'push2' , 'push2' , 'push2' , 'push2']
    all_ref = ['vis1'  , 'push1' , 'retract1', 'vis2', 'wait2']
    num = 0
    dates_label = ['vis1-push1', 'push1-push2' , 'retract1-push2' , 'vis2-push2' , 'wait2-push2']
    for (state, ref) in zip(all_states, all_ref):
        print(ref , ' to ', state)
        [session_align_data, session_align_time, session_trial_type, session_outcome, session_opto, session_comp_time] = get_js_average_time(push_data, state, ref)
        cmap = plt.cm.inferno
        norm = plt.Normalize(vmin=0, vmax=len(push_data['outcome'])+1)
        
        for i in range(len(session_align_data)):
            color_tag = cmap(norm(len(push_data['outcome'])+1-i))
            #plot_beh(axs[num], session_align_data[i], session_align_time[i], session_trial_type[i], session_outcome[i], state, color_tag)
            condition1 = session_trial_type[i]
            condition2 = session_opto[i]
            condition = condition1
            for j in range(len(condition)):
                if condition1[j] == 1 and condition2[j] == 0:
                    condition[j] =0 
                elif condition1[j] == 1 and condition2[j] == 1:
                    condition[j] =1 
                elif condition1[j] == 2 and condition2[j] == 0:
                    condition[j] =2 
                elif condition1[j] == 2 and condition2[j] == 1:
                    condition[j] =3 
            x , y , c = plot_time_dev_all(axs[num], session_comp_time[i],  session_trial_type[i], session_outcome[i], state, condition, 0, color_tag, 0, i, dates_label[num])
            #plot_time_dev(axs1, session_comp_time[i], session_trial_type[i], session_outcome[i], state, condition, 1, color_tag, num, i, 'opto short')
            x1, y1 , c= plot_time_dev_all(axs[num], session_comp_time[i],  session_trial_type[i], session_outcome[i], state, condition, 2, color_tag, 1, i,dates_label[num])
            axs[num].plot([x,x1],[y,y1],color = c)
            #plot_time_dev(axs3, session_comp_time[i],  session_trial_type[i], session_outcome[i], state, condition, 3, color_tag, num, i, 'opto long')
            tick_index = [0.4, 1.4]
            axs[num].set_xticks(np.sort(tick_index))
            axs[num].set_xticklabels(['short', 'long'], rotation='vertical')
            
        num = num + 1
    #tick_index = [0.1 , 1.1 , 2.1, 3.1, 4.1]
    
    
    #axs.set_xticks(np.sort(tick_index))
    # axs.set_xticklabels(dates_label, rotation='vertical')
    # axs1.set_xticks(np.sort(tick_index))
    # axs1.set_xticklabels(dates_label, rotation='vertical')
    # axs2.set_xticks(np.sort(tick_index))
    # axs2.set_xticklabels(dates_label, rotation='vertical')
    # axs3.set_xticks(np.sort(tick_index))
    # axs3.set_xticklabels(dates_label, rotation='vertical')
    # axs.axhline(y = 1, color = 'gray', linestyle='--')
    # axs.axhline(y = 0.25, color = 'gray', linestyle='--')
    # axs1.axhline(y = 1, color = 'gray', linestyle='--')
    # axs1.axhline(y = 0.25, color = 'gray', linestyle='--')
    # axs2.axhline(y = 1, color = 'gray', linestyle='--')
    # axs2.axhline(y = 0.25, color = 'gray', linestyle='--')
    # axs3.axhline(y = 1, color = 'gray', linestyle='--')
    # axs3.axhline(y = 0.25, color = 'gray', linestyle='--')