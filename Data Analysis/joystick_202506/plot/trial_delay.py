# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 20:23:22 2024

@author: saminnaji3
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import random
from matplotlib.lines import Line2D
import math
from scipy.stats import ttest_ind

def block_start_end(trial_type , warmup_num):
    diff = np.diff(trial_type)
    
    short_start = np.where(diff == 255)[0] + 1
    long_start = np.where(diff == 1)[0] + 1
    short_end = np.where(diff == 1)[0] + 1
    long_end = np.where(diff == 255)[0] + 1
    
    if trial_type[0] == 1:
        short_start = np.insert(short_start,0,warmup_num)
    else:
        long_start = np.insert(long_start,0,warmup_num)
        
        
    if len(short_start) > len(short_end):
        short_end = np.insert(short_end,0,len(trial_type))
        
    elif len(long_start) > len(long_end) and len(long_end) > 0:
        long_end = np.insert(long_end,-1,len(trial_type))
    elif len(long_start) > len(long_end):
        long_end = np.insert(long_end,0,len(trial_type))
        
        
    return short_start , short_end , long_start , long_end

def realize(block_outcome , win):
    block_realize = np.nan
    for i in range(len(block_outcome)-win):
        if not np.sum(np.array(block_outcome[i:i + win]) == 'Reward') < win/2 :
            block_realize = i
            for j in range(win):
                if block_outcome[i+j] == 'Reward':
                    block_realize = i+j
                    break
            break
    return block_realize

def adaptation(block_delay , block_type):
    block_realize = np.nan
    for i in range(len(block_delay)-1):
        if block_type == 1:
            if np.nanmean(block_delay[:i]) >= np.nanmean(block_delay[i:]):
                block_realize = i
            elif i > 1:
                break
        else:
            if np.nanmean(block_delay[:i]) <= np.nanmean(block_delay[i:]):
                block_realize = i
            elif i > 0:
                break
    return block_realize

def delay_change(previous_delay, block_delay , block_type):
    block_realize = np.nan
    if len(previous_delay) < 10:
        refrence = np.nanmin(previous_delay)
    else:
        refrence = np.nanmin(previous_delay[-10:])
    #print(refrence)
    for i in range(1,len(block_delay)):
        if block_type == 1:
            if np.nanmean(block_delay[:i]) <= refrence*0.7:
                block_realize = i-1
                #print(block_type, np.nanmean(block_delay[:i]))
                break
            
        else:
            if np.nanmean(block_delay[:i]) >= refrence*1.3:
                block_realize = i-1
                #print(block_type, np.nanmean(block_delay[:i]))
                break
    
    return block_realize

def delay_ttest(previous_delay, block_delay , block_type):
    block_realize = np.nan
    if len(previous_delay) < 10:
        refrence = previous_delay
    else:
        refrence = previous_delay[-10:]
    for i in range(1,len(block_delay)):
        t_stat, p_value = ttest_ind(refrence, block_delay[:i], nan_policy='omit', equal_var=False)
        if p_value < 0.1:
            block_realize = i
            break
    
    return block_realize
        
def read_block(
        session_data, push_data
        ):
    
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_id = np.arange(len(outcomes)) + 1
    session_raw = session_data['raw']
    
    session_outcome = []
    session_lowerband = []
    session_upperband = []
    session_type = []
    session_numtrials = []
    session_delay = []
    session_start = []
    session_end = []
    session_numblocks = []
    session_short_id = []
    session_long_id =[]
    session_QC_index = []
    session_block_realize = []
    session_block_adaptation = []
    session_opto = []
    session_opto_block = []
    session_delay_all = []
    session_delay_change = []
    session_delay_ttest = []
    
    for i in range(0 , len(session_id)):
        outcome = outcomes[i]
        delay = np.array(session_data['session_comp_delay'][i])
        print('processing delays for ' + dates[i])
        opto = session_data['session_opto_tag'][i]
        trial_types = session_raw[i]['TrialTypes'][0:len(outcome)]
        warmup_num = session_raw[i]['IsWarmupTrial'].count(1)
        predelay2 = session_raw[i]['PrePress2Delay']
        trial_setting = session_raw[i]['TrialSettings']
        
        trial_outcome = []
        trial_lowerband = []
        trial_upperband = []
        trial_type = []
        trial_numtrials = []
        trial_delay = []
        trial_start = []
        trial_end = []
        trial_short_id = []
        trial_long_id =[]
        trial_QC_index = []
        trial_block_realize = []
        trial_delay_change = []
        trial_delay_ttest = []
        trial_block_adaptation = []
        trial_opto = []
        trial_opto_block = np.zeros(len(outcomes[i]))
        
        short_start , short_end , long_start , long_end = block_start_end(trial_types , warmup_num)
        
        trial_start = np.sort(np.append(short_start , long_start))
        trial_end = np.sort(np.append(short_end , long_end))
        trial_num_blocks = len(trial_start)
        win = 6
        
        #print(session_data['dates'][i])
        for j in range(0 , trial_num_blocks):
            trial_opto.append(int(any(opto[trial_start[j] : trial_end[j]])))
            trial_opto_block[trial_start[j] : trial_end[j]] = int(any(opto[trial_start[j] : trial_end[j]]))
            trial_delay.append(delay[trial_start[j] : trial_end[j]])
            trial_outcome.append(outcome[trial_start[j] : trial_end[j]])
            #print(trial_start[j], len(trial_types))
            trial_type.append(trial_types[trial_start[j]])
            trial_block_realize.append(realize(outcome[trial_start[j] : trial_end[j]] , win))
            if j == 0:
                trial_delay_change.append(np.nan)
                trial_delay_ttest.append(np.nan)
            else:
                trial_delay_change.append(delay_change(trial_delay[-2], trial_delay[-1] , trial_type[-1]))
                trial_delay_ttest.append(delay_ttest(trial_delay[-2], trial_delay[-1] , trial_type[-1]))
            trial_block_adaptation.append(adaptation(trial_delay[-1] , trial_type[-1]))
            if trial_types[trial_start[j]] == 1:
                trial_short_id.append(j)
                if 'Press2WindowShort_s' in trial_setting[trial_start[j]]['GUI'].keys():
                    window = np.array([trial_setting[k]['GUI']['Press2WindowShort_s'] for k in range(trial_start[j] , trial_end[j])])
            else:
                trial_long_id.append(j)
                if 'Press2WindowLong_s' in trial_setting[trial_start[j]]['GUI'].keys():
                    window = np.array([trial_setting[k]['GUI']['Press2WindowLong_s'] for k in range(trial_start[j] , trial_end[j])])
            trial_numtrials.append(trial_end[j]-trial_start[j])
            if 'Press2Window_s' in trial_setting[trial_start[j]]['GUI'].keys():
                window = np.array([trial_setting[k]['GUI']['Press2Window_s'] for k in range(trial_start[j] , trial_end[j])])
            #window = np.array([trial_setting[k]['GUI']['Press2Window_s'] for k in range(trial_start[j] , trial_end[j])])
            
            if session_data['isSelfTimedMode'][i][0] == 1:
                prepress2delay = np.array(predelay2[trial_start[j] : trial_end[j]])
                trial_lowerband.append(prepress2delay)
                trial_upperband.append(prepress2delay + window)
            else:
                prepress2delay = np.array(predelay2[trial_start[j] : trial_end[j]]) + 0.15
                trial_lowerband.append(prepress2delay)
                trial_upperband.append(prepress2delay + window + 0.15)
            
            total_rewarded = 0
            both_rewarded = 0
            if j == 0:
                trial_QC_index.append(np.nan)
            else:
                for k in range(len(trial_outcome[-1])):
                    if trial_outcome[-1][k] == 'Reward':
                        total_rewarded = total_rewarded + 1 
                        if trial_delay[-1][k] > trial_lowerband[-2][-1] and trial_delay[-1][k] < trial_upperband[-2][-1]:
                            both_rewarded = both_rewarded + 1
                if total_rewarded > 0:
                    trial_QC_index.append(1-both_rewarded/total_rewarded)
                else:
                    trial_QC_index.append(np.nan)
        
        session_outcome.append(trial_outcome)
        session_lowerband.append(trial_lowerband)
        session_upperband.append(trial_upperband)
        session_type.append(trial_type)
        session_numtrials.append(trial_numtrials)
        session_delay.append(trial_delay)
        session_start.append(trial_start)
        session_end.append(trial_end)
        session_numblocks.append(trial_num_blocks)
        session_short_id.append(trial_short_id)
        session_long_id.append(trial_long_id)
        session_QC_index.append(trial_QC_index)
        session_block_realize.append(trial_block_realize)
        session_delay_change.append(trial_delay_change)
        session_delay_ttest.append(trial_delay_ttest)
        session_block_adaptation.append(trial_block_adaptation)
        session_opto.append(trial_opto)
        session_opto_block.append(trial_opto_block)
        session_delay_all.append(delay)
    
    block_data = {
            'NumBlocks': session_numblocks ,
            'short_id': session_short_id ,
            'long_id': session_long_id , 
            'start': session_start ,
            'end': session_end ,
            'delay': session_delay ,
            'delay_all': session_delay_all,
            'outcome': session_outcome ,
            'NumTrials': session_numtrials ,
            'Type': session_type ,
            'LowerBand': session_lowerband ,
            'UpperBand': session_upperband ,
            'block_realize': session_block_realize,
            'block_adaptation': session_block_adaptation,
            'delay_change': session_delay_change,
            'delay_ttest': session_delay_ttest,
            'QC_index': session_QC_index,
            'Opto' : session_opto,
            'block_opto' : session_opto_block,
            'date': dates ,
            'subject': subject}
    return block_data 