#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem


def run(ax, subject_session_data):
    max_sessions = 20
    subject = subject_session_data['subject']
    dates = subject_session_data['dates']
    session_id = np.arange(len(dates)) + 0.75
    jitter_flag = subject_session_data['jitter_flag']
    jitter_session = np.array([np.sum(j) for j in jitter_flag])
    jitter_session[jitter_session!=0] = 1
    chemo_labels = subject_session_data['Chemo']
    #post_isi_type = subject_session_data['post_isi_type']
    percentage = []
    percentage_actual = []
    percentage_short = []
    percentage_actual_short = []
    width = 0.25
    
    for i in range(len(dates)):
        
        long = sum(trial_type > 0 for trial_type in subject_session_data['post_isi_type'][i])
        long_actual = sum(trial_type > 500 for trial_type in subject_session_data['post_isi'][i])
        percentage.append(long/len(subject_session_data['post_isi_type'][i]))
        #percentage_actual.append(long_actual/len(subject_session_data['post_isi'][i]))
        short = sum(trial_type < 1 for trial_type in subject_session_data['post_isi_type'][i])
        short_actual = sum(trial_type < 500 for trial_type in subject_session_data['post_isi'][i])
        percentage_short.append(short/len(subject_session_data['post_isi_type'][i]))
        #percentage_actual_short.append(short_actual/len(subject_session_data['post_isi'][i]))
        percentage_actual_short.append(short_actual/(short_actual+long_actual))
        percentage_actual.append(long_actual/(short_actual+long_actual))
    ax.bar(
        session_id, percentage,
        edgecolor='white',
        width=width,
        color='black',
        label='long ISI(emp)')
    ax.bar(
        session_id, percentage_short,
        edgecolor='white',
        width=width,
        bottom = percentage,
        color='gray',
        label='Short ISI(emp)')
    ax.bar(
        session_id+width, percentage_actual,
        edgecolor='white',
        width=width,
        color='darkblue',
        label='long ISI')
    ax.bar(
        session_id+width, percentage_actual_short,
        edgecolor='white',
        width=width,
        bottom = percentage_actual,
        color='darkblue',
        alpha = 0.3,
        label='Short ISI')
    ax.hlines(0.5,0,len(dates)+1, linestyle='--' , color='silver' , lw = 0.5)
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax.yaxis.grid(True)
    ax.set_xlabel('training session')
    ax.set_ylabel('fraction of trials')
    ax.set_ylim(0 , 1)
    ax.set_xticks(np.arange(len(dates))+1)
    ax.set_xticklabels(dates, rotation='vertical')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    ax.set_title('short vs long ISI percentage')
    ind = 0
    for xtick in ax.get_xticklabels():
        if jitter_session[ind] == 1:
            xtick.set_color('limegreen')
        if chemo_labels[ind] == 1:
            xtick.set_color('red')
        ind = ind + 1

