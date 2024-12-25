#!/usr/bin/env python
# coding: utf-8

# In[6]:


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
    session_id = np.arange(len(dates)) + 1
    jitter_flag = subject_session_data['jitter_flag']
    jitter_session = np.array([np.sum(j) for j in jitter_flag])
    jitter_session[jitter_session!=0] = 1
    # chemo_labels = subject_session_data['Chemo']
    chemo_labels = []
    choice = []
    choice_right = []

    width = 0.5
    
    for i in range(len(dates)):
        
        decision = subject_session_data['decision'][i]
        decision = np.concatenate(decision, axis=1)
        direction = decision[1 , :]
        left = sum(trial_type == 0 for trial_type in direction)
        right = sum(trial_type == 1 for trial_type in direction)
        choice.append(left/(left+right))
        choice_right.append(right/(left+right))
    
    ax.bar(
        session_id, choice_right,
        edgecolor='white',
        width=width,
        color='mediumpurple',
        label='Right choice')
        # bottom = choice,    
    ax.bar(
        session_id, choice,
        edgecolor='white',
        width=width,
        bottom = choice_right,        
        color='lightcoral',
        label='Left choice')

    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.hlines(0.5,0,len(dates)+1, linestyle='--' , color='silver' , lw = 0.5)
    #ax.yaxis.grid(True)
    ax.set_xlabel('training session')
    ax.set_ylabel('fraction of left choice')
    ax.set_ylim(0 , 1)
    ax.set_xticks(np.arange(len(dates))+1)
    ax.set_xticklabels(dates, rotation=45)
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    ax.set_title('left vs right choice fraction')
    ind = 0
    for xtick in ax.get_xticklabels():
        if jitter_session[ind] == 1:
            xtick.set_color('limegreen')
        # if chemo_labels[ind] == 1:
        #     xtick.set_color('red')
        ind = ind + 1

