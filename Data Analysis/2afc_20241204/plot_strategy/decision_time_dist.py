#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

def get_bin_stat(decision, isi='post'):
    bin_size=1
    row = 4
    least_trials=5
    if len(decision[row,:]) > 0:
        last_number = np.nanmax(decision[row,:])
    else:
        last_number = 1
    bins = np.arange(0, last_number + bin_size, bin_size)
    bins = bins - bin_size / 2
    bin_indices = np.digitize(decision[row,:], bins) - 1
    bin_mean = []
    bin_sem = []
    for i in range(len(bins)-1):
        direction = decision[1, bin_indices == i].copy()
        m = np.mean(direction) if len(direction) > least_trials else np.nan
        s = sem(direction) if len(direction) > least_trials else np.nan
        bin_mean.append(m)
        bin_sem.append(s)
    bin_mean = np.array(bin_mean)
    bin_sem  = np.array(bin_sem)
    bin_isi  = bins[:-1] + (bins[1]-bins[0]) / 2
    non_nan  = (1-np.isnan(bin_mean)).astype('bool')
    bin_mean = bin_mean[non_nan]
    bin_sem  = bin_sem[non_nan]
    bin_isi  = bin_isi[non_nan]
    return bin_mean, bin_sem, bin_isi

def plot_curves(ax, subject,jitter_session, dates, decision,stim_start, post_isi_mean,number_flash,j,r, k ,n_jitter,n_control , n_chemo, chemo_labels):
    stim_start = np.concatenate(stim_start).reshape(-1)
    decision_time = decision[0 , :] - stim_start
    decision = np.concatenate([decision, post_isi_mean,number_flash], axis=0)
    non_nan = (1-np.isnan(np.sum(decision, axis=0))).astype('bool')
    extra = 1
    decision = decision[:,non_nan]
    bin_mean, bin_sem, bin_isi = get_bin_stat(decision)
    c1 = (n_control+extra-r)/(n_control+extra)
    c2 = (n_jitter+extra-j)/(n_jitter+extra)
    c3 = (n_chemo+extra-k)/(n_chemo+extra)
    
    
    num_bin = 20
    bin_lims = np.linspace(1300,4500,num_bin+1)
    bin_centers = 0.5*(bin_lims[:-1]+bin_lims[1:])

    ##computing the histograms
    hist1, _ = np.histogram(decision_time, bins=bin_lims)
    

    ##normalizing
    hist1 = hist1/np.max(hist1)
    
    
    if chemo_labels == 1:
        c = [1 , c3 , c3]
    elif jitter_session == 1:
        c = [c2 , 1 , c2]
    else:
        c = [c1 , c1 , 1]
#     ax.hist(
#         decision_time, bins = 50 ,histtype='step', stacked=True, fill=False, density = True,
#         color = c, label = dates[4:])
    ax.plot(bin_centers , hist1 , color = c, label = dates[4:])
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([1200 , 4500])
    #ax.set_ylim([0 , 0.04])
    ax.set_xlabel('desicion time')
    ax.set_ylabel('fraction')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    ax.set_title('decision time distribution')

def run(axs, subject_session_data):
    max_sessions = 6
    subject = subject_session_data['subject']
    dates = subject_session_data['dates']
    chemo_labels = subject_session_data['Chemo']
    
    decision = subject_session_data['decision']
    decision = [np.concatenate(d, axis=1) for d in decision]
    stim_start = subject_session_data['stim_start']
    isi_post_emp = subject_session_data['isi_post_emp']
    isi_post_emp = [np.array(isi).reshape(1,-1) for isi in isi_post_emp]
    number_flash = subject_session_data['number_flash']
    number_flash = [np.array(nf).reshape(1,-1) for nf in number_flash]
    
    jitter_flag = subject_session_data['jitter_flag']
    if len(axs)*max_sessions <= len(dates):
        dates = dates[-len(axs)*max_sessions:]
        stim_start = stim_start[-len(axs)*max_sessions:]
        decision = decision[-len(axs)*max_sessions:]
        isi_post_emp = isi_post_emp[-len(axs)*max_sessions:]
        jitter_flag = jitter_flag[-len(axs)*max_sessions:]
        chemo_labels = chemo_labels[-len(axs)*max_sessions:]
        number_flash = number_flash[-len(axs)*max_sessions:]
    jitter_session = np.array([np.sum(j) for j in jitter_flag])
    jitter_session[jitter_session!=0] = 1
    r = 0
    j = 0
    k = 0
    n_jitter = []
    n_control = []
    n_chemo = []
    for i in range(len(dates)//max_sessions+1):
        a = jitter_session[i*max_sessions:min(len(dates) , (i+1)*max_sessions)]
        b = chemo_labels[i*max_sessions:min(len(dates) , (i+1)*max_sessions)]
        n_chemo1 = np.count_nonzero(b)
        n_jitter1 = np.count_nonzero(a)
        if len(a) - n_jitter1-n_chemo1 >-1:
            n_control1 = len(a) - n_jitter1-n_chemo1
        else:
            n_control1 = 0
        n_jitter.append(n_jitter1)
        n_chemo.append(n_chemo1)
        n_control.append(n_control1)
    for i in range(len(dates)):
        if i%max_sessions != 0:
            if jitter_session[i] == 1:
                j = j + 1
            elif chemo_labels[i] == 1:
                k = k + 1
            else:
                r = r + 1
        else:
            j = 0
            r = 0
            k = 0
            if jitter_session[i] == 1:
                j = j + 1
            elif chemo_labels[i] == 1:
                k = k + 1
            else:
                r = r + 1
        plot_curves(
            axs[i//max_sessions], subject,
            jitter_session[i], dates[i], decision[i],stim_start[i], isi_post_emp[i],number_flash[i], j, r, k ,n_jitter[i//max_sessions],n_control[i//max_sessions] , n_chemo[i//max_sessions] , chemo_labels[i])

