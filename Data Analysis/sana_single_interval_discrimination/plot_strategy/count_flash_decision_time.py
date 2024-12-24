#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import matplotlib.colors as colors
import matplotlib.cm as cmx


# bin the data with timestamps.

def get_bin_stat(decision , lick_number):
    
    bin_size=100
    least_trials=5
    bins = np.arange(0, 1000 + bin_size, bin_size)
    bins = bins - bin_size / 2
    decision = decision[:,decision[6,:]==lick_number]
    
    
    bin_size=250
    least_trials=3
    max_time = 5000
    bins = np.arange(0, max_time + bin_size, bin_size)
    bins = bins - bin_size / 2
    bin_indices = np.digitize(decision[0,:], bins) - 1
    bin_mean = []
    bin_sem = []
    for i in range(len(bins)-1):
        correctness = decision[2, bin_indices == i].copy()
        m = np.mean(correctness) if len(correctness) > least_trials else np.nan
        s = sem(correctness) if len(correctness) > least_trials else np.nan
        bin_mean.append(m)
        bin_sem.append(s)
    bin_mean = np.array(bin_mean)
    bin_sem  = np.array(bin_sem)
    bin_time = bins[:-1] + (bins[1]-bins[0]) / 2
    non_nan  = (1-np.isnan(bin_mean)).astype('bool')
    bin_mean = bin_mean[non_nan]
    bin_sem  = bin_sem[non_nan]
    bin_time = bin_time[non_nan]
    return bin_mean, bin_sem, bin_time


def separate_fix_jitter(decision):
    decision_fix = decision[:,decision[3,:]==0]
    decision_jitter = decision[:,decision[3,:]==1]
    decision_chemo = decision[:,decision[3,:]==2]
    decision_opto = decision[:,decision[3,:]==3]
    return decision_fix, decision_jitter, decision_chemo ,decision_opto


def get_decision(subject_session_data):
    decision = subject_session_data['decision']
    decision = [np.concatenate(d, axis=1) for d in decision]
    decision = np.concatenate(decision, axis=1)
    jitter_flag = subject_session_data['jitter_flag']
    jitter_flag = np.concatenate(jitter_flag).reshape(1,-1)
    opto_flag = subject_session_data['opto_flag']
    opto_flag = np.concatenate(opto_flag).reshape(1,-1)
    jitter_flag[0 , :] = jitter_flag[0 , :] + opto_flag[0 , :]*3
    outcomes = subject_session_data['outcomes']
    all_trials = 0
    chemo_labels = subject_session_data['Chemo']
    for j in range(len(chemo_labels)):
        if chemo_labels[j] == 1:
            jitter_flag[0 , all_trials:all_trials+len(outcomes[j])] = 2*np.ones(len(outcomes[j]))
        all_trials += len(outcomes[j])
    number_flash = subject_session_data['number_flash']
    number_flash = np.concatenate(number_flash).reshape(1,-1)
    pre_isi = subject_session_data['pre_isi']
    pre_isi = np.concatenate(pre_isi).reshape(1,-1)
    post_isi = subject_session_data['post_isi']
    post_isi = np.concatenate(post_isi).reshape(1,-1)
    post_isi_mean = subject_session_data['isi_post_emp']
    post_isi_mean = np.concatenate(post_isi_mean).reshape(1,-1)
    number_flash = subject_session_data['number_flash']
    number_flash = np.concatenate(number_flash).reshape(1,-1)
    stim_start = subject_session_data['stim_start']
    stim_start = np.concatenate(stim_start).reshape(-1)
    decision = np.concatenate([decision, jitter_flag, pre_isi, post_isi_mean, number_flash], axis=0)
    decision[0,:] -= stim_start
    non_nan = (1-np.isnan(np.sum(decision, axis=0))).astype('bool')
    decision = decision[:,non_nan]
    # row 0: time.
    # row 1: direction.
    # row 2: correctness.
    # row 3: jitter flag.
    # row 4: pre pert isi.
    # row 5: post pert isi.
    # row 6: number of flashes.
    decision_fix, decision_jitter, decision_chemo, decision_opto = separate_fix_jitter(decision)
    return decision_fix, decision_jitter, decision_chemo, decision_opto


def run(ax, ax1, ax2, ax3, subject_session_data):
    decision_fix, decision_jitter, decision_chemo, decision_opto = get_decision(subject_session_data)
    last_isi = 8
    jet = cm = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=last_isi)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    for i in range(last_isi+1):
        colorVal = scalarMap.to_rgba(i)
        
        decision_fix, decision_jitter, decision_chemo, decision_opto = get_decision(subject_session_data)
        bin_mean_fix, bin_sem_fix, bin_isi_fix = get_bin_stat(decision_fix , i+1)
        bin_mean_jitter, bin_sem_jitter, bin_isi_jitter = get_bin_stat(decision_jitter , i+1)
        bin_mean_chemo, bin_sem_chemo, bin_isi_chemo = get_bin_stat(decision_chemo , i+1)
        bin_mean_opto, bin_sem_opto, bin_isi_opto = get_bin_stat(decision_opto , i+1)
        ax.plot(
            bin_isi_fix,
            bin_mean_fix,
            color=colorVal, marker='.', label=str(i)+'flash', markersize=4)

        ax1.plot(
            bin_isi_jitter,
            bin_mean_jitter,
            color=colorVal, marker='.', label=str(i)+'flash', markersize=4)

        ax2.plot(
            bin_isi_chemo,
            bin_mean_chemo,
            color=colorVal, marker='.', label=str(i)+'flash', markersize=4)
        ax3.plot(
            bin_isi_opto,
            bin_mean_opto,
            color=colorVal, marker='.', label=str(i)+'flash', markersize=4)
    
    max_time = 5000 
    ax.hlines(
        0.5, 0.0, max_time,
        linestyle=':', color='grey')
    ax.vlines(
        1300, 0.0, 1.0,
        linestyle=':', color='mediumseagreen')
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([0,max_time])
    ax.set_ylim([-0.05,1])
    ax.set_yticks(np.arange(5)*0.25)
    ax.set_xlabel('decision time')
    ax.set_ylabel('prob. of choosing the correct side (mean$\pm$sem)')
    ax.set_title('average decision time curve, fix')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    
    ax1.tick_params(tick1On=False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_xlim([0,max_time])
    ax1.set_ylim([-0.05,1])
    ax1.set_yticks(np.arange(5)*0.25)
    ax1.set_xlabel('decision time')
    ax1.set_ylabel('prob. of choosing the correct side (mean$\pm$sem)')
    ax1.set_title('average decision time curve, jitter')
    ax1.hlines(
        0.5, 0.0, max_time,
        linestyle=':', color='grey')
    ax1.vlines(
        1300, 0.0, 1.0,
        linestyle=':', color='mediumseagreen')
    ax1.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    
    ax2.tick_params(tick1On=False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_xlim([0,max_time])
    ax2.set_ylim([-0.05,1])
    ax2.set_yticks(np.arange(5)*0.25)
    ax2.set_xlabel('decision time')
    ax2.set_ylabel('prob. of choosing the correct side (mean$\pm$sem)')
    ax2.set_title('average decision time curve, chemo')
    ax2.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    ax2.hlines(
        0.5, 0.0, max_time,
        linestyle=':', color='grey')
    ax2.vlines(
        1300, 0.0, 1.0,
        linestyle=':', color='mediumseagreen')
    
    
    
    ax3.tick_params(tick1On=False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.set_xlim([0,max_time])
    ax3.set_ylim([-0.05,1])
    ax3.set_yticks(np.arange(5)*0.25)
    ax3.set_xlabel('decision time')
    ax3.set_ylabel('prob. of choosing the correct side (mean$\pm$sem)')
    ax3.set_title('average decision time curve, opto')
    ax3.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    ax3.set_xlabel('post perturbation flah count')
    ax3.hlines(
        0.5, 0.0, max_time,
        linestyle=':', color='grey')
    ax3.vlines(
        1300, 0.0, 1.0,
        linestyle=':', color='mediumseagreen')
   

