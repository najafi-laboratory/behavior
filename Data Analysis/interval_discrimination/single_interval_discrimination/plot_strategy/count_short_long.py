#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem


# bin the data with timestamps.

def get_bin_stat(decision):
    bin_size=1
    row = 6
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


def separate_fix_jitter(decision):
    decision_fix = decision[:,decision[3,:]==0]
    decision_jitter = decision[:,decision[3,:]==1]
    decision_chemo = decision[:,decision[3,:]==2]
    decision_opto = decision[:,decision[3,:]==3]
    return decision_fix, decision_jitter, decision_chemo, decision_opto

def separate_short_long(decision):
    decision_long = decision[:,decision[5,:]>500]
    decision_short = decision[:,decision[5,:]<500]
    return decision_long, decision_short

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
    decision = np.concatenate([decision, jitter_flag, pre_isi, post_isi,number_flash], axis=0)
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
    decision_fix_l , decision_fix_s = separate_short_long(decision_fix)
    decision_jitter_l , decision_jitter_s = separate_short_long(decision_jitter)
    decision_chemo_l , decision_chemo_s = separate_short_long(decision_chemo)
    decision_opto_l , decision_opto_s = separate_short_long(decision_opto)
    return decision_fix_l, decision_jitter_l, decision_chemo_l , decision_opto_l, decision_fix_s, decision_jitter_s, decision_chemo_s, decision_opto_s


def run(ax1 , ax2, subject_session_data):
    decision_fix_l, decision_jitter_l, decision_chemo_l ,decision_opto_l, decision_fix_s, decision_jitter_s, decision_chemo_s, decision_opto_s = get_decision(subject_session_data)
    bin_mean_fix_l, bin_sem_fix_l, bin_isi_fix_l = get_bin_stat(decision_fix_l)
    bin_mean_jitter_l, bin_sem_jitter_l, bin_isi_jitter_l = get_bin_stat(decision_jitter_l)
    bin_mean_chemo_l, bin_sem_chemo_l, bin_isi_chemo_l = get_bin_stat(decision_chemo_l)
    bin_mean_opto_l, bin_sem_opto_l, bin_isi_opto_l = get_bin_stat(decision_opto_l)
    ax1.plot(
        bin_isi_fix_l,
        bin_mean_fix_l,
        color='black', marker='.', label='fix', markersize=4)
    ax1.fill_between(
        bin_isi_fix_l,
        bin_mean_fix_l - bin_sem_fix_l,
        bin_mean_fix_l + bin_sem_fix_l,
        color='black', alpha=0.2)
    ax1.plot(
        bin_isi_jitter_l,
        bin_mean_jitter_l,
        color='limegreen', marker='.', label='jitter', markersize=4)
    ax1.fill_between(
        bin_isi_jitter_l,
        bin_mean_jitter_l - bin_sem_jitter_l,
        bin_mean_jitter_l + bin_sem_jitter_l,
        color='limegreen', alpha=0.2)
    ax1.plot(
        bin_isi_chemo_l,
        bin_mean_chemo_l,
        color='red', marker='.', label='chemo', markersize=4)
    ax1.fill_between(
        bin_isi_chemo_l,
        bin_mean_chemo_l - bin_sem_chemo_l,
        bin_mean_chemo_l + bin_sem_chemo_l,
        color='red', alpha=0.2)
    ax1.plot(
        bin_isi_opto_l,
        bin_mean_opto_l,
        color='dodgerblue', marker='.', label='opto', markersize=4)
    ax1.fill_between(
        bin_isi_opto_l,
        bin_mean_opto_l - bin_sem_opto_l,
        bin_mean_opto_l + bin_sem_opto_l,
        color='dodgerblue', alpha=0.2)
    ax1.tick_params(tick1On=False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    if len(bin_mean_jitter_l) > 0:
        max_limit = max(bin_isi_jitter_l[-1] , bin_isi_jitter_l[-1])
    else:
        max_limit = np.max(bin_isi_fix_l)
    ax1.set_xlim([-0.05,max_limit+2])
    ax1.set_ylim([-0.05,1])
    ax1.set_xticks(np.arange(max_limit+2))
    ax1.set_yticks(np.arange(5)*0.25)
    ax1.set_xlabel('post perturbation flah count')
    ax1.set_ylabel('prob. of choosing the right side (mean$\pm$sem)')
    ax1.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    ax1.set_title('average right side licks for post per flash count (long)')
    ax1.hlines(0.5, -0.05, max_limit+2, linestyle=':', color='grey')
    
    bin_mean_fix_s, bin_sem_fix_s, bin_isi_fix_s = get_bin_stat(decision_fix_s)
    bin_mean_jitter_s, bin_sem_jitter_s, bin_isi_jitter_s = get_bin_stat(decision_jitter_s)
    bin_mean_chemo_s, bin_sem_chemo_s, bin_isi_chemo_s = get_bin_stat(decision_chemo_s)
    bin_mean_opto_s, bin_sem_opto_s, bin_isi_opto_s = get_bin_stat(decision_opto_s)
    ax2.plot(
        bin_isi_fix_s,
        bin_mean_fix_s,
        color='black', marker='.', label='fix', markersize=4)
    ax2.fill_between(
        bin_isi_fix_s,
        bin_mean_fix_s - bin_sem_fix_s,
        bin_mean_fix_s + bin_sem_fix_s,
        color='black', alpha=0.2)
    ax2.plot(
        bin_isi_jitter_s,
        bin_mean_jitter_s,
        color='limegreen', marker='.', label='jitter', markersize=4)
    ax2.fill_between(
        bin_isi_jitter_s,
        bin_mean_jitter_s - bin_sem_jitter_s,
        bin_mean_jitter_s + bin_sem_jitter_s,
        color='limegreen', alpha=0.2)
    ax2.plot(
        bin_isi_chemo_s,
        bin_mean_chemo_s,
        color='red', marker='.', label='chemo', markersize=4)
    ax2.fill_between(
        bin_isi_chemo_s,
        bin_mean_chemo_s - bin_sem_chemo_s,
        bin_mean_chemo_s + bin_sem_chemo_s,
        color='red', alpha=0.2)
    ax2.plot(
        bin_isi_opto_s,
        bin_mean_opto_s,
        color='dodgerblue', marker='.', label='opto', markersize=4)
    ax2.fill_between(
        bin_isi_opto_s,
        bin_mean_opto_s - bin_sem_opto_s,
        bin_mean_opto_s + bin_sem_opto_s,
        color='dodgerblue', alpha=0.2)
    ax2.tick_params(tick1On=False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    if len(bin_mean_jitter_s) > 0:
        max_limit = max(bin_isi_jitter_s[-1] , bin_isi_jitter_s[-1])
    else:
        max_limit = np.max(bin_isi_fix_s)
    ax2.set_xlim([-0.05,max_limit+2])
    ax2.set_ylim([-0.05,1])
    ax2.set_xticks(np.arange(max_limit+2))
    ax2.set_yticks(np.arange(5)*0.25)
    ax2.set_xlabel('post perturbation flah count')
    ax2.set_ylabel('prob. of choosing the right side (mean$\pm$sem)')
    ax2.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    ax2.set_title('average right side licks for post per flash count (short)')
    ax2.hlines(0.5, -0.05, max_limit+2, linestyle=':', color='grey')

