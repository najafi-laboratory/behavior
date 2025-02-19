import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem


# bin the data with timestamps.

def get_bin_stat(decision, isi='post'):
    bin_size=50
    least_trials=5
    # set bins across isi range
    # short ISI: [50, 400, 750]ms.  associated with left lick
    # long ISI: [750, 1100, 1450]ms.  associated with right lick
    bins = np.arange(0, 1500 + bin_size, bin_size)
    bins = bins - bin_size / 2
    if isi=='pre':
        row = 4
    if isi=='post':
        row = 5
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
    # for j in range(len(chemo_labels)):
    #     if chemo_labels[j] == 1:
    #         jitter_flag[0 , all_trials:all_trials+len(outcomes[j])] = 2*np.ones(len(outcomes[j]))
    #     all_trials += len(outcomes[j])
    isi_pre_emp = subject_session_data['isi_pre_emp']
    isi_pre_emp = np.concatenate(isi_pre_emp).reshape(1,-1)
    isi_post_emp = subject_session_data['isi_post_emp']
    isi_post_emp = np.concatenate(isi_post_emp).reshape(1,-1)
    decision = np.concatenate([decision, jitter_flag, isi_pre_emp, isi_post_emp], axis=0)
    non_nan = (1-np.isnan(np.sum(decision, axis=0))).astype('bool')
    decision = decision[:,non_nan]
    # row 0: time.
    # row 1: direction.
    # row 2: correctness.
    # row 3: jitter flag.
    # row 4: pre pert isi.
    # row 5: post pert isi.
    decision_fix, decision_jitter, decision_chemo, decision_opto= separate_fix_jitter(decision)
    return decision_fix, decision_jitter, decision_chemo, decision_opto


def run(ax, subject_session_data, start_from='std'):

    subject_session_data_copy = subject_session_data.copy()
    
    if not start_from=='std':
        start_date = subject_session_data[start_from]
        dates = subject_session_data['dates']
        if start_date in dates:
            start_idx = dates.index(start_date)
        else:
            return
            
        for key in subject_session_data_copy.keys():
            # print(key)
            if isinstance(subject_session_data_copy[key], list) and len(subject_session_data_copy[key]) == len(dates):
                subject_session_data_copy[key] = subject_session_data_copy[key][start_idx:]  
    
    decision_fix, decision_jitter, decision_chemo, decision_opto = get_decision(subject_session_data_copy)
    bin_mean_fix, bin_sem_fix, bin_isi_fix = get_bin_stat(decision_fix)
    bin_mean_jitter, bin_sem_jitter, bin_isi_jitter = get_bin_stat(decision_jitter)
    bin_mean_chemo, bin_sem_chemo, bin_isi_chemo = get_bin_stat(decision_chemo)
    bin_mean_opto, bin_sem_opto, bin_isi_opto = get_bin_stat(decision_opto)
    ax.plot(
        bin_isi_fix,
        bin_mean_fix,
        color='indigo', marker='.', label='fix', markersize=4)
    ax.fill_between(
        bin_isi_fix,
        bin_mean_fix - bin_sem_fix,
        bin_mean_fix + bin_sem_fix,
        color='violet', alpha=0.2)
    ax.vlines(
        700, 0.0, 1.0,
        linestyle='--', color='mediumseagreen',
        label='Category Boundary')      
    # ax.plot(
    #     bin_isi_jitter,
    #     bin_mean_jitter,
    #     color='limegreen', marker='.', label='jitter', markersize=4)
    # ax.fill_between(
    #     bin_isi_jitter,
    #     bin_mean_jitter - bin_sem_jitter,
    #     bin_mean_jitter + bin_sem_jitter,
    #     color='limegreen', alpha=0.2)
    # ax.plot(
    #     bin_isi_chemo,
    #     bin_mean_chemo,
    #     color='red', marker='.', label='chemo', markersize=4)
    # ax.fill_between(
    #     bin_isi_chemo,
    #     bin_mean_chemo - bin_sem_chemo,
    #     bin_mean_chemo + bin_sem_chemo,
    #     color='red', alpha=0.2)
    # ax.plot(
    #     bin_isi_opto,
    #     bin_mean_opto,
    #     color='dodgerblue', marker='.', label='opto', markersize=4)
    # ax.fill_between(
    #     bin_isi_opto,
    #     bin_mean_opto - bin_sem_opto,
    #     bin_mean_opto + bin_sem_opto,
    #     color='dodgerblue', alpha=0.2)
    ax.hlines(0.5, 0.0, 1500, linestyle='--', color='grey')
    # ax.vlines(500, 0.0, 1.0, linestyle=':', color='grey')
    ax.tick_params(tick1On=False)
    ax.tick_params(axis='x', rotation=45)    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([-50,1600])
    ax.set_ylim([-0.05,1.05])
    # ax.set_xticks(np.arange(6)*200)
    ax.set_xticks(np.arange(11)*150)
    ax.set_yticks(np.arange(5)*0.25)
    ax.set_xlabel('post perturbation isi')
    ax.set_ylabel('prob. of choosing the right side (mean$\pm$sem)')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    if start_from=='start_date':
        ax.set_title('average psychometric function from ' + start_date)
    elif start_from=='non_naive':
        ax.set_title('average psychometric function non-naive')
    else:
        ax.set_title('average psychometric function')