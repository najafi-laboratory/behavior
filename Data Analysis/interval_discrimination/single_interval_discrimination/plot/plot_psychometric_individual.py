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


def get_decision(subject_session_data, session_num):
    decision = subject_session_data['decision'][session_num]
    # decision = [np.concatenate(d, axis=1) for d in decision]
    decision = np.concatenate(decision, axis=1)
    jitter_flag = subject_session_data['jitter_flag'][session_num]
    # jitter_flag = np.concatenate(jitter_flag).reshape(1,-1)
    jitter_flag = np.array(jitter_flag).reshape(1,-1)
    # opto_flag = subject_session_data['opto_flag']
    opto_flag = subject_session_data['opto_trial'][session_num]
    opto_flag = np.array(opto_flag).reshape(1,-1)
    jitter_flag[0 , :] = jitter_flag[0 , :] + opto_flag[0 , :]*3
    # jitter_flag = jitter_flag + opto_flag*3
    # jitter_flag = [j + o * 3 for j, o in zip(jitter_flag, opto_flag)]
    opto_side = subject_session_data['opto_side'][session_num]
    opto_side = np.array(opto_side).reshape(1,-1)
    outcomes = subject_session_data['outcomes'][session_num]
    all_trials = 0
    chemo_labels = subject_session_data['Chemo'][session_num]
    # for j in range(len(chemo_labels)):
    #     if chemo_labels[j] == 1:
    #         jitter_flag[0 , all_trials:all_trials+len(outcomes[j])] = 2*np.ones(len(outcomes[j]))
    #     all_trials += len(outcomes[j])
    isi_pre_emp = subject_session_data['isi_pre_emp'][session_num]
    # isi_pre_emp = np.concatenate(isi_pre_emp).reshape(1,-1)
    isi_pre_emp = np.array(isi_pre_emp).reshape(1,-1)
    
    isi_post_emp = subject_session_data['isi_post_emp'][session_num]
    isi_post_emp = np.array(isi_post_emp).reshape(1,-1)
    # isi_post_emp = np.concatenate(isi_post_emp).reshape(1,-1)
    decision = np.concatenate([decision, jitter_flag, isi_pre_emp, isi_post_emp, opto_side], axis=0)
    non_nan = (1-np.isnan(np.sum(decision, axis=0))).astype('bool')
    decision = decision[:,non_nan]
    # row 0: time.
    # row 1: direction.
    # row 2: correctness.
    # row 3: jitter flag.
    # row 4: pre pert isi.
    # row 5: post pert isi.
    # row 6: opto side
    decision_fix, decision_jitter, decision_chemo, decision_opto= separate_fix_jitter(decision)
    return decision_fix, decision_jitter, decision_chemo, decision_opto


def run(ax, subject_session_data, session_num):
    # Check that the session number is valid.
    subject_session_data_copy = subject_session_data.copy()
    total_sessions = subject_session_data['total_sessions']
    if session_num >= total_sessions:
        raise ValueError(f"session_num {session_num} out of range (total_sessions: {total_sessions}).")

    decision_fix, decision_jitter, decision_chemo, decision_opto = get_decision(subject_session_data_copy, session_num)
    bin_mean_fix, bin_sem_fix, bin_isi_fix = get_bin_stat(decision_fix)

    # Plot the psychometric function for the given session.
    ax.plot(bin_isi_fix, bin_mean_fix, color='indigo', marker='.', label='control', markersize=4)
    ax.fill_between(bin_isi_fix,
                    bin_mean_fix - bin_sem_fix,
                    bin_mean_fix + bin_sem_fix,
                    color='violet', alpha=0.2)
    ax.vlines(750, 0.0, 1.0, linestyle='--', color='mediumseagreen', label='Category Boundary')
    ax.hlines(0.5, 0.0, 1500, linestyle='--', color='grey')
    ax.tick_params(axis='x', rotation=45)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([-50, 1600])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xticks(np.arange(11)*150)
    ax.set_yticks(np.arange(5)*0.25)
    ax.set_xlabel('post perturbation isi')
    ax.set_ylabel('prob. of choosing the right side (mean±sem)')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    date = subject_session_data_copy['dates'][session_num]
    ax.set_title('Psychometric Function ' + date)