import numpy as np
from scipy.stats import sem

def get_bin_stat(decision, max_time):
    bin_size=250
    least_trials=3
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
    return decision_fix, decision_jitter

def get_decision(subject_session_data):
    decision = subject_session_data['decision']
    decision = [np.concatenate(d, axis=1) for d in decision]
    decision = np.concatenate(decision, axis=1)
    jitter_flag = subject_session_data['jitter_flag']
    jitter_flag = np.concatenate(jitter_flag).reshape(1,-1)
    isi_pre_emp = subject_session_data['isi_pre_emp']
    isi_pre_emp = np.concatenate(isi_pre_emp).reshape(1,-1)
    isi_post_emp = subject_session_data['isi_post_emp']
    isi_post_emp = np.concatenate(isi_post_emp).reshape(1,-1)
    stim_start = subject_session_data['stim_start']
    stim_start = np.concatenate(stim_start).reshape(-1)
    decision = np.concatenate([decision, jitter_flag, isi_pre_emp, isi_post_emp], axis=0)
    decision[0,:] -= stim_start
    non_nan = (1-np.isnan(np.sum(decision, axis=0))).astype('bool')
    decision = decision[:,non_nan]
    # row 0: time.
    # row 1: direction.
    # row 2: correctness.
    # row 3: jitter flag.
    # row 4: pre pert isi.
    # row 5: post pert isi.
    decision_fix, decision_jitter = separate_fix_jitter(decision)
    return decision_fix, decision_jitter


def run(ax, subject_session_data):
    max_time = 5000
    decision_fix, decision_jitter = get_decision(subject_session_data)
    bin_mean_fix, bin_sem_fix, bin_time_fix = get_bin_stat(decision_fix, max_time)
    bin_mean_jitter, bin_sem_jitter, bin_time_jitter = get_bin_stat(decision_jitter, max_time)
    ax.plot(
        bin_time_fix,
        bin_mean_fix,
        color='hotpink',
        marker='.',
        label='fix',
        markersize=4)
    ax.fill_between(
        bin_time_fix,
        bin_mean_fix - bin_sem_fix,
        bin_mean_fix + bin_sem_fix,
        color='hotpink',
        alpha=0.2)
    ax.plot(
        bin_time_jitter,
        bin_mean_jitter,
        color='royalblue',
        marker='.',
        label='jitter',
        markersize=4)
    ax.fill_between(
        bin_time_jitter,
        bin_mean_jitter - bin_sem_jitter,
        bin_mean_jitter + bin_sem_jitter,
        color='royalblue',
        alpha=0.2)
    ax.hlines(
        0.5, 0.0, max_time,
        linestyle=':', color='grey',
        label='chance level')
    ax.vlines(
        1300, 0.0, 1.0,
        linestyle=':', color='mediumseagreen',
        label='perturbation')
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([0, max_time])
    ax.set_ylim([0.20, 1.05])
    ax.set_xlabel('decision time (since stim onset) / s')
    ax.set_ylabel('correct prob.')
    ax.set_xticks(np.arange(0, max_time, 1000))
    ax.set_yticks([0.25, 0.50, 0.75, 1])
    ax.legend(loc='lower right', ncol=1)
    ax.set_title('average decision time curve')