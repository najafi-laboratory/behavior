import numpy as np
from scipy.stats import sem

def get_bin_stat(reaction, max_time):
    bin_size=250
    least_trials=3
    bins = np.arange(0, max_time + bin_size, bin_size)
    bins = bins - bin_size / 2
    bin_indices = np.digitize(reaction[0,:], bins) - 1
    bin_mean = []
    bin_sem = []
    for i in range(len(bins)-1):
        correctness = reaction[2, bin_indices == i].copy()
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


def separate_fix_jitter(reaction):
    reaction_fix = reaction[:,reaction[3,:]==0]
    reaction_jitter = reaction[:,reaction[3,:]==1]
    return reaction_fix, reaction_jitter

def get_reaction(subject_session_data):
    reaction = subject_session_data['reaction']
    reaction = [np.concatenate(d, axis=1) for d in reaction]
    reaction = np.concatenate(reaction, axis=1)
    jitter_flag = subject_session_data['jitter_flag']
    jitter_flag = np.concatenate(jitter_flag).reshape(1,-1)
    pre_isi = subject_session_data['pre_isi']
    pre_isi = np.concatenate(pre_isi).reshape(1,-1)
    post_isi_mean = subject_session_data['post_isi_mean']
    post_isi_mean = np.concatenate(post_isi_mean).reshape(1,-1)
    stim_start = subject_session_data['stim_start']
    stim_start = np.concatenate(stim_start).reshape(-1)
    reaction = np.concatenate([reaction, jitter_flag, pre_isi, post_isi_mean], axis=0)
    reaction[0,:] -= stim_start
    non_nan = (1-np.isnan(np.sum(reaction, axis=0))).astype('bool')
    reaction = reaction[:,non_nan]
    # row 0: time.
    # row 1: direction.
    # row 2: correctness.
    # row 3: jitter flag.
    # row 4: pre pert isi.
    # row 5: post pert isi.
    reaction_fix, reaction_jitter = separate_fix_jitter(reaction)
    return reaction_fix, reaction_jitter


def run(ax, subject_session_data):
    max_time = 5000
    reaction_fix, reaction_jitter = get_reaction(subject_session_data)
    bin_mean_fix, bin_sem_fix, bin_time_fix = get_bin_stat(reaction_fix, max_time)
    bin_mean_jitter, bin_sem_jitter, bin_time_jitter = get_bin_stat(reaction_jitter, max_time)
    ax.plot(
        bin_time_fix,
        bin_mean_fix,
        color='dodgerblue',
        marker='.',
        label='fix',
        markersize=4)
    ax.fill_between(
        bin_time_fix,
        bin_mean_fix - bin_sem_fix,
        bin_mean_fix + bin_sem_fix,
        color='dodgerblue',
        alpha=0.2)
    ax.plot(
        bin_time_jitter,
        bin_mean_jitter,
        color='coral',
        marker='.',
        label='jitter',
        markersize=4)
    ax.fill_between(
        bin_time_jitter,
        bin_mean_jitter - bin_sem_jitter,
        bin_mean_jitter + bin_sem_jitter,
        color='coral',
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
    ax.set_xlabel('reaction time (since stim onset) / s')
    ax.set_ylabel('correct prob.')
    ax.set_xticks(np.arange(0, max_time, 1000))
    ax.set_yticks([0.25, 0.50, 0.75, 1])
    ax.legend(loc='lower right', ncol=1)
    ax.set_title('average reaction time curve')