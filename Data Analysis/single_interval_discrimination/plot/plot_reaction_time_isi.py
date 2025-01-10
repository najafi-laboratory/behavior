import numpy as np
from scipy.stats import sem

def get_bin_stat(reaction, max_time):
    bin_size=100
    least_trials=5
    bins = np.arange(0, 1000 + bin_size, bin_size)
    bins = bins - bin_size / 2
    bin_indices = np.digitize(reaction[5,:], bins) - 1
    bin_mean = []
    bin_sem = []
    for i in range(len(bins)-1):
        correctness = reaction[0, bin_indices == i].copy()
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
    reaction_chemo = reaction[:,reaction[3,:]==2]
    reaction_opto = reaction[:,reaction[3,:]==3]
    return reaction_fix, reaction_jitter, reaction_chemo, reaction_opto

def get_reaction(subject_session_data):
    reaction = subject_session_data['reaction']
    reaction = [np.concatenate(d, axis=1) for d in reaction]
    reaction = np.concatenate(reaction, axis=1)
    jitter_flag = subject_session_data['jitter_flag']
    jitter_flag = np.concatenate(jitter_flag).reshape(1,-1)
    opto_flag = subject_session_data['opto_flag']
    opto_flag = np.concatenate(opto_flag).reshape(1,-1)
    jitter_flag[0 , :] = jitter_flag[0 , :] + opto_flag[0 , :]*3
    outcomes = subject_session_data['outcomes']
    all_trials = 0
    chemo_labels = subject_session_data['Chemo']
    for j in range(len(chemo_labels)):
        if chemo_labels[j] == 1 :
            jitter_flag[0 , all_trials:all_trials+len(outcomes[j])] = 2*np.ones(len(outcomes[j]))
        all_trials += len(outcomes[j])
    pre_isi = subject_session_data['pre_isi']
    pre_isi = np.concatenate(pre_isi).reshape(1,-1)
    post_isi_mean = subject_session_data['isi_post_emp']
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
    reaction_fix, reaction_jitter, reaction_chemo, reaction_opto = separate_fix_jitter(reaction)
    return reaction_fix, reaction_jitter, reaction_chemo, reaction_opto


def run(ax, subject_session_data):
    max_time = 5000
    reaction_fix, reaction_jitter, reaction_chemo , reaction_opto = get_reaction(subject_session_data)
    bin_mean_fix, bin_sem_fix, bin_time_fix = get_bin_stat(reaction_fix, max_time)
    bin_mean_jitter, bin_sem_jitter, bin_time_jitter = get_bin_stat(reaction_jitter, max_time)
    bin_mean_chemo, bin_sem_chemo, bin_time_chemo = get_bin_stat(reaction_chemo, max_time)
    bin_mean_opto, bin_sem_opto, bin_time_opto = get_bin_stat(reaction_opto, max_time)
    ax.plot(
        bin_time_fix,
        bin_mean_fix,
        color='black',
        marker='.',
        label='fix',
        markersize=4)
    ax.fill_between(
        bin_time_fix,
        bin_mean_fix - bin_sem_fix,
        bin_mean_fix + bin_sem_fix,
        color='black',
        alpha=0.2)
    ax.plot(
        bin_time_jitter,
        bin_mean_jitter,
        color='limegreen',
        marker='.',
        label='jitter',
        markersize=4)
    ax.fill_between(
        bin_time_jitter,
        bin_mean_jitter - bin_sem_jitter,
        bin_mean_jitter + bin_sem_jitter,
        color='limegreen',
        alpha=0.2)
    ax.plot(
        bin_time_chemo,
        bin_mean_chemo,
        color='red',
        marker='.',
        label='chemo',
        markersize=4)
    ax.fill_between(
        bin_time_chemo,
        bin_mean_chemo - bin_sem_chemo,
        bin_mean_chemo + bin_sem_chemo,
        color='red',
        alpha=0.2)
    ax.plot(
        bin_time_opto,
        bin_mean_opto,
        color='dodgerblue',
        marker='.',
        label='opto',
        markersize=4)
    ax.fill_between(
        bin_time_opto,
        bin_mean_opto - bin_sem_opto,
        bin_mean_opto + bin_sem_opto,
        color='dodgerblue',
        alpha=0.2)
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.vlines(500, 1200, 3500, linestyle=':', color='grey')
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([-50,1050])
    ax.set_xticks(np.arange(6)*200)
    ax.set_xlabel('post perturbation isi')
    ax.set_ylabel('Reaction Time (ms)')
    ax.set_title('average reaction time for percepted post pert isi')
    ax.legend(loc='lower right', ncol=1)
    