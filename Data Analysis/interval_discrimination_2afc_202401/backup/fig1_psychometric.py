import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from fit_psyche.psychometric_curve import PsychometricCurve


def bin_trials(trial_choice, bin_size=5, least_trials=0):
    num_bins = int(1000/bin_size)
    bin_stat = []
    for i in range(num_bins):
        center = i*bin_size
        idx = np.where(np.abs(trial_choice[:,0]*1000-center)<bin_size/2)[0]
        if len(idx) > least_trials:
            bin_stat.append([
                center/1000,
                np.mean(trial_choice[idx,1]),
                sem(trial_choice[idx,1])])
    bin_stat = np.array(bin_stat).reshape(-1, 3)
    return bin_stat


def separate_fix_jitter_choice(choice, jitter_flag):
    choice_fix = []
    choice_jitter = []
    for session_id in range(len(choice)):
        for trial_id in range(len(choice[session_id])):
            if jitter_flag[session_id][trial_id] == 0:
                choice_fix.append(choice[session_id][trial_id])
            else:
                choice_jitter.append(choice[session_id][trial_id])
    choice_fix = np.concatenate(choice_fix).reshape(-1,3)
    choice_jitter = np.concatenate(choice_jitter).reshape(-1,3)
    return choice_fix, choice_jitter


def get_jitter_flag(choice):
    jitter_flag = []
    for sess_choice in choice:
        trial_jitter_flag = []
        for trial_choice in sess_choice:
            if trial_choice[0] == 0.5:
                trial_jitter_flag.append(0)
            else:
                trial_jitter_flag.append(1)
        jitter_flag.append(trial_jitter_flag)
    return jitter_flag


def separate_pre_post_choice(choice_fix, choice_jitter):
    choice_fix_pre      = np.zeros((choice_fix.shape[0],2))
    choice_fix_pre[:,0] = choice_fix[:,0]
    choice_fix_pre[:,1] = choice_fix[:,2]
    choice_fix_post      = np.zeros((choice_fix.shape[0],2))
    choice_fix_post[:,0] = choice_fix[:,1]
    choice_fix_post[:,1] = choice_fix[:,2]
    choice_jitter_pre      = np.zeros((choice_jitter.shape[0],2))
    choice_jitter_pre[:,0] = choice_jitter[:,0]
    choice_jitter_pre[:,1] = choice_jitter[:,2]
    choice_jitter_post      = np.zeros((choice_jitter.shape[0],2))
    choice_jitter_post[:,0] = choice_jitter[:,1]
    choice_jitter_post[:,1] = choice_jitter[:,2]
    return [choice_fix_pre, choice_fix_post,
            choice_jitter_pre, choice_jitter_post]


def plot_subject(subject_session_data):
    subject = subject_session_data['subject']
    choice = subject_session_data['choice']
    jitter_flag = get_jitter_flag(choice)
    choice_fix, choice_jitter = separate_fix_jitter_choice(
        choice, jitter_flag)
    [choice_fix_pre, choice_fix_post,
     choice_jitter_pre, choice_jitter_post] = separate_pre_post_choice(
         choice_fix, choice_jitter)

    fig, axs = plt.subplots(1, 2, figsize=(8, 3))

    # pre
    bin_stat_fix_pre = bin_trials(choice_fix_pre)
    bin_stat_jitter_pre = bin_trials(choice_jitter_pre)
    pf_fix = PsychometricCurve(model='wh').fit(
        bin_stat_fix_pre[:,0], bin_stat_fix_pre[:,1])
    axs[0].plot(
        np.linspace(0,1,100), pf_fix.predict(np.linspace(0,1,100)),
        color='dodgerblue',
        marker='.',
        label='fix',
        markersize=0.5)
    pf_jitter = PsychometricCurve(model='wh').fit(
        bin_stat_jitter_pre[:,0], bin_stat_jitter_pre[:,1])
    axs[0].plot(
        np.linspace(0,1,100), pf_jitter.predict(np.linspace(0,1,100)),
        color='coral',
        marker='.',
        label='jitter',
        markersize=0.5)
    axs[0].set_title('psychometric function for pre-perturbation')
    # post
    bin_stat_fix_post = bin_trials(choice_fix_post)
    bin_stat_jitter_post = bin_trials(choice_jitter_post)
    pf_fix = PsychometricCurve(model='wh').fit(
        bin_stat_fix_post[:,0], bin_stat_fix_post[:,1])
    axs[1].plot(
        np.linspace(0,1,100), pf_fix.predict(np.linspace(0,1,100)),
        color='dodgerblue',
        marker='.',
        label='fix',
        markersize=0.5)
    pf_jitter = PsychometricCurve(model='wh').fit(
        bin_stat_jitter_post[:,0], bin_stat_jitter_post[:,1])
    axs[1].plot(
        np.linspace(0,1,100), pf_jitter.predict(np.linspace(0,1,100)),
        color='coral',
        marker='.',
        label='jitter',
        markersize=0.5)
    axs[1].set_title('psychometric function for post-perturbation')
    # adjust layout.
    for i in range(2):
        axs[i].hlines(
            0.5, 0.0, 1.0,
            linestyle=':', color='grey')
        axs[i].vlines(
            0.5, 0.0, 1.0,
            linestyle=':', color='grey')
        axs[i].tick_params(tick1On=False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        axs[i].set_xlim([-0.05,1.05])
        axs[i].set_ylim([-0.05,1.05])
        axs[i].set_xticks(np.arange(6)*0.2)
        axs[i].set_yticks(np.arange(5)*0.25)
        axs[i].set_xlabel('isi')
        axs[i].set_ylabel('right fraction')
        axs[i].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    fig.suptitle(subject + ' psychometric functions grand average')
    fig.set_size_inches(8, 3)
    fig.tight_layout()
    fig.savefig('./figures/fig4_psychometric_epoch_'+subject+'.pdf', dpi=300)
    fig.savefig('./figures/fig4_psychometric_epoch_'+subject+'.png', dpi=300)

    '''
    axs[0].plot(
        bin_stat_fix_pre[:,0], bin_stat_fix_pre[:,1],
        color='dodgerblue',
        marker='.',
        label='fix',
        markersize=0.5)
    axs[0].fill_between(
        bin_stat_fix_pre[:,0],
        bin_stat_fix_pre[:,1] - bin_stat_fix_pre[:,2],
        bin_stat_fix_pre[:,1] + bin_stat_fix_pre[:,2],
        color='dodgerblue',
        alpha=0.2)
    axs[0].plot(
        bin_stat_jitter_pre[:,0], bin_stat_jitter_pre[:,1],
        color='coral',
        marker='.',
        label='jitter',
        markersize=0.5)
    axs[0].fill_between(
        bin_stat_jitter_pre[:,0],
        bin_stat_jitter_pre[:,1] - bin_stat_jitter_pre[:,2],
        bin_stat_jitter_pre[:,1] + bin_stat_jitter_pre[:,2],
        color='coral',
        alpha=0.2)
    axs[1].plot(
        bin_stat_fix_post[:,0], bin_stat_fix_post[:,1],
        color='dodgerblue',
        marker='.',
        label='fix',
        markersize=0.5)
    axs[1].fill_between(
        bin_stat_fix_post[:,0],
        bin_stat_fix_post[:,1] - bin_stat_fix_post[:,2],
        bin_stat_fix_post[:,1] + bin_stat_fix_post[:,2],
        color='dodgerblue',
        alpha=0.2)
    axs[1].plot(
        bin_stat_jitter_post[:,0], bin_stat_jitter_post[:,1],
        color='coral',
        marker='.',
        label='jitter',
        markersize=0.5)
    axs[1].fill_between(
        bin_stat_jitter_post[:,0],
        bin_stat_jitter_post[:,1] - bin_stat_jitter_post[:,2],
        bin_stat_jitter_post[:,1] + bin_stat_jitter_post[:,2],
        color='coral',
        alpha=0.2)
    '''


def plot_fig1(session_data):
    for i in range(len(session_data)):
        plot_subject(session_data[i])
    print('Completed fig1')
