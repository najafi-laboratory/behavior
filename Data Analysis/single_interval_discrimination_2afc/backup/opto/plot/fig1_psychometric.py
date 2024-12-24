import numpy as np
import matplotlib.pyplot as plt


def bin_trials(trial_choice, bin_size=100, least_trials=3):
    num_bins = int(1000/bin_size)
    bin_stat = []
    for i in range(num_bins):
        center = i*bin_size
        idx = np.where(np.abs(trial_choice[:,0]*1000-center)<bin_size/2)[0]
        if len(idx) > least_trials:
            bin_stat.append([
                center/1000,
                np.mean(trial_choice[idx,1]),
                np.std(trial_choice[idx,1])])
    bin_stat = np.array(bin_stat).reshape(-1, 3)
    return bin_stat


def separate_control_opto_choice(choice, opto_flag):
    choice_control = []
    choice_opto = []
    for session_id in range(len(choice)):
        for trial_id in range(len(choice[session_id])):
            if opto_flag[session_id][trial_id] == 0:
                choice_control.append(choice[session_id][trial_id])
            else:
                choice_opto.append(choice[session_id][trial_id])
    choice_control = np.concatenate(choice_control).reshape(-1,2)
    choice_opto = np.concatenate(choice_opto).reshape(-1,2)
    return choice_control, choice_opto


def plot_subject(subject_session_data):
    subject = subject_session_data['subject']
    choice = subject_session_data['choice']
    opto_flag = subject_session_data['opto_flag']
    choice_control, choice_opto = separate_control_opto_choice(
        choice, opto_flag)

    fig, axs = plt.subplots(1, 1, figsize=(4, 3))

    bin_stat_control = bin_trials(choice_control)
    bin_stat_opto = bin_trials(choice_opto)
    axs.plot(
        bin_stat_control[:,0], bin_stat_control[:,1],
        color='dodgerblue',
        marker='.',
        label='control',
        markersize=0.5)
    axs.plot(
        bin_stat_opto[:,0], bin_stat_opto[:,1],
        color='coral',
        marker='.',
        label='opto',
        markersize=0.5)
    axs.hlines(
        0.5, 0.0, 1.0,
        linestyle=':', color='grey')
    axs.vlines(
        0.5, 0.0, 1.0,
        linestyle=':', color='grey')
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlim([-0.05,1.05])
    axs.set_ylim([-0.05,1.05])
    axs.set_xticks(np.arange(6)*0.2)
    axs.set_yticks(np.arange(5)*0.25)
    axs.set_xlabel('isi')
    axs.set_ylabel('right fraction')
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    fig.suptitle(subject + ' psychometric functions grand average')
    fig.set_size_inches(4, 3)
    fig.tight_layout()


def plot_fig1(session_data):
    for i in range(len(session_data)):
        plot_subject(session_data[i])
    print('Completed fig1')
