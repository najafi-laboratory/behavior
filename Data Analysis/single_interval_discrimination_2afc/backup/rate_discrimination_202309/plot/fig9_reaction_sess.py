import numpy as np
import matplotlib.pyplot as plt


def bin_trials(trial_reaction, max_time, bin_size=0.5, least_trials=2):
    num_bins = int(max_time/bin_size)
    bin_stat = []
    for i in range(num_bins):
        center = i*bin_size
        idx = np.where(np.abs(trial_reaction[:,0]-center)<bin_size/2)[0]
        if len(idx) > least_trials:
            bin_stat.append([
                center, np.sum(trial_reaction[idx,1])/len(idx)])
    bin_stat = np.array(bin_stat).reshape(-1, 2)
    return bin_stat


def plot_subject(
        ax,
        subject_session_data,
        max_sessions
        ):
    subject = subject_session_data['subject']
    dates = subject_session_data['dates']
    reaction = subject_session_data['reaction']
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    dates = dates[start_idx:]
    reaction = reaction[start_idx:]
    processed_reaction = []
    for i in range(len(reaction)):
        if len(reaction[i]) > 0:
            r = np.concatenate(reaction[i], axis=0)
            r = r[:,0].reshape(-1)
            processed_reaction.append(r)
        else:
            processed_reaction.append(np.nan)
    mean = [np.mean(r) for r in processed_reaction]
    std = [np.std(r) for r in processed_reaction]
    loc = np.arange(1, len(dates)+1)
    ax.errorbar(
        loc, mean, yerr=std,
        linestyle='none',
        color='dodgerblue',
        capsize=2,
        marker='o',
        markeredgecolor='white',
        markeredgewidth=1)
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.fill_between(
        [0, len(dates)+1], 0, 1.3,
        color='gold', alpha=0.2,
        label='Pre-perturb stim')
    ax.fill_between(
        [0, len(dates)+1], 1.3, 4.3,
        color='coral', alpha=0.2,
        label='Post-perturb stim')
    ax.set_ylim([0.0, 8])
    ax.set_xlabel('Dates')
    ax.set_ylabel('Reaction time (since stim onset) / s')
    ax.set_xticks(loc)
    ax.set_xticklabels(dates, rotation='vertical')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    ax.set_title(subject + ' Reaction time (the 1st side lick since stim onset) mean/std across sessions')


def plot_fig9(
        session_data,
        max_sessions=20
        ):
    fig, axs = plt.subplots(
        len(session_data), 1,
        figsize=(16, 8*len(session_data)))
    plt.subplots_adjust(hspace=2)
    for i in range(len(session_data)):
        plot_subject(
            axs[i],
            session_data[i],
            max_sessions=max_sessions)
    print('Completed fig9')
    fig.set_size_inches(10, len(session_data)*4)
    fig.tight_layout()
    fig.savefig('./figures/fig9_reaction_sess.pdf', dpi=300)
    fig.savefig('./figures/fig9_reaction_sess.png', dpi=300)
    plt.close()