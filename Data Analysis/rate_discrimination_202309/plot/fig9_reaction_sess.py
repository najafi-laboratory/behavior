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


def plot_fig9(
    session_data,
    max_sessions=20
    ):
    fig, axs = plt.subplots(1, figsize=(10, 4))
    subject = session_data['subject']
    dates = session_data['dates']
    reaction = session_data['reaction']
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
    axs.errorbar(
        loc, mean, yerr=std,
        linestyle='none',
        color='dodgerblue',
        capsize=2,
        marker='o',
        markeredgecolor='white',
        markeredgewidth=1)
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.fill_between(
        [0, max_sessions], 0, 1.3,
        color='gold', alpha=0.2,
        label='Pre-perturb stim')
    axs.fill_between(
        [0, max_sessions], 1.3, 4.3,
        color='coral', alpha=0.2,
        label='Post-perturb stim')
    axs.set_ylim([0.0, 8])
    axs.set_xlabel('Dates')
    axs.set_ylabel('Reaction time (since stim onset) / s')
    axs.set_xticks(loc)
    axs.set_xticklabels(dates, rotation='vertical')
    axs.set_title(subject + ' ')
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    fig.suptitle('Reaction time (the 1st side lick since stim onset) mean/std across sessions')
    fig.tight_layout()
    print('Completed fig8 for ' + subject)
    fig.savefig('./figures/fig8_'+subject+'_reaction_sess.pdf', dpi=300)
    fig.savefig('./figures/fig8_'+subject+'_reaction_sess.png', dpi=300)
    plt.close()