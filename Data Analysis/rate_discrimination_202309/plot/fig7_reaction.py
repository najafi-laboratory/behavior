import numpy as np
import matplotlib.pyplot as plt


def bin_trials(trial_reaction, max_time, bin_size=0.2, least_trials=3):
    num_bins = int(max_time/bin_size)
    bin_stat = []
    for i in range(num_bins):
        center = i*bin_size
        idx = np.where(np.abs(trial_reaction-center)<bin_size/2)[0]
        if len(idx) > least_trials:
            bin_stat.append([
                center, np.sum(trial_reaction[idx])/len(idx)])
    bin_stat = np.array(bin_stat).reshape(-1, 2)
    return bin_stat


def plot_curves(axs, session_data, max_time=10, max_sessions=10):
    subject = session_data['subject']
    reaction = session_data['reaction'][session_data['LR12_start']:]
    dates = session_data['dates'][session_data['LR12_start']:]
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    reaction = reaction[start_idx:]
    dates = dates[start_idx:]
    cmap = plt.cm.RdBu(np.arange(len(reaction))/len(reaction))
    for i in range(len(reaction)):
        trial_reaction = np.concatenate(reaction)
        bin_stat = bin_trials(trial_reaction, max_time)
        axs.plot(
            bin_stat[:,0], bin_stat[:,1],
            color=cmap[i],
            label=dates[i])
        axs.scatter(
            bin_stat[:,0], bin_stat[:,1],
            color=cmap[i],
            s=5)
    axs.tick_params(tick1On=False)
    axs.spines['left'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.axvspan(
        0, 1.8,
        linestyle=':', color='coral', alpha=0.1,
        label='pre-perturb stim')
    axs.axvspan(
        1.8, 3.7,
        linestyle=':', color='violet', alpha=0.1,
        label='post-perturb stim')
    axs.set_xlim([0, max_time])
    axs.set_ylim([-0.05, 1.05])
    axs.set_xlabel('time since stim starts / s')
    axs.set_ylabel('probability of correct choice')
    axs.set_yticks([0, 0.25, 0.50, 0.75, 1])
    axs.yaxis.grid(True)
    axs.set_title(subject)
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)


def plot_fig7(
    session_data_1,
    session_data_2,
    session_data_3,
    session_data_4
    ):
    fig, axs = plt.subplots(2, 2, figsize=(24, 8))
    plt.subplots_adjust(hspace=0.7)
    plot_curves(axs[0,0], session_data_1)
    plot_curves(axs[0,1], session_data_2)
    plot_curves(axs[1,0], session_data_3)
    plot_curves(axs[1,1], session_data_4)
    fig.suptitle('reaction time vs probability of correctness')
    fig.tight_layout()
    print('Plot fig7 completed.')
    fig.savefig('./figures/fig7_reaction.pdf', dpi=300)
    fig.savefig('./figures/fig7_reaction.png', dpi=300)