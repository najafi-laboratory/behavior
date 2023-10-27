import numpy as np
import matplotlib.pyplot as plt


def label_early(trial_outcomes):
    trial_early = np.array(trial_outcomes)=='EarlyChoice'
    trial_early = trial_early.astype('int32')
    return trial_early


def bin_trials(trial_early, iti, max_time, bin_size=0.5, least_trials=3):
    centers = np.arange(0, int(max_time/bin_size)) * bin_size
    bin_stat = []
    for c in centers:
        idx = np.where(np.abs(np.array(iti)-c)<bin_size/2)[0]
        if len(idx) > least_trials:
            bin_stat.append([
                c, np.sum(trial_early[idx])/len(idx)])
    bin_stat = np.array(bin_stat).reshape(-1, 2)
    return bin_stat


def plot_curves(axs, session_data, max_time=6, max_sessions=10):
    subject = session_data['subject']
    iti = session_data['iti'][session_data['LR12_start']:]
    outcomes = session_data['outcomes'][session_data['LR12_start']:]
    dates = session_data['dates'][session_data['LR12_start']:]
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    iti = iti[start_idx:]
    outcomes = outcomes[start_idx:]
    dates = dates[start_idx:]
    cmap = plt.cm.plasma_r(np.arange(len(iti))/len(iti))
    for i in range(len(iti)):
        trial_early = label_early(outcomes[i])
        bin_stat = bin_trials(trial_early, iti[i], max_time=max_time)
        axs.plot(
            bin_stat[:,0], bin_stat[:,1],
            color=cmap[i],
            label=dates[i])
        axs.scatter(
            bin_stat[:,0], bin_stat[:,1],
            color=cmap[i],
            s=5)
    axs.tick_params(tick1On=False)
    axs.tick_params(tick1On=False)
    axs.spines['left'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlim([0.9, 5.1])
    axs.set_ylim([0, 1])
    axs.set_yticks([0, 0.25, 0.50, 0.75, 1])
    axs.yaxis.grid(True)
    axs.set_xlabel('iti / s')
    axs.set_ylabel('probability of early choice')
    axs.set_title(subject)
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)


def plot_fig7(
    session_data_1,
    session_data_2,
    session_data_3,
    session_data_4
    ):
    fig, axs = plt.subplots(2, 2, figsize=(20, 8))
    plt.subplots_adjust(hspace=0.7)
    plot_curves(axs[0,0], session_data_1)
    plot_curves(axs[0,1], session_data_2)
    plot_curves(axs[1,0], session_data_3)
    plot_curves(axs[1,1], session_data_4)
    fig.suptitle('iti and probability of early choice')
    fig.tight_layout()
    print('Plot fig7 completed.')
    fig.savefig('./figures/fig7_early_iti.pdf', dpi=300)
    fig.savefig('./figures/fig7_early_iti.png', dpi=300)
