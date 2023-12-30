import numpy as np
import matplotlib.pyplot as plt


states = [
    'Reward',
    'RewardNaive',
    'Punish',
    'PunishNaive',
    'WrongInitiation',
    'EarlyChoice',
    'DidNotChoose',
    'DidNotConfirm',
    'ChangingMindReward',
    'Habituation']
colors = [
    'limegreen',
    'springgreen',
    'coral',
    'lightcoral',
    'orange',
    'dodgerblue',
    'deeppink',
    'violet',
    'purple',
    'grey']


def count_bin(outcomes, states, bin_size):
    session_counts = []
    for i in range(len(outcomes)):
        trial_counts = []
        num_bins = int(len(outcomes[i])/bin_size)
        for j in range(num_bins):
            end_idx = len(outcomes[i]) if j == num_bins-1 else (j+1)*bin_size
            bin_data = outcomes[i][j*bin_size:end_idx]
            trial_counts.append(count_label(bin_data, states))
        session_counts.append(np.array(trial_counts))
    return session_counts


def count_label(outcomes, states):
    counts = np.zeros(len(states))
    for j in range(len(states)):
        counts[j] = np.sum(np.array(outcomes) == states[j])
    counts = counts / (np.sum(counts)+1e-5)
    return counts


def plot_line(axs, session_data, session_id=-1, bin_size=10):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    counts = count_bin(outcomes, states, bin_size)
    counts = counts[session_id]
    for j in range(len(states)):
        if len(counts)>0:
            axs.plot(
                np.arange(len(counts))+1,
                counts[:,j],
                color=colors[j],
                label=states[j])
    axs.tick_params(tick1On=False)
    axs.spines['left'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.yaxis.grid(True)
    axs.set_xlabel('bins with size {}'.format(bin_size))
    axs.set_ylabel('percentage')
    axs.set_title(subject + ' ' + dates[session_id])


def plot_fig3(
    session_data_1,
    session_data_2,
    session_data_3,
    session_data_4
    ):
    fig, axs = plt.subplots(2, 2, figsize=(20,8))
    plt.subplots_adjust(hspace=0.7)
    plot_line(axs[0,0], session_data_1)
    plot_line(axs[0,1], session_data_2)
    plot_line(axs[1,0], session_data_3)
    plot_line(axs[1,1], session_data_4)
    axs[0,1].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    fig.suptitle('reward/punish percentage for completed trials across epoches')
    fig.tight_layout()
    print('Plot fig3 completed.')
    fig.savefig('./figures/fig3_percentage_epoch.pdf', dpi=300)
    fig.savefig('./figures/fig3_percentage_epoch.png', dpi=300)

