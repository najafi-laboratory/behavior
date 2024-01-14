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


def plot_fig3(
    session_data,
    session_id=-1,
    bin_size=10
    ):
    fig, axs = plt.subplots(1, figsize=(10,4))
    plt.subplots_adjust(hspace=0.7)
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
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    fig.suptitle('reward/punish percentage for completed trials across epoches')
    fig.tight_layout()
    print('Completed fig3 for ' + subject)
    fig.savefig('./figures/fig3_'+subject+'_percentage_epoch.pdf', dpi=300)
    fig.savefig('./figures/fig3_'+subject+'_opercentage_epoch.png', dpi=300)
    plt.close()