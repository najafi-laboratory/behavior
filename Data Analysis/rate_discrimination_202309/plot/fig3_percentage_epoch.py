import numpy as np
import matplotlib.pyplot as plt


states = [
    'Reward',
    'RewardNaive',
    'ChangingMindReward',
    'Punish',
    'PunishNaive',
    'WrongInitiation',
    'DidNotChoose']
colors = [
    'limegreen',
    'springgreen',
    'dodgerblue',
    'coral',
    'violet',
    'orange',
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


def plot_subject(
        ax,
        subject_session_data,
        session_id,
        bin_size,
        ):
    subject = subject_session_data['subject']
    outcomes = subject_session_data['outcomes']
    dates = subject_session_data['dates']
    counts = count_bin(outcomes, states, bin_size)
    counts = counts[session_id]
    for j in range(len(states)):
        if len(counts)>0:
            ax.plot(
                np.arange(len(counts))+1,
                counts[:,j],
                color=colors[j],
                label=states[j])
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.grid(True)
    ax.set_xlabel('bins with size {}'.format(bin_size))
    ax.set_ylabel('percentage')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    ax.set_title(
        subject + ' ' + dates[session_id] + 
        ' outcome percentage across epoches')
    
    
def plot_fig3(
        session_data,
        session_id=-1,
        bin_size=10
        ):
    fig, axs = plt.subplots(
        len(session_data), 1,
        figsize=(16, 8*len(session_data)))
    plt.subplots_adjust(hspace=2)
    for i in range(len(session_data)):
        plot_subject(
            axs[i],
            session_data[i],
            session_id=session_id,
            bin_size=bin_size)
    print('Completed fig3')
    fig.set_size_inches(10, len(session_data)*3)
    fig.tight_layout()
    fig.savefig('./figures/fig3_percentage_epoch.pdf', dpi=300)
    fig.savefig('./figures/fig3_percentage_epoch.png', dpi=300)
    plt.close()