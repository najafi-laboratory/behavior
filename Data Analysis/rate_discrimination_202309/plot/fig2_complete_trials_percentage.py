import numpy as np
import matplotlib.pyplot as plt


states = [
    'Reward',
    'RewardNaive',
    'ChangingMindReward',
    'Punish',
    'PunishNaive']
colors = [
    'limegreen',
    'springgreen',
    'purple',
    'coral',
    'lightcoral']


def count_label(outcomes, states):
    num_session = len(outcomes)
    counts = np.zeros((num_session, len(states)))
    for i in range(num_session):
        for j in range(len(states)):
            counts[i,j] = np.sum(np.array(outcomes[i]) == states[j])
        counts[i,:] = counts[i,:] / (np.sum(counts[i,:])+1e-5)
    return counts


def plot_fig2(
    session_data,
    max_sessions=25
    ):
    fig, axs = plt.subplots(1, figsize=(10, 4))
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    outcomes = outcomes[start_idx:]
    dates = dates[start_idx:]
    counts = count_label(outcomes, states)
    session_id = np.arange(len(outcomes)) + 1
    bottom = np.cumsum(counts, axis=1)
    bottom[:,1:] = bottom[:,:-1]
    bottom[:,0] = 0
    width = 0.5
    for i in range(len(states)):
        axs.bar(
            session_id, counts[:,i],
            bottom=bottom[:,i],
            edgecolor='white',
            width=width,
            color=colors[i],
            label=states[i])
    axs.tick_params(tick1On=False)
    axs.spines['left'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.yaxis.grid(True)
    axs.set_xlabel('training session')
    axs.set_ylabel('number of trials')
    axs.set_xticks(np.arange(len(outcomes))+1)
    axs.set_yticks(np.arange(6)*0.2)
    axs.set_xticklabels(dates, rotation='vertical')
    axs.set_title(subject)
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    fig.suptitle('reward/punish percentage for completed trials across sessions')
    fig.tight_layout()
    print('Completed fig2 for ' + subject)
    fig.savefig('./figures/fig2_'+subject+'_complete_trials_percentage.pdf', dpi=300)
    fig.savefig('./figures/fig2_'+subject+'_complete_trials_percentage.png', dpi=300)
    plt.close()