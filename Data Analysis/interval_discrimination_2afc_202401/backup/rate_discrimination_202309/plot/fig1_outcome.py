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


def count_label(session_label, states, norm=True):
    num_session = len(session_label)
    counts = np.zeros((num_session, len(states)))
    for i in range(num_session):
        for j in range(len(states)):
            if norm:
                counts[i,j] = np.sum(
                    np.array(session_label[i]) == states[j]
                    ) / len(session_label[i])
            else:
                counts[i,j] = np.sum(
                    np.array(session_label[i]) == states[j]
                    )
    return counts


def plot_subject(
        ax,
        subject_session_data,
        max_sessions,
        ):
    subject = subject_session_data['subject']
    outcomes = subject_session_data['outcomes']
    dates = subject_session_data['dates']
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
        ax.bar(
            session_id, counts[:,i],
            bottom=bottom[:,i],
            edgecolor='white',
            width=width,
            color=colors[i],
            label=states[i])
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.grid(True)
    ax.set_xlabel('training session')
    ax.set_ylabel('number of trials')
    ax.set_xticks(np.arange(len(outcomes))+1)
    ax.set_xticklabels(dates, rotation='vertical')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    ax.set_title(subject + ' outcome percentage')

    
def plot_fig1(
        session_data,
        max_sessions=25
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
    print('Completed fig1')
    fig.set_size_inches(12, len(session_data)*3)
    fig.tight_layout()
    fig.savefig('./figures/fig1_outcome.pdf', dpi=300)
    fig.savefig('./figures/fig1_outcome.png', dpi=300)
    plt.close()