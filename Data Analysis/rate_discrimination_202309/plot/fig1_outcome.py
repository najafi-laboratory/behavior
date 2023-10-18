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
    'DidNotLickCenter']
colors = [
    'springgreen',
    'limegreen',
    'coral',
    'lightcoral',
    'orange',
    'dodgerblue',
    'deeppink',
    'violet',
    'mediumorchid']


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


def plot_bar(axs, session_data, max_sessions=25):
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
    axs.set_xticklabels(dates, rotation='vertical')
    axs.set_title(subject)


def plot_fig1(
    session_data_1,
    session_data_2,
    session_data_3,
    session_data_4
    ):
    fig, axs = plt.subplots(2, 2, figsize=(20, 8))
    plt.subplots_adjust(hspace=0.7)
    plot_bar(axs[0,0], session_data_1)
    plot_bar(axs[0,1], session_data_2)
    plot_bar(axs[1,0], session_data_3)
    plot_bar(axs[1,1], session_data_4)
    axs[0,1].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    fig.suptitle('reward/punish percentage for completed trials across sessions')
    fig.tight_layout()
    print('Plot fig1 completed.')
    fig.savefig('./figures/fig1_outcome.pdf', dpi=300)
    fig.savefig('./figures/fig1_outcome.png', dpi=300)
