import os
import numpy as np
import matplotlib.pyplot as plt


# states = [
#     'Reward',
#     'RewardNaive',
#     'Punish',
#     'PunishNaive',
#     'WrongInitiation',
#     'EarlyChoice',
#     'DidNotChoose',
#     'DidNotConfirm',
#     'DidNotLickCenter',
#     'ChangingMindReward',
#     'Habituation']
# colors = [
#     'limegreen',
#     'springgreen',
#     'coral',
#     'lightcoral',
#     'orange',
#     'dodgerblue',
#     'deeppink',
#     'violet',
#     'mediumorchid',
#     'purple',
#     'grey']

states = [
    'Reward',
    'Punish']
colors = [
    'limegreen',
    'deeppink',
    'coral',
    'lightcoral',
    'orange',
    'dodgerblue',
    'deeppink',
    'violet',
    'mediumorchid',
    'purple',
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


def plot_fig1(
        session_data,
        max_sessions=25
        ):
    fig, axs = plt.subplots(1, figsize=(10, 4))
    plt.subplots_adjust(hspace=0.7)
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
    
    axs.set_ylabel('outcome percentages')
    
    axs.set_xticks(np.arange(len(outcomes))+1)
    axs.set_xticklabels(dates, rotation='vertical')
    axs.set_title(subject)
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    fig.suptitle('reward/punish percentage for completed trials across sessions')
    fig.tight_layout()
    print('Completed fig1 for ' + subject)
    print()
    for i in range(len(session_id)):
        Trials = len(outcomes[i])
        Reward = outcomes[i].count('Reward')
        Punish = outcomes[i].count('Punish')
        HitRate = Reward/Trials
        print(subject, dates[i], 'Counts')
        print('Trials:', Trials)
        print('Reward:', Reward)
        print('Punish:', Punish)
        print('Hit Rate:', format(HitRate, ".2%"))
        print()
    # print(session_id 'Outcome counts:')
    # print('Outcome counts:')
    # print()
    os.makedirs('./figures/'+subject+'/outcome', exist_ok = True)
    fig.savefig('./figures/'+subject+'/fig1_'+subject+'_outcome.pdf', dpi=300)
    fig.savefig('./figures/'+subject+'/outcome/fig1_'+subject+'_outcome.png', dpi=300)
    plt.close()
    
    
    
# session_data = session_data_1
# plot_fig1(session_data)

# session_data = session_data_3
# plot_fig1(session_data)