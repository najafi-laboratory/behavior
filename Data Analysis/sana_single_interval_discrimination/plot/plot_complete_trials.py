import numpy as np

states = [
    'Reward',
    'RewardNaive',
    'ChangingMindReward',
    'Punish',
    'PunishNaive']
colors = [
    'limegreen',
    'springgreen',
    'dodgerblue',
    'coral',
    'violet']

def count_label(outcomes, states):
    num_session = len(outcomes)
    counts = np.zeros((num_session, len(states)))
    for i in range(num_session):
        for j in range(len(states)):
            counts[i,j] = np.sum(np.array(outcomes[i]) == states[j])
        counts[i,:] = counts[i,:] / (np.sum(counts[i,:])+1e-5)
    return counts

def run(ax, subject_session_data):
    max_sessions=25
    outcomes = subject_session_data['outcomes']
    dates = subject_session_data['dates']
    chemo_labels = subject_session_data['Chemo']
    jitter_flag = subject_session_data['jitter_flag']
    jitter_session = np.array([np.sum(j) for j in jitter_flag])
    jitter_session[jitter_session!=0] = 1
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    outcomes = outcomes[start_idx:]
    dates = dates[start_idx:]
    jitter_session = jitter_session[start_idx:]
    chemo_labels = chemo_labels[start_idx:]
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
    #ax.yaxis.grid(True)
    ax.set_xlabel('training session')
    ax.set_ylabel('number of trials')
    ax.set_xticks(np.arange(len(outcomes))+1)
    ax.set_yticks(np.arange(6)*0.2)
    #ax.set_xticklabels(dates, rotation='vertical')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    ax.set_title('reward/punish percentage for completed trials')
    dates_label = dates
    for i in range(0 , len(chemo_labels)):
        if chemo_labels[i] == 1:
            dates_label[i] = dates[i] + '(chemo)'
        if jitter_session[i] == 1:
            dates_label[i] =  dates_label[i] + '(jittered)'
    ax.set_xticklabels(dates_label, rotation=45)
    ind = 0
    for xtick in ax.get_xticklabels():
        if jitter_session[ind] == 1:
            xtick.set_color('limegreen')
        if chemo_labels[ind] == 1:
            xtick.set_color('red')
        ind = ind + 1
