import numpy as np

states = [
    'Reward',
    'Punish']
colors = [
    'limegreen',
    'coral']

# states = [
#     'Reward',
#     'RewardNaive',
#     'ChangingMindReward',
#     'Punish',
#     'PunishNaive']
# colors = [
#     'limegreen',
#     'springgreen',
#     'dodgerblue',
#     'coral',
#     'violet']

def get_side_outcomes(outcomes, states):
    num_session = len(outcomes)
    counts = np.zeros((num_session, len(states)))
    for i in range(num_session):
        for j in range(len(states)):
            counts[i,j] = np.sum(np.array(outcomes[i]) == states[j])
        counts[i,:] = counts[i,:] / (np.sum(counts[i,:])+1e-5)
    return counts

def run(ax, subject_session_data):
    max_sessions=25                
    outcomes_left = subject_session_data['outcomes_left']
    outcomes_right = subject_session_data['outcomes_right']
    dates = subject_session_data['dates']
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    dates = dates[start_idx:]
    outcomes_left = outcomes_left[start_idx:]
    outcomes_right = outcomes_right[start_idx:]
    session_id = np.arange(len(outcomes_left)) + 1    
    left_counts = get_side_outcomes(outcomes_left, states)
    right_counts = get_side_outcomes(outcomes_right, states)    
    left_bottom = np.cumsum(left_counts, axis=1)
    left_bottom[:,1:] = left_bottom[:,:-1]
    left_bottom[:,0] = 0        
    right_bottom = np.cumsum(right_counts, axis=1)
    right_bottom[:,1:] = right_bottom[:,:-1]
    right_bottom[:,0] = 0       
    width = 0.25
    for i in range(len(states)):
        # Plot the left bars
        ax.bar(
            session_id - width / 2, left_counts[:,i],  # Shift left by width/2
            bottom=left_bottom[:,i],
            edgecolor='white',
            width=width,
            color=colors[i],
            label=states[i])        
        # Plot the right bars
        ax.bar(
            session_id + width / 2, right_counts[:,i],  # Shift right by width/2
            bottom=right_bottom[:,i],
            edgecolor='white',
            width=width,
            color=colors[i])  # Optionally update label for right bars  

   
    ax.tick_params(tick1On=False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.hlines(0.5,0,len(dates)+1, linestyle='--' , color='silver' , lw = 0.5)
    ax.hlines(0.75,0,len(dates)+1, linestyle='--' , color='silver' , lw = 0.5)    
    ax.yaxis.grid(False)
    ax.set_xlabel('training session')
    ax.set_ylabel('number of trials')
    top_labels = []
    for i in range(len(dates)):
        top_labels.append('L')  # Label for the first bar
        top_labels.append('R')  # Label for the second bar
    # Update x-ticks and x-tick labels
    tick_positions_bottom = np.arange(len(outcomes_left))+1
    tick_positions_top = np.repeat(tick_positions_bottom, 2)  # Repeat x positions for each set of bars (top labels)
    ax.set_xticks(tick_positions_bottom)
    # ax.set_xticks(tick_positions_top)  # Set the tick positions for the top labels
    # ax.set_xticklabels(top_labels, rotation=45, ha='right')  # Set the top labels and rotate    
    ax.set_yticks(np.arange(6)*0.2)
    ax.set_xticklabels(dates, rotation=45)
    

    
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    ax.set_title('reward/punish percentage for completed trials per side')
