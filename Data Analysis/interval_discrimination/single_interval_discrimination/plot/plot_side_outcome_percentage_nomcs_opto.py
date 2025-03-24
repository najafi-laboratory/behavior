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
    
    opto_side = subject_session_data['opto_side']
    
    outcomes_left_opto_on = subject_session_data['outcomes_left_opto_on']
    outcomes_right_opto_on = subject_session_data['outcomes_right_opto_on']    
    
    outcomes_left_opto_off = subject_session_data['outcomes_left_opto_off']
    outcomes_right_opto_off = subject_session_data['outcomes_right_opto_off']      
    
    dates = subject_session_data['dates']
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    dates = dates[start_idx:]
    
    opto_side = opto_side[start_idx:]
    opto_sides = []
    
    for side_array in opto_side:
        if 1 in side_array:
            opto_sides.append('Left')
        elif 2 in side_array:
            opto_sides.append('Right')
    
    outcomes_left_opto_off = outcomes_left_opto_off[start_idx:]
    outcomes_left_opto_on = outcomes_left_opto_on[start_idx:]    
    outcomes_right_opto_on = outcomes_right_opto_on[start_idx:]
    outcomes_right_opto_off = outcomes_right_opto_off[start_idx:]
    
    session_id = np.arange(len(outcomes_left)) + 1    
    left_counts = get_side_outcomes(outcomes_left, states)
    right_counts = get_side_outcomes(outcomes_right, states)   
        
    left_opto_off_counts = get_side_outcomes(outcomes_left_opto_off, states)
    left_opto_on_counts = get_side_outcomes(outcomes_left_opto_on, states)
    right_opto_on_counts = get_side_outcomes(outcomes_right_opto_on, states)
    right_opto_off_counts = get_side_outcomes(outcomes_right_opto_off, states)
        
    left_opto_off_bottom = np.cumsum(left_opto_off_counts, axis=1)
    left_opto_off_bottom[:,1:] = left_opto_off_bottom[:,:-1]
    left_opto_off_bottom[:,0] = 0
    
    left_opto_on_bottom = np.cumsum(left_opto_on_counts, axis=1)
    left_opto_on_bottom[:,1:] = left_opto_on_bottom[:,:-1]
    left_opto_on_bottom[:,0] = 0    
    
    right_opto_on_bottom = np.cumsum(right_opto_on_counts, axis=1)
    right_opto_on_bottom[:,1:] = right_opto_on_bottom[:,:-1]
    right_opto_on_bottom[:,0] = 0    
    
    right_opto_off_bottom = np.cumsum(right_opto_off_counts, axis=1)
    right_opto_off_bottom[:,1:] = right_opto_off_bottom[:,:-1]
    right_opto_off_bottom[:,0] = 0    
    
    left_bottom = np.cumsum(left_counts, axis=1)
    left_bottom[:,1:] = left_bottom[:,:-1]
    left_bottom[:,0] = 0        
    right_bottom = np.cumsum(right_counts, axis=1)
    right_bottom[:,1:] = right_bottom[:,:-1]
    right_bottom[:,0] = 0     
    
    width = 0.125
    # ax.show()
    # Get the figure from the axes
    # fig_from_ax = ax.figure
    # Show the figure
    # fig_from_ax.show()
    
    for i in range(len(states)):
        # # Plot the left bars
        # ax.bar(
        #     session_id - width / 2, left_counts[:,i],  # Shift left by width/2
        #     bottom=left_bottom[:,i],
        #     edgecolor='white',
        #     width=width,
        #     color=colors[i],
        #     label=states[i])        
        # # Plot the right bars
        # ax.bar(
        #     session_id + width / 2, right_counts[:,i],  # Shift right by width/2
        #     bottom=right_bottom[:,i],
        #     edgecolor='white',
        #     width=width,
        #     color=colors[i])  # Optionally update label for right bars  

        # Plot the left bars
        bars_left_off = ax.bar(
            session_id - (3*width) / 2, left_opto_off_counts[:,i],  # Shift left by width/2
            bottom=left_opto_off_bottom[:,i],
            edgecolor='white',
            width=width,
            color=colors[i],
            label=states[i]) 

        bars_left_on = ax.bar(
            session_id - width / 2, left_opto_on_counts[:,i],  # Shift left by width/2
            bottom=left_opto_on_bottom[:,i],
            edgecolor='white',
            width=width,
            color=colors[i])         

        # Plot the right bars
        bars_right_off = ax.bar(
            session_id + width / 2, right_opto_off_counts[:,i],  # Shift right by width/2
            bottom=right_opto_off_bottom[:,i],
            edgecolor='white',
            width=width,
            color=colors[i])  # Optionally update label for right bars    
        
        bars_right_on = ax.bar(
            session_id + (3*width) / 2, right_opto_on_counts[:,i],  # Shift right by width/2
            bottom=right_opto_on_bottom[:,i],
            edgecolor='white',
            width=width,
            color=colors[i])  # Optionally update label for right bars          
        # fig_from_ax.show()        
   
    # Add labels to each bar set
    ax.bar_label(bars_left_off, padding=3, labels=['L']*len(bars_left_off))
    ax.bar_label(bars_left_on, padding=3, labels=['LO']*len(bars_left_on))
    ax.bar_label(bars_right_on, padding=3, labels=['RO']*len(bars_right_on))
    ax.bar_label(bars_right_off, padding=3, labels=['R']*len(bars_right_off))    
   
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
    
    date_labels = []
    for i in range(len(dates)):
        date_labels.append(dates[i] + ' ' + opto_sides[i])
        
    ax.set_xticklabels(date_labels, rotation=45)
    

    
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    ax.set_title('reward/punish percentage for completed trials per side, separated by opto', pad=20)
