import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import random
import re

states = [
    'Reward' , 
    'DidNotPress1' , 
    'DidNotPress2' , 
    'EarlyPress' , 
    'EarlyPress1' , 
    'EarlyPress2' ,
    'VisStimInterruptDetect1' ,         #int1
    'VisStimInterruptDetect2' ,         #int2
    'VisStimInterruptGray1' ,           #int3
    'VisStimInterruptGray2' ,           #int4
    'Other']                            #int
states_name = [
    'Reward' , 
    'DidNotPress1' , 
    'DidNotPress2' , 
    'EarlyPress' , 
    'EarlyPress1' , 
    'EarlyPress2' , 
    'VisStimInterruptDetect1' ,
    'VisStimInterruptDetect2' ,
    'VisStimInterruptGray1' ,
    'VisStimInterruptGray2' ,
    'VisInterrupt']
colors = [
    '#4CAF50',
    '#FFB74D',
    '#FB8C00',
    'r',
    '#64B5F6',
    '#1976D2',
    '#967bb6',
    '#9932CC',
    '#800080',
    '#4B0082',
    '#2E003E',
    'purple',
    'deeppink',
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


def deduplicate_chemo(strings):
    result = []
    for string in strings:
        # Find all occurrences of (chemo)
        chemo_occurrences = re.findall(r'\(chemo\)', string)
        # If more than one (chemo) found, replace all but the first with empty string
        if len(chemo_occurrences) > 1:
            # Keep only one (chemo)
            string = re.sub(r'\(chemo\)', '', string)
            string = string + '(chemo)'
        result.append(string)
    return result

def plot_fig1(
        session_data,
        output_dir_onedrive, 
        output_dir_local
        ):
    
    max_sessions=100
    fig, axs = plt.subplots(1, figsize=(14,7))
    plt.subplots_adjust(hspace=0.7)
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    chemo_labels = session_data['chemo']
    start_idx = 0
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    outcomes = outcomes[start_idx:]
    dates = dates[start_idx:]  
    new_dates = []
    for date_item in dates:
        new_dates.append(date_item[2:])
    dates = new_dates
      
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
            label=states_name[i])
    axs.tick_params(tick1On=False)
    axs.spines['left'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('Training session')
    
    axs.set_ylabel('Outcome percentages')
    
    axs.set_xticks(np.arange(len(outcomes))+1)
    
    dates_label = dates
    for i in range(0 , len(chemo_labels)):
        if chemo_labels[i] == 1:
            dates_label[i] = dates[i] + '(chemo)'
    dates_label = deduplicate_chemo(dates_label)
    axs.set_xticklabels(dates_label, rotation='vertical')
    ind = 0
    for xtick in axs.get_xticklabels():
        if chemo_labels[ind] == 1:
            xtick.set_color('r')
        ind = ind + 1
    axs.set_title(subject)
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    fig.suptitle('Reward percentage for completed trials across sessions')
    fig.tight_layout()
    print('Completed fig1 outcome percentages for ' + subject)
    print()
    
    # Saving the reference of the standard output
    original_stdout = sys.stdout
    today = date.today()
    today_formatted = str(today)[2:]
    year = today_formatted[0:2]
    month = today_formatted[3:5]
    day = today_formatted[6:]
    today_string = year + month + day
    output_dir = 'C:\\data analysis\\behavior\\joystick\\'
    output_logs_dir = output_dir +'logs\\'
    output_logs_fname = output_logs_dir + subject + 'outcome_log_' + today_string + '.txt'
    os.makedirs(output_logs_dir, exist_ok = True)
    
    Trials = []
    Reward = []
    Punish = []
    HitRate = []
    
    for i in range(len(session_id)):
        if 'Other' in outcomes[i]:
            Trials.append(len(outcomes[i]) - outcomes[i].count('Other'))
            Punish.append(len(outcomes[i]) - outcomes[i].count('Reward') - outcomes[i].count('Other'))
        else:
            Trials.append(len(outcomes[i]))
            Punish.append(len(outcomes[i])-outcomes[i].count('Reward'))
        Reward.append(outcomes[i].count('Reward'))
        HitRate.append(Reward[i]/Trials[i])
    
    with open(output_logs_fname, 'w') as f:
        sys.stdout = f
           
    # Reset the standard output
    sys.stdout = original_stdout 

    for i in range(len(session_id)):
        print(subject, dates[i], 'Counts')
        print('Trials:', Trials[i])
        print('Reward:', Reward[i])
        print('NonRewarded:', Punish[i])
        print('Hit Rate:', format(HitRate[i], ".2%"))
        print()
    
         
    output_figs_dir = output_dir_onedrive + subject + '/'    
    output_imgs_dir = output_dir_local + subject + '/outcome_imgs/'    
    os.makedirs(output_figs_dir, exist_ok = True)
    os.makedirs(output_imgs_dir, exist_ok = True)
    fig.savefig(output_figs_dir + subject + '_Outcome.pdf', dpi=300)
    fig.savefig(output_imgs_dir + subject + '_Outcome.png', dpi=300)
    plt.close()
    
