import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import random


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
                if 'Other' in session_label[i]:
                    counts[i,j] = np.sum(
                        np.array(session_label[i]) == states[j]
                        ) / (len(session_label[i]) - session_label[i].count('Other'))
                else:
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
        output_dir_onedrive, 
        output_dir_local
        ):
    
    max_sessions=25
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
            label=states[i])
    axs.tick_params(tick1On=False)
    axs.spines['left'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.yaxis.grid(True)
    axs.set_xlabel('Training session')
    
    axs.set_ylabel('Outcome percentages')
    
    axs.set_xticks(np.arange(len(outcomes))+1)
    axs.set_xticklabels(dates, rotation='vertical')
    axs.set_title(subject)
    axs.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    fig.suptitle('Reward/punish percentage for completed trials across sessions')
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
        else:
            Trials.append(len(outcomes[i]))
        Reward.append(outcomes[i].count('Reward'))
        Punish.append(outcomes[i].count('Punish'))
        HitRate.append(Reward[i]/Trials[i])
    
    with open(output_logs_fname, 'w') as f:
        sys.stdout = f
    
    
        for i in range(len(session_id)):
            # Trials = len(outcomes[i])
            # Reward = outcomes[i].count('Reward')
            # Punish = outcomes[i].count('Punish')
            # HitRate = Reward/Trials
            # print to log file
            print(subject, dates[i], 'Counts')
            print('Trials:', Trials[i])
            print('Reward:', Reward[i])
            print('Punish:', Punish[i])
            print('Hit Rate:', format(HitRate[i], ".2%"))
            print()
           
    # Reset the standard output
    sys.stdout = original_stdout 

    for i in range(len(session_id)):
        # Trials = len(outcomes[i])
        # Reward = outcomes[i].count('Reward')
        # Punish = outcomes[i].count('Punish')
        # HitRate = Reward/Trials
        # print to console
        print(subject, dates[i], 'Counts')
        print('Trials:', Trials[i])
        print('Reward:', Reward[i])
        print('Punish:', Punish[i])
        print('Hit Rate:', format(HitRate[i], ".2%"))
        print()
    
         
    # update to better code later
    # save_dir = './figures/'+ subject + '/outcome'
    # save_dir_outcome = save_dir + '/outcome'
    # save_fn = 
    # os.makedirs('./figures/'+subject+'/outcome', exist_ok = True)
    # fig.savefig('./figures/'+subject+'/fig1_'+subject+'_outcome.pdf', dpi=300)
    # fig.savefig('./figures/'+subject+'/outcome/fig1_'+subject+'_outcome.png', dpi=300)
    
    # output_imgs_dir = 'C:\\data analysis\\behavior\\joystick\\figures\\'+subject+'\\' + 'outcome_imgs\\'
    
    
    # output_figs_dir = 'C:/Users/timst/OneDrive - Georgia Institute of Technology/Najafi_Lab/0_Data_analysis/Behavior/Joystick/' +subject+'\\'    
    # output_imgs_dir = 'C:/Users/timst/OneDrive - Georgia Institute of Technology/Najafi_Lab/0_Data_analysis/Behavior/Joystick/'+subject+'\\' + 'outcome_imgs\\'
    output_figs_dir = output_dir_onedrive + subject + '/'    
    output_imgs_dir = output_dir_local + subject + '/outcome_imgs/'    
    os.makedirs(output_figs_dir, exist_ok = True)
    os.makedirs(output_imgs_dir, exist_ok = True)
    fig.savefig(output_figs_dir + subject + '_Outcome.pdf', dpi=300)
    fig.savefig(output_imgs_dir + subject + '_Outcome.png', dpi=300)
    # fig.savefig(output_figs_dir + 'fig1_'+subject+'_outcome.pdf', dpi=300)
    # fig.savefig(output_imgs_dir + '\\fig1_'+subject+'_outcome.png', dpi=300)
    
    # os.makedirs('C:\\behavior\\joystick\\figures\\'+subject+'\\outcome', exist_ok = True)
    # fig.savefig('C:\\behavior\\joystick\\figures\\'+subject+'\\fig1_'+subject+'_outcome.pdf', dpi=300)
    # fig.savefig('C:\\behavior\\joystick\\figures\\'+subject+'\\outcome\\fig1_'+subject+'_outcome.png', dpi=300)
    plt.close()
    
# debugging

# session_data = session_data_1
# plot_fig1(session_data)

# session_data = session_data_2
# plot_fig1(session_data)
    
# session_data = session_data_3
# plot_fig1(session_data)

# session_data = session_data_4
# plot_fig1(session_data)