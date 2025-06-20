import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from datetime import date
from matplotlib.lines import Line2D
import re
import seaborn as sns

def run_epoch(axs , session_data , block_data):
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_raw = session_data['raw']
    
    num_session = len(block_data['NumBlocks'])
    
    
    all_lens = np.concatenate(block_data['NumTrials'][:])
    unique = set(all_lens)
    max_trial_num_temp = max(all_lens)
    max_trial_num = max_trial_num_temp
    i = 1
   
    max_trial_num = max_trial_num_temp
    
    
    max_num = num_session
    num_plotting =  num_session
    initial_start = 0
    epoch_len = 10
    short_first = []
    short_last = []
    long_first = []
    long_last = []
    
    
    for i in range(num_session):
        short_delay_first = []
        short_delay_last = []
        
        long_delay_first = []
        long_delay_last = []
        
        short_id = block_data['short_id'][i]
        long_id = block_data['long_id'][i]
        max_trial_num = np.max(block_data['NumTrials'][i])
        if short_id[0] == 0:
            short_id = short_id[1:]
        if len(long_id) > 0:
            if long_id[0] ==0:
                long_id = long_id[1:]
        for j in range(len(short_id)):
            current_block = block_data['delay'][i][short_id[j]]
            
            if len(current_block)>1.5*epoch_len:
                short_delay_first.append(np.nanmean(current_block[0:epoch_len]))
                short_delay_last.append(np.nanmean(current_block[-epoch_len:]))
                
        for j in range(len(long_id)):
            current_block = block_data['delay'][i][long_id[j]]
            if len(current_block)>1.5*epoch_len:
                long_delay_first.append(np.nanmean(current_block[0:epoch_len]))
                long_delay_last.append(np.nanmean(current_block[epoch_len:]))
                
        short_first.append(np.mean(short_delay_first))
        short_last.append(np.mean(short_delay_last))
        long_first.append(np.mean(long_delay_first))
        long_last.append(np.mean(long_delay_last))
        
    size = 3
    x_first = np.arange(0 ,len(short_first))/len(short_first)
    x_last = np.arange(0 ,len(short_last))/len(short_last)+2
    cmap = plt.cm.Greys
    norm = plt.Normalize(vmin=0, vmax=num_session+1)
    
    color1 = []
    for i in range(max(len(short_last),len(short_first))):
        color1.append(np.array(cmap(norm(i+1))))
        axs.plot([x_first[i] , x_last[i]],[short_first[i] , short_last[i]], color = np.array(cmap(norm(i+1))))
    
    axs.scatter(x_first, short_first, color = color1 , label = date , s = size)
    axs.scatter(x_last, short_last, color = color1 , s = size)
    
    x_first = np.arange(0 ,len(long_first))/len(long_first)+5
    x_last = np.arange(0 ,len(long_last))/len(long_last)+7
    cmap = plt.cm.Greys
    norm = plt.Normalize(vmin=0, vmax=num_session+1)
    color1 = []
    for i in range(max(len(long_last),len(long_first))):
        color1.append(np.array(cmap(norm(i+1))))
        axs.plot([x_first[i] , x_last[i]],[long_first[i] , long_last[i]], color = np.array(cmap(norm(i+1))))
    
    axs.scatter(x_first, long_first, color = color1, s= size)
    axs.scatter(x_last, long_last, color = color1, s= size)
    
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlabel('trial number')
    axs.set_ylabel('delay (s)')
    axs.set_title('Average Delay for first and last epoches ') 
    
   
    axs.set_xticks([0.5 , 2.5 , 5.5 , 7.5])
    axs.set_xticklabels(['Short First' , 'Short Last' , 'Long First' , 'Long Last'], rotation='vertical')
