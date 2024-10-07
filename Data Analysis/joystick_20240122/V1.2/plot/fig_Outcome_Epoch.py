import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfFileReader, PdfFileWriter, PdfWriter, PdfReader
from datetime import date
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import re
from scipy import stats
import warnings


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

def epoching_session(nTrials , Block_size):
    epoch1_block_size = int(np.ceil(Block_size/5))
    epoch2_block_size = int(np.ceil((Block_size - epoch1_block_size)/2))
    epoch3_block_size = Block_size - (epoch1_block_size + epoch2_block_size)

    # define the count of block in a session
    nBlock = int(np.ceil(nTrials/Block_size))
    epoch1 = []
    epoch2 = []
    epoch3 = []
    for i in range(0,nBlock):
        epoch1 = np.concatenate((epoch1, np.arange(i * Block_size, ((i * Block_size) + epoch1_block_size))))
        epoch2 = np.concatenate((epoch2, np.arange((i * Block_size) + epoch1_block_size , (i * Block_size) + epoch1_block_size + epoch2_block_size)))
        epoch3 = np.concatenate((epoch3, np.arange((i * Block_size) + epoch1_block_size + epoch2_block_size , (i * Block_size) + epoch1_block_size + epoch2_block_size + epoch3_block_size)))
        
    # Now we have arrays for epochs, just to drop the trials larger than nTrials

    epoch1 = epoch1[epoch1 <= nTrials]
    epoch2 = epoch2[epoch2 <= nTrials]
    epoch3 = epoch3[epoch3 <= nTrials]
    return epoch1, epoch2, epoch3

def process_matrix(matrix):
    count_ones = 0
    count_zeros = 0
    
    for row in matrix:
        for element in row:
            if element == 1:
                count_ones += 1
            elif element == 0:
                count_zeros += 1
    
    if count_ones > count_zeros:
        print("selftime")
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == 0:
                    matrix[i][j] = np.nan
    elif count_zeros > count_ones:
        print("VisGuided")
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == 1:
                    matrix[i][j] = np.nan
    else:
        print("The number of 1s and 0s is equal")
    
    return matrix

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



def plot_outcome_epoh(
        session_data,
        output_dir_onedrive, 
        output_dir_local
        ):
    max_sessions=100
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    start_idx = 0
    dates = session_data['dates']
    if max_sessions != -1 and len(dates) > max_sessions:
        start_idx = len(dates) - max_sessions
    outcomes = outcomes[start_idx:]
    session_id = np.arange(len(outcomes)) + 1

    today = date.today()
    today_formatted = str(today)[2:]
    year = today_formatted[0:2]
    month = today_formatted[3:5]
    day = today_formatted[6:]
    today_string = year + month + day
    numSessions = session_data['total_sessions']

    encoder_time_max = 100
    ms_per_s = 1000
    savefiles = 1
    subject = session_data['subject']
    outcomes = session_data['outcomes']
    dates = session_data['dates']
    session_id = np.arange(len(outcomes)) + 1
    isSelfTimedMode  = process_matrix (session_data['isSelfTimedMode'])

    chemo_labels = session_data['chemo']
    for i in range(0 , len(chemo_labels)):
            if chemo_labels[i] == 1:
                dates[i] = dates[i] + '(chemo)'

    dates = deduplicate_chemo(dates)
    numeric_dates = np.arange(len(dates))

    if any(0 in row for row in isSelfTimedMode):
        print('Visually Guided')
    else:
        print('selftime')

    #################################### POOLED ##################################
    p_epoch1_reward_chemo_s = 0
    p_epoch1_reward_opto_s = 0
    p_epoch1_reward_con_s = 0

    p_epoch1_reward_chemo_l = 0
    p_epoch1_reward_opto_l = 0
    p_epoch1_reward_con_l = 0

    p_epoch2_reward_chemo_s = 0
    p_epoch2_reward_opto_s = 0
    p_epoch2_reward_con_s = 0

    p_epoch2_reward_chemo_l = 0
    p_epoch2_reward_opto_l = 0
    p_epoch2_reward_con_l = 0

    p_epoch3_reward_chemo_s = 0
    p_epoch3_reward_opto_s = 0
    p_epoch3_reward_con_s = 0

    p_epoch3_reward_chemo_l = 0
    p_epoch3_reward_opto_l = 0
    p_epoch3_reward_con_l = 0

    p_epoch1_dnp1_chemo_s = 0
    p_epoch1_dnp1_opto_s = 0
    p_epoch1_dnp1_con_s = 0

    p_epoch1_dnp1_chemo_l = 0
    p_epoch1_dnp1_opto_l = 0 
    p_epoch1_dnp1_con_l = 0

    p_epoch2_dnp1_chemo_s = 0
    p_epoch2_dnp1_opto_s = 0
    p_epoch2_dnp1_con_s = 0

    p_epoch2_dnp1_chemo_l = 0
    p_epoch2_dnp1_opto_l = 0
    p_epoch2_dnp1_con_l = 0

    p_epoch3_dnp1_chemo_s = 0
    p_epoch3_dnp1_opto_s = 0
    p_epoch3_dnp1_con_s = 0

    p_epoch3_dnp1_chemo_l = 0
    p_epoch3_dnp1_opto_l = 0
    p_epoch3_dnp1_con_l = 0

    p_epoch1_dnp2_chemo_s = 0
    p_epoch1_dnp2_opto_s = 0
    p_epoch1_dnp2_con_s = 0

    p_epoch1_dnp2_chemo_l = 0
    p_epoch1_dnp2_opto_l = 0 
    p_epoch1_dnp2_con_l = 0

    p_epoch2_dnp2_chemo_s = 0
    p_epoch2_dnp2_opto_s = 0
    p_epoch2_dnp2_con_s = 0

    p_epoch2_dnp2_chemo_l = 0
    p_epoch2_dnp2_opto_l = 0
    p_epoch2_dnp2_con_l = 0

    p_epoch3_dnp2_chemo_s = 0
    p_epoch3_dnp2_opto_s = 0
    p_epoch3_dnp2_con_s = 0

    p_epoch3_dnp2_chemo_l = 0
    p_epoch3_dnp2_opto_l = 0
    p_epoch3_dnp2_con_l = 0

    p_epoch1_ep_chemo_s = 0
    p_epoch1_ep_opto_s = 0
    p_epoch1_ep_con_s = 0

    p_epoch1_ep_chemo_l = 0
    p_epoch1_ep_opto_l = 0
    p_epoch1_ep_con_l = 0

    p_epoch2_ep_chemo_s = 0
    p_epoch2_ep_opto_s = 0
    p_epoch2_ep_con_s = 0

    p_epoch2_ep_chemo_l = 0
    p_epoch2_ep_opto_l = 0
    p_epoch2_ep_con_l = 0

    p_epoch3_ep_chemo_s = 0
    p_epoch3_ep_opto_s = 0
    p_epoch3_ep_con_s = 0

    p_epoch3_ep_chemo_l = 0
    p_epoch3_ep_opto_l = 0
    p_epoch3_ep_con_l = 0

    p_epoch1_ep1_chemo_s = 0
    p_epoch1_ep1_opto_s = 0
    p_epoch1_ep1_con_s = 0

    p_epoch1_ep1_chemo_l = 0
    p_epoch1_ep1_opto_l = 0
    p_epoch1_ep1_con_l = 0

    p_epoch2_ep1_chemo_s = 0
    p_epoch2_ep1_opto_s = 0
    p_epoch2_ep1_con_s = 0

    p_epoch2_ep1_chemo_l = 0
    p_epoch2_ep1_opto_l = 0
    p_epoch2_ep1_con_l = 0

    p_epoch3_ep1_chemo_s = 0
    p_epoch3_ep1_opto_s = 0
    p_epoch3_ep1_con_s = 0

    p_epoch3_ep1_chemo_l = 0
    p_epoch3_ep1_opto_l = 0
    p_epoch3_ep1_con_l = 0

    p_epoch1_ep2_chemo_s = 0
    p_epoch1_ep2_opto_s = 0
    p_epoch1_ep2_con_s = 0

    p_epoch1_ep2_chemo_l = 0
    p_epoch1_ep2_opto_l = 0
    p_epoch1_ep2_con_l = 0

    p_epoch2_ep2_chemo_s = 0
    p_epoch2_ep2_opto_s = 0
    p_epoch2_ep2_con_s = 0

    p_epoch2_ep2_chemo_l = 0
    p_epoch2_ep2_opto_l = 0
    p_epoch2_ep2_con_l = 0

    p_epoch3_ep2_chemo_s = 0
    p_epoch3_ep2_opto_s = 0
    p_epoch3_ep2_con_s = 0

    p_epoch3_ep2_chemo_l = 0
    p_epoch3_ep2_opto_l = 0
    p_epoch3_ep2_con_l = 0
    
    p_epoch1_int1_chemo_s = 0
    p_epoch1_int1_opto_s = 0
    p_epoch1_int1_con_s = 0

    p_epoch1_int1_chemo_l = 0
    p_epoch1_int1_opto_l = 0
    p_epoch1_int1_con_l = 0

    p_epoch2_int1_chemo_s = 0
    p_epoch2_int1_opto_s = 0
    p_epoch2_int1_con_s = 0

    p_epoch2_int1_chemo_l = 0
    p_epoch2_int1_opto_l = 0
    p_epoch2_int1_con_l = 0

    p_epoch3_int1_chemo_s = 0
    p_epoch3_int1_opto_s = 0
    p_epoch3_int1_con_s = 0

    p_epoch3_int1_chemo_l = 0
    p_epoch3_int1_opto_l = 0
    p_epoch3_int1_con_l = 0
    
    p_epoch1_int2_chemo_s = 0
    p_epoch1_int2_opto_s = 0
    p_epoch1_int2_con_s = 0

    p_epoch1_int2_chemo_l = 0
    p_epoch1_int2_opto_l = 0
    p_epoch1_int2_con_l = 0

    p_epoch2_int2_chemo_s = 0
    p_epoch2_int2_opto_s = 0
    p_epoch2_int2_con_s = 0

    p_epoch2_int2_chemo_l = 0
    p_epoch2_int2_opto_l = 0
    p_epoch2_int2_con_l = 0

    p_epoch3_int2_chemo_s = 0
    p_epoch3_int2_opto_s = 0
    p_epoch3_int2_con_s = 0

    p_epoch3_int2_chemo_l = 0
    p_epoch3_int2_opto_l = 0
    p_epoch3_int2_con_l = 0
    
    p_epoch1_int3_chemo_s = 0
    p_epoch1_int3_opto_s = 0
    p_epoch1_int3_con_s = 0

    p_epoch1_int3_chemo_l = 0
    p_epoch1_int3_opto_l = 0
    p_epoch1_int3_con_l = 0

    p_epoch2_int3_chemo_s = 0
    p_epoch2_int3_opto_s = 0
    p_epoch2_int3_con_s = 0

    p_epoch2_int3_chemo_l = 0
    p_epoch2_int3_opto_l = 0
    p_epoch2_int3_con_l = 0

    p_epoch3_int3_chemo_s = 0
    p_epoch3_int3_opto_s = 0
    p_epoch3_int3_con_s = 0

    p_epoch3_int3_chemo_l = 0
    p_epoch3_int3_opto_l = 0
    p_epoch3_int3_con_l = 0
    
    p_epoch1_int4_chemo_s = 0
    p_epoch1_int4_opto_s = 0
    p_epoch1_int4_con_s = 0

    p_epoch1_int4_chemo_l = 0
    p_epoch1_int4_opto_l = 0
    p_epoch1_int4_con_l = 0

    p_epoch2_int4_chemo_s = 0
    p_epoch2_int4_opto_s = 0
    p_epoch2_int4_con_s = 0

    p_epoch2_int4_chemo_l = 0
    p_epoch2_int4_opto_l = 0
    p_epoch2_int4_con_l = 0

    p_epoch3_int4_chemo_s = 0
    p_epoch3_int4_opto_s = 0
    p_epoch3_int4_con_s = 0

    p_epoch3_int4_chemo_l = 0
    p_epoch3_int4_opto_l = 0
    p_epoch3_int4_con_l = 0

    p_epoch1_int_chemo_s = 0
    p_epoch1_int_opto_s = 0
    p_epoch1_int_con_s = 0

    p_epoch1_int_chemo_l = 0
    p_epoch1_int_opto_l = 0
    p_epoch1_int_con_l = 0

    p_epoch2_int_chemo_s = 0
    p_epoch2_int_opto_s = 0
    p_epoch2_int_con_s = 0

    p_epoch2_int_chemo_l = 0
    p_epoch2_int_opto_l = 0
    p_epoch2_int_con_l = 0

    p_epoch3_int_chemo_s = 0
    p_epoch3_int_opto_s = 0
    p_epoch3_int_con_s = 0

    p_epoch3_int_chemo_l = 0
    p_epoch3_int_opto_l = 0
    p_epoch3_int_con_l = 0
    ##################################### Grand ############################
    grand_short_epoch1 = []
    grand_short_epoch2 = []
    grand_short_epoch3 = []

    grand_long_epoch1 = []
    grand_long_epoch2 = []
    grand_long_epoch3 = []

    # Plotting
    top_ticks_mega = []
    for i in range(0, len(session_id)):
        top_ticks_mega.append('ep1|ep2|ep3')
    fig1, axs1 = plt.subplots(nrows=2,figsize=(7 + 1.2 *len(session_id) , 12))
    fig1.suptitle('Reward percentage for each session ' +  subject)
    tick_index = (np.arange(len(session_id)) + 1)
    width = 0.8
    axs1[0].set_xticks(tick_index + width/3)
    secax = axs1[0].secondary_xaxis('top')
    secax.set_xticks(tick_index + width/3)
    secax.set_xticklabels(top_ticks_mega)
    axs1[0].set_title('epoch analysis for Short trials' + '\n')

    axs1[1].set_xticks(tick_index + width/3)
    secax = axs1[1].secondary_xaxis('top')
    secax.set_xticks(tick_index + width/3)
    secax.set_xticklabels(top_ticks_mega)
    axs1[1].set_title('epoch analysis for Long trials' + '\n')
    ind = 0
    dates_label = dates
    axs1[0].set_xticklabels(dates_label, rotation='vertical')
        
    for xtick in axs1[0].get_xticklabels():
        if 'chemo' in dates_label[ind]:
            xtick.set_color('r')        
        else:
            xtick.set_color('k')
        ind = ind + 1
            
    axs1[1].set_xticklabels(dates_label, rotation='vertical')
    ind = 0
    for xtick in axs1[1].get_xticklabels():
        if 'chemo' in dates_label[ind]:
            xtick.set_color('r')        
        else:
            xtick.set_color('k')
        ind = ind + 1
    # Starting loop for analyzing sessions
    for i in range(0 , len(session_id)):
        
        # Variables
        epoch1_reward_s = 0
        epoch2_reward_s = 0
        epoch3_reward_s = 0
        
        epoch1_reward_l = 0
        epoch2_reward_l = 0
        epoch3_reward_l = 0
        
        epoch1_dnp1_s = 0
        epoch2_dnp1_s = 0 
        epoch3_dnp1_s = 0
        
        epoch1_dnp1_l = 0
        epoch2_dnp1_l = 0 
        epoch3_dnp1_l = 0
        
        epoch1_dnp2_s = 0
        epoch2_dnp2_s = 0 
        epoch3_dnp2_s = 0
        
        epoch1_dnp2_l = 0
        epoch2_dnp2_l = 0 
        epoch3_dnp2_l = 0
        
        epoch1_ep_s = 0 
        epoch2_ep_s = 0
        epoch3_ep_s= 0
        
        epoch1_ep_l = 0 
        epoch2_ep_l = 0
        epoch3_ep_l= 0
        
        epoch1_ep1_s = 0 
        epoch2_ep1_s = 0
        epoch3_ep1_s= 0
        
        epoch1_ep1_l = 0 
        epoch2_ep1_l = 0
        epoch3_ep1_l= 0
        
        epoch1_ep2_s = 0 
        epoch2_ep2_s = 0
        epoch3_ep2_s= 0
        
        epoch1_ep2_l = 0 
        epoch2_ep2_l = 0
        epoch3_ep2_l= 0
        
        epoch1_int1_s = 0 
        epoch2_int1_s = 0
        epoch3_int1_s= 0
        
        epoch1_int1_l = 0 
        epoch2_int1_l = 0
        epoch3_int1_l= 0
        
        epoch1_int2_s = 0 
        epoch2_int2_s = 0
        epoch3_int2_s= 0
        
        epoch1_int2_l = 0 
        epoch2_int2_l = 0
        epoch3_int2_l= 0
        
        epoch1_int3_s = 0 
        epoch2_int3_s = 0
        epoch3_int3_s= 0
        
        epoch1_int3_l = 0 
        epoch2_int3_l = 0
        epoch3_int3_l= 0
        
        epoch1_int4_s = 0 
        epoch2_int4_s = 0
        epoch3_int4_s= 0
        
        epoch1_int4_l = 0 
        epoch2_int4_l = 0
        epoch3_int4_l= 0
        
        epoch1_int_s = 0 
        epoch2_int_s = 0
        epoch3_int_s= 0
        
        epoch1_int_l = 0 
        epoch2_int_l = 0
        epoch3_int_l= 0
        # Grand: ###############################
        G_epoch1_reward_chemo_s = 0
        G_epoch1_reward_opto_s = 0
        G_epoch1_reward_con_s = 0

        G_epoch1_reward_chemo_l = 0
        G_epoch1_reward_opto_l = 0
        G_epoch1_reward_con_l = 0

        G_epoch2_reward_chemo_s = 0
        G_epoch2_reward_opto_s = 0
        G_epoch2_reward_con_s = 0

        G_epoch2_reward_chemo_l = 0
        G_epoch2_reward_opto_l = 0
        G_epoch2_reward_con_l = 0

        G_epoch3_reward_chemo_s = 0
        G_epoch3_reward_opto_s = 0
        G_epoch3_reward_con_s = 0

        G_epoch3_reward_chemo_l = 0
        G_epoch3_reward_opto_l = 0
        G_epoch3_reward_con_l = 0
        
        G_epoch1_dnp1_chemo_s = 0
        G_epoch1_dnp1_opto_s = 0
        G_epoch1_dnp1_con_s = 0

        G_epoch1_dnp1_chemo_l = 0
        G_epoch1_dnp1_opto_l = 0 
        G_epoch1_dnp1_con_l = 0

        G_epoch2_dnp1_chemo_s = 0
        G_epoch2_dnp1_opto_s = 0
        G_epoch2_dnp1_con_s = 0

        G_epoch2_dnp1_chemo_l = 0
        G_epoch2_dnp1_opto_l = 0
        G_epoch2_dnp1_con_l = 0

        G_epoch3_dnp1_chemo_s = 0
        G_epoch3_dnp1_opto_s = 0
        G_epoch3_dnp1_con_s = 0

        G_epoch3_dnp1_chemo_l = 0
        G_epoch3_dnp1_opto_l = 0
        G_epoch3_dnp1_con_l = 0
        
        G_epoch1_dnp2_chemo_s = 0
        G_epoch1_dnp2_opto_s = 0
        G_epoch1_dnp2_con_s = 0

        G_epoch1_dnp2_chemo_l = 0
        G_epoch1_dnp2_opto_l = 0 
        G_epoch1_dnp2_con_l = 0

        G_epoch2_dnp2_chemo_s = 0
        G_epoch2_dnp2_opto_s = 0
        G_epoch2_dnp2_con_s = 0

        G_epoch2_dnp2_chemo_l = 0
        G_epoch2_dnp2_opto_l = 0
        G_epoch2_dnp2_con_l = 0

        G_epoch3_dnp2_chemo_s = 0
        G_epoch3_dnp2_opto_s = 0
        G_epoch3_dnp2_con_s = 0

        G_epoch3_dnp2_chemo_l = 0
        G_epoch3_dnp2_opto_l = 0
        G_epoch3_dnp2_con_l = 0
        
        G_epoch1_ep_chemo_s = 0
        G_epoch1_ep_opto_s = 0
        G_epoch1_ep_con_s = 0

        G_epoch1_ep_chemo_l = 0
        G_epoch1_ep_opto_l = 0
        G_epoch1_ep_con_l = 0

        G_epoch2_ep_chemo_s = 0
        G_epoch2_ep_opto_s = 0
        G_epoch2_ep_con_s = 0

        G_epoch2_ep_chemo_l = 0
        G_epoch2_ep_opto_l = 0
        G_epoch2_ep_con_l = 0

        G_epoch3_ep_chemo_s = 0
        G_epoch3_ep_opto_s = 0
        G_epoch3_ep_con_s = 0

        G_epoch3_ep_chemo_l = 0
        G_epoch3_ep_opto_l = 0
        G_epoch3_ep_con_l = 0
        
        G_epoch1_ep1_chemo_s = 0
        G_epoch1_ep1_opto_s = 0
        G_epoch1_ep1_con_s = 0

        G_epoch1_ep1_chemo_l = 0
        G_epoch1_ep1_opto_l = 0
        G_epoch1_ep1_con_l = 0

        G_epoch2_ep1_chemo_s = 0
        G_epoch2_ep1_opto_s = 0
        G_epoch2_ep1_con_s = 0

        G_epoch2_ep1_chemo_l = 0
        G_epoch2_ep1_opto_l = 0
        G_epoch2_ep1_con_l = 0

        G_epoch3_ep1_chemo_s = 0
        G_epoch3_ep1_opto_s = 0
        G_epoch3_ep1_con_s = 0

        G_epoch3_ep1_chemo_l = 0
        G_epoch3_ep1_opto_l = 0
        G_epoch3_ep1_con_l = 0
        
        G_epoch1_ep2_chemo_s = 0
        G_epoch1_ep2_opto_s = 0
        G_epoch1_ep2_con_s = 0

        G_epoch1_ep2_chemo_l = 0
        G_epoch1_ep2_opto_l = 0
        G_epoch1_ep2_con_l = 0

        G_epoch2_ep2_chemo_s = 0
        G_epoch2_ep2_opto_s = 0
        G_epoch2_ep2_con_s = 0

        G_epoch2_ep2_chemo_l = 0
        G_epoch2_ep2_opto_l = 0
        G_epoch2_ep2_con_l = 0

        G_epoch3_ep2_chemo_s = 0
        G_epoch3_ep2_opto_s = 0
        G_epoch3_ep2_con_s = 0

        G_epoch3_ep2_chemo_l = 0
        G_epoch3_ep2_opto_l = 0
        G_epoch3_ep2_con_l = 0
        
        G_epoch1_int1_chemo_s = 0
        G_epoch1_int1_opto_s = 0
        G_epoch1_int1_con_s = 0

        G_epoch1_int1_chemo_l = 0
        G_epoch1_int1_opto_l = 0
        G_epoch1_int1_con_l = 0

        G_epoch2_int1_chemo_s = 0
        G_epoch2_int1_opto_s = 0
        G_epoch2_int1_con_s = 0

        G_epoch2_int1_chemo_l = 0
        G_epoch2_int1_opto_l = 0
        G_epoch2_int1_con_l = 0

        G_epoch3_int1_chemo_s = 0
        G_epoch3_int1_opto_s = 0
        G_epoch3_int1_con_s = 0

        G_epoch3_int1_chemo_l = 0
        G_epoch3_int1_opto_l = 0
        G_epoch3_int1_con_l = 0
        
        G_epoch1_int2_chemo_s = 0
        G_epoch1_int2_opto_s = 0
        G_epoch1_int2_con_s = 0

        G_epoch1_int2_chemo_l = 0
        G_epoch1_int2_opto_l = 0
        G_epoch1_int2_con_l = 0

        G_epoch2_int2_chemo_s = 0
        G_epoch2_int2_opto_s = 0
        G_epoch2_int2_con_s = 0

        G_epoch2_int2_chemo_l = 0
        G_epoch2_int2_opto_l = 0
        G_epoch2_int2_con_l = 0

        G_epoch3_int2_chemo_s = 0
        G_epoch3_int2_opto_s = 0
        G_epoch3_int2_con_s = 0

        G_epoch3_int2_chemo_l = 0
        G_epoch3_int2_opto_l = 0
        G_epoch3_int2_con_l = 0
        
        G_epoch1_int3_chemo_s = 0
        G_epoch1_int3_opto_s = 0
        G_epoch1_int3_con_s = 0

        G_epoch1_int3_chemo_l = 0
        G_epoch1_int3_opto_l = 0
        G_epoch1_int3_con_l = 0

        G_epoch2_int3_chemo_s = 0
        G_epoch2_int3_opto_s = 0
        G_epoch2_int3_con_s = 0

        G_epoch2_int3_chemo_l = 0
        G_epoch2_int3_opto_l = 0
        G_epoch2_int3_con_l = 0

        G_epoch3_int3_chemo_s = 0
        G_epoch3_int3_opto_s = 0
        G_epoch3_int3_con_s = 0

        G_epoch3_int3_chemo_l = 0
        G_epoch3_int3_opto_l = 0
        G_epoch3_int3_con_l = 0
        
        G_epoch1_int4_chemo_s = 0
        G_epoch1_int4_opto_s = 0
        G_epoch1_int4_con_s = 0

        G_epoch1_int4_chemo_l = 0
        G_epoch1_int4_opto_l = 0
        G_epoch1_int4_con_l = 0

        G_epoch2_int4_chemo_s = 0
        G_epoch2_int4_opto_s = 0
        G_epoch2_int4_con_s = 0

        G_epoch2_int4_chemo_l = 0
        G_epoch2_int4_opto_l = 0
        G_epoch2_int4_con_l = 0

        G_epoch3_int4_chemo_s = 0
        G_epoch3_int4_opto_s = 0
        G_epoch3_int4_con_s = 0

        G_epoch3_int4_chemo_l = 0
        G_epoch3_int4_opto_l = 0
        G_epoch3_int4_con_l = 0
        
        G_epoch1_int_chemo_s = 0
        G_epoch1_int_opto_s = 0
        G_epoch1_int_con_s = 0

        G_epoch1_int_chemo_l = 0
        G_epoch1_int_opto_l = 0
        G_epoch1_int_con_l = 0

        G_epoch2_int_chemo_s = 0
        G_epoch2_int_opto_s = 0
        G_epoch2_int_con_s = 0

        G_epoch2_int_chemo_l = 0
        G_epoch2_int_opto_l = 0
        G_epoch2_int_con_l = 0

        G_epoch3_int_chemo_s = 0
        G_epoch3_int_opto_s = 0
        G_epoch3_int_con_s = 0

        G_epoch3_int_chemo_l = 0
        G_epoch3_int_opto_l = 0
        G_epoch3_int_con_l = 0
        
        TrialOutcomes = session_data['outcomes'][i]
        # We have Raw data and extract every thing from it (Times)
        raw_data = session_data['raw'][i]
        session_date = dates[i][2:]
        trial_types = raw_data['TrialTypes']
        # iswarmup = raw_data['IsWarmupTrial']
        opto = session_data['session_opto_tag'][i]
        print('Epoch based analysis:' + session_date)
        
        Block_size = session_data['raw'][i]['TrialSettings'][0]['GUI']['NumTrialsPerBlock']
        nTrials = session_data['raw'][i]['nTrials']
        epoch1 , epoch2 , epoch3 = epoching_session(nTrials, Block_size)
        
        if nTrials <= Block_size:
            print('not adequate trials for analysis in session:',i)
            continue
        
        # NOTE: we pass the first block
        for trial in range(Block_size ,len(TrialOutcomes)):
            
            if np.isnan(isSelfTimedMode[i][trial]):
                continue
            
            # Passing warmup trials 
            # if iswarmup == 1:
            #     continue
            
            if TrialOutcomes[trial] == 'Reward':
                if trial in epoch1:
                    if trial_types[trial] == 1:
                        epoch1_reward_s = epoch1_reward_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_reward_chemo_s = p_epoch1_reward_chemo_s + 1
                            G_epoch1_reward_chemo_s = G_epoch1_reward_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch1_reward_opto_s = p_epoch1_reward_opto_s + 1
                            G_epoch1_reward_opto_s = G_epoch1_reward_opto_s + 1
                        else:
                            p_epoch1_reward_con_s = p_epoch1_reward_con_s + 1
                            G_epoch1_reward_con_s = G_epoch1_reward_con_s + 1
                    else:
                        epoch1_reward_l = epoch1_reward_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_reward_chemo_l = p_epoch1_reward_chemo_l + 1
                            G_epoch1_reward_chemo_l = G_epoch1_reward_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch1_reward_opto_l = p_epoch1_reward_opto_l + 1
                            G_epoch1_reward_opto_l = G_epoch1_reward_opto_l + 1
                        else:
                            p_epoch1_reward_con_l = p_epoch1_reward_con_l + 1
                            G_epoch1_reward_con_l = G_epoch1_reward_con_l + 1
                    
                elif trial in epoch2:
                    if trial_types[trial] == 1:
                        epoch2_reward_s = epoch2_reward_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_reward_chemo_s = p_epoch2_reward_chemo_s + 1
                            G_epoch2_reward_chemo_s = G_epoch2_reward_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch2_reward_opto_s = p_epoch2_reward_opto_s + 1
                            G_epoch2_reward_opto_s = G_epoch2_reward_opto_s + 1
                        else:
                            p_epoch2_reward_con_s = p_epoch2_reward_con_s + 1
                            G_epoch2_reward_con_s = G_epoch2_reward_con_s + 1
                    else:
                        epoch2_reward_l = epoch2_reward_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_reward_chemo_l = p_epoch2_reward_chemo_l + 1
                            G_epoch2_reward_chemo_l = G_epoch2_reward_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch2_reward_opto_l = p_epoch2_reward_opto_l + 1
                            G_epoch2_reward_opto_l = G_epoch2_reward_opto_l + 1
                        else:
                            p_epoch2_reward_con_l = p_epoch2_reward_con_l + 1
                            G_epoch2_reward_con_l = G_epoch2_reward_con_l + 1
                            
                elif trial in epoch3:
                    if trial_types[trial] == 1:
                        epoch3_reward_s = epoch3_reward_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_reward_chemo_s = p_epoch3_reward_chemo_s + 1
                            G_epoch3_reward_chemo_s = G_epoch3_reward_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch3_reward_opto_s = p_epoch3_reward_opto_s + 1
                            G_epoch3_reward_opto_s = G_epoch3_reward_opto_s + 1
                        else:
                            p_epoch3_reward_con_s = p_epoch3_reward_con_s + 1
                            G_epoch3_reward_con_s = G_epoch3_reward_con_s + 1
                    else:
                        epoch3_reward_l = epoch3_reward_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_reward_chemo_l = p_epoch3_reward_chemo_l + 1
                            G_epoch3_reward_chemo_l = G_epoch3_reward_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch3_reward_opto_l = p_epoch3_reward_opto_l + 1
                            G_epoch3_reward_opto_l = G_epoch3_reward_opto_l + 1
                        else:
                            p_epoch3_reward_con_l = p_epoch3_reward_con_l + 1
                            G_epoch3_reward_con_l = G_epoch3_reward_con_l + 1
            
            elif TrialOutcomes[trial] == 'DidNotPress1':
                if trial in epoch1:
                    if trial_types[trial] == 1:
                        epoch1_dnp1_s = epoch1_dnp1_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_dnp1_chemo_s = p_epoch1_dnp1_chemo_s + 1
                            G_epoch1_dnp1_chemo_s = G_epoch1_dnp1_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch1_dnp1_opto_s = p_epoch1_dnp1_opto_s + 1
                            G_epoch1_dnp1_opto_s = G_epoch1_dnp1_opto_s + 1
                        else:
                            p_epoch1_dnp1_con_s = p_epoch1_dnp1_con_s + 1
                            G_epoch1_dnp1_con_s = G_epoch1_dnp1_con_s + 1
                    else:
                        epoch1_dnp1_l = epoch1_dnp1_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_dnp1_chemo_l = p_epoch1_dnp1_chemo_l + 1
                            G_epoch1_dnp1_chemo_l = G_epoch1_dnp1_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch1_dnp1_opto_l = p_epoch1_dnp1_opto_l + 1
                            G_epoch1_dnp1_opto_l = G_epoch1_dnp1_opto_l + 1
                        else:
                            p_epoch1_dnp1_con_l = p_epoch1_dnp1_con_l + 1
                            G_epoch1_dnp1_con_l = G_epoch1_dnp1_con_l + 1
                    
                elif trial in epoch2:
                    if trial_types[trial] == 1:
                        epoch2_dnp1_s = epoch2_dnp1_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_dnp1_chemo_s = p_epoch2_dnp1_chemo_s + 1
                            G_epoch2_dnp1_chemo_s = G_epoch2_dnp1_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch2_dnp1_opto_s = p_epoch2_dnp1_opto_s + 1
                            G_epoch2_dnp1_opto_s = G_epoch2_dnp1_opto_s + 1
                        else:
                            p_epoch2_dnp1_con_s = p_epoch2_dnp1_con_s + 1
                            G_epoch2_dnp1_con_s = G_epoch2_dnp1_con_s + 1
                    else:
                        epoch2_dnp1_l = epoch2_dnp1_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_dnp1_chemo_l = p_epoch2_dnp1_chemo_l + 1
                            G_epoch2_dnp1_chemo_l = G_epoch2_dnp1_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch2_dnp1_opto_l = p_epoch2_dnp1_opto_l + 1
                            G_epoch2_dnp1_opto_l = G_epoch2_dnp1_opto_l + 1
                        else:
                            p_epoch2_dnp1_con_l = p_epoch2_dnp1_con_l + 1
                            G_epoch2_dnp1_con_l = G_epoch2_dnp1_con_l + 1
                            
                elif trial in epoch3:
                    if trial_types[trial] == 1:
                        epoch3_dnp1_s = epoch3_dnp1_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_dnp1_chemo_s = p_epoch3_dnp1_chemo_s + 1
                            G_epoch3_dnp1_chemo_s = G_epoch3_dnp1_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch3_dnp1_opto_s = p_epoch3_dnp1_opto_s + 1
                            G_epoch3_dnp1_opto_s = G_epoch3_dnp1_opto_s + 1
                        else:
                            p_epoch3_dnp1_con_s = p_epoch3_dnp1_con_s + 1
                            G_epoch3_dnp1_con_s = G_epoch3_dnp1_con_s + 1
                    else:
                        epoch3_dnp1_l = epoch3_dnp1_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_dnp1_chemo_l = p_epoch3_dnp1_chemo_l + 1
                            G_epoch3_dnp1_chemo_l = G_epoch3_dnp1_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch3_dnp1_opto_l = p_epoch3_dnp1_opto_l + 1
                            G_epoch3_dnp1_opto_l = G_epoch3_dnp1_opto_l + 1
                        else:
                            p_epoch3_dnp1_con_l = p_epoch3_dnp1_con_l + 1
                            G_epoch3_dnp1_con_l = G_epoch3_dnp1_con_l + 1
                            
            elif TrialOutcomes[trial] == 'DidNotPress2':
                if trial in epoch1:
                    if trial_types[trial] == 1:
                        epoch1_dnp2_s = epoch1_dnp2_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_dnp2_chemo_s = p_epoch1_dnp2_chemo_s + 1
                            G_epoch1_dnp2_chemo_s = G_epoch1_dnp2_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch1_dnp2_opto_s = p_epoch1_dnp2_opto_s + 1
                            G_epoch1_dnp2_opto_s = G_epoch1_dnp2_opto_s + 1
                        else:
                            p_epoch1_dnp2_con_s = p_epoch1_dnp2_con_s + 1
                            G_epoch1_dnp2_con_s = G_epoch1_dnp2_con_s + 1
                    else:
                        epoch1_dnp2_l = epoch1_dnp2_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_dnp2_chemo_l = p_epoch1_dnp2_chemo_l + 1
                            G_epoch1_dnp2_chemo_l = G_epoch1_dnp2_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch1_dnp2_opto_l = p_epoch1_dnp2_opto_l + 1
                            G_epoch1_dnp2_opto_l = G_epoch1_dnp2_opto_l + 1
                        else:
                            p_epoch1_dnp2_con_l = p_epoch1_dnp2_con_l + 1
                            G_epoch1_dnp2_con_l = G_epoch1_dnp2_con_l + 1
                    
                elif trial in epoch2:
                    if trial_types[trial] == 1:
                        epoch2_dnp2_s = epoch2_dnp2_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_dnp2_chemo_s = p_epoch2_dnp2_chemo_s + 1
                            G_epoch2_dnp2_chemo_s = G_epoch2_dnp2_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch2_dnp2_opto_s = p_epoch2_dnp2_opto_s + 1
                            G_epoch2_dnp2_opto_s = G_epoch2_dnp2_opto_s + 1
                        else:
                            p_epoch2_dnp2_con_s = p_epoch2_dnp2_con_s + 1
                            G_epoch2_dnp2_con_s = G_epoch2_dnp2_con_s + 1
                    else:
                        epoch2_dnp2_l = epoch2_dnp2_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_dnp2_chemo_l = p_epoch2_dnp2_chemo_l + 1
                            G_epoch2_dnp2_chemo_l = G_epoch2_dnp2_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch2_dnp2_opto_l = p_epoch2_dnp2_opto_l + 1
                            G_epoch2_dnp2_opto_l = G_epoch2_dnp2_opto_l + 1
                        else:
                            p_epoch2_dnp2_con_l = p_epoch2_dnp2_con_l + 1
                            G_epoch2_dnp2_con_l = G_epoch2_dnp2_con_l + 1
                            
                elif trial in epoch3:
                    if trial_types[trial] == 1:
                        epoch3_dnp2_s = epoch3_dnp2_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_dnp2_chemo_s = p_epoch3_dnp2_chemo_s + 1
                            G_epoch3_dnp2_chemo_s = G_epoch3_dnp2_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch3_dnp2_opto_s = p_epoch3_dnp2_opto_s + 1
                            G_epoch3_dnp2_opto_s = G_epoch3_dnp2_opto_s + 1
                        else:
                            p_epoch3_dnp2_con_s = p_epoch3_dnp2_con_s + 1
                            G_epoch3_dnp2_con_s = G_epoch3_dnp2_con_s + 1
                    else:
                        epoch3_dnp2_l = epoch3_dnp2_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_dnp2_chemo_l = p_epoch3_dnp2_chemo_l + 1
                            G_epoch3_dnp2_chemo_l = G_epoch3_dnp2_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch3_dnp2_opto_l = p_epoch3_dnp2_opto_l + 1
                            G_epoch3_dnp2_opto_l = G_epoch3_dnp2_opto_l + 1
                        else:
                            p_epoch3_dnp2_con_l = p_epoch3_dnp2_con_l + 1
                            G_epoch3_dnp2_con_l = G_epoch3_dnp2_con_l + 1            
                            
            elif TrialOutcomes[trial] == 'EarlyPress':
                if trial in epoch1:
                    if trial_types[trial] == 1:
                        epoch1_ep_s = epoch1_ep_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_ep_chemo_s = p_epoch1_ep_chemo_s + 1
                            G_epoch1_ep_chemo_s = G_epoch1_ep_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch1_ep_opto_s = p_epoch1_ep_opto_s + 1
                            G_epoch1_ep_opto_s = G_epoch1_ep_opto_s + 1
                        else:
                            p_epoch1_ep_con_s = p_epoch1_ep_con_s + 1
                            G_epoch1_ep_con_s = G_epoch1_ep_con_s + 1
                    else:
                        epoch1_ep_l = epoch1_ep_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_ep_chemo_l = p_epoch1_ep_chemo_l + 1
                            G_epoch1_ep_chemo_l = G_epoch1_ep_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch1_ep_opto_l = p_epoch1_ep_opto_l + 1
                            G_epoch1_ep_opto_l = G_epoch1_ep_opto_l + 1
                        else:
                            p_epoch1_ep_con_l = p_epoch1_ep_con_l + 1
                            G_epoch1_ep_con_l = G_epoch1_ep_con_l + 1
                    
                elif trial in epoch2:
                    if trial_types[trial] == 1:
                        epoch2_ep_s = epoch2_ep_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_ep_chemo_s = p_epoch2_ep_chemo_s + 1
                            G_epoch2_ep_chemo_s = G_epoch2_ep_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch2_ep_opto_s = p_epoch2_ep_opto_s + 1
                            G_epoch2_ep_opto_s = G_epoch2_ep_opto_s + 1
                        else:
                            p_epoch2_ep_con_s = p_epoch2_ep_con_s + 1
                            G_epoch2_ep_con_s = G_epoch2_ep_con_s + 1
                    else:
                        epoch2_ep_l = epoch2_ep_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_ep_chemo_l = p_epoch2_ep_chemo_l + 1
                            G_epoch2_ep_chemo_l = G_epoch2_ep_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch2_ep_opto_l = p_epoch2_ep_opto_l + 1
                            G_epoch2_ep_opto_l = G_epoch2_ep_opto_l + 1
                        else:
                            p_epoch2_ep_con_l = p_epoch2_ep_con_l + 1
                            G_epoch2_ep_con_l = G_epoch2_ep_con_l + 1
                            
                elif trial in epoch3:
                    if trial_types[trial] == 1:
                        epoch3_ep_s = epoch3_ep_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_ep_chemo_s = p_epoch3_ep_chemo_s + 1
                            G_epoch3_ep_chemo_s = G_epoch3_ep_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch3_ep_opto_s = p_epoch3_ep_opto_s + 1
                            G_epoch3_ep_opto_s = G_epoch3_ep_opto_s + 1
                        else:
                            p_epoch3_ep_con_s = p_epoch3_ep_con_s + 1
                            G_epoch3_ep_con_s = G_epoch3_ep_con_s + 1
                    else:
                        epoch3_ep_l = epoch3_ep_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_ep_chemo_l = p_epoch3_ep_chemo_l + 1
                            G_epoch3_ep_chemo_l = G_epoch3_ep_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch3_ep_opto_l = p_epoch3_ep_opto_l + 1
                            G_epoch3_ep_opto_l = G_epoch3_ep_opto_l + 1
                        else:
                            p_epoch3_ep_con_l = p_epoch3_ep_con_l + 1
                            G_epoch3_ep_con_l = G_epoch3_ep_con_l + 1            

            elif TrialOutcomes[trial] == 'EarlyPress1':
                if trial in epoch1:
                    if trial_types[trial] == 1:
                        epoch1_ep1_s = epoch1_ep1_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_ep1_chemo_s = p_epoch1_ep1_chemo_s + 1
                            G_epoch1_ep1_chemo_s = G_epoch1_ep1_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch1_ep1_opto_s = p_epoch1_ep1_opto_s + 1
                            G_epoch1_ep1_opto_s = G_epoch1_ep1_opto_s + 1
                        else:
                            p_epoch1_ep1_con_s = p_epoch1_ep1_con_s + 1
                            G_epoch1_ep1_con_s = G_epoch1_ep1_con_s + 1
                    else:
                        epoch1_ep1_l = epoch1_ep1_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_ep1_chemo_l = p_epoch1_ep1_chemo_l + 1
                            G_epoch1_ep1_chemo_l = G_epoch1_ep1_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch1_ep1_opto_l = p_epoch1_ep1_opto_l + 1
                            G_epoch1_ep1_opto_l = G_epoch1_ep1_opto_l + 1
                        else:
                            p_epoch1_ep1_con_l = p_epoch1_ep1_con_l + 1
                            G_epoch1_ep1_con_l = G_epoch1_ep1_con_l + 1
                    
                elif trial in epoch2:
                    if trial_types[trial] == 1:
                        epoch2_ep1_s = epoch2_ep1_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_ep1_chemo_s = p_epoch2_ep1_chemo_s + 1
                            G_epoch2_ep1_chemo_s = G_epoch2_ep1_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch2_ep1_opto_s = p_epoch2_ep1_opto_s + 1
                            G_epoch2_ep1_opto_s = G_epoch2_ep1_opto_s + 1
                        else:
                            p_epoch2_ep1_con_s = p_epoch2_ep1_con_s + 1
                            G_epoch2_ep1_con_s = G_epoch2_ep1_con_s + 1
                    else:
                        epoch2_ep1_l = epoch2_ep1_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_ep1_chemo_l = p_epoch2_ep1_chemo_l + 1
                            G_epoch2_ep1_chemo_l = G_epoch2_ep1_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch2_ep1_opto_l = p_epoch2_ep1_opto_l + 1
                            G_epoch2_ep1_opto_l = G_epoch2_ep1_opto_l + 1
                        else:
                            p_epoch2_ep1_con_l = p_epoch2_ep1_con_l + 1
                            G_epoch2_ep1_con_l = G_epoch2_ep1_con_l + 1
                            
                elif trial in epoch3:
                    if trial_types[trial] == 1:
                        epoch3_ep1_s = epoch3_ep1_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_ep1_chemo_s = p_epoch3_ep1_chemo_s + 1
                            G_epoch3_ep1_chemo_s = G_epoch3_ep1_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch3_ep1_opto_s = p_epoch3_ep1_opto_s + 1
                            G_epoch3_ep1_opto_s = G_epoch3_ep1_opto_s + 1
                        else:
                            p_epoch3_ep1_con_s = p_epoch3_ep1_con_s + 1
                            G_epoch3_ep1_con_s = G_epoch3_ep1_con_s + 1
                    else:
                        epoch3_ep1_l = epoch3_ep1_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_ep1_chemo_l = p_epoch3_ep1_chemo_l + 1
                            G_epoch3_ep1_chemo_l = G_epoch3_ep1_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch3_ep1_opto_l = p_epoch3_ep1_opto_l + 1
                            G_epoch3_ep1_opto_l = G_epoch3_ep1_opto_l + 1
                        else:
                            p_epoch3_ep1_con_l = p_epoch3_ep1_con_l + 1
                            G_epoch3_ep1_con_l = G_epoch3_ep1_con_l + 1            
            elif TrialOutcomes[trial] == 'EarlyPress2':
                if trial in epoch1:
                    if trial_types[trial] == 1:
                        epoch1_ep2_s = epoch1_ep2_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_ep2_chemo_s = p_epoch1_ep2_chemo_s + 1
                            G_epoch1_ep2_chemo_s = G_epoch1_ep2_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch1_ep2_opto_s = p_epoch1_ep2_opto_s + 1
                            G_epoch1_ep2_opto_s = G_epoch1_ep2_opto_s + 1
                        else:
                            p_epoch1_ep2_con_s = p_epoch1_ep2_con_s + 1
                            G_epoch1_ep2_con_s = G_epoch1_ep2_con_s + 1
                    else:
                        epoch1_ep2_l = epoch1_ep2_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_ep2_chemo_l = p_epoch1_ep2_chemo_l + 1
                            G_epoch1_ep2_chemo_l = G_epoch1_ep2_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch1_ep2_opto_l = p_epoch1_ep2_opto_l + 1
                            G_epoch1_ep2_opto_l = G_epoch1_ep2_opto_l + 1
                        else:
                            p_epoch1_ep2_con_l = p_epoch1_ep2_con_l + 1
                            G_epoch1_ep2_con_l = G_epoch1_ep2_con_l + 1
                    
                elif trial in epoch2:
                    if trial_types[trial] == 1:
                        epoch2_ep2_s = epoch2_ep2_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_ep2_chemo_s = p_epoch2_ep2_chemo_s + 1
                            G_epoch2_ep2_chemo_s = G_epoch2_ep2_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch2_ep2_opto_s = p_epoch2_ep2_opto_s + 1
                            G_epoch2_ep2_opto_s = G_epoch2_ep2_opto_s + 1
                        else:
                            p_epoch2_ep2_con_s = p_epoch2_ep2_con_s + 1
                            G_epoch2_ep2_con_s = G_epoch2_ep2_con_s + 1
                    else:
                        epoch2_ep2_l = epoch2_ep2_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_ep2_chemo_l = p_epoch2_ep2_chemo_l + 1
                            G_epoch2_ep2_chemo_l = G_epoch2_ep2_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch2_ep2_opto_l = p_epoch2_ep2_opto_l + 1
                            G_epoch2_ep2_opto_l = G_epoch2_ep2_opto_l + 1
                        else:
                            p_epoch2_ep2_con_l = p_epoch2_ep2_con_l + 1
                            G_epoch2_ep2_con_l = G_epoch2_ep2_con_l + 1
                            
                elif trial in epoch3:
                    if trial_types[trial] == 1:
                        epoch3_ep2_s = epoch3_ep1_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_ep2_chemo_s = p_epoch3_ep2_chemo_s + 1
                            G_epoch3_ep2_chemo_s = G_epoch3_ep2_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch3_ep2_opto_s = p_epoch3_ep2_opto_s + 1
                            G_epoch3_ep2_opto_s = G_epoch3_ep2_opto_s + 1
                        else:
                            p_epoch3_ep1_con_s = p_epoch3_ep2_con_s + 1
                            G_epoch3_ep1_con_s = G_epoch3_ep2_con_s + 1
                    else:
                        epoch3_ep2_l = epoch3_ep2_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_ep2_chemo_l = p_epoch3_ep2_chemo_l + 1
                            G_epoch3_ep2_chemo_l = G_epoch3_ep2_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch3_ep2_opto_l = p_epoch3_ep2_opto_l + 1
                            G_epoch3_ep2_opto_l = G_epoch3_ep2_opto_l + 1
                        else:
                            p_epoch3_ep2_con_l = p_epoch3_ep2_con_l + 1
                            G_epoch3_ep2_con_l = G_epoch3_ep2_con_l + 1            
            ### 
            elif TrialOutcomes[trial] == 'VisStimInterruptDetect1':
                if trial in epoch1:
                    if trial_types[trial] == 1:
                        epoch1_int1_s = epoch1_int1_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_int1_chemo_s = p_epoch1_int1_chemo_s + 1
                            G_epoch1_int1_chemo_s = G_epoch1_int1_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch1_int1_opto_s = p_epoch1_int1_opto_s + 1
                            G_epoch1_int1_opto_s = G_epoch1_int1_opto_s + 1
                        else:
                            p_epoch1_int1_con_s = p_epoch1_int1_con_s + 1
                            G_epoch1_int1_con_s = G_epoch1_int1_con_s + 1
                    else:
                        epoch1_int1_l = epoch1_int1_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_int1_chemo_l = p_epoch1_int1_chemo_l + 1
                            G_epoch1_int1_chemo_l = G_epoch1_int1_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch1_int1_opto_l = p_epoch1_int1_opto_l + 1
                            G_epoch1_int1_opto_l = G_epoch1_int1_opto_l + 1
                        else:
                            p_epoch1_int1_con_l = p_epoch1_int1_con_l + 1
                            G_epoch1_int1_con_l = G_epoch1_int1_con_l + 1
                    
                elif trial in epoch2:
                    if trial_types[trial] == 1:
                        epoch2_int1_s = epoch2_int1_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_int1_chemo_s = p_epoch2_int1_chemo_s + 1
                            G_epoch2_int1_chemo_s = G_epoch2_int1_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch2_int1_opto_s = p_epoch2_int1_opto_s + 1
                            G_epoch2_int1_opto_s = G_epoch2_int1_opto_s + 1
                        else:
                            p_epoch2_int1_con_s = p_epoch2_int1_con_s + 1
                            G_epoch2_int1_con_s = G_epoch2_int1_con_s + 1
                    else:
                        epoch2_int1_l = epoch2_int1_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_int1_chemo_l = p_epoch2_int1_chemo_l + 1
                            G_epoch2_int1_chemo_l = G_epoch2_int1_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch2_int1_opto_l = p_epoch2_int1_opto_l + 1
                            G_epoch2_int1_opto_l = G_epoch2_int1_opto_l + 1
                        else:
                            p_epoch2_int1_con_l = p_epoch2_int1_con_l + 1
                            G_epoch2_int1_con_l = G_epoch2_int1_con_l + 1
                            
                elif trial in epoch3:
                    if trial_types[trial] == 1:
                        epoch3_int1_s = epoch3_ep1_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_int1_chemo_s = p_epoch3_int1_chemo_s + 1
                            G_epoch3_int1_chemo_s = G_epoch3_int1_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch3_int1_opto_s = p_epoch3_int1_opto_s + 1
                            G_epoch3_int1_opto_s = G_epoch3_int1_opto_s + 1
                        else:
                            p_epoch3_ep1_con_s = p_epoch3_int1_con_s + 1
                            G_epoch3_ep1_con_s = G_epoch3_int1_con_s + 1
                    else:
                        epoch3_int1_l = epoch3_int1_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_int1_chemo_l = p_epoch3_int1_chemo_l + 1
                            G_epoch3_int1_chemo_l = G_epoch3_int1_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch3_int1_opto_l = p_epoch3_int1_opto_l + 1
                            G_epoch3_int1_opto_l = G_epoch3_int1_opto_l + 1
                        else:
                            p_epoch3_int1_con_l = p_epoch3_int1_con_l + 1
                            G_epoch3_int1_con_l = G_epoch3_int1_con_l + 1
            elif TrialOutcomes[trial] == 'VisStimInterruptDetect2':
                if trial in epoch1:
                    if trial_types[trial] == 1:
                        epoch1_int2_s = epoch1_int2_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_int2_chemo_s = p_epoch1_int2_chemo_s + 1
                            G_epoch1_int2_chemo_s = G_epoch1_int2_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch1_int2_opto_s = p_epoch1_int2_opto_s + 1
                            G_epoch1_int2_opto_s = G_epoch1_int2_opto_s + 1
                        else:
                            p_epoch1_int2_con_s = p_epoch1_int2_con_s + 1
                            G_epoch1_int2_con_s = G_epoch1_int2_con_s + 1
                    else:
                        epoch1_int2_l = epoch1_int2_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_int2_chemo_l = p_epoch1_int2_chemo_l + 1
                            G_epoch1_int2_chemo_l = G_epoch1_int2_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch1_int2_opto_l = p_epoch1_int2_opto_l + 1
                            G_epoch1_int2_opto_l = G_epoch1_int2_opto_l + 1
                        else:
                            p_epoch1_int2_con_l = p_epoch1_int2_con_l + 1
                            G_epoch1_int2_con_l = G_epoch1_int2_con_l + 1
                    
                elif trial in epoch2:
                    if trial_types[trial] == 1:
                        epoch2_int2_s = epoch2_int2_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_int2_chemo_s = p_epoch2_int2_chemo_s + 1
                            G_epoch2_int2_chemo_s = G_epoch2_int2_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch2_int2_opto_s = p_epoch2_int2_opto_s + 1
                            G_epoch2_int2_opto_s = G_epoch2_int2_opto_s + 1
                        else:
                            p_epoch2_int2_con_s = p_epoch2_int2_con_s + 1
                            G_epoch2_int2_con_s = G_epoch2_int2_con_s + 1
                    else:
                        epoch2_int2_l = epoch2_int2_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_int2_chemo_l = p_epoch2_int2_chemo_l + 1
                            G_epoch2_int2_chemo_l = G_epoch2_int2_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch2_int2_opto_l = p_epoch2_int2_opto_l + 1
                            G_epoch2_int2_opto_l = G_epoch2_int2_opto_l + 1
                        else:
                            p_epoch2_int2_con_l = p_epoch2_int2_con_l + 1
                            G_epoch2_int2_con_l = G_epoch2_int2_con_l + 1
                            
                elif trial in epoch3:
                    if trial_types[trial] == 1:
                        epoch3_int2_s = epoch3_ep1_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_int2_chemo_s = p_epoch3_int2_chemo_s + 1
                            G_epoch3_int2_chemo_s = G_epoch3_int2_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch3_int2_opto_s = p_epoch3_int2_opto_s + 1
                            G_epoch3_int2_opto_s = G_epoch3_int2_opto_s + 1
                        else:
                            p_epoch3_ep1_con_s = p_epoch3_int2_con_s + 1
                            G_epoch3_ep1_con_s = G_epoch3_int2_con_s + 1
                    else:
                        epoch3_int2_l = epoch3_int2_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_int2_chemo_l = p_epoch3_int2_chemo_l + 1
                            G_epoch3_int2_chemo_l = G_epoch3_int2_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch3_int2_opto_l = p_epoch3_int2_opto_l + 1
                            G_epoch3_int2_opto_l = G_epoch3_int2_opto_l + 1
                        else:
                            p_epoch3_int2_con_l = p_epoch3_int2_con_l + 1
                            G_epoch3_int2_con_l = G_epoch3_int2_con_l + 1          
            elif TrialOutcomes[trial] == 'VisStimInterruptGray1':
                if trial in epoch1:
                    if trial_types[trial] == 1:
                        epoch1_int3_s = epoch1_int3_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_int3_chemo_s = p_epoch1_int3_chemo_s + 1
                            G_epoch1_int3_chemo_s = G_epoch1_int3_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch1_int3_opto_s = p_epoch1_int3_opto_s + 1
                            G_epoch1_int3_opto_s = G_epoch1_int3_opto_s + 1
                        else:
                            p_epoch1_int3_con_s = p_epoch1_int3_con_s + 1
                            G_epoch1_int3_con_s = G_epoch1_int3_con_s + 1
                    else:
                        epoch1_int3_l = epoch1_int3_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_int3_chemo_l = p_epoch1_int3_chemo_l + 1
                            G_epoch1_int3_chemo_l = G_epoch1_int3_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch1_int3_opto_l = p_epoch1_int3_opto_l + 1
                            G_epoch1_int3_opto_l = G_epoch1_int3_opto_l + 1
                        else:
                            p_epoch1_int3_con_l = p_epoch1_int3_con_l + 1
                            G_epoch1_int3_con_l = G_epoch1_int3_con_l + 1
                    
                elif trial in epoch2:
                    if trial_types[trial] == 1:
                        epoch2_int3_s = epoch2_int3_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_int3_chemo_s = p_epoch2_int3_chemo_s + 1
                            G_epoch2_int3_chemo_s = G_epoch2_int3_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch2_int3_opto_s = p_epoch2_int3_opto_s + 1
                            G_epoch2_int3_opto_s = G_epoch2_int3_opto_s + 1
                        else:
                            p_epoch2_int3_con_s = p_epoch2_int3_con_s + 1
                            G_epoch2_int3_con_s = G_epoch2_int3_con_s + 1
                    else:
                        epoch2_int3_l = epoch2_int3_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_int3_chemo_l = p_epoch2_int3_chemo_l + 1
                            G_epoch2_int3_chemo_l = G_epoch2_int3_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch2_int3_opto_l = p_epoch2_int3_opto_l + 1
                            G_epoch2_int3_opto_l = G_epoch2_int3_opto_l + 1
                        else:
                            p_epoch2_int3_con_l = p_epoch2_int3_con_l + 1
                            G_epoch2_int3_con_l = G_epoch2_int3_con_l + 1
                            
                elif trial in epoch3:
                    if trial_types[trial] == 1:
                        epoch3_int3_s = epoch3_ep1_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_int3_chemo_s = p_epoch3_int3_chemo_s + 1
                            G_epoch3_int3_chemo_s = G_epoch3_int3_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch3_int3_opto_s = p_epoch3_int3_opto_s + 1
                            G_epoch3_int3_opto_s = G_epoch3_int3_opto_s + 1
                        else:
                            p_epoch3_ep1_con_s = p_epoch3_int3_con_s + 1
                            G_epoch3_ep1_con_s = G_epoch3_int3_con_s + 1
                    else:
                        epoch3_int3_l = epoch3_int3_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_int3_chemo_l = p_epoch3_int3_chemo_l + 1
                            G_epoch3_int3_chemo_l = G_epoch3_int3_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch3_int3_opto_l = p_epoch3_int3_opto_l + 1
                            G_epoch3_int3_opto_l = G_epoch3_int3_opto_l + 1
                        else:
                            p_epoch3_int3_con_l = p_epoch3_int3_con_l + 1
                            G_epoch3_int3_con_l = G_epoch3_int3_con_l + 1
            elif TrialOutcomes[trial] == 'VisStimInterruptGray2':
                if trial in epoch1:
                    if trial_types[trial] == 1:
                        epoch1_int4_s = epoch1_int4_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_int4_chemo_s = p_epoch1_int4_chemo_s + 1
                            G_epoch1_int4_chemo_s = G_epoch1_int4_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch1_int4_opto_s = p_epoch1_int4_opto_s + 1
                            G_epoch1_int4_opto_s = G_epoch1_int4_opto_s + 1
                        else:
                            p_epoch1_int4_con_s = p_epoch1_int4_con_s + 1
                            G_epoch1_int4_con_s = G_epoch1_int4_con_s + 1
                    else:
                        epoch1_int4_l = epoch1_int4_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_int4_chemo_l = p_epoch1_int4_chemo_l + 1
                            G_epoch1_int4_chemo_l = G_epoch1_int4_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch1_int4_opto_l = p_epoch1_int4_opto_l + 1
                            G_epoch1_int4_opto_l = G_epoch1_int4_opto_l + 1
                        else:
                            p_epoch1_int4_con_l = p_epoch1_int4_con_l + 1
                            G_epoch1_int4_con_l = G_epoch1_int4_con_l + 1
                    
                elif trial in epoch2:
                    if trial_types[trial] == 1:
                        epoch2_int4_s = epoch2_int4_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_int4_chemo_s = p_epoch2_int4_chemo_s + 1
                            G_epoch2_int4_chemo_s = G_epoch2_int4_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch2_int4_opto_s = p_epoch2_int4_opto_s + 1
                            G_epoch2_int4_opto_s = G_epoch2_int4_opto_s + 1
                        else:
                            p_epoch2_int4_con_s = p_epoch2_int4_con_s + 1
                            G_epoch2_int4_con_s = G_epoch2_int4_con_s + 1
                    else:
                        epoch2_int4_l = epoch2_int4_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_int4_chemo_l = p_epoch2_int4_chemo_l + 1
                            G_epoch2_int4_chemo_l = G_epoch2_int4_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch2_int4_opto_l = p_epoch2_int4_opto_l + 1
                            G_epoch2_int4_opto_l = G_epoch2_int4_opto_l + 1
                        else:
                            p_epoch2_int4_con_l = p_epoch2_int4_con_l + 1
                            G_epoch2_int4_con_l = G_epoch2_int4_con_l + 1
                            
                elif trial in epoch3:
                    if trial_types[trial] == 1:
                        epoch3_int4_s = epoch3_ep1_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_int4_chemo_s = p_epoch3_int4_chemo_s + 1
                            G_epoch3_int4_chemo_s = G_epoch3_int4_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch3_int4_opto_s = p_epoch3_int4_opto_s + 1
                            G_epoch3_int4_opto_s = G_epoch3_int4_opto_s + 1
                        else:
                            p_epoch3_ep1_con_s = p_epoch3_int4_con_s + 1
                            G_epoch3_ep1_con_s = G_epoch3_int4_con_s + 1
                    else:
                        epoch3_int4_l = epoch3_int4_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_int4_chemo_l = p_epoch3_int4_chemo_l + 1
                            G_epoch3_int4_chemo_l = G_epoch3_int4_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch3_int4_opto_l = p_epoch3_int4_opto_l + 1
                            G_epoch3_int4_opto_l = G_epoch3_int4_opto_l + 1
                        else:
                            p_epoch3_int4_con_l = p_epoch3_int4_con_l + 1
                            G_epoch3_int4_con_l = G_epoch3_int4_con_l + 1          
            ####                
            else:
                if trial in epoch1:
                    if trial_types[trial] == 1:
                        epoch1_int_s = epoch1_int_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_int_chemo_s = p_epoch1_int_chemo_s + 1
                            G_epoch1_int_chemo_s = G_epoch1_int_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch1_int_opto_s = p_epoch1_int_opto_s + 1
                            G_epoch1_int_opto_s = G_epoch1_int_opto_s + 1
                        else:
                            p_epoch1_int_con_s = p_epoch1_int_con_s + 1
                            G_epoch1_int_con_s = G_epoch1_int_con_s + 1
                    else:
                        epoch1_int_l = epoch1_int_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch1_int_chemo_l = p_epoch1_int_chemo_l + 1
                            G_epoch1_int_chemo_l = G_epoch1_int_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch1_int_opto_l = p_epoch1_int_opto_l + 1
                            G_epoch1_int_opto_l = G_epoch1_int_opto_l + 1
                        else:
                            p_epoch1_int_con_l = p_epoch1_int_con_l + 1
                            G_epoch1_int_con_l = G_epoch1_int_con_l + 1
                    
                elif trial in epoch2:
                    if trial_types[trial] == 1:
                        epoch2_int_s = epoch2_int_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_int_chemo_s = p_epoch2_int_chemo_s + 1
                            G_epoch2_int_chemo_s = G_epoch2_int_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch2_int_opto_s = p_epoch2_int_opto_s + 1
                            G_epoch2_int_opto_s = G_epoch2_int_opto_s + 1
                        else:
                            p_epoch2_int_con_s = p_epoch2_int_con_s + 1
                            G_epoch2_int_con_s = G_epoch2_int_con_s + 1
                    else:
                        epoch2_int_l = epoch2_int_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch2_int_chemo_l = p_epoch2_int_chemo_l + 1
                            G_epoch2_int_chemo_l = G_epoch2_int_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch2_int_opto_l = p_epoch2_int_opto_l + 1
                            G_epoch2_int_opto_l = G_epoch2_int_opto_l + 1
                        else:
                            p_epoch2_int_con_l = p_epoch2_int_con_l + 1
                            G_epoch2_int_con_l = G_epoch2_int_con_l + 1
                            
                elif trial in epoch3:
                    if trial_types[trial] == 1:
                        epoch3_int_s = epoch3_int_s + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_int_chemo_s = p_epoch3_int_chemo_s + 1
                            G_epoch3_int_chemo_s = G_epoch3_int_chemo_s + 1
                        elif opto[trial] == 1:
                            p_epoch3_int_opto_s = p_epoch3_int_opto_s + 1
                            G_epoch3_int_opto_s = G_epoch3_int_opto_s + 1
                        else:
                            p_epoch3_int_con_s = p_epoch3_int_con_s + 1
                            G_epoch3_int_con_s = G_epoch3_int_con_s + 1
                    else:
                        epoch3_int_l = epoch3_int_l + 1
                        if chemo_labels[i] == 1:
                            p_epoch3_int_chemo_l = p_epoch3_int_chemo_l + 1
                            G_epoch3_int_chemo_l = G_epoch3_int_chemo_l + 1
                        elif opto[trial] == 1:
                            p_epoch3_int_opto_l = p_epoch3_int_opto_l + 1
                            G_epoch3_int_opto_l = G_epoch3_int_opto_l + 1
                        else:
                            p_epoch3_int_con_l = p_epoch3_int_con_l + 1
                            G_epoch3_int_con_l = G_epoch3_int_con_l + 1         
        
        #Creating the arrays
        short_epoch1 = [epoch1_reward_s,epoch1_dnp1_s,epoch1_dnp2_s,epoch1_ep_s,epoch1_ep1_s,epoch1_ep2_s,epoch1_int1_s,epoch1_int2_s,epoch1_int3_s,epoch1_int4_s,epoch1_int_s] 
        short_epoch2 = [epoch2_reward_s,epoch2_dnp1_s,epoch2_dnp2_s,epoch2_ep_s,epoch2_ep1_s,epoch2_ep2_s,epoch2_int1_s,epoch2_int2_s,epoch2_int3_s,epoch2_int4_s,epoch2_int_s]
        short_epoch3 = [epoch3_reward_s,epoch3_dnp1_s,epoch3_dnp2_s,epoch3_ep_s,epoch3_ep1_s,epoch3_ep2_s,epoch3_int1_s,epoch3_int2_s,epoch3_int3_s,epoch3_int4_s,epoch3_int_s]
        
        long_epoch1 = [epoch1_reward_l,epoch1_dnp1_l,epoch1_dnp2_l,epoch1_ep_l,epoch1_ep1_l,epoch1_ep2_l,epoch1_int1_l,epoch1_int2_l,epoch1_int3_l,epoch1_int4_l,epoch1_int_l]
        long_epoch2 = [epoch2_reward_l,epoch2_dnp1_l,epoch2_dnp2_l,epoch2_ep_l,epoch2_ep1_l,epoch2_ep2_l,epoch2_int1_l,epoch2_int2_l,epoch2_int3_l,epoch2_int4_l,epoch2_int_l]
        long_epoch3 = [epoch3_reward_l,epoch3_dnp1_l,epoch3_dnp2_l,epoch3_ep_l,epoch3_ep1_l,epoch3_ep2_l,epoch3_int1_l,epoch3_int2_l,epoch3_int3_l,epoch3_int4_l,epoch3_int_l]
        
        short = [short_epoch1/np.sum(short_epoch1),short_epoch2/np.sum(short_epoch2),short_epoch3/np.sum(short_epoch3)]
        long = [long_epoch1/np.sum(long_epoch1),long_epoch2/np.sum(long_epoch2),long_epoch3/np.sum(long_epoch3)]
        
        short_bottom = np.cumsum(short, axis=1)
        short_bottom[:,1:] = short_bottom[:,:-1]
        short_bottom[:,0] = 0

        long_bottom = np.cumsum(long, axis=1)
        long_bottom[:,1:] = long_bottom[:,:-1]
        long_bottom[:,0] = 0
        
        
        axs1[0].tick_params(tick1On=False)
        axs1[0].spines['left'].set_visible(False)
        axs1[0].spines['right'].set_visible(False)
        axs1[0].spines['top'].set_visible(False)
        axs1[0].set_xlabel('sessions')
        axs1[0].set_ylabel('Outcome percentages')
        
        axs1[1].tick_params(tick1On=False)
        axs1[1].spines['left'].set_visible(False)
        axs1[1].spines['right'].set_visible(False)
        axs1[1].spines['top'].set_visible(False)
        axs1[1].set_xlabel('sessions')
        axs1[1].set_ylabel('Outcome percentages')
        
        
        for k in range(len(states)):
            if i < 1:
                axs1[0].bar(
                    i + 1, short[0][k],
                    bottom=short_bottom[0][k],
                    edgecolor='white',
                    width=width/3,
                    color=colors[k],
                    label=states_name[k]
                )
            else:
                axs1[0].bar(
                    i + 1, short[0][k],
                    bottom=short_bottom[0][k],
                    edgecolor='white',
                    width=width/3,
                    color=colors[k]
                )
            axs1[0].bar(
                i+1 + width/3,short[1][k],
                bottom=short_bottom[1][k],
                edgecolor='white',
                width=width/3,
                color=colors[k])
            axs1[0].bar(
                i + 1 + 2*width/3,short[2][k],
                bottom=short_bottom[2][k],
                edgecolor='white',
                width=width/3,
                color=colors[k])
            if i < 1:
                axs1[1].bar(
                    i+1, long[0][k],
                    bottom=long_bottom[0][k],
                    edgecolor='white',
                    width=width/3,
                    color=colors[k],
                    label=states_name[k])
            else:
                axs1[1].bar(
                    i+1, long[0][k],
                    bottom=long_bottom[0][k],
                    edgecolor='white',
                    width=width/3,
                    color=colors[k])
                
            axs1[1].bar(
                i+1 + width/3,long[1][k],
                bottom=long_bottom[1][k],
                edgecolor='white',
                width=width/3,
                color=colors[k])
            axs1[1].bar(
                i+1 + 2*width/3,long[2][k],
                bottom=long_bottom[2][k],
                edgecolor='white',
                width=width/3,
                color=colors[k])
            
        axs1[0].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
        axs1[1].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
        # ############################### Grand Average ###############################
        grand_short_epoch1_chemo = [G_epoch1_reward_chemo_s,G_epoch1_dnp1_chemo_s,G_epoch1_dnp2_chemo_s,G_epoch1_ep_chemo_s,G_epoch1_ep1_chemo_s,G_epoch1_ep2_chemo_s,G_epoch1_int1_chemo_s,G_epoch1_int2_chemo_s,G_epoch1_int3_chemo_s,G_epoch1_int4_chemo_s,G_epoch1_int_chemo_s]
        grand_short_epoch1_opto = [G_epoch1_reward_opto_s,G_epoch1_dnp1_opto_s,G_epoch1_dnp2_opto_s,G_epoch1_ep_opto_s,G_epoch1_ep1_opto_s,G_epoch1_ep2_opto_s,G_epoch1_int1_opto_s,G_epoch1_int2_opto_s,G_epoch1_int3_opto_s,G_epoch1_int4_opto_s,G_epoch1_int_opto_s]
        grand_short_epoch1_con = [G_epoch1_reward_con_s,G_epoch1_dnp1_con_s,G_epoch1_dnp2_con_s,G_epoch1_ep_con_s,G_epoch1_ep1_con_s,G_epoch1_ep2_con_s,G_epoch1_int1_con_s,G_epoch1_int2_con_s,G_epoch1_int3_con_s,G_epoch1_int4_con_s,G_epoch1_int_con_s]

        grand_short_epoch2_chemo = [G_epoch2_reward_chemo_s,G_epoch2_dnp1_chemo_s,G_epoch2_dnp2_chemo_s,G_epoch2_ep_chemo_s,G_epoch2_ep1_chemo_s,G_epoch2_ep2_chemo_s,G_epoch2_int1_chemo_s,G_epoch2_int2_chemo_s,G_epoch2_int3_chemo_s,G_epoch2_int4_chemo_s,G_epoch2_int_chemo_s]
        grand_short_epoch2_opto = [G_epoch2_reward_opto_s,G_epoch2_dnp1_opto_s,G_epoch2_dnp2_opto_s,G_epoch2_ep_opto_s,G_epoch2_ep1_opto_s,G_epoch2_ep2_opto_s,G_epoch2_int1_opto_s,G_epoch2_int2_opto_s,G_epoch2_int3_opto_s,G_epoch2_int4_opto_s,G_epoch2_int_opto_s]
        grand_short_epoch2_con = [G_epoch2_reward_con_s,G_epoch2_dnp1_con_s,G_epoch2_dnp2_con_s,G_epoch2_ep_con_s,G_epoch2_ep1_con_s,G_epoch2_ep2_con_s,G_epoch2_int1_con_s,G_epoch2_int2_con_s,G_epoch2_int3_con_s,G_epoch2_int4_con_s,G_epoch2_int_con_s]

        grand_short_epoch3_chemo = [G_epoch3_reward_chemo_s,G_epoch3_dnp1_chemo_s,G_epoch3_dnp2_chemo_s,G_epoch3_ep_chemo_s,G_epoch3_ep1_chemo_s,G_epoch3_ep2_chemo_s,G_epoch3_int1_chemo_s,G_epoch3_int2_chemo_s,G_epoch3_int3_chemo_s,G_epoch3_int4_chemo_s,G_epoch3_int_chemo_s]
        grand_short_epoch3_opto = [G_epoch3_reward_opto_s,G_epoch3_dnp1_opto_s,G_epoch3_dnp2_opto_s,G_epoch3_ep_opto_s,G_epoch3_ep1_opto_s,G_epoch3_ep2_opto_s,G_epoch3_int1_opto_s,G_epoch3_int2_opto_s,G_epoch3_int3_opto_s,G_epoch3_int4_opto_s,G_epoch3_int_opto_s]
        grand_short_epoch3_con = [G_epoch3_reward_con_s,G_epoch3_dnp1_con_s,G_epoch3_dnp2_con_s,G_epoch3_ep_con_s,G_epoch3_ep1_con_s,G_epoch3_ep2_con_s,G_epoch3_int1_con_s,G_epoch3_int2_con_s,G_epoch3_int3_con_s,G_epoch3_int4_con_s,G_epoch3_int_con_s]

        grand_long_epoch1_chemo = [G_epoch1_reward_chemo_l,G_epoch1_dnp1_chemo_l,G_epoch1_dnp2_chemo_l,G_epoch1_ep_chemo_l,G_epoch1_ep1_chemo_l,G_epoch1_ep2_chemo_l,G_epoch1_int1_chemo_l,G_epoch1_int2_chemo_l,G_epoch1_int3_chemo_l,G_epoch1_int4_chemo_l,G_epoch1_int_chemo_l]
        grand_long_epoch1_opto = [G_epoch1_reward_opto_l,G_epoch1_dnp1_opto_l,G_epoch1_dnp2_opto_l,G_epoch1_ep_opto_l,G_epoch1_ep1_opto_l,G_epoch1_ep2_opto_l,G_epoch1_int1_opto_l,G_epoch1_int2_opto_l,G_epoch1_int3_opto_l,G_epoch1_int4_opto_l,G_epoch1_int_opto_l]
        grand_long_epoch1_con = [G_epoch1_reward_con_l,G_epoch1_dnp1_con_l,G_epoch1_dnp2_con_l,G_epoch1_ep_con_l,G_epoch1_ep1_con_l,G_epoch1_ep2_con_l,G_epoch1_int1_con_l,G_epoch1_int2_con_l,G_epoch1_int3_con_l,G_epoch1_int4_con_l,G_epoch1_int_con_l]

        grand_long_epoch2_chemo = [G_epoch2_reward_chemo_l,G_epoch2_dnp1_chemo_l,G_epoch2_dnp2_chemo_l,G_epoch2_ep_chemo_l,G_epoch2_ep1_chemo_l,G_epoch2_ep2_chemo_l,G_epoch2_int1_chemo_l,G_epoch2_int2_chemo_l,G_epoch2_int3_chemo_l,G_epoch2_int4_chemo_l,G_epoch2_int_chemo_l]
        grand_long_epoch2_opto = [G_epoch2_reward_opto_l,G_epoch2_dnp1_opto_l,G_epoch2_dnp2_opto_l,G_epoch2_ep_opto_l,G_epoch2_ep1_opto_l,G_epoch2_ep2_opto_l,G_epoch2_int1_opto_l,G_epoch2_int2_opto_l,G_epoch2_int3_opto_l,G_epoch2_int4_opto_l,G_epoch2_int_opto_l]
        grand_long_epoch2_con = [G_epoch2_reward_con_l,G_epoch2_dnp1_con_l,G_epoch2_dnp2_con_l,G_epoch2_ep_con_l,G_epoch2_ep1_con_l,G_epoch2_ep2_con_l,G_epoch2_int1_con_l,G_epoch2_int2_con_l,G_epoch2_int3_con_l,G_epoch2_int4_con_l,G_epoch2_int_con_l]

        grand_long_epoch3_chemo = [G_epoch3_reward_chemo_l,G_epoch3_dnp1_chemo_l,G_epoch3_dnp2_chemo_l,G_epoch3_ep_chemo_l,G_epoch3_ep1_chemo_l,G_epoch3_ep2_chemo_l,G_epoch3_int1_chemo_l,G_epoch3_int2_chemo_l,G_epoch3_int3_chemo_l,G_epoch3_int4_chemo_l,G_epoch3_int_chemo_l]
        grand_long_epoch3_opto = [G_epoch3_reward_opto_l,G_epoch3_dnp1_opto_l,G_epoch3_dnp2_opto_l,G_epoch3_ep_opto_l,G_epoch3_ep1_opto_l,G_epoch3_ep2_opto_l,G_epoch3_int1_opto_l,G_epoch3_int2_opto_l,G_epoch3_int3_opto_l,G_epoch3_int4_opto_l,G_epoch3_int_opto_l]
        grand_long_epoch3_con = [G_epoch3_reward_con_l,G_epoch3_dnp1_con_l,G_epoch3_dnp2_con_l,G_epoch3_ep_con_l,G_epoch3_ep1_con_l,G_epoch3_ep2_con_l,G_epoch3_int1_con_l,G_epoch3_int2_con_l,G_epoch3_int3_con_l,G_epoch3_int4_con_l,G_epoch3_int_con_l]
        
        # Suppress RuntimeWarning (if we have nan value this suppres the devide warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            grand_short_epoch1.append([grand_short_epoch1_con/np.sum(grand_short_epoch1_con),grand_short_epoch1_chemo/np.sum(grand_short_epoch1_chemo),grand_short_epoch1_opto/np.sum(grand_short_epoch1_opto)])
            grand_short_epoch2.append([grand_short_epoch2_con/np.sum(grand_short_epoch2_con),grand_short_epoch2_chemo/np.sum(grand_short_epoch2_chemo),grand_short_epoch2_opto/np.sum(grand_short_epoch2_opto)])
            grand_short_epoch3.append([grand_short_epoch3_con/np.sum(grand_short_epoch3_con),grand_short_epoch3_chemo/np.sum(grand_short_epoch3_chemo),grand_short_epoch3_opto/np.sum(grand_short_epoch3_opto)])

            grand_long_epoch1.append([grand_long_epoch1_con/np.sum(grand_long_epoch1_con),grand_long_epoch1_chemo/np.sum(grand_long_epoch1_chemo),grand_long_epoch1_opto/np.sum(grand_long_epoch1_opto)])
            grand_long_epoch2.append([grand_long_epoch2_con/np.sum(grand_long_epoch2_con),grand_long_epoch2_chemo/np.sum(grand_long_epoch2_chemo),grand_long_epoch2_opto/np.sum(grand_long_epoch2_opto)])
            grand_long_epoch3.append([grand_long_epoch3_con/np.sum(grand_long_epoch3_con),grand_long_epoch3_chemo/np.sum(grand_long_epoch3_chemo),grand_long_epoch3_opto/np.sum(grand_long_epoch3_opto)])
        

    fig1.tight_layout()
    ################################### POOLED ####################################

    fig2, axs2 = plt.subplots(nrows=2,figsize=(15 , 10))
    fig2.suptitle('Reward percentage for POOLED sessions ' +  subject)
    width = 0.2

    short_epoch1_chemo = [p_epoch1_reward_chemo_s,p_epoch1_dnp1_chemo_s,p_epoch1_dnp2_chemo_s,p_epoch1_ep_chemo_s,p_epoch1_ep1_chemo_s,p_epoch1_ep2_chemo_s,p_epoch1_int1_chemo_s,p_epoch1_int2_chemo_s,p_epoch1_int3_chemo_s,p_epoch1_int4_chemo_s,p_epoch1_int_chemo_s]
    short_epoch1_opto = [p_epoch1_reward_opto_s,p_epoch1_dnp1_opto_s,p_epoch1_dnp2_opto_s,p_epoch1_ep_opto_s,p_epoch1_ep1_opto_s,p_epoch1_ep2_opto_s,p_epoch1_int1_opto_s,p_epoch1_int2_opto_s,p_epoch1_int3_opto_s,p_epoch1_int4_opto_s,p_epoch1_int_opto_s]
    short_epoch1_con = [p_epoch1_reward_con_s,p_epoch1_dnp1_con_s,p_epoch1_dnp2_con_s,p_epoch1_ep_con_s,p_epoch1_ep1_con_s,p_epoch1_ep2_con_s,p_epoch1_int1_con_s,p_epoch1_int2_con_s,p_epoch1_int3_con_s,p_epoch1_int4_con_s,p_epoch1_int_con_s]

    short_epoch2_chemo = [p_epoch2_reward_chemo_s,p_epoch2_dnp1_chemo_s,p_epoch2_dnp2_chemo_s,p_epoch2_ep_chemo_s,p_epoch2_ep1_chemo_s,p_epoch2_ep2_chemo_s,p_epoch2_int1_chemo_s,p_epoch2_int2_chemo_s,p_epoch2_int3_chemo_s,p_epoch2_int4_chemo_s,p_epoch2_int_chemo_s]
    short_epoch2_opto = [p_epoch2_reward_opto_s,p_epoch2_dnp1_opto_s,p_epoch2_dnp2_opto_s,p_epoch2_ep_opto_s,p_epoch2_ep1_opto_s,p_epoch2_ep2_opto_s,p_epoch2_int1_opto_s,p_epoch2_int2_opto_s,p_epoch2_int3_opto_s,p_epoch2_int4_opto_s,p_epoch2_int_opto_s]
    short_epoch2_con = [p_epoch2_reward_con_s,p_epoch2_dnp1_con_s,p_epoch2_dnp2_con_s,p_epoch2_ep_con_s,p_epoch2_ep1_con_s,p_epoch2_ep2_con_s,p_epoch2_int1_con_s,p_epoch2_int2_con_s,p_epoch2_int3_con_s,p_epoch2_int4_con_s,p_epoch2_int_con_s]

    short_epoch3_chemo = [p_epoch3_reward_chemo_s,p_epoch3_dnp1_chemo_s,p_epoch3_dnp2_chemo_s,p_epoch3_ep_chemo_s,p_epoch3_ep1_chemo_s,p_epoch3_ep2_chemo_s,p_epoch3_int1_chemo_s,p_epoch3_int2_chemo_s,p_epoch3_int3_chemo_s,p_epoch3_int4_chemo_s,p_epoch3_int_chemo_s]
    short_epoch3_opto = [p_epoch3_reward_opto_s,p_epoch3_dnp1_opto_s,p_epoch3_dnp2_opto_s,p_epoch3_ep_opto_s,p_epoch3_ep1_opto_s,p_epoch3_ep2_opto_s,p_epoch3_int1_opto_s,p_epoch3_int2_opto_s,p_epoch3_int3_opto_s,p_epoch3_int4_opto_s,p_epoch3_int_opto_s]
    short_epoch3_con = [p_epoch3_reward_con_s,p_epoch3_dnp1_con_s,p_epoch3_dnp2_con_s,p_epoch3_ep_con_s,p_epoch3_ep1_con_s,p_epoch3_ep2_con_s,p_epoch3_int1_con_s,p_epoch3_int2_con_s,p_epoch3_int3_con_s,p_epoch3_int4_con_s,p_epoch3_int_con_s]

    long_epoch1_chemo = [p_epoch1_reward_chemo_l,p_epoch1_dnp1_chemo_l,p_epoch1_dnp2_chemo_l,p_epoch1_ep_chemo_l,p_epoch1_ep1_chemo_l,p_epoch1_ep2_chemo_l,p_epoch1_int1_chemo_l,p_epoch1_int2_chemo_l,p_epoch1_int3_chemo_l,p_epoch1_int4_chemo_l,p_epoch1_int_chemo_l]
    long_epoch1_opto = [p_epoch1_reward_opto_l,p_epoch1_dnp1_opto_l,p_epoch1_dnp2_opto_l,p_epoch1_ep_opto_l,p_epoch1_ep1_opto_l,p_epoch1_ep2_opto_l,p_epoch1_int1_opto_l,p_epoch1_int2_opto_l,p_epoch1_int3_opto_l,p_epoch1_int4_opto_l,p_epoch1_int_opto_l]
    long_epoch1_con = [p_epoch1_reward_con_l,p_epoch1_dnp1_con_l,p_epoch1_dnp2_con_l,p_epoch1_ep_con_l,p_epoch1_ep1_con_l,p_epoch1_ep2_con_l,p_epoch1_int1_con_l,p_epoch1_int2_con_l,p_epoch1_int3_con_l,p_epoch1_int4_con_l,p_epoch1_int_con_l]

    long_epoch2_chemo = [p_epoch2_reward_chemo_l,p_epoch2_dnp1_chemo_l,p_epoch2_dnp2_chemo_l,p_epoch2_ep_chemo_l,p_epoch2_ep1_chemo_l,p_epoch2_ep2_chemo_l,p_epoch2_int1_chemo_l,p_epoch2_int2_chemo_l,p_epoch2_int3_chemo_l,p_epoch2_int4_chemo_l,p_epoch2_int_chemo_l]
    long_epoch2_opto = [p_epoch2_reward_opto_l,p_epoch2_dnp1_opto_l,p_epoch2_dnp2_opto_l,p_epoch2_ep_opto_l,p_epoch2_ep1_opto_l,p_epoch2_ep2_opto_l,p_epoch2_int1_opto_l,p_epoch2_int2_opto_l,p_epoch2_int3_opto_l,p_epoch2_int4_opto_l,p_epoch2_int_opto_l]
    long_epoch2_con = [p_epoch2_reward_con_l,p_epoch2_dnp1_con_l,p_epoch2_dnp2_con_l,p_epoch2_ep_con_l,p_epoch2_ep1_con_l,p_epoch2_ep2_con_l,p_epoch2_int1_con_l,p_epoch2_int2_con_l,p_epoch2_int3_con_l,p_epoch2_int4_con_l,p_epoch2_int_con_l]

    long_epoch3_chemo = [p_epoch3_reward_chemo_l,p_epoch3_dnp1_chemo_l,p_epoch3_dnp2_chemo_l,p_epoch3_ep_chemo_l,p_epoch3_ep1_chemo_l,p_epoch3_ep2_chemo_l,p_epoch3_int1_chemo_l,p_epoch3_int2_chemo_l,p_epoch3_int3_chemo_l,p_epoch3_int4_chemo_l,p_epoch3_int_chemo_l]
    long_epoch3_opto = [p_epoch3_reward_opto_l,p_epoch3_dnp1_opto_l,p_epoch3_dnp2_opto_l,p_epoch3_ep_opto_l,p_epoch3_ep1_opto_l,p_epoch3_ep2_opto_l,p_epoch3_int1_opto_l,p_epoch3_int2_opto_l,p_epoch3_int3_opto_l,p_epoch3_int4_opto_l,p_epoch3_int_opto_l]
    long_epoch3_con = [p_epoch3_reward_con_l,p_epoch3_dnp1_con_l,p_epoch3_dnp2_con_l,p_epoch3_ep_con_l,p_epoch3_ep1_con_l,p_epoch3_ep2_con_l,p_epoch3_int1_con_l,p_epoch3_int2_con_l,p_epoch3_int3_con_l,p_epoch3_int4_con_l,p_epoch3_int_con_l]

    session_id_mega = np.arange(3) + 1
    top_ticks_mega = []
    for i in range(0,3):
        top_ticks_mega.append('ep1|ep2|ep3')

    short_epoch1 = [short_epoch1_con/np.sum(short_epoch1_con),short_epoch1_chemo/np.sum(short_epoch1_chemo),short_epoch1_opto/np.sum(short_epoch1_opto)]
    short_epoch2 = [short_epoch2_con/np.sum(short_epoch2_con),short_epoch2_chemo/np.sum(short_epoch2_chemo),short_epoch2_opto/np.sum(short_epoch2_opto)]
    short_epoch3 = [short_epoch3_con/np.sum(short_epoch3_con),short_epoch3_chemo/np.sum(short_epoch3_chemo),short_epoch3_opto/np.sum(short_epoch3_opto)]

    long_epoch1 = [long_epoch1_con/np.sum(long_epoch1_con),long_epoch1_chemo/np.sum(long_epoch1_chemo),long_epoch1_opto/np.sum(long_epoch1_opto)]
    long_epoch2 = [long_epoch2_con/np.sum(long_epoch2_con),long_epoch2_chemo/np.sum(long_epoch2_chemo),long_epoch2_opto/np.sum(long_epoch2_opto)]
    long_epoch3 = [long_epoch3_con/np.sum(long_epoch3_con),long_epoch3_chemo/np.sum(long_epoch3_chemo),long_epoch3_opto/np.sum(long_epoch3_opto)]

    short_bottom_1 = np.cumsum(short_epoch1, axis=1)
    short_bottom_1[:,1:] = short_bottom_1[:,:-1]
    short_bottom_1[:,0] = 0

    short_bottom_2 = np.cumsum(short_epoch2, axis=1)
    short_bottom_2[:,1:] = short_bottom_2[:,:-1]
    short_bottom_2[:,0] = 0

    short_bottom_3 = np.cumsum(short_epoch3, axis=1)
    short_bottom_3[:,1:] = short_bottom_3[:,:-1]
    short_bottom_3[:,0] = 0

    long_bottom_1 = np.cumsum(long_epoch1, axis=1)
    long_bottom_1[:,1:] = long_bottom_1[:,:-1]
    long_bottom_1[:,0] = 0

    long_bottom_2 = np.cumsum(long_epoch2, axis=1)
    long_bottom_2[:,1:] = long_bottom_2[:,:-1]
    long_bottom_2[:,0] = 0

    long_bottom_3 = np.cumsum(long_epoch3, axis=1)
    long_bottom_3[:,1:] = long_bottom_3[:,:-1]
    long_bottom_3[:,0] = 0

    axs2[0].tick_params(tick1On=False)
    axs2[0].spines['left'].set_visible(False)
    axs2[0].spines['right'].set_visible(False)
    axs2[0].spines['top'].set_visible(False)
    axs2[0].set_xlabel('Mega session')
    axs2[0].set_ylabel('Outcome percentages')
    axs2[0].set_title('Epoch Analysis for Short Trials')

    tick_index = np.arange(3)+1
    axs2[0].set_xticks(tick_index + width/3)
    dates_label = ['Control','Chemo','Opto']

    secax = axs2[0].secondary_xaxis('top')
    secax.set_xticks(tick_index + width/3)
    secax.set_xticklabels(top_ticks_mega)

    axs2[1].tick_params(tick1On=False)
    axs2[1].spines['left'].set_visible(False)
    axs2[1].spines['right'].set_visible(False)
    axs2[1].spines['top'].set_visible(False)
    axs2[1].set_xlabel('Mega session')
    axs2[1].set_ylabel('Outcome percentages')
    axs2[1].set_title('Epoch Analysis for Long Trials')

    axs2[1].set_xticks(tick_index + width/3)
    dates_label = ['Control','Chemo','Opto']

    secax = axs2[1].secondary_xaxis('top')
    secax.set_xticks(tick_index + width/3)
    secax.set_xticklabels(top_ticks_mega)

    axs2[0].set_xticklabels(dates_label, rotation='vertical')
    ind = 0
    for xtick in axs2[0].get_xticklabels():
        if dates_label[ind] == 'Chemo':
            xtick.set_color('r')
        elif dates_label[ind] == 'Opto':
            xtick.set_color('deepskyblue')        
        ind = ind + 1
        
    axs2[1].set_xticklabels(dates_label, rotation='vertical')
    ind = 0
    for xtick in axs2[1].get_xticklabels():
        if dates_label[ind] == 'Chemo':
            xtick.set_color('r')
        elif dates_label[ind] == 'Opto':
            xtick.set_color('deepskyblue')        
        ind = ind + 1
        
    for i in range(len(states)):
        axs2[0].bar(
            session_id_mega, [short_epoch1[0][i],short_epoch1[1][i],short_epoch1[2][i]],
            bottom=short_bottom_1[:,i],
            edgecolor='white',
            width=width/3,
            color=colors[i],
            label=states_name[i])
        axs2[0].bar(
            session_id_mega  + width/3, [short_epoch2[0][i],short_epoch2[1][i],short_epoch2[2][i]],
            bottom=short_bottom_2[:,i],
            edgecolor='white',
            width=width/3,
            color=colors[i])
        axs2[0].bar(
            session_id_mega + 2*width/3, [short_epoch3[0][i],short_epoch3[1][i],short_epoch3[2][i]],
            bottom=short_bottom_3[:,i],
            edgecolor='white',
            width=width/3,
            color=colors[i])
        
        axs2[1].bar(
            session_id_mega, [long_epoch1[0][i],long_epoch1[1][i],long_epoch1[2][i]],
            bottom=long_bottom_1[:,i],
            edgecolor='white',
            width=width/3,
            color=colors[i],
            label=states_name[i])
        axs2[1].bar(
            session_id_mega  + width/3, [long_epoch2[0][i],long_epoch2[1][i],long_epoch2[2][i]],
            bottom=long_bottom_2[:,i],
            edgecolor='white',
            width=width/3,
            color=colors[i])
        axs2[1].bar(
            session_id_mega + 2*width/3, [long_epoch3[0][i],long_epoch3[1][i],long_epoch3[2][i]],
            bottom=long_bottom_3[:,i],
            edgecolor='white',
            width=width/3,
            color=colors[i])
            
    axs2[0].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs2[1].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)

    fig2.tight_layout()
    ##################################### GRAND PLOTTING #################################
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sem_grand_short_epoch1 = stats.sem(grand_short_epoch1, axis=0,nan_policy = 'omit')
        sem_grand_short_epoch2 = stats.sem(grand_short_epoch2, axis=0,nan_policy = 'omit')
        sem_grand_short_epoch3 = stats.sem(grand_short_epoch3, axis=0,nan_policy = 'omit')

        sem_grand_long_epoch1 = stats.sem(grand_long_epoch1, axis=0,nan_policy = 'omit')
        sem_grand_long_epoch2 = stats.sem(grand_long_epoch2, axis=0,nan_policy = 'omit')
        sem_grand_long_epoch3 = stats.sem(grand_long_epoch3, axis=0,nan_policy = 'omit')

    grand_short_epoch1 = np.nanmean(grand_short_epoch1, axis=0)
    grand_short_epoch2 = np.nanmean(grand_short_epoch2, axis=0)
    grand_short_epoch3 = np.nanmean(grand_short_epoch3, axis=0)

    grand_long_epoch1 = np.nanmean(grand_long_epoch1, axis=0)
    grand_long_epoch2 = np.nanmean(grand_long_epoch2, axis=0)
    grand_long_epoch3 = np.nanmean(grand_long_epoch3, axis=0)

    # Define colors
    black_shades = [(150, 150, 150), (100, 100, 100), (50, 50, 50)]
    red_shades = [(255, 102, 102), (255, 51, 51), (204, 0, 0)]
    skyblue_shades = [(135, 206, 235), (70, 130, 180), (0, 105, 148)]

    # Normalize the colors to [0, 1] range for matplotlib
    black_shades = [tuple(c/255 for c in shade) for shade in black_shades]
    red_shades = [tuple(c/255 for c in shade) for shade in red_shades]
    skyblue_shades = [tuple(c/255 for c in shade) for shade in skyblue_shades]

    fig3, axs3 = plt.subplots(nrows= 2 , figsize=(15, 10))
    fig3.suptitle('Reward percentage (Grand Average)')
    offset = 0.08
    # Base x-values for the categories
    x_rewarded = 0
    x_dnp1 = 1
    x_dnp2 = 2
    x_ep1 = 3
    x_ep2 = 4

    # Set x-ticks to show the original category labels
    axs3[0].set_xticks([x_rewarded, x_dnp1, x_dnp2, x_ep1, x_ep2])

    #  PLOTTING
    axs3[0].errorbar(x_rewarded - 3.5*offset, grand_short_epoch1[0][0] , sem_grand_short_epoch1[0][0], color=black_shades[0], fmt='o', capsize=4, label='Control_epoch1_short')
    axs3[0].errorbar(x_rewarded - 3*offset, grand_short_epoch2[0][0] , sem_grand_short_epoch2[0][0], color=black_shades[1], fmt='o', capsize=4, label='Control_epoch2_short')
    axs3[0].errorbar(x_rewarded - 2.5*offset, grand_short_epoch3[0][0] , sem_grand_short_epoch3[0][0], color=black_shades[2], fmt='o', capsize=4, label='Control_epoch3_short')

    axs3[0].errorbar(x_rewarded - .5*offset, grand_short_epoch1[1][0] , sem_grand_short_epoch1[1][0], color=red_shades[0], fmt='o', capsize=4, label='Chemo_epoch1_short')
    axs3[0].errorbar(x_rewarded , grand_short_epoch2[1][0] , sem_grand_short_epoch2[1][0], color=red_shades[1], fmt='o', capsize=4, label='Chemo_epoch2_short')
    axs3[0].errorbar(x_rewarded + .5*offset, grand_short_epoch3[1][0] , sem_grand_short_epoch3[1][0], color=red_shades[2], fmt='o', capsize=4, label='Chemo_epoch3_short')

    axs3[0].errorbar(x_rewarded + 2.5*offset, grand_short_epoch1[2][0] , sem_grand_short_epoch1[2][0], color=skyblue_shades[0], fmt='o', capsize=4, label='Opto_epoch1_short')
    axs3[0].errorbar(x_rewarded + 3*offset, grand_short_epoch2[2][0] , sem_grand_short_epoch2[2][0], color=skyblue_shades[1], fmt='o', capsize=4, label='Opto_epoch2_short')
    axs3[0].errorbar(x_rewarded + 3.5*offset, grand_short_epoch3[2][0] , sem_grand_short_epoch3[2][0], color=skyblue_shades[2], fmt='o', capsize=4, label='Opto_epoch3_short')
    #######
    axs3[0].errorbar(x_dnp1 - 3.5*offset, grand_short_epoch1[0][1] , sem_grand_short_epoch1[0][1], color=black_shades[0], fmt='o', capsize=4)
    axs3[0].errorbar(x_dnp1 - 3*offset, grand_short_epoch2[0][1] , sem_grand_short_epoch2[0][1], color=black_shades[1], fmt='o', capsize=4)
    axs3[0].errorbar(x_dnp1 - 2.5*offset, grand_short_epoch3[0][1] , sem_grand_short_epoch3[0][1], color=black_shades[2], fmt='o', capsize=4)

    axs3[0].errorbar(x_dnp1 - .5*offset, grand_short_epoch1[1][1] , sem_grand_short_epoch1[1][1], color=red_shades[0], fmt='o', capsize=4)
    axs3[0].errorbar(x_dnp1 , grand_short_epoch2[1][1] , sem_grand_short_epoch2[1][1], color=red_shades[1], fmt='o', capsize=4)
    axs3[0].errorbar(x_dnp1 + .5*offset, grand_short_epoch3[1][1] , sem_grand_short_epoch3[1][1], color=red_shades[2], fmt='o', capsize=4)

    axs3[0].errorbar(x_dnp1 + 2.5*offset, grand_short_epoch1[2][1] , sem_grand_short_epoch1[2][1], color=skyblue_shades[0], fmt='o', capsize=4)
    axs3[0].errorbar(x_dnp1 + 3*offset, grand_short_epoch2[2][1] , sem_grand_short_epoch2[2][1], color=skyblue_shades[1], fmt='o', capsize=4)
    axs3[0].errorbar(x_dnp1 + 3.5*offset, grand_short_epoch3[2][1] , sem_grand_short_epoch3[2][1], color=skyblue_shades[2], fmt='o', capsize=4)
    #######
    axs3[0].errorbar(x_dnp2 - 3.5*offset, grand_short_epoch1[0][2] , sem_grand_short_epoch1[0][2], color=black_shades[0], fmt='o', capsize=4)
    axs3[0].errorbar(x_dnp2 - 3*offset, grand_short_epoch2[0][2] , sem_grand_short_epoch2[0][2], color=black_shades[1], fmt='o', capsize=4)
    axs3[0].errorbar(x_dnp2 - 2.5*offset, grand_short_epoch3[0][2] , sem_grand_short_epoch3[0][2], color=black_shades[2], fmt='o', capsize=4)

    axs3[0].errorbar(x_dnp2 - .5*offset, grand_short_epoch1[1][2] , sem_grand_short_epoch1[1][2], color=red_shades[0], fmt='o', capsize=4)
    axs3[0].errorbar(x_dnp2 , grand_short_epoch2[1][2] , sem_grand_short_epoch2[1][2], color=red_shades[1], fmt='o', capsize=4)
    axs3[0].errorbar(x_dnp2 + .5*offset, grand_short_epoch3[1][2] , sem_grand_short_epoch3[1][2], color=red_shades[2], fmt='o', capsize=4)

    axs3[0].errorbar(x_dnp2 + 2.5*offset, grand_short_epoch1[2][2] , sem_grand_short_epoch1[2][2], color=skyblue_shades[0], fmt='o', capsize=4)
    axs3[0].errorbar(x_dnp2 + 3*offset, grand_short_epoch2[2][2] , sem_grand_short_epoch2[2][2], color=skyblue_shades[1], fmt='o', capsize=4)
    axs3[0].errorbar(x_dnp2 + 3.5*offset, grand_short_epoch3[2][2] , sem_grand_short_epoch3[2][2], color=skyblue_shades[2], fmt='o', capsize=4)
    ######
    axs3[0].errorbar(x_ep1 - 3.5*offset, grand_short_epoch1[0][4] , sem_grand_short_epoch1[0][4], color=black_shades[0], fmt='o', capsize=4)
    axs3[0].errorbar(x_ep1 - 3*offset, grand_short_epoch2[0][4] , sem_grand_short_epoch2[0][4], color=black_shades[1], fmt='o', capsize=4)
    axs3[0].errorbar(x_ep1 - 2.5*offset, grand_short_epoch3[0][4] , sem_grand_short_epoch3[0][4], color=black_shades[2], fmt='o', capsize=4)

    axs3[0].errorbar(x_ep1 - .5*offset, grand_short_epoch1[1][4] , sem_grand_short_epoch1[1][4], color=red_shades[0], fmt='o', capsize=4)
    axs3[0].errorbar(x_ep1 , grand_short_epoch2[1][4] , sem_grand_short_epoch2[1][4], color=red_shades[1], fmt='o', capsize=4)
    axs3[0].errorbar(x_ep1 + .5*offset, grand_short_epoch3[1][4] , sem_grand_short_epoch3[1][4], color=red_shades[2], fmt='o', capsize=4)

    axs3[0].errorbar(x_ep1 + 2.5*offset, grand_short_epoch1[2][4] , sem_grand_short_epoch1[2][4], color=skyblue_shades[0], fmt='o', capsize=4)
    axs3[0].errorbar(x_ep1 + 3*offset, grand_short_epoch2[2][4] , sem_grand_short_epoch2[2][4], color=skyblue_shades[1], fmt='o', capsize=4)
    axs3[0].errorbar(x_ep1 + 3.5*offset, grand_short_epoch3[2][4] , sem_grand_short_epoch3[2][4], color=skyblue_shades[2], fmt='o', capsize=4)
    ######
    axs3[0].errorbar(x_ep2 - 3.5*offset, grand_short_epoch1[0][5] , sem_grand_short_epoch1[0][5], color=black_shades[0], fmt='o', capsize=4)
    axs3[0].errorbar(x_ep2 - 3*offset, grand_short_epoch2[0][5] , sem_grand_short_epoch2[0][5], color=black_shades[1], fmt='o', capsize=4)
    axs3[0].errorbar(x_ep2 - 2.5*offset, grand_short_epoch3[0][5] , sem_grand_short_epoch3[0][5], color=black_shades[2], fmt='o', capsize=4)

    axs3[0].errorbar(x_ep2 - .5*offset, grand_short_epoch1[1][5] , sem_grand_short_epoch1[1][5], color=red_shades[0], fmt='o', capsize=4)
    axs3[0].errorbar(x_ep2 , grand_short_epoch2[1][5] , sem_grand_short_epoch2[1][5], color=red_shades[1], fmt='o', capsize=4)
    axs3[0].errorbar(x_ep2 + .5*offset, grand_short_epoch3[1][5] , sem_grand_short_epoch3[1][5], color=red_shades[2], fmt='o', capsize=4)

    axs3[0].errorbar(x_ep2 + 2.5*offset, grand_short_epoch1[2][5] , sem_grand_short_epoch1[2][5], color=skyblue_shades[0], fmt='o', capsize=4)
    axs3[0].errorbar(x_ep2 + 3*offset, grand_short_epoch2[2][5] , sem_grand_short_epoch2[2][5], color=skyblue_shades[1], fmt='o', capsize=4)
    axs3[0].errorbar(x_ep2 + 3.5*offset, grand_short_epoch3[2][5] , sem_grand_short_epoch3[2][5], color=skyblue_shades[2], fmt='o', capsize=4)
    ######
    axs3[0].set_xticklabels(['Rewarded', 'DidNotPress1', 'DidNotPress2', 'EarlyPress1', 'EarlyPress2'])

    axs3[0].set_ylim(-0.1,1.1)
    axs3[0].spines['right'].set_visible(False)
    axs3[0].spines['top'].set_visible(False)
    axs3[0].set_title('Epoch Analysis for Short Trials')
    axs3[0].set_ylabel('outcome percentages mean +/- SEM')

    axs3[1].set_xticks([x_rewarded, x_dnp1, x_dnp2, x_ep1, x_ep2])

    axs3[1].errorbar(x_rewarded - 3.5*offset, grand_long_epoch1[0][0] , sem_grand_long_epoch1[0][0], color=black_shades[0], fmt='o', capsize=4, label='Control_epoch1_long')
    axs3[1].errorbar(x_rewarded - 3*offset, grand_long_epoch2[0][0] , sem_grand_long_epoch2[0][0], color=black_shades[1], fmt='o', capsize=4, label='Control_epoch2_long')
    axs3[1].errorbar(x_rewarded - 2.5*offset, grand_long_epoch3[0][0] , sem_grand_long_epoch3[0][0], color=black_shades[2], fmt='o', capsize=4, label='Control_epoch3_long')

    axs3[1].errorbar(x_rewarded - .5*offset, grand_long_epoch1[1][0] , sem_grand_long_epoch1[1][0], color=red_shades[0], fmt='o', capsize=4, label='Chemo_epoch1_long')
    axs3[1].errorbar(x_rewarded , grand_long_epoch2[1][0] , sem_grand_long_epoch2[1][0], color=red_shades[1], fmt='o', capsize=4, label='Chemo_epoch2_long')
    axs3[1].errorbar(x_rewarded + .5*offset, grand_long_epoch3[1][0] , sem_grand_long_epoch3[1][0], color=red_shades[2], fmt='o', capsize=4, label='Chemo_epoch3_long')

    axs3[1].errorbar(x_rewarded + 2.5*offset, grand_long_epoch1[2][0] , sem_grand_long_epoch1[2][0], color=skyblue_shades[0], fmt='o', capsize=4, label='Opto_epoch1_long')
    axs3[1].errorbar(x_rewarded + 3*offset, grand_long_epoch2[2][0] , sem_grand_long_epoch2[2][0], color=skyblue_shades[1], fmt='o', capsize=4, label='Opto_epoch2_long')
    axs3[1].errorbar(x_rewarded + 3.5*offset, grand_long_epoch3[2][0] , sem_grand_long_epoch3[2][0], color=skyblue_shades[2], fmt='o', capsize=4, label='Opto_epoch3_long')
    #######
    axs3[1].errorbar(x_dnp1 - 3.5*offset, grand_long_epoch1[0][1] , sem_grand_long_epoch1[0][1], color=black_shades[0], fmt='o', capsize=4)
    axs3[1].errorbar(x_dnp1 - 3*offset, grand_long_epoch2[0][1] , sem_grand_long_epoch2[0][1], color=black_shades[1], fmt='o', capsize=4)
    axs3[1].errorbar(x_dnp1 - 2.5*offset, grand_long_epoch3[0][1] , sem_grand_long_epoch3[0][1], color=black_shades[2], fmt='o', capsize=4)

    axs3[1].errorbar(x_dnp1 - .5*offset, grand_long_epoch1[1][1] , sem_grand_long_epoch1[1][1], color=red_shades[0], fmt='o', capsize=4)
    axs3[1].errorbar(x_dnp1 , grand_long_epoch2[1][1] , sem_grand_long_epoch2[1][1], color=red_shades[1], fmt='o', capsize=4)
    axs3[1].errorbar(x_dnp1 + .5*offset, grand_long_epoch3[1][1] , sem_grand_long_epoch3[1][1], color=red_shades[2], fmt='o', capsize=4)

    axs3[1].errorbar(x_dnp1 + 2.5*offset, grand_long_epoch1[2][1] , sem_grand_long_epoch1[2][1], color=skyblue_shades[0], fmt='o', capsize=4)
    axs3[1].errorbar(x_dnp1 + 3*offset, grand_long_epoch2[2][1] , sem_grand_long_epoch2[2][1], color=skyblue_shades[1], fmt='o', capsize=4)
    axs3[1].errorbar(x_dnp1 + 3.5*offset, grand_long_epoch3[2][1] , sem_grand_long_epoch3[2][1], color=skyblue_shades[2], fmt='o', capsize=4)
    #######
    axs3[1].errorbar(x_dnp2 - 3.5*offset, grand_long_epoch1[0][2] , sem_grand_long_epoch1[0][2], color=black_shades[0], fmt='o', capsize=4)
    axs3[1].errorbar(x_dnp2 - 3*offset, grand_long_epoch2[0][2] , sem_grand_long_epoch2[0][2], color=black_shades[1], fmt='o', capsize=4)
    axs3[1].errorbar(x_dnp2 - 2.5*offset, grand_long_epoch3[0][2] , sem_grand_long_epoch3[0][2], color=black_shades[2], fmt='o', capsize=4)

    axs3[1].errorbar(x_dnp2 - .5*offset, grand_long_epoch1[1][2] , sem_grand_long_epoch1[1][2], color=red_shades[0], fmt='o', capsize=4)
    axs3[1].errorbar(x_dnp2 , grand_long_epoch2[1][2] , sem_grand_long_epoch2[1][2], color=red_shades[1], fmt='o', capsize=4)
    axs3[1].errorbar(x_dnp2 + .5*offset, grand_long_epoch3[1][2] , sem_grand_long_epoch3[1][2], color=red_shades[2], fmt='o', capsize=4)

    axs3[1].errorbar(x_dnp2 + 2.5*offset, grand_long_epoch1[2][2] , sem_grand_long_epoch1[2][2], color=skyblue_shades[0], fmt='o', capsize=4)
    axs3[1].errorbar(x_dnp2 + 3*offset, grand_long_epoch2[2][2] , sem_grand_long_epoch2[2][2], color=skyblue_shades[1], fmt='o', capsize=4)
    axs3[1].errorbar(x_dnp2 + 3.5*offset, grand_long_epoch3[2][2] , sem_grand_long_epoch3[2][2], color=skyblue_shades[2], fmt='o', capsize=4)
    ######
    axs3[1].errorbar(x_ep1 - 3.5*offset, grand_long_epoch1[0][4] , sem_grand_long_epoch1[0][4], color=black_shades[0], fmt='o', capsize=4)
    axs3[1].errorbar(x_ep1 - 3*offset, grand_long_epoch2[0][4] , sem_grand_long_epoch2[0][4], color=black_shades[1], fmt='o', capsize=4)
    axs3[1].errorbar(x_ep1 - 2.5*offset, grand_long_epoch3[0][4] , sem_grand_long_epoch3[0][4], color=black_shades[2], fmt='o', capsize=4)

    axs3[1].errorbar(x_ep1 - .5*offset, grand_long_epoch1[1][4] , sem_grand_long_epoch1[1][4], color=red_shades[0], fmt='o', capsize=4)
    axs3[1].errorbar(x_ep1 , grand_long_epoch2[1][4] , sem_grand_long_epoch2[1][4], color=red_shades[1], fmt='o', capsize=4)
    axs3[1].errorbar(x_ep1 + .5*offset, grand_long_epoch3[1][4] , sem_grand_long_epoch3[1][4], color=red_shades[2], fmt='o', capsize=4)

    axs3[1].errorbar(x_ep1 + 2.5*offset, grand_long_epoch1[2][4] , sem_grand_long_epoch1[2][4], color=skyblue_shades[0], fmt='o', capsize=4)
    axs3[1].errorbar(x_ep1 + 3*offset, grand_long_epoch2[2][4] , sem_grand_long_epoch2[2][4], color=skyblue_shades[1], fmt='o', capsize=4)
    axs3[1].errorbar(x_ep1 + 3.5*offset, grand_long_epoch3[2][4] , sem_grand_long_epoch3[2][4], color=skyblue_shades[2], fmt='o', capsize=4)
    ######
    axs3[1].errorbar(x_ep2 - 3.5*offset, grand_long_epoch1[0][5] , sem_grand_long_epoch1[0][5], color=black_shades[0], fmt='o', capsize=4)
    axs3[1].errorbar(x_ep2 - 3*offset, grand_long_epoch2[0][5] , sem_grand_long_epoch2[0][5], color=black_shades[1], fmt='o', capsize=4)
    axs3[1].errorbar(x_ep2 - 2.5*offset, grand_long_epoch3[0][5] , sem_grand_long_epoch3[0][5], color=black_shades[2], fmt='o', capsize=4)

    axs3[1].errorbar(x_ep2 - .5*offset, grand_long_epoch1[1][5] , sem_grand_long_epoch1[1][5], color=red_shades[0], fmt='o', capsize=4)
    axs3[1].errorbar(x_ep2 , grand_long_epoch2[1][5] , sem_grand_long_epoch2[1][5], color=red_shades[1], fmt='o', capsize=4)
    axs3[1].errorbar(x_ep2 + .5*offset, grand_long_epoch3[1][5] , sem_grand_long_epoch3[1][5], color=red_shades[2], fmt='o', capsize=4)

    axs3[1].errorbar(x_ep2 + 2.5*offset, grand_long_epoch1[2][5] , sem_grand_long_epoch1[2][5], color=skyblue_shades[0], fmt='o', capsize=4)
    axs3[1].errorbar(x_ep2 + 3*offset, grand_long_epoch2[2][5] , sem_grand_long_epoch2[2][5], color=skyblue_shades[1], fmt='o', capsize=4)
    axs3[1].errorbar(x_ep2 + 3.5*offset, grand_long_epoch3[2][5] , sem_grand_long_epoch3[2][5], color=skyblue_shades[2], fmt='o', capsize=4)
    ######
    axs3[1].set_xticklabels(['Rewarded', 'DidNotPress1', 'DidNotPress2', 'EarlyPress1', 'EarlyPress2'])
    axs3[1].set_ylim(-0.1,1.1)
    axs3[1].spines['right'].set_visible(False)
    axs3[1].spines['top'].set_visible(False)
    axs3[1].set_title('Epoch Analysis for Long Trials')
    axs3[1].set_ylabel('outcome percentages mean +/- SEM')

    axs3[0].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    axs3[1].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    fig3.tight_layout()

    output_figs_dir = output_dir_onedrive + subject + '/'  
    pdf_path = os.path.join(output_figs_dir, subject + '_Learning_Epoch_Outcome_Short_Long.pdf')
    
    plt.rcParams['pdf.fonttype'] = 42  # Ensure text is kept as text (not outlines)
    plt.rcParams['ps.fonttype'] = 42   # For compatibility with EPS as well, if needed

    # Save both plots into a single PDF file with each on a separate page
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig3)

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)