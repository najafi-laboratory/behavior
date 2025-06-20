# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 15:19:35 2025

@author: saminnaji3
"""

import numpy as np

def calculate_interal(push_data, block_data, session_data):
    session_vis1_push1 = []
    session_vis2_push2 = []
    session_push1_push2 = []
    session_retract1_init_push2 = []
    session_retract1_end_push2 = []
    session_vis1_push2 = []
    session_wait2_push2 = []
    for i in range(len(block_data['start'])):
        trial_start = block_data['start'][i]
        trial_end = block_data['end'][i]
        trial_num_blocks = block_data['NumBlocks'][i]
        
        vis1_push1 = np.concatenate(np.array(push_data['push1'][i]))/1000 - np.array([push_data['vis1'][i][t][0] for t in range(len(push_data['vis1'][i]))])/1000
        vis2_push2 = np.concatenate(np.array(push_data['push2'][i]))/1000 - np.array([push_data['vis2'][i][t][0] for t in range(len(push_data['vis2'][i]))])/1000
        wait2_push2 = np.concatenate(np.array(push_data['push2'][i]))/1000 - np.array([push_data['wait2'][i][t][0] for t in range(len(push_data['wait2'][i]))])/1000
        wait2_push2[wait2_push2 < 0] = np.nan
        push1_push2 = np.concatenate(np.array(push_data['push2'][i]))/1000 - np.concatenate(np.array(push_data['push1'][i]))/1000
        retract1_init_push2 = np.concatenate(np.array(push_data['push2'][i]))/1000 - np.array([push_data['retract1_init'][i][t][0] for t in range(len(push_data['retract1_init'][i]))])/1000
        retract1_end_push2 = np.concatenate(np.array(push_data['push2'][i]))/1000 - np.array([push_data['retract1'][i][t][0] for t in range(len(push_data['retract1'][i]))])/1000
        vis1_push2 = np.concatenate(np.array(push_data['push2'][i]))/1000 - np.array([push_data['vis1'][i][t][0] for t in range(len(push_data['vis1'][i]))])/1000
        trial_vis1_push1 = []
        trial_vis2_push2 = []
        trial_push1_push2 = []
        trial_retract1_init_push2 = []
        trial_retract1_end_push2 = []
        trial_vis1_push2 = []
        trial_wait2_push2 = []
        
        for j in range(0 , trial_num_blocks):
            trial_vis1_push1.append(vis1_push1[trial_start[j] : trial_end[j]])
            trial_vis2_push2.append(vis2_push2[trial_start[j] : trial_end[j]])
            trial_push1_push2.append(push1_push2[trial_start[j] : trial_end[j]])
            trial_retract1_init_push2.append(retract1_init_push2[trial_start[j] : trial_end[j]])
            trial_retract1_end_push2.append(retract1_end_push2[trial_start[j] : trial_end[j]])
            trial_vis1_push2.append(vis1_push2[trial_start[j] : trial_end[j]])
            trial_wait2_push2.append(wait2_push2[trial_start[j] : trial_end[j]])
            
        session_vis1_push1.append(trial_vis1_push1)
        session_vis2_push2.append(trial_vis2_push2)
        session_push1_push2.append(trial_push1_push2)
        session_retract1_init_push2.append(trial_retract1_init_push2)
        session_retract1_end_push2.append(trial_retract1_end_push2)
        session_vis1_push2.append(trial_vis1_push2)
        session_wait2_push2.append(trial_wait2_push2)
        
        
    interval_Data = {'vis1_push1': session_vis1_push1,
                     'vis2_push2': session_vis2_push2,
                     'push1_push2': session_push1_push2,
                     'retract1_init_push2': session_retract1_init_push2,
                     'retract1_end_push2': session_retract1_end_push2,
                     'vis1_push2': session_vis1_push2,
                     'wait2_push2': session_wait2_push2
                     }
    
    return interval_Data