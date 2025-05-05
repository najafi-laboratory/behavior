# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 21:39:54 2025

@author: timst
"""
import pandas as pd
import numpy as np

def prepare_session_overview_trial_type(df):
    """
    Prepare trial type overview data for plotting.
    Returns a DataFrame with trial index, trial side, opto, naive, and correctness flags.
    """
    df_tt = pd.DataFrame({
        'trial_index': df.index,
        'trial_side': df['trial_side'],                  # 'left' or 'right'
        'is_right': df['is_right'],                      # 1 for right, 0 for left
        'naive': df.get('naive', 0),                     # 1 if naive trial
        'is_opto': df.get('is_opto', 0),                       # 1 if opto trial
        'correct': df.get('mouse_correct', np.nan),      # 1 if correct, 0 if incorrect
        'outcome': df.get('outcome', 'unknown'),         # outcome string    
        'MoveCorrectSpout': df.get('MoveCorrectSpout', 0),
    })
    
    return df_tt

def prepare_session_overview_rt(df):
    """
    Prepare reaction time overview data for plotting.
    Returns a DataFrame with trial index, reaction time, opto, naive, and correctness flags.
    """

    df_rt = pd.DataFrame({
        'trial_index': df.index,
        'reaction_time': df['RT'],   
        'trial_side': df['trial_side'],                  # 'left' or 'right'
        'isi': df['isi'],                                # isi
        'is_right': df['is_right'],                      # 1 for right, 0 for left
        'naive': df.get('naive', 0),                     # 1 if naive trial
        'is_opto': df.get('is_opto', 0),                       # 1 if opto trial
        'correct': df.get('mouse_correct', np.nan),      # 1 if correct, 0 if incorrect
        'outcome': df.get('outcome', 'unknown'),         # outcome string    
        'MoveCorrectSpout': df.get('MoveCorrectSpout', 0),
    })
    
    return df_rt
