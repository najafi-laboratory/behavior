import pandas as pd
import numpy as np

def transform_data_to_dataframe(sessions_data, session_index=0):
    """
    Transforms session data into a pandas DataFrame for analysis.
    Args:
        sessions_data (dict): Dictionary containing session data with keys:
            'outcomes', 'trial_types', 'trial_isi', 'block_type'.
        session_index (int): Index of the session to transform.
    Returns:
        pd.DataFrame: DataFrame containing trial information for the specified session.
    """
    if session_index >= len(sessions_data['outcomes']):
        return pd.DataFrame()

    outcomes = np.asarray(sessions_data['outcomes'][session_index])
    trial_types_int = np.asarray(sessions_data['trial_types'][session_index])
    trial_isi = np.asarray(sessions_data['trial_isi'][session_index])
    block_types = np.asarray(sessions_data['block_type'][session_index])

    if len(outcomes) == 0:
        return pd.DataFrame()

    mouse_choices = []
    for outcome, tt in zip(outcomes, trial_types_int):
        rewarded = outcome in ['Reward', 'RewardNaive']
        # Note: tt has already been swapped if reversed. 
        # So tt=1 always means "Logic requires Left".
        is_short = (tt == 1)
        if rewarded:
            mouse_choices.append('left' if is_short else 'right')
        else:
            mouse_choices.append('right' if is_short else 'left')

    df = pd.DataFrame({
        'isi': trial_isi,
        'trial_type': ['short' if tt == 1 else 'long' for tt in trial_types_int],
        'block_type': ['neutral' if b == 0 else 'short_block' if b == 1 else 'long_block' for b in block_types],
        'mouse_choice': mouse_choices,
        'rewarded': [1 if o in ['Reward', 'RewardNaive'] else 0 for o in outcomes]
    })
    return df.dropna().reset_index(drop=True)