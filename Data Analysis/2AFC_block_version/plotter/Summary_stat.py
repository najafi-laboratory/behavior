import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import os
import pandas as pd

def plot_summary_stats(sessions_data, subject, data_paths, save_path=None):
    """
    Generates a 2x3 summary figure:
    Row 1: Performance (Reward Rate)
    Row 2: Reaction Time (RT)

    Columns:
    1. Stimulus Probability Effect (Rare vs Majority)
    2. Opto Effect by Probability (Split by Opto/Control within Rare/Majority)
    3. Overall Opto Effect by Stimulus
    """

    # --- Helper Functions ---

    def reconstruct_rt_array(lick_props, n_trials):
        """
        Extract RTs from lick properties into a trial-aligned array.
        """
        rt_array = np.full(n_trials, np.nan)
        keys = [
            'short_ISI_reward_left_correct_lick', 'short_ISI_reward_right_incorrect_lick',
            'short_ISI_punish_right_incorrect_lick', 'short_ISI_punish_left_correct_lick',
            'long_ISI_reward_right_correct_lick', 'long_ISI_reward_left_incorrect_lick',
            'long_ISI_punish_left_incorrect_lick', 'long_ISI_punish_right_correct_lick'
        ]
        
        for key in keys:
            if key in lick_props:
                data = lick_props[key]
                trials = data.get('trial_number', [])
                rts = data.get('Lick_reaction_time', [])
                for t, rt in zip(trials, rts):
                    if 0 <= t < n_trials:
                        rt_array[t] = rt
        return rt_array

    def get_stats(data_array, mask):
        """ Calculate Mean and SEM for a subset of data defined by mask. """
        subset = data_array[mask]
        # Remove NaNs (crucial for RT)
        subset = subset[~np.isnan(subset)]
        
        if len(subset) == 0:
            return np.nan, np.nan
        
        return np.mean(subset), sem(subset)

    # --- Data Extraction and Pooling ---

    dates = sessions_data['dates']
    outcomes_list = sessions_data['outcomes']
    block_types_list = sessions_data['block_type']
    trial_types_list = sessions_data.get('trial_types', [None]*len(dates))
    lick_props_list = sessions_data.get('lick_properties', [{} for _ in dates])
    opto_list = sessions_data.get('opto_tags', [[] for _ in dates])

    # Containers for pooled data
    pool_perf = [] # 1 for Reward, 0 for other
    pool_rt = []   # RT in seconds
    pool_blk = []  # Block Type (1=Short, 2=Long)
    pool_tt = []   # Trial Type (1=Short, 2=Long)
    pool_opto = [] # 1=Opto, 0=Control

    for i in range(len(dates)):
        n_trials = len(block_types_list[i])
        
        # 1. RT
        rt_arr = reconstruct_rt_array(lick_props_list[i], n_trials)
        
        # 2. Performance (Binary)
        # 'Reward' -> 1, Anything else (Punish, Miss, etc.) -> 0
        out_arr = np.array([1 if o == 'Reward' else 0 for o in outcomes_list[i]])
        
        # 3. Block & Trial Types
        blk_arr = np.array(block_types_list[i])
        tt_arr = trial_types_list[i]
        if tt_arr is None: 
            tt_arr = np.full(n_trials, np.nan)
        else: 
            tt_arr = np.array(tt_arr)

        # 4. Opto Tags
        raw_opto = opto_list[i]
        if raw_opto is None or len(raw_opto) == 0:
            opto_arr = np.zeros(n_trials)
        else:
            opto_temp = pd.to_numeric(raw_opto, errors='coerce')
            opto_arr = np.nan_to_num(opto_temp, nan=0.0)
            if len(opto_arr) < n_trials:
                opto_arr = np.pad(opto_arr, (0, n_trials - len(opto_arr)), constant_values=0)
            elif len(opto_arr) > n_trials:
                opto_arr = opto_arr[:n_trials]

        pool_perf.append(out_arr)
        pool_rt.append(rt_arr)
        pool_blk.append(blk_arr)
        pool_tt.append(tt_arr)
        pool_opto.append(opto_arr)

    # Concatenate all sessions
    if not pool_perf:
        print("No data available for summary plots.")
        return

    p_perf = np.concatenate(pool_perf)
    p_rt = np.concatenate(pool_rt)
    p_blk = np.concatenate(pool_blk)
    p_tt = np.concatenate(pool_tt)
    p_opto = np.concatenate(pool_opto)

    # --- Plotting ---

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Summary Statistics: Performance & RT - {subject}", fontsize=16, y=0.98)

    # Define rows: (Data Source, Y-Label, Row Title Prefix)
    rows_config = [
        (p_perf, "Performance (Reward Rate)", "Perf"),
        (p_rt,   "Reaction Time (s)",         "RT")
    ]

    # Plot Settings
    capsize = 5
    fmt = 'o'
    linewidth = 1.5
    markersize = 8

    for row_idx, (data, ylabel, prefix) in enumerate(rows_config):
        
        # ==========================================
        # Column 1: Probability Effect (Rare vs Majority)
        # ==========================================
        ax = axes[row_idx, 0]
        
        # Define Ticks Logic
        # Tick 1: Short Block (Blk=1)
        #   - Rare: Long Stim (tt=2)
        #   - Majority: Short Stim (tt=1)
        mask_sb = (p_blk == 1)
        m_sb_rare = mask_sb & (p_tt == 2)
        m_sb_maj = mask_sb & (p_tt == 1)
        
        # Tick 2: Long Block (Blk=2)
        #   - Rare: Short Stim (tt=1)
        #   - Majority: Long Stim (tt=2)
        mask_lb = (p_blk == 2)
        m_lb_rare = mask_lb & (p_tt == 1)
        m_lb_maj = mask_lb & (p_tt == 2)

        # Plot Short Block Tick (x=0)
        # Rare (Gray) -> Majority (Black)
        mu_r, err_r = get_stats(data, m_sb_rare)
        mu_m, err_m = get_stats(data, m_sb_maj)
        
        ax.errorbar(0 - 0.1, mu_r, yerr=err_r, fmt=fmt, color='gray', 
                    label='Rare' if row_idx == 0 else "", capsize=capsize, elinewidth=linewidth, ms=markersize)
        ax.errorbar(0 + 0.1, mu_m, yerr=err_m, fmt=fmt, color='black', 
                    label='Majority' if row_idx == 0 else "", capsize=capsize, elinewidth=linewidth, ms=markersize)

        # Plot Long Block Tick (x=1)
        mu_r, err_r = get_stats(data, m_lb_rare)
        mu_m, err_m = get_stats(data, m_lb_maj)
        
        ax.errorbar(1 - 0.1, mu_r, yerr=err_r, fmt=fmt, color='gray', capsize=capsize, elinewidth=linewidth, ms=markersize)
        ax.errorbar(1 + 0.1, mu_m, yerr=err_m, fmt=fmt, color='black', capsize=capsize, elinewidth=linewidth, ms=markersize)

        # Formatting
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Short Block", "Long Block"])
        ax.set_ylabel(ylabel)
        if row_idx == 0:
            ax.set_title("Probability Effect")
            ax.legend(loc='best', fontsize='small')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ==========================================
        # Column 2: Opto Effect by Probability
        # ==========================================
        ax = axes[row_idx, 1]
        
        # Order on X-axis per tick: 
        # 1. Opto Rare, 2. Control Rare, 3. Opto Majority, 4. Control Majority
        offsets = [-0.3, -0.1, 0.1, 0.3]
        
        # Colors: Opto(Blue), Control Rare(Gray), Opto(Blue), Control Maj(Black)
        # Using shades for Opto to distinguish slightly or keeping uniform blue? 
        # Request: "opto use blue shades".
        c_opto_rare = 'lightskyblue'
        c_ctl_rare = 'gray'
        c_opto_maj = 'dodgerblue' # Slightly darker blue for majority
        c_ctl_maj = 'black'
        
        col2_colors = [c_opto_rare, c_ctl_rare, c_opto_maj, c_ctl_maj]
        
        # --- Short Block (x=0) ---
        # Rare (Stim 2), Majority (Stim 1)
        # 1. Opto Rare
        m1 = m_sb_rare & (p_opto == 1)
        # 2. Ctl Rare
        m2 = m_sb_rare & (p_opto == 0)
        # 3. Opto Maj
        m3 = m_sb_maj & (p_opto == 1)
        # 4. Ctl Maj
        m4 = m_sb_maj & (p_opto == 0)
        
        masks_sb = [m1, m2, m3, m4]
        
        for i, m in enumerate(masks_sb):
            mu, err = get_stats(data, m)
            ax.errorbar(0 + offsets[i], mu, yerr=err, fmt=fmt, color=col2_colors[i], 
                        capsize=capsize, elinewidth=linewidth, ms=markersize)

        # --- Long Block (x=1) ---
        # Rare (Stim 1), Majority (Stim 2)
        # 1. Opto Rare
        m1 = m_lb_rare & (p_opto == 1)
        # 2. Ctl Rare
        m2 = m_lb_rare & (p_opto == 0)
        # 3. Opto Maj
        m3 = m_lb_maj & (p_opto == 1)
        # 4. Ctl Maj
        m4 = m_lb_maj & (p_opto == 0)
        
        masks_lb = [m1, m2, m3, m4]
        
        for i, m in enumerate(masks_lb):
            mu, err = get_stats(data, m)
            ax.errorbar(1 + offsets[i], mu, yerr=err, fmt=fmt, color=col2_colors[i], 
                        capsize=capsize, elinewidth=linewidth, ms=markersize)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Short Block", "Long Block"])
        if row_idx == 0: ax.set_title("Opto x Probability")
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ==========================================
        # Column 3: Overall Opto Effect by Stimulus
        # ==========================================
        ax = axes[row_idx, 2]
        
        # Ticks: Short Trials (0), Long Trials (1)
        # Order: Opto (Blue), Control (Black)
        
        # Short Trials (tt=1)
        m_st = (p_tt == 1)
        m_st_opto = m_st & (p_opto == 1)
        m_st_ctl = m_st & (p_opto == 0)
        
        # Long Trials (tt=2)
        m_lt = (p_tt == 2)
        m_lt_opto = m_lt & (p_opto == 1)
        m_lt_ctl = m_lt & (p_opto == 0)
        
        # Plot Short Trials Tick
        mu, err = get_stats(data, m_st_opto)
        ax.errorbar(0 - 0.1, mu, yerr=err, fmt=fmt, color='blue', label='Opto' if row_idx == 0 else "", capsize=capsize, elinewidth=linewidth, ms=markersize)
        
        mu, err = get_stats(data, m_st_ctl)
        ax.errorbar(0 + 0.1, mu, yerr=err, fmt=fmt, color='black', label='Control' if row_idx == 0 else "", capsize=capsize, elinewidth=linewidth, ms=markersize)
        
        # Plot Long Trials Tick
        mu, err = get_stats(data, m_lt_opto)
        ax.errorbar(1 - 0.1, mu, yerr=err, fmt=fmt, color='blue', capsize=capsize, elinewidth=linewidth, ms=markersize)
        
        mu, err = get_stats(data, m_lt_ctl)
        ax.errorbar(1 + 0.1, mu, yerr=err, fmt=fmt, color='black', capsize=capsize, elinewidth=linewidth, ms=markersize)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Short Trials", "Long Trials"])
        if row_idx == 0:
            ax.set_title("Overall Opto Effect")
            ax.legend(loc='best', fontsize='small')
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        filename = f"Summary_Stats_Perf_RT_{subject}.pdf"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path)
        print(f"Summary figure saved to: {full_path}")
        
    plt.close(fig)