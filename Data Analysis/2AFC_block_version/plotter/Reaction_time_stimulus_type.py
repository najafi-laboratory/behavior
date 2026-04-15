import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import sem
import os
import pandas as pd

def plot_rt_by_block_status(sessions_data, subject, data_paths, save_path=None):
    """
    Plot Reaction Time (RT) aligned on block transitions.
    Generates a 2-page PDF:
      Page 1: RT by Stimulus (Short vs Long)
      Page 2: RT by Stimulus & Outcome (Short/Long x Reward/Punish)
    
    Groups (Columns):
    1. No Opto Trials (Cols 0-1)
    2. All Trials (Cols 3-4)
    3. Opto Trials Only (Cols 6-7)
    4. Destination is Control Block (Cols 9-10)
    5. Destination is Opto Block (Cols 12-13)
    """

    # --- Helper Functions ---

    def reconstruct_rt_array(lick_props, n_trials):
        """
        Consolidate RTs into a single array aligned with trial indices.
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

    def classify_transitions_by_dest_block(transitions, block_types, opto_tags):
        """
        Classify transitions based on whether the DESTINATION block (the block starting at t)
        is a Control Block (no opto trials) or an Opto Block (contains opto trials).
        """
        trans_control = []
        trans_opto = []

        total_trials = len(block_types)

        for t in transitions:
            current_block_type = block_types[t]
            
            # Find the end of the current block
            end_idx = t + 1
            while end_idx < total_trials and block_types[end_idx] == current_block_type:
                end_idx += 1
            
            # Slice the opto tags for this block
            if isinstance(opto_tags, list):
                 block_opto_slice = opto_tags[t:end_idx]
            else:
                 block_opto_slice = opto_tags[t:end_idx]

            # Check if any trial in this block was Opto
            if np.sum(block_opto_slice) > 0:
                trans_opto.append(t)
            else:
                trans_control.append(t)
                
        return trans_control, trans_opto

    def get_filtered_data(transitions, rt_array, outcome_array, opto_array, trial_types, mode='all', window=20):
        """
        Extract valid data points surrounding transitions, filtered by opto mode.
        Returns a list of dicts: [{'rt': val, 'out': val, 'tt': val, 'pos': relative_pos}, ...]
        """
        data_points = []
        if not transitions:
            return data_points

        for t in transitions:
            for pos in range(-window, window + 1):
                trial_idx = t + pos
                
                # Bounds check
                if 0 <= trial_idx < len(rt_array):
                    # Filter by Opto Mode
                    is_opto = (opto_array[trial_idx] != 0)
                    keep = False
                    if mode == 'all': keep = True
                    elif mode == 'no_opto' and not is_opto: keep = True
                    elif mode == 'opto' and is_opto: keep = True
                    
                    if keep and not np.isnan(rt_array[trial_idx]):
                        data_points.append({
                            'rt': rt_array[trial_idx],
                            'out': outcome_array[trial_idx],
                            'tt': trial_types[trial_idx] if trial_types is not None else None,
                            'pos': pos
                        })
        return data_points

    def calc_stats_page1(data_points, window=20):
        """
        Page 1 Logic: Short vs Long Stimulus.
        """
        trial_positions = np.arange(-window, window + 1)
        
        # Containers: [pos_index] -> list of RTs
        vals_short = [[] for _ in trial_positions]
        vals_long = [[] for _ in trial_positions]
        
        pos_map = {p: i for i, p in enumerate(trial_positions)}

        for dp in data_points:
            p_idx = pos_map.get(dp['pos'])
            if p_idx is not None and dp['tt'] is not None:
                if dp['tt'] == 1: # Short
                    vals_short[p_idx].append(dp['rt'])
                elif dp['tt'] == 2: # Long
                    vals_long[p_idx].append(dp['rt'])
        
        # Calculate Stats
        def get_mean_sem(vals_list):
            means = np.full(len(vals_list), np.nan)
            sems = np.full(len(vals_list), np.nan)
            for i, v in enumerate(vals_list):
                if v:
                    means[i] = np.mean(v)
                    sems[i] = sem(v)
            return means, sems

        ms, ss = get_mean_sem(vals_short)
        ml, sl = get_mean_sem(vals_long)
        
        return (ms, ss, ml, sl), trial_positions

    def calc_stats_page2(data_points, window=20):
        """
        Page 2 Logic: Short/Long x Reward/Punish (4 traces).
        """
        trial_positions = np.arange(-window, window + 1)
        pos_map = {p: i for i, p in enumerate(trial_positions)}

        # Containers
        v_sr = [[] for _ in trial_positions] # Short Reward
        v_lr = [[] for _ in trial_positions] # Long Reward
        v_sp = [[] for _ in trial_positions] # Short Punish
        v_lp = [[] for _ in trial_positions] # Long Punish

        for dp in data_points:
            p_idx = pos_map.get(dp['pos'])
            if p_idx is not None and dp['tt'] is not None:
                is_rew = (dp['out'] == 'Reward')
                
                if dp['tt'] == 1: # Short
                    if is_rew: v_sr[p_idx].append(dp['rt'])
                    else:      v_sp[p_idx].append(dp['rt'])
                elif dp['tt'] == 2: # Long
                    if is_rew: v_lr[p_idx].append(dp['rt'])
                    else:      v_lp[p_idx].append(dp['rt'])

        def get_mean_sem(vals_list):
            means = np.full(len(vals_list), np.nan)
            sems = np.full(len(vals_list), np.nan)
            for i, v in enumerate(vals_list):
                if v:
                    means[i] = np.mean(v)
                    sems[i] = sem(v)
            return means, sems

        return (
            get_mean_sem(v_sr), # Short Reward
            get_mean_sem(v_lr), # Long Reward
            get_mean_sem(v_sp), # Short Punish
            get_mean_sem(v_lp)  # Long Punish
        ), trial_positions

    def plot_page1_traces(ax, stats, positions, title, n_trans, show_ylabel=False):
        """ Plot Short (Blue) vs Long (Red) """
        (ms, ss, ml, sl) = stats
        
        # Short (Blue)
        if not np.all(np.isnan(ms)):
            ax.plot(positions, ms, 'o-', color='#1f77b4', label='Short', markersize=3, alpha=0.8)
            mask = ~np.isnan(ms)
            if np.any(mask):
                ax.fill_between(positions[mask], (ms-ss)[mask], (ms+ss)[mask], color='#1f77b4', alpha=0.2)
        
        # Long (Red)
        if not np.all(np.isnan(ml)):
            ax.plot(positions, ml, 'o-', color='#d62728', label='Long', markersize=3, alpha=0.8)
            mask = ~np.isnan(ml)
            if np.any(mask):
                ax.fill_between(positions[mask], (ml-sl)[mask], (ml+sl)[mask], color='#d62728', alpha=0.2)

        _finalize_plot(ax, title, n_trans, show_ylabel)

    def plot_page2_traces(ax, stats, positions, title, n_trans, show_ylabel=False):
        """ Plot 4 traces: Short/Long x Reward/Punish with high contrast Green/Red shades """
        ((msr, ssr), (mlr, slr), (msp, ssp), (mlp, slp)) = stats
        
        # Define high contrast colors
        # Reward: Green Shades
        c_short_rew = '#006400' # Dark Green
        c_long_rew = '#32CD32'  # Lime Green
        
        # Punish: Red Shades
        c_short_pun = '#8B0000' # Dark Red
        c_long_pun = '#FF6347'  # Tomato (Light Red/Orange)
        
        # Short Reward (Dark Green Solid)
        if not np.all(np.isnan(msr)):
            ax.plot(positions, msr, 'o-', color=c_short_rew, label='Short Rew', markersize=3, alpha=0.8)
            mask = ~np.isnan(msr)
            if np.any(mask):
                ax.fill_between(positions[mask], (msr-ssr)[mask], (msr+ssr)[mask], color=c_short_rew, alpha=0.2)

        # Long Reward (Lime Green Solid)
        if not np.all(np.isnan(mlr)):
            ax.plot(positions, mlr, 'o-', color=c_long_rew, label='Long Rew', markersize=3, alpha=0.8)
            mask = ~np.isnan(mlr)
            if np.any(mask):
                ax.fill_between(positions[mask], (mlr-slr)[mask], (mlr+slr)[mask], color=c_long_rew, alpha=0.2)

        # Short Punish (Dark Red Dashed)
        if not np.all(np.isnan(msp)):
            ax.plot(positions, msp, 'o--', color=c_short_pun, label='Short Pun', markersize=3, alpha=0.8)
            
        # Long Punish (Tomato Dashed)
        if not np.all(np.isnan(mlp)):
            ax.plot(positions, mlp, 'o--', color=c_long_pun, label='Long Pun', markersize=3, alpha=0.8)

        _finalize_plot(ax, title, n_trans, show_ylabel)

    def _finalize_plot(ax, title, n_trans, show_ylabel):
        ax.axvline(x=0, color='black', linestyle=':', alpha=0.5)
        ax.set_xticks(np.arange(-20, 21, 10))
        if show_ylabel: ax.set_ylabel('RT (s)')
        else: ax.set_yticklabels([])
        
        if n_trans > 0: ax.set_title(title, fontsize=9)
        else: ax.set_title(f"{title}\n(No Trans)", fontsize=8, color='gray')
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(True, axis='y', linestyle=':', alpha=0.3)


    # --- Main Execution Block ---

    dates = sessions_data['dates']
    outcomes_list = sessions_data['outcomes']
    block_types_list = sessions_data['block_type']
    lick_props_list = sessions_data['lick_properties']
    # Ensure trial types are retrieved
    trial_types_list = sessions_data.get('trial_types', [None] * len(dates))
    opto_list = sessions_data.get('opto_tags', [[]]*len(dates))
    n_sessions = len(dates)

    # Pre-process Data
    sess_arrays = [] 
    
    # Pools
    pool_rt, pool_out, pool_tt, pool_blk, pool_opto = [], [], [], [], []

    for i in range(n_sessions):
        n_trials = len(block_types_list[i])
        rt_arr = reconstruct_rt_array(lick_props_list[i], n_trials)
        out_arr = np.array(outcomes_list[i])
        blk_arr = np.array(block_types_list[i])
        
        tt_arr = trial_types_list[i]
        if tt_arr is None: tt_arr = np.full(n_trials, np.nan) # Handle missing
        else: tt_arr = np.array(tt_arr)

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

        sess_arrays.append({
            'rt': rt_arr, 'out': out_arr, 'tt': tt_arr, 'blk': blk_arr, 'opto': opto_arr
        })
        
        pool_rt.append(rt_arr)
        pool_out.append(out_arr)
        pool_tt.append(tt_arr)
        pool_blk.append(blk_arr)
        pool_opto.append(opto_arr)

    # Concatenate Pooled
    p_rt = np.concatenate(pool_rt)
    p_out = np.concatenate(pool_out)
    p_tt = np.concatenate(pool_tt)
    p_blk = np.concatenate(pool_blk)
    p_opto = np.concatenate(pool_opto)

    # Classify Pooled Transitions
    s2l_all_p = [i for i in range(1, len(p_blk)) if p_blk[i-1] == 1 and p_blk[i] == 2]
    l2s_all_p = [i for i in range(1, len(p_blk)) if p_blk[i-1] == 2 and p_blk[i] == 1]
    s2l_ctl_p, s2l_opto_p = classify_transitions_by_dest_block(s2l_all_p, p_blk, p_opto)
    l2s_ctl_p, l2s_opto_p = classify_transitions_by_dest_block(l2s_all_p, p_blk, p_opto)

    # --- Figure Generator Function ---
    def generate_figure(page_type):
        width_ratios = [1, 1, 0.2, 1, 1, 0.2, 1, 1, 0.2, 1, 1, 0.2, 1, 1]
        fig = plt.figure(figsize=(55, 3.5 * (n_sessions + 1) + 2)) 
        gs = gridspec.GridSpec(n_sessions + 1, len(width_ratios), figure=fig, width_ratios=width_ratios)
        
        subtitle = "Short vs Long Stimulus" if page_type == 'p1' else "Stimulus x Outcome (4 Traces)"
        fig.suptitle(f"Reaction Time: {subtitle}\nSubject: {subject}", fontsize=18, y=0.98)

        # Plot Groups Config
        # Groups: (Name, Col_SL, Col_LS, Mode)
        groups = [
            ("No Opto", 0, 1, 'no_opto'),
            ("All", 3, 4, 'all'),
            ("Opto Only", 6, 7, 'opto'),
            ("Control Blk", 9, 10, 'all'),
            ("Opto Blk", 12, 13, 'all')
        ]

        # 1. Plot Pooled Row
        # Define transitions map for pooled
        trans_map_p = {
            "No Opto": (s2l_all_p, l2s_all_p),
            "All": (s2l_all_p, l2s_all_p),
            "Opto Only": (s2l_all_p, l2s_all_p),
            "Control Blk": (s2l_ctl_p, l2s_ctl_p),
            "Opto Blk": (s2l_opto_p, l2s_opto_p)
        }

        for name, c_sl, c_ls, mode in groups:
            tsl, tls = trans_map_p[name]
            
            # S->L
            ax_sl = fig.add_subplot(gs[0, c_sl])
            data_sl = get_filtered_data(tsl, p_rt, p_out, p_opto, p_tt, mode=mode)
            if page_type == 'p1':
                stats, pos = calc_stats_page1(data_sl)
                plot_page1_traces(ax_sl, stats, pos, f"Pooled {name}: S->L", len(tsl), show_ylabel=(c_sl==0))
            else:
                stats, pos = calc_stats_page2(data_sl)
                plot_page2_traces(ax_sl, stats, pos, f"Pooled {name}: S->L", len(tsl), show_ylabel=(c_sl==0))

            if c_sl == 0: ax_sl.legend(loc='lower right', fontsize='x-small')

            # L->S
            ax_ls = fig.add_subplot(gs[0, c_ls])
            data_ls = get_filtered_data(tls, p_rt, p_out, p_opto, p_tt, mode=mode)
            if page_type == 'p1':
                stats, pos = calc_stats_page1(data_ls)
                plot_page1_traces(ax_ls, stats, pos, f"Pooled {name}: L->S", len(tls), show_ylabel=False)
            else:
                stats, pos = calc_stats_page2(data_ls)
                plot_page2_traces(ax_ls, stats, pos, f"Pooled {name}: L->S", len(tls), show_ylabel=False)

        # 2. Plot Session Rows
        for i, sd in enumerate(sess_arrays):
            # Calc transitions for this session
            s_s2l = [j for j in range(1, len(sd['blk'])) if sd['blk'][j-1] == 1 and sd['blk'][j] == 2]
            s_l2s = [j for j in range(1, len(sd['blk'])) if sd['blk'][j-1] == 2 and sd['blk'][j] == 1]
            
            s_s2l_c, s_s2l_o = classify_transitions_by_dest_block(s_s2l, sd['blk'], sd['opto'])
            s_l2s_c, s_l2s_o = classify_transitions_by_dest_block(s_l2s, sd['blk'], sd['opto'])
            
            trans_map_s = {
                "No Opto": (s_s2l, s_l2s),
                "All": (s_s2l, s_l2s),
                "Opto Only": (s_s2l, s_l2s),
                "Control Blk": (s_s2l_c, s_l2s_c),
                "Opto Blk": (s_s2l_o, s_l2s_o)
            }

            for name, c_sl, c_ls, mode in groups:
                tsl, tls = trans_map_s[name]
                
                # S->L
                ax_sl = fig.add_subplot(gs[i+1, c_sl])
                data_sl = get_filtered_data(tsl, sd['rt'], sd['out'], sd['opto'], sd['tt'], mode=mode)
                if page_type == 'p1':
                    stats, pos = calc_stats_page1(data_sl)
                    plot_page1_traces(ax_sl, stats, pos, f"{dates[i]} {name}", len(tsl), show_ylabel=(c_sl==0))
                else:
                    stats, pos = calc_stats_page2(data_sl)
                    plot_page2_traces(ax_sl, stats, pos, f"{dates[i]} {name}", len(tsl), show_ylabel=(c_sl==0))

                # L->S
                ax_ls = fig.add_subplot(gs[i+1, c_ls])
                data_ls = get_filtered_data(tls, sd['rt'], sd['out'], sd['opto'], sd['tt'], mode=mode)
                if page_type == 'p1':
                    stats, pos = calc_stats_page1(data_ls)
                    plot_page1_traces(ax_ls, stats, pos, f"{dates[i]} {name}", len(tls), show_ylabel=False)
                else:
                    stats, pos = calc_stats_page2(data_ls)
                    plot_page2_traces(ax_ls, stats, pos, f"{dates[i]} {name}", len(tls), show_ylabel=False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    # --- Save PDF ---
    if save_path:
        filename = f'RT_by_BlockTypes_MultiPage_{subject}.pdf'
        output_path = os.path.join(save_path, filename)
        
        with PdfPages(output_path) as pdf:
            print("Generating Page 1: Short vs Long...")
            fig1 = generate_figure('p1')
            pdf.savefig(fig1)
            plt.close(fig1)
            
            print("Generating Page 2: Stimulus x Outcome...")
            fig2 = generate_figure('p2')
            pdf.savefig(fig2)
            plt.close(fig2)
            
        print(f"Saved multi-page PDF to {output_path}")