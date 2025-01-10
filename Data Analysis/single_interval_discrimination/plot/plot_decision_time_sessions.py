import numpy as np
from scipy.stats import sem

def get_bin_stat(decision, max_time):
    # bin_size=250
    # bin_size=25
    bin_size=5    
    least_trials=3
    bins = np.arange(0, max_time + bin_size, bin_size)
    bins = bins - bin_size / 2
    bin_indices = np.digitize(decision[0,:], bins) - 1
    bin_mean = []
    bin_sem = []
    trials_per_bin = []
    for i in range(len(bins)-1):        
        correctness = decision[2, bin_indices == i].copy()
        m = np.mean(correctness) if len(correctness) > least_trials else np.nan
        s = sem(correctness) if len(correctness) > least_trials else np.nan
        num_trials = np.sum(bin_indices[bin_indices == i]) if len(correctness) > least_trials else np.nan
        trials_per_bin.append(num_trials)  
        bin_mean.append(m)
        bin_sem.append(s)
    bin_mean = np.array(bin_mean)
    bin_sem  = np.array(bin_sem)
    bin_time = bins[:-1] + (bins[1]-bins[0]) / 2
    non_nan  = (1-np.isnan(bin_mean)).astype('bool')
    bin_mean = bin_mean[non_nan]
    bin_sem  = bin_sem[non_nan]
    bin_time = bin_time[non_nan]
    trials_per_bin = np.array(trials_per_bin)
    trials_per_bin = trials_per_bin[non_nan]
    return bin_mean, bin_sem, bin_time, trials_per_bin


def separate_fix_jitter(decision):
    decision_fix = decision[:,decision[3,:]==0]
    decision_jitter = decision[:,decision[3,:]==1]
    decision_chemo = decision[:,decision[3,:]==2]
    decision_opto = decision[:,decision[3,:]==3]
    return decision_fix, decision_jitter, decision_chemo, decision_opto

def get_decision(subject_session_data):
    decision = subject_session_data['decision']
    num_non_nan = []
    for session in decision:
        session = np.concatenate(session, axis=1)
        sess_non_nan = (1-np.isnan(np.sum(session, axis=0)))
        # num_non_nan.append(np.sum(sess_non_nan)-1)
        num_non_nan.append(np.sum(sess_non_nan))
    sess_trial_start = [1] + num_non_nan[0:-1]
    sess_trial_start = np.cumsum(sess_trial_start)
    # num_trials = [len(session) + 1 for session in decision]
    decision = [np.concatenate(d, axis=1) for d in decision]
    decision = np.concatenate(decision, axis=1)
    jitter_flag = subject_session_data['jitter_flag']
    jitter_flag = np.concatenate(jitter_flag).reshape(1,-1)
    opto_flag = subject_session_data['opto_flag']
    opto_flag = np.concatenate(opto_flag).reshape(1,-1)
    jitter_flag[0 , :] = jitter_flag[0 , :] + opto_flag[0 , :]*3
    outcomes = subject_session_data['outcomes']
    all_trials = 0
    chemo_labels = subject_session_data['Chemo']
    # for j in range(len(chemo_labels)):
    #     if chemo_labels[j] == 1:
    #         jitter_flag[0 , all_trials:all_trials+len(outcomes[j])] = 2*np.ones(len(outcomes[j]))
    #     all_trials += len(outcomes[j])
    pre_isi = subject_session_data['pre_isi']
    pre_isi = np.concatenate(pre_isi).reshape(1,-1)
    post_isi_mean = subject_session_data['isi_post_emp']
    post_isi_mean = np.concatenate(post_isi_mean).reshape(1,-1)
    choice_start = subject_session_data['choice_start']
    choice_start = np.concatenate(choice_start).reshape(-1)     
    # stim_start = subject_session_data['stim_start']
    # stim_start = np.concatenate(stim_start).reshape(-1)
    # decision = np.concatenate([decision, jitter_flag, pre_isi, post_isi_mean], axis=0)
    decision = np.concatenate([decision, jitter_flag, post_isi_mean], axis=0)    
    # decision[0,:] -= stim_start
    # decision[0,:] -= choice_start
    decision[0,:] -= 1000*choice_start
    non_nan = (1-np.isnan(np.sum(decision, axis=0))).astype('bool')
    decision = decision[:,non_nan]
    # row 0: time.
    # row 1: direction.
    # row 2: correctness.
    # row 3: jitter flag.
    # row 4: pre pert isi.
    # row 5: post pert isi.
    decision_fix, decision_jitter, decision_chemo, decision_opto = separate_fix_jitter(decision)
    return decision_fix, decision_jitter, decision_chemo, decision_opto, sess_trial_start


def run(ax, subject_session_data, max_rt=700, plot_type='std', start_from='std'):
        
 
    subject_session_data_copy = subject_session_data.copy()
    
    if not start_from=='std':
        start_date = subject_session_data[start_from]
        dates = subject_session_data['dates']
        if start_date in dates:
            start_idx = dates.index(start_date)
        else:
            return
        
        for key in subject_session_data_copy.keys():
            # print(key)
            if isinstance(subject_session_data_copy[key], list) and len(subject_session_data_copy[key]) == len(dates):
                subject_session_data_copy[key] = subject_session_data_copy[key][start_idx:]     
    
    # max_time = 5000
    max_time = 1000 # choice window is 5s, although most licks are 1s or less
    decision_fix, decision_jitter, decision_chemo, decision_opto, sess_trial_start = get_decision(subject_session_data_copy)
    
    
    trial_num_fix = range(1, len(decision_fix[0])+1)
    
    correctness_fix = decision_fix[2,:]
    reward_fix = decision_fix[:,decision_fix[2,:] == 1][0]
    reward_fix_trial_num = np.where(decision_fix[2,:] == 1)[0]+1
    punish_fix = decision_fix[:,decision_fix[2,:] == 0][0]
    punish_fix_trial_num = np.where(decision_fix[2,:] == 0)[0]+1
    
    if plot_type=='std':
        ax.plot(
            trial_num_fix,
            decision_fix[0],
            color='indigo',
            marker='.',
            label='fix',
            markersize=4,
            alpha=0.2)    
    
    if plot_type=='trial-side':
        # not yet implemented
        left_idx = np.where(decision_fix[1,:] == 0)
        right_idx = np.where(decision_fix[1,:] == 1)
        left_trials_fix = decision_fix[0,left_idx][0]
        trial_num_left = (left_idx[0]+1).tolist()
        right_trials_fix = decision_fix[0,right_idx][0]
        trial_num_right = (right_idx[0]+1).tolist()
    elif plot_type=='lick-side':
        left_idx = np.where(decision_fix[1,:] == 0)
        right_idx = np.where(decision_fix[1,:] == 1)
        left_trials_fix = decision_fix[0,left_idx][0]
        trial_num_left = (left_idx[0]+1).tolist()
        right_trials_fix = decision_fix[0,right_idx][0]
        trial_num_right = (right_idx[0]+1).tolist()

    if (plot_type=='trial-side') or (plot_type=='lick-side'):
        ax.plot(
            trial_num_left,
            left_trials_fix,
            color='dodgerblue',
            marker='.',
            label='left',
            markersize=4,
            alpha=0.8)    
        
        ax.plot(
            trial_num_right,
            right_trials_fix,
            color='indianred',
            marker='.',
            label='right',
            markersize=4,
            alpha=0.8)     

    
    ax.scatter(
        reward_fix_trial_num,
        reward_fix,
        color='green',
        marker='.',
        label='reward')
    
    ax.scatter(
        punish_fix_trial_num,
        punish_fix,
        color='red',
        marker='.',
        label='punish')    
    
    bin_mean_fix, bin_sem_fix, bin_time_fix, trials_per_bin_fix = get_bin_stat(decision_fix, max_time)
    bin_mean_jitter, bin_sem_jitter, bin_time_jitter, trials_per_bin_jitter = get_bin_stat(decision_jitter, max_time)
    bin_mean_chemo, bin_sem_chemo, bin_time_chemo, trials_per_bin_chemo = get_bin_stat(decision_chemo, max_time)
    bin_mean_opto, bin_sem_opto, bin_time_opto, trials_per_bin_opto = get_bin_stat(decision_opto, max_time)
    # ax.plot(
    #     bin_time_fix,
    #     bin_mean_fix,
    #     color='indigo',
    #     marker='.',
    #     label='fix',
    #     markersize=4)
    # ax.fill_between(
    #     bin_time_fix,
    #     bin_mean_fix - bin_sem_fix,
    #     bin_mean_fix + bin_sem_fix,
    #     color='violet',
    #     alpha=0.2)
    # ax.plot(
    #     bin_time_jitter,
    #     bin_mean_jitter,
    #     color='limegreen',
    #     marker='.',
    #     label='jitter',
    #     markersize=4)
    # ax.fill_between(
    #     bin_time_jitter,
    #     bin_mean_jitter - bin_sem_jitter,
    #     bin_mean_jitter + bin_sem_jitter,
    #     color='limegreen',
    #     alpha=0.2)
    # ax.plot(
    #     bin_time_chemo,
    #     bin_mean_chemo,
    #     color='red',
    #     marker='.',
    #     label='chemo',
    #     markersize=4)
    # ax.fill_between(
    #     bin_time_chemo,
    #     bin_mean_chemo - bin_sem_chemo,
    #     bin_mean_chemo + bin_sem_chemo,
    #     color='red',
    #     alpha=0.2)
    # ax.plot(
    #     bin_time_opto,
    #     bin_mean_opto,
    #     color='dodgerblue',
    #     marker='.',
    #     label='opto',
    #     markersize=4)
    # ax.fill_between(
    #     bin_time_opto,
    #     bin_mean_opto - bin_sem_opto,
    #     bin_mean_opto + bin_sem_opto,
    #     color='dodgerblue',
    #     alpha=0.2)
    # ax.hlines(
    #     0.5, 0.0, max_time,
    #     linestyle=':', color='grey')
    # ax.vlines(
    #     1300, 0.0, 1.0,
    #     linestyle=':', color='mediumseagreen')
    y_axis_lim = max_rt
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_xlim([0, max_time])
    # ax.set_xlim([100, 300])
    # ax.set_xlim([0, 1000])
    ax.hlines(np.arange(0, y_axis_lim, 50), 0, len(trial_num_fix), linestyle=':', color='grey')
    ax.vlines(sess_trial_start, 0, y_axis_lim, linestyle=':', color='grey')
    ax.set_ylim([0.0, 600])
    ax.set_xlabel('trial number')
    ax.set_ylabel('decision time across trials (since choice window onset) / s')
    # ax.set_xticks(np.arange(0, max_time, 1000))
    # ax.set_xticks(np.arange(0, max_time, 100))
    
    
    # ax.set_xticks(np.arange(0, len(trial_num_fix), 100))
    ax.set_xticks(sess_trial_start, dates[start_idx:], rotation=45)
    
    
    ax.tick_params(axis='x', rotation=45)
    # ax.set_xticklabels(rotation=45)
    # ax.set_yticks([0.25, 0.50, 0.75, 1])
    # ax.set_yticks(np.arange(0, max_time, 1000))
    ax.set_yticks(np.arange(0, y_axis_lim, 50))
    
    
    # Create a second axis on the right side with a different scale
    # ax2 = ax.figure.add_axes(ax.get_position())  # Copy position from ax1
    
    # ax2.set_frame_on(False)  # Hide the box of the second axis
    # ax2.plot(x, y2, 'b-', label='2*cos(x)')
    # ax2.tick_params(axis='y', labelcolor='b')    
    
    # ax2 = ax.twinx()
    # ax2.set_ylabel('trials per bin')
    # ax2.plot(
    #     bin_time_fix,
    #     trials_per_bin_fix,
    #     color='gray',
    #     marker='.',
    #     label='fix',
    #     markersize=4)
    
    
    # ax.legend(loc='upper left', ncol=1, bbox_to_anchor=(1, 1))
    ax.legend(loc='best', ncol=1, bbox_to_anchor=(1, 1))
    # ax.set_title('average decision time curve')
    
    if start_from=='start_date':
        ax.set_title('response time across trials from ' + start_date)
    elif start_from=='non_naive':
        ax.set_title('response time across trials non-naive')
    else:
        ax.set_title('response time across trials')      