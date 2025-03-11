import numpy as np
from scipy.stats import sem

def get_bin_stat(decision, max_time):
    # bin_size=250
    bin_size=25
    # bin_size=5    
    # bin_size=100
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
        # num_trials = np.sum(bin_indices[bin_indices == i]) if len(correctness) > least_trials else np.nan
        num_trials = len(bin_indices[bin_indices == i]) if len(correctness) > least_trials else np.nan
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
    return decision_fix, decision_jitter, decision_chemo, decision_opto


def run(ax, subject_session_data, start_from='std'):
    
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
    decision_fix, decision_jitter, decision_chemo, decision_opto = get_decision(subject_session_data_copy)
    bin_mean_fix, bin_sem_fix, bin_time_fix, trials_per_bin_fix = get_bin_stat(decision_fix, max_time)
    bin_mean_jitter, bin_sem_jitter, bin_time_jitter, trials_per_bin_jitter = get_bin_stat(decision_jitter, max_time)
    bin_mean_chemo, bin_sem_chemo, bin_time_chemo, trials_per_bin_chemo = get_bin_stat(decision_chemo, max_time)
    bin_mean_opto, bin_sem_opto, bin_time_opto, trials_per_bin_opto = get_bin_stat(decision_opto, max_time)
    ax.plot(
        bin_time_fix,
        bin_mean_fix,
        color='indigo',
        marker='.',
        label='fix',
        markersize=4)
    ax.fill_between(
        bin_time_fix,
        bin_mean_fix - bin_sem_fix,
        bin_mean_fix + bin_sem_fix,
        color='violet',
        alpha=0.2)
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
    ax.hlines(
        0.5, 0.0, max_time,
        linestyle=':', color='grey')
    # ax.vlines(
    #     1300, 0.0, 1.0,
    #     linestyle=':', color='mediumseagreen')
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_xlim([0, max_time])
    ax.set_xlim([0, 650])
    # ax.set_xlim([0, 1000])
    ax.set_ylim([0.20, 1.05])
    ax.set_xlabel('decision time (since choice window onset) / s')
    ax.set_ylabel('correct prob.')
    # ax.set_xticks(np.arange(0, max_time, 1000))
    # ax.set_xticks(np.arange(0, max_time, 100))
    ax.set_xticks(np.arange(0, 650, 100))
    ax.tick_params(axis='x', rotation=45)
    # ax.set_xticklabels(rotation=45)
    ax.set_yticks([0.25, 0.50, 0.75, 1])
    
    
    # Create a second axis on the right side with a different scale
    # ax2 = ax.figure.add_axes(ax.get_position())  # Copy position from ax1
    ax2 = ax.twinx()
    # ax2.set_frame_on(False)  # Hide the box of the second axis
    # ax2.plot(x, y2, 'b-', label='2*cos(x)')
    ax2.set_ylabel('trials per bin')
    # ax2.tick_params(axis='y', labelcolor='b')    
    
    ax2.plot(
        bin_time_fix,
        trials_per_bin_fix,
        color='gray',
        marker='.',
        label='fix',
        markersize=4)
    
    
    # ax.legend(loc='upper left', ncol=1, bbox_to_anchor=(1, 1))
    ax.legend(loc='best', ncol=1, bbox_to_anchor=(1, 1))
    ax.set_title('average decision time curve')
    
    if start_from=='start_date':
        ax.set_title('average decision time curve from ' + start_date)
    elif start_from=='non_naive':
        ax.set_title('average decision time curve non-naive')
    else:
        ax.set_title('average decision time curve')      