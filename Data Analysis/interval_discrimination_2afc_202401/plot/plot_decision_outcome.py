import numpy as np
from scipy.stats import sem

def get_cate_decision_stat(decision):
    reward_all = decision[:,decision[2,:]==1]
    reward_left = reward_all[:, reward_all[1,:]==0]
    reward_right = reward_all[:, reward_all[1,:]==1]
    punish_all = decision[:,decision[2,:]==0]
    punish_left = punish_all[:, punish_all[1,:]==0]
    punish_right = punish_all[:, punish_all[1,:]==1]
    reward_mean = [
        np.mean(reward_all[0,:]),
        np.mean(reward_left[0,:]),
        np.mean(reward_right[0,:])]
    reward_sem = [
        sem(reward_all[0,:]),
        sem(reward_left[0,:]),
        sem(reward_right[0,:])]
    punish_mean = [
        np.mean(punish_all[0,:]),
        np.mean(punish_left[0,:]),
        np.mean(punish_right[0,:])]
    punish_sem = [
        sem(punish_all[0,:]),
        sem(punish_left[0,:]),
        sem(punish_right[0,:])]
    return [reward_mean, reward_sem, punish_mean, punish_sem]

def separate_fix_jitter(decision):
    decision_fix = decision[:,decision[3,:]==0]
    decision_jitter = decision[:,decision[3,:]==1]
    return decision_fix, decision_jitter

def get_decision(subject_session_data):
    decision = subject_session_data['decision']
    decision = [np.concatenate(d, axis=1) for d in decision]
    decision = np.concatenate(decision, axis=1)
    jitter_flag = subject_session_data['jitter_flag']
    jitter_flag = np.concatenate(jitter_flag).reshape(1,-1)
    isi_pre_emp = subject_session_data['isi_pre_emp']
    isi_pre_emp = np.concatenate(isi_pre_emp).reshape(1,-1)
    isi_post_emp = subject_session_data['isi_post_emp']
    isi_post_emp = np.concatenate(isi_post_emp).reshape(1,-1)
    stim_start = subject_session_data['stim_start']
    stim_start = np.concatenate(stim_start).reshape(-1)
    decision = np.concatenate([decision, jitter_flag, isi_pre_emp, isi_post_emp], axis=0)
    decision[0,:] -= stim_start
    non_nan = (1-np.isnan(np.sum(decision, axis=0))).astype('bool')
    decision = decision[:,non_nan]
    # row 0: time.
    # row 1: direction.
    # row 2: correctness.
    # row 3: jitter flag.
    # row 4: pre pert isi.
    # row 5: post pert isi.
    decision_fix, decision_jitter = separate_fix_jitter(decision)
    return [decision_fix, decision_jitter]

def run(ax, subject_session_data):
    r_fix_jitter = get_decision(subject_session_data)
    pos = [-0.1, 0, 0.1]
    offset = [-0.02, 0.02]
    colors = [['mediumseagreen', 'royalblue', 'brown'],
              ['#A4CB9E', '#9DB4CE', '#EDA1A4']]
    label = ['all', 'left', 'right']
    ax.hlines(1300, -0.2, 3, linestyle=':', color='grey', label='perturbation')
    for j in range(2):
        [reward_mean, reward_sem,
         punish_mean, punish_sem
         ] = get_cate_decision_stat(r_fix_jitter[j])
        for i in range(3):
            ax.errorbar(
                0 + pos[i] + offset[j],
                reward_mean[i], reward_sem[i],
                linestyle='none', color=colors[j][i], capsize=2, marker='o',
                markeredgecolor='white', markeredgewidth=1)
            ax.errorbar(
                1 + pos[i] + offset[j],
                punish_mean[i], punish_sem[i],
                linestyle='none', color=colors[j][i], capsize=2, marker='o',
                markeredgecolor='white', markeredgewidth=1)
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('decision time (since stim onset) / s')
    ax.set_xticks([0,1])
    ax.set_xticklabels(['reward', 'punish'])
    ax.set_xlim([-0.5, 3])
    for i in range(3):
        ax.plot([], label=label[i], color=colors[0][i])
    ax.legend(loc='upper right')
    ax.set_title('decision time V.S. outcome')