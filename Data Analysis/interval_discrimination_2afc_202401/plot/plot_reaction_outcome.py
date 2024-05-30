import numpy as np
from scipy.stats import sem

def get_cate_reaction_stat(reaction):
    reward_all = reaction[:,reaction[2,:]==1]
    reward_left = reward_all[:, reward_all[1,:]==0]
    reward_right = reward_all[:, reward_all[1,:]==1]
    punish_all = reaction[:,reaction[2,:]==0]
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

def separate_fix_jitter(reaction):
    reaction_fix = reaction[:,reaction[3,:]==0]
    reaction_jitter = reaction[:,reaction[3,:]==1]
    return reaction_fix, reaction_jitter

def get_reaction(subject_session_data):
    reaction = subject_session_data['reaction']
    reaction = [np.concatenate(d, axis=1) for d in reaction]
    reaction = np.concatenate(reaction, axis=1)
    jitter_flag = subject_session_data['jitter_flag']
    jitter_flag = np.concatenate(jitter_flag).reshape(1,-1)
    pre_isi = subject_session_data['pre_isi']
    pre_isi = np.concatenate(pre_isi).reshape(1,-1)
    post_isi_mean = subject_session_data['post_isi_mean']
    post_isi_mean = np.concatenate(post_isi_mean).reshape(1,-1)
    stim_start = subject_session_data['stim_start']
    stim_start = np.concatenate(stim_start).reshape(-1)
    reaction = np.concatenate([reaction, jitter_flag, pre_isi, post_isi_mean], axis=0)
    reaction[0,:] -= stim_start
    non_nan = (1-np.isnan(np.sum(reaction, axis=0))).astype('bool')
    reaction = reaction[:,non_nan]
    # row 0: time.
    # row 1: direction.
    # row 2: correctness.
    # row 3: jitter flag.
    # row 4: pre pert isi.
    # row 5: post pert isi.
    reaction_fix, reaction_jitter = separate_fix_jitter(reaction)
    return [reaction_fix, reaction_jitter]

def run(ax, subject_session_data):
    r_fix_jitter = get_reaction(subject_session_data)
    pos = [-0.1, 0, 0.1]
    offset = [-0.02, 0.02]
    colors = [['mediumseagreen', 'royalblue', 'brown'],
              ['#A4CB9E', '#9DB4CE', '#EDA1A4']]
    label = ['all', 'left', 'right']
    ax.hlines(1300, -0.2, 3, linestyle=':', color='grey', label='perturbation')
    for j in range(2):
        [reward_mean, reward_sem,
         punish_mean, punish_sem
         ] = get_cate_reaction_stat(r_fix_jitter[j])
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
    ax.set_ylabel('reaction time (since stim onset) / s')
    ax.set_xticks([0,1])
    ax.set_xticklabels(['reward', 'punish'])
    ax.set_xlim([-0.5, 3])
    for i in range(3):
        ax.plot([], label=label[i], color=colors[0][i])
    ax.legend(loc='upper right')
    ax.set_title('reaction time V.S. outcome')