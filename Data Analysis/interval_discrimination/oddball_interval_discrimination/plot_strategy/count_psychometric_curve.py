
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import matplotlib.colors as colors
import matplotlib.cm as cmx


# bin the data with timestamps.

def get_bin_stat(decision , lick_number):
    bin_size=100
    least_trials=5
    bins = np.arange(0, 1000 + bin_size, bin_size)
    bins = bins - bin_size / 2
    decision = decision[:,decision[6,:]==lick_number]
    bin_indices = np.digitize(decision[5,:], bins) - 1
    bin_mean = []
    bin_sem = []
    for i in range(len(bins)-1):
        direction = decision[1, bin_indices == i].copy()
        m = np.mean(direction) if len(direction) > least_trials else np.nan
        s = sem(direction) if len(direction) > least_trials else np.nan
        bin_mean.append(m)
        bin_sem.append(s)
    bin_mean = np.array(bin_mean)
    bin_sem  = np.array(bin_sem)
    if len(bins) > 1:
        bin_isi  = bins[:-1] + (bins[1]-bins[0]) / 2
    else:
        bin_isi  = bins[:-1] + bin_size / 2
    non_nan  = (1-np.isnan(bin_mean)).astype('bool')
    bin_mean = bin_mean[non_nan]
    bin_sem  = bin_sem[non_nan]
    bin_isi  = bin_isi[non_nan]
    return bin_mean, bin_sem, bin_isi


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
    for j in range(len(chemo_labels)):
        if chemo_labels[j] == 1:
            jitter_flag[0 , all_trials:all_trials+len(outcomes[j])] = 2*np.ones(len(outcomes[j]))
        all_trials += len(outcomes[j])
    number_flash = subject_session_data['number_flash']
    number_flash = np.concatenate(number_flash).reshape(1,-1)
    pre_isi = subject_session_data['pre_isi']
    pre_isi = np.concatenate(pre_isi).reshape(1,-1)
    post_isi = subject_session_data['post_isi']
    post_isi = np.concatenate(post_isi).reshape(1,-1)
    decision = np.concatenate([decision, jitter_flag, pre_isi, post_isi,number_flash], axis=0)
    non_nan = (1-np.isnan(np.sum(decision, axis=0))).astype('bool')
    decision = decision[:,non_nan]
    # row 0: time.
    # row 1: direction.
    # row 2: correctness.
    # row 3: jitter flag.
    # row 4: pre pert isi.
    # row 5: post pert isi.
    # row 6: number of flashes.
    decision_fix, decision_jitter, decision_chemo, decision_opto = separate_fix_jitter(decision)
    return decision_fix, decision_jitter, decision_chemo, decision_opto


def run(ax, ax1, ax2, ax3 , subject_session_data):
    decision_fix, decision_jitter, decision_chemo, decision_opto = get_decision(subject_session_data)
    last_flash = 9
    jet = cm = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=last_flash)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    for i in range(last_flash+1):
        colorVal = scalarMap.to_rgba(i)
        
        bin_mean_fix, bin_sem_fix, bin_isi_fix = get_bin_stat(decision_fix , i)
        bin_mean_jitter, bin_sem_jitter, bin_isi_jitter = get_bin_stat(decision_jitter , i)
        bin_mean_chemo, bin_sem_chemo, bin_isi_chemo = get_bin_stat(decision_chemo , i)
        bin_mean_opto, bin_sem_opto, bin_isi_opto = get_bin_stat(decision_opto , i)
        ax.plot(
            bin_isi_fix,
            bin_mean_fix,
            color=colorVal, marker='.', label=str(i)+'flash', markersize=4)

        ax1.plot(
            bin_isi_jitter,
            bin_mean_jitter,
            color=colorVal, marker='.', label=str(i)+'flash', markersize=4)

        ax2.plot(
            bin_isi_chemo,
            bin_mean_chemo,
            color=colorVal, marker='.', label=str(i)+'flash', markersize=4)
        
        ax3.plot(
            bin_isi_opto,
            bin_mean_opto,
            color=colorVal, marker='.', label=str(i)+'flash', markersize=4)

    ax.hlines(0.5, 0.0, 1000, linestyle=':', color='grey')
    ax.vlines(500, 0.0, 1.0, linestyle=':', color='grey')
    ax.tick_params(tick1On=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([-50,1050])
    ax.set_ylim([-0.05,1.05])
    ax.set_xticks(np.arange(6)*200)
    ax.set_yticks(np.arange(5)*0.25)
    ax.set_xlabel('post perturbation isi')
    ax.set_ylabel('prob. of choosing the right side (mean$\pm$sem)')
    ax.set_title('average psychometric function, fix')
    ax.hlines(0.5, 0.0, 1000, linestyle=':', color='grey')
    ax.vlines(500, 0.0, 1.0, linestyle=':', color='grey')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    
    ax1.tick_params(tick1On=False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_xlim([-50,1050])
    ax1.set_ylim([-0.05,1.05])
    ax1.set_xticks(np.arange(6)*200)
    ax1.set_yticks(np.arange(5)*0.25)
    ax1.set_xlabel('post perturbation isi')
    ax1.set_ylabel('prob. of choosing the right side (mean$\pm$sem)')
    ax1.set_title('average psychometric function, jitter')
    ax1.hlines(0.5, 0.0, 1000, linestyle=':', color='grey')
    ax1.vlines(500, 0.0, 1.0, linestyle=':', color='grey')
    ax1.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    
    ax2.tick_params(tick1On=False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_xlim([-50,1050])
    ax2.set_ylim([-0.05,1.05])
    ax2.set_xticks(np.arange(6)*200)
    ax2.set_yticks(np.arange(5)*0.25)
    ax2.set_xlabel('post perturbation isi')
    ax2.set_ylabel('prob. of choosing the right side (mean$\pm$sem)')
    ax2.set_title('average psychometric function, chemo')
    ax2.hlines(0.5, 0.0, 1000, linestyle=':', color='grey')
    ax2.vlines(500, 0.0, 1.0, linestyle=':', color='grey')
    ax2.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    
    ax3.tick_params(tick1On=False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.set_xlim([-50,1050])
    ax3.set_ylim([-0.05,1.05])
    ax3.set_xticks(np.arange(6)*200)
    ax3.set_yticks(np.arange(5)*0.25)
    ax3.set_xlabel('post perturbation isi')
    ax3.set_ylabel('prob. of choosing the right side (mean$\pm$sem)')
    ax3.set_title('average psychometric function, opto')
    ax3.hlines(0.5, 0.0, 1000, linestyle=':', color='grey')
    ax3.vlines(500, 0.0, 1.0, linestyle=':', color='grey')
    ax3.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)

