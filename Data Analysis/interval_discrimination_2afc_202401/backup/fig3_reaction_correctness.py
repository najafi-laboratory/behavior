import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

def get_bin_stat(reaction, max_time, bin_size=0.5, min_samples=2):
    bins = np.arange(0, max_time + bin_size, bin_size)
    bins = bins - bin_size / 2
    bin_indices = np.digitize(reaction[:,0], bins) - 1
    bin_mean = []
    bin_sem = []
    for i in range(len(bins)-1):
        bin_values = reaction[bin_indices == i,1]
        m = np.mean(bin_values) if len(bin_values) > min_samples else np.nan
        s = sem(bin_values) if len(bin_values) > min_samples else np.nan
        bin_mean.append(m)
        bin_sem.append(s)
    bin_mean = np.array(bin_mean)
    bin_sem = np.array(bin_sem)
    return bins, bin_mean, bin_sem


def plot_subject(subject_session_data, max_time):
    subject = subject_session_data['subject']
    reaction = subject_session_data['reaction']
    reaction = np.concatenate(reaction).reshape(-1,3)
    reaction_correct_all = reaction[reaction[:,1]==1,:]
    reaction_wrong_all = reaction[reaction[:,1]==0,:]
    targets_correct = reaction_correct_all[:,2]
    targets_wrong = 1-reaction_wrong_all[:,2]
    reaction_correct_short = reaction_correct_all[targets_correct==0,:]
    reaction_correct_long = reaction_correct_all[targets_correct==1,:]
    reaction_wrong_short = reaction_wrong_all[targets_wrong==0,:]
    reaction_wrong_long = reaction_wrong_all[targets_wrong==1,:]
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    axs.errorbar(
        -0.1,
        np.mean(reaction_correct_all[:,0]),
        yerr=sem(reaction_correct_all[:,0]),
        linestyle='none',
        color='dodgerblue',
        capsize=2,
        marker='o',
        markeredgecolor='white',
        label='all',
        markeredgewidth=1)
    axs.errorbar(
        0,
        np.mean(reaction_correct_short[:,0]),
        yerr=sem(reaction_correct_short[:,0]),
        linestyle='none',
        color='mediumseagreen',
        capsize=2,
        marker='o',
        markeredgecolor='white',
        label='short',
        markeredgewidth=1)
    axs.errorbar(
        0.1,
        np.mean(reaction_correct_long[:,0]),
        yerr=sem(reaction_correct_long[:,0]),
        linestyle='none',
        color='violet',
        capsize=2,
        marker='o',
        markeredgecolor='white',
        label='long',
        markeredgewidth=1)
    axs.errorbar(
        1-0.1,
        np.mean(reaction_wrong_all[:,0]),
        yerr=sem(reaction_wrong_all[:,0]),
        linestyle='none',
        color='dodgerblue',
        capsize=2,
        marker='o',
        markeredgecolor='white',
        label='all',
        markeredgewidth=1)
    axs.errorbar(
        1,
        np.mean(reaction_wrong_short[:,0]),
        yerr=sem(reaction_wrong_short[:,0]),
        linestyle='none',
        color='mediumseagreen',
        capsize=2,
        marker='o',
        markeredgecolor='white',
        label='short',
        markeredgewidth=1)
    axs.errorbar(
        1+0.1,
        np.mean(reaction_wrong_long[:,0]),
        yerr=sem(reaction_wrong_long[:,0]),
        linestyle='none',
        color='violet',
        capsize=2,
        marker='o',
        markeredgecolor='white',
        label='long',
        markeredgewidth=1)
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_ylabel('Reaction time (since stim onset) / s')
    axs.set_xticks([0,1])
    axs.set_xticklabels(['correct', 'wrong'])
    handles, labels = axs.get_legend_handles_labels()
    axs.legend(handles[:3], labels[:3], loc='lower center')
    fig.suptitle(subject + ' reaction time')
    fig.set_size_inches(8, 6)
    print('Completed fig3 for ' + subject)
    fig.savefig('./figures/fig3_reaction_correctness_'+subject+'.pdf', dpi=300)
    fig.savefig('./figures/fig3_reaction_correctness_'+subject+'.png', dpi=300)
    plt.close()


def plot_fig3(session_data, max_time=5):
    for i in range(len(session_data)):
        plot_subject(session_data[i], max_time)
