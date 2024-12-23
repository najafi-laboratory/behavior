import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

def get_bin_stat(licking, max_time, bin_size=0.5, min_samples=2):
    bins = np.arange(0, max_time + bin_size, bin_size)
    bins = bins - bin_size / 2
    bin_indices = np.digitize(licking[:,0], bins) - 1
    bin_mean = []
    bin_sem = []
    for i in range(len(bins)-1):
        bin_values = licking[bin_indices == i,2]
        m = np.mean(bin_values) if len(bin_values) > min_samples else np.nan
        s = sem(bin_values) if len(bin_values) > min_samples else np.nan
        bin_mean.append(m)
        bin_sem.append(s)
    bin_mean = np.array(bin_mean)
    bin_sem = np.array(bin_sem)
    return bins, bin_mean, bin_sem


def plot_subject(subject_session_data, max_time):
    subject = subject_session_data['subject']
    licking = subject_session_data['licking']
    licking = [np.concatenate(l,axis=1) for l in licking]
    licking = np.concatenate(licking,axis=1).transpose()
    targets = np.bitwise_xor(
        licking[:,2].astype('bool'),
        (1-licking[:,1]).astype('bool')).astype('int32')
    bins, bin_mean_all, bin_sem_all = get_bin_stat(
        licking, max_time)
    bins, bin_mean_short, bin_sem_short = get_bin_stat(
        licking[targets==0,:], max_time)
    bins, bin_mean_long, bin_sem_long = get_bin_stat(
        licking[targets==1,:], max_time)
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    axs.plot(
        bins[:-1] + (bins[1]-bins[0]) / 2,
        bin_mean_all,
        color='dodgerblue',
        marker='.',
        label='all',
        markersize=4)
    axs.fill_between(
        bins[:-1] + (bins[1]-bins[0]) / 2,
        bin_mean_all - bin_sem_all,
        bin_mean_all + bin_sem_all,
        color='dodgerblue',
        alpha=0.2)
    axs.plot(
        bins[:-1] + (bins[1]-bins[0]) / 2,
        bin_mean_short,
        color='mediumseagreen',
        marker='.',
        label='short',
        markersize=4)
    axs.fill_between(
        bins[:-1] + (bins[1]-bins[0]) / 2,
        bin_mean_short - bin_sem_short,
        bin_mean_short + bin_sem_short,
        color='mediumseagreen',
        alpha=0.2)
    axs.plot(
        bins[:-1] + (bins[1]-bins[0]) / 2,
        bin_mean_long,
        color='violet',
        marker='.',
        label='long',
        markersize=4)
    axs.fill_between(
        bins[:-1] + (bins[1]-bins[0]) / 2,
        bin_mean_long - bin_sem_long,
        bin_mean_long + bin_sem_long,
        color='violet',
        alpha=0.2)
    axs.hlines(
        0.5, 0.0, max_time,
        linestyle=':', color='grey',
        label='chance level')
    axs.fill_between(
        [0, 1.3], 1, 0,
        color='gold', alpha=0.1,
        label='Pre-perturb stim')
    axs.fill_between(
        [1.3, 4.3], 1, 0,
        color='coral', alpha=0.1,
        label='Post-perturb stim')
    axs.tick_params(tick1On=False)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlim([0, max_time])
    axs.set_ylim([-0.05, 1.05])
    axs.set_xlabel('Licking events time (since stim onset) / s')
    axs.set_ylabel('Prob. of right side licking')
    axs.set_xticks(np.arange(0, max_time, 1))
    axs.set_yticks([0, 0.25, 0.50, 0.75, 1])
    axs.legend(loc='lower right', ncol=1)
    fig.suptitle(subject + ' licking time')
    fig.set_size_inches(8, 6)
    print('Completed fig4 for ' + subject)
    fig.savefig('./figures/fig4_licking_'+subject+'.pdf', dpi=300)
    fig.savefig('./figures/fig4_licking_'+subject+'.png', dpi=300)
    plt.close()


def plot_fig4(session_data, max_time=5):
    for i in range(len(session_data)):
        plot_subject(session_data[i], max_time)
